"""PNG charts for the event-study report.

Each chart is drawn on its own matplotlib Figure with the Agg canvas (no pyplot, so no global figure state) and
follows one visual system: a light surface, ink-toned text, hairline solid y gridlines, and fixed colors by role
(bullish blue, bearish orange, placebo/SPY/reference gray). Every function returns the PNG path it wrote, or None
when there is nothing to draw.
"""

from __future__ import annotations

import math
import textwrap
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, Formatter, FuncFormatter, MaxNLocator

from ..timeutil import NY
from .study import DAILY_WINDOWS, INTRADAY_PATH_MINUTES, MIN_GROUP_POSTS, PATH_DAYS, Z_CRITICAL

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BULLISH = "#2a78d6"
BEARISH = "#eb6834"
NEUTRAL = "#1baf7a"
REFERENCE = "#898781"
OFF_HOURS = "#efeee8"  # background wash for pre-market / after-hours minutes
STANCE_COLORS = {"bullish": BULLISH, "bearish": BEARISH, "neutral": NEUTRAL}
UNLABELLED = INK_2  # a post without a stance must not borrow a stance's role color

WIDE = (8.0, 4.5)
SMALL = (6.0, 3.2)
DPI = 200
# The design sizes are CSS pixels (1/96 in) at 100% zoom; matplotlib measures lines and markers in points.
PX = 0.75
LINE_W = 2 * PX
HAIRLINE = 0.8 * PX
MARKER = 10 * PX  # 8 px of fill inside a 2 px surface ring
RING = 2 * PX
BAND_ALPHA = 0.15
MAX_AUTHOR_GROUPS = 15
BAR_MAX_IN = 24 / 96

TITLE_SIZE = 13.5
SUBTITLE_SIZE = 9.5
LABEL_SIZE = 9.0
TICK_SIZE = 8.5
NOTE_SIZE = 8.5
EDGE_IN = 0.2  # left edge of the title, and of the y tick labels under it

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "axes.unicode_minus": True,
    "savefig.facecolor": SURFACE,
}

INTRADAY_BEFORE = timedelta(minutes=60)
INTRADAY_AFTER = timedelta(minutes=120)
DAILY_SPAN = 10
# An off-hours print that jumps at least this far (and this many typical one-minute moves) from the price on both
# sides, then comes straight back within SPIKE_MAX_BARS bars, is drawn as a mark instead of joining the line.
SPIKE_MIN_JUMP = 0.0025
SPIKE_TYPICAL_MOVES = 20
SPIKE_MAX_BARS = 2


@dataclass(frozen=True)
class EventPrices:
    """Price change of one event's ticker and of SPY, as fractions of a common base.

    intraday: indexed by minutes from t0, base = the reference price (last 1-minute close before the post);
    `regular` is that day's regular session (open, close) in minutes from t0, so extended hours can be shaded.
    daily: indexed by trading sessions from d0, base = the d0-10 close (split- and dividend-adjusted)."""

    kind: Literal["intraday", "daily"]
    ticker: pd.Series
    spy: pd.Series
    t0: datetime
    regular: tuple[float, float] | None = None


@dataclass(frozen=True)
class AuthorRows:
    shown: pd.DataFrame  # at most MAX_AUTHOR_GROUPS rows with a finite mean, most posts first
    folded: pd.DataFrame  # drawable groups beyond the cap
    undrawable: pd.DataFrame  # groups with no signed mean (e.g. only neutral posts)


# ---------------------------------------------------------------- shared value helpers


def is_number(value: object) -> bool:
    if value is None or isinstance(value, bool | str):
        return False
    try:
        return math.isfinite(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def as_utc(value: object) -> datetime | None:
    """A post time from the Study (Timestamp, datetime or ISO text) as an aware UTC datetime."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        ts = pd.Timestamp(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    ts = ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)
    return ts.to_pydatetime()


def as_date(value: object) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        ts = pd.Timestamp(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if pd.isna(ts) else ts.date()


def format_ny(t: datetime) -> str:
    """'Sep 14, 2026 13:33 ET' in New York time."""
    local = t.astimezone(NY)
    return f"{local:%b} {local.day}, {local:%Y %H:%M} ET"


def stance_color(stance: object) -> str:
    """The role color of a stance; unlabelled or unknown stances get a neutral ink, never a stance's color."""
    return STANCE_COLORS.get(stance, UNLABELLED) if isinstance(stance, str) else UNLABELLED


def isolated_prints(series: pd.Series, regular: tuple[float, float] | None) -> pd.Series:
    """Boolean mask over an intraday series (indexed by minutes from t0 at each bar's end) marking off-hours prints
    that jump away from the price and come straight back within SPIKE_MAX_BARS bars: thin-trading bad ticks that
    would otherwise set a chart's scale. Display only; nothing is removed from the data."""
    x = series.to_numpy(float)
    flagged = np.zeros(len(x), dtype=bool)
    if regular is None or len(x) < 3:
        return pd.Series(flagged, index=series.index)
    minutes = series.index.to_numpy(float)
    open_, close = regular
    off = (minutes - 1 < open_ - 1e-6) | (minutes > close + 1e-6)
    moves = np.abs(np.diff(x))
    # A typical move in thin trading, not the busier regular session, sets the bar.
    off_moves = moves[off[1:] & off[:-1]]
    typical = float(np.median(off_moves if off_moves.size >= 10 else moves))
    threshold = max(SPIKE_MIN_JUMP, SPIKE_TYPICAL_MOVES * typical)
    i = 1
    while i < len(x) - 1:
        step = 1
        for run in range(1, SPIKE_MAX_BARS + 1):
            after = i + run
            if after >= len(x):
                break
            dev = x[i:after] - (x[i - 1] + x[after]) / 2
            if (
                abs(x[after] - x[i - 1]) < threshold / 2
                and off[i:after].all()
                and (np.abs(dev) > threshold).all()
                and ((dev > 0).all() or (dev < 0).all())
            ):
                flagged[i:after] = True
                step = run + 1  # the bar after the run is back at the price: not a spike
                break
        i += step
    return pd.Series(flagged, index=series.index)


def pct_text(value: object, decimals: int = 2) -> str:
    """Signed percent for chart text ('+1.23%'); an em dash for missing values."""
    if not is_number(value):
        return "—"
    v = float(value)  # type: ignore[arg-type]
    text = f"{abs(v) * 100:.{decimals}f}%"
    if float(text[:-1]) == 0:
        return text
    return ("+" if v > 0 else "−") + text


def author_rows(groups: pd.DataFrame, max_groups: int = MAX_AUTHOR_GROUPS) -> AuthorRows:
    """Author groups for the chart: main subset, event window, the `max_groups` with the most posts."""
    rows = groups[(groups["family"] == "author") & (groups["subset"] == "main") & (groups["window"] == "event")]
    rows = rows.sort_values(["n_posts", "group"], ascending=[False, True], kind="stable")
    drawable = rows["mean_signed_car"].map(is_number).astype(bool)
    ok = rows[drawable]
    return AuthorRows(shown=ok.head(max_groups), folded=ok.iloc[max_groups:], undrawable=rows[~drawable])


# ---------------------------------------------------------------- figure scaffolding


class _Canvas:
    def __init__(self, size: tuple[float, float]) -> None:
        self.fig = Figure(figsize=size, dpi=DPI, facecolor=SURFACE)
        self.agg = FigureCanvasAgg(self.fig)
        self.width, self.height = size
        self.top_in = 0.0

    def text_width_in(self, text: str, size: float) -> float:
        probe = self.fig.text(0, 0, text, fontsize=size)
        width = probe.get_window_extent(self.agg.get_renderer()).width / DPI
        probe.remove()
        return width

    def wrap(self, text: str, size: float, max_in: float) -> str:
        width = self.text_width_in(text, size)
        if width <= max_in:
            return text
        chars = max(20, int(len(text) * max_in / width * 0.97))
        return textwrap.fill(text, chars)

    def header(self, title: str, subtitle: str) -> None:
        max_in = self.width - 2 * EDGE_IN
        x = EDGE_IN / self.width
        title = self.wrap(title, TITLE_SIZE, max_in)
        self.fig.text(
            x, 1 - 0.14 / self.height, title, ha="left", va="top", fontsize=TITLE_SIZE, fontweight=600, color=INK
        )
        title_lines = title.count("\n") + 1
        y = 0.14 + title_lines * 0.27
        subtitle = self.wrap(subtitle, SUBTITLE_SIZE, max_in)
        self.fig.text(
            x, 1 - y / self.height, subtitle, ha="left", va="top", fontsize=SUBTITLE_SIZE, color=INK_2, linespacing=1.35
        )
        self.top_in = y + (subtitle.count("\n") + 1) * 0.2 + 0.1

    def legend(self, handles: Sequence[object], labels: Sequence[str]) -> None:
        # One row when it fits the figure, otherwise two.
        em_in = LABEL_SIZE / 72
        row_in = sum(self.text_width_in(label, LABEL_SIZE) + (1.8 + 0.5 + 1.6) * em_in for label in labels)
        ncol = len(labels) if row_in <= self.width - 2 * EDGE_IN else math.ceil(len(labels) / 2)
        self.fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(EDGE_IN / self.width - 0.004, 1 - self.top_in / self.height),
            ncol=ncol,
            frameon=False,
            fontsize=LABEL_SIZE,
            labelcolor=INK_2,
            handlelength=1.8,
            handletextpad=0.5,
            columnspacing=1.6,
            borderaxespad=0,
            borderpad=0,
        )
        self.top_in += 0.34 + (0.2 if ncol < len(labels) else 0)

    def axes(self):
        ax = self.fig.add_axes((0.1, 0.1, 0.8, 0.8))
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
            ax.spines[side].set_linewidth(HAIRLINE)
        ax.tick_params(colors=MUTED, labelcolor=MUTED, labelsize=TICK_SIZE, width=HAIRLINE, length=3)
        ax.set_axisbelow(True)
        return ax

    def place(self, ax, *, bottom_in: float = 0.62, right_in: float = 0.3) -> None:
        """Size the axes so the y tick labels start at EDGE_IN under the title."""
        ax.set_position((0.1, bottom_in / self.height, 0.8, 1 - (self.top_in + bottom_in) / self.height))
        self.fig.draw_without_rendering()
        renderer = self.agg.get_renderer()
        widths = [t.get_window_extent(renderer).width for t in ax.get_yticklabels() if t.get_text()]
        left_in = EDGE_IN + (max(widths) / DPI if widths else 0) + 0.08
        ax.set_position(
            (
                left_in / self.width,
                bottom_in / self.height,
                1 - (left_in + right_in) / self.width,
                1 - (self.top_in + bottom_in) / self.height,
            )
        )

    def note(self, text: str) -> None:
        self.fig.text(
            EDGE_IN / self.width,
            0.06 / self.height,
            self.wrap(text, NOTE_SIZE, self.width - 2 * EDGE_IN),
            ha="left",
            va="bottom",
            fontsize=NOTE_SIZE,
            color=INK_2,
        )

    def save(self, out: Path) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(out, dpi=DPI, facecolor=SURFACE, metadata={"Software": None})
        return out


def _y_grid(ax) -> None:
    ax.yaxis.grid(True, color=GRID, linewidth=HAIRLINE, linestyle="-")
    ax.xaxis.grid(False)


def _zero_line(ax) -> None:
    ax.axhline(0, color=AXIS, linewidth=HAIRLINE * 1.4, zorder=1.5)


def _post_rule(ax, x: float, label: str) -> None:
    ax.axvline(x, color=AXIS, linewidth=HAIRLINE * 1.4, zorder=1.5)
    ax.annotate(
        label,
        xy=(x, 1),
        xycoords=("data", "axes fraction"),
        xytext=(0, 3),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=TICK_SIZE,
        color=MUTED,
    )


class _PercentFormatter(Formatter):
    """Fractions as percent ticks, with exactly as many decimals as the tick step needs."""

    def __init__(self, signed: bool) -> None:
        self.signed = signed
        self.decimals = 0

    def set_locs(self, locs) -> None:
        super().set_locs(locs)
        steps = np.diff(np.sort(np.asarray(locs, dtype=float)))
        steps = steps[steps > 0]
        step_pct = float(steps.min()) * 100 if steps.size else 1.0
        self.decimals = max(0, min(4, -math.floor(math.log10(step_pct) + 1e-9)))

    def __call__(self, x: float, pos: int | None = None) -> str:
        if abs(x) < 0.5 * 10 ** -(self.decimals + 2):
            return "0%"
        text = f"{abs(x) * 100:.{self.decimals}f}%"
        if x < 0:
            return "−" + text
        return ("+" if self.signed else "") + text


def _pct_axis(axis, signed: bool = True) -> None:
    # No 2.5 steps: a 2.5% tick would need one more decimal than its neighbours.
    axis.set_major_locator(MaxNLocator(nbins=6, steps=[1, 2, 5, 10], min_n_ticks=4))
    axis.set_major_formatter(_PercentFormatter(signed))


def _padded_limits(values: Sequence[float], include_zero: bool = True, pad: float = 0.1) -> tuple[float, float]:
    finite = [v for v in values if is_number(v)]
    if include_zero:
        finite.append(0.0)
    lo, hi = (min(finite), max(finite)) if finite else (-0.01, 0.01)
    span = hi - lo or max(abs(hi), 0.005)
    return lo - pad * span, hi + pad * span


def _series(frame: pd.DataFrame, name: str, x_col: str, y_col: str) -> pd.DataFrame | None:
    rows = frame[(frame["series"] == name) & frame[y_col].map(is_number).astype(bool)]
    if rows.empty:
        return None
    return rows.sort_values(x_col).astype({x_col: float, y_col: float})


def _max_n(rows: pd.DataFrame) -> int:
    n = pd.to_numeric(rows["n"], errors="coerce").max()
    return int(n) if is_number(n) else 0


def _line_with_band(ax, rows: pd.DataFrame, x_col: str, y_col: str, color: str, markers: bool) -> None:
    x = rows[x_col].to_numpy(float)
    lo = pd.to_numeric(rows["ci_low"], errors="coerce").to_numpy(float)
    hi = pd.to_numeric(rows["ci_high"], errors="coerce").to_numpy(float)
    ok = np.isfinite(lo) & np.isfinite(hi)
    if ok.any():
        ax.fill_between(
            x, np.where(ok, lo, 0), np.where(ok, hi, 0), where=ok, color=color, alpha=BAND_ALPHA, linewidth=0, zorder=2
        )
    ax.plot(
        x,
        rows[y_col].to_numpy(float),
        color=color,
        linewidth=LINE_W,
        solid_joinstyle="round",
        solid_capstyle="round",
        zorder=3,
        marker="o" if markers else None,
        markersize=MARKER,
        markerfacecolor=color,
        markeredgecolor=SURFACE,
        markeredgewidth=RING,
    )


def _handle(color: str, markers: bool = True) -> Line2D:
    return Line2D(
        [],
        [],
        color=color,
        linewidth=LINE_W,
        marker="o" if markers else None,
        markersize=MARKER - RING,
        markerfacecolor=color,
        markeredgewidth=0,
    )


def _end_labels(ax, points: list[tuple[float, float, str]], min_gap_pt: float = 13) -> None:
    """Value labels at line ends, in ink. If two ends sit too close to separate, the legend and table carry them."""
    if not points:
        return
    ax.figure.draw_without_rendering()
    to_px = ax.transData.transform
    ys = sorted(to_px((x, y))[1] for x, y, _ in points)
    min_gap_px = min_gap_pt * DPI / 72
    if any(b - a < min_gap_px for a, b in zip(ys, ys[1:], strict=False)):
        return
    for x, y, text in points:
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=LABEL_SIZE,
            color=INK,
        )


def _path_chart(
    frame: pd.DataFrame,
    out: Path,
    *,
    x_col: str,
    y_col: str,
    xs: Sequence[int],
    title: str,
    subtitle_metric: str,
    x_label: str,
    with_placebo: bool,
) -> Path | None:
    lines = {name: _series(frame, name, x_col, y_col) for name in ("bullish", "bearish")}
    lines = {name: rows for name, rows in lines.items() if rows is not None}
    if not lines:
        return None
    placebo = _series(frame, "placebo", x_col, y_col) if with_placebo else None

    with mpl.rc_context(_RC):
        c = _Canvas(WIDE)
        parts = [f"{name} n={_max_n(rows)}" for name, rows in lines.items()]
        if placebo is not None:
            parts.append(f"placebo n={_max_n(placebo)}")
        c.header(title, f"{subtitle_metric} · {', '.join(parts)}")

        handles, labels = [], []
        for name, rows in lines.items():
            n = _max_n(rows)
            handles.append(_handle(STANCE_COLORS[name]))
            labels.append(f"{name} posts" + (f" (n={n}, fewer than {MIN_GROUP_POSTS})" if n < MIN_GROUP_POSTS else ""))
        if placebo is not None:
            handles.append(_handle(REFERENCE, markers=False))
            labels.append("placebo days (no post)")
        c.legend(handles, labels)

        ax = c.axes()
        _y_grid(ax)
        _zero_line(ax)
        values: list[float] = []
        for name, rows in lines.items():
            _line_with_band(ax, rows, x_col, y_col, STANCE_COLORS[name], markers=True)
            values += [
                *rows[y_col],
                *pd.to_numeric(rows["ci_low"], errors="coerce"),
                *pd.to_numeric(rows["ci_high"], errors="coerce"),
            ]
        if placebo is not None:
            ax.plot(
                placebo[x_col], placebo[y_col], color=REFERENCE, linewidth=LINE_W, zorder=2.5, solid_capstyle="round"
            )
            values += list(placebo[y_col])
        lo, hi = _padded_limits(values)
        ax.set_ylim(lo, hi)
        _pct_axis(ax.yaxis)
        ax.set_xlim(min(xs) - (max(xs) - min(xs)) * 0.03, max(xs) + (max(xs) - min(xs)) * 0.03)
        ax.xaxis.set_major_locator(FixedLocator(list(xs)))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:+.0f}".replace("-", "−") if v else "0"))
        ax.set_xlabel(x_label, fontsize=LABEL_SIZE, color=MUTED, labelpad=4)
        _post_rule(ax, 0, "post")
        ends = []
        for rows in lines.values():
            last = rows.iloc[-1]
            ends.append((float(last[x_col]), float(last[y_col]), pct_text(last[y_col])))
        label_in = max(c.text_width_in(text, LABEL_SIZE) for _, _, text in ends)
        c.place(ax, right_in=label_in + 0.3)
        _end_labels(ax, ends)
        return c.save(out)


# ---------------------------------------------------------------- the report's charts


def car_path_chart(car_path: pd.DataFrame, out: Path) -> Path | None:
    """Mean cumulative abnormal return from day -5 to +5, bullish vs bearish posts with 95% CI, placebo in gray."""
    return _path_chart(
        car_path,
        out,
        x_col="day",
        y_col="mean_car",
        xs=PATH_DAYS,
        title="Average cumulative abnormal return around a post",
        subtitle_metric="Cumulative abnormal return vs SPY market model, mean of posts, 95% CI",
        x_label="day (trading days from the post; day 0 = first session at or after it)",
        with_placebo=True,
    )


def intraday_chart(intraday_path: pd.DataFrame, out: Path) -> Path | None:
    """Mean abnormal return path from 60 minutes before to 60 minutes after regular-session posts."""
    return _path_chart(
        intraday_path,
        out,
        x_col="minute",
        y_col="mean_ar",
        xs=INTRADAY_PATH_MINUTES,
        title="Minute by minute around regular-session posts",
        subtitle_metric="Abnormal return (return − β × SPY return) since the price at the post, mean, 95% CI",
        x_label="minutes from the post",
        with_placebo=False,
    )


def author_chart(groups: pd.DataFrame, out: Path) -> Path | None:
    """Mean signed CAR[0,+1] per author (main subset), sorted, with 95% CI whiskers; groups under the test minimum
    in muted gray, marked 'n<10', without whiskers (a 2-post interval would set the scale for everyone)."""
    rows = author_rows(groups)
    if rows.shown.empty:
        return None
    shown = rows.shown.sort_values(["mean_signed_car", "group"], ascending=[True, False], kind="stable")
    n_groups = len(shown)
    size = (WIDE[0], max(WIDE[1], 1.6 + 0.3 * n_groups))

    with mpl.rc_context(_RC):
        c = _Canvas(size)
        total = int(pd.to_numeric(shown["n_posts"], errors="coerce").fillna(0).sum())
        c.header(
            "Signed abnormal return by author, days 0 to +1",
            f"Mean signed CAR[0,+1] (bearish posts count −CAR), 95% CI · {n_groups} authors, {total} posts "
            f"· gray: n<{MIN_GROUP_POSTS}, not tested (no CI)",
        )
        ax = c.axes()
        ax.xaxis.grid(True, color=GRID, linewidth=HAIRLINE, linestyle="-")
        ax.yaxis.grid(False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)

        notes: list[str] = []
        if not rows.folded.empty:
            folded_posts = int(pd.to_numeric(rows.folded["n_posts"], errors="coerce").fillna(0).sum())
            notes.append(f"+{len(rows.folded)} more authors ({folded_posts} posts) are listed in the report's table.")
        if not rows.undrawable.empty:
            notes.append(f"{len(rows.undrawable)} author(s) with no bullish or bearish post are not drawn.")
        bottom_in = 0.62 + (0.26 if notes else 0)

        c.place(ax, bottom_in=bottom_in, right_in=0.55)
        ys = np.arange(n_groups, dtype=float)
        slot_in = ax.get_position().height * size[1] / max(n_groups, 1)
        bar_h = min(0.62, BAR_MAX_IN / slot_in)
        means = shown["mean_signed_car"].astype(float).to_numpy()
        ok = (shown["status"] == "ok").to_numpy()
        colors = [BULLISH if good else MUTED for good in ok]
        ax.barh(ys, means, height=bar_h, color=colors, linewidth=0, zorder=3)
        lo = pd.to_numeric(shown["ci_low"], errors="coerce").to_numpy(float)
        hi = pd.to_numeric(shown["ci_high"], errors="coerce").to_numpy(float)
        has_ci = np.isfinite(lo) & np.isfinite(hi) & ok
        if has_ci.any():
            ax.hlines(ys[has_ci], lo[has_ci], hi[has_ci], color=INK_2, linewidth=HAIRLINE * 1.25, zorder=4)
            ax.vlines(
                np.concatenate([lo[has_ci], hi[has_ci]]),
                np.tile(ys[has_ci], 2) - bar_h * 0.35,
                np.tile(ys[has_ci], 2) + bar_h * 0.35,
                color=INK_2,
                linewidth=HAIRLINE * 1.25,
                zorder=4,
            )
        ax.axvline(0, color=AXIS, linewidth=HAIRLINE * 1.4, zorder=3.5)

        labels = [
            f"{g} (n={int(n)})" if is_number(n) else str(g)
            for g, n in zip(shown["group"], shown["n_posts"], strict=True)
        ]
        ax.set_yticks(ys, labels=labels, fontsize=LABEL_SIZE, color=INK_2)
        ax.tick_params(axis="y", labelcolor=INK_2)
        ax.set_ylim(-0.7, n_groups - 0.3)
        values = [*means, *lo[has_ci], *hi[has_ci]]
        xlo, xhi = _padded_limits(values, pad=0.12)
        ax.set_xlim(xlo, xhi)
        _pct_axis(ax.xaxis)
        ax.set_xlabel("mean signed CAR[0,+1]", fontsize=LABEL_SIZE, color=MUTED, labelpad=4)
        c.place(ax, bottom_in=bottom_in, right_in=0.55)

        for y, mean, good in zip(ys, means, ok, strict=True):
            if good:
                continue
            ax.annotate(
                f"n<{MIN_GROUP_POSTS}",
                xy=(mean, y),
                xytext=(5 if mean >= 0 else -5, 0),
                textcoords="offset points",
                ha="left" if mean >= 0 else "right",
                va="center",
                fontsize=TICK_SIZE,
                color=MUTED,
            )
        if notes:
            c.note(" ".join(notes))
        return c.save(out)


def placebo_chart(magnitude: pd.DataFrame, out: Path) -> Path | None:
    """Share of |z| > 1.96 per daily window: real events (blue) vs placebo days (gray), with the 5% chance line."""
    windows = list(DAILY_WINDOWS)
    cells: dict[tuple[str, str], tuple[float, int]] = {}
    for _, row in magnitude.iterrows():
        n = row["n"]
        if row["window"] in windows and is_number(row["share_abs_z_gt_crit"]) and is_number(n) and n > 0:
            cells[(row["window"], row["sample"])] = (float(row["share_abs_z_gt_crit"]), int(n))
    if not any(sample == "events" for _, sample in cells):
        return None

    with mpl.rc_context(_RC):
        c = _Canvas(WIDE)
        ev = cells.get(("event", "events"))
        pl = cells.get(("event", "placebo"))
        n_text = f"events n={ev[1] if ev else 0}, placebo days n={pl[1] if pl else 0} (event window)"
        c.header(
            f"How often a move is unusually large (|z| > {Z_CRITICAL:g})",
            f"Share of events vs placebo days (same tickers, random days without a post) · {n_text}",
        )
        c.legend(
            [Patch(color=BULLISH), Patch(color=REFERENCE), Line2D([], [], color=INK_2, linewidth=HAIRLINE * 1.6)],
            ["real events", "placebo days", "5%: expected by chance if |z| were normal"],
        )
        ax = c.axes()
        _y_grid(ax)
        shares = [share for share, _ in cells.values()]
        top = max([*shares, 0.05]) * 1.22
        ax.set_ylim(0, top)
        _pct_axis(ax.yaxis, signed=False)
        if top > 1:
            # The headroom is for the value labels; a share cannot pass 100%, so no tick may either.
            ax.yaxis.set_major_locator(FixedLocator([0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        ax.set_xlim(-0.6, len(windows) - 0.4)
        c.place(ax, right_in=0.45)

        axis_w_in = ax.get_position().width * WIDE[0]
        gap = (2 / 96) / (axis_w_in / (len(windows) + 0.2))
        width = min(0.3, (BAR_MAX_IN * 1.6) / (axis_w_in / (len(windows) + 0.2)))
        for i, window in enumerate(windows):
            for sample, color, sign in (("events", BULLISH, -1), ("placebo", REFERENCE, 1)):
                cell = cells.get((window, sample))
                if cell is None:
                    continue
                x = i + sign * (width + gap) / 2
                ax.bar(x, cell[0], width=width, color=color, linewidth=0, zorder=3)
                # A zero-height bar is invisible; say 0% so it does not read as missing.
                if window == "event" or cell[0] == 0:
                    ax.annotate(
                        f"{cell[0] * 100:.1f}%",
                        xy=(x, cell[0]),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=LABEL_SIZE,
                        color=INK,
                        # Shares near 5% are the expected null result: the chance line must not strike them out.
                        bbox={"boxstyle": "square,pad=0.12", "facecolor": SURFACE, "edgecolor": "none"},
                        zorder=5,
                    )
        ax.axhline(0.05, color=INK_2, linewidth=HAIRLINE * 1.25, zorder=4)
        ax.annotate(
            "5%",
            xy=(1, 0.05),
            xycoords=("axes fraction", "data"),
            xytext=(4, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=TICK_SIZE,
            color=INK_2,
        )
        ticks = []
        for window in windows:
            start, end = DAILY_WINDOWS[window]
            days = f"days {start:+d} to {end:+d}".replace("+0", "0").replace("-", "−")
            n_ev = cells.get((window, "events"), (0, 0))[1]
            n_pl = cells.get((window, "placebo"), (0, 0))[1]
            ticks.append(f"{window}: {days}\nn={n_ev} events / {n_pl} placebo")
        ax.set_xticks(range(len(windows)), labels=ticks, fontsize=TICK_SIZE, color=MUTED)
        ax.tick_params(axis="x", length=0, pad=5)
        _zero_line(ax)
        c.place(ax, bottom_in=0.62, right_in=0.45)
        return c.save(out)


def top_event_chart(event: Mapping[str, object], prices: EventPrices | None, out: Path) -> Path | None:
    """One event's price move around the post, with SPY in gray on the same % axis. Title: ticker, author, New York
    time; subtitle: stance and CAR[0,+1] / z."""
    if prices is None or prices.ticker.dropna().size < 2:
        return None
    ticker = str(event.get("ticker", ""))
    stance = event.get("stance")
    stance = str(stance) if isinstance(stance, str) and stance else "unlabelled"
    color = stance_color(stance)

    with mpl.rc_context(_RC):
        c = _Canvas(SMALL)
        c.header(
            f"{ticker} · {event.get('author', '')} · {format_ny(prices.t0)}",
            f"{stance} post · CAR[0,+1] {pct_text(event.get('car_event'))} "
            f"(z {_z_text(event.get('z_event'))}) · "
            + (
                "% change since the price at the post"
                if prices.kind == "intraday"
                else f"% change since the close {DAILY_SPAN} sessions before"
            ),
        )
        raw_series, raw_spy = prices.ticker.dropna(), prices.spy.dropna()
        regular = prices.regular if prices.kind == "intraday" else None
        stray = raw_series[isolated_prints(raw_series, regular)]
        stray_spy = raw_spy[isolated_prints(raw_spy, regular)]
        series, spy = raw_series.drop(stray.index), raw_spy.drop(stray_spy.index)
        lo, hi = _padded_limits([*series, *spy])
        handles, labels = [_handle(color, markers=False)], [ticker]
        if not spy.empty:
            handles.append(_handle(REFERENCE, markers=False))
            labels.append("SPY")
        off_hours = _off_hours_spans(prices)
        if off_hours:
            handles.append(Patch(color=OFF_HOURS))
            labels.append("outside regular hours (thin trading)")
        strays = pd.concat([stray, stray_spy])
        if not strays.empty:
            handles.append(_stray_handle(INK_2, "o"))
            off_scale = bool(((strays < lo) | (strays > hi)).any())
            tail = "; arrowheads are off the scale" if off_scale else " to the line"
            labels.append(f"stray one-minute prints, not joined{tail}")
        c.legend(handles, labels)
        ax = c.axes()
        _y_grid(ax)
        _zero_line(ax)
        drawstyle = "steps-post" if prices.kind == "intraday" else "default"
        if not spy.empty:
            ax.plot(
                spy.index, spy.to_numpy(float), color=REFERENCE, linewidth=LINE_W * 0.8, zorder=2.5, drawstyle=drawstyle
            )
        ax.plot(
            series.index,
            series.to_numpy(float),
            color=color,
            linewidth=LINE_W,
            zorder=3,
            drawstyle=drawstyle,
            solid_joinstyle="round",
        )
        last_x, last_y = float(series.index[-1]), float(series.iloc[-1])
        ax.plot(
            [last_x],
            [last_y],
            marker="o",
            markersize=MARKER,
            color=color,
            markeredgecolor=SURFACE,
            markeredgewidth=RING,
            zorder=4,
        )
        ax.set_ylim(lo, hi)
        _pct_axis(ax.yaxis)
        _stray_marks(ax, stray, color, lo, hi)
        _stray_marks(ax, stray_spy, REFERENCE, lo, hi)

        if prices.kind == "intraday":
            x_lo, x_hi = float(min(series.index.min(), 0.0)), float(max(series.index.max(), 0.0))
            ax.set_xlim(x_lo - 2, x_hi + 2)
            ticks = _clock_ticks(prices.t0, x_lo, x_hi)
            ax.xaxis.set_major_locator(FixedLocator([m for m, _ in ticks]))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p, t=dict(ticks): t.get(v, "")))
            ax.set_xlabel("New York time", fontsize=LABEL_SIZE, color=MUTED, labelpad=3)
            for start, end in off_hours:
                ax.axvspan(start, end, color=OFF_HOURS, linewidth=0, zorder=0.5)
            _post_rule(ax, 0, f"post {prices.t0.astimezone(NY):%H:%M}")
        else:
            ax.set_xlim(-DAILY_SPAN - 0.4, DAILY_SPAN + 0.4)
            ax.xaxis.set_major_locator(FixedLocator(list(range(-DAILY_SPAN, DAILY_SPAN + 1, 2))))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:+.0f}".replace("-", "−") if v else "0"))
            ax.set_xlabel("trading days from the post day", fontsize=LABEL_SIZE, color=MUTED, labelpad=3)
            _post_rule(ax, 0, "post day")
        end_label = f"{ticker} {pct_text(last_y)}"
        c.place(ax, bottom_in=0.55, right_in=c.text_width_in(end_label, TICK_SIZE) + 0.25)
        ax.annotate(
            end_label,
            xy=(last_x, last_y),
            xytext=(7, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=TICK_SIZE,
            color=INK,
        )
        return c.save(out)


_STRAY_STYLE = {
    "linestyle": "none",
    "markersize": MARKER * 0.7,
    "markerfacecolor": SURFACE,
    "markeredgewidth": RING * 0.75,
}


def _stray_handle(color: str, marker: str) -> Line2D:
    return Line2D([], [], marker=marker, markeredgecolor=color, **_STRAY_STYLE)


def _stray_marks(ax, stray: pd.Series, color: str, lo: float, hi: float) -> None:
    """Hollow marks where isolated prints are; one beyond the scale sits on the chart edge as an arrowhead."""
    for x, y in stray.items():
        marker, at = ("o", float(y)) if lo <= y <= hi else (("v", lo) if y < lo else ("^", hi))
        ax.plot([float(x)], [at], marker=marker, markeredgecolor=color, zorder=4, clip_on=False, **_STRAY_STYLE)


def _off_hours_spans(prices: EventPrices) -> list[tuple[float, float]]:
    """Parts of an intraday chart's x range (minutes from t0) outside the regular session."""
    series = prices.ticker.dropna()
    if prices.kind != "intraday" or prices.regular is None or series.empty:
        return []
    lo, hi = float(min(series.index.min(), 0.0)), float(max(series.index.max(), 0.0))
    open_, close = prices.regular
    spans = [(lo - 2, min(open_, hi + 2)), (max(close, lo - 2), hi + 2)]
    return [(a, b) for a, b in spans if b > a]


def _z_text(value: object) -> str:
    return f"{float(value):+.2f}".replace("-", "−") if is_number(value) else "—"  # type: ignore[arg-type]


def _clock_ticks(t0: datetime, lo_min: float, hi_min: float) -> list[tuple[float, str]]:
    """Ticks at whole and half hours of New York time, as minutes from t0."""
    local = (t0 + timedelta(minutes=lo_min)).astimezone(NY)
    first = math.ceil((local.hour * 60 + local.minute + local.second / 60) / 30) * 30
    tick = local.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(minutes=first)
    out: list[tuple[float, str]] = []
    while (tick - t0).total_seconds() / 60 <= hi_min:
        out.append(((tick - t0).total_seconds() / 60, f"{tick:%H:%M}"))
        tick += timedelta(minutes=30)
    return out
