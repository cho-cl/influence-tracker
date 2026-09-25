"""`influence events`: list, inspect and export events so they can be checked by hand against price charts."""

from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from rich import box
from rich.console import Console, JustifyMethod
from rich.table import Table
from rich.text import Text

from .config import Ticker
from .logsetup import make_console
from .status import _clip, _row
from .timeutil import NY, from_iso

WINDOWS = ("pre_leg", "post_leg", "pre60", "p5", "p15", "p30", "p60")
DEFAULT_LIMIT = 30
SNIPPET_CHARS = 60
# Redirected output has no terminal to fit, so don't squeeze the table into 160 columns.
FILE_WIDTH = 250
FLAG_LEGEND = "flags: E earnings  S split  C clustered  O spans open  U intraday unavailable  ? earnings unknown"

_GAP = 2  # cells between columns in the list table (collapsed padding + blank divider)
_TEXT_MIN_WIDTH = 20
_WINDOW_FIELDS = (
    "ret",
    "spy_ret",
    "start_et",
    "end_et",
    "start_price",
    "end_price",
    "spy_start_price",
    "spy_end_price",
    "truncated",
)
_WINDOW_COLUMNS: tuple[tuple[str, JustifyMethod], ...] = (
    ("window", "left"),
    ("start bar", "left"),
    ("start", "right"),
    ("end bar", "left"),
    ("end", "right"),
    ("return", "right"),
    ("SPY start", "right"),
    ("SPY end", "right"),
    ("SPY return", "right"),
    ("truncated", "left"),
)
# Excel runs a cell that starts with one of these as a formula; post text is untrusted.
_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")
_STANCE_STYLE = {"bullish": "green", "bearish": "red"}
_STATUS_STYLE = {"pending": "yellow", "complete": "green"}

_SELECT = """SELECT e.*, p.author AS post_author, p.text AS post_text, p.url AS post_url,
                    p.stance AS post_stance, p.stance_conf AS post_stance_conf, p.stance_model AS post_stance_model
             FROM events e LEFT JOIN posts p ON p.platform = e.platform AND p.native_id = e.native_id"""


@dataclass(frozen=True)
class EventFilter:
    ticker: str | None = None
    platform: str | None = None
    status: str | None = None

    def _pairs(self) -> list[tuple[str, str]]:
        ticker = self.ticker.upper() if self.ticker else None
        pairs = (("ticker", ticker), ("platform", self.platform), ("status", self.status))
        return [(k, v) for k, v in pairs if v]

    def sql(self) -> tuple[str, list[object]]:
        """A WHERE clause over `events e` (empty when nothing is filtered) and its parameters."""
        pairs = self._pairs()
        if not pairs:
            return "", []
        return " WHERE " + " AND ".join(f"e.{k} = ?" for k, _ in pairs), [v for _, v in pairs]

    def describe(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self._pairs())


def events_console() -> Console:
    console = make_console()
    if not console.is_terminal:
        console.width = max(console.width, FILE_WIDTH)
    return console


# ---------------------------------------------------------------- queries


def _event_columns(conn: sqlite3.Connection) -> list[str]:
    return [r["name"] for r in conn.execute("PRAGMA table_info(events)")]


def _fetch_events(conn: sqlite3.Connection, flt: EventFilter, limit: int | None) -> list[sqlite3.Row]:
    where, params = flt.sql()
    sql = f"{_SELECT}{where} ORDER BY e.t0 DESC, e.id DESC"
    if limit is not None:
        sql += " LIMIT ?"
        params.append(limit)
    return conn.execute(sql, params).fetchall()


def _fetch_windows(conn: sqlite3.Connection, where: str, params: list[object]) -> dict[int, dict[str, sqlite3.Row]]:
    out: dict[int, dict[str, sqlite3.Row]] = {}
    for w in conn.execute(f"SELECT w.* FROM event_windows w JOIN events e ON e.id = w.event_id{where}", params):
        out.setdefault(w["event_id"], {})[w["win"]] = w
    return out


def _window_order(names: set[str]) -> list[str]:
    return [*WINDOWS, *sorted(names - set(WINDOWS))]


# ---------------------------------------------------------------- formatting


def _bar_start(epoch: int) -> datetime:
    return datetime.fromtimestamp(epoch, UTC).astimezone(NY)


def format_t0(iso: str) -> str:
    """'Sep 24 10:31:07 ET' in New York time, EDT or EST as the date requires."""
    return from_iso(iso).astimezone(NY).strftime("%b %d %H:%M:%S ET")


def _stamp(ts: datetime) -> str:
    return ts.astimezone(NY).strftime("%Y-%m-%d %H:%M:%S")


def _pct(value: float | None, truncated: bool = False) -> str:
    if value is None:
        return "-"
    return f"{value * 100:+.2f}%" + ("*" if truncated else "")


def _price(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}" if abs(value) >= 1 else f"{value:.4f}"


def _stance(stance: str | None, conf: float | None) -> Text:
    if not stance:
        return Text("-")
    label = stance if conf is None else f"{stance} {conf:.2f}"
    return Text(label, style=_STANCE_STYLE.get(stance, ""))


def _ref(row: sqlite3.Row) -> str:
    if row["ref_ts"] is None or row["ref_price"] is None:
        return "-"
    bar = _bar_start(row["ref_ts"])
    posted = from_iso(row["t0"]).astimezone(NY)
    # A weekend or overnight post takes its reference from an earlier day's last bar; say which day.
    clock = bar.strftime("%H:%M") if bar.date() == posted.date() else bar.strftime("%a %H:%M")
    return f"{_price(row['ref_price'])} @{clock}"


def flags(row: sqlite3.Row) -> str:
    marks = (
        ("E", row["earnings_flag"] == 1),
        ("S", row["split_flag"] == 1),
        ("C", bool(row["clustered"])),
        ("O", bool(row["spans_open"])),
        ("U", row["intraday_state"] == "unavailable"),
        # Flags are only set at completion, so NULL on a pending event is not "unknown" yet.
        ("?", row["status"] == "complete" and row["earnings_flag"] is None),
    )
    return "".join(letter for letter, on in marks if on)


def _window_ret(wins: dict[str, sqlite3.Row], name: str, spy: bool = False) -> str:
    w = wins.get(name)
    if w is None:
        return "-"
    return _pct(w["spy_ret"] if spy else w["ret"], bool(w["truncated"]))


def _yes_no(value: object) -> str:
    return "-" if value is None else ("yes" if value else "no")


# ---------------------------------------------------------------- list


@dataclass(frozen=True)
class _Column:
    header: str
    justify: JustifyMethod = "left"
    # Columns that give way, lowest first, when the console is too narrow; None = always shown.
    drop_order: int | None = None


_LIST_COLUMNS = (
    _Column("id", "right"),
    _Column("platform", drop_order=1),
    _Column("@author", drop_order=9),
    _Column("ticker"),
    _Column("t0"),
    _Column("phase", drop_order=2),
    _Column("d0", drop_order=8),
    _Column("stance", drop_order=6),
    _Column("ref"),
    _Column("pre_leg", "right"),
    _Column("post_leg", "right"),
    _Column("p15", "right"),
    _Column("SPY p15", "right", drop_order=3),
    _Column("status", drop_order=5),
    _Column("flags", drop_order=4),
    _Column("text", drop_order=7),
)


def _list_cells(row: sqlite3.Row, wins: dict[str, sqlite3.Row]) -> list[Text]:
    cells: list[str | Text] = [
        str(row["id"]),
        row["platform"],
        f"@{row['post_author']}" if row["post_author"] else "-",
        row["ticker"],
        format_t0(row["t0"]),
        row["session_phase"],
        row["d0"],
        _stance(row["post_stance"], row["post_stance_conf"]),
        _ref(row),
        _window_ret(wins, "pre_leg"),
        _window_ret(wins, "post_leg"),
        _window_ret(wins, "p15"),
        _window_ret(wins, "p15", spy=True),
        Text(row["status"], style=_STATUS_STYLE.get(row["status"], "")),
        flags(row),
        _clip(row["post_text"], SNIPPET_CHARS),
    ]
    # Plain str cells would be parsed as rich markup; post text like "[/x]" must print literally.
    return [Text(c) if isinstance(c, str) else c for c in cells]


def _plain_table() -> Table:
    return Table(box=box.SIMPLE_HEAD, show_edge=False, pad_edge=False, collapse_padding=True, header_style="bold")


def _fit_columns(rows: list[list[Text]], width: int) -> tuple[list[int], int, int]:
    """Drop low-priority columns until the table fits `width`. Returns the indexes of the columns to show, the width
    left for the snippet (cut to fit so every event stays on one line) and the width the shown columns need, which
    exceeds `width` only when even the columns that are never dropped don't fit."""
    text = next(i for i, col in enumerate(_LIST_COLUMNS) if col.header == "text")
    widths = [max([len(col.header), *(row[i].cell_len for row in rows)]) for i, col in enumerate(_LIST_COLUMNS)]
    widths[text] = min(widths[text], _TEXT_MIN_WIDTH)
    shown = list(range(len(_LIST_COLUMNS)))
    droppable = sorted(
        (i for i, col in enumerate(_LIST_COLUMNS) if col.drop_order is not None),
        key=lambda i: _LIST_COLUMNS[i].drop_order,
    )

    def needed() -> int:
        return sum(widths[j] for j in shown) + _GAP * (len(shown) - 1)

    for i in droppable:
        if needed() <= width:
            break
        shown.remove(i)
    others = sum(widths[j] for j in shown if j != text) + _GAP * (len(shown) - 1)
    return shown, max(width - others, _TEXT_MIN_WIDTH), needed()


def show_events(
    conn: sqlite3.Connection, flt: EventFilter, limit: int = DEFAULT_LIMIT, console: Console | None = None
) -> int:
    """Print the newest `limit` events matching `flt` as a table. Returns how many were shown."""
    console = console or events_console()
    where, params = flt.sql()
    total = conn.execute(f"SELECT COUNT(*) FROM events e{where}", params).fetchone()[0]
    described = flt.describe()
    if total == 0:
        if described:
            console.print(Text(f"No events match {described}."))
        else:
            console.print(Text("No events yet. 'influence enrich' (part of 'influence daily') builds them."))
        return 0

    rows = _fetch_events(conn, flt, limit)
    windows = _fetch_windows(conn, where, params)
    cells = [_list_cells(r, windows.get(r["id"], {})) for r in rows]
    shown, text_width, needed = _fit_columns(cells, console.width)
    # Too narrow even for the core columns: wrap rows, since rich would otherwise cut ids and returns to "…".
    wrap = needed > console.width

    table = _plain_table()
    for i in shown:
        col = _LIST_COLUMNS[i]
        max_width = text_width if col.header == "text" else None
        overflow = "fold" if wrap else "ellipsis"
        table.add_column(col.header, justify=col.justify, no_wrap=not wrap, overflow=overflow, max_width=max_width)
    for row_cells in cells:
        table.add_row(*(row_cells[i] for i in shown))

    title = f"Events, newest t0 first: {len(rows)} of {total}"
    console.print(Text(title + (f" ({described})" if described else ""), style="bold cyan"))
    console.print(table)
    notes = [
        "times are New York (ET); ref = close of the 1-minute bar starting at the time shown; "
        "* = window truncated at the session close",
        FLAG_LEGEND,
    ]
    hidden = [_LIST_COLUMNS[i].header for i in range(len(_LIST_COLUMNS)) if i not in shown]
    if hidden:
        notes.append(f"hidden to fit the terminal: {', '.join(hidden)} (widen it, or use --csv)")
    if wrap:
        notes.append(f"rows wrap: {needed} columns are needed to keep each event on one line")
    notes.append("'influence events --id N' shows one event in full; '--csv PATH' exports them all")
    for note in notes:
        console.print(Text(note, style="dim"))
    return len(rows)


# ---------------------------------------------------------------- one event


def _field(row: sqlite3.Row, name: str) -> str:
    value = row[name]
    if value is None:
        if name in ("earnings_flag", "split_flag"):
            return "unknown" if row["status"] == "complete" else "not set until the event completes"
        return "-"
    if name == "t0":
        return f"{format_t0(value)}  ({value})"
    if name == "d0":
        return f"{value} ({date.fromisoformat(value):%a})"
    if name == "ref_ts":
        return f"{value}  (bar start {_bar_start(value):%a %b %d %H:%M} ET)"
    if name == "ref_price":
        return _price(value)
    if name in ("created_at", "completed_at"):
        return f"{_stamp(from_iso(value))} ET"
    if name in ("earnings_flag", "split_flag", "clustered", "spans_open"):
        return _yes_no(value)
    return str(value)


def _check_hint(row: sqlite3.Row) -> str:
    if row["ref_ts"] is not None and row["ref_price"] is not None:
        bar = _bar_start(row["ref_ts"])
        yahoo = Ticker(symbol=row["ticker"]).yahoo_symbol
        return (
            f"To check it: open a 1-minute {yahoo} chart for {bar:%a %Y-%m-%d} with pre/post-market shown and read "
            f"the close of the {bar:%H:%M} ET bar; it should be {_price(row['ref_price'])}. Every price above is the "
            "close of the bar that starts at the time shown."
        )
    if row["intraday_state"] == "unavailable":
        return "No reference price: 1-minute bars for this event were unavailable, so it has daily data only."
    return "No reference price yet: enrich sets it once the 1-minute bars around t0 are stored."


def show_event(conn: sqlite3.Connection, event_id: int, console: Console | None = None) -> bool:
    """Print every field of one event, its windows and its post. False if there is no such event."""
    console = console or events_console()
    row = conn.execute(f"{_SELECT} WHERE e.id = ?", (event_id,)).fetchone()
    if row is None:
        return False
    wins = _fetch_windows(conn, " WHERE e.id = ?", [event_id]).get(event_id, {})

    author = f"@{row['post_author']}" if row["post_author"] else "(post not found)"
    console.rule(Text(f"Event {event_id}: {row['ticker']}, {author} on {row['platform']}"))
    fields = Table.grid(padding=(0, 2))
    fields.add_column(style="bold")
    fields.add_column(overflow="fold")
    for name in _event_columns(conn):
        _row(fields, name, _field(row, name))
    _row(fields, "flags", flags(row) or "-")
    console.print(fields)

    console.print()
    console.print(Text("Post", style="bold cyan"))
    post = Table.grid(padding=(0, 2))
    post.add_column(style="bold")
    post.add_column(overflow="fold")
    stance = _stance(row["post_stance"], row["post_stance_conf"])
    if row["post_stance_model"]:
        stance.append(f"  ({row['post_stance_model']})", style="dim")
    _row(post, "author", author)
    _row(post, "stance", stance)
    _row(post, "url", row["post_url"] or "-")
    console.print(post)
    console.print(Text(row["post_text"] if row["post_text"] is not None else "(post not found)"))

    console.print()
    if not wins:
        console.print(Text("Windows: none stored for this event.", style="bold cyan"))
    else:
        console.print(Text("Windows (bar start times, New York; prices are those bars' closes)", style="bold cyan"))
        table = _plain_table()
        for header, justify in _WINDOW_COLUMNS:
            table.add_column(header, justify=justify, no_wrap=True)
        for name in _window_order(set(wins)):
            w = wins.get(name)
            if w is None:
                continue
            _row(
                table,
                name,
                f"{_bar_start(w['start_ts']):%a %b %d %H:%M}",
                _price(w["start_price"]),
                f"{_bar_start(w['end_ts']):%a %b %d %H:%M}",
                _price(w["end_price"]),
                _pct(w["ret"]),
                _price(w["spy_start_price"]),
                _price(w["spy_end_price"]),
                _pct(w["spy_ret"]),
                _yes_no(w["truncated"]),
            )
        console.print(table)

    console.print()
    console.print(Text(_check_hint(row), style="italic"))
    return True


# ---------------------------------------------------------------- CSV


def _excel_safe(value: object) -> object:
    if isinstance(value, str) and value.startswith(_FORMULA_START):
        return "'" + value
    return value


def _excel_id(native_id: str) -> object:
    """X (19-digit) and Truth Social (18-digit) ids as the ="..." text formula: Excel reads a bare digit string as a
    number and keeps only 15 significant digits. Only ever built from ASCII digits, so it can't carry a formula."""
    if native_id.isascii() and native_id.isdigit():
        return f'="{native_id}"'
    return _excel_safe(native_id)


def _window_values(w: sqlite3.Row | None) -> list[object]:
    if w is None:
        return [None] * len(_WINDOW_FIELDS)
    return [
        w["ret"],
        w["spy_ret"],
        _stamp(_bar_start(w["start_ts"])),
        _stamp(_bar_start(w["end_ts"])),
        w["start_price"],
        w["end_price"],
        w["spy_start_price"],
        w["spy_end_price"],
        w["truncated"],
    ]


def export_csv(conn: sqlite3.Connection, path: Path, flt: EventFilter, limit: int | None = None) -> int:
    """Write one row per matching event (newest t0 first) to `path`. Returns the number of rows written.

    utf-8-sig: without the BOM, Excel on Windows reads the file as cp1252 and mangles emoji and accents. Numeric
    native_ids are written as ="<id>" so Excel shows them exactly; the url column carries the id as well."""
    columns = _event_columns(conn)
    id_col = columns.index("native_id")
    rows = _fetch_events(conn, flt, limit)
    where, params = flt.sql()
    windows = _fetch_windows(conn, where, params)
    names = _window_order({name for wins in windows.values() for name in wins})
    header = [*columns, "t0_et", "ref_time_et", "author", "stance", "stance_conf", "stance_model"]
    header += [f"{name}_{field}" for name in names for field in _WINDOW_FIELDS]
    header += ["url", "text"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            wins = windows.get(r["id"], {})
            values: list[object] = [r[c] for c in columns]
            values += [
                _stamp(from_iso(r["t0"])),
                _stamp(_bar_start(r["ref_ts"])) if r["ref_ts"] is not None else None,
                r["post_author"],
                r["post_stance"],
                r["post_stance_conf"],
                r["post_stance_model"],
            ]
            for name in names:
                values += _window_values(wins.get(name))
            values += [r["post_url"], r["post_text"]]
            cells = [_excel_safe(v) for v in values]
            cells[id_col] = _excel_id(r["native_id"])
            writer.writerow(cells)
    return len(rows)
