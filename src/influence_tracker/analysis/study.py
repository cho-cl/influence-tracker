"""The contract between the event-study computation (metrics.py) and the report (report.py).

metrics.compute_study() fills a Study; report.write_report() only reads it (plus raw bars for price charts).
Every DataFrame carries exactly the columns listed here, in this order; missing values are NaN / None.
Returns are simple returns as fractions (0.012 = +1.2%). Windows are in trading sessions relative to d0.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd

# Daily windows, all market-model abnormal returns cumulated over sessions relative to d0.
#   pre   = CAR[-5,-1]  did the stock already move before the post? (move vs react)
#   event = CAR[0,+1]   the event window (headline metric)
#   post  = CAR[+2,+5]  did the move continue or reverse afterwards?
DAILY_WINDOWS: dict[str, tuple[int, int]] = {"pre": (-5, -1), "event": (0, 1), "post": (2, 5)}
PATH_DAYS: tuple[int, ...] = tuple(range(-5, 6))
# Intraday windows (regular-session posts with 1m data), named as in event_windows.
INTRADAY_WINDOWS: tuple[str, ...] = ("pre60", "p5", "p15", "p30", "p60")
INTRADAY_PATH_MINUTES: tuple[int, ...] = (-60, 0, 5, 15, 30, 60)

ESTIMATION_WINDOW: tuple[int, int] = (-130, -11)  # market-model estimation sessions relative to d0 (120 sessions)
MIN_ESTIMATION_OBS = 60
PLACEBO_RANGE: tuple[int, int] = (-130, -10)  # placebo days are drawn from these sessions before d0
PLACEBO_CLEARANCE = 5  # no event on the same ticker within +-5 sessions of a placebo day
INTRADAY_SIGMA_SESSIONS = 20
MIN_INTRADAY_SIGMA_OBS = 10
MIN_GROUP_POSTS = 10
Z_CRITICAL = 1.96

# Reddit posts have no watchlist account: their "author" group is the subreddit and their category is this.
REDDIT_CATEGORY = "reddit_crowd"

EXCLUSION_REASONS = ("earnings", "split", "clustered", "no_model")

EVENT_COLUMNS: tuple[str, ...] = (
    "event_id", "platform", "native_id", "author", "category", "ticker",
    "t0", "t0_et", "d0", "session_phase",
    "stance", "stance_conf", "sign",            # sign: +1 bullish, -1 bearish, 0 neutral, NaN unlabelled
    "excluded_reason",                          # '' when included, else one of EXCLUSION_REASONS
    "earnings_flag", "split_flag", "clustered", "intraday_state",
    "alpha", "beta", "sigma", "n_est",          # market model r_i = alpha + beta * r_SPY; sigma = residual std
    "ar_dm5", "ar_dm4", "ar_dm3", "ar_dm2", "ar_dm1", "ar_d0", "ar_dp1", "ar_dp2", "ar_dp3", "ar_dp4", "ar_dp5",
    "car_pre", "z_pre", "car_event", "z_event", "car_post", "z_post",
    "ar_pre_leg", "z_pre_leg", "ar_post_leg", "z_post_leg",   # day-0 split at the post (intraday_state == 'ok')
    "iar_pre60", "iz_pre60", "iar_p5", "iz_p5", "iar_p15", "iz_p15", "iar_p30", "iz_p30", "iar_p60", "iz_p60",
    "rel_volume",                               # d0 volume / mean volume over sessions [-25, -6]
    "spy_ret_d0", "spy_post_leg",               # raw SPY moves, for market-wide (politician) posts
    "text", "url",
)  # fmt: skip

POST_COLUMNS: tuple[str, ...] = (
    "platform", "native_id", "author", "category", "t0", "t0_et", "stance", "stance_conf", "sign",
    "n_events", "tickers",
    # means over the post's included events; signed_* = sign * value (NaN when sign is 0 or NaN)
    "car_pre", "car_event", "car_post", "z_pre", "z_event", "z_post",
    "signed_car_pre", "signed_car_event", "signed_car_post",
    "ar_dm5", "ar_dm4", "ar_dm3", "ar_dm2", "ar_dm1", "ar_d0", "ar_dp1", "ar_dp2", "ar_dp3", "ar_dp4", "ar_dp5",
    "ar_pre_leg", "ar_post_leg",
    "iar_pre60", "iar_p5", "iar_p15", "iar_p30", "iar_p60",
    "text", "url",
)  # fmt: skip

# Directional (stance-signed) tests, one row per (family, group, subset, window). Unit of observation: the post.
#   family: 'all' | 'author' | 'category' | 'platform' | 'stance'
#   subset: 'main' | 'confident' (stance_conf >= StudyOptions.min_conf_robust)
#   status: 'ok' | 'insufficient' (n_posts < MIN_GROUP_POSTS: no test, p-values NaN)
#   p_holm: Holm-adjusted within the same (family, subset, window) across groups with status 'ok'
GROUP_COLUMNS: tuple[str, ...] = (
    "family", "group", "subset", "window", "n_posts",
    "mean_signed_car", "median_signed_car", "ci_low", "ci_high", "mean_abs_z", "share_abs_z_gt_crit",
    "t_stat", "p_value", "p_holm", "status",
)  # fmt: skip

# Magnitude, unsigned, at the EVENT level: real events vs placebo days, per daily window.
#   sample: 'events' | 'placebo'
MAGNITUDE_COLUMNS: tuple[str, ...] = (
    "window", "sample", "n", "mean_car", "mean_abs_car", "median_abs_car", "share_abs_z_gt_crit",
)  # fmt: skip

# Cumulative abnormal return from day -5 through `day`, averaged per series with a 95% CI.
#   series: 'bullish' | 'bearish' | 'neutral' (post level, unsigned) | 'placebo' (placebo-day level)
PATH_COLUMNS: tuple[str, ...] = ("series", "day", "mean_car", "ci_low", "ci_high", "n")

# Intraday path relative to the reference price (0 at minute 0): -iar_pre60 at -60, iar_pN at +N.
INTRADAY_PATH_COLUMNS: tuple[str, ...] = ("series", "minute", "mean_ar", "ci_low", "ci_high", "n")

# Reddit attention (ApeWisdom 'all-stocks'): mention spike vs the next session's abnormal return.
ATTENTION_COLUMNS: tuple[str, ...] = (
    "ticker", "snapshot_date", "mentions", "trailing_mean", "spike", "next_session", "next_ar", "next_z",
)  # fmt: skip


@dataclass(frozen=True)
class StudyOptions:
    since: date | None = None  # d0 >= since
    until: date | None = None  # d0 <= until
    include_earnings: bool = False
    include_splits: bool = False
    include_clustered: bool = False
    min_conf_robust: float = 0.6
    placebo_draws: int = 5
    seed: int = 20260928


@dataclass
class Study:
    options: StudyOptions
    generated_at: datetime
    events: pd.DataFrame  # EVENT_COLUMNS, every complete event in the date range, included or excluded
    posts: pd.DataFrame  # POST_COLUMNS, posts with >= 1 included event
    groups: pd.DataFrame  # GROUP_COLUMNS
    magnitude: pd.DataFrame  # MAGNITUDE_COLUMNS
    car_path: pd.DataFrame  # PATH_COLUMNS
    intraday_path: pd.DataFrame  # INTRADAY_PATH_COLUMNS
    attention: pd.DataFrame  # ATTENTION_COLUMNS, possibly empty
    # events_complete, events_included, excluded_<reason> for each reason, earnings_unknown, events_intraday,
    # posts_included, posts_signed, placebo_days, attention_days
    counts: dict[str, int] = field(default_factory=dict)
    # Plain-language data-coverage caveats for the report ("ApeWisdom: 2 days collected, 8 needed ...").
    notes: list[str] = field(default_factory=list)
