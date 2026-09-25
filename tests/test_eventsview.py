from __future__ import annotations

import csv
import io
import re
from datetime import datetime

import pytest
from rich.console import Console

from influence_tracker import eventsview
from influence_tracker.eventsview import EventFilter
from influence_tracker.timeutil import NY

CREATED = "2026-09-25T00:31:00Z"
LONG_TEXT = (
    "Tesla is going to CRUSH it this quarter 🚀🚀 robotaxi everywhere, buying more $TSLA today and holding forever"
)
MARKUP_TEXT = "[bold red]BUY INTEL[/bold red] now! [/x]"
MULTILINE_TEXT = "GME squeeze is over.\n\nPuts printing 📉 café"
FORMULA_TEXT = '=HYPERLINK("http://evil.example","NVDA") nvidia still cheap'


def _et(y: int, mo: int, d: int, h: int, mi: int) -> int:
    """Epoch seconds of a New York wall-clock minute."""
    return int(datetime(y, mo, d, h, mi, tzinfo=NY).timestamp())


def _post(conn, platform, native_id, author, text, stance=None, conf=None) -> None:
    conn.execute(
        """INSERT INTO posts (platform, native_id, author, created_at_utc, text, url, stance, stance_conf,
                              stance_model, collected_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            platform,
            native_id,
            author,
            CREATED,
            text,
            f"https://example.com/{platform}/{native_id}",
            stance,
            conf,
            "StephanAkkerman/FinTwitBERT-sentiment" if stance else None,
            CREATED,
        ),
    )


def _event(conn, platform, native_id, ticker, t0, d0, phase, status, **extra) -> int:
    values = {
        "platform": platform,
        "native_id": native_id,
        "ticker": ticker,
        "t0": t0,
        "d0": d0,
        "session_phase": phase,
        "status": status,
        "created_at": CREATED,
        **extra,
    }
    cur = conn.execute(
        f"INSERT INTO events ({', '.join(values)}) VALUES ({', '.join('?' * len(values))})", list(values.values())
    )
    return int(cur.lastrowid)


def _window(conn, event_id, win, start_ts, end_ts, start_price, end_price, ret, spy=(None, None, None), truncated=0):
    conn.execute(
        """INSERT INTO event_windows (event_id, win, start_ts, end_ts, start_price, end_price, ret,
                                      spy_start_price, spy_end_price, spy_ret, truncated)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (event_id, win, start_ts, end_ts, start_price, end_price, ret, *spy, truncated),
    )


@pytest.fixture
def ids(conn) -> dict[str, int]:
    """Five hand-built events; newest t0 first they are B, A, D, E, C."""
    with conn:
        _post(conn, "x", "1001", "elonmusk", LONG_TEXT, "bullish", 0.87)
        a = _event(
            conn, "x", "1001", "TSLA", "2026-09-24T14:31:07Z", "2026-09-24", "regular", "complete",
            intraday_state="ok", ref_ts=_et(2026, 9, 24, 10, 30), ref_price=182.41,
            earnings_flag=1, split_flag=0, clustered=1, completed_at="2026-10-01T00:30:00Z",
        )  # fmt: skip
        ref = _et(2026, 9, 24, 10, 30)
        _window(conn, a, "pre_leg", _et(2026, 9, 23, 15, 59), ref, 180.0, 182.41, 0.013389, (660.0, 661.0, 0.001515))
        _window(conn, a, "post_leg", ref, _et(2026, 9, 24, 15, 59), 182.41, 180.0, -0.013212, (661.0, 662.5, 0.002269))
        _window(conn, a, "pre60", _et(2026, 9, 24, 9, 30), ref, 181.0, 182.41, 0.00779)
        _window(conn, a, "p5", ref, _et(2026, 9, 24, 10, 35), 182.41, 182.9, 0.002686)
        _window(conn, a, "p15", ref, _et(2026, 9, 24, 10, 45), 182.41, 183.0, 0.003234, (661.0, 660.67, -0.0005))
        _window(conn, a, "p30", ref, _et(2026, 9, 24, 11, 0), 182.41, 183.5, 0.005975)
        _window(conn, a, "p60", ref, _et(2026, 9, 24, 11, 30), 182.41, 184.0, 0.008717)

        # After the Nov 1 DST switch New York is UTC-5: 15:00:05Z is 10:00:05 ET.
        _post(conn, "truthsocial", "115000000000000001", "realDonaldTrump", MARKUP_TEXT)
        b = _event(
            conn, "truthsocial", "115000000000000001", "INTC", "2026-11-02T15:00:05Z", "2026-11-02", "regular",
            "pending", ref_ts=_et(2026, 11, 2, 9, 59), ref_price=24.5,
        )  # fmt: skip

        # Saturday post: d0 is Monday and the reference bar is Friday's last extended-hours bar.
        _post(conn, "reddit", "t3_abc", "DeepValueDan", MULTILINE_TEXT, "bearish", 0.61)
        c = _event(
            conn, "reddit", "t3_abc", "GME", "2026-09-19T16:00:00Z", "2026-09-21", "closed", "complete",
            intraday_state="ok", ref_ts=_et(2026, 9, 18, 19, 59), ref_price=25.1, earnings_flag=None, split_flag=1,
            completed_at="2026-09-29T00:30:00Z",
        )  # fmt: skip
        _window(conn, c, "pre_leg", _et(2026, 9, 18, 15, 59), _et(2026, 9, 18, 19, 59), 25.6, 25.1, -0.019531)
        _window(conn, c, "post_leg", _et(2026, 9, 18, 19, 59), _et(2026, 9, 21, 15, 59), 25.1, 24.0, -0.043825)

        # Ten minutes before the close: the +15 minute window stops at the last regular bar.
        _post(conn, "x", "1002", "jimcramer", FORMULA_TEXT, "neutral", 0.55)
        d = _event(
            conn, "x", "1002", "NVDA", "2026-09-22T19:50:30Z", "2026-09-22", "regular", "complete",
            intraday_state="ok", ref_ts=_et(2026, 9, 22, 15, 49), ref_price=180.1, earnings_flag=0, split_flag=0,
            completed_at="2026-09-30T00:30:00Z",
        )  # fmt: skip
        _window(
            conn, d, "p15", _et(2026, 9, 22, 15, 49), _et(2026, 9, 22, 15, 59), 180.1, 180.5, 0.002221,
            (660.0, 660.2, 0.000303), truncated=1,
        )  # fmt: skip

        _post(conn, "x", "1003", "jimcramer", "Apple earnings next week, stock looks heavy")
        e = _event(
            conn, "x", "1003", "AAPL", "2026-09-21T13:30:20Z", "2026-09-21", "regular", "complete",
            intraday_state="unavailable", earnings_flag=0, split_flag=0, spans_open=1,
            completed_at="2026-09-29T00:30:00Z",
        )  # fmt: skip
    return {"A": a, "B": b, "C": c, "D": d, "E": e}


def _console(width: int = 240) -> Console:
    return Console(file=io.StringIO(), width=width, color_system=None)


def _list(conn, flt: EventFilter | None = None, limit: int = eventsview.DEFAULT_LIMIT, width: int = 240) -> str:
    console = _console(width)
    eventsview.show_events(conn, flt or EventFilter(), limit, console)
    return console.file.getvalue()


def _detail(conn, event_id: int) -> str:
    console = _console()
    assert eventsview.show_event(conn, event_id, console)
    return console.file.getvalue()


def _cells(out: str, needle: str, starts: bool = False) -> list[str]:
    """The non-empty cells of the one table line containing (or starting with) `needle`; cells are 2+ spaces apart."""
    [line] = [line for line in out.splitlines() if (line.startswith(needle) if starts else needle in line)]
    return re.split(r"\s{2,}", line.strip())


def _field(out: str, name: str) -> str:
    [line] = [line for line in out.splitlines() if line.startswith(name + " ")]
    return line[len(name) :].strip()


# ---------------------------------------------------------------- list


def test_empty_database(conn, tmp_path):
    console = _console()
    assert eventsview.show_events(conn, EventFilter(), 30, console) == 0
    assert "No events yet" in console.file.getvalue()
    assert not eventsview.show_event(conn, 1, _console())

    path = tmp_path / "events.csv"
    assert eventsview.export_csv(conn, path, EventFilter()) == 0
    with open(path, encoding="utf-8-sig", newline="") as f:
        [header] = list(csv.reader(f))
    assert header[:3] == ["id", "platform", "native_id"] and header[-2:] == ["url", "text"]


def test_table_rows_are_newest_first_with_times_in_new_york(conn, ids):
    out = _list(conn)

    assert "Events, newest t0 first: 5 of 5" in out
    order = [out.index(t) for t in ("Nov 02 10:00:05 ET", "Sep 24 10:31:07 ET", "Sep 22 15:50:30 ET")]
    assert order == sorted(order)
    order = [out.index(t) for t in ("Sep 22 15:50:30 ET", "Sep 21 09:30:20 ET", "Sep 19 12:00:00 ET")]
    assert order == sorted(order)

    snippet = LONG_TEXT[:57] + "..."
    assert _cells(out, "Sep 24 10:31:07 ET") == [
        str(ids["A"]), "x", "@elonmusk", "TSLA", "Sep 24 10:31:07 ET", "regular", "2026-09-24", "bullish 0.87",
        "182.41 @10:30", "+1.34%", "-1.32%", "+0.32%", "-0.05%", "complete", "EC", snippet,
    ]  # fmt: skip
    # EST after Nov 1 (UTC-5); a pending event with no windows or stance; markup-like text printed literally.
    assert _cells(out, "Nov 02 10:00:05 ET") == [
        str(ids["B"]), "truthsocial", "@realDonaldTrump", "INTC", "Nov 02 10:00:05 ET", "regular", "2026-11-02", "-",
        "24.50 @09:59", "-", "-", "-", "-", "pending", MARKUP_TEXT,
    ]  # fmt: skip
    # Weekend post: the reference bar is Friday's, so its day is shown; whitespace in the snippet is collapsed.
    assert _cells(out, "Sep 19 12:00:00 ET") == [
        str(ids["C"]), "reddit", "@DeepValueDan", "GME", "Sep 19 12:00:00 ET", "closed", "2026-09-21", "bearish 0.61",
        "25.10 @Fri 19:59", "-1.95%", "-4.38%", "-", "-", "complete", "S?",
        "GME squeeze is over. Puts printing 📉 café",
    ]  # fmt: skip
    # A window clipped at the close is starred.
    assert _cells(out, "Sep 22 15:50:30 ET")[7:15] == [
        "neutral 0.55", "180.10 @15:49", "-", "-", "+0.22%*", "+0.03%*", "complete", FORMULA_TEXT,
    ]  # fmt: skip
    assert _cells(out, "Sep 21 09:30:20 ET")[7:13] == ["-", "-", "-", "-", "-", "-"]
    assert _cells(out, "Sep 21 09:30:20 ET")[13:15] == ["complete", "OU"]
    assert "hidden to fit" not in out
    assert "E earnings  S split  C clustered  O spans open  U intraday unavailable  ? earnings unknown" in out


@pytest.mark.parametrize(
    ("flt", "limit", "expected_ids", "heading"),
    [
        (EventFilter(ticker="tsla"), 30, ["A"], "1 of 1 (ticker=TSLA)"),
        (EventFilter(platform="reddit"), 30, ["C"], "1 of 1 (platform=reddit)"),
        (EventFilter(status="pending"), 30, ["B"], "1 of 1 (status=pending)"),
        (EventFilter(platform="x", status="complete"), 30, ["A", "D", "E"], "3 of 3 (platform=x, status=complete)"),
        (EventFilter(), 2, ["B", "A"], "2 of 5"),
    ],
)
def test_filters_and_limit(conn, ids, flt, limit, expected_ids, heading):
    out = _list(conn, flt, limit)
    assert f"Events, newest t0 first: {heading}" in out
    shown = [int(line.split()[0]) for line in out.splitlines() if line.split() and line.split()[0].isdigit()]
    assert shown == [ids[k] for k in expected_ids]


def test_no_match_names_the_filter(conn, ids):
    assert "No events match ticker=ZZZ, status=pending." in _list(conn, EventFilter(ticker="zzz", status="pending"))


@pytest.mark.parametrize(
    ("width", "hidden", "a_cells"),
    [
        (
            160,
            "platform, phase, SPY p15",
            ["@elonmusk", "TSLA", "Sep 24 10:31:07 ET", "2026-09-24", "bullish 0.87", "182.41 @10:30", "+1.34%",
             "-1.32%", "+0.32%", "complete", "EC", "Tesla is going to CR…"],
        ),
        (
            120,
            "platform, phase, stance, SPY p15, status, flags, text",
            ["@elonmusk", "TSLA", "Sep 24 10:31:07 ET", "2026-09-24", "182.41 @10:30", "+1.34%", "-1.32%", "+0.32%"],
        ),
        (
            100,
            "platform, phase, d0, stance, SPY p15, status, flags, text",
            ["@elonmusk", "TSLA", "Sep 24 10:31:07 ET", "182.41 @10:30", "+1.34%", "-1.32%", "+0.32%"],
        ),
        (
            80,
            "platform, @author, phase, d0, stance, SPY p15, status, flags, text",
            ["TSLA", "Sep 24 10:31:07 ET", "182.41 @10:30", "+1.34%", "-1.32%", "+0.32%"],
        ),
    ],
)  # fmt: skip
def test_a_narrow_console_hides_low_priority_columns_instead_of_squeezing(conn, ids, width, hidden, a_cells):
    out = _list(conn, width=width)

    assert f"hidden to fit the terminal: {hidden} (widen it, or use --csv)" in " ".join(out.split())
    table = out[out.index("\n") + 1 : out.index("times are New York")].splitlines()
    assert len(table) == 2 + 5, "one line per event"
    assert all(len(line.rstrip()) <= width for line in table)
    assert _cells(out, "Sep 24 10:31:07 ET") == [str(ids["A"]), *a_cells]
    # Only the snippet is ever cut short.
    upto = None if hidden.endswith("text") else -1
    assert all("…" not in cell for line in table[2:] for cell in re.split(r"\s{2,}", line.strip())[:upto])
    assert "rows wrap" not in out


def test_too_narrow_for_one_line_per_event_wraps_rows_instead_of_cutting_values(conn, ids):
    out = _list(conn, width=64)

    joined = " ".join(out.split())
    assert "rows wrap: 76 columns are needed to keep each event on one line" in joined
    table = out[out.index("\n") + 1 : out.index("times are New York")].splitlines()
    assert all(len(line.rstrip()) <= 64 for line in table)
    assert "…" not in out
    assert [line.split()[0] for line in table if re.match(r"\s*\d+\s", line)] == [
        str(ids[k]) for k in ("B", "A", "D", "E", "C")
    ]
    for value in ("182.41", "@10:30", "+1.34%", "-1.32%", "+0.32%", "25.10", "@Fri", "-4.38%", "+0.22%*", "24.50"):
        assert value in out, value


def test_no_value_but_the_snippet_is_cut_at_any_width(conn, ids):
    for width in range(60, 241):
        out = _list(conn, width=width)
        lines = out.splitlines()
        table = lines[1 : next(i for i, line in enumerate(lines) if line.startswith("times are New York"))]
        assert all(len(line.rstrip()) <= width for line in table), width
        text_shown = table[0].rstrip().endswith("text")
        for line in table[2:]:
            cells = re.split(r"\s{2,}", line.strip())
            assert not any("…" in cell for cell in (cells[:-1] if text_shown else cells)), (width, line)
        # The id is what '--id N' needs; the three returns are never dropped.
        assert [line.split()[0] for line in table[2:] if re.match(r"\s*\d+\s", line)] == [
            str(ids[k]) for k in ("B", "A", "D", "E", "C")
        ], width
        for value in ("+1.34%", "-1.32%", "+0.32%", "-1.95%", "-4.38%", "+0.22%*"):
            assert value in out, (width, value)


# ---------------------------------------------------------------- one event


def test_detail_shows_every_field_every_window_the_post_and_how_to_check_it(conn, ids):
    out = _detail(conn, ids["A"])

    for column in [r["name"] for r in conn.execute("PRAGMA table_info(events)")]:
        assert _field(out, column), column
    assert _field(out, "t0") == "Sep 24 10:31:07 ET  (2026-09-24T14:31:07Z)"
    assert _field(out, "d0") == "2026-09-24 (Thu)"
    assert _field(out, "ref_ts") == f"{_et(2026, 9, 24, 10, 30)}  (bar start Thu Sep 24 10:30 ET)"
    assert _field(out, "ref_price") == "182.41"
    assert (_field(out, "earnings_flag"), _field(out, "split_flag"), _field(out, "clustered")) == ("yes", "no", "yes")
    assert _field(out, "completed_at") == "2026-09-30 20:30:00 ET"
    assert _field(out, "flags") == "EC"

    rows = {name: _cells(out, name + " ", starts=True) for name in eventsview.WINDOWS}
    assert rows["pre_leg"] == [
        "pre_leg", "Wed Sep 23 15:59", "180.00", "Thu Sep 24 10:30", "182.41", "+1.34%", "660.00", "661.00", "+0.15%",
        "no",
    ]  # fmt: skip
    assert rows["p15"] == [
        "p15", "Thu Sep 24 10:30", "182.41", "Thu Sep 24 10:45", "183.00", "+0.32%", "661.00", "660.67", "-0.05%", "no",
    ]  # fmt: skip
    assert rows["p60"][:6] == ["p60", "Thu Sep 24 10:30", "182.41", "Thu Sep 24 11:30", "184.00", "+0.87%"]
    assert rows["p60"][6:] == ["-", "-", "-", "no"]
    window_lines = [line.split()[0] for line in out[out.index("Windows (") :].splitlines()[3:10]]
    assert window_lines == list(eventsview.WINDOWS)

    assert LONG_TEXT in out
    assert "https://example.com/x/1001" in out
    assert "bullish 0.87  (StephanAkkerman/FinTwitBERT-sentiment)" in out
    hint = " ".join(out[out.index("To check it") :].split())
    assert hint.startswith(
        "To check it: open a 1-minute TSLA chart for Thu 2026-09-24 with pre/post-market shown and read the close of "
        "the 10:30 ET bar; it should be 182.41."
    )


def test_detail_of_a_pending_event_without_windows(conn, ids):
    out = _detail(conn, ids["B"])

    assert _field(out, "t0") == "Nov 02 10:00:05 ET  (2026-11-02T15:00:05Z)"
    assert _field(out, "earnings_flag") == "not set until the event completes"
    assert _field(out, "intraday_state") == "-"
    assert "Windows: none stored for this event." in out
    assert MARKUP_TEXT in out
    assert "1-minute INTC chart for Mon 2026-11-02" in " ".join(out.split())
    assert "close of the 09:59 ET bar; it should be 24.50." in " ".join(out.split())


def test_detail_without_a_reference_price(conn, ids):
    out = _detail(conn, ids["E"])
    assert "No reference price: 1-minute bars for this event were unavailable" in out
    assert _field(out, "flags") == "OU"

    with conn:
        _post(conn, "x", "1004", "jimcramer", "AMD looks cheap")
        pending = _event(conn, "x", "1004", "AMD", "2026-09-24T21:00:00Z", "2026-09-25", "after", "pending")
    assert "No reference price yet: enrich sets it" in _detail(conn, pending)


def test_detail_uses_the_yahoo_symbol_and_survives_a_missing_post(conn):
    with conn:
        event_id = _event(
            conn, "x", "gone", "BRK.B", "2026-09-24T14:31:07Z", "2026-09-24", "regular", "pending",
            ref_ts=_et(2026, 9, 24, 10, 30), ref_price=480.0,
        )  # fmt: skip
    out = _detail(conn, event_id)
    assert "(post not found)" in out
    assert "1-minute BRK-B chart" in out


# ---------------------------------------------------------------- CSV


def _read_csv(path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def test_csv_round_trip(conn, ids, tmp_path):
    path = tmp_path / "events.csv"

    assert eventsview.export_csv(conn, path, EventFilter()) == 5

    assert path.read_bytes().startswith(b"\xef\xbb\xbf")
    rows = _read_csv(path)
    assert [int(r["id"]) for r in rows] == [ids[k] for k in ("B", "A", "D", "E", "C")]
    event_columns = [r["name"] for r in conn.execute("PRAGMA table_info(events)")]
    header = list(rows[0])
    assert header[: len(event_columns)] == event_columns
    for name in eventsview.WINDOWS:
        for field in ("ret", "spy_ret", "start_et", "end_et", "start_price", "end_price", "truncated"):
            assert f"{name}_{field}" in header
    assert header[-2:] == ["url", "text"]

    a = next(r for r in rows if int(r["id"]) == ids["A"])
    assert a["text"] == LONG_TEXT
    assert (a["author"], a["stance"], float(a["stance_conf"])) == ("elonmusk", "bullish", 0.87)
    assert (a["t0"], a["t0_et"], a["ref_time_et"]) == (
        "2026-09-24T14:31:07Z",
        "2026-09-24 10:31:07",
        "2026-09-24 10:30:00",
    )
    assert float(a["ref_price"]) == 182.41
    assert (a["earnings_flag"], a["split_flag"], a["clustered"], a["spans_open"]) == ("1", "0", "1", "0")
    assert float(a["p15_ret"]) == pytest.approx(0.003234)
    assert float(a["p15_spy_ret"]) == pytest.approx(-0.0005)
    assert (a["p15_start_et"], a["p15_end_et"], a["p15_truncated"]) == (
        "2026-09-24 10:30:00",
        "2026-09-24 10:45:00",
        "0",
    )
    assert (a["pre_leg_start_et"], float(a["pre_leg_start_price"])) == ("2026-09-23 15:59:00", 180.0)
    assert a["url"] == "https://example.com/x/1001"

    b = next(r for r in rows if int(r["id"]) == ids["B"])
    assert b["text"] == MARKUP_TEXT
    assert (b["p15_ret"], b["pre_leg_ret"], b["earnings_flag"], b["stance"]) == ("", "", "", "")
    assert b["ref_time_et"] == "2026-11-02 09:59:00"

    c = next(r for r in rows if int(r["id"]) == ids["C"])
    assert c["text"] == MULTILINE_TEXT
    assert c["ref_time_et"] == "2026-09-18 19:59:00"

    # Excel would run this as a formula; the leading apostrophe keeps it text.
    d = next(r for r in rows if int(r["id"]) == ids["D"])
    assert d["text"] == "'" + FORMULA_TEXT
    assert d["p15_truncated"] == "1"


def test_csv_keeps_long_numeric_post_ids_exact_in_excel(conn, ids, tmp_path):
    with conn:
        _post(conn, "x", "1970000000000000001", "elonmusk", "$TSLA to the moon")
        _event(conn, "x", "1970000000000000001", "TSLA", "2026-09-23T14:00:00Z", "2026-09-23", "regular", "pending")
        # Not ASCII digits, so neither may become a ="..." formula; the second still gets the formula guard.
        _event(conn, "x", "١٢٣", "TSLA", "2026-09-23T13:00:00Z", "2026-09-23", "regular", "pending")
        _event(conn, "x", "=1+1", "TSLA", "2026-09-23T12:00:00Z", "2026-09-23", "regular", "pending")
    path = tmp_path / "events.csv"

    eventsview.export_csv(conn, path, EventFilter())

    rows = {r["url"] or r["native_id"]: r for r in _read_csv(path)}
    # A bare number keeps only 15 significant digits in Excel; ="..." is a text formula it shows in full.
    assert rows["https://example.com/truthsocial/115000000000000001"]["native_id"] == '="115000000000000001"'
    assert rows["https://example.com/x/1970000000000000001"]["native_id"] == '="1970000000000000001"'
    assert rows["https://example.com/x/1001"]["native_id"] == '="1001"'
    assert rows["https://example.com/reddit/t3_abc"]["native_id"] == "t3_abc"
    assert rows["١٢٣"]["native_id"] == "١٢٣"
    assert rows["'=1+1"]["native_id"] == "'=1+1"
    assert '"=""115000000000000001"""' in path.read_text(encoding="utf-8-sig")
    assert not any(r["native_id"].isascii() and r["native_id"].isdigit() for r in rows.values())


def test_csv_applies_filters_and_limit_and_creates_the_folder(conn, ids, tmp_path):
    path = tmp_path / "reports" / "m2" / "complete.csv"

    assert eventsview.export_csv(conn, path, EventFilter(status="complete")) == 4
    assert {r["status"] for r in _read_csv(path)} == {"complete"}

    assert eventsview.export_csv(conn, path, EventFilter(platform="x"), limit=1) == 1
    assert [int(r["id"]) for r in _read_csv(path)] == [ids["A"]]
