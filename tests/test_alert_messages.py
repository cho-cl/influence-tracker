from __future__ import annotations

from datetime import UTC, datetime

from influence_tracker.alerts import messages as m
from influence_tracker.alerts.notify import MAX_BODY_BYTES
from influence_tracker.events import PricePoint

T0 = datetime(2026, 9, 29, 14, 31, tzinfo=UTC)  # 10:31 ET


def post(**kw) -> m.PostInfo:
    base = dict(
        platform="truthsocial",
        native_id="1",
        author="realDonaldTrump",
        created_at=T0,
        text="NVIDIA is doing GREAT",
        url="https://truthsocial.com/@realDonaldTrump/1",
        stance="bullish",
        stance_conf=0.91,
        tickers=("NVDA",),
    )
    base.update(kw)
    return m.PostInfo(**base)


def test_et_format():
    assert m.et(datetime(2026, 9, 9, 12, 5, tzinfo=UTC)) == "Wed Sep 9 08:05 ET"


def test_heads_up_basic():
    msg = m.heads_up(post(), set(), {"NVDA": PricePoint(int(T0.timestamp()) - 60, 182.41)}, None)
    assert msg.title == "NVDA · realDonaldTrump · bullish (0.91)"
    assert msg.priority == 3 and msg.tags == ("chart_with_upwards_trend",)
    assert msg.click == "https://truthsocial.com/@realDonaldTrump/1"
    assert "NVIDIA is doing GREAT" in msg.body
    assert "Posted Tue Sep 29 10:31 ET" in msg.body
    assert "NVDA $182.41" in msg.body
    assert "History: no study numbers available for realDonaldTrump." in msg.body


def test_heads_up_holdings_first_with_star_and_high_priority():
    msg = m.heads_up(post(tickers=("AAPL", "NVDA")), {"NVDA"}, {}, None)
    assert msg.title.startswith("⭐ NVDA, AAPL · ")
    assert msg.priority == 4 and "star" in msg.tags
    assert "Price at post: unavailable" in msg.body


def test_heads_up_folds_many_tickers_and_caps_long_text():
    text = "🚀 Huge news for American companies! " * 100
    msg = m.heads_up(post(text=text, tickers=("AAPL", "INTC", "MSFT", "NVDA", "QCOM", "TSLA")), {"TSLA"}, {}, None)
    assert msg.title.startswith("⭐ TSLA, AAPL +4 · ")
    first_line = msg.body.splitlines()[0]
    assert len(first_line) <= 221 and first_line.endswith("…")
    assert len(msg.body.encode("utf-8")) <= MAX_BODY_BYTES


def test_heads_up_price_as_of_note_for_old_bars():
    stale = PricePoint(int(T0.timestamp()) - 3 * 3600, 99.5)
    msg = m.heads_up(post(), set(), {"NVDA": stale}, None)
    assert "NVDA $99.50 (as of Tue Sep 29 07:31 ET)" in msg.body


def test_heads_up_without_stance():
    msg = m.heads_up(post(stance=None, stance_conf=None), set(), {}, None)
    assert msg.title.endswith("· stance unavailable") and msg.tags == ("grey_question",)


def test_history_line_significant_and_not():
    ok = m.heads_up(post(), set(), {}, m.History(71, -0.00604, 0.50))
    assert (
        "History: after realDonaldTrump's bullish/bearish posts, the stocks moved 0.60% against the direction the "
        "post pointed, beyond what SPY explains, on average over 2 days (n=71, Holm p=0.50, not significant)."
    ) in ok.body
    sig = m.heads_up(post(), set(), {}, m.History(40, 0.0123, 0.01))
    assert (
        "the stocks moved 1.23% in the direction the post pointed, beyond what SPY explains, on average over 2 days "
        "(n=40, Holm p=0.01, significant at the 5% level)."
    ) in sig.body


def history_line(msg: m.Message) -> str:
    return next(line for line in msg.body.splitlines() if line.startswith("History:"))


def test_history_line_reads_the_signed_car_against_the_post_direction():
    # mean_signed_car is +CAR after bullish posts and -CAR after bearish ones: +0.60% for a bearish-heavy author means
    # the stocks fell, so the line must say which way relative to the post, never print a bare signed move.
    bearish = post(stance="bearish", stance_conf=0.9)
    fell = history_line(m.heads_up(bearish, set(), {}, m.History(30, 0.006, 0.20)))
    assert "moved 0.60% in the direction the post pointed" in fell
    rose = history_line(m.heads_up(bearish, set(), {}, m.History(30, -0.006, 0.20)))
    assert "moved 0.60% against the direction the post pointed" in rose
    for line in (fell, rose):
        assert "+0.60%" not in line and "-0.60%" not in line and "bullish/bearish posts" in line


def test_history_line_small_untested_and_missing():
    # n_posts counts only bullish/bearish posts with a CAR; None means no study row or a failed study run.
    few = history_line(m.heads_up(post(), set(), {}, m.History(6, 0.01, None)))
    assert few == "History: fewer than 10 past bullish/bearish posts by realDonaldTrump with price data."
    untested = history_line(m.heads_up(post(), set(), {}, m.History(12, 0.004, None)))
    assert "the stocks moved 0.40% in the direction the post pointed" in untested
    assert untested.endswith("(n=12, no p-value).") and "fewer than" not in untested
    missing = history_line(m.heads_up(post(), set(), {}, None))
    assert missing == "History: no study numbers available for realDonaldTrump."


def test_follow_60m_lines():
    rows = [m.Follow60Row("NVDA", 0.0084, 0.0010, 0.0069), m.Follow60Row("AAPL", -0.002, None, None)]
    msg = m.follow_60m(post(tickers=("AAPL", "NVDA")), set(), rows, "From the post (10:31 ET) to 11:31 ET")
    assert msg.title == "AAPL, NVDA · 60 min after realDonaldTrump's post"
    assert "NVDA +0.84% vs SPY +0.10% → abnormal +0.69%" in msg.body
    assert "AAPL -0.20% (SPY unavailable)" in msg.body
    assert "From the post (10:31 ET) to 11:31 ET" in msg.body
    assert msg.priority == 2


def test_follow_d1_wording():
    rows = [
        m.D1Row("NVDA", 0.012, 1.95, ()),
        m.D1Row("AAPL", -0.041, -2.40, ("earnings day — the move may be the report, not the post",)),
        m.D1Row("MSFT", None, None, ()),
    ]
    msg = m.follow_d1(post(tickers=("AAPL", "MSFT", "NVDA")), {"AAPL"}, rows)
    assert "NVDA CAR[0,+1] +1.20% (z +1.95) — within the normal range" in msg.body
    assert "AAPL CAR[0,+1] -4.10% (z -2.40) — unusually large (|z| = 2.40); earnings day" in msg.body
    assert "MSFT: no market model (too little price history)" in msg.body
    assert msg.priority == 3 and msg.title.startswith("⭐ AAPL, MSFT +1 · day-after check")


def test_followups_say_how_many_tickers_were_left_out():
    tickers = ("AAPL", "AMD", "INTC", "MSFT", "NVDA", "QCOM", "TSLA")
    shown = tickers[:5]
    msg60 = m.follow_60m(
        post(tickers=tickers), set(), [m.Follow60Row(t, 0.01, 0.001, 0.009) for t in shown], "From the post"
    )
    lines = msg60.body.splitlines()
    assert lines[5] == "+2 more — run: influence events" and lines[6] == "From the post"
    msg_d1 = m.follow_d1(post(tickers=tickers), set(), [m.D1Row(t, 0.01, 0.5, ()) for t in shown])
    assert msg_d1.body.splitlines()[5] == "+2 more — run: influence events"
    every = m.follow_60m(post(), set(), [m.Follow60Row("NVDA", 0.01, 0.001, 0.009)], "From the post")
    assert "more — run: influence events" not in every.body


def test_digest_caps_at_eight():
    posts = [post(native_id=str(i), tickers=("NVDA", "INTC")) for i in range(11)]
    msg = m.digest(posts)
    assert msg.title == "While you were away: 11 stock posts"
    lines = msg.body.splitlines()
    assert len(lines) == 9 and lines[-1] == "+3 more — run: influence events"
    assert lines[0] == "Tue Sep 29 10:31 ET · realDonaldTrump · INTC, NVDA · bullish"
    assert msg.priority == 2
