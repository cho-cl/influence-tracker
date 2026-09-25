from __future__ import annotations

import html
import re
import string
from collections.abc import Iterable, Mapping, Sequence

from .config import Watchlist
from .models import MatchResult, MatchType, Mention

_TYPE_ORDER: dict[str, int] = {"cashtag": 0, "name": 1, "bare": 2}

# Both translations are one-for-one, so match spans in the normalized text index the original text too.
_NORMALIZE = str.maketrans({0x2018: "'", 0x2019: "'", 0x02BC: "'", 0x2010: "-", 0x2011: "-"})
_ASCII_LOWER = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)

_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
# "$BRK.B", Yahoo's "$BRK-B" and Bloomberg's "$BRK/B" are the same class share.
_CASHTAG = re.compile(r"(?<![\w$])\$([A-Za-z]{1,5}(?:[./-][A-Za-z])?)(?!\w)")
_CASHTAG_SYMBOL = re.compile(r"[A-Za-z]{1,5}(?:[./-][A-Za-z])?")
_CLASS_SEPARATORS = str.maketrans("-/", "..")

# A letter or digit (Unicode-aware; underscore counts as punctuation). Names and context words must not touch one.
_ALNUM = r"[^\W_]"
_NOT_ALNUM_BEFORE = rf"(?<!{_ALNUM})"
_NOT_ALNUM_AFTER = rf"(?!{_ALNUM})"
_WORD_OR_DOLLAR = r"[\w$]"

# Lowercased-text matches. Bare long/short/calls/puts/options are left out: "Ford calls on Congress", "a short film".
_CONTEXT_WORDS = (
    r"stocks",
    # Retail "in stock" / "out of stock" / "stock photo" is not equity.
    r"(?<!\bin\s)(?<!\bof\s)stock(?!\s+(?:photos?|images?|footage)\b)",
    r"(?:stock|share)holders?",
    # "Meta shares a video" is the verb, not equity.
    r"shares(?!\s+(?:a|an|the|his|her|their|its|my|our|your|this|that|these|those|some|new)\b)",
    r"share\s+price",
    r"tickers?",
    r"earnings",
    r"revenues?",
    r"dividends?",
    r"buybacks?",
    r"ipo(?:s|ed)?",
    r"market\s+cap(?:italization)?",
    r"valuations?",
    r"price\s+targets?",
    r"bullish",
    r"bearish",
    r"short\s+(?:sellers?|selling|positions?|interest|squeeze)",
    r"shorting",
    r"shorted",
    r"long\s+positions?",
    r"buys?",
    r"buying",
    r"bought",
    # "Target sold out of consoles" is retail, not trading.
    r"(?:sells?|selling|sold)(?![\s-]+out\b)",
    r"sell-?offs?",
    r"invest(?:s|ed|ing|ors?|ments?)?",
    r"portfolios?",
    r"hedge\s+funds?",
    r"(?:call|put)\s+options?",
    r"options?\s+(?:chain|flow|activity|trading|expir\w*)",
    r"strike\s+price",
    r"0dte",
    r"nasdaq",
    r"nyse",
    r"dow\s+jones",
    r"wall\s+street",
    r"ceo",
)
_CONTEXT = re.compile(
    rf"{_NOT_ALNUM_BEFORE}(?:{'|'.join(_CONTEXT_WORDS)}){_NOT_ALNUM_AFTER}"
    r"|s&p|[$\N{EURO SIGN}\N{POUND SIGN}]\s?\d|\d\s?%"
)
_STRIKE = r"\d{2,}(?:\.\d+)?"
# Original-case matches.
_CONTEXT_CASED = re.compile(
    rf"{_NOT_ALNUM_BEFORE}(?:"
    # "eps" is also slang for episodes, so earnings-per-share only counts in capitals.
    r"EPS"
    # Strikes are lowercase ("15c") since "40C" is a temperature and "1080p" a resolution, unless an expiry follows.
    rf"|(?!(?:144|240|360|480|720|1080|1440|2160|4320)p){_STRIKE}[cp]"
    rf"|{_STRIKE}[cCpP]\s+\d{{1,2}}/\d{{1,2}}"
    rf"){_NOT_ALNUM_AFTER}"
)

# Lowercased-text phrases in which a name is trader or everyday jargon, not the company.
_JARGON_WORDS = (
    r"price\s+targets?",
    # "Block trades higher" is the company's stock moving.
    r"block\s+trade(?:s(?!\s+(?:higher|lower|up|down|flat|at|near|above|below)\b))?",
    r"shell\s+(?:compan(?:y|ies)|corporations?|corps?|entit(?:y|ies))",
    r"oracle\s+of\s+omaha",
    r"google\s+(?:it|them)",
)
_JARGON = re.compile(
    rf"{_NOT_ALNUM_BEFORE}(?:(?:{'|'.join(_JARGON_WORDS)}){_NOT_ALNUM_AFTER}"
    r"|target\s+price(?=\s*(?::|of\b|to\b|\$|\d))|target\s*:\s*\$?\d|targets?\s+hit(?=\s*(?:[^\w\s]|$)))"
)
# Original-case: 'then google "investopedia delta"' is the verb; 'Google "Gemini" launch' is the company.
_JARGON_CASED = re.compile(rf'{_NOT_ALNUM_BEFORE}google\s+["\N{{LEFT DOUBLE QUOTATION MARK}}]')


def _blank(m: re.Match[str]) -> str:
    return " " * len(m.group())


def _fold(s: str) -> str:
    # Per-character lower() rather than casefold(), which would turn a configured "ß" into "ss" and never match it.
    if s.isascii():
        return s.lower()
    return "".join(low if len(low := c.lower()) == 1 else c for c in s)


def _name_key(name: str) -> str:
    return _fold(" ".join(name.translate(_NORMALIZE).split()))


def _lookup_key(name: str) -> str:
    # A hyphenated name also matches its spaced spelling ("Coca Cola"), so both spellings share one key.
    return _name_key(name).replace("-", " ")


def _trie_pattern(words: Iterable[str], boundary_before: str, special: Mapping[str, str], fold_non_ascii: bool) -> str:
    """A regex for any of `words`, factored as a character trie: longest word wins; `special` maps a char to a regex."""
    trie: dict[str, dict] = {}
    for word in words:
        node = trie
        for ch in word:
            node = node.setdefault(ch, {})
        node[""] = {}

    def atom(ch: str) -> str:
        if ch in special:
            return special[ch]
        upper = ch.upper()
        # Text is only ASCII-lowercased; non-ASCII letters need both cases (if upper folds back, the lookup works).
        if fold_non_ascii and not ch.isascii() and upper != ch and len(upper) == 1 and _fold(upper) == ch:
            return f"[{re.escape(ch)}{re.escape(upper)}]"
        return re.escape(ch)

    def emit(node: dict[str, dict], first: bool) -> str:
        # The boundary lookbehind follows the first char, not precedes it, so sre keeps its fast first-char scan.
        branches = [
            atom(ch) + (f"(?<!{boundary_before}.)" if first else "") + emit(child, False)
            for ch, child in sorted(node.items())
            if ch
        ]
        if not branches:
            return ""
        if len(branches) == 1 and "" not in node:
            return branches[0]
        group = f"(?:{'|'.join(branches)})"
        return group + "?" if "" in node else group

    return emit(trie, True)


class MentionMatcher:
    """Finds configured tickers in post text via cashtags, company names and (Reddit only) bare tickers."""

    def __init__(self, watchlist: Watchlist):
        self._universe: dict[str, str] = {}
        for t in watchlist.tickers:
            for sym in (t.symbol, *t.aliases):
                self._universe[sym] = t.symbol

        bases: dict[str, set[str]] = {}
        for sym, ticker in self._universe.items():
            base, dot, _ = sym.partition(".")
            if dot and base not in self._universe:
                bases.setdefault(base, set()).add(ticker)
        # "$BRK" means BRK.B when no other ticker has a share class on that base.
        self._class_bases = {base: next(iter(owners)) for base, owners in bases.items() if len(owners) == 1}

        self._names: dict[str, list[tuple[str, str]]] = {}
        name_keys: set[str] = set()
        for t in watchlist.tickers:
            kinds = [
                *((n, "plain") for n in t.names),
                *((n, "cased") for n in t.cased_names),
                *((n, "ambiguous") for n in t.ambiguous_names),
            ]
            for name, kind in kinds:
                if not (key := _name_key(name)):
                    continue
                name_keys.add(key)
                entries = self._names.setdefault(_lookup_key(key), [])
                if (t.symbol, kind) not in entries:
                    entries.append((t.symbol, kind))
        # Matched against ASCII-lowercased text.
        name_special = {" ": r"\s+", "-": r"(?:-|\s+)"}
        self._name_re = (
            re.compile(f"(?:{_trie_pattern(name_keys, _ALNUM, name_special, True)}){_NOT_ALNUM_AFTER}")
            if name_keys
            else None
        )

        stop = set(watchlist.bare_ticker_stoplist)
        bare = [s for s in self._universe if len(s.replace(".", "")) >= 3 and s not in stop]
        self._bare_re = (
            re.compile(rf"(?:{_trie_pattern(bare, _WORD_OR_DOLLAR, {'.': '[./-]'}, False)})(?!\w)") if bare else None
        )

    def match(self, text: str, platform: str, cashtag_hints: Sequence[str] = ()) -> MatchResult:
        found: dict[tuple[str, MatchType], str] = {}
        unknown: set[str] = set()
        if "&" in text:
            # X returns post text HTML-escaped ("AT&amp;T").
            text = html.unescape(text)
        work = text.translate(_NORMALIZE)
        if "://" in work or "www." in work.translate(_ASCII_LOWER):
            work = _URL.sub(_blank, work)

        saw_cashtag = False

        def on_cashtag(m: re.Match[str]) -> str:
            nonlocal saw_cashtag
            saw_cashtag = True
            symbol = m.group(1).upper().translate(_CLASS_SEPARATORS)
            ticker = self._resolve_cashtag(symbol)
            if ticker is None:
                unknown.add(symbol)
                return m.group()
            found.setdefault((ticker, "cashtag"), text[m.start() : m.end()])
            # Blank out known cashtags so "$META" / "$AMD" don't also count as names or bare tickers.
            return _blank(m)

        masked = _CASHTAG.sub(on_cashtag, work) if "$" in work else work

        for hint in cashtag_hints:
            tag = hint.strip().lstrip("$")
            if not _CASHTAG_SYMBOL.fullmatch(tag):
                continue
            saw_cashtag = True
            symbol = tag.upper().translate(_CLASS_SEPARATORS)
            ticker = self._resolve_cashtag(symbol)
            if ticker is None:
                unknown.add(symbol)
            else:
                found.setdefault((ticker, "cashtag"), f"${tag}")

        bare_tokens: set[str] = set()
        if platform == "reddit" and self._bare_re is not None:
            for m in self._bare_re.finditer(masked):
                ticker = self._universe[m.group().translate(_CLASS_SEPARATORS)]
                if (ticker, "cashtag") not in found:
                    found.setdefault((ticker, "bare"), m.group())
                    bare_tokens.add(m.group())

        if self._name_re is not None:
            lowered = masked.translate(_ASCII_LOWER)
            context: bool | None = None
            jargon: list[tuple[int, int]] | None = None
            for m in self._name_re.finditer(lowered):
                if jargon is None:
                    jargon = [j.span() for j in (*_JARGON.finditer(lowered), *_JARGON_CASED.finditer(masked))]
                if any(start < m.end() and m.start() < end for start, end in jargon):
                    continue
                surface = text[m.start() : m.end()]
                for ticker, kind in self._names.get(_lookup_key(m.group()), ()):
                    if (ticker, "name") in found:
                        continue
                    if kind != "plain" and not surface[0].isupper():
                        continue
                    if kind == "ambiguous":
                        if context is None:
                            context = saw_cashtag or _has_context(masked, lowered)
                        # A bare "META" can't vouch for itself being Meta the company.
                        if not (context or any(tok != surface for tok in bare_tokens)):
                            continue
                    found[(ticker, "name")] = surface

        mentions = [
            Mention(ticker, match_type, surface)
            for (ticker, match_type), surface in sorted(found.items(), key=lambda kv: (kv[0][0], _TYPE_ORDER[kv[0][1]]))
        ]
        return MatchResult(mentions=mentions, unknown_cashtags=sorted(unknown))

    def _resolve_cashtag(self, symbol: str) -> str | None:
        ticker = self._universe.get(symbol)
        if ticker is not None:
            return ticker
        # "$TSLA.I regret it": a sentence period glued to the cashtag, not a share class.
        base = symbol.split(".", 1)[0]
        return self._universe.get(base) or self._class_bases.get(base)


def _has_context(text: str, lowered: str) -> bool:
    # Whitespace runs are collapsed so the fixed-width "in stock" lookbehind also catches "in  stock".
    return _CONTEXT.search(" ".join(lowered.split())) is not None or _CONTEXT_CASED.search(text) is not None
