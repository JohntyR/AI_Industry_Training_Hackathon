"""
query_data — deterministic tool layer for the Cognitivo hackathon agent.

This is the ENGINE behind the 40% hidden-question score. The models NEVER do math:
Qwen (brain) decides which metric to call; this code computes the exact number;
fine-tuned Nemotron only phrases the verified result.

Pure Python stdlib (csv, json, re, math, datetime) — NO third-party deps, so it
drops onto the box with zero install risk.

REPRODUCIBILITY RULES (getting one wrong = 0 on that component, even with perfect
architecture). All verified against Participant_Package/public_questions.jsonl:
  - EXCLUDE Tabcorp (TAH.AX) from ASX rankings/baskets/extremes unless explicitly asked.
    Its +2660% is a flagged data artifact and it also skews average-volume.
  - ASX returns = first-to-last CLOSE, simple return ((last/first)-1)*100.
  - Basket = arithmetic mean of the 17 non-Tabcorp constituents' individual returns.
  - Max drawdown = min over rows of (close/running_peak - 1), peak & trough dates reported.
  - AFR pattern search = case-insensitive, across HEADLINE+SUBHEAD+INTRO+TEXT COMBINED,
    counted ONCE per record. Whole-word patterns must use \\b...\\b anchors.
  - RBA "rate in force on date D" = the Cash rate target of the latest row with
    Effective Date <= D.
  - RBA change sign: 'Change % points' like '+0.25' / '-0.25' / '0.00'.
  - Tolerances (see each public Q's tolerance_note): dates/counts/rates/rankings EXACT;
    returns/drawdowns/volatility/shares +/-0.02pp; correlations +/-0.001;
    quoted closes +/-0.0001; average volume +/-1 share.

Usage:
    from query_data import query_data
    query_data("rba", "count_changes")
    query_data("asx", "rank_annual_returns", year=2018, exclude_tabcorp=True)
    query_data("afr", "count", pattern=r"\\bunemployment\\b")
"""

import csv
import glob
import json
import math
import os
import re
import threading
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Data location. Override with env HACKATHON_DATA_DIR on the box.
# Default points at the cloned repo's "data set" folder.
# ---------------------------------------------------------------------------
# This file lives in <repo>/src/, and the organizer-supplied datasets sit in
# <repo>/data set/, so the default is one level up. HACKATHON_DATA_DIR wins.
_DEFAULT_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data set",
)
DATA_DIR = os.environ.get("HACKATHON_DATA_DIR", os.path.abspath(_DEFAULT_DATA))

RBA_CSV = os.path.join(DATA_DIR, "RBA Rates", "RBA-rates.csv")
ASX_DIR = os.path.join(DATA_DIR, "ASX")
AFR_DIR = os.path.join(DATA_DIR, "AFR")

TABCORP = "TAH.AX"

# ---------------------------------------------------------------------------
# Caches
#
# The harness sends up to THREE concurrent /query requests, and LangChain runs
# sync tools in a threadpool, so every cache is built under a lock: three
# simultaneous cold AFR loads would otherwise triple both the parse time and
# the ~1 GB of resident memory. Call warmup() at server startup so no request
# ever pays the cold-load cost.
# ---------------------------------------------------------------------------
_rba_cache = None
_asx_cache = {}       # ticker -> list[dict] sorted by date
_asx_index = None     # ticker -> file path
_afr_cache = None     # list[dict] all records
_afr_haystacks = None  # parallel list[str]: HEADLINE+SUBHEAD+INTRO+TEXT per record

_rba_lock = threading.Lock()
_asx_lock = threading.Lock()
_afr_lock = threading.Lock()


def _parse_date(s):
    """Accept ISO 'YYYY-MM-DD' or RBA-style '3 Feb 2010'."""
    s = s.strip()
    for fmt in ("%Y-%m-%d", "%d %b %Y", "%d %B %Y", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            pass
    raise ValueError(f"unparseable date: {s!r}")


def _iso(s):
    """Any accepted date spelling -> 'YYYY-MM-DD' (the ASX row key format)."""
    return _parse_date(s).strftime("%Y-%m-%d")


def warmup():
    """Preload every dataset. Call once at startup, before serving traffic."""
    _load_rba()
    for ticker in _all_tickers():
        _load_asx(ticker)
    _load_afr()
    return {
        "rba_rows": len(_rba_cache),
        "asx_tickers": len(_asx_cache),
        "afr_records": len(_afr_cache),
    }


# ===========================================================================
# RBA
# ===========================================================================
def _load_rba():
    """Return list of {date: datetime, date_str, change: float, target: float} in file order."""
    global _rba_cache
    if _rba_cache is not None:
        return _rba_cache
    with _rba_lock:
        if _rba_cache is not None:
            return _rba_cache
        rows = []
        # utf-8-sig handles the BOM whether or not it's present.
        with open(RBA_CSV, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                # Field names (exact): 'Effective Date', 'Change % points', 'Cash rate target%'
                date_str = r["Effective Date"].strip()
                change = float(r["Change % points"].replace("+", "").strip())
                target = float(r["Cash rate target%"].strip())
                rows.append({
                    "date": datetime.strptime(date_str, "%d %b %Y"),
                    "date_str": date_str,
                    "change": change,
                    "target": target,
                })
        _rba_cache = rows
        return rows


def _decision_date(effective):
    """RBA board decisions are announced the day before they take effect."""
    return (effective - timedelta(days=1)).strftime("%d %b %Y").lstrip("0")


def _rba(metric, **p):
    rows = _load_rba()
    changes = [r for r in rows if r["change"] != 0.0]

    if metric == "count":
        return {"total_records": len(rows)}

    if metric == "count_changes":
        inc = [r for r in changes if r["change"] > 0]
        dec = [r for r in changes if r["change"] < 0]
        return {
            "total_records": len(rows),
            "changes": len(changes),
            "increases": len(inc),
            "decreases": len(dec),
        }

    if metric == "count_increases":
        return {"increases": len([r for r in changes if r["change"] > 0])}

    if metric == "count_decreases":
        return {"decreases": len([r for r in changes if r["change"] < 0])}

    if metric == "extremes":
        # highest / lowest cash-rate target, first effective date + record count at that rate
        targets = [r["target"] for r in rows]
        hi, lo = max(targets), min(targets)
        hi_rows = [r for r in rows if r["target"] == hi]
        lo_rows = [r for r in rows if r["target"] == lo]
        # The RBA board announces a decision the day BEFORE it takes effect, and
        # the reference answers are inconsistent about which one they quote (see
        # handout/03_scoring_and_examples.md, where the judge expected 2010-11-02
        # for a rate effective 3 Nov 2010). Both are reported so the answer can
        # state the effective date and name the decision date alongside it.
        return {
            "highest_rate": hi,
            "highest_first_date": hi_rows[0]["date_str"],
            "highest_first_decision_date": _decision_date(hi_rows[0]["date"]),
            "highest_record_count": len(hi_rows),
            "lowest_rate": lo,
            "lowest_first_date": lo_rows[0]["date_str"],
            "lowest_first_decision_date": _decision_date(lo_rows[0]["date"]),
            "lowest_record_count": len(lo_rows),
        }

    if metric == "lookup_rate":
        # rate in force on a given date (latest row with Effective Date <= date)
        d = _parse_date(p["date"])
        applicable = [r for r in rows if r["date"] <= d]
        if not applicable:
            return {"error": f"no RBA record on or before {p['date']}"}
        r = applicable[-1]  # rows are in chronological file order
        return {"date": p["date"], "rate": r["target"], "effective_date": r["date_str"]}

    if metric == "max_hold_streak":
        # longest gap (in days) between two consecutive NON-ZERO changes
        pts = changes
        best = {"days": 0}
        for a, b in zip(pts, pts[1:]):
            days = (b["date"] - a["date"]).days
            if days > best["days"]:
                best = {
                    "days": days,
                    "start": a["date_str"], "end": b["date_str"],
                    "rate_during": a["target"], "rate_after": b["target"],
                }
        return best

    if metric == "period_summary":
        # cuts/hikes within [start_year, end_year] inclusive; cumulative change; endpoint rates
        y0, y1 = p["start_year"], p["end_year"]
        sub = [r for r in changes if y0 <= r["date"].year <= y1]
        if not sub:
            return {"window": [y0, y1], "n_changes": 0, "n_cuts": 0, "n_hikes": 0,
                    "by_year": {}, "cumulative_change": 0.0,
                    "note": f"the RBA made no rate changes between {y0} and {y1}"}
        cuts = [r for r in sub if r["change"] < 0]
        hikes = [r for r in sub if r["change"] > 0]
        # rate before first change in window = target of latest row strictly before first sub row
        first = sub[0]
        idx = rows.index(first)
        pre_rate = rows[idx - 1]["target"] if idx > 0 else first["target"] - first["change"]
        cum = round(sum(r["change"] for r in sub), 2)
        by_year = {}
        for r in sub:
            by_year[r["date"].year] = by_year.get(r["date"].year, 0) + 1
        return {
            "window": [y0, y1],
            "n_changes": len(sub), "n_cuts": len(cuts), "n_hikes": len(hikes),
            "by_year": by_year,
            "cumulative_change": cum,
            "rate_before": pre_rate,
            "rate_after": sub[-1]["target"],
        }

    if metric == "list":
        y = p.get("year")
        out = rows if y is None else [r for r in rows if r["date"].year == y]
        return {"rows": [{"date": r["date_str"], "change": r["change"], "target": r["target"]} for r in out]}

    return {"error": f"unknown rba metric '{metric}'"}


# ===========================================================================
# ASX
# ===========================================================================
def _load_asx_index():
    """ticker -> file path, built once. The in-file ticker field is authoritative."""
    global _asx_index
    if _asx_index is not None:
        return _asx_index
    with _asx_lock:
        if _asx_index is not None:
            return _asx_index
        index = {}
        for f in sorted(glob.glob(os.path.join(ASX_DIR, "*.jsonl"))):
            with open(f, encoding="utf-8") as fh:
                index[json.loads(fh.readline())["ticker"]] = f
        _asx_index = index
        return index


def _load_asx(ticker):
    """ticker like 'BHP.AX' -> list of row dicts sorted by date ascending."""
    cached = _asx_cache.get(ticker)
    if cached is not None:
        return cached
    path = _load_asx_index().get(ticker)
    if path is None:
        raise KeyError(f"no ASX file for ticker {ticker}; known tickers: {_all_tickers()}")
    with _asx_lock:
        cached = _asx_cache.get(ticker)
        if cached is not None:
            return cached
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        rows.sort(key=lambda r: r["date"])
        _asx_cache[ticker] = rows
        return rows


def _all_tickers():
    return sorted(_load_asx_index())


def _norm(ticker):
    """'bhp' / 'BHP' / 'bhp.ax' -> 'BHP.AX'."""
    t = str(ticker).strip().upper()
    return t if t.endswith(".AX") else t + ".AX"


def _ticker_universe(exclude_tabcorp=True):
    tks = _all_tickers()
    if exclude_tabcorp:
        tks = [t for t in tks if t != TABCORP]
    return tks


def _rows_in_year(rows, year):
    return [r for r in rows if r["date"][:4] == str(year)]


def _simple_return(rows):
    """first-to-last close simple return in percent."""
    if len(rows) < 2:
        return None
    first, last = rows[0]["close"], rows[-1]["close"]
    return (last / first - 1.0) * 100.0


def _max_drawdown(rows):
    """returns (dd_pct_negative, peak_date, trough_date)."""
    peak = -math.inf
    peak_date = None
    worst = 0.0
    worst_peak_date = None
    worst_trough_date = None
    for r in rows:
        c = r["close"]
        if c > peak:
            peak = c
            peak_date = r["date"]
        dd = (c / peak - 1.0) * 100.0
        if dd < worst:
            worst = dd
            worst_peak_date = peak_date
            worst_trough_date = r["date"]
    return worst, worst_peak_date, worst_trough_date


def _daily_returns(rows):
    out = []
    for a, b in zip(rows, rows[1:]):
        out.append(b["close"] / a["close"] - 1.0)
    return out


def _mean(xs):
    return sum(xs) / len(xs)


def _std(xs, sample=True):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - (1 if sample else 0))
    return math.sqrt(var)


def _fmt_date(iso):
    """'2015-03-20' -> '20 Mar 2015' to match reference-answer style."""
    return datetime.strptime(iso, "%Y-%m-%d").strftime("%d %b %Y").lstrip("0")


def _asx(metric, **p):
    exclude = p.get("exclude_tabcorp", True)

    if metric in ("dimensions", "profile"):
        tks = _all_tickers()
        rows0 = _load_asx(tks[0])
        return {
            "n_tickers": len(tks),
            "rows_per_ticker": len(rows0),
            "start_date": rows0[0]["date"],
            "end_date": rows0[-1]["date"],
            "tickers": tks,
        }

    if metric == "annual_return":
        rows = _rows_in_year(_load_asx(p["ticker"]), p["year"])
        return {"ticker": p["ticker"], "year": p["year"], "return_pct": round(_simple_return(rows), 2)}

    if metric == "full_sample_return":
        rows = _load_asx(p["ticker"])
        return {"ticker": p["ticker"], "return_pct": round(_simple_return(rows), 2)}

    if metric == "rank_annual_returns":
        year = p["year"]
        res = []
        for t in _ticker_universe(exclude):
            rows = _rows_in_year(_load_asx(t), year)
            res.append((t, _simple_return(rows)))
        res.sort(key=lambda x: x[1], reverse=True)
        return {
            "year": year,
            "excluded_tabcorp": exclude,
            "ranking": [{"ticker": t, "return_pct": round(v, 2)} for t, v in res],
            "best": {"ticker": res[0][0], "return_pct": round(res[0][1], 2)},
            "worst": {"ticker": res[-1][0], "return_pct": round(res[-1][1], 2)},
        }

    if metric == "rank_full_sample_returns":
        res = []
        for t in _ticker_universe(exclude):
            res.append((t, _simple_return(_load_asx(t))))
        res.sort(key=lambda x: x[1], reverse=True)
        return {"ranking": [{"ticker": t, "return_pct": round(v, 2)} for t, v in res]}

    if metric == "avg_volume":
        # average daily volume over full sample, ranked
        res = []
        for t in _ticker_universe(exclude):
            rows = _load_asx(t)
            res.append((t, _mean([r["volume"] for r in rows])))
        res.sort(key=lambda x: x[1], reverse=True)
        return {
            "excluded_tabcorp": exclude,
            "ranking": [{"ticker": t, "avg_volume": round(v, 2)} for t, v in res],
            "highest": {"ticker": res[0][0], "avg_volume": round(res[0][1], 2)},
        }

    if metric == "max_drawdown":
        # single ticker
        if "ticker" in p:
            dd, pk, tr = _max_drawdown(_load_asx(p["ticker"]))
            return {"ticker": p["ticker"], "drawdown_pct": round(dd, 2),
                    "peak_date": _fmt_date(pk), "trough_date": _fmt_date(tr)}
        # rank worst drawdowns across universe
        res = []
        for t in _ticker_universe(exclude):
            dd, pk, tr = _max_drawdown(_load_asx(t))
            res.append((t, dd, pk, tr))
        res.sort(key=lambda x: x[1])  # most negative first
        top = p.get("top", 3)
        return {
            "excluded_tabcorp": exclude,
            "worst": [
                {"rank": i + 1, "ticker": t, "drawdown_pct": round(dd, 2),
                 "peak_date": _fmt_date(pk), "trough_date": _fmt_date(tr)}
                for i, (t, dd, pk, tr) in enumerate(res[:top])
            ],
        }

    if metric == "window_return":
        # return between two exact dates for one ticker
        rows = _load_asx(p["ticker"])
        d0, d1 = p["start"], p["end"]
        sel = [r for r in rows if d0 <= r["date"] <= d1]
        if len(sel) < 2:
            return {"error": "need >=2 rows in window"}
        return {"ticker": p["ticker"], "start": d0, "end": d1,
                "return_pct": round((sel[-1]["close"] / sel[0]["close"] - 1.0) * 100.0, 2)}

    if metric == "basket_window_return":
        # arithmetic mean of individual constituent window returns
        d0, d1 = p["start"], p["end"]
        tickers = p.get("tickers") or _ticker_universe(exclude)
        rets = []
        detail = {}
        for t in tickers:
            rows = _load_asx(t)
            sel = [r for r in rows if d0 <= r["date"] <= d1]
            if len(sel) >= 2:
                r = (sel[-1]["close"] / sel[0]["close"] - 1.0) * 100.0
                rets.append(r)
                detail[t] = round(r, 2)
        return {"start": d0, "end": d1, "n": len(rets),
                "basket_return_pct": round(_mean(rets), 2), "constituents": detail}

    if metric == "volatility":
        rows = _load_asx(p["ticker"])
        if "year" in p:
            rows = _rows_in_year(rows, p["year"])
        dr = _daily_returns(rows)
        ann = _std(dr) * math.sqrt(252) * 100.0
        return {"ticker": p["ticker"], "daily_vol_pct": round(_std(dr) * 100, 4),
                "annualized_vol_pct": round(ann, 2)}

    if metric == "correlation":
        a = _load_asx(p["ticker_a"])
        b = _load_asx(p["ticker_b"])
        # align on common dates
        bd = {r["date"]: r["close"] for r in b}
        pairs = [(r["close"], bd[r["date"]]) for r in a if r["date"] in bd]
        ra = _daily_returns([{"close": c} for c, _ in pairs])
        rb = _daily_returns([{"close": c} for _, c in pairs])
        ma, mb = _mean(ra), _mean(rb)
        cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
        da = math.sqrt(sum((x - ma) ** 2 for x in ra))
        db = math.sqrt(sum((y - mb) ** 2 for y in rb))
        return {"pair": [p["ticker_a"], p["ticker_b"]], "correlation": round(cov / (da * db), 3)}

    if metric == "quote":
        # exact close on a date
        rows = _load_asx(p["ticker"])
        want = _iso(p["date"])
        for r in rows:
            if r["date"] == want:
                return {"ticker": p["ticker"], "date": r["date"], "close": r["close"],
                        "open": r["open"], "high": r["high"], "low": r["low"], "volume": r["volume"]}
        return {"error": f"no {p['ticker']} row on {want} (market closed or outside 2015-2021)"}

    if metric == "event_study":
        return _asx_event_study(**p)

    return {"error": f"unknown asx metric '{metric}'"}


def _trading_calendar():
    """Sorted session dates ('YYYY-MM-DD'). Every ticker shares the same 1,774 rows."""
    return [r["date"] for r in _load_asx(_all_tickers()[0])]


def _asx_event_study(event_dates, horizon_days=7, horizon_sessions=None,
                     start_from="event_date", tickers=None, exclude_tabcorp=True):
    """Price reaction to one or more dated events (RBA decisions, AFR articles).

    Collapses the four-call 'look up the rate -> find the window -> basket
    return -> per-ticker returns' pattern into a single deterministic call, which
    is what keeps the multi-event questions inside the 60-second budget.

    ``horizon_days`` measures the window in calendar days from the event date
    ("the 5-12 Jun return"); ``horizon_sessions`` measures it in trading sessions
    from the window start ("the first five-session window after publication") and
    takes precedence when supplied.
    """
    calendar = _trading_calendar()
    universe = [_norm(t) for t in tickers] if tickers else _ticker_universe(exclude_tabcorp)
    if isinstance(event_dates, str):
        event_dates = [event_dates]

    events = []
    for raw in event_dates:
        event_dt = _parse_date(str(raw))
        event_iso = event_dt.strftime("%Y-%m-%d")

        # Window start: the event session itself, or the first session after it.
        if start_from == "next_session":
            starts = [i for i, d in enumerate(calendar) if d > event_iso]
        else:
            starts = [i for i, d in enumerate(calendar) if d >= event_iso]
        if not starts:
            events.append({"event_date": event_iso,
                           "error": "event date is after the ASX coverage window (ends 2021-12-30)"})
            continue
        i0 = starts[0]

        if horizon_sessions is not None:
            i1 = min(i0 + int(horizon_sessions), len(calendar) - 1)
        else:
            limit = (event_dt + timedelta(days=int(horizon_days))).strftime("%Y-%m-%d")
            ends = [i for i, d in enumerate(calendar) if d <= limit]
            i1 = max(ends[-1], i0)
        d0, d1 = calendar[i0], calendar[i1]

        per_ticker, returns = {}, []
        for t in universe:
            rows = [r for r in _load_asx(t) if d0 <= r["date"] <= d1]
            if len(rows) >= 2:
                pct = (rows[-1]["close"] / rows[0]["close"] - 1.0) * 100.0
                per_ticker[t] = round(pct, 2)
                returns.append(pct)

        rate = _rba("lookup_rate", date=event_iso)
        events.append({
            "event_date": event_iso,
            "window_start": d0,
            "window_end": d1,
            "sessions": i1 - i0 + 1,
            "rba_target_in_force": rate.get("rate"),
            "rba_effective_date": rate.get("effective_date"),
            "basket_return_pct": round(_mean(returns), 2) if returns else None,
            "n_constituents": len(returns),
            "excluded_tabcorp": exclude_tabcorp and not tickers,
            "ticker_returns": per_ticker,
        })

    return {"events": events, "basis": "first-to-last close over the window, simple return"}


# ===========================================================================
# AFR
# ===========================================================================
def _afr_haystack(rec):
    """HEADLINE + SUBHEAD + INTRO + TEXT combined (the mandated search scope)."""
    return " ".join(str(rec.get(k, "") or "") for k in ("HEADLINE", "SUBHEAD", "INTRO", "TEXT"))


def _load_afr():
    """Returns (records, haystacks) — two parallel lists.

    The combined four-field haystack is built ONCE at load time. Rebuilding it
    per query re-joins 219k x 4 fields on every pattern count, which dominated
    the AFR response time.
    """
    global _afr_cache, _afr_haystacks
    if _afr_cache is not None:
        return _afr_cache, _afr_haystacks
    with _afr_lock:
        if _afr_cache is not None:
            return _afr_cache, _afr_haystacks
        recs = []
        for f in sorted(glob.glob(os.path.join(AFR_DIR, "*.jsonl"))):
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        recs.append(json.loads(line))
        # Lowercased: every search is case-insensitive, so folding once at load
        # lets the prefilter below use plain substring search.
        _afr_haystacks = [_afr_haystack(r).lower() for r in recs]
        _afr_cache = recs
        return _afr_cache, _afr_haystacks


# Reference regexes for recurring AFR themes. `rba_rates` is the exact pattern
# behind the published MHQ084 answer (3,181 records in 2019) -- a paraphrase of
# it returns a different count and scores zero, so it is pinned here rather than
# left to the brain to reinvent.
AFR_PATTERN_PRESETS = {
    "rba_rates": r"interest rates?|cash rate|rate cut|rate hike|\bRBA\b",
    "unemployment": r"\bunemployment\b",
    "inflation": r"\binflation\b",
    "recession": r"\brecession\b",
    "covid": r"\bcovid\b|\bcoronavirus\b|\bpandemic\b",
    "housing": r"\bhousing\b|\bhouse prices\b|\bproperty prices\b",
}

_REGEX_META = set(r"\|()[]{}.*+?^$")


def normalize_pattern(pattern=None, preset=None):
    """Resolve a preset name, and word-anchor bare terms.

    A bare acronym like ``QBE`` substring-matches unrelated words and inflates
    the count, which the setup instructions call out explicitly. Anything with
    regex metacharacters is passed through untouched.
    """
    if preset:
        key = str(preset).strip().lower()
        if key not in AFR_PATTERN_PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {sorted(AFR_PATTERN_PRESETS)}")
        return AFR_PATTERN_PRESETS[key]
    if not pattern:
        raise ValueError("either pattern or preset is required for an AFR search")
    pattern = str(pattern).strip()
    if not (_REGEX_META & set(pattern)):
        return r"\b" + pattern + r"\b"
    return pattern


# Stopwords dropped when keyword-matching a headline query (paraphrase-robust search).
_STOP = {
    "the", "a", "an", "on", "in", "of", "to", "for", "and", "or", "as", "at", "by",
    "with", "from", "is", "are", "be", "its", "it", "this", "that", "these", "those",
    "article", "story", "piece", "report", "retrieve", "find", "about", "afr", "headline",
    "published", "dated", "titled", "entitled", "news",
}

# Light synonym expansion for finance-headline paraphrases (stocks<->shares etc.).
_SYN = {
    "stocks": {"shares", "equities", "stock"}, "shares": {"stocks", "equities", "share"},
    "rise": {"gain", "climb", "jump", "rally", "soar", "surge", "up"},
    "rises": {"gains", "climbs", "jumps"}, "fall": {"drop", "decline", "slump", "sink", "down"},
    "vaccine": {"vaccination", "immunisation", "immunization", "jab"},
    "rollout": {"roll-out", "rollouts"}, "rates": {"rate"}, "rate": {"rates"},
    "rba": {"reserve bank", "central bank"},
}


def _tokens(text):
    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if t not in _STOP and len(t) > 1]


def _expand(tokens):
    out = set(tokens)
    for t in tokens:
        out |= _SYN.get(t, set())
    return out


def _norm_afr_date(d):
    """Accept 'YYYYMMDD', 'YYYY-MM-DD', '23 Feb 2021' -> 'YYYYMMDD' (or None)."""
    if not d:
        return None
    d = str(d).strip()
    if re.fullmatch(r"\d{8}", d):
        return d
    for fmt in ("%Y-%m-%d", "%d %b %Y", "%d %B %Y"):
        try:
            return datetime.strptime(d, fmt).strftime("%Y%m%d")
        except ValueError:
            pass
    return None


def _article_payload(r, score=None, method=None, candidates=None):
    out = {
        "HEADLINE": r["HEADLINE"],
        "PUBLICATIONDATE": r["PUBLICATIONDATE"],
        "SUBHEAD": r.get("SUBHEAD", ""),
        "INTRO": r.get("INTRO", ""),
        "TEXT": (r.get("TEXT", "") or "")[:4000],
    }
    if score is not None:
        out["match_score"] = score
    if method:
        out["match_method"] = method
    if candidates:
        out["other_candidates"] = candidates
    return out


def _afr_year(rec):
    return rec["PUBLICATIONDATE"][:4]


def _afr_month(rec):
    return rec["PUBLICATIONDATE"][:6]  # YYYYMM


def _split_alternation(pattern):
    """Split a regex on top-level '|' only (ignoring | inside (...) or [...])."""
    parts, depth, cur = [], 0, []
    i = 0
    while i < len(pattern):
        c = pattern[i]
        if c == "\\" and i + 1 < len(pattern):
            cur.append(pattern[i:i + 2])
            i += 2
            continue
        if c in "([":
            depth += 1
        elif c in ")]":
            depth -= 1
        if c == "|" and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    parts.append("".join(cur))
    return parts


def _literal_runs(branch):
    """Guaranteed-literal substrings of one alternation branch."""
    runs, cur, i = [], [], 0
    while i < len(branch):
        c = branch[i]
        if c == "\\":
            nxt = branch[i + 1] if i + 1 < len(branch) else ""
            if nxt and not nxt.isalnum():   # an escaped literal, e.g. \. or \-
                cur.append(nxt)
            else:                            # a class or anchor: \b \d \w \s
                runs.append("".join(cur))
                cur = []
            i += 2
            continue
        if c in "?*":                        # preceding char is optional
            if cur:
                cur.pop()
            runs.append("".join(cur))
            cur = []
        elif c in "+{}()[].^$":
            runs.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    runs.append("".join(cur))
    return [r for r in runs if r]


def _prefilter_literals(pattern):
    """One required lowercase literal per branch, or None if any branch has none.

    A record that contains none of these literals cannot match the pattern, so
    the expensive regex only runs on the survivors. Scanning 219k documents with
    the regex alone takes ~5s; the substring pass cuts that by an order of
    magnitude and keeps AFR questions inside the 60-second budget.
    """
    literals = []
    for branch in _split_alternation(pattern):
        runs = _literal_runs(branch)
        best = max(runs, key=len) if runs else ""
        if len(best) < 3:
            return None
        literals.append(best.lower())
    return literals or None


def _fold_pattern(pattern):
    """Lowercase a pattern's literal characters, leaving escapes intact.

    The haystacks are already folded, so matching a folded pattern without
    re.IGNORECASE is ~1.6x faster on multi-branch searches and gives identical
    results. Escapes are copied verbatim because \\b and \\B (or \\w and \\W)
    are different assertions.
    """
    out, i = [], 0
    while i < len(pattern):
        if pattern[i] == "\\" and i + 1 < len(pattern):
            out.append(pattern[i:i + 2])
            i += 2
            continue
        out.append(pattern[i].lower())
        i += 1
    return "".join(out)


def _afr(metric, **p):
    recs, hays = _load_afr()

    pattern = None
    if p.get("preset") or p.get("pattern"):
        pattern = normalize_pattern(pattern=p.get("pattern"), preset=p.get("preset"))
    rx = re.compile(_fold_pattern(pattern)) if pattern else None
    prefilter = _prefilter_literals(pattern) if pattern else None

    def hits():
        """Matching records, once per record across the four combined fields."""
        if prefilter:
            return [r for r, h in zip(recs, hays)
                    if any(lit in h for lit in prefilter) and rx.search(h)]
        return [r for r, h in zip(recs, hays) if rx.search(h)]

    if metric == "count":
        n = len(hits())
        return {"pattern": pattern, "count": n, "total_records": len(recs)}

    if metric == "count_by_year":
        by = {}
        for r in hits():
            by[_afr_year(r)] = by.get(_afr_year(r), 0) + 1
        peak_year = max(by, key=by.get) if by else None
        return {"pattern": pattern, "by_year": by, "peak_year": peak_year,
                "peak_count": by.get(peak_year)}

    if metric == "count_by_month":
        by = {}
        for r in hits():
            by[_afr_month(r)] = by.get(_afr_month(r), 0) + 1
        peak_month = max(by, key=by.get) if by else None
        return {"pattern": pattern, "by_month": by, "peak_month": peak_month,
                "peak_count": by.get(peak_month)}

    if metric == "count_year":
        yr = str(p["year"])
        n = sum(1 for r in hits() if _afr_year(r) == yr)
        return {"pattern": pattern, "year": yr, "count": n}

    if metric == "peak_year_and_month":
        by_y, by_m = {}, {}
        for r in hits():
            by_y[_afr_year(r)] = by_y.get(_afr_year(r), 0) + 1
            by_m[_afr_month(r)] = by_m.get(_afr_month(r), 0) + 1
        if not by_y:
            return {"pattern": pattern, "peak_year": None, "peak_month": None,
                    "note": "no AFR record matches this pattern"}
        py = max(by_y, key=by_y.get)
        pm = max(by_m, key=by_m.get)
        return {"pattern": pattern, "peak_year": py, "peak_year_count": by_y[py],
                "peak_month": pm, "peak_month_count": by_m[pm],
                "by_year": by_y}

    if metric == "find_article":
        # Robust to paraphrase: keyword-overlap ranking over HEADLINE (weighted) + INTRO,
        # anchored by date when given. Handles "Travel stocks take off on vaccine rollout"
        # vs "travel shares rising on the vaccine rollout" via token overlap + synonyms.
        hq = p.get("headline", "")
        dq = _norm_afr_date(p.get("date"))  # normalize to YYYYMMDD if provided

        # 1) Exact substring fast-path (cheap, and unambiguous when it hits).
        exact = [r for r in recs
                 if hq.lower().strip() and hq.lower().strip() in r["HEADLINE"].lower()
                 and (dq is None or r["PUBLICATIONDATE"] == dq)]
        if len(exact) == 1:
            return _article_payload(exact[0], score=None, method="exact")

        # 2) Keyword-overlap ranking.
        q_tokens = _expand(_tokens(hq))
        if not q_tokens:
            return {"error": "no searchable terms in headline query"}

        # Date window: exact day, else same month, else whole corpus (widen only if needed).
        def _pool(filter_fn):
            return [r for r in recs if filter_fn(r)]

        pools = []
        if dq:
            pools.append(_pool(lambda r: r["PUBLICATIONDATE"] == dq))          # exact day
            pools.append(_pool(lambda r: r["PUBLICATIONDATE"][:6] == dq[:6]))  # same month
        pools.append(recs)                                                     # everything

        for pool in pools:
            scored = []
            for r in pool:
                htok = _expand(_tokens(r["HEADLINE"]))
                itok = _tokens(r.get("INTRO", ""))
                # headline overlap weighted x3; intro overlap x1
                h_hits = len(q_tokens & htok)
                i_hits = len(q_tokens & set(itok))
                score = 3 * h_hits + i_hits
                if score > 0:
                    scored.append((score, h_hits, r))
            if scored:
                scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
                best = scored[0][2]
                runners = [{"HEADLINE": r["HEADLINE"], "PUBLICATIONDATE": r["PUBLICATIONDATE"],
                            "score": s} for s, _, r in scored[1:4]]
                return _article_payload(best, score=scored[0][0], method="keyword",
                                        candidates=runners)
        return {"error": "article not found", "query_terms": sorted(q_tokens)}

    if metric == "share":
        # fraction of records in a year (or overall) matching pattern
        yr = str(p["year"]) if p.get("year") else None
        pool = [(r, h) for r, h in zip(recs, hays) if (yr is None or _afr_year(r) == yr)]
        n = sum(1 for _, h in pool
                if (not prefilter or any(lit in h for lit in prefilter)) and rx.search(h))
        return {"pattern": pattern, "year": yr, "matches": n, "pool": len(pool),
                "share_pct": round(100.0 * n / len(pool), 2) if pool else 0.0}

    return {"error": f"unknown afr metric '{metric}'"}


# ===========================================================================
# Dispatch
# ===========================================================================
def query_data(dataset, metric, **params):
    """Single entry point the agent runtime calls on Qwen's behalf."""
    ds = dataset.lower()
    if ds == "rba":
        return _rba(metric, **params)
    if ds == "asx":
        return _asx(metric, **params)
    if ds == "afr":
        return _afr(metric, **params)
    return {"error": f"unknown dataset '{dataset}'"}


def coverage():
    """Shape and date range of all three datasets.

    Answers both the 'what are the dimensions of the ASX dataset' family and the
    'can the supplied data support this analysis' family (MHQ040, MHQ090) from a
    single call. AFR bounds come from the monthly filenames so this stays cheap
    even before the corpus is loaded.
    """
    rba = _load_rba()
    tickers = _all_tickers()
    asx0 = _load_asx(tickers[0])
    months = sorted(
        os.path.basename(f).replace("AFR_", "").split("-")[0]
        for f in glob.glob(os.path.join(AFR_DIR, "AFR_*.jsonl"))
    )
    return {
        "rba": {
            "start": rba[0]["date_str"], "end": rba[-1]["date_str"],
            "records": len(rba), "unit": "cash-rate decision records",
        },
        "asx": {
            "start": asx0[0]["date"], "end": asx0[-1]["date"],
            "n_tickers": len(tickers), "rows_per_ticker": len(asx0), "tickers": tickers,
        },
        "afr": {
            "start": f"{months[0][:4]}-{months[0][4:6]}" if months else None,
            "end": f"{months[-1][:4]}-{months[-1][4:6]}" if months else None,
            "monthly_files": len(months),
        },
        "note": (
            "ASX and AFR both end on 30 Dec 2021 / Dec 2021; RBA runs to 2026. Any "
            "question needing ASX prices or AFR news after 2021 is unsupported by "
            "the supplied evidence."
        ),
    }


if __name__ == "__main__":
    import pprint
    pprint.pprint(query_data("rba", "count_changes"))
    pprint.pprint(coverage())
