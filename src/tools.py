"""LangChain tool surface over the deterministic dataset engine.

ARCHITECTURE
------------
Three layers, deliberately separated (Challenge_Brief.md, Required Model Roles):

    tools.py          typed tool surface the Qwen agent-brain calls
    query_data.py     deterministic engine -- pure stdlib, no LLM, unit-tested
    data set/         the approved local datasets, read-only

Qwen chooses a tool and its arguments; this module validates and executes the
call; the fine-tuned Nemotron synthesises the answer from what comes back. No
model ever performs arithmetic.

WHY TASK-SHAPED TOOLS RATHER THAN ONE query_data(dataset, metric, ...)
----------------------------------------------------------------------
The handout's reference interface is a single tool with a free-text ``metric``
and a bag of optional parameters. That shape makes the brain guess two things at
once -- which metric name exists, and which parameters pair with it -- and on
the public set it guessed wrong in ways that cost real points (MHQ084 invented
its own AFR regex and returned 1,283 where the reference is 3,181).

Each tool here covers one question family with a closed ``Literal`` argument
set, so an invalid call is largely unrepresentable. Every one still routes
through the single deterministic entry point ``query_data.query_data()``, so
there remains exactly one code path to the data. ``query_data`` itself stays
registered last as a fallback for question families the narrow tools do not
anticipate -- roughly 75 of the ~90 benchmark questions are unseen.

WHAT EVERY TOOL RETURNS
-----------------------
JSON with the raw computed fields plus two presentation keys built in
``summaries.py``: ``summary`` (a judge-ready sentence, already signed and
formatted) and ``must_state`` (the individual facts the answer must contain).
The synthesis model composes from these instead of re-deriving prose from raw
JSON, which is where requested components were being dropped.

Three further jobs this layer does that the engine deliberately does not:

1. **Type coercion.** vLLM's ``qwen3_xml`` tool-call parser yields every
   ``<parameter=...>`` value as a STRING, so ``year`` arrives as ``"2018"`` and
   ``exclude_tabcorp`` as ``"true"``.
2. **Argument repair.** Ticker aliases, date spellings and bare AFR search terms
   are normalised into the exact forms the reference answers were computed with.
3. **Errors as data.** LangGraph's ToolNode already converts a raised exception
   into a ToolMessage, but a raw traceback tells the brain nothing about how to
   fix the call, so it burns a retry -- and each retry is another round trip
   against a 60-second budget. Every failure here comes back as
   ``{"error", "hint", <the arguments as received>}`` instead.
"""

from __future__ import annotations

import json
from typing import Any, Literal, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field, field_validator

import query_data as qd
import summaries


def _coerce_list(value):
    """Accept a real list, a JSON array string, or a comma-separated string.

    vLLM's ``qwen3_xml`` parser hands every argument over as a string, so a
    list-typed field arrives as ``"['CBA.AX', 'NAB.AX']"`` or ``"CBA,NAB"`` and
    would fail schema validation before the tool body ever runs. Coercing in a
    ``mode="before"`` validator fixes it at the boundary rather than forcing the
    model to retry.
    """
    if value is None or isinstance(value, list):
        return value
    text = str(value).strip()
    if text.startswith("["):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            text = text.strip("[]")
    return [part.strip().strip("'\"") for part in text.split(",") if part.strip()]

# ---------------------------------------------------------------------------
# Shared coercion helpers
# ---------------------------------------------------------------------------

_TICKER_ALIASES = {
    "agl": "AGL.AX", "amp": "AMP.AX", "anz": "ANZ.AX", "aurizon": "AZJ.AX",
    "bhp": "BHP.AX", "cba": "CBA.AX", "commonwealth bank": "CBA.AX",
    "cromwell": "CMW.AX", "gpt": "GPT.AX", "iag": "IAG.AX", "nab": "NAB.AX",
    "national australia bank": "NAB.AX", "qantas": "QAN.AX", "qbe": "QBE.AX",
    "rio": "RIO.AX", "rio tinto": "RIO.AX", "stockland": "SGP.AX",
    "suncorp": "SUN.AX", "tabcorp": "TAH.AX", "transurban": "TCL.AX", "tpg": "TPG.AX",
}


def _int(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _bool(value, default=True):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in ("false", "0", "no", "none", "")


def _ticker(value):
    """'qantas' / 'QAN' / 'qan.ax' -> 'QAN.AX'."""
    if not value:
        return None
    key = str(value).strip().lower()
    if key in _TICKER_ALIASES:
        return _TICKER_ALIASES[key]
    key = key.replace(".ax", "")
    if key in _TICKER_ALIASES:
        return _TICKER_ALIASES[key]
    upper = str(value).strip().upper()
    return upper if upper.endswith(".AX") else upper + ".AX"


def _tickers(value):
    """Accept a list, a JSON array string, or a comma-separated string."""
    if not value:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                value = text.strip("[]").split(",")
        else:
            value = text.split(",")
    return [_ticker(v) for v in value if str(v).strip()] or None


def _emit(result: dict, summarizer=None) -> str:
    """Attach the judge-ready summary and serialise for the model."""
    if summarizer and not result.get("error"):
        try:
            result = {**result, **summarizer(result)}
        except Exception as exc:                       # never let formatting kill a call
            result = {**result, "summary_error": f"{type(exc).__name__}: {exc}"}
    return json.dumps(result, default=str)


def _fail(message: str, hint: str, **context) -> str:
    return json.dumps({"error": message, "hint": hint, **context})


def _guard(summarizer=None):
    """Run an engine call, converting any exception into readable tool output."""

    def runner(fn, hint, **context):
        try:
            return _emit(fn(), summarizer)
        except (KeyError, ValueError, TypeError) as exc:
            return _fail(f"{type(exc).__name__}: {exc}", hint, **context)
        except Exception as exc:
            return _fail(f"unexpected {type(exc).__name__}: {exc}", hint, **context)

    return runner


_run = _guard()


# ===========================================================================
# RBA -- cash-rate decisions, 3 Feb 2010 to 17 Jun 2026, 175 records
# ===========================================================================


class RateOnDateArgs(BaseModel):
    date: str = Field(description="e.g. 2021-02-23, 23 Feb 2021 or 20210223.")


@tool("rba_rate_on_date", args_schema=RateOnDateArgs)
def rba_rate_on_date(date: str) -> str:
    """Cash-rate target in force ON a date: the latest decision at or before it."""
    return _guard(summaries.rba_rate_on_date)(
        lambda: qd.query_data("rba", "lookup_rate", date=date),
        "Pass a single calendar date within 2010-2026.",
        date=date,
    )


class RateChangesArgs(BaseModel):
    start_year: Optional[int] = Field(default=None, description="Omit for the whole dataset.")
    end_year: Optional[int] = Field(default=None, description="Omit for the whole dataset.")


@tool("rba_rate_changes", args_schema=RateChangesArgs)
def rba_rate_changes(start_year: Optional[int] = None, end_year: Optional[int] = None) -> str:
    """Rate-change counts and the cumulative move.

    No years: dataset totals (changed / increases / decreases). With years: the
    cuts, hikes, split by year, cycle dates and endpoint targets for that period.
    """
    y0, y1 = _int(start_year), _int(end_year)
    if y0 is None and y1 is None:
        return _guard(summaries.rba_rate_changes)(
            lambda: qd.query_data("rba", "count_changes"),
            "No arguments are needed for dataset-wide totals.",
        )
    y0 = y0 if y0 is not None else y1
    y1 = y1 if y1 is not None else y0
    return _guard(summaries.rba_rate_changes)(
        lambda: qd.query_data("rba", "period_summary", start_year=y0, end_year=y1),
        "Pass start_year and end_year as four-digit years.",
        start_year=y0, end_year=y1,
    )


@tool("rba_rate_extremes")
def rba_rate_extremes() -> str:
    """Highest and lowest cash-rate targets, each with first effective date,
    board decision date, and how many records show that rate.
    """
    return _guard(summaries.rba_rate_extremes)(
        lambda: qd.query_data("rba", "extremes"), "No arguments required."
    )


@tool("rba_longest_hold")
def rba_longest_hold() -> str:
    """Longest stretch with rates unchanged: days, both dates, rate held, rate after."""
    return _guard(summaries.rba_longest_hold)(
        lambda: qd.query_data("rba", "max_hold_streak"), "No arguments required."
    )


# ===========================================================================
# ASX -- 18 tickers x 1,774 sessions, 2 Jan 2015 to 30 Dec 2021
# ===========================================================================


class ReturnsArgs(BaseModel):
    scope: Literal["ticker", "basket", "ranking"] = Field(
        description="ticker=one company; basket=equal-weighted mean; ranking=all, best to worst."
    )
    ticker: Optional[str] = Field(default=None, description="For scope=ticker, e.g. BHP.AX.")
    tickers: Optional[list[str]] = Field(default=None, description="Basket constituents.")
    year: Optional[int] = Field(default=None, description="Calendar year, e.g. 2018.")
    start: Optional[str] = Field(default=None, description="Window start, e.g. 2019-06-05.")
    end: Optional[str] = Field(default=None, description="Window end.")
    exclude_tabcorp: Optional[bool] = Field(default=True, description="Keep true unless asked.")

    _coerce_tickers = field_validator("tickers", mode="before")(_coerce_list)


@tool("asx_returns", args_schema=ReturnsArgs)
def asx_returns(
    scope: str,
    ticker: Optional[str] = None,
    tickers: Optional[list[str]] = None,
    year: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    exclude_tabcorp: Optional[bool] = True,
) -> str:
    """First-to-last close returns: one ticker, an equal-weighted basket, or a ranking.

    Pass year, or start+end, or neither for the full 2015-2021 sample.
    scope='ranking' gives best, worst and every position in one call.
    """
    year = _int(year)
    exclude = _bool(exclude_tabcorp)
    basket = _tickers(tickers)
    label = (
        f"{year}" if year
        else f"{summaries.date(start)} to {summaries.date(end)}" if start and end
        else "the full 2015-2021 sample"
    )

    def call():
        if scope == "ranking":
            if year:
                out = qd.query_data("asx", "rank_annual_returns", year=year, exclude_tabcorp=exclude)
            else:
                out = qd.query_data("asx", "rank_full_sample_returns", exclude_tabcorp=exclude)
                out["best"], out["worst"] = out["ranking"][0], out["ranking"][-1]
                out["excluded_tabcorp"] = exclude
            return {**out, "period_label": label}

        if scope == "basket":
            if not (start and end):
                # A calendar-year or full-sample basket is the mean of the
                # constituents' individual returns over that period.
                ranked = (
                    qd.query_data("asx", "rank_annual_returns", year=year, exclude_tabcorp=exclude)
                    if year else
                    qd.query_data("asx", "rank_full_sample_returns", exclude_tabcorp=exclude)
                )
                rows = [r for r in ranked["ranking"] if not basket or r["ticker"] in basket]
                mean = sum(r["return_pct"] for r in rows) / len(rows)
                return {"basket_return_pct": round(mean, 2), "n": len(rows),
                        "excluded_tabcorp": exclude, "period_label": label,
                        "constituents": {r["ticker"]: r["return_pct"] for r in rows}}
            out = qd.query_data("asx", "basket_window_return", start=start, end=end,
                                tickers=basket, exclude_tabcorp=exclude)
            out["excluded_tabcorp"] = exclude and not basket
            return {**out, "period_label": label}

        resolved = _ticker(ticker)
        if not resolved:
            raise ValueError("scope='ticker' requires a ticker")
        if year:
            out = qd.query_data("asx", "annual_return", ticker=resolved, year=year)
        elif start and end:
            out = qd.query_data("asx", "window_return", ticker=resolved, start=start, end=end)
        else:
            out = qd.query_data("asx", "full_sample_return", ticker=resolved)
        return {**out, "period_label": label}

    return _guard(summaries.asx_returns)(
        call,
        "scope must be ticker, basket or ranking; give year, or start and end, or neither.",
        scope=scope, ticker=ticker, year=year, start=start, end=end,
    )


class RiskArgs(BaseModel):
    measure: Literal["max_drawdown", "volatility", "correlation"]
    ticker: Optional[str] = Field(default=None, description="Omit for a drawdown ranking.")
    ticker_a: Optional[str] = Field(default=None, description="Correlation only.")
    ticker_b: Optional[str] = Field(default=None, description="Correlation only.")
    year: Optional[int] = Field(default=None, description="Volatility in one year.")
    top: Optional[int] = Field(default=3, description="How many ranked drawdowns.")
    exclude_tabcorp: Optional[bool] = Field(default=True, description="Keep true unless asked.")


@tool("asx_risk", args_schema=RiskArgs)
def asx_risk(
    measure: str,
    ticker: Optional[str] = None,
    ticker_a: Optional[str] = None,
    ticker_b: Optional[str] = None,
    year: Optional[int] = None,
    top: Optional[int] = 3,
    exclude_tabcorp: Optional[bool] = True,
) -> str:
    """Risk stats: max drawdown (with peak and trough dates, single or ranked),
    annualised volatility, or pairwise daily-return correlation.
    """
    exclude = _bool(exclude_tabcorp)
    year, top = _int(year), _int(top) or 3

    def call():
        if measure == "correlation":
            return qd.query_data("asx", "correlation",
                                 ticker_a=_ticker(ticker_a), ticker_b=_ticker(ticker_b))
        if measure == "volatility":
            params = {"ticker": _ticker(ticker)}
            if year:
                params["year"] = year
            return qd.query_data("asx", "volatility", **params)
        if ticker:
            return qd.query_data("asx", "max_drawdown", ticker=_ticker(ticker))
        return qd.query_data("asx", "max_drawdown", top=top, exclude_tabcorp=exclude)

    return _guard(summaries.asx_risk)(
        call,
        "correlation needs ticker_a and ticker_b; volatility needs ticker; "
        "max_drawdown takes a ticker or nothing.",
        measure=measure, ticker=ticker,
    )


class MarketDataArgs(BaseModel):
    measure: Literal["quote", "avg_volume"] = Field(
        description="quote=OHLCV on one date; avg_volume=average daily volume ranking."
    )
    ticker: Optional[str] = Field(default=None, description="For quote.")
    date: Optional[str] = Field(default=None, description="For quote, e.g. 2020-03-23.")
    exclude_tabcorp: Optional[bool] = Field(default=True, description="Keep true unless asked.")


@tool("asx_market_data", args_schema=MarketDataArgs)
def asx_market_data(
    measure: str,
    ticker: Optional[str] = None,
    date: Optional[str] = None,
    exclude_tabcorp: Optional[bool] = True,
) -> str:
    """Raw price and volume facts: an exact OHLCV row, or the average-daily-volume ranking."""
    exclude = _bool(exclude_tabcorp)

    def call():
        if measure == "quote":
            return qd.query_data("asx", "quote", ticker=_ticker(ticker), date=date)
        return qd.query_data("asx", "avg_volume", exclude_tabcorp=exclude)

    return _guard(summaries.asx_market_data)(
        call, "quote needs ticker and date; avg_volume needs neither.",
        measure=measure, ticker=ticker, date=date,
    )


class EventStudyArgs(BaseModel):
    event_dates: list[str] = Field(description="One or more event dates.")
    horizon_days: Optional[int] = Field(
        default=7, description="Window in CALENDAR days, e.g. 7 for 5-12 Jun."
    )
    horizon_sessions: Optional[int] = Field(
        default=None, description="Window in TRADING SESSIONS instead; wins over horizon_days."
    )
    start_from: Literal["event_date", "next_session"] = Field(
        default="event_date",
        description="event_date for RBA effective dates; next_session for article publication.",
    )
    tickers: Optional[list[str]] = Field(
        default=None, description="Also report these individually; the basket stays all 17."
    )
    exclude_tabcorp: Optional[bool] = Field(default=True, description="Keep true unless asked.")

    _coerce_dates = field_validator("event_dates", mode="before")(_coerce_list)
    _coerce_tickers = field_validator("tickers", mode="before")(_coerce_list)


@tool("asx_event_study", args_schema=EventStudyArgs)
def asx_event_study(
    event_dates: list[str],
    horizon_days: Optional[int] = 7,
    horizon_sessions: Optional[int] = None,
    start_from: str = "event_date",
    tickers: Optional[list[str]] = None,
    exclude_tabcorp: Optional[bool] = True,
) -> str:
    """Market reaction to dated events. Per event: RBA target in force, session
    window, basket return, per-ticker returns.

    Use for any "what happened after the rate cut / after the article" question.
    Handles several events in ONE call.
    """
    return _guard(summaries.asx_event_study)(
        lambda: qd.query_data(
            "asx", "event_study",
            event_dates=[str(d) for d in (event_dates or [])],
            horizon_days=_int(horizon_days) if horizon_days is not None else 7,
            horizon_sessions=_int(horizon_sessions),
            start_from=start_from,
            tickers=_tickers(tickers),
            exclude_tabcorp=_bool(exclude_tabcorp),
        ),
        "event_dates must be a list of dates inside 2015-2021; set horizon_sessions "
        "for a session-counted window or horizon_days for a calendar window.",
        event_dates=event_dates,
    )


# ===========================================================================
# AFR -- 219,538 articles, Jan 2015 to Dec 2021
# ===========================================================================


class AfrCountArgs(BaseModel):
    pattern: Optional[str] = Field(
        default=None, description="Search term; plain terms are word-anchored for you."
    )
    preset: Optional[Literal["rba_rates", "unemployment", "inflation",
                             "recession", "covid", "housing"]] = Field(
        default=None,
        description="Pinned reference regex. Use rba_rates for any rate/RBA count.",
    )
    group_by: Literal["total", "year", "month", "peak", "share"] = Field(
        default="total", description="peak = highest year AND highest month together."
    )
    year: Optional[int] = Field(default=None, description="Restrict to one calendar year.")


@tool("afr_count", args_schema=AfrCountArgs)
def afr_count(
    pattern: Optional[str] = None,
    preset: Optional[str] = None,
    group_by: str = "total",
    year: Optional[int] = None,
) -> str:
    """Count AFR articles matching a term: case-insensitive, once per record,
    across HEADLINE + SUBHEAD + INTRO + TEXT combined.
    """
    year = _int(year)
    metric = {
        "total": "count_year" if year else "count",
        "year": "count_by_year",
        "month": "count_by_month",
        "peak": "peak_year_and_month",
        "share": "share",
    }[group_by]

    params: dict[str, Any] = {"pattern": pattern, "preset": preset}
    if year and metric in ("count_year", "share"):
        params["year"] = year

    return _guard(summaries.afr_count)(
        lambda: qd.query_data("afr", metric, **params),
        r"Supply pattern or preset. Bare terms are word-anchored for you.",
        pattern=pattern, preset=preset, group_by=group_by, year=year,
    )


class AfrArticleArgs(BaseModel):
    headline: str = Field(description="Exact or approximate headline.")
    date: Optional[str] = Field(default=None, description="Publication date; improves the match.")


@tool("afr_find_article", args_schema=AfrArticleArgs)
def afr_find_article(headline: str, date: Optional[str] = None) -> str:
    """Fetch one AFR article's text for a sentiment question. Pair with rba_rate_on_date."""
    return _guard(summaries.afr_find_article)(
        lambda: qd.query_data("afr", "find_article", headline=headline, date=date),
        "Give the headline as written in the question, plus the publication date if stated.",
        headline=headline, date=date,
    )


# ===========================================================================
# Meta and fallback
# ===========================================================================


@tool("dataset_coverage")
def dataset_coverage() -> str:
    """Shape, date range and ticker list of all three datasets. Also settles whether
    a question is answerable: ASX and AFR both end in December 2021.
    """
    return _guard(summaries.dataset_coverage)(qd.coverage, "No arguments required.")


class RawQueryArgs(BaseModel):
    dataset: Literal["rba", "asx", "afr"]
    metric: str = Field(
        description="rba: count_increases, count_decreases, list. "
                    "asx: rank_full_sample_returns, window_return, basket_window_return. "
                    "afr: count_by_year, share."
    )
    params: Optional[dict] = Field(default=None, description="Arguments as an object.")


@tool("query_data", args_schema=RawQueryArgs)
def query_data_tool(dataset: str, metric: str, params: Optional[dict] = None) -> str:
    """FALLBACK only, when no tool above fits. Raw engine access by metric name."""
    supplied = dict(params or {})
    if "ticker" in supplied:
        supplied["ticker"] = _ticker(supplied["ticker"])
    return _run(
        lambda: qd.query_data(dataset, metric, **supplied),
        "Check the metric name and its arguments in the metric list.",
        dataset=dataset, metric=metric, params=supplied,
    )


# Order matters: the brain reads this list top to bottom, and the generic
# fallback deliberately sits last.
ALL_TOOLS = [
    rba_rate_on_date,
    rba_rate_changes,
    rba_rate_extremes,
    rba_longest_hold,
    asx_returns,
    asx_risk,
    asx_market_data,
    asx_event_study,
    afr_count,
    afr_find_article,
    dataset_coverage,
    query_data_tool,
]
