"""Deterministic data-query tools for the approved local datasets.

Interface follows ``Participant_Package/handout/02_execution_guide.md``:
``query_data(dataset, metric, ...)``. RBA/ASX/AFR facts must come from exact
parsing/calculation, never from model memory (see Challenge_Brief.md).

The computation lives in ``query_data.py`` (pure stdlib, verified to reproduce
all 15 public reference answers exactly -- run ``tests/test_public.py``). This
module is only the LangChain tool surface over it, and it does three jobs that
the engine itself deliberately does not:

1. **Explicit typed schema.** The brain sees one flat parameter per argument
   (``ticker``, ``year``, ``pattern``, ...) rather than a nested ``kwargs``
   blob, so it can actually express the call it wants.
2. **Type coercion.** vLLM's ``qwen3_xml`` tool-call parser extracts XML
   ``<parameter=...>`` values as STRINGS, so ``year`` arrives as ``"2018"`` and
   ``exclude_tabcorp`` as ``"true"``. Everything is coerced here before it
   reaches the engine.
3. **Errors as data.** A raised exception inside a tool escapes the LangGraph
   run and kills the whole request (a 500 with no ``answer`` field = zero score
   for that question). Every failure is returned as ``{"error": ...}`` instead,
   which comes back as a ToolMessage the brain can read and recover from.
"""

from __future__ import annotations

import json
from typing import Literal, Optional

from langchain.tools import tool
from pydantic import BaseModel, Field

import query_data as qd

# Full metric table, kept in the schema so the brain never has to guess a name.
_METRIC_DOC = (
    "RBA: count | count_changes (total/increases/decreases) | count_increases | "
    "count_decreases | extremes (highest/lowest rate + first date + record count) | "
    "lookup_rate (needs date) | max_hold_streak | "
    "period_summary (needs start_year,end_year -> cuts/hikes/by_year/cumulative/endpoints) | "
    "list (needs year). "
    "ASX: dimensions | annual_return (ticker,year) | full_sample_return (ticker) | "
    "rank_annual_returns (year) | rank_full_sample_returns | avg_volume | "
    "max_drawdown (ticker OR ranked worst via top=N) | window_return (ticker,start,end) | "
    "basket_window_return (start,end) | volatility (ticker[,year]) | "
    "correlation (ticker_a,ticker_b) | quote (ticker,date). "
    "AFR: count (pattern) | count_year (pattern,year) | count_by_year (pattern) | "
    "count_by_month (pattern) | peak_year_and_month (pattern) | share (pattern[,year]) | "
    "find_article (headline[,date]). "
    "META: coverage."
)

_TICKER_ALIASES = {
    "tabcorp": "TAH.AX", "qantas": "QAN.AX", "rio": "RIO.AX", "transurban": "TCL.AX",
    "aurizon": "AZJ.AX", "cromwell": "CMW.AX", "stockland": "SGP.AX", "suncorp": "SUN.AX",
}


def _as_int(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _as_bool(v, default=True):
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() not in ("false", "0", "no", "none", "")


def _norm_ticker(t):
    """'qantas' / 'QAN' / 'qan.ax' -> 'QAN.AX'. The in-file ticker is authoritative."""
    if not t:
        return None
    low = str(t).strip().lower().replace(".ax", "")
    if low in _TICKER_ALIASES:
        return _TICKER_ALIASES[low]
    up = str(t).strip().upper()
    return up if up.endswith(".AX") else up + ".AX"


class QueryDataArgs(BaseModel):
    """Flat, typed argument surface for the brain."""

    dataset: Literal["rba", "asx", "afr", "meta"] = Field(
        description="rba=cash-rate decisions; asx=18-company OHLCV 2015-2021; "
                    "afr=news corpus 2015-2021; meta=dataset coverage."
    )
    metric: str = Field(description=_METRIC_DOC)
    ticker: Optional[str] = Field(default=None, description="ASX ticker, e.g. BHP.AX, QAN.AX, AMP.AX.")
    ticker_a: Optional[str] = Field(default=None, description="First ticker for correlation.")
    ticker_b: Optional[str] = Field(default=None, description="Second ticker for correlation.")
    year: Optional[int] = Field(default=None, description="Calendar year, e.g. 2018.")
    start_year: Optional[int] = Field(default=None, description="First year of a period_summary window.")
    end_year: Optional[int] = Field(default=None, description="Last year of a period_summary window.")
    date: Optional[str] = Field(default=None, description="A single date: YYYY-MM-DD (or '3 Feb 2010').")
    start: Optional[str] = Field(default=None, description="Window start date YYYY-MM-DD.")
    end: Optional[str] = Field(default=None, description="Window end date YYYY-MM-DD.")
    pattern: Optional[str] = Field(
        default=None,
        description=r"AFR regex. Use word boundaries for whole-word counts, e.g. \bQBE\b, \bunemployment\b.",
    )
    headline: Optional[str] = Field(default=None, description="Article headline for find_article.")
    top: Optional[int] = Field(default=None, description="How many ranked rows to return, e.g. top=3.")
    exclude_tabcorp: Optional[bool] = Field(
        default=True,
        description="Defaults to true. Only set false if the question explicitly includes Tabcorp (TAH.AX).",
    )


@tool("query_data", args_schema=QueryDataArgs)
def query_data_tool(
    dataset: str,
    metric: str,
    ticker: Optional[str] = None,
    ticker_a: Optional[str] = None,
    ticker_b: Optional[str] = None,
    year: Optional[int] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    date: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    pattern: Optional[str] = None,
    headline: Optional[str] = None,
    top: Optional[int] = None,
    exclude_tabcorp: Optional[bool] = True,
) -> str:
    """Run a deterministic query against an approved local dataset (RBA/ASX/AFR).

    Returns exact structured numbers and dates as a JSON string. Every figure in
    the final answer must come from this tool -- never from memory.
    """
    ds = (dataset or "").strip().lower()
    mt = (metric or "").strip()

    # Coerce: the XML tool-call parser hands every value over as a string.
    params = {
        "ticker": _norm_ticker(ticker),
        "ticker_a": _norm_ticker(ticker_a),
        "ticker_b": _norm_ticker(ticker_b),
        "year": _as_int(year),
        "start_year": _as_int(start_year),
        "end_year": _as_int(end_year),
        "date": date,
        "start": start,
        "end": end,
        "pattern": pattern,
        "headline": headline,
        "top": _as_int(top),
    }
    params = {k: v for k, v in params.items() if v is not None}

    # Tabcorp is excluded by default; only ASX metrics accept the flag.
    if ds == "asx":
        params["exclude_tabcorp"] = _as_bool(exclude_tabcorp, default=True)

    try:
        if ds == "meta" or mt == "coverage":
            return json.dumps(qd.coverage(), default=str)
        result = qd.query_data(ds, mt, **params)
        return json.dumps(result, default=str)
    except TypeError as e:
        # Wrong/extra argument for this metric -- tell the brain precisely what to fix.
        return json.dumps({
            "error": f"bad arguments for {ds}/{mt}: {e}",
            "hint": "Check the required arguments for this metric in the metric list.",
            "metric_reference": _METRIC_DOC,
        })
    except Exception as e:
        return json.dumps({
            "error": f"{type(e).__name__}: {e}",
            "dataset": ds,
            "metric": mt,
            "hint": "Pick a valid dataset/metric pair from the metric list.",
            "metric_reference": _METRIC_DOC,
        })


class AfrArticleArgs(BaseModel):
    headline: str = Field(description="Article headline, exact or approximate.")
    date: Optional[str] = Field(
        default=None, description="PUBLICATIONDATE as YYYYMMDD, e.g. 20210223. Optional but improves the match."
    )


@tool("afr_get_article", args_schema=AfrArticleArgs)
def afr_get_article(headline: str, date: Optional[str] = None) -> str:
    """Fetch one AFR article's text by headline and optional date (YYYYMMDD).

    Use for article-grounded SENTIMENT questions: retrieve the article, then
    classify its sentiment and likely market direction from the returned text.
    Matching is paraphrase-tolerant, so an approximate headline still resolves.
    """
    try:
        result = qd.query_data("afr", "find_article", headline=headline, date=date)
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {e}", "headline": headline, "date": date})


ALL_TOOLS = [query_data_tool, afr_get_article]
