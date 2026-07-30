"""Deterministic data-query tool for the approved local datasets.

Interface follows ``Participant_Package/handout/02_execution_guide.md``:
``query_data(dataset, metric, ...)``. RBA and ASX facts must come from exact
parsing/calculation, never from model memory (see Challenge_Brief.md).

Only ``dataset="rba", metric="count_changes"`` is implemented so far, as the
first proof that the reason -> act loop round-trips real data. ASX and AFR
metrics raise ``NotImplementedError`` naming the metric table so gaps are
loud rather than guessed.
"""

from __future__ import annotations

import csv
from pathlib import Path

from langchain.tools import tool

REPO_ROOT = Path(__file__).resolve().parent.parent
RBA_CSV_PATH = REPO_ROOT / "data set" / "RBA Rates" / "RBA-rates.csv"

RBA_METRICS = {
    "count",
    "count_changes",
    "count_increases",
    "count_decreases",
    "extremes",
    "max_hold_streak",
    "lookup_rate",
    "list",
}
ASX_METRICS = {
    "annual_return",
    "rank_annual_returns",
    "full_sample_return",
    "volatility",
    "correlation",
    "max_drawdown",
}
AFR_METRICS = {"count", "count_by_month", "share"}


def _load_rba_rows() -> list[dict[str, str]]:
    with RBA_CSV_PATH.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _rba_count_changes() -> str:
    rows = _load_rba_rows()
    changes = [float(row["Change % points"]) for row in rows]
    non_zero = [c for c in changes if c != 0]
    increases = [c for c in non_zero if c > 0]
    decreases = [c for c in non_zero if c < 0]
    return (
        f"{len(non_zero)} of the {len(rows)} decision records changed the "
        f"rate: {len(increases)} increases and {len(decreases)} decreases."
    )


@tool
def query_data(dataset: str, metric: str, **kwargs: str) -> str:
    """Run a deterministic query against an approved local dataset.

    Args:
        dataset: One of "rba", "asx", "afr".
        metric: The metric to compute. See handout/02_execution_guide.md for
            the full metric table per dataset.
        **kwargs: Extra metric-specific arguments (e.g. ``pattern`` for AFR
            searches, ``ticker``/``year`` for ASX).
    """
    dataset = dataset.strip().lower()

    if dataset == "rba":
        if metric == "count_changes":
            return _rba_count_changes()
        if metric not in RBA_METRICS:
            raise NotImplementedError(
                f"Unknown RBA metric {metric!r}. Known RBA metrics: "
                f"{sorted(RBA_METRICS)}"
            )
        raise NotImplementedError(f"RBA metric {metric!r} is not implemented yet.")

    if dataset == "asx":
        if metric not in ASX_METRICS:
            raise NotImplementedError(
                f"Unknown ASX metric {metric!r}. Known ASX metrics: "
                f"{sorted(ASX_METRICS)}"
            )
        raise NotImplementedError(f"ASX metric {metric!r} is not implemented yet.")

    if dataset == "afr":
        if metric not in AFR_METRICS:
            raise NotImplementedError(
                f"Unknown AFR metric {metric!r}. Known AFR metrics: "
                f"{sorted(AFR_METRICS)}"
            )
        raise NotImplementedError(f"AFR metric {metric!r} is not implemented yet.")

    raise NotImplementedError(f"Unknown dataset {dataset!r}. Expected rba/asx/afr.")


ALL_TOOLS = [query_data]
