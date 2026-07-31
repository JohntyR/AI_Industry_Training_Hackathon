"""Does the tool layer actually supply every fact the judge asks for?

WHAT THIS MEASURES
------------------
An end-to-end score confounds three things: whether the brain routed to the
right tool, whether the tool returned the right facts, and whether the synthesis
model stated them. This isolates the middle one.

For each public question it executes the tools the brain is *supposed* to call,
concatenates the ``summary`` and ``must_state`` text those tools produce, and
grades that text with the same component grader used on real runs. The result is
an EVIDENCE CEILING: the best score achievable if routing and synthesis were
perfect.

  ceiling 100%  -> the tools supply everything; every remaining loss is routing
                   or synthesis, and building more tools will not help
  ceiling < 100% -> a graded fact is missing from the tool layer itself, and no
                   amount of prompt or fine-tuning work can recover it

It needs no model server, so it runs in seconds and can gate a commit.

The TOOL_PLANS below double as the routing specification: they are what the
system prompt is trying to get the brain to do.

Usage:
    python training/eval/tool_evidence_audit.py
    python training/eval/tool_evidence_audit.py --verbose   # show missing facts
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, _HERE)

import query_data as qd            # noqa: E402
import tools                       # noqa: E402
from grade_components import grade_answer, load_questions  # noqa: E402

# The intended tool call(s) for each public question: (tool_name, arguments).
# Arguments are written as the XML tool-call parser would deliver them -- strings
# for numbers and lists -- so this exercises the coercion path too.
TOOL_PLANS: dict[str, list[tuple[str, dict]]] = {
    "MHQ001": [("rba_rate_changes", {})],
    "MHQ035": [("rba_rate_changes", {"start_year": "2011", "end_year": "2013"})],
    "MHQ040": [("dataset_coverage", {})],
    "MHQ045": [("asx_returns", {"scope": "ranking", "year": "2018"})],
    "MHQ049": [("asx_market_data", {"measure": "avg_volume"})],
    "MHQ055": [("asx_risk", {"measure": "max_drawdown", "top": "3"})],
    "MHQ058": [
        ("afr_find_article", {"headline": "Travel stocks take off on vaccine rollout",
                              "date": "20210223"}),
        ("rba_rate_on_date", {"date": "2021-02-23"}),
    ],
    "MHQ061": [("afr_count", {"pattern": "unemployment", "group_by": "peak"})],
    "MHQ067": [
        ("afr_find_article", {"headline": "Why investors don't believe the RBA on interest rates",
                              "date": "20211125"}),
        ("rba_rate_on_date", {"date": "2021-11-25"}),
    ],
    "MHQ072": [("asx_event_study", {"event_dates": "2019-06-05", "horizon_days": "7",
                                    "tickers": "CBA,NAB,ANZ,BHP,RIO"})],
    "MHQ074": [("asx_event_study", {"event_dates": "['2019-06-05','2019-07-03','2019-10-02']"})],
    "MHQ076": [
        ("afr_count", {"pattern": "QBE", "year": "2021"}),
        ("asx_returns", {"scope": "ranking", "year": "2021"}),
    ],
    "MHQ080": [
        ("afr_find_article", {"headline": "Energy stocks shine as vaccines fuel oil rally",
                              "date": "20201128"}),
        ("asx_event_study", {"event_dates": "2020-11-28", "horizon_sessions": "5",
                             "start_from": "next_session"}),
    ],
    "MHQ084": [
        ("rba_rate_changes", {"start_year": "2019", "end_year": "2019"}),
        ("afr_count", {"preset": "rba_rates", "year": "2019"}),
        ("asx_returns", {"scope": "basket", "year": "2019"}),
    ],
    "MHQ090": [
        ("dataset_coverage", {}),
        ("rba_rate_changes", {"start_year": "2022", "end_year": "2023"}),
    ],
}

# Facts a question asks for that are a JUDGEMENT rather than a dataset lookup:
# sentiment labels and market direction come from the fine-tuned model reading
# the article text the tool returned. The tool layer's job is to deliver that
# text; it cannot be expected to emit the label itself, so these components are
# reported separately instead of counted as a tool-layer gap.
JUDGEMENT_COMPONENTS = {
    "MHQ058": {"C02", "C03"},
    "MHQ067": {"C02", "C03"},
    "MHQ080": {"C02", "C03", "C07"},
    "MHQ090": {"C01", "C03"},
}

_BY_NAME = {t.name: t for t in tools.ALL_TOOLS}


def run_plan(plan: list[tuple[str, dict]]) -> tuple[str, int, float]:
    """Execute a plan and return (evidence text, tool calls, seconds)."""
    started = time.monotonic()
    parts: list[str] = []
    for name, kwargs in plan:
        raw = _BY_NAME[name].invoke(kwargs)
        result = json.loads(raw)
        if result.get("error"):
            parts.append(f"[{name} ERROR] {result['error']}")
            continue
        parts.append(result.get("summary", ""))
        parts.extend(result.get("must_state", []))
        # Article text is evidence the synthesis model reads, not a stated fact.
        if result.get("TEXT"):
            parts.append(str(result["TEXT"])[:2000])
    return " ".join(p for p in parts if p), len(plan), time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="List unmatched components.")
    args = parser.parse_args()

    print("Loading datasets...")
    qd.warmup()

    questions = load_questions()
    total_earned = total_possible = 0.0
    total_calls = 0
    judgement_points = 0.0
    gaps: list[str] = []

    print()
    print(f"{'QUESTION':<10} {'EVIDENCE':>9} {'CALLS':>6} {'SECONDS':>8}  UNSUPPLIED")
    print("-" * 78)

    for qid, question in questions.items():
        plan = TOOL_PLANS.get(qid)
        if not plan:
            print(f"{qid:<10} {'NO PLAN':>9}")
            continue

        evidence, n_calls, seconds = run_plan(plan)
        graded = grade_answer(question, evidence)
        judgement = JUDGEMENT_COMPONENTS.get(qid, set())

        missing = [c for c in graded["components"] if not c["matched"]]
        tool_gaps = [c for c in missing if c["component_id"] not in judgement]
        deferred = [c for c in missing if c["component_id"] in judgement]
        judgement_points += sum(c["max_points"] for c in deferred)

        total_earned += graded["earned"]
        total_possible += graded["possible"]
        total_calls += n_calls

        note = ", ".join(c["component_id"] for c in tool_gaps) or (
            f"({', '.join(c['component_id'] for c in deferred)} = model judgement)"
            if deferred else "-"
        )
        print(f"{qid:<10} {graded['earned']:>5.1f}/{graded['possible']:<3.0f} "
              f"{n_calls:>6} {seconds:>8.2f}  {note}")

        for comp in tool_gaps:
            gaps.append(f"{qid} {comp['component_id']}: {comp['expected_fact']}")
        if args.verbose:
            for comp in missing:
                kind = "JUDGEMENT" if comp["component_id"] in judgement else "TOOL GAP"
                print(f"           {kind}: {comp['expected_fact']}")

    supplied = total_earned + judgement_points
    print("-" * 78)
    print(f"{'TOTAL':<10} {total_earned:>5.1f}/{total_possible:<3.0f} "
          f"{total_calls:>6} {'':>8}  evidence ceiling "
          f"{100.0 * total_earned / total_possible:.1f}%")
    print(f"{'':<10} {'':>9} {'':>6} {'':>8}  including model-judgement components: "
          f"{100.0 * supplied / total_possible:.1f}%")
    print(f"{'':<10} {'':>9} {total_calls / len(TOOL_PLANS):>6.2f} calls/question")

    if gaps:
        print()
        print("TOOL-LAYER GAPS -- a graded fact no tool supplies:")
        for gap in gaps:
            print(f"  - {gap}")
        return 1

    print()
    print("No tool-layer gaps: every dataset-derived graded fact is supplied by a tool.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
