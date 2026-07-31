"""Run the 15 public questions through the live agent and score them.

This is the end-to-end measurement: the real Qwen brain routing to the real
tools, with the real synthesis model writing the answer. It mirrors how the
official harness behaves -- three questions in flight at once, per-question
latency recorded, the 60-second slow-penalty threshold flagged.

The output file has the same shape as ``logs/langchain_public_eval.json`` and is
graded by ``training/eval/grade_components.py``, so two runs are directly
comparable: grade the old log, grade the new log, and the difference is
attributable to whatever changed between them.

Requires a running agent (``uvicorn server:app --port 5000`` from ``src/``) and
reachable model endpoints.

Usage:
    python scripts/run_public_eval.py --out logs/public_eval_new.json
    python scripts/run_public_eval.py --endpoint http://127.0.0.1:5000 --workers 3
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import sys
import time

import httpx

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_REPO, "training", "eval"))

from grade_components import grade_all, load_questions  # noqa: E402

SLOW_PENALTY_S = 60.0
TIMEOUT_S = 300.0


def ask(endpoint: str, question: dict) -> dict:
    """Post one question, recording latency and whatever came back."""
    started = time.monotonic()
    try:
        response = httpx.post(
            f"{endpoint.rstrip('/')}/query",
            json={"question": question["prompt"]},
            timeout=TIMEOUT_S,
        )
        response.raise_for_status()
        payload = response.json()
        answer = payload.get("answer", "")
        trace = payload.get("tool_trace", [])
        steps = payload.get("steps", 0)
        error = ""
    except Exception as exc:
        answer, trace, steps = "", [], 0
        error = f"{type(exc).__name__}: {exc}"

    elapsed = time.monotonic() - started
    return {
        "id": question["id"],
        "difficulty": question.get("difficulty"),
        "latency_s": round(elapsed, 2),
        "slow": elapsed > SLOW_PENALTY_S,
        "steps": steps,
        "tool_calls": len(trace),
        "tools": trace,
        "answer": answer,
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:5000")
    parser.add_argument("--workers", type=int, default=3,
                        help="Concurrent requests, matching the official harness default.")
    parser.add_argument("--out", default=os.path.join(_REPO, "logs", "public_eval_run.json"))
    args = parser.parse_args()

    try:
        health = httpx.get(f"{args.endpoint.rstrip('/')}/health", timeout=10).json()
    except Exception as exc:
        print(f"health check failed against {args.endpoint}: {exc}")
        print("start the agent first:  cd src && uvicorn server:app --host 0.0.0.0 --port 5000")
        return 2
    print(f"health: {health}")
    if not health.get("evaluation_ready", True):
        print("WARNING: agent reports it is not evaluation-ready (check DOMAIN_PREDICT_MODE)")

    questions = load_questions()
    print(f"running {len(questions)} public questions, {args.workers} at a time\n")

    started = time.monotonic()
    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda q: ask(args.endpoint, q), questions.values()))
    wall = time.monotonic() - started

    graded = grade_all({r["id"]: r["answer"] for r in results}, questions)
    by_id = {g["id"]: g for g in graded["per_question"]}

    penalised = 0.0
    print(f"{'QUESTION':<10} {'SCORE':>11} {'LATENCY':>9} {'CALLS':>6}  MISSING")
    print("-" * 74)
    for result in sorted(results, key=lambda r: r["id"]):
        grade = by_id.get(result["id"], {})
        earned, possible = grade.get("earned", 0.0), grade.get("possible", 10.0)
        # Challenge_Brief.md -> Response-Time Rules: over 60s loses 20% of the
        # points earned on that question.
        after_penalty = earned * (0.8 if result["slow"] else 1.0)
        penalised += after_penalty
        missing = ", ".join(
            c["component_id"] for c in grade.get("components", []) if not c["matched"]
        ) or "-"
        flag = " SLOW" if result["slow"] else ""
        print(f"{result['id']:<10} {earned:>5.1f}/{possible:<5.0f} "
              f"{result['latency_s']:>8.1f}s{flag:<5} {result['tool_calls']:>3}  {missing}")
        if result["error"]:
            print(f"           ERROR: {result['error']}")

    possible = graded["possible"]
    latencies = sorted(r["latency_s"] for r in results)
    calls = [r["tool_calls"] for r in results]
    print("-" * 74)
    print(f"raw score        {graded['earned']:.1f}/{possible:.0f}  ({graded['pct']:.1f}%)")
    print(f"after slow penalty {penalised:.1f}/{possible:.0f}  "
          f"({100.0 * penalised / possible:.1f}%)")
    print(f"latency          median {latencies[len(latencies) // 2]:.1f}s  "
          f"p95 {latencies[int(len(latencies) * 0.95)]:.1f}s  max {latencies[-1]:.1f}s")
    print(f"tool calls       mean {sum(calls) / len(calls):.2f}  max {max(calls)}")
    print(f"wall clock       {wall:.1f}s for {len(results)} questions at {args.workers} workers")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for result in results:
        grade = by_id.get(result["id"], {})
        result["score"] = round(grade.get("earned", 0.0) / max(grade.get("possible", 10.0), 1), 3)
        result["missing_components"] = [
            c["component_id"] for c in grade.get("components", []) if not c["matched"]
        ]
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
