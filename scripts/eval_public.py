"""End-to-end evaluation: drive the 15 public questions through POST /query.

This is the only check that exercises the whole submitted pipeline the way the
organizers will -- over HTTP, against the running agent, with Qwen planning,
the runtime executing tools, and the fine-tuned Nemotron writing the answer.
Everything else in tests/ replaces part of that chain with a fake and therefore
cannot tell you whether the agent works.

It records what the harness records (answer, steps, tool_trace, wall-clock) and
scores it the way the harness scores it: component-based partial credit, plus
the response-time penalty, which is part of the real score and invisible if you
only look at correctness.

    # agent already serving on :5000
    python scripts/eval_public.py
    python scripts/eval_public.py --endpoint http://10.0.1.10:5000 --concurrency 3
    python scripts/eval_public.py --ids MHQ001,MHQ074 --verbose

Writes logs/langchain_public_eval.json (the recorded run, also consumed by
training/eval/compare_base_vs_ft.py) and logs/public_eval_summary.md.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time

import httpx

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))
sys.path.insert(0, os.path.join(_REPO, "training", "eval"))

from grade_components import grade_answer, load_questions  # noqa: E402

DEFAULT_LOG = os.path.join(_REPO, "logs", "langchain_public_eval.json")
DEFAULT_SUMMARY = os.path.join(_REPO, "logs", "public_eval_summary.md")

# Challenge_Brief.md -> Response-Time Rules.
FULL_CREDIT_S = 60.0
TIMEOUT_S = 300.0
SLOW_PENALTY = 0.20


def time_penalty(elapsed: float) -> tuple[float, str]:
    """(multiplier applied to earned points, label)."""
    if elapsed > TIMEOUT_S:
        return 0.0, "TIMEOUT"
    if elapsed > FULL_CREDIT_S:
        return 1.0 - SLOW_PENALTY, "-20%"
    return 1.0, ""


def ask(base_url: str, question: str, timeout: float) -> dict:
    """One POST /query, timed. Never raises -- a dead request is a zero, not a crash."""
    started = time.monotonic()
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(f"{base_url.rstrip('/')}/query", json={"question": question})
        elapsed = time.monotonic() - started
        if response.status_code != 200:
            return {"elapsed_s": elapsed, "error": f"HTTP {response.status_code}",
                    "answer": "", "steps": 0, "tools": []}
        payload = response.json()
        return {
            "elapsed_s": elapsed,
            "error": "" if isinstance(payload.get("answer"), str) and payload["answer"].strip()
                     else "empty or missing 'answer' (scores zero)",
            "answer": payload.get("answer", ""),
            "steps": payload.get("steps", 0),
            "tools": payload.get("tool_trace", []) or [],
        }
    except Exception as exc:
        return {"elapsed_s": time.monotonic() - started,
                "error": f"{type(exc).__name__}: {exc}", "answer": "", "steps": 0, "tools": []}


def evaluate(question: dict, base_url: str, timeout: float) -> dict:
    """Ask one question and grade the response."""
    outcome = ask(base_url, question["prompt"], timeout)
    graded = grade_answer(question, outcome["answer"])
    multiplier, penalty_label = time_penalty(outcome["elapsed_s"])

    # server.py records every degradation as an underscore-prefixed pseudo-tool.
    # Separating those from real tool calls matters: an answer produced from
    # zero tool results is ungrounded, and can still score by coincidence when
    # a generic refusal happens to match a label-only component. A run like
    # that looks partly successful and is worth nothing.
    tool_names = [t.get("tool", "") for t in outcome["tools"]]
    real_calls = [t for t in tool_names if not t.startswith("_")]
    degradations = [t for t in tool_names if t.startswith("_")]

    return {
        "id": question["id"],
        "difficulty": question.get("difficulty"),
        "datasets": question.get("datasets"),
        "prompt": question["prompt"],
        "answer": outcome["answer"],
        "steps": outcome["steps"],
        "tools": outcome["tools"],
        "tool_calls": len(real_calls),
        "degradations": degradations,
        "ungrounded": not real_calls,
        "elapsed_s": round(outcome["elapsed_s"], 2),
        "error": outcome["error"],
        "earned": graded["earned"],
        "possible": graded["possible"],
        "pct": graded["pct"],
        "effective": round(graded["earned"] * multiplier, 2),
        "penalty": penalty_label,
        "missing_components": [
            {"component_id": c["component_id"], "expected_fact": c["expected_fact"],
             "missing_numbers": c.get("missing_numbers", []),
             "missing_dates": c.get("missing_dates", []),
             "missing_tickers": c.get("missing_tickers", []),
             "missing_keywords": c.get("missing_keywords", [])}
            for c in graded["components"] if not c["matched"]
        ],
    }


def write_summary(path: str, rows: list[dict], totals: dict, base_url: str) -> None:
    lines = [
        "# Public-Question Evaluation",
        "",
        f"Run against `{base_url}` over `POST /query` -- the full pipeline: Qwen planning,",
        "runtime tool execution, fine-tuned Nemotron synthesis.",
        "",
        f"| Metric | Value |",
        f"|---|---:|",
        f"| Component score | {totals['earned']}/{totals['possible']} ({totals['pct']}%) |",
        f"| After time penalties | {totals['effective']}/{totals['possible']} ({totals['effective_pct']}%) |",
        f"| Slowest response | {totals['slowest']}s |",
        f"| Over 60s | {totals['slow']} of {len(rows)} |",
        f"| Tool calls made | {totals['tool_calls']} |",
        f"| Answers with no tool evidence | {totals['ungrounded']} |",
        f"| Questions hitting a degraded path | {totals['degraded']} |",
        f"| Failed requests | {totals['errors']} |",
        "",
    ]
    if totals["ungrounded"]:
        lines += [
            f"> **This run is not valid.** {totals['ungrounded']} of {len(rows)} answers were "
            "produced from zero tool results, so any points shown are coincidence rather than "
            "evidence — a generic refusal can satisfy a label-only component.",
            "",
        ]
    lines += [
        "| Question | Difficulty | Time | Tools | Score | Penalty | Missing components |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        missing = ", ".join(c["component_id"] or "?" for c in row["missing_components"]) or "-"
        lines.append(
            f"| {row['id']} | {row['difficulty']} | {row['elapsed_s']}s | {row['tool_calls']} | "
            f"{row['earned']}/{row['possible']} | {row['penalty'] or '-'} | {missing} |"
        )

    failed = [r for r in rows if r["error"]]
    if failed:
        lines += ["", "## Failed requests", ""]
        lines += [f"- **{r['id']}**: {r['error']}" for r in failed]

    lines += ["", "## Answers", ""]
    for row in rows:
        lines += [f"### {row['id']} — {row['earned']}/{row['possible']}", "",
                  f"> {row['prompt']}", "",
                  row["answer"] or "_(no answer)_", ""]
        if row["missing_components"]:
            lines.append("Missing:")
            lines += [f"- `{c['component_id']}` {c['expected_fact']}" for c in row["missing_components"]]
            lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", default="http://localhost:5000",
                        help="Agent base URL. Default: http://localhost:5000")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Simultaneous requests. Use 3 to mirror the harness default; "
                             "1 gives clean per-question timings.")
    parser.add_argument("--timeout", type=float, default=310.0,
                        help="Per-request timeout in seconds. Default: 310 (harness allows 300).")
    parser.add_argument("--ids", help="Comma-separated question ids to run (default: all).")
    parser.add_argument("--log", default=DEFAULT_LOG)
    parser.add_argument("--summary", default=DEFAULT_SUMMARY)
    parser.add_argument("--verbose", action="store_true", help="Print each answer as it arrives.")
    args = parser.parse_args()

    base_url = args.endpoint.rstrip("/")

    # Fail fast and clearly rather than 15 connection errors.
    try:
        health = httpx.get(f"{base_url}/health", timeout=10.0)
        if health.status_code != 200:
            print(f"GET {base_url}/health returned HTTP {health.status_code}; "
                  "the harness would skip this team entirely.", file=sys.stderr)
            return 2
        body = health.json() if health.headers.get("content-type", "").startswith("application/json") else {}
        if body.get("evaluation_ready") is False:
            print(f"WARNING: agent reports evaluation_ready=false "
                  f"(synthesis_mode={body.get('synthesis_mode')})\n", file=sys.stderr)
    except Exception as exc:
        print(f"Cannot reach {base_url}/health: {type(exc).__name__}: {exc}\n"
              f"Start the agent first:  uvicorn server:app --app-dir src --port 5000",
              file=sys.stderr)
        return 2

    questions = load_questions()
    if args.ids:
        wanted = {qid.strip() for qid in args.ids.split(",")}
        questions = {k: v for k, v in questions.items() if k in wanted}
        if not questions:
            print(f"No public questions matched {args.ids}", file=sys.stderr)
            return 2

    print(f"Running {len(questions)} question(s) against {base_url} "
          f"(concurrency {args.concurrency})\n")

    started = time.monotonic()
    ordered = list(questions.values())
    if args.concurrency > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            rows = list(pool.map(lambda q: evaluate(q, base_url, args.timeout), ordered))
    else:
        rows = []
        for question in ordered:
            row = evaluate(question, base_url, args.timeout)
            rows.append(row)
            flag = f"  {row['penalty']}" if row["penalty"] else ""
            print(f"  {row['id']:<8} {row['elapsed_s']:>6.1f}s  "
                  f"{row['earned']:>5.2f}/{row['possible']:<6.2f}{flag}"
                  f"{'  ERROR: ' + row['error'] if row['error'] else ''}")
            if args.verbose:
                print(f"           {row['answer'][:160]}")
    wall = time.monotonic() - started
    rows.sort(key=lambda r: r["id"])

    totals = {
        "earned": round(sum(r["earned"] for r in rows), 2),
        "possible": round(sum(r["possible"] for r in rows), 2),
        "effective": round(sum(r["effective"] for r in rows), 2),
        "slowest": round(max((r["elapsed_s"] for r in rows), default=0.0), 2),
        "slow": sum(1 for r in rows if r["elapsed_s"] > FULL_CREDIT_S),
        "errors": sum(1 for r in rows if r["error"]),
        "degraded": sum(1 for r in rows if r["degradations"]),
        "ungrounded": sum(1 for r in rows if r["ungrounded"]),
        "tool_calls": sum(r["tool_calls"] for r in rows),
    }
    totals["pct"] = round(100.0 * totals["earned"] / totals["possible"], 1) if totals["possible"] else 0.0
    totals["effective_pct"] = (round(100.0 * totals["effective"] / totals["possible"], 1)
                               if totals["possible"] else 0.0)

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=1)
    write_summary(args.summary, rows, totals, base_url)

    print(f"\n{'QUESTION':<10}{'TIME':>8}{'SCORE':>14}  MISSING")
    print("-" * 72)
    for row in rows:
        missing = ", ".join(c["component_id"] or "?" for c in row["missing_components"]) or "-"
        print(f"{row['id']:<10}{row['elapsed_s']:>7.1f}s"
              f"{row['earned']:>7.2f}/{row['possible']:<6.2f}{row['penalty']:<3} {missing}")
    print("-" * 72)
    print(f"{'TOTAL':<10}{wall:>7.1f}s{totals['earned']:>7.2f}/{totals['possible']:<6.2f}"
          f"    {totals['pct']}%")
    if totals["effective"] != totals["earned"]:
        print(f"{'EFFECTIVE':<10}{'':>8}{totals['effective']:>7.2f}/{totals['possible']:<6.2f}"
              f"    {totals['effective_pct']}%  after response-time penalties")
    if totals["ungrounded"]:
        print(f"\n  !! {totals['ungrounded']}/{len(rows)} answer(s) used ZERO tool results.")
        print("     Any points scored there are coincidence, not evidence -- a generic")
        print("     refusal can match a label-only component. Treat this run as invalid.")
    if totals["degraded"]:
        seen = sorted({d for r in rows for d in r["degradations"]})
        print(f"  !! {totals['degraded']}/{len(rows)} question(s) hit a degraded path: {', '.join(seen)}")
    if totals["errors"]:
        print(f"  !! {totals['errors']} request(s) failed -- see {os.path.relpath(args.summary, _REPO)}")
    print(f"\nWrote {os.path.relpath(args.log, _REPO)} and {os.path.relpath(args.summary, _REPO)}")

    return 1 if (totals["errors"] or totals["degraded"]) else 0


if __name__ == "__main__":
    sys.exit(main())
