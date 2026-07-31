"""Base-versus-fine-tuned comparison for the Nemotron synthesis model.

A documented comparison between the supplied base model and the team's
fine-tuned model is a required deliverable (Challenge_Brief.md -> Required
Deliverables) and is assessed directly under Fine-Tuned Model Quality.

METHOD -- this isolates the synthesis step, which is the only thing the
fine-tune changes:

  * Both models receive IDENTICAL, already-verified tool evidence, replayed
    from a recorded agent run. Neither model gets to make its own tool calls,
    so no score difference can come from a luckier reasoning path.
  * Both models receive the IDENTICAL system prompt used in production
    (imported from src/domain_model.py, not copied, so this cannot drift).
  * Both answers are graded by the same component-based scorer, at the
    tolerances declared in each question's grading.tolerance_note.

Any delta is therefore attributable to the fine-tune.

Usage:
    python training/eval/compare_base_vs_ft.py \
        --base-model nemotron-base \
        --ft-model   domain-ft \
        --base-url   http://<model-node>:8001/v1
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import httpx

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))

from grade_components import grade_all, load_questions  # noqa: E402
from domain_model import SYNTH_SYSTEM_PROMPT            # noqa: E402  (production prompt)

DEFAULT_EVIDENCE = os.path.join(_REPO, "logs", "langchain_public_eval.json")
DEFAULT_OUT = os.path.join(_REPO, "training", "metrics")


def load_evidence(path: str) -> dict[str, list[str]]:
    """{question_id: [tool result strings]} from a recorded agent run."""
    with open(path, encoding="utf-8") as fh:
        run = json.load(fh)
    return {
        row["id"]: [t.get("result", "") for t in row.get("tools", []) if t.get("result")]
        for row in run
    }


def synthesize(client: httpx.Client, base_url: str, api_key: str, model: str,
               question: str, evidence: list[str]) -> str:
    """One synthesis call against an OpenAI-compatible endpoint."""
    body = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYNTH_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Question: {question}\n\nVerified tool results:\n"
                f"{chr(10).join(evidence) or 'No tool evidence was returned.'}\n\nFinal answer:"
            )},
        ],
    }
    response = client.post(
        f"{base_url.rstrip('/')}/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def run_model(model: str, questions: dict, evidence: dict[str, list[str]],
              base_url: str, api_key: str) -> tuple[dict[str, str], dict]:
    answers: dict[str, str] = {}
    with httpx.Client() as client:
        for qid, question in questions.items():
            if qid not in evidence:
                continue
            try:
                answers[qid] = synthesize(
                    client, base_url, api_key, model, question["prompt"], evidence[qid]
                )
            except Exception as exc:                     # keep going; a dead call scores zero
                answers[qid] = ""
                print(f"  {qid}: FAILED ({type(exc).__name__}: {exc})", file=sys.stderr)
            print(f"  {qid}: {answers[qid][:90]!r}")
    return answers, grade_all(answers, questions)


def write_markdown(path: str, base_model: str, ft_model: str,
                   base_report: dict, ft_report: dict,
                   base_answers: dict[str, str], ft_answers: dict[str, str]) -> None:
    base_by_id = {r["id"]: r for r in base_report["per_question"]}
    ft_by_id = {r["id"]: r for r in ft_report["per_question"]}

    lines = [
        "# Base vs Fine-Tuned Synthesis Comparison",
        "",
        "Identical verified tool evidence and identical production system prompt for both models;",
        "graded by `training/eval/grade_components.py` at the official tolerances.",
        "",
        f"| Model | Score | Points |",
        f"|---|---:|---:|",
        f"| Base — `{base_model}` | {base_report['pct']}% | {base_report['earned']}/{base_report['possible']} |",
        f"| Fine-tuned — `{ft_model}` | {ft_report['pct']}% | {ft_report['earned']}/{ft_report['possible']} |",
        f"| **Delta** | **{round(ft_report['pct'] - base_report['pct'], 1):+}pp** | "
        f"**{round(ft_report['earned'] - base_report['earned'], 2):+}** |",
        "",
        "## Per question",
        "",
        "| Question | Difficulty | Base | Fine-tuned | Delta |",
        "|---|---|---:|---:|---:|",
    ]
    for qid in sorted(base_by_id):
        b, f = base_by_id[qid], ft_by_id.get(qid, {"earned": 0.0, "possible": b["possible"]})
        lines.append(
            f"| {qid} | {b['difficulty']} | {b['earned']}/{b['possible']} | "
            f"{f['earned']}/{f['possible']} | {round(f['earned'] - b['earned'], 2):+} |"
        )

    lines += ["", "## Sample answers", ""]
    for qid in sorted(base_by_id):
        lines += [
            f"### {qid}",
            "",
            f"- **Base:** {base_answers.get(qid, '') or '_(no answer)_'}",
            f"- **Fine-tuned:** {ft_answers.get(qid, '') or '_(no answer)_'}",
            "",
        ]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-model", required=True, help="Supplied base Nemotron model/alias.")
    parser.add_argument("--ft-model", default=os.getenv("DOMAIN_FT_MODEL", "domain-ft"),
                        help="Fine-tuned model/alias. Default: $DOMAIN_FT_MODEL.")
    parser.add_argument("--base-url", default=os.getenv("DOMAIN_BASE_URL", ""),
                        help="OpenAI-compatible base URL. Default: $DOMAIN_BASE_URL.")
    parser.add_argument("--api-key", default=os.getenv("DOMAIN_KEY", ""))
    parser.add_argument("--evidence", default=DEFAULT_EVIDENCE,
                        help="Recorded agent run supplying the tool evidence.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()

    if not args.base_url:
        parser.error("--base-url is required (or set DOMAIN_BASE_URL)")

    questions = load_questions()
    evidence = load_evidence(args.evidence)
    shared = [q for q in questions if q in evidence]
    if not shared:
        print(f"No overlap between {args.evidence} and the public questions.", file=sys.stderr)
        return 1
    print(f"Comparing on {len(shared)} questions with recorded evidence.\n")

    print(f"BASE — {args.base_model}")
    base_answers, base_report = run_model(args.base_model, questions, evidence,
                                          args.base_url, args.api_key)
    print(f"\nFINE-TUNED — {args.ft_model}")
    ft_answers, ft_report = run_model(args.ft_model, questions, evidence,
                                      args.base_url, args.api_key)

    os.makedirs(args.out_dir, exist_ok=True)
    json_path = os.path.join(args.out_dir, "base_vs_ft.json")
    md_path = os.path.join(args.out_dir, "base_vs_ft.md")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({
            "base": {"model": args.base_model, "report": base_report, "answers": base_answers},
            "fine_tuned": {"model": args.ft_model, "report": ft_report, "answers": ft_answers},
            "delta_pct": round(ft_report["pct"] - base_report["pct"], 1),
            "evidence_source": os.path.relpath(args.evidence, _REPO),
        }, fh, indent=2)
    write_markdown(md_path, args.base_model, args.ft_model,
                   base_report, ft_report, base_answers, ft_answers)

    print(f"\n{'':<14}{'SCORE':>8}{'POINTS':>16}")
    print(f"{'base':<14}{base_report['pct']:>7}%{base_report['earned']:>10}/{base_report['possible']}")
    print(f"{'fine-tuned':<14}{ft_report['pct']:>7}%{ft_report['earned']:>10}/{ft_report['possible']}")
    print(f"{'delta':<14}{round(ft_report['pct'] - base_report['pct'], 1):>+7}pp")
    print(f"\nWrote {os.path.relpath(json_path, _REPO)} and {os.path.relpath(md_path, _REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
