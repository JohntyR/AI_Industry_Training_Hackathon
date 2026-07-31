"""Component-based grader approximating the official hidden-question judge.

The official judge awards the points attached to each independently satisfied
component of a question and accepts equivalent expression -- date formats,
harmless numeric formatting, and sentiment synonyms (Challenge_Brief.md ->
Hidden-Question Evaluation). This module reproduces that shape locally so a
change to the agent or to the fine-tuned model can be measured instead of
eyeballed.

It is deliberately STRICT: a component counts as satisfied only when every
anchor in its ``expected_fact`` (dates, tickers, numbers within the declared
tolerance) is present in the answer. The real judge is an LLM and will be more
forgiving about phrasing. Use the score to compare two systems on identical
questions, where the same strictness applies to both; do not read the absolute
percentage as a predicted leaderboard score.

Tolerances follow each question's ``grading.tolerance_note``:
counts/dates/rankings exact; returns/drawdowns/volatility/shares +/-0.02;
correlations +/-0.001; quoted closes +/-0.0001; average volume +/-1 share.

Usage:
    python training/eval/grade_components.py logs/langchain_public_eval.json
    from grade_components import grade_answer, load_questions
"""

from __future__ import annotations

import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
PUBLIC_QUESTIONS = os.path.join(_REPO, "Participant_Package", "public_questions.jsonl")

_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Spelled counts are matched softly: either the word or the digit satisfies them.
_SPELLED = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
}

_STOP = {
    "the", "a", "an", "of", "to", "for", "and", "or", "as", "at", "by", "with",
    "from", "is", "are", "was", "were", "be", "been", "its", "it", "this",
    "that", "these", "those", "in", "on", "there", "their", "which", "than",
    "then", "s", "t",
}

# Sentiment / direction synonyms accepted by the official judge.
_SYN = {
    "positive": {"positive", "bullish", "optimistic", "upbeat", "favourable", "favorable"},
    "negative": {"negative", "bearish", "pessimistic", "downbeat", "unfavourable", "unfavorable"},
    "mixed": {"mixed", "ambivalent", "two-sided", "cautious"},
    "upward": {"upward", "up", "higher", "rise", "rises", "rising", "gain", "gains", "positive"},
    "downward": {"downward", "down", "lower", "fall", "falls", "falling", "decline", "declines"},
    "confirmed": {"confirmed", "confirms", "consistent", "supports", "supported"},
    "contradicted": {"contradicted", "contradicts", "inconsistent", "refutes"},
    "unsupported": {"unsupported", "unsupportable", "cannot", "not", "no", "insufficient"},
    "shares": {"shares", "stocks", "equities"},
    "cut": {"cut", "cuts", "reduction", "eased", "easing", "lowered"},
    "hike": {"hike", "hikes", "increase", "increases", "raised", "tightening"},
    "percentage": {"percentage", "percent", "pct", "pp", "points"},
    "records": {"records", "record", "rows", "entries", "decisions"},
}

_DATE_PATTERNS = [
    # 20 Mar 2015 / 3 February 2010
    (re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\.?\s+(\d{4})\b"), "dmy"),
    # 2015-03-20
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "iso"),
    # March 2015 / Mar 2015  (month precision)
    (re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{4})\b"), "my"),
    # 202005 (YYYYMM as emitted by the AFR metrics)
    (re.compile(r"\b(\d{4})(0[1-9]|1[0-2])\b"), "yyyymm"),
]

_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")
_TICKER_RE = re.compile(r"\b[A-Z]{2,4}\.AX\b")


def _norm_dates(text: str) -> tuple[set[str], str]:
    """Return (ISO-ish date strings, text with those dates blanked out).

    Dates are removed before numeric extraction so that day and year digits are
    not double-counted as standalone numbers.
    """
    found: set[str] = set()
    residue = text
    for pattern, kind in _DATE_PATTERNS:
        for m in list(pattern.finditer(residue)):
            try:
                if kind == "dmy":
                    day, mon, year = m.group(1), m.group(2)[:3].lower(), m.group(3)
                    if mon not in _MONTHS:
                        continue
                    found.add(f"{year}-{_MONTHS[mon]:02d}-{int(day):02d}")
                elif kind == "iso":
                    found.add(f"{m.group(1)}-{m.group(2)}-{int(m.group(3)):02d}")
                elif kind == "my":
                    mon = m.group(1)[:3].lower()
                    if mon not in _MONTHS:
                        continue
                    found.add(f"{m.group(2)}-{_MONTHS[mon]:02d}")
                else:  # yyyymm
                    found.add(f"{m.group(1)}-{m.group(2)}")
            except (ValueError, KeyError):
                continue
            residue = residue[: m.start()] + " " * (m.end() - m.start()) + residue[m.end():]
    return found, residue


def _numbers(text: str) -> list[tuple[float, int]]:
    """Extract (value, decimal_places) pairs, ignoring thousands separators."""
    out = []
    for raw in _NUM_RE.findall(text):
        clean = raw.replace(",", "").rstrip("%")
        if clean in ("", "-", "+"):
            continue
        try:
            value = float(clean)
        except ValueError:
            continue
        decimals = len(clean.split(".")[1]) if "." in clean else 0
        out.append((value, decimals))
    return out


def _tolerance(value: float, decimals: int) -> float:
    """Tolerance implied by the questions' ``tolerance_note`` for this figure."""
    if decimals == 0:
        return 0.0                      # counts, years, rankings: exact
    if abs(value) >= 10_000:
        return 1.0                      # average volume: +/-1 share
    if decimals >= 4:
        return 0.0001                   # quoted closes
    if decimals == 3:
        return 0.001                    # correlations
    return 0.02                         # returns, drawdowns, rates, shares


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9']+", text.lower()) if t not in _STOP}


def _token_satisfied(token: str, answer_tokens: set[str]) -> bool:
    if token in answer_tokens:
        return True
    for canon, group in _SYN.items():
        if token == canon or token in group:
            if answer_tokens & group:
                return True
    return False


def component_matched(expected_fact: str, answer: str) -> tuple[bool, dict]:
    """Is every anchor in ``expected_fact`` present in ``answer``?"""
    # Drop ranking prefixes like "1) " so the list marker is not read as a value.
    expected = re.sub(r"^\s*\d+\)\s*", "", expected_fact)

    exp_dates, exp_residue = _norm_dates(expected)
    ans_dates, ans_residue = _norm_dates(answer)

    missing_dates = sorted(
        d for d in exp_dates
        if d not in ans_dates and not any(a.startswith(d) or d.startswith(a) for a in ans_dates)
    )

    exp_tickers = set(_TICKER_RE.findall(expected))
    ans_tickers = set(_TICKER_RE.findall(answer))
    missing_tickers = sorted(exp_tickers - ans_tickers)

    # Expected numbers come from the residue (so date digits are not demanded
    # twice), but the answer pool is the full text: a day or year the answer
    # happens to express as part of a date still satisfies a bare number.
    ans_numbers = _numbers(answer)
    missing_numbers = []
    for value, decimals in _numbers(exp_residue):
        tol = _tolerance(value, decimals)
        if not any(abs(value - a) <= max(tol, 1e-9) for a, _ in ans_numbers):
            missing_numbers.append(value)

    # Spelled counts ("Eight cuts"): the word or the digit satisfies them.
    ans_tokens = _tokens(answer)
    for word, digit in _SPELLED.items():
        if re.search(rf"\b{word}\b", expected.lower()):
            if word not in ans_tokens and not any(abs(digit - a) < 1e-9 for a, _ in ans_numbers):
                missing_numbers.append(float(digit))

    detail = {
        "missing_dates": missing_dates,
        "missing_tickers": missing_tickers,
        "missing_numbers": missing_numbers,
    }
    has_anchors = bool(exp_dates or exp_tickers or _numbers(exp_residue))
    if has_anchors:
        matched = not (missing_dates or missing_tickers or missing_numbers)
        return matched, detail

    # Label-only components (sentiment, direction, "No."): keyword overlap with
    # synonym expansion, since there is no number to anchor on.
    exp_tokens = _tokens(expected)
    if not exp_tokens:
        return False, detail
    hits = sum(1 for t in exp_tokens if _token_satisfied(t, ans_tokens))
    ratio = hits / len(exp_tokens)
    detail["keyword_ratio"] = round(ratio, 2)
    detail["missing_keywords"] = sorted(t for t in exp_tokens if not _token_satisfied(t, ans_tokens))
    return ratio >= 0.6, detail


def grade_answer(question: dict, answer: str) -> dict:
    """Score one answer against one question's grading components."""
    components = question.get("grading", {}).get("components", [])
    results, earned, possible = [], 0.0, 0.0
    for comp in components:
        points = float(comp.get("points", 0))
        possible += points
        matched, detail = component_matched(comp["expected_fact"], answer or "")
        if matched:
            earned += points
        results.append({
            "component_id": comp.get("component_id"),
            "expected_fact": comp["expected_fact"],
            "matched": matched,
            "points": points if matched else 0.0,
            "max_points": points,
            **detail,
        })
    return {
        "id": question.get("id"),
        "difficulty": question.get("difficulty"),
        "datasets": question.get("datasets"),
        "earned": round(earned, 2),
        "possible": round(possible, 2),
        "pct": round(100.0 * earned / possible, 1) if possible else 0.0,
        "components": results,
    }


def load_questions(path: str = PUBLIC_QUESTIONS) -> dict[str, dict]:
    """Load public_questions.jsonl keyed by question id."""
    with open(path, encoding="utf-8") as fh:
        return {q["id"]: q for q in (json.loads(line) for line in fh if line.strip())}


def grade_all(answers_by_id: dict[str, str], questions: dict[str, dict] | None = None) -> dict:
    """Grade a whole run: {question_id: answer} -> aggregate report."""
    questions = questions or load_questions()
    per_question = [
        grade_answer(questions[qid], answer)
        for qid, answer in answers_by_id.items()
        if qid in questions
    ]
    earned = sum(r["earned"] for r in per_question)
    possible = sum(r["possible"] for r in per_question)
    return {
        "n_questions": len(per_question),
        "earned": round(earned, 2),
        "possible": round(possible, 2),
        "pct": round(100.0 * earned / possible, 1) if possible else 0.0,
        "per_question": per_question,
    }


def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        print("usage: python training/eval/grade_components.py <run.json>")
        print("  <run.json>: list of {id, answer} objects (e.g. logs/langchain_public_eval.json)")
        return 2

    with open(argv[1], encoding="utf-8") as fh:
        run = json.load(fh)
    answers = {row["id"]: row.get("answer", "") for row in run}

    report = grade_all(answers)
    print(f"{'QUESTION':<10} {'SCORE':>12}  MISSING COMPONENTS")
    print("-" * 78)
    for row in report["per_question"]:
        missed = [c["component_id"] for c in row["components"] if not c["matched"]]
        print(f"{row['id']:<10} {row['earned']:>5.2f}/{row['possible']:<6.2f} "
              f"{row['pct']:>4.0f}%  {', '.join(missed) if missed else '-'}")
    print("-" * 78)
    print(f"{'TOTAL':<10} {report['earned']:>5.2f}/{report['possible']:<6.2f} {report['pct']:>4.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
