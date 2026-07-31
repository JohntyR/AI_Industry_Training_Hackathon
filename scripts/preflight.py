"""Pre-submission check: everything that silently scores zero if left wrong.

Three failure modes cost far more than they should, and all three are invisible
from inside a working dev loop:

  1. ``submission.json`` still holds template values, so the harness calls the
     example endpoint, ``GET /health`` fails, and the team is SKIPPED -- zero on
     the whole 40% hidden-question category, regardless of agent quality.
  2. The agent is still in ``DOMAIN_PREDICT_MODE=mock``, so the fine-tuned
     Nemotron model is never called. Responses still look valid.
  3. The declared commit does not exist on the public remote, so architecture
     review inspects the wrong tree -- or nothing at all.

Run this from the repository root before submitting:

    python scripts/preflight.py                 # static + live checks
    python scripts/preflight.py --skip-live     # static checks only
    python scripts/preflight.py --endpoint http://10.0.0.5:5000

Exit code is 0 only when there are no blocking failures.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, ".."))

SUBMISSION = os.path.join(_REPO, "submission.json")

# Values carried over from submission_template.json. Any of these left in place
# means the file was never filled in.
PLACEHOLDERS = (
    "mock-team", "Mock Team", "example.com", "github.com/example",
    "172.20.x.x", "0123456789abcdef", "your-team", "<", "x.x",
)

SAMPLE_QUESTION = (
    "From the first RBA record to the last, how many cash-rate decisions "
    "changed the rate, and how many were increases versus decreases?"
)

_results: list[tuple[str, str, str]] = []   # (level, check, detail)


def record(level: str, check: str, detail: str = "") -> None:
    _results.append((level, check, detail))


def ok(check: str, detail: str = "") -> None:
    record("PASS", check, detail)


def warn(check: str, detail: str = "") -> None:
    record("WARN", check, detail)


def fail(check: str, detail: str = "") -> None:
    record("FAIL", check, detail)


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------
def check_submission() -> dict:
    if not os.path.exists(SUBMISSION):
        fail("submission.json exists", SUBMISSION)
        return {}
    try:
        with open(SUBMISSION, encoding="utf-8") as fh:
            sub = json.load(fh)
    except json.JSONDecodeError as exc:
        fail("submission.json parses", str(exc))
        return {}
    ok("submission.json parses")

    for key in ("team_id", "team_name", "github_url", "commit_sha", "agent", "model"):
        if key not in sub:
            fail(f"submission.json has '{key}'")

    blob = json.dumps(sub)
    hit = [p for p in PLACEHOLDERS if p in blob]
    if hit:
        fail("submission.json has no template placeholders", f"found: {', '.join(hit)}")
    else:
        ok("submission.json has no template placeholders")

    endpoint = (sub.get("agent") or {}).get("endpoint", "")
    if not endpoint:
        fail("agent.endpoint is set")
    elif any(p in endpoint for p in PLACEHOLDERS):
        pass  # already reported by the placeholder check; don't claim it is reachable
    elif re.search(r"localhost|127\.0\.0\.1|0\.0\.0\.0", endpoint):
        fail("agent.endpoint is externally reachable",
             f"{endpoint} is not reachable from the organizer machine")
    else:
        ok("agent.endpoint is externally reachable", endpoint)

    sha = sub.get("commit_sha", "")
    if not re.fullmatch(r"[0-9a-f]{40}", str(sha)):
        fail("commit_sha is a full 40-character SHA", str(sha))
    else:
        ok("commit_sha is a full 40-character SHA")

    model = sub.get("model") or {}
    if model.get("endpoint") and model.get("model_name"):
        ok("fine-tuned model declared", f"{model['model_name']} @ {model['endpoint']}")
    else:
        fail("fine-tuned model declared",
             "model.endpoint and model.model_name are needed for direct assessment")
    return sub


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", "-C", _REPO, *args], capture_output=True,
                              text=True, timeout=15).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def check_git(sub: dict) -> None:
    head = _git("rev-parse", "HEAD")
    declared = sub.get("commit_sha", "")
    if head and declared and re.fullmatch(r"[0-9a-f]{40}", str(declared)):
        if head == declared:
            ok("declared commit_sha matches HEAD", head[:12])
        else:
            fail("declared commit_sha matches HEAD",
                 f"HEAD={head[:12]} declared={str(declared)[:12]}")

    dirty = _git("status", "--porcelain")
    if dirty:
        warn("working tree is clean",
             f"{len(dirty.splitlines())} uncommitted change(s) -- judges inspect the pinned commit")
    else:
        ok("working tree is clean")

    unpushed = _git("log", "--oneline", "@{u}..HEAD")
    if unpushed:
        fail("HEAD is pushed to the remote",
             f"{len(unpushed.splitlines())} commit(s) not on origin -- the declared SHA will 404")
    elif _git("rev-parse", "--abbrev-ref", "@{u}"):
        ok("HEAD is pushed to the remote")

    tracked = _git("ls-files")
    leaked = [f for f in tracked.splitlines()
              if re.search(r"(^|/)\.env$|\.env\.|(^|/)(id_rsa|.*\.pem)$", f)]
    if leaked:
        fail("no credential files tracked by git", ", ".join(leaked))
    else:
        ok("no credential files tracked by git")


def check_required_paths() -> None:
    for path, why in [
        ("README.md", "architecture, run instructions, limitations"),
        ("src", "agent source"),
        ("training", "fine-tuning evidence"),
        ("logs", "diagnostic logs"),
        ("Participant_Package", "challenge materials"),
    ]:
        full = os.path.join(_REPO, path)
        if not os.path.exists(full):
            fail(f"{path} exists", why)
        elif os.path.isdir(full) and not [
            f for f in os.listdir(full) if not f.startswith(".")
        ]:
            fail(f"{path} is populated", f"empty -- required for {why}")
        else:
            ok(f"{path} is populated")


def check_config(sub: dict) -> None:
    sys.path.insert(0, os.path.join(_REPO, "src"))
    try:
        import config
    except Exception as exc:
        fail("src/config.py imports", f"{type(exc).__name__}: {exc}")
        return
    problems = config.evaluation_readiness()
    if problems:
        for problem in problems:
            fail("agent configuration is evaluation-ready", problem)
    else:
        ok("agent configuration is evaluation-ready",
           f"brain={config.BRAIN_MODEL} synthesis={config.DOMAIN_FT_MODEL} mode=llm")

    # A declared model name that differs from the served one means the judges
    # assess a model the agent never calls.
    declared = ((sub.get("model") or {}).get("model_name") or "")
    if declared and not any(p in declared for p in PLACEHOLDERS):
        if declared == config.DOMAIN_FT_MODEL:
            ok("declared model_name matches DOMAIN_FT_MODEL", declared)
        else:
            fail("declared model_name matches DOMAIN_FT_MODEL",
                 f"submission.json='{declared}' but the agent calls '{config.DOMAIN_FT_MODEL}'")


# ---------------------------------------------------------------------------
# Live checks
# ---------------------------------------------------------------------------
def _validate_response(payload: dict) -> list[str]:
    """Check a /query response against Participant_Package/validate.json."""
    errors = []
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        errors.append("'answer' must be a non-empty string (missing answer scores zero)")
    if "steps" in payload and not (isinstance(payload["steps"], int)
                                   and not isinstance(payload["steps"], bool)
                                   and payload["steps"] >= 0):
        errors.append("'steps' must be a non-negative integer")
    if "tool_trace" in payload:
        trace = payload["tool_trace"]
        if not isinstance(trace, list):
            errors.append("'tool_trace' must be an array")
        else:
            for i, entry in enumerate(trace):
                if not isinstance(entry, dict):
                    errors.append(f"tool_trace[{i}] must be an object")
                elif not isinstance(entry.get("args", {}), dict):
                    errors.append(f"tool_trace[{i}].args must be an object")
    return errors


def check_live(endpoint: str) -> None:
    try:
        import httpx
    except ImportError:
        warn("live checks", "httpx not installed; skipping")
        return

    base = endpoint.rstrip("/")
    try:
        response = httpx.get(f"{base}/health", timeout=10.0)
        if response.status_code == 200:
            ok("GET /health returns 200", f"{base}/health")
            body = response.json() if response.headers.get(
                "content-type", "").startswith("application/json") else {}
            if body.get("evaluation_ready") is False:
                fail("served agent reports evaluation_ready",
                     f"synthesis_mode={body.get('synthesis_mode')}")
        else:
            fail("GET /health returns 200",
                 f"HTTP {response.status_code} -- the team is SKIPPED entirely")
            return
    except Exception as exc:
        fail("GET /health returns 200", f"{type(exc).__name__}: {exc} -- the team is SKIPPED")
        return

    def query(question: str) -> tuple[float, dict | str]:
        import time
        started = time.monotonic()
        try:
            response = httpx.post(f"{base}/query", json={"question": question}, timeout=310.0)
            elapsed = time.monotonic() - started
            return elapsed, response.json()
        except Exception as exc:
            return time.monotonic() - started, f"{type(exc).__name__}: {exc}"

    elapsed, payload = query(SAMPLE_QUESTION)
    if isinstance(payload, str):
        fail("POST /query returns valid JSON", payload)
        return
    errors = _validate_response(payload)
    if errors:
        for error in errors:
            fail("POST /query matches validate.json", error)
    else:
        ok("POST /query matches validate.json", f"answer: {payload['answer'][:70]}...")

    if elapsed <= 60:
        ok("response time within the full-credit window", f"{elapsed:.1f}s")
    elif elapsed <= 300:
        warn("response time within the full-credit window",
             f"{elapsed:.1f}s -- 20% of earned points deducted for this question")
    else:
        fail("response time within the full-credit window", f"{elapsed:.1f}s -- timeout, zero points")

    # The harness sends up to three concurrent requests by default.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        outcomes = list(pool.map(query, [SAMPLE_QUESTION] * 3))
    bad = [p for _, p in outcomes if isinstance(p, str) or _validate_response(p)]
    if bad:
        fail("3 concurrent /query requests all valid", f"{len(bad)}/3 failed")
    else:
        slowest = max(e for e, _ in outcomes)
        ok("3 concurrent /query requests all valid", f"slowest {slowest:.1f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--endpoint", help="Agent base URL. Default: agent.endpoint from submission.json.")
    parser.add_argument("--skip-live", action="store_true", help="Static checks only.")
    args = parser.parse_args()

    sub = check_submission()
    check_git(sub)
    check_required_paths()
    check_config(sub)

    if not args.skip_live:
        endpoint = args.endpoint or (sub.get("agent") or {}).get("endpoint", "")
        if endpoint and not any(p in endpoint for p in PLACEHOLDERS):
            check_live(endpoint)
        else:
            warn("live checks", "no usable endpoint; pass --endpoint to run them")

    width = max(len(check) for _, check, _ in _results)
    icons = {"PASS": "✓", "WARN": "!", "FAIL": "✗"}
    print()
    for level, check, detail in _results:
        print(f" {icons[level]} {level:<4} {check:<{width}}  {detail}")

    failures = sum(1 for level, _, _ in _results if level == "FAIL")
    warnings = sum(1 for level, _, _ in _results if level == "WARN")
    print(f"\n{len(_results)} checks: {len(_results) - failures - warnings} passed, "
          f"{warnings} warnings, {failures} blocking failures")
    if failures:
        print("\nNOT READY TO SUBMIT -- fix the failures above.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
