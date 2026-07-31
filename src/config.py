"""Environment configuration for the hackathon agent.

Reads the variables documented in ``Participant_Package/Setup_Instructions.md``
under "Reference Configuration". Keep all endpoints and credentials in the
environment (``.env``) rather than hard-coding them here.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "http://localhost:4000")
LITELLM_KEY = os.getenv("LITELLM_KEY", "")

BRAIN_MODEL = os.getenv("BRAIN_MODEL", "agent-brain")
DOMAIN_FT_MODEL = os.getenv("DOMAIN_FT_MODEL", "domain-ft")

# The brain and the synthesis model are served by DIFFERENT nodes when LiteLLM
# is not fronting them (Qwen on this box :8000, Nemotron on the model node
# :8001). Defaults to LITELLM_BASE_URL so a single LiteLLM gateway still works
# unchanged; set DOMAIN_BASE_URL to talk to the model node directly.
DOMAIN_BASE_URL = os.getenv("DOMAIN_BASE_URL", "") or LITELLM_BASE_URL
DOMAIN_KEY = os.getenv("DOMAIN_KEY", "") or LITELLM_KEY

# "mock" (bootstrap default) or "llm" (must be set before official evaluation).
DOMAIN_PREDICT_MODE = os.getenv("DOMAIN_PREDICT_MODE", "mock").strip().lower()

EMBED_MODEL = os.getenv("EMBED_MODEL", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "")

MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "10"))

# ---------------------------------------------------------------------------
# Response-time budget
#
# Scoring is time-sensitive (Challenge_Brief.md -> Response-Time Rules):
#   <= 60s   full earned points
#   <= 300s  20% of the earned points deducted
#   >  300s  timeout, zero for that question
#
# So the whole request is run against a wall-clock budget rather than left to
# finish whenever it finishes. The budget is split: the brain loop gets most of
# it, and a reserve is held back so the fine-tuned model always gets a chance to
# write the answer. Answering from partial evidence still earns per-component
# partial credit; returning raw tool JSON, or nothing, earns close to zero.
# ---------------------------------------------------------------------------

# Total wall-clock target for POST /query. Below 60 so a slow tail still lands
# inside the full-credit window.
REQUEST_DEADLINE_S = float(os.getenv("REQUEST_DEADLINE_S", "55"))

# Held back from the brain loop so synthesis is never starved.
SYNTH_RESERVE_S = float(os.getenv("SYNTH_RESERVE_S", "15"))

# Synthesis gets at least this long even if the brain overran, because a
# 20% slow-penalty on a real answer beats a fast answer made of raw JSON.
SYNTH_MIN_S = float(os.getenv("SYNTH_MIN_S", "12"))

# Per-call HTTP timeouts, so one hung model call cannot consume the budget.
# Qwen3 thinking mode. Off by default: the hidden reasoning block dominated
# brain latency (503 output tokens for one tool call) and caused timeouts under
# concurrency. Set BRAIN_THINKING=1 to measure it back on.
BRAIN_THINKING = os.getenv("BRAIN_THINKING", "").strip().lower() in ("1", "true", "yes")

BRAIN_TIMEOUT_S = float(os.getenv("BRAIN_TIMEOUT_S", "30"))
SYNTH_TIMEOUT_S = float(os.getenv("SYNTH_TIMEOUT_S", "20"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "1"))


def brain_budget_s() -> float:
    """Wall-clock allowance for the Qwen reasoning loop."""
    return max(5.0, REQUEST_DEADLINE_S - SYNTH_RESERVE_S)


def evaluation_readiness() -> list[str]:
    """Return blocking configuration problems for official evaluation.

    The cluster bootstrap starts in ``DOMAIN_PREDICT_MODE=mock``, which
    concatenates raw tool JSON instead of calling the fine-tuned model. Running
    the official evaluation in that mode means the submitted agent is not using
    the fine-tuned Nemotron model at all, which forfeits model-quality and
    architecture credit -- and it fails silently, because the responses still
    look like valid JSON. Everything that must be true before evaluation is
    checked here, in one place, so ``scripts/preflight.py`` and the server
    startup banner cannot disagree about what "ready" means.
    """
    problems: list[str] = []

    if DOMAIN_PREDICT_MODE != "llm":
        problems.append(
            f"DOMAIN_PREDICT_MODE is '{DOMAIN_PREDICT_MODE}', not 'llm' -- the fine-tuned "
            "Nemotron model is NOT being used to synthesize answers."
        )
    if not DOMAIN_FT_MODEL:
        problems.append("DOMAIN_FT_MODEL is unset -- no fine-tuned model to synthesize with.")
    if not DOMAIN_BASE_URL:
        problems.append("DOMAIN_BASE_URL (or LITELLM_BASE_URL) is unset.")
    if not LITELLM_BASE_URL:
        problems.append("LITELLM_BASE_URL is unset -- the Qwen brain is unreachable.")
    if not BRAIN_MODEL:
        problems.append("BRAIN_MODEL is unset.")

    return problems
