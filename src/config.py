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

# "mock" (bootstrap default) or "llm" (must be set before official evaluation).
DOMAIN_PREDICT_MODE = os.getenv("DOMAIN_PREDICT_MODE", "mock").strip().lower()

EMBED_MODEL = os.getenv("EMBED_MODEL", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "")

MAX_AGENT_STEPS = int(os.getenv("MAX_AGENT_STEPS", "10"))
