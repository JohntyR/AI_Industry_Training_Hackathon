"""Split what the brain sees from what the synthesis model sees.

Every tool result is appended to the brain's message history and re-sent on
every subsequent turn, so a 400-token article body is paid for again on each
call. With the brain served at a 4,096-token context -- shared between the
system prompt, the tool schemas, the question, every result so far and the
reply -- three tool calls overflow it and vLLM rejects the request outright.

But the brain does not need the bulk. It needs enough to decide whether another
call is required; it never writes the answer and never does arithmetic. The
article body, the 17-ticker ranking, the per-month histogram -- those are for
the synthesis model, which is called exactly once with no accumulation.

So the tool returns a compact view to the brain, carrying a ``ref``, and the
full payload is stashed here. ``server.py`` resolves each ref back to the full
result when it assembles the evidence for synthesis, so nothing is lost.

The store is a plain bounded dict rather than a ``ContextVar`` on purpose:
LangGraph runs sync tools in an executor thread, and context variables do not
reliably propagate across that boundary. Refs are UUIDs, so concurrent requests
cannot collide.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections import OrderedDict

# Ceiling on the compact view handed to the brain, in characters (~4 chars per
# token). Three calls at this size cost roughly 450 tokens of history.
MAX_BRAIN_CHARS = 600

# Fields that are pure bulk for planning purposes: long-form article text and
# full-population collections. The deterministic ``summary`` already states
# whatever the answer needs from them.
BULKY_FIELDS = frozenset({
    "TEXT", "INTRO", "SUBHEAD", "other_candidates",
    "constituents", "ranking", "by_year", "by_month", "rows", "events",
})

# Always preserved: without these the brain cannot plan or recover from errors.
ESSENTIAL_FIELDS = ("error", "hint", "summary", "must_state")

_MAX_ENTRIES = 512
_store: OrderedDict[str, str] = OrderedDict()
_lock = threading.Lock()


def stash(full_result: str) -> str:
    """Store a full tool result and return its reference id."""
    ref = uuid.uuid4().hex[:12]
    with _lock:
        _store[ref] = full_result
        while len(_store) > _MAX_ENTRIES:
            _store.popitem(last=False)
    return ref


def lookup(ref: str) -> str | None:
    with _lock:
        return _store.get(ref)


def compact(full_result: str) -> str:
    """Build the brain's view of a tool result and stash the full one.

    Never raises: a compaction failure must not break a working tool call, so
    anything unexpected falls back to a truncated copy of the original.
    """
    try:
        ref = stash(full_result)

        try:
            data = json.loads(full_result)
        except (json.JSONDecodeError, TypeError):
            return json.dumps({"ref": ref, "result": str(full_result)[:MAX_BRAIN_CHARS]})

        if not isinstance(data, dict):
            return json.dumps({"ref": ref, "result": str(data)[:MAX_BRAIN_CHARS]})

        view: dict = {}
        omitted: list[str] = []

        for key in ESSENTIAL_FIELDS:
            if data.get(key):
                view[key] = data[key]

        for key, value in data.items():
            if key in view or key == "ref":
                continue
            if key in BULKY_FIELDS:
                omitted.append(key)
                continue
            rendered = json.dumps(value, default=str)
            if len(rendered) > 200:          # an unexpectedly large field
                omitted.append(key)
                continue
            view[key] = value

        view["ref"] = ref
        if omitted:
            view["omitted"] = omitted

        out = json.dumps(view, default=str)
        if len(out) <= MAX_BRAIN_CHARS:
            return out

        # Still too big: fall back to the essentials alone, which is all the
        # brain needs to decide whether the question is fully answered.
        minimal = {k: data[k] for k in ESSENTIAL_FIELDS if data.get(k)}
        minimal["ref"] = ref
        minimal["omitted"] = omitted + [k for k in data if k not in minimal and k not in omitted]
        out = json.dumps(minimal, default=str)
        return out if len(out) <= MAX_BRAIN_CHARS else out[:MAX_BRAIN_CHARS]

    except Exception:
        return str(full_result)[:MAX_BRAIN_CHARS]


def resolve(brain_view: str) -> str:
    """Given what the brain saw, return the full result if we still hold it."""
    try:
        parsed = json.loads(brain_view)
    except (json.JSONDecodeError, TypeError):
        return brain_view
    if not isinstance(parsed, dict):
        return brain_view
    ref = parsed.get("ref")
    if not ref:
        return brain_view
    return lookup(ref) or brain_view
