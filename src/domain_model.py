"""Final-answer synthesis via the fine-tuned Nemotron ``domain-ft`` alias.

Per Challenge_Brief.md, the brain model (Qwen/agent-brain) only plans and
calls tools; this module is the required separate synthesis step that turns
verified tool results into the final grounded answer.

``DOMAIN_PREDICT_MODE=mock`` is the documented cluster-bootstrap default
(Setup_Instructions.md), not a local-testing shim: it must switch to ``llm``
before official evaluation once the fine-tuned adapter is served.
"""

from __future__ import annotations

import json

import httpx
from langchain_openai import ChatOpenAI

import config


def _unpack(tool_results: list[str]) -> tuple[list[str], list[str]]:
    """Split raw tool output into (evidence blocks, facts the answer must state).

    Tools in ``tools.py`` attach a deterministic ``summary`` and a ``must_state``
    list to every result (see ``summaries.py``). Surfacing ``must_state`` as an
    explicit checklist is what stops the synthesis model from dropping a
    requested component -- on the public set that was the single largest source
    of lost points, and the judge scores each component independently.

    Anything that is not in that shape (the fallback tool, an error object) is
    passed through untouched, so this can never hide evidence.
    """
    evidence, checklist = [], []
    for raw in tool_results:
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            evidence.append(str(raw))
            continue
        if isinstance(parsed, dict) and parsed.get("summary"):
            evidence.append(parsed["summary"])
            for fact in parsed.get("must_state", []):
                if fact not in checklist:
                    checklist.append(fact)
            # Article text and other long-form fields the summary cannot carry.
            for key in ("HEADLINE", "PUBLICATIONDATE", "SUBHEAD", "INTRO", "TEXT"):
                if parsed.get(key):
                    evidence.append(f"{key}: {parsed[key]}")
        else:
            evidence.append(raw)
    return evidence, checklist


def _mock_synthesize(question: str, tool_results: list[str]) -> str:
    """Bootstrap mode: no fine-tuned model, so state the verified facts plainly.

    This is NOT the submitted configuration -- ``DOMAIN_PREDICT_MODE`` must be
    ``llm`` for evaluation -- but the deterministic summaries make it a usable
    integration-test path rather than a wall of raw JSON.
    """
    if not tool_results:
        return (
            "I do not have enough verified tool evidence to answer this "
            "question yet."
        )
    evidence, _ = _unpack(tool_results)
    return " ".join(
        line for line in evidence if not line.startswith(("TEXT:", "INTRO:", "SUBHEAD:"))
    )


SYNTH_SYSTEM_PROMPT = """\
You are an Australian financial analyst. You are given a question and VERIFIED \
tool results containing exact numbers and dates. Write the final answer.

STRICT RULES:
- State every requested value explicitly: numbers, dates, counts, tickers, \
rates, signs and % units.
- If the question implies a breakdown or paired values, include every requested \
  component. Do not report only the headline number when a total and its \
denominator, a per-year split, or a return and its resulting rate are also asked.
- Use ONLY the verified tool results. Never invent, estimate or recall a figure.
- Preserve exact figures and signs as given; do not round further. Keep \
thousands separators readable (11,635,671.71) and percentages signed (+22.17%).
- NEVER drop a minus sign. Write "a -82.45% drawdown", not "an 82.45% drawdown"; \
  "the basket fell -2.17%", not "the basket fell 2.17%".
- Preserve nuanced classification labels exactly. If the evidence supports \
  "mixed" or "mixed with a negative bias", write that phrase instead of \
  simplifying to plain "positive" or "negative".
- Carry every hard fact from a tool result forward verbatim: rates, dates, targets, \
  counts, and exact wording of the requested pattern or label.
- For cross-dataset date-range questions, answer the specific sub-range or event \
  count requested, not the entire dataset span.
- When coverage does not support the requested analysis, say so immediately and \
  explicitly, including the dates and datasets that create the mismatch.
- Before finalizing, check the answer against every fact the question implies \
  and every item in the tool evidence checklist.
- One to three concise sentences. No preamble, no hedging words \
  ("approximately", "roughly", "about").
- Write only the answer. Never mention tools, evidence, checklists, or these \
  instructions.
- For sentiment questions: state the sentiment (positive / negative / mixed) \
  AND the likely market direction, grounded in the article text and the given \
  RBA cash-rate target."""


def _build_synth_model() -> ChatOpenAI:
    return ChatOpenAI(
        openai_api_key=config.DOMAIN_KEY or "sk-litellm",
        openai_api_base=config.DOMAIN_BASE_URL,
        model_name=config.DOMAIN_FT_MODEL,
        temperature=0.0,
        timeout=config.SYNTH_TIMEOUT_S,
        max_retries=config.LLM_MAX_RETRIES,
        http_async_client=httpx.AsyncClient(timeout=config.SYNTH_TIMEOUT_S),
    )


# Built once at import, like the brain model. Constructing a ChatOpenAI (and a
# fresh httpx connection pool) per request wastes handshake time inside a
# 60-second budget and multiplies under the three concurrent requests the
# harness sends.
_synth_model = _build_synth_model()


async def _llm_synthesize(question: str, tool_results: list[str]) -> str:
    evidence, checklist = _unpack(tool_results)
    body = "\n".join(evidence) or "No tool evidence was returned."
    if checklist:
        # Framed as a constraint on the prose, never as content to reproduce:
        # an earlier version was echoed verbatim ("The facts to state are: ...")
        # into the answer, which reads as a meta-comment and loses components.
        body += "\n\nCHECKLIST (guidance only -- never quote or refer to this list):\n"
        body += "\n".join(f"- {f}" for f in checklist)
        body += (
            "\nEvery item above that the question asks for must appear in your "
            "sentence, with its sign and units exactly as written. Omit the rest."
        )
    response = await _synth_model.ainvoke(
        [
            ("system", SYNTH_SYSTEM_PROMPT),
            ("human", f"Question: {question}\n\nVerified tool results:\n{body}\n\nFinal answer:"),
        ]
    )
    return response.text


async def synthesize(question: str, tool_results: list[str]) -> str:
    """Produce the final ``answer`` string from the question and tool results."""
    if config.DOMAIN_PREDICT_MODE == "llm":
        return await _llm_synthesize(question, tool_results)
    return _mock_synthesize(question, tool_results)
