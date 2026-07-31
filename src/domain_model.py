"""Final-answer synthesis via the fine-tuned Nemotron ``domain-ft`` alias.

Per Challenge_Brief.md, the brain model (Qwen/agent-brain) only plans and
calls tools; this module is the required separate synthesis step that turns
verified tool results into the final grounded answer.

``DOMAIN_PREDICT_MODE=mock`` is the documented cluster-bootstrap default
(Setup_Instructions.md), not a local-testing shim: it must switch to ``llm``
before official evaluation once the fine-tuned adapter is served.
"""

from __future__ import annotations

import httpx
from langchain_openai import ChatOpenAI

import config


def _mock_synthesize(question: str, tool_results: list[str]) -> str:
    if not tool_results:
        return (
            "I do not have enough verified tool evidence to answer this "
            "question yet."
        )
    return " ".join(tool_results)


SYNTH_SYSTEM_PROMPT = """\
You are an Australian financial analyst. You are given a question and VERIFIED \
tool results containing exact numbers and dates. Write the final answer.

STRICT RULES:
- State every requested value explicitly: numbers, dates, counts, tickers, \
rates, signs and % units.
- Use ONLY the verified tool results. Never invent, estimate or recall a figure.
- Preserve exact figures and signs as given; do not round further. Keep \
thousands separators readable (11,635,671.71) and percentages signed (+22.17%).
- One to three concise sentences. No preamble, no hedging words \
("approximately", "roughly", "about").
- If the results show the data cannot support the question, say so plainly and \
explain the coverage gap.
- For sentiment questions: state the sentiment (positive / negative / mixed) \
AND the likely market direction, grounded in the article text and the given \
RBA cash-rate target."""


async def _llm_synthesize(question: str, tool_results: list[str]) -> str:
    model = ChatOpenAI(
        openai_api_key=config.DOMAIN_KEY or "sk-litellm",
        openai_api_base=config.DOMAIN_BASE_URL,
        model_name=config.DOMAIN_FT_MODEL,
        temperature=0.0,
        http_async_client=httpx.AsyncClient(),
    )
    evidence = "\n".join(tool_results) or "No tool evidence was returned."
    response = await model.ainvoke(
        [
            ("system", SYNTH_SYSTEM_PROMPT),
            ("human", f"Question: {question}\n\nVerified tool results:\n{evidence}\n\nFinal answer:"),
        ]
    )
    return response.text


async def synthesize(question: str, tool_results: list[str]) -> str:
    """Produce the final ``answer`` string from the question and tool results."""
    if config.DOMAIN_PREDICT_MODE == "llm":
        return await _llm_synthesize(question, tool_results)
    return _mock_synthesize(question, tool_results)
