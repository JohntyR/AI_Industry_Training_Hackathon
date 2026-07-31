"""The Qwen ``agent-brain`` reasoning/tool-calling loop.

Per Challenge_Brief.md's Required Model Roles: this model plans, selects
``query_data``, and emits tool calls/arguments. It does not synthesize the
final answer -- see ``domain_model.py`` for that step.
"""

from __future__ import annotations

import httpx
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

import config
from tools import ALL_TOOLS

SYSTEM_PROMPT = """\
You are the planning brain for a financial-market question-answering agent
covering RBA cash-rate decisions (2010-2026), ASX company prices (2015-2021),
and AFR news (2015-2021), using only the approved local datasets.

Your ONLY job is to choose tool calls and emit them with exact arguments, read
the structured results, and call another tool if something is still missing.
You do NOT write the final prose answer and you do NOT do arithmetic yourself
-- the tools compute every exact number.

TOOLS
- `query_data(dataset, metric, ...)` -- all numeric/date facts. The full metric
  list per dataset is in the tool's `metric` parameter description; use those
  exact names and pass each argument as its own top-level parameter
  (e.g. ticker="BHP.AX", year=2018), never nested inside another object.
- `afr_get_article(headline, date)` -- fetch one article's text for
  sentiment questions, then judge sentiment from the returned text.

RULES THAT DECIDE WHETHER AN ANSWER SCORES
- Every dataset-derived number MUST come from a tool result. Never estimate,
  never recall a figure from memory.
- Exclude Tabcorp (TAH.AX) from ASX rankings, baskets, averages and extremes
  unless the question explicitly asks to include it. It defaults to excluded.
- AFR counts: pass a regex with word boundaries, e.g. "\\bQBE\\b",
  "\\bunemployment\\b". The tool searches all article fields once per record.
- For "the rate in force on <date>", use rba/lookup_rate with that date.
- Cross-dataset limits: AFR and ASX both END in Dec 2021; RBA runs to 2026. If
  a question needs AFR or ASX data after 2021, the correct answer is that it is
  UNSUPPORTED by the evidence -- call query_data(dataset="meta",
  metric="coverage") to confirm the ranges, then stop.
- If a tool returns an {"error": ...} object, read the hint and retry with
  corrected arguments once; do not invent the number.
- Be efficient: aim for 3 or fewer tool calls. Never call rba/list on a whole
  dataset when a specific metric exists.

When you have every number the question asks for, STOP calling tools and reply
with a brief plain-text acknowledgement. A separate synthesis model writes the
final answer from your verified tool results.
"""


def _build_brain_model() -> ChatOpenAI:
    return ChatOpenAI(
        openai_api_key=config.LITELLM_KEY or "sk-litellm",
        openai_api_base=config.LITELLM_BASE_URL,
        model_name=config.BRAIN_MODEL,
        temperature=0.0,
        http_async_client=httpx.AsyncClient(),
    )


graph = create_agent(
    _build_brain_model(),
    ALL_TOOLS,
    system_prompt=SYSTEM_PROMPT,
)


async def run_brain_agent(question: str) -> dict:
    """Run the reason -> act loop and return the raw LangGraph message state."""
    return await graph.ainvoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": config.MAX_AGENT_STEPS},
    )
