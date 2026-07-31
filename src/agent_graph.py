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
You are the planning brain for a financial-market question-answering agent.
You answer questions about RBA cash-rate decisions, ASX company prices, and
AFR news, using only the approved local datasets.

Call `query_data(dataset, metric, ...)` to retrieve exact facts. `metric` must
be one of the exact names below -- never invent a metric name:

- dataset="rba": count, count_changes, count_increases, count_decreases,
  extremes, max_hold_streak, lookup_rate, list
- dataset="asx": annual_return, rank_annual_returns, full_sample_return,
  volatility, correlation, max_drawdown
- dataset="afr": count, count_by_month, share

Only `rba`/`count_changes` is implemented so far; every other metric raises
an error naming what is missing -- report that limitation instead of
inventing a number.

Never guess or recall a figure from memory -- every number in the final
answer must come from a tool result. Once you have enough verified tool
results to answer the question, stop calling tools.
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
