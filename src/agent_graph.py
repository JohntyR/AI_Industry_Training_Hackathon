"""The Qwen ``agent-brain`` reasoning/tool-calling loop.

Per Challenge_Brief.md's Required Model Roles: this model plans, selects
``query_data``, and emits tool calls/arguments. It does not synthesize the
final answer -- see ``domain_model.py`` for that step.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

import httpx
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

import config
from tools import ALL_TOOLS

# Kept deliberately short. The brain is served with a 4,096-token context
# window, shared between this prompt, the tool schemas, the question, every tool
# result and the reply. Prose here is paid for on every single request, so the
# tool names and enums carry the routing and only score-critical rules are
# spelled out.
SYSTEM_PROMPT = """\
You plan tool calls for a financial-market agent over RBA cash rates
(2010-2026), ASX prices and AFR news (both 2015-2021).

Choose tools and arguments, read the results, call again only if a requested
fact is still missing. You never do arithmetic and never write the final
answer; a separate model does that from your results.

RULES
- Every number comes from a tool. Never estimate or recall one.
- Tabcorp (TAH.AX) stays excluded from ASX rankings, baskets and averages
  unless the question names it.
- For any rate/RBA article count use afr_count(preset="rba_rates"). Other terms
  are word-anchored for you, so pass the plain word.
- Sentiment questions need afr_find_article AND rba_rate_on_date.
- Needs ASX or AFR data after 2021? Call dataset_coverage and stop: the answer
  is that the evidence does not support it.
- On {"error": ...}, read the hint and retry once.
- ANY question about what prices did after dated events -- a rate cut, an
  article, "the one-week return after each effective date" -- is ONE
  asx_event_study call with the list of dates. It returns the rate in force,
  the window, the basket return and per-ticker returns together. Never answer
  that family with rba_rate_changes or a rate lookup alone.
- 3 tool calls maximum.

When you have every requested fact, stop and reply "done".
"""


def _build_brain_model() -> ChatOpenAI:
    return ChatOpenAI(
        openai_api_key=config.LITELLM_KEY or "sk-litellm",
        openai_api_base=config.LITELLM_BASE_URL,
        model_name=config.BRAIN_MODEL,
        temperature=0.0,
        timeout=config.BRAIN_TIMEOUT_S,
        max_retries=config.LLM_MAX_RETRIES,
        http_async_client=httpx.AsyncClient(timeout=config.BRAIN_TIMEOUT_S),
    )


graph = create_agent(
    _build_brain_model(),
    ALL_TOOLS,
    system_prompt=SYSTEM_PROMPT,
)


@dataclass
class BrainRun:
    """Outcome of the reasoning loop, including partial outcomes.

    ``messages`` holds the last state observed, so it is populated even when the
    loop timed out or raised. That is the whole point: evidence already gathered
    is worth per-component partial credit, and discarding it because a later
    step failed converts a partial score into a zero.
    """

    messages: list = field(default_factory=list)
    status: str = "complete"          # complete | timeout | error
    error: str = ""
    elapsed_s: float = 0.0

    @property
    def degraded(self) -> bool:
        return self.status != "complete"


async def run_brain_agent(question: str, deadline_s: float | None = None) -> BrainRun:
    """Run the reason -> act loop under a wall-clock deadline, salvaging partials.

    The loop is streamed rather than awaited as a single call. ``ainvoke``
    returns the final state or nothing at all, so an exception at step 5 throws
    away the tool results from steps 1-4. Streaming keeps the most recent state
    after every step, so a timeout or a crash still leaves usable evidence.
    """
    deadline_s = config.brain_budget_s() if deadline_s is None else deadline_s
    run = BrainRun()
    started = time.monotonic()

    try:
        async with asyncio.timeout(deadline_s):
            async for chunk in graph.astream(
                {"messages": [{"role": "user", "content": question}]},
                config={"recursion_limit": config.MAX_AGENT_STEPS},
                stream_mode="values",
            ):
                messages = chunk.get("messages") if isinstance(chunk, dict) else None
                if messages:
                    run.messages = messages
    except (asyncio.TimeoutError, TimeoutError):
        run.status = "timeout"
        run.error = f"brain loop exceeded {deadline_s:.0f}s"
    except Exception as exc:  # brain unreachable, recursion limit, tool crash
        run.status = "error"
        run.error = f"{type(exc).__name__}: {exc}"[:300]

    run.elapsed_s = time.monotonic() - started
    return run
