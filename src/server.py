"""Agent API contract required by submission-guide.md.

``GET /health`` -> 200 or the team is skipped entirely.
``POST /query`` -> {"question": str} -> {"answer", "steps", "tool_trace"}.
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager

from langchain_core.messages import AIMessage, ToolMessage
from fastapi import FastAPI
from pydantic import BaseModel

import config
import query_data
from agent_graph import run_brain_agent
from domain_model import synthesize

logger = logging.getLogger("agent")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the datasets, then announce the synthesis mode loudly.

    The AFR corpus is 219k records and takes seconds to parse. Loading it here
    rather than on first use means no graded question pays that cost, and the
    three concurrent requests the harness sends cannot each trigger a cold load
    at once. ``GET /health`` only starts answering once this returns, so the
    harness never sees a warm-up as availability.

    Shipping in bootstrap ``mock`` mode is silent and costly, so the one place
    it cannot be missed is the log line printed before the first request.
    """
    loaded = await asyncio.to_thread(query_data.warmup)
    logger.info(
        "datasets loaded: %d RBA records, %d ASX tickers, %d AFR articles",
        loaded["rba_rows"], loaded["asx_tickers"], loaded["afr_records"],
    )

    problems = config.evaluation_readiness()
    if problems:
        logger.warning("=" * 72)
        logger.warning("AGENT IS NOT EVALUATION-READY:")
        for problem in problems:
            logger.warning("  - %s", problem)
        logger.warning("Run `python scripts/preflight.py` for the full check.")
        logger.warning("=" * 72)
    else:
        logger.info(
            "Evaluation-ready: brain=%s synthesis=%s (mode=llm)",
            config.BRAIN_MODEL, config.DOMAIN_FT_MODEL,
        )
    yield


app = FastAPI(title="Cognitivo Hackathon Agent", lifespan=lifespan)


class QueryRequest(BaseModel):
    question: str


class ToolTraceEntry(BaseModel):
    tool: str
    args: dict
    result: str


class QueryResponse(BaseModel):
    answer: str
    steps: int
    tool_trace: list[ToolTraceEntry]


@app.get("/health")
async def health() -> dict:
    """Always HTTP 200.

    This is a hard gate: a non-200 here means the team is skipped and scores
    zero on the hidden questions. It therefore never depends on a model server
    being reachable. The extra fields are diagnostics only.
    """
    return {
        "status": "ok",
        "synthesis_mode": config.DOMAIN_PREDICT_MODE,
        "evaluation_ready": not config.evaluation_readiness(),
    }


def _extract_trace(messages: list) -> tuple[int, list[ToolTraceEntry]]:
    """Turn the raw LangGraph message list into (steps, tool_trace).

    ``steps`` counts each brain reasoning turn (one per AIMessage). Each tool
    call is paired with its corresponding ToolMessage result by call id.
    """
    steps = 0
    results_by_call_id: dict[str, str] = {}
    for message in messages:
        if isinstance(message, ToolMessage):
            results_by_call_id[message.tool_call_id] = str(message.content)

    tool_trace: list[ToolTraceEntry] = []
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        steps += 1
        for call in message.tool_calls or []:
            tool_trace.append(
                ToolTraceEntry(
                    tool=call["name"],
                    args=call["args"],
                    result=results_by_call_id.get(call["id"], ""),
                )
            )
    return steps, tool_trace


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Always returns a valid, non-empty ``answer``, inside a wall-clock budget.

    Two things score zero and both are avoidable. An uncaught exception returns
    HTTP 500 with no ``answer`` field; exceeding 300 seconds times out. Between
    them sits a third, subtler loss: a request that runs long enough to take the
    20% slow-penalty, or that discards evidence it already had.

    So the request is run against a budget. The brain loop is streamed under a
    deadline and its partial state is kept whatever happens; a reserve is held
    back so the fine-tuned model always gets to write an answer from whatever
    evidence exists. Every degradation is recorded in ``tool_trace`` with an
    underscore-prefixed pseudo-tool, which keeps it out of the evidence passed
    to synthesis while leaving it visible in organizer diagnostics.
    """
    question = (request.question or "").strip()
    if not question:
        return QueryResponse(answer="No question was provided.", steps=0, tool_trace=[])

    started = time.monotonic()

    run = await run_brain_agent(question, deadline_s=config.brain_budget_s())
    steps, tool_trace = _extract_trace(run.messages)

    tool_results = [
        entry.result for entry in tool_trace if entry.result and not entry.tool.startswith("_")
    ]
    if run.degraded:
        tool_trace.append(ToolTraceEntry(
            tool=f"_brain_{run.status}",
            args={"elapsed_s": round(run.elapsed_s, 1)},
            result=f"{run.error}; synthesizing from {len(tool_results)} tool result(s) already collected",
        ))

    # Whatever the brain loop consumed, synthesis still gets a workable slice:
    # a slow real answer scores far better than a fast unusable one.
    remaining = config.REQUEST_DEADLINE_S - (time.monotonic() - started)
    synth_budget = max(config.SYNTH_MIN_S, min(config.SYNTH_TIMEOUT_S, remaining))

    answer = ""
    try:
        async with asyncio.timeout(synth_budget):
            answer = await synthesize(question, tool_results)
    except (asyncio.TimeoutError, TimeoutError):
        tool_trace.append(ToolTraceEntry(
            tool="_synth_timeout", args={"budget_s": round(synth_budget, 1)},
            result=f"synthesis exceeded {synth_budget:.0f}s; falling back to raw tool evidence",
        ))
    except Exception as exc:
        tool_trace.append(ToolTraceEntry(
            tool="_synth_error", args={}, result=str(exc)[:300],
        ))

    if not answer or not answer.strip():
        answer = " ".join(tool_results).strip()
    if not answer:
        answer = (
            "Based on the supplied datasets, a definitive answer could not be "
            "produced for this question."
        )

    elapsed = time.monotonic() - started
    logger.info(
        "query done in %.1fs (brain %.1fs, %s) steps=%d tools=%d%s",
        elapsed, run.elapsed_s, run.status, steps, len(tool_results),
        "  [OVER 60s SLOW PENALTY]" if elapsed > 60 else "",
    )

    return QueryResponse(answer=answer.strip(), steps=steps, tool_trace=tool_trace)
