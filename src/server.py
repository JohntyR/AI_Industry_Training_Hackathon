"""Agent API contract required by submission-guide.md.

``GET /health`` -> 200 or the team is skipped entirely.
``POST /query`` -> {"question": str} -> {"answer", "steps", "tool_trace"}.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, ToolMessage
from fastapi import FastAPI
from pydantic import BaseModel

from agent_graph import run_brain_agent
from domain_model import synthesize

app = FastAPI(title="Cognitivo Hackathon Agent")


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
    return {"status": "ok"}


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
    """Always returns a valid, non-empty ``answer``.

    An uncaught exception here would be a 500 with no ``answer`` field, which
    the harness scores as zero. A degraded-but-valid answer can still earn
    partial credit, so every failure path is contained: if the brain loop dies
    we still synthesize from whatever tool results we did collect, and if that
    fails too we return a plain statement of the limitation.
    """
    question = (request.question or "").strip()
    if not question:
        return QueryResponse(answer="No question was provided.", steps=0, tool_trace=[])

    steps, tool_trace = 0, []
    try:
        state = await run_brain_agent(question)
        steps, tool_trace = _extract_trace(state["messages"])
    except Exception as exc:  # brain unreachable, recursion limit, tool crash
        tool_trace.append(
            ToolTraceEntry(tool="_brain_error", args={}, result=str(exc)[:300])
        )

    tool_results = [
        entry.result for entry in tool_trace if entry.result and not entry.tool.startswith("_")
    ]
    try:
        answer = await synthesize(question, tool_results)
    except Exception as exc:
        tool_trace.append(
            ToolTraceEntry(tool="_synth_error", args={}, result=str(exc)[:300])
        )
        answer = " ".join(tool_results) if tool_results else ""

    if not answer or not answer.strip():
        answer = (
            "Based on the supplied datasets, a definitive answer could not be "
            "produced for this question."
        )

    return QueryResponse(answer=answer.strip(), steps=steps, tool_trace=tool_trace)
