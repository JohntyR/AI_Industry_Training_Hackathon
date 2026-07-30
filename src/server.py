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
    state = await run_brain_agent(request.question)
    messages = state["messages"]
    steps, tool_trace = _extract_trace(messages)

    tool_results = [entry.result for entry in tool_trace if entry.result]
    answer = await synthesize(request.question, tool_results)

    return QueryResponse(answer=answer, steps=steps, tool_trace=tool_trace)
