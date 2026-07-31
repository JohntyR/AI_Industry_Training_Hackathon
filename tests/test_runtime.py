"""Runtime tests: response-time budget, partial-evidence salvage, concurrency.

These cover the paths that never execute during a healthy dev loop and are
therefore the ones most likely to be broken when they finally matter -- a brain
loop that overruns the budget, one that crashes halfway, a synthesis call that
hangs, and three requests in flight at once.

No model server is required: the brain graph and the synthesis call are replaced
with controllable fakes, so the orchestration in server.py is what is under
test.

Run:  python tests/test_runtime.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

import agent_graph  # noqa: E402
import config       # noqa: E402
import server       # noqa: E402

P = F = 0


def check(label, got, expected):
    global P, F
    ok = got == expected
    P, F = (P + 1, F) if ok else (P, F + 1)
    print(f"{'PASS' if ok else 'FAIL'} | {label}")
    if not ok:
        print(f"      got:      {got!r}")
        print(f"      expected: {expected!r}")


def check_true(label, got):
    check(label, bool(got), True)


class _FakeGraph:
    """Stands in for the compiled LangGraph agent."""

    def __init__(self, steps=3, delay=0.0, raise_at=None, hang_at=None):
        self.steps, self.delay = steps, delay
        self.raise_at, self.hang_at = raise_at, hang_at

    async def astream(self, _inputs, config=None, stream_mode=None):
        messages = []
        for i in range(self.steps):
            if self.raise_at == i:
                raise RuntimeError("brain exploded")
            if self.hang_at == i:
                await asyncio.sleep(3600)
            messages = messages + [
                AIMessage(content="", tool_calls=[{
                    "name": "query_data",
                    "args": {"dataset": "rba", "metric": "count_changes"},
                    "id": f"call-{i}",
                }]),
                ToolMessage(content=f'{{"step": {i}}}', tool_call_id=f"call-{i}"),
            ]
            await asyncio.sleep(self.delay)
            yield {"messages": messages}


def _run(coro):
    return asyncio.run(coro)


def _tools_in(trace):
    return [entry.tool for entry in trace]


# ---------------------------------------------------------------------------
print("=" * 70)
print("Brain loop completes normally")
agent_graph.graph = _FakeGraph(steps=3)
run = _run(agent_graph.run_brain_agent("q", deadline_s=5.0))
check("  status=complete", run.status, "complete")
check("  degraded=False", run.degraded, False)
check("  3 reasoning steps captured", len([m for m in run.messages if isinstance(m, AIMessage)]), 3)

# ---------------------------------------------------------------------------
print("=" * 70)
print("Brain loop times out -> evidence from completed steps survives")
agent_graph.graph = _FakeGraph(steps=5, delay=0.05, hang_at=3)
run = _run(agent_graph.run_brain_agent("q", deadline_s=0.5))
check("  status=timeout", run.status, "timeout")
check("  3 completed steps salvaged", len([m for m in run.messages if isinstance(m, AIMessage)]), 3)
check_true("  tool results salvaged", [m for m in run.messages if isinstance(m, ToolMessage)])

# ---------------------------------------------------------------------------
print("=" * 70)
print("Brain loop raises -> evidence from completed steps survives")
agent_graph.graph = _FakeGraph(steps=5, raise_at=2)
run = _run(agent_graph.run_brain_agent("q", deadline_s=5.0))
check("  status=error", run.status, "error")
check("  error is reported", run.error.startswith("RuntimeError"), True)
check("  2 completed steps salvaged", len([m for m in run.messages if isinstance(m, AIMessage)]), 2)

# ---------------------------------------------------------------------------
print("=" * 70)
print("Endpoint salvages a timed-out brain loop into a real answer")
config.REQUEST_DEADLINE_S, config.SYNTH_RESERVE_S = 2.0, 1.0
config.SYNTH_MIN_S, config.SYNTH_TIMEOUT_S = 1.0, 1.0
agent_graph.graph = _FakeGraph(steps=5, delay=0.05, hang_at=3)


async def _fake_synth(question, tool_results):
    return f"answer from {len(tool_results)} result(s)"


server.synthesize = _fake_synth
response = _run(server.query(server.QueryRequest(question="Q1")))
check("  answer built from partial evidence", response.answer, "answer from 3 result(s)")
check("  degradation recorded in tool_trace", "_brain_timeout" in _tools_in(response.tool_trace), True)
check("  real tool calls still traced", _tools_in(response.tool_trace).count("query_data"), 3)

# ---------------------------------------------------------------------------
print("=" * 70)
print("Synthesis hangs -> falls back to raw evidence, never an empty answer")
agent_graph.graph = _FakeGraph(steps=2)


async def _hanging_synth(question, tool_results):
    await asyncio.sleep(3600)


server.synthesize = _hanging_synth
started = time.monotonic()
response = _run(server.query(server.QueryRequest(question="Q2")))
elapsed = time.monotonic() - started
check("  synthesis timeout recorded", "_synth_timeout" in _tools_in(response.tool_trace), True)
check("  answer is non-empty", bool(response.answer.strip()), True)
check("  answer contains the tool evidence", '"step": 0' in response.answer, True)
check("  bounded by the synthesis budget", elapsed < 3.0, True)

# ---------------------------------------------------------------------------
print("=" * 70)
print("Total-loss path still returns a valid answer")
agent_graph.graph = _FakeGraph(steps=1, raise_at=0)
server.synthesize = _hanging_synth
response = _run(server.query(server.QueryRequest(question="Q3")))
check("  brain error recorded", "_brain_error" in _tools_in(response.tool_trace), True)
check("  answer is non-empty", bool(response.answer.strip()), True)
check("  steps is a valid integer", isinstance(response.steps, int) and response.steps >= 0, True)

# ---------------------------------------------------------------------------
print("=" * 70)
print("Three concurrent requests do not mix state")
config.REQUEST_DEADLINE_S, config.SYNTH_RESERVE_S = 10.0, 3.0
config.SYNTH_MIN_S, config.SYNTH_TIMEOUT_S = 3.0, 3.0
agent_graph.graph = _FakeGraph(steps=2, delay=0.05)


async def _echo_synth(question, tool_results):
    await asyncio.sleep(0.05)                     # force interleaving
    return f"answer for {question}"


server.synthesize = _echo_synth


async def _concurrent():
    return await asyncio.gather(*[
        server.query(server.QueryRequest(question=f"Question-{i}")) for i in range(3)
    ])


responses = _run(_concurrent())
for i, response in enumerate(responses):
    check(f"  response {i} matches its own question", response.answer, f"answer for Question-{i}")
    check(f"  response {i} has its own trace", len(response.tool_trace), 2)

# ---------------------------------------------------------------------------
print("=" * 70)
print("Budget arithmetic")
config.REQUEST_DEADLINE_S, config.SYNTH_RESERVE_S = 55.0, 15.0
check("  brain budget = deadline - reserve", config.brain_budget_s(), 40.0)
config.REQUEST_DEADLINE_S, config.SYNTH_RESERVE_S = 10.0, 20.0
check("  brain budget never negative", config.brain_budget_s(), 5.0)

print("=" * 70)
print(f"RESULT: {P} passed, {F} failed")
sys.exit(1 if F else 0)
