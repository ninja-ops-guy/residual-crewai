"""Integration tests against CrewAI's public API (crewai 1.15.22).

Uses a FakeLLM subclassing the native OpenAI provider with call() stubbed —
real Crew/Agent/Task/kickoff machinery, no network, no API keys.
"""

import pytest
from crewai import Agent, Crew, Task
from crewai.llms.providers.openai.completion import OpenAICompletion

from residual_crewai import ResidualCrewAIListener
from residual_sdk import GateSet, SQLiteResidualBackend, verify_attestation


class FakeLLM(OpenAICompletion):
    def call(self, messages, tools=None, callbacks=None, available_functions=None, **kw):
        return "Final Answer: 42"


def make_crew(roles=("researcher",)):
    agents = [
        Agent(role=r, goal=f"do {r} work", backstory="test", llm=FakeLLM(model="gpt-4o", provider="openai", api_key="sk-fake"))
        for r in roles
    ]
    tasks = [
        Task(description=f"task for {r}", expected_output="text", agent=a)
        for r, a in zip(roles, agents)
    ]
    return Crew(agents=agents, tasks=tasks)


def make_listener(blocked=()):
    backend = SQLiteResidualBackend(
        ":memory:", gate_set=GateSet(blocked_modules=set(blocked))
    )
    return ResidualCrewAIListener(backend=backend), backend


def test_crew_kickoff_attested():
    listener, backend = make_listener()
    out = make_crew().kickoff()
    listener.wait()
    assert "42" in out.raw
    assert not listener.failures, listener.failures
    tok = listener.attestation_for()
    assert verify_attestation(tok).ok
    kinds = [e["kind"] for e in backend.events()]
    assert kinds[0] == "run.started" and "attestation.issued" in kinds
    modules = [e["payload"]["module"] for e in backend.events() if e["kind"] == "module.called"]
    assert any(m.startswith("task:") for m in modules) or any(m.startswith("agent:") for m in modules), modules


def test_role_based_agents_recorded():
    listener, backend = make_listener()
    make_crew(("researcher", "writer")).kickoff()
    listener.wait()
    modules = [e["payload"]["module"] for e in backend.events() if e["kind"] == "module.called"]
    agents = {m for m in modules if m.startswith("agent:")}
    assert "agent:researcher" in agents, modules


def test_gate_fire_on_blocked_task():
    listener, backend = make_listener(blocked={"task:task for researcher"})
    make_crew().kickoff()
    listener.wait()
    # CrewAI swallows handler exceptions; the failure record is the channel.
    assert listener.failures, "gate fire must be recorded in listener.failures"
    fired = [e for e in backend.events() if e["kind"] == "gate.fired"]
    assert fired
    tok = listener.attestation_for()
    assert any(v != "PASS" for v in tok["verdicts"].values())


def test_tool_calling_and_delegation():
    from crewai.tools import tool as crew_tool

    @crew_tool("calculator")
    def calc(x: str) -> str:
        """Calculate."""
        return "3"

    class ToolLLM(OpenAICompletion):
        def supports_function_calling(self):
            return False  # force ReAct loop so tool usage events fire

        def call(self, messages, tools=None, callbacks=None, available_functions=None, **kw):
            n = getattr(self, "_calls", 0)
            object.__setattr__(self, "_calls", n + 1)
            if n == 0:
                return 'Thought: use the tool\nAction: calculator\nAction Input: {"x": "1+2"}'
            return "Final Answer: 3"

    llm = ToolLLM(model="gpt-4o", provider="openai", api_key="sk-fake")
    agent = Agent(role="calc", goal="compute", backstory="t", llm=llm, tools=[calc])
    task = Task(description="compute 1+2", expected_output="number", agent=agent)
    listener, backend = make_listener()
    Crew(agents=[agent], tasks=[task]).kickoff()
    listener.wait()
    modules = [e["payload"]["module"] for e in backend.events() if e["kind"] == "module.called"]
    assert any(m.startswith("tool:calculator") for m in modules), modules


def test_parallel_crews_independent_runs():
    listener, backend = make_listener()
    make_crew(("a",)).kickoff()
    make_crew(("b",)).kickoff()
    listener.wait()
    starts = [e for e in backend.events() if e["kind"] == "run.started"]
    assert len({e["run_id"] for e in starts}) == 2
    atts = [e for e in backend.events() if e["kind"] == "attestation.issued"]
    assert len(atts) == 2
