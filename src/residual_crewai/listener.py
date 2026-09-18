"""ResidualCrewAIListener — CrewAI event-bus adapter for RESIDUAL.

Lifecycle mapping (CrewAI 1.x events -> RESIDUAL):
  * CrewKickoffStartedEvent      -> on_run_start (one residual run per crew kickoff)
  * TaskStartedEvent             -> on_module_call("task:<name>")
  * AgentExecutionStartedEvent   -> on_module_call("agent:<role>")  (role-based agents)
  * ToolUsageStartedEvent        -> on_module_call("tool:<tool_name>")
    (delegation between agents surfaces as tool usage in CrewAI; it is
    attested at that granularity)
  * CrewKickoffCompletedEvent    -> on_run_complete("success")
  * CrewKickoffFailedEvent       -> on_run_complete("error")

Parallel tasks / concurrent kickoffs: each kickoff gets an independent
residual run keyed by UUID, so overlapping crews never share a run.

Fail-closed: gate verdicts (FAIL/UNKNOWN/BLOCKED) raise GateFiredError from
the event handler; CrewAI's bus logs handler errors instead of raising, so
the listener records every failure in ``self.failures`` — callers MUST drain
it (or inspect attestation verdicts) after kickoff + ``wait()``.
"""

from __future__ import annotations

import uuid
from typing import Any

from crewai.events.base_event_listener import BaseEventListener
from crewai.events.types.agent_events import (
    AgentExecutionCompletedEvent,
    AgentExecutionErrorEvent,
    AgentExecutionStartedEvent,
)
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
)
from crewai.events.types.task_events import (
    TaskCompletedEvent,
    TaskFailedEvent,
    TaskStartedEvent,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageStartedEvent,
)

from residual_sdk import GateFiredError, ResidualBackend, SQLiteResidualBackend
from residual_sdk.attestation import canonical_json, sha256_hex


class ResidualCrewAIListener(BaseEventListener):
    """Drop-in CrewAI event listener attesting crew runs to RESIDUAL."""

    def __init__(self, backend: ResidualBackend | None = None,
                 spec_id: str = "crewai-crew@1.0.0"):
        super().__init__()
        self.backend = backend or SQLiteResidualBackend(":memory:")
        self.spec_id = spec_id
        self._active_run: str | None = None
        self._attestations: dict[str, dict] = {}
        self.failures: list[BaseException] = []

    # -- helpers -------------------------------------------------------------

    def _module(self, module: str, payload: Any) -> None:
        if self._active_run is None:
            return
        verdict = self.backend.on_module_call(
            self._active_run, module, str(uuid.uuid4()),
            sha256_hex(canonical_json(_jsonable(payload))),
        )
        if verdict != "PASS":
            err = GateFiredError("G0-BLOCKED-MODULE", verdict, module)
            try:
                self.backend.on_run_complete(self._active_run, "aborted")
            finally:
                self._capture()
            raise err  # recorded by _guard; bus logs, failures[] is the record

    def _capture(self) -> None:
        if self._active_run is None:
            return
        try:
            self._attestations[self._active_run] = self.backend.get_attestation(
                self._active_run
            )
        except Exception as exc:  # fail-closed: absence is evidence
            self.failures.append(exc)

    def _guard(self, fn):
        """Record handler exceptions in self.failures before re-raising.

        CrewAI's event bus dispatches handlers on futures and logs (does not
        raise) handler errors, so fail-closed surfacing requires this record;
        callers and the conformance handle drain ``failures`` after flush().
        """

        def wrapped(*args: Any) -> None:
            try:
                fn(*args)
            except BaseException as exc:
                self.failures.append(exc)
                raise

        return wrapped

    def _kickoff_start(self, *args: Any) -> None:
        run_id = str(uuid.uuid4())
        self._active_run = run_id
        self.backend.on_run_start(
            run_id, self.spec_id,
            {"framework": "crewai",
             "crew": getattr(args[-1], "crew_name", None) or "crew"},
        )

    def _kickoff_done(self, outcome: str) -> None:
        if self._active_run is not None:
            self.backend.on_run_complete(self._active_run, outcome)
            self._capture()

    # -- listener registration -------------------------------------------------

    def setup_listeners(self, crewai_event_bus) -> None:  # noqa: D102
        @crewai_event_bus.on(CrewKickoffStartedEvent)
        def _on_kickoff_start(*args: Any) -> None:
            self._guard(self._kickoff_start)(*args)

        @crewai_event_bus.on(CrewKickoffCompletedEvent)
        def _on_kickoff_done(*args: Any) -> None:
            self._guard(self._kickoff_done)("success")

        @crewai_event_bus.on(CrewKickoffFailedEvent)
        def _on_kickoff_failed(*args: Any) -> None:
            self._guard(self._kickoff_done)("error")

        @crewai_event_bus.on(TaskStartedEvent)
        def _on_task_start(*args: Any) -> None:
            def body(*a):
                e = a[-1]
                name = getattr(e, "task_name", None) or str(getattr(e, "task_id", "task"))
                self._module(f"task:{name}", {"task_id": str(getattr(e, "task_id", ""))})
            self._guard(body)(*args)

        @crewai_event_bus.on(TaskCompletedEvent)
        def _on_task_done(*args: Any) -> None:
            pass  # completion is crew-level; tasks attest at start granularity

        @crewai_event_bus.on(TaskFailedEvent)
        def _on_task_failed(*args: Any) -> None:
            self._guard(self._kickoff_done)("error")

        @crewai_event_bus.on(AgentExecutionStartedEvent)
        def _on_agent_start(*args: Any) -> None:
            def body(*a):
                e = a[-1]
                role = getattr(e, "agent_role", None) or getattr(getattr(e, "agent", None), "role", "agent")
                self._module(f"agent:{role}", {"agent_id": str(getattr(e, "agent_id", ""))})
            self._guard(body)(*args)

        @crewai_event_bus.on(AgentExecutionCompletedEvent)
        def _on_agent_done(*args: Any) -> None:
            pass

        @crewai_event_bus.on(AgentExecutionErrorEvent)
        def _on_agent_error(*args: Any) -> None:
            self._guard(self._kickoff_done)("error")

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def _on_tool_start(*args: Any) -> None:
            def body(*a):
                e = a[-1]
                name = getattr(e, "tool_name", None) or "unknown"
                self._module(f"tool:{name}", {"tool_class": str(getattr(e, "tool_class", ""))})
            self._guard(body)(*args)

        @crewai_event_bus.on(ToolUsageErrorEvent)
        def _on_tool_error(*args: Any) -> None:
            e = args[-1]
            err = getattr(e, "error", None)
            self.failures.append(RuntimeError(f"tool error: {err}"))

    # -- results ----------------------------------------------------------------

    def wait(self, timeout: float = 30.0) -> None:
        """Flush the CrewAI event bus so all queued handler futures finish.

        CrewAI dispatches bus events asynchronously; call this after kickoff()
        before reading attestations.
        """
        from crewai.events import crewai_event_bus

        crewai_event_bus.flush()

    def attestation_for(self, run_id: str | None = None) -> dict:
        """Attestation for a completed kickoff (most recent if unspecified)."""
        if run_id is not None:
            return self._attestations[run_id]
        if not self._attestations:
            raise KeyError("no attestation yet")
        return list(self._attestations.values())[-1]


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return repr(obj)
