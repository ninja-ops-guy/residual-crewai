"""Runs the canonical adapter conformance suite against ResidualCrewAIListener.

The handle emits real CrewAI event objects onto the global crewai_event_bus
(the same path CrewAI itself uses), then flushes the bus, so the listener's
mapping code is exercised end-to-end.

Vendored suite: tests/vendor/adapter_conformance_suite.py
SHA-256: b8e14c9babc0eb4c7e07b04768afdda6e49c7a404e6e4ba1fc4f700f276b290e
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "vendor"))

from adapter_conformance_suite import AdapterConformanceTests, AdapterHandle

from crewai.events import crewai_event_bus
from crewai.events.types.crew_events import (
    CrewKickoffCompletedEvent,
    CrewKickoffFailedEvent,
    CrewKickoffStartedEvent,
)
from crewai.events.types.task_events import TaskStartedEvent
from crewai.events.types.tool_usage_events import ToolUsageStartedEvent

from residual_crewai import ResidualCrewAIListener
from residual_sdk import (
    CoreUnreachableError,
    GateFiredError,
    GateSet,
    LedgerWriteError,
    SQLiteResidualBackend,
)


class _ChaosTransport:
    def __init__(self):
        self.reachable = True
        self.writable = True

    def ping(self):
        if not self.reachable:
            raise ConnectionRefusedError("core unreachable")
        return True

    def commit(self):
        if not self.writable:
            raise OSError("ledger commit failed")


class _Src:
    """Minimal event source object."""


class CrewAIHandle(AdapterHandle):
    core_unreachable_exc = CoreUnreachableError
    ledger_write_exc = LedgerWriteError
    gate_fired_exc = GateFiredError

    def __init__(self):
        self.transport = _ChaosTransport()
        self.blocked: set[str] = set()
        self._new()

    def _new(self):
        self.backend = SQLiteResidualBackend(
            ":memory:",
            gate_set=GateSet(blocked_modules=self.blocked),
            transport=self.transport,
        )
        self.listener = ResidualCrewAIListener(backend=self.backend)

    def _flush(self):
        crewai_event_bus.flush()
        # surface listener-side failures (bus swallows handler exceptions)
        while self.listener.failures:
            err = self.listener.failures.pop(0)
            raise err

    def start_run(self, run_id, spec_id):
        self.listener.spec_id = spec_id
        try:
            crewai_event_bus.emit(_Src(), CrewKickoffStartedEvent(crew_name="conf", inputs=None))
            crewai_event_bus.flush()
        finally:
            self._runs = getattr(self, "_runs", {})
            self._runs[run_id] = self.listener._active_run
        # propagate core-unreachable raised synchronously
        while self.listener.failures:
            raise self.listener.failures.pop(0)

    def _r(self, run_id):
        return self._runs.get(run_id, run_id)

    def module_call(self, run_id, module):
        kind, _, name = module.partition(":")
        if kind == "task":
            crewai_event_bus.emit(_Src(), TaskStartedEvent(context=name, task_name=name))
        else:
            crewai_event_bus.emit(_Src(), ToolUsageStartedEvent(tool_name=name, tool_args={}))
        crewai_event_bus.flush()
        if self.listener.failures:
            err = self.listener.failures.pop(0)
            if isinstance(err, GateFiredError):
                raise err
            raise err
        return "PASS"

    def complete_run(self, run_id, outcome="success"):
        ev = CrewKickoffCompletedEvent(crew_name="conf", output=None) if outcome == "success" else CrewKickoffFailedEvent(crew_name="conf", error="x")
        crewai_event_bus.emit(_Src(), ev)
        crewai_event_bus.flush()
        while self.listener.failures:
            raise self.listener.failures.pop(0)

    def events(self, run_id):
        return self.backend.events(self._r(run_id))

    def get_attestation(self, run_id):
        return self.backend.get_attestation(self._r(run_id))

    def break_core(self):
        self.transport.reachable = False

    def heal_core(self):
        self.transport.reachable = True

    def break_ledger(self):
        self.transport.writable = False

    def heal_ledger(self):
        self.transport.writable = True

    def block_module(self, module):
        self.blocked.add(module)
        self._new()

    def try_mutate_ledger(self):
        self.backend._conn.execute("UPDATE events SET kind='x'")


class TestCrewAIConformance(AdapterConformanceTests):
    def make_handle(self):
        return CrewAIHandle()


class MutantListener(ResidualCrewAIListener):
    """Deliberately non-compliant: swallows gate fires."""

    def _module(self, module, payload):
        try:
            super()._module(module, payload)
        except GateFiredError:
            self.failures.clear()


class MutantHandle(CrewAIHandle):
    def _new(self):
        self.backend = SQLiteResidualBackend(
            ":memory:",
            gate_set=GateSet(blocked_modules=self.blocked),
            transport=self.transport,
        )
        self.listener = MutantListener(backend=self.backend)


class TestMutantRejected(unittest.TestCase):
    def test_gate_swallowing_mutant_detected(self):
        class Bound(AdapterConformanceTests):
            def make_handle(self):
                return MutantHandle()

        result = unittest.TestResult()
        unittest.TestLoader().loadTestsFromTestCase(Bound).run(result)
        failed = {t._testMethodName for t, _ in result.failures + result.errors}
        assert result.testsRun > 0
        assert {"test_gate_firing_propagates", "test_verdict_not_coerced"} & failed, failed


if __name__ == "__main__":
    unittest.main(verbosity=2)
