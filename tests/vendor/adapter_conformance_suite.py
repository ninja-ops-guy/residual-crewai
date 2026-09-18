"""Adapter conformance suite — canonical, stdlib-only.

Version: ADAPTERS-CONFORMANCE 1.0.0 (versioned independently of the core
conformance suite; see VERSION in this directory).

Every RESIDUAL adapter (SDK backend, LangChain handler, CrewAI handler, ...)
MUST pass this suite. Binding: subclass ``AdapterConformanceTests`` and
implement ``make_handle()`` returning an ``AdapterHandle`` driving your
adapter. The suite file is intended to be vendored verbatim into adapter
repos (record the SHA-256 of the vendored copy).

Normative sources:
  * adapters/interface.md (ResidualBackend contract, error semantics)
  * specs/ATTESTATION-VERSIONING.md (token format, verdict vocabulary)
  * specs/INTEGRATION-CONTRACTS.md (failure semantics: UNKNOWN != PASS;
    fail-closed posture)

Anti-vacuity: negative paths assert the operation actually happened before
asserting detection, per GOVERNANCE.md mutation-verification rules.
"""

from __future__ import annotations

import hashlib
import json
import re
import unittest

SUITE_VERSION = "1.0.0"
GENESIS_HASH = "0" * 64
VALID_VERDICTS = frozenset({"PASS", "FAIL", "UNKNOWN", "BLOCKED"})
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_HEX40 = re.compile(r"^[0-9a-f]{40}$")

REQUIRED_TOKEN_FIELDS = (
    "attestation_id", "spec_version", "spec_head_sha", "gate_version",
    "gate_set_hash", "implementation", "evidence_manifest_hash", "verdicts",
    "issued_ns", "issuer", "prev_attestation_hash",
)


def _canon(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def recompute_attestation_id(token: dict) -> str:
    return _sha(_canon({k: v for k, v in token.items() if k != "attestation_id"}))


class AdapterHandle:
    """Binding surface an adapter provides to the suite.

    Implementations drive the adapter under test at the RESIDUAL event level
    (not framework level), so the same suite binds every adapter.
    """

    def start_run(self, run_id: str, spec_id: str) -> None:
        raise NotImplementedError

    def module_call(self, run_id: str, module: str) -> str:
        """Emit a module call through the adapter; returns the gate verdict."""
        raise NotImplementedError

    def complete_run(self, run_id: str, outcome: str = "success") -> None:
        raise NotImplementedError

    def events(self, run_id: str) -> list[dict]:
        """All ledger events for the run, in ledger order."""
        raise NotImplementedError

    def get_attestation(self, run_id: str) -> dict:
        raise NotImplementedError

    # -- chaos knobs (fail-closed probes) -----------------------------------
    def break_core(self) -> None:
        raise NotImplementedError

    def heal_core(self) -> None:
        raise NotImplementedError

    def break_ledger(self) -> None:
        raise NotImplementedError

    def heal_ledger(self) -> None:
        raise NotImplementedError

    def block_module(self, module: str) -> None:
        raise NotImplementedError

    def try_mutate_ledger(self) -> None:
        """Attempt an UPDATE or DELETE on recorded evidence. MUST raise."""
        raise NotImplementedError

    # -- expected exception types (bind to the adapter's own classes) -------
    core_unreachable_exc: type[BaseException] = Exception
    ledger_write_exc: type[BaseException] = Exception
    gate_fired_exc: type[BaseException] = Exception


class AdapterConformanceTests(unittest.TestCase):
    """Subclass and implement make_handle(). Run with unittest or pytest."""

    def make_handle(self) -> AdapterHandle:
        raise NotImplementedError

    def setUp(self):
        if type(self) is AdapterConformanceTests:
            self.skipTest("abstract suite — subclass and implement make_handle()")

    # A-ORD-1: event ordering ------------------------------------------------
    def test_event_ordering(self):
        h = self.make_handle()
        h.start_run("r1", "spec@1.0.0")
        v = h.module_call("r1", "llm:test")
        self.assertEqual(v, "PASS")
        h.complete_run("r1")
        kinds = [e["kind"] for e in h.events("r1")]
        self.assertEqual(kinds[0], "run.started")
        self.assertLess(kinds.index("module.called"), kinds.index("run.completed"))
        self.assertLess(kinds.index("run.completed"), kinds.index("attestation.issued"))

    # A-TOK-1: attestation token format ---------------------------------------
    def test_attestation_format(self):
        h = self.make_handle()
        h.start_run("r1", "spec@1.0.0")
        h.module_call("r1", "llm:test")
        h.complete_run("r1")
        tok = h.get_attestation("r1")
        for f in REQUIRED_TOKEN_FIELDS:
            self.assertIn(f, tok, f"missing field {f}")
        self.assertEqual(tok["attestation_id"], recompute_attestation_id(tok),
                         "content address must recompute")
        self.assertRegex(tok["gate_set_hash"], _HEX64)
        self.assertRegex(tok["prev_attestation_hash"], _HEX64)
        self.assertRegex(tok["evidence_manifest_hash"], _HEX64)
        self.assertRegex(tok["spec_head_sha"], _HEX40)
        impl = tok["implementation"]
        self.assertRegex(impl["commit_sha"], _HEX40)
        self.assertRegex(impl["tree_sha"], _HEX40)
        self.assertTrue(tok["verdicts"], "verdicts must be non-empty")
        for g, v in tok["verdicts"].items():
            self.assertIn(v, VALID_VERDICTS, f"verdict {g}={v} outside vocabulary")

    # A-VER-1: verdicts verbatim, UNKNOWN/BLOCKED never coerced to PASS -------
    def test_verdict_not_coerced(self):
        h = self.make_handle()
        h.block_module("tool:shell")
        h.start_run("r1", "spec@1.0.0")
        try:
            verdict = h.module_call("r1", "tool:shell")
        except h.gate_fired_exc as e:
            verdict = getattr(e, "verdict", None) or "NONPASS"
        # anti-vacuity: the gate actually fired
        self.assertNotEqual(verdict, "PASS")
        already_terminal = any(e["kind"] == "run.completed" for e in h.events("r1"))
        if not already_terminal:
            try:
                h.complete_run("r1", "aborted")
            except h.gate_fired_exc:
                pass  # adapters may surface the gate at completion instead
        tok = h.get_attestation("r1")
        self.assertTrue(
            any(v != "PASS" for v in tok["verdicts"].values()),
            f"non-PASS verdict was coerced/lost: {tok['verdicts']}",
        )
        fired = [e for e in h.events("r1") if e["kind"] == "gate.fired"]
        self.assertTrue(fired, "gate.fired event missing")
        self.assertNotEqual(fired[0]["payload"]["verdict"], "PASS")

    # A-ERR-1: core unreachable -> fail closed --------------------------------
    def test_core_unreachable_fail_closed(self):
        h = self.make_handle()
        h.break_core()
        try:
            with self.assertRaises(h.core_unreachable_exc):
                h.start_run("r1", "spec@1.0.0")
        finally:
            h.heal_core()
        # anti-vacuity + fail-closed: no evidence was recorded for the run
        self.assertEqual(h.events("r1"), [])

    # A-ERR-2: ledger write failure -> fail closed, no attestation ------------
    def test_ledger_write_failure_fail_closed(self):
        h = self.make_handle()
        h.break_ledger()
        try:
            with self.assertRaises(h.ledger_write_exc):
                h.start_run("r1", "spec@1.0.0")
        finally:
            h.heal_ledger()
        self.assertEqual(h.events("r1"), [])
        with self.assertRaises(Exception):
            h.get_attestation("r1")

    # A-GATE-1: gate firing propagates to the caller ---------------------------
    def test_gate_firing_propagates(self):
        h = self.make_handle()
        h.block_module("tool:shell")
        h.start_run("r1", "spec@1.0.0")
        fired = False
        try:
            verdict = h.module_call("r1", "tool:shell")
            if verdict != "PASS":
                fired = True  # verdict-return style
        except h.gate_fired_exc:
            fired = True  # exception style
        self.assertTrue(fired, "gate fire was not propagated to caller")
        completed = [e for e in h.events("r1")
                     if e["kind"] == "run.completed"]
        if completed:  # adapter may defer terminal to caller; if present must be aborted
            self.assertEqual(completed[0]["payload"]["outcome"], "aborted")

    # A-LED-1: hash chain integrity --------------------------------------------
    def test_hash_chain_integrity(self):
        h = self.make_handle()
        h.start_run("r1", "spec@1.0.0")
        h.module_call("r1", "llm:test")
        h.complete_run("r1")
        prev = GENESIS_HASH
        for e in h.events("r1"):
            self.assertEqual(e["prev_hash"], prev, f"chain broken at seq {e['seq']}")
            body = {
                "domain": e["domain"], "event_id": e["event_id"], "kind": e["kind"],
                "payload": e["payload"], "prev_hash": e["prev_hash"],
                "run_id": e["run_id"], "seq": e["seq"],
            }
            self.assertEqual(_sha(_canon(body)), e["digest"],
                             f"digest mismatch at seq {e['seq']}")
            prev = e["digest"]
        self.assertNotEqual(prev, GENESIS_HASH, "anti-vacuity: no events recorded")

    # A-LED-2: ledger is append-only -------------------------------------------
    def test_ledger_append_only(self):
        h = self.make_handle()
        h.start_run("r1", "spec@1.0.0")
        h.module_call("r1", "llm:test")
        h.complete_run("r1")
        self.assertTrue(h.events("r1"), "anti-vacuity: nothing to mutate")
        with self.assertRaises(Exception):
            h.try_mutate_ledger()

    # A-LED-3: exactly one terminal event per run ------------------------------
    def test_exactly_one_terminal(self):
        h = self.make_handle()
        h.start_run("r1", "spec@1.0.0")
        h.module_call("r1", "llm:test")
        h.complete_run("r1")
        with self.assertRaises(Exception):
            h.complete_run("r1")
        terminals = [e for e in h.events("r1") if e["kind"] == "run.completed"]
        self.assertEqual(len(terminals), 1)

    # A-ATT-1: attestation admitted idempotently (event_id = attestation_id) ---
    def test_attestation_idempotent_admission(self):
        h = self.make_handle()
        h.start_run("r1", "spec@1.0.0")
        h.module_call("r1", "llm:test")
        h.complete_run("r1")
        att = [e for e in h.events("r1") if e["kind"] == "attestation.issued"]
        self.assertEqual(len(att), 1)
        tok = h.get_attestation("r1")
        self.assertEqual(att[0]["event_id"], tok["attestation_id"],
                         "AT-3: attestation event_id must equal attestation_id")


# ---------------------------------------------------------------------------
# Self-binding: a minimal compliant in-memory adapter + non-compliant mutants.
# Lets this suite verify itself without any external dependency.
# ---------------------------------------------------------------------------

class _SelfTestCore:
    def __init__(self):
        self.reachable = True
        self.writable = True
        self.blocked: set[str] = set()
        self.events_log: list[dict] = []
        self.attestations: dict[str, dict] = {}
        self.seq = 0

    def append(self, run_id, domain, kind, payload, event_id=None):
        if not self.writable:
            raise _LedgerWrite("ledger write failed")
        self.seq += 1
        prev = self.events_log[-1]["digest"] if self.events_log else GENESIS_HASH
        event_id = event_id or f"ev-{self.seq}"
        e = {"seq": self.seq, "event_id": event_id, "run_id": run_id, "domain": domain,
             "kind": kind, "payload": payload, "prev_hash": prev}
        e["digest"] = _sha(_canon({
            "domain": domain, "event_id": event_id, "kind": kind, "payload": payload,
            "prev_hash": prev, "run_id": run_id, "seq": self.seq}))
        if any(x["event_id"] == event_id for x in self.events_log):
            self.seq -= 1  # idempotent admission: resolve duplicate
            return [x for x in self.events_log if x["event_id"] == event_id][0]["digest"]
        self.events_log.append(e)
        return e["digest"]


class _CoreUnreachable(Exception):
    pass


class _LedgerWrite(Exception):
    pass


class _GateFired(Exception):
    pass


class _AppendOnlyViolation(Exception):
    pass


class SelfTestHandle(AdapterHandle):
    """Minimal compliant adapter used for suite self-verification."""

    core_unreachable_exc = _CoreUnreachable
    ledger_write_exc = _LedgerWrite
    gate_fired_exc = _GateFired

    GATE_VERSION = "G0@1.0.0"
    SPEC_SHA = "a" * 40
    IMPL_SHA = "b" * 40

    def __init__(self):
        self.core = _SelfTestCore()
        self.terminal: set[str] = set()
        self.verdicts: dict[str, dict[str, str]] = {}

    def start_run(self, run_id, spec_id):
        if not self.core.reachable:
            raise _CoreUnreachable("core unreachable")
        self.core.append(run_id, "run", "run.started", {"spec_id": spec_id})

    def module_call(self, run_id, module):
        verdict = "BLOCKED" if module in self.core.blocked else "PASS"
        self.core.append(run_id, "module", "module.called",
                         {"module": module, "verdict": verdict})
        if verdict != "PASS":
            self.verdicts.setdefault(run_id, {})["G0-BLOCKED-MODULE"] = verdict
            self.core.append(run_id, "gate", "gate.fired",
                             {"gate_id": "G0-BLOCKED-MODULE", "module": module,
                              "verdict": verdict})
            self.complete_run(run_id, "aborted")
            raise _GateFired(f"{module}: {verdict}")
        return verdict

    def complete_run(self, run_id, outcome="success"):
        if run_id in self.terminal:
            raise _LedgerWrite("exactly-one-terminal violated")
        self.terminal.add(run_id)
        self.verdicts.setdefault(run_id, {}).setdefault("G0-BLOCKED-MODULE", "PASS")
        self.core.append(run_id, "run", "run.completed", {"outcome": outcome})
        self._attest(run_id)

    def _attest(self, run_id):
        manifest = {"run_id": run_id,
                    "events": [e for e in self.core.events_log if e["run_id"] == run_id]}
        tok = {
            "spec_version": "1.0.0", "spec_head_sha": self.SPEC_SHA,
            "gate_version": self.GATE_VERSION,
            "gate_set_hash": _sha(_canon(["G0-BLOCKED-MODULE"])),
            "implementation": {"repo": "self-test", "commit_sha": self.IMPL_SHA,
                               "tree_sha": self.IMPL_SHA},
            "evidence_manifest_hash": _sha(_canon(manifest)),
            "verdicts": self.verdicts[run_id],
            "issued_ns": 1, "issuer": "self-test",
            "prev_attestation_hash": GENESIS_HASH,
        }
        tok["attestation_id"] = recompute_attestation_id(tok)
        self.core.attestations[run_id] = tok
        self.core.append(run_id, "attestation", "attestation.issued",
                         {"attestation_id": tok["attestation_id"]},
                         event_id=tok["attestation_id"])

    def events(self, run_id):
        return [e for e in self.core.events_log if e["run_id"] == run_id]

    def get_attestation(self, run_id):
        if run_id not in self.core.attestations:
            raise KeyError(run_id)
        return self.core.attestations[run_id]

    def break_core(self):
        self.core.reachable = False

    def heal_core(self):
        self.core.reachable = True

    def break_ledger(self):
        self.core.writable = False

    def heal_ledger(self):
        self.core.writable = True

    def block_module(self, module):
        self.core.blocked.add(module)

    def try_mutate_ledger(self):
        if not self.core.events_log:
            raise _AppendOnlyViolation("nothing to mutate")
        raise _AppendOnlyViolation("append-only: mutation refused")


class CompliantSelfTest(AdapterConformanceTests):
    def make_handle(self):
        return SelfTestHandle()


# -- Non-compliant mutants (negative conformance evidence) --------------------

class MutantCoercesVerdicts(SelfTestHandle):
    """MUTANT A: coerces non-PASS gate verdicts to PASS in the attestation."""

    def _attest(self, run_id):
        super()._attest(run_id)
        self.core.attestations[run_id]["verdicts"] = {
            g: "PASS" for g in self.core.attestations[run_id]["verdicts"]
        }
        # keep content address consistent so only the coercion is detectable
        tok = self.core.attestations[run_id]
        tok["attestation_id"] = recompute_attestation_id(tok)


class MutantFailOpenCore(SelfTestHandle):
    """MUTANT B: unreachable core is ignored; run proceeds and records events."""

    def start_run(self, run_id, spec_id):
        # silently proceeds even when core unreachable
        self.core.append(run_id, "run", "run.started", {"spec_id": spec_id})


class MutantSwallowsGate(SelfTestHandle):
    """MUTANT C: gate fire is recorded but never propagated to the caller."""

    def module_call(self, run_id, module):
        try:
            return super().module_call(run_id, module)
        except _GateFired:
            return "PASS"  # swallow


class MutantMutableLedger(SelfTestHandle):
    """MUTANT D: ledger allows retroactive mutation."""

    def try_mutate_ledger(self):
        self.core.events_log.clear()  # mutation succeeds — violation


MUTANTS = {
    "A-coerces-verdicts": (MutantCoercesVerdicts, {"test_verdict_not_coerced"}),
    "B-fail-open-core": (MutantFailOpenCore, {"test_core_unreachable_fail_closed"}),
    "C-swallows-gate": (MutantSwallowsGate, {"test_gate_firing_propagates"}),
    "D-mutable-ledger": (MutantMutableLedger, {"test_ledger_append_only"}),
}


class TestMutantsRejected(unittest.TestCase):
    """Mutation verification: each mutant MUST fail the suite, and the
    specific test designed to catch it MUST be among the failures."""

    def _run_against(self, handle_cls):
        class Bound(AdapterConformanceTests):
            def make_handle(self):
                return handle_cls()

        suite = unittest.TestLoader().loadTestsFromTestCase(Bound)
        result = unittest.TestResult()
        suite.run(result)
        failed_names = {
            t._testMethodName for t, _ in result.failures + result.errors
        }
        return result, failed_names

    def test_compliant_handle_passes(self):
        result, failed = self._run_against(SelfTestHandle)
        self.assertEqual(failed, set(), f"compliant handle failed: {failed}")
        self.assertGreater(result.testsRun, 0, "anti-vacuity: suite ran zero tests")

    def test_each_mutant_detected(self):
        for name, (mutant, expected_failures) in MUTANTS.items():
            with self.subTest(mutant=name):
                result, failed = self._run_against(mutant)
                self.assertTrue(failed, f"mutant {name} was NOT detected")
                self.assertTrue(
                    expected_failures & failed,
                    f"mutant {name}: expected {expected_failures} to fail, got {failed}",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
