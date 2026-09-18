# residual-crewai

CrewAI adapter for RESIDUAL run attestation: a `BaseEventListener` on the
CrewAI event bus that turns crew kickoffs into spec-compliant attestations.

## Quickstart (<20 lines to first attestation)

```python
from crewai import Agent, Crew, Task
from residual_crewai import ResidualCrewAIListener
from residual_sdk import SQLiteResidualBackend, verify_attestation

listener = ResidualCrewAIListener(backend=SQLiteResidualBackend("evidence.db"))

agent = Agent(role="researcher", goal="answer questions", backstory="...")
crew = Crew(agents=[agent],
            tasks=[Task(description="Q", expected_output="A", agent=agent)])
crew.kickoff()
listener.wait()                      # CrewAI dispatches events async

token = listener.attestation_for()
assert verify_attestation(token).ok
```

## Lifecycle mapping

| CrewAI event | RESIDUAL |
|---|---|
| `CrewKickoffStartedEvent` | `on_run_start` (one run per kickoff) |
| `TaskStartedEvent` | `on_module_call("task:<name>")` |
| `AgentExecutionStartedEvent` | `on_module_call("agent:<role>")` — role-based agents |
| `ToolUsageStartedEvent` | `on_module_call("tool:<name>")` — **delegation** between agents surfaces here in CrewAI and is attested at tool granularity |
| `CrewKickoffCompletedEvent` | `on_run_complete("success")` |
| `CrewKickoffFailedEvent` / `TaskFailedEvent` / `AgentExecutionErrorEvent` | `on_run_complete("error")` |

Parallel tasks/concurrent kickoffs: each kickoff gets an independent residual
run id; overlapping crews never share evidence.

## What the attestation proves / does not prove

See residual-sdk README. Additionally: CrewAI dispatches bus events on
futures and **logs rather than raises** handler exceptions — so fail-closed
signals surface in `listener.failures` (drained by `wait()` callers and by
the conformance binding). Always check `listener.failures` after kickoff; a
quiet kickoff with a failed ledger write is otherwise invisible. This adapter
does NOT attest individual agent messages or LLM token streams.

## Troubleshooting

- `KeyError: no attestation yet` after kickoff → call `listener.wait()`
  first (async bus dispatch).
- Gate fire mid-crew → recorded in `listener.failures` as `GateFiredError`;
  run outcome is `aborted`; verdict stored verbatim in the attestation.
- Silent ledger failure → `LedgerWriteError` in `listener.failures`; the run
  produced NO attestation (fail-closed by design).

## Migration from direct core use

Delete manual bookkeeping around `crew.kickoff()`; instantiate the listener
once per process (it self-registers on the global event bus), call
`listener.wait()` after kickoff, read tokens via `attestation_for()`. Gate
configuration lives on the backend's `GateSet`.

## Tests

```
pip install -e . pytest
python3 -m pytest tests/ -q   # 16 passed (real Crew/Agent/Task kickoffs with a stubbed native LLM provider, conformance binding, mutant rejection)
```

Verified against crewai 1.15.22 with real `Crew.kickoff()` runs (native
OpenAI provider subclass with `call()` stubbed — no network, no API keys).
CI YAML omitted (workflow-scope token limitation); see RELEASES.md.
