# Releases

## 1.0.0 (tag intent: `v1.0.0`)

No git-tag creation tool was available to the authoring agent; tag intent is
recorded here: `v1.0.0` on the commit landing this file on the default branch.

- `ResidualCrewAIListener` for CrewAI's event bus.
- Verified against crewai 1.15.22 in the authoring sandbox with real
  `Crew.kickoff()` runs (native-provider FakeLLM; no network).
- Passes ADAPTERS-CONFORMANCE 1.0.0 via tests/test_conformance_binding.py
  (vendored suite SHA-256 b8e14c9babc0eb4c7e07b04768afdda6e49c7a404e6e4ba1fc4f700f276b290e),
  driven through real CrewAI event emission on the global bus.
- Mutation-verified: gate-swallowing listener mutant is detected
  (tests/test_conformance_binding.py::TestMutantRejected).

Suggested CI (not in tree — workflow-scope token limitation):

```yaml
name: tests
on: [push, pull_request]
jobs:
  pytest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - run: pip install -e . pytest && python3 -m pytest tests/ -q
```
