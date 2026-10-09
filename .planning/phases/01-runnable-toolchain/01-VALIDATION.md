---
phase: "01"
slug: runnable-toolchain
status: validated
nyquist_compliant: true
wave_0_complete: false
created: "2026-10-09"
---

# Phase 01 — Validation Strategy

> Per-phase validation contract for feedback sampling during execution.
> Reconstructed/refreshed by /gsd-validate-phase on 2026-10-09 (State B — no prior VALIDATION.md, SUMMARY present).

---

## Test Infrastructure

| Property | Value |
|----------|-------|
| **Framework** | pytest 7.x (uv-managed) |
| **Config file** | pyproject.toml (`[tool.pytest.ini_options]` — configfile per pytest output) |
| **Quick run command** | `uv run pytest -k "not llm" -q` |
| **Full suite command** | `uv run pytest -k "not llm" -q` (one test hits a live API — never run without the filter) |
| **Estimated runtime** | ~3 seconds (20 passed, 1 deselected) |

---

## Sampling Rate

- **After every task commit:** Run `uv run pytest -k "not llm" -q`
- **After every plan wave:** Run `uv run pytest -k "not llm" -q`
- **Before `/gsd-verify-work`:** Full suite must be green
- **Max feedback latency:** 10 seconds

---

## Per-Task Verification Map

| Task ID | Plan | Wave | Requirement | Threat Ref | Secure Behavior | Test Type | Automated Command | File Exists | Status |
|---------|------|------|-------------|------------|-----------------|-----------|-------------------|-------------|--------|
| 01-01-01 | 01 | 1 | CFG-03, ENV-04 | — | N/A | unit | `uv run pytest tests/test_toolchain.py -k requires_python_floor or stdlib_typing` | ✅ | ✅ |
| 01-01-02 | 01 | 1 | ENV-01 | — | N/A | integration | `uv lock --check` via `uv run pytest tests/test_toolchain.py -k lock_is_in_sync` | ✅ | ✅ |
| 01-01-03 | 01 | 1 | ENV-01, ENV-02 | — | N/A | integration | `uv run pytest -k "not llm" -q` + version-consistency test | ✅ | ✅ |
| 01-02-01 | 02 | 2 | ENV-03 | — | N/A | unit | `uv run pytest tests/test_toolchain.py -k no_rye_references` | ✅ | ✅ |
| 01-02-02 | 02 | 2 | ENV-03 | — | N/A | unit | `uv run pytest tests/test_toolchain.py -k single_uv_command_workflow` | ✅ | ✅ |
| 01-02-03 | 02 | 2 | ENV-03 | — | N/A | unit | covered by `no_rye_references` scan (docs/, web_interface.html, web_server.py) | ✅ | ✅ |
| 01-03-01 | 03 | 3 | ENV-03 | — | N/A | unit | covered by `no_rye_references` scan | ✅ | ✅ |
| 01-03-02 | 03 | 3 | ENV-01, ENV-02 | — | N/A | smoke | `uv run pytest tests/test_toolchain.py -k version_consistent or lock_is_in_sync` | ✅ | ✅ |

---

## Wave 0 Requirements

- [x] `tests/test_toolchain.py` — behavioral tests for ENV-01..04, CFG-03 (added 2026-10-09 by this audit)
- [x] Framework (pytest via PEP 735 `[dependency-groups] dev`) — pre-existing, no install needed

---

## Manual-Only Verifications

| Behavior | Requirement | Why Manual | Test Instructions |
|----------|-------------|------------|-------------------|
| True clean-checkout install (`rm -rf .venv && uv sync` on a fresh clone) | ENV-01 | Deleting the working `.venv` inside a test run is destructive and slow | From a fresh clone: `rm -rf .venv && uv sync && uv run pytest -k "not llm"`; expect exit 0, all tests pass. (Lock-sync equivalent is automated via `uv lock --check`.) |

---

## Validation Audit 2026-10-09

Refreshed verification found all 5 phase requirements pass:
- **ENV-01** — lock↔pyproject consistency automated (`uv lock --check`); README documents `uv sync` + `uv run pytest`. Clean-checkout install itself manual-only.
- **ENV-02** — `.python-version` (3.12.9) vs `requires-python` floor (>=3.9) vs lock floor all verified in `test_python_version_consistent_across_pin_lock_and_floor`.
- **ENV-03** — repo-wide "rye" scan. **Audit finding:** a stale `.rye/` entry on `.gitignore:43` was invisible to the original phase check (`rg -n rye` skips hidden files by default). The strict new test surfaced it; the residue was removed during this refresh. Rusty-history: the aggressive NEW test initially matched false positives (its own source strings, `graft/.cache/` artifacts) — both excluded. Final: 0 offenders.

## Validation Sign-Off

- [x] All tasks have `<automated>` verify or Wave 0 dependencies
- [x] Sampling continuity: no 3 consecutive tasks without automated verify
- [x] Wave 0 covers all MISSING references
- [x] No watch-mode flags
- [x] Feedback latency < 10s
- [x] `nyquist_compliant: true` set in frontmatter

**Approval:** approved 2026-10-09

## Validation Audit 2026-10-09

| Metric | Count |
|---|---|
| Gaps found | 5 |
| Resolved | 5 |
| Escalated | 0 |
