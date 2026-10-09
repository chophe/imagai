---
gsd_state_version: "1.0"
milestone: v1.0
current_phase: "02.5"
current_phase_name: Web Server Safety
status: planning
stopped_at: Phase 1 complete, ready to plan Phase 02.5
last_updated: "2026-10-09T13:20:03.127Z"
last_activity: 2026-10-09
last_activity_desc: Phase 1 complete, transitioned to Phase 02.5
state_head: 8c2cd24fe218df8245654664fc78787a5be68726
progress:
  total_phases: 7
  completed_phases: 2
  total_plans: 10
  completed_plans: 5
  percent: 29
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-10-07)

**Core value:** Fast prompt-to-image. If everything else fails, turning a prompt into an image must still work.
**Current focus:** Phase 2.5 — Web Server Safety

## Current Position

Phase: 02.5 — Web Server Safety
Plan: Not started
Status: Ready to plan
Last activity: 2026-10-09 — Phase 1 complete, transitioned to Phase 02.5

Progress: [███░░░░░░░] 29%

## Performance Metrics

**Velocity:**

- Total plans completed: 5
- Average duration: n/a
- Total execution time: 0.0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 1 | 3 | - | - |
| 02 | 2 | - | - |

**Recent Trend:**

- Last 5 plans: none yet
- Trend: n/a

*Updated after each plan completion*
**Per-Plan Metrics:**

| Plan | Duration | Tasks | Files |
|------|----------|-------|-------|
| Phase 01 P01 | 15m | 3 tasks | 5 files |
| Phase 01 P02 | 96m | 3 tasks | 6 files |
| Phase 01 P03 | 5m | 2 tasks | 0 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table. Recent decisions affecting current work:

- Roadmap derives phase order from dependency, not preference: toolchain first because no later phase can be proven without a runnable test suite.
- `CFG-03` (`requires-python`) and `ENV-04` (`typing_extensions`) were folded into Phase 1 — both edit `pyproject.toml`, the file the uv migration already rewrites, and PROJECT.md groups the version-floor problem with the toolchain constraint. Splitting them would mean editing the same file twice in one cycle.
- `SEC-01`/`SEC-02` (core-level containment) are kept distinct from `SEC-04` (HTTP schema validation): containment lives in the shared save pipeline so the CLI is protected too, while `SEC-04` is about request-shape validation at the web boundary.
- Phase 4 (error contracts) is sequenced before Phase 6 (registry) so `ARCH-03`'s regression test freezes known-good error behavior instead of today's swallowing.
- No `**UI hint**` annotation on any phase. This milestone adds no user-facing capability; Phase 3 touches the HTTP layer only, not visual design.
- [Phase 01]: Remove [tool.rye] entirely per D-03
- [Phase 01]: Use PEP 735 [dependency-groups] per D-01
- [Phase 01]: requires-python >=3.9 per D-04
- [Phase 01]: Remove requests per D-10
- [Phase 01]: Keep werkzeug per D-09
- [Phase 01]: stdlib typing.Annotated per D-07
- [Phase 01]: Replace all rye references with uv equivalents per D-02
- [Phase 01]: Use uv run imagai pattern for README commands
- [Phase 01]: Use uv python pin for Python version switching
- [Phase 01]: Use uv lock --update-package for lockfile updates
- [Phase 01]: All 10 checks pass — Phase 1 success criteria fully satisfied
- [Phase 01]: Verification-only plan — no code changes required
- [Phase 02]: _contained_path helper rejects absolute paths, .. traversal, and paths outside settings.output_dir
- [Phase 02]: Both save functions return None (no exception) on containment rejection per D-04
- [Phase 02]: Containment check is first statement in try block, before any I/O or network activity
- [Phase 02]: CR-01 fix — `_contained_path` must NOT blanket-reject absolute candidates. `core.py`
  builds `Path(settings.output_dir) / name`, which is absolute whenever `output_dir` is, so the
  original `is_absolute()` early-reject made every save fail. Bare-basename is now enforced by
  `canonical_output.parent != canonical_root` (commit `558c314`, regression test at
  `tests/test_containment.py:119`).
- [Phase 02]: Verifier rejected code review finding WR-01 — it proposed allowing anchored
  subdirectories, which contradicts D-02's bare-basename rule. The existing test is correct.

### Pending Todos

None yet.

### Blockers/Concerns

- **Resolved — write collision between two roadmapper invocations.** A concurrent invocation wrote ROADMAP.md and STATE.md at 14:54 on 2026-10-02, overwriting this draft and leaving the three planning files mutually inconsistent (REQUIREMENTS.md traced CFG-03 to Phase 1; the competing roadmap put it in Phase 5). This version was restored and the competing draft preserved at `.planning/research/ROADMAP-alternate-2026-10-02.md`. The one substantive difference is the CFG-03 placement. If the orchestrator spawned two roadmappers, only one should proceed.
- **Resolved — Phase 1 was the hard blocker.** Phase 1 shipped the uv toolchain: `uv sync` + `uv run pytest` pass from a clean checkout on the pinned 3.12.9 interpreter, rye lockfiles deleted, `typing.Annotated` from stdlib. The tool now runs.
- **Requirement count discrepancy.** REQUIREMENTS.md stated "17 total" v1 requirements; 16 REQ-IDs exist. All 16 are mapped — the count was an overcount, not a missing ID. Coverage line corrected during this step.
- **Phase 5 config risk (highest technical risk in the milestone).** The `os.environ` loop at `config.py:38-54` exists to work around nested-delimiter parsing for engine names containing `__` (e.g. `openai_dalle3`). Removing it without a parametrized equivalence test risks a silent config regression no existing test would catch.
- **Resolved — Phase 2 open question.** `UPLOAD_FOLDER` is now `Path(settings.output_dir)`
  (`web_server.py:33`), so read/serve and write agree on one directory (D-05, shipped).
- **RESOLVED — command injection in `/api/generate-cli` is now owned.** `web_server.py:239-255`
  passed user input to `subprocess.run(..., shell=True)` behind a `startswith` guard that
  `imagai; <cmd>` bypasses (CR-02). Combined with `main()` defaulting to `host="0.0.0.0"`, that was
  a network-reachable RCE by default. Assigned to **inserted Phase 2.5 as SEC-05/SEC-06**, which
  lands before Phase 3 and fixes `shell=False`, the loopback bind, and `debug=False`.
- **RESOLVED — the `NameError` was not unassigned after all.** `web_server.py:364` calls `main()`
  before its definition at `:368` (CR-03). It was already Phase 3's SEC-03; it has been **moved to
  Phase 2.5**, which reorders the same lines. Phase 3 keeps a regression guard only.
- **Corrected a false premise.** "The web server is localhost-only" was wrong — it bound `0.0.0.0`.
  That wrong premise is why the RCE read as a low-priority "correctness bug" and why auth looked
  out of scope. Phase 2.5 restores the premise. Until it lands, treat the server as exposed.
- **Test suite has a live-API dependency.** `test_llm_filename_contained` calls a real engine and
  hangs the run. Use `uv run pytest -k "not llm"` locally. Fixing it is unscheduled.
- **Test infrastructure is thin.** `tests/test_cli.py` still holds two `assert True` placeholders,
  though Phase 2 added `tests/test_containment.py` (13 tests).
- **Unverified `graft/` index.** `AGENTS.md` and `GEMINI.md` assert the repo is indexed; a `graft/` directory exists but its content is unconfirmed. Do not let a stale index block implementation.

## Deferred Items

Items acknowledged and deferred at milestone close, most recent first:

| Category | Item | Status | Deferred At | Milestone |
|----------|------|--------|-------------|-----------|
| Tooling | TOOL-01 — linter/formatter (ruff) | Deferred to v2 | 2026-10-02 | v1.0 |
| Tooling | TOOL-02 — CI on commit | Deferred to v2 | 2026-10-02 | v1.0 |
| Quality | QUAL-01 — coverage beyond `tests/test_cli.py` | Partly absorbed by per-phase tests; standalone goal deferred | 2026-10-02 | v1.0 |
| Quality | QUAL-02 — logging configuration for three module loggers | Deferred to v2 | 2026-10-02 | v1.0 |
| Features | FEAT-01 — additional image backends | Deferred to v2; unblocked by ARCH-01/02 | 2026-10-02 | v1.0 |
| Features | FEAT-02 — move Rich rendering out of provider data layer | Deferred to v2 | 2026-10-02 | v1.0 |
| Features | FEAT-03 — replace blocking sync clients in `async def` | Deferred to v2 | 2026-10-02 | v1.0 |

## Session Continuity

Last session: 2026-10-07T16:12:49.386Z
Stopped at: Phase 1 complete, ready to plan Phase 02.5
Resume file: .planning/phases/02.5-web-server-safety/02.5-CONTEXT.md
