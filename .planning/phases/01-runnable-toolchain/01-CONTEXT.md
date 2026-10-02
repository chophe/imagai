# Phase 1: Runnable Toolchain - Context

**Gathered:** 2026-10-02
**Status:** Ready for planning

<domain>
## Phase Boundary

The project installs, imports, and tests on macOS using only uv. This phase migrates the
toolchain off rye and leaves the environment runnable. It delivers **no user-facing
capability** — every change is build/config/docs surface.

Requirements: ENV-01, ENV-02, ENV-03, ENV-04, CFG-03.

Explicitly out of scope: the `config.py` env-loop cleanup (Phase 5), any provider work
(Phase 6), any path-safety or HTTP work (Phases 2–3).

</domain>

<decisions>
## Implementation Decisions

### Dev dependency wiring
- **D-01:** Replace `[tool.rye] dev-dependencies` with PEP 735
  `[dependency-groups]` and a `dev = ["pytest>=7.0.0"]` table. Current uv docs
  recommend `dependency-groups.dev`; `[tool.uv] dev-dependencies` still works but is
  explicitly marked "not recommended" and on a deprecation path.
- **D-02:** Generate a single `uv.lock` via `uv lock`, then delete both
  `requirements.lock` and `requirements-dev.lock`. Required for ENV-03 — both are
  rye-generated, so criterion 3 (`rg -n rye` returns nothing) cannot pass while they exist.
- **D-03:** Remove the `[tool.rye]` section entirely, not just `managed = true`.
  Leaving a partially-populated `[tool.rye]` table is what makes `rg -n rye` still hit.

### Python floor vs pin
- **D-04:** `requires-python = ">=3.9"` is the honest floor; `.python-version` keeps
  pinning `3.12.9`. Floor and pin are different jobs — floor governs what uv will resolve
  for, `.python-version` picks the reproducible dev interpreter. Do not collapse them.
- **D-05:** Verify 3.9 compatibility on the **pinned 3.12.9 interpreter only**. Python 3.9
  reached EOL in October 2025. Supporting evidence that 3.9 was never actually verified:
  `requirements.lock` contains zero `python_version` markers, so it was resolved for a
  single interpreter.
- **D-06:** **ROADMAP EDIT REQUIRED.** Criterion 5 currently reads "the project resolves
  and installs on a 3.9 interpreter". That gate is not real per D-05. It must be rewritten
  to assert that declared imports are satisfiable on the pinned interpreter. Do not treat
  criterion 5 as written as a 3.9 test-matrix requirement.

### ENV-04 resolution — the `typing_extensions` import
- **D-07:** Change `src/imagai/cli.py:2` from
  `from typing_extensions import Annotated` to `from typing import Annotated`.
  `Annotated` has been stdlib since Python 3.9 (PEP 593). **Add no new dependency** — do
  not add `typing_extensions` to `[project]`.
- **D-08:** D-07 and the CFG-03 floor bump land in **one plan**, not two. The import change
  is invalid in isolation: on the current (wrong) `>=3.8` floor a 3.8 interpreter has no
  `typing.Annotated`. The two edits are coupled by the 3.9 boundary.
- **D-09:** Keep `werkzeug` as an explicit dependency. It *is* imported directly
  (`secure_filename` in `src/imagai/web_server.py`). This **corrects** `.planning/codebase/STACK.md`,
  which claims the explicit pin is redundant because Flask provides it transitively — true
  of availability, false of the import being direct.

### Dependency hygiene
- **D-10:** *(AGENT DEFAULT — user did not confirm; three interactive prompts were killed
  by server restarts. Reverse if you disagree.)* Remove `requests>=2.32.5` from
  `[project] dependencies`. Evidence: zero imports across `src/` and `tests/` — `httpx` is
  what performs HTTP. Removal cannot break an import; it only shrinks the lockfile and
  stops misleading readers.

### Local environment cleanup
- **D-11:** *(AGENT DEFAULT — user did not confirm; same interruption as D-10.
  Reverse if you disagree.)* Deleting the local Windows-built `.venv/` is a **manual
  prerequisite, not a plan task**. It is gitignored (`.gitignore:26`) with zero tracked
  files, so no repository change can express it. The plan should state it as a pre-flight
  step; the executor deletes it locally before the first `uv sync`.
- **D-12:** The plan must not attempt to remove rye from the machine or otherwise mutate
  anything outside the repository. Scope is the repo plus the developer's local venv.

### the agent's Discretion
- Exact `uv.lock` contents and package versions resolved at lock time — the current rye
  lockfile pins exact versions, but uv will resolve fresh and may pick different ones.
  Anything satisfying the declared floors is acceptable.
- Whether README command examples use `uv run imagai ...` or a `uv run` alias. The
  criterion is that they are uv commands and contain no rye references.
- Exact wording of README prose, provided `rg -n rye` returns nothing.

</decisions>

<specifics>
## Specific Ideas

Verification is expected to be mechanical, not narrative. The phase's own criteria are
mostly greppable or runnable:

- `rg -n rye` returns nothing (excluding historical planning notes under `.planning/`)
- `uv sync` from a clean venv succeeds
- `uv run pytest` completes with no import errors
- `uv run python -c "import imagai.cli"` succeeds
- `uv run python -V` reports 3.12.9

Note that `.planning/` will legitimately contain the word "rye" — ROADMAP notes,
PROJECT.md decisions, and this file all discuss the migration. The criterion means the
*build and docs surface*, not the planning archive. Write the grep so it excludes
`.planning/`.

</specifics>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Build and dependency config
- `pyproject.toml` — the primary artifact. `[tool.rye]` (dev-deps) is removed,
  `requires-python` moves to `>=3.9`, `requests` is dropped, `[dependency-groups]` is added.
- `.python-version` — pins `3.12.9`; keep as-is per D-04.
- `requirements.lock`, `requirements-dev.lock` — rye-generated, 226 lines combined; both deleted per D-02.
- `src/imagai/cli.py` — line 2 is the `typing_extensions` import changed by D-07.

### Project requirements and constraints
- `.planning/REQUIREMENTS.md` — ENV-01 through ENV-04 and CFG-03, plus the v2 items explicitly excluded from this phase.
- `.planning/ROADMAP.md` — Phase 1 success criteria. **Criterion 5 is stale; see D-06.**
- `.planning/PROJECT.md` — Key Decisions: migrate rye → uv, harden before adding features.
- `.planning/codebase/STACK.md` — Package Manager and Runtime sections. **Contains one known error, corrected by D-09.**

### Verification surface
- `tests/test_cli.py` — the only test file; `uv run pytest` must pass it.
- `.gitignore:26` — proves `.venv/` is ignored (bears on D-11).

</canonical_refs>

<deferred>
## Deferred Ideas

- **General dependency audit beyond `requests`** (D-10) — v2 TOOL-01, tied to adding a
  linter. `werkzeug` was checked and is legitimately used; no other unused pins were found
  in this phase's scope.
- **`.venv/` deletion** is local-only and cannot be a tracked change (D-11).

</deferred>

---

*Decisions captured 2026-10-02 during `/gsd:discuss-phase 1`. D-10 and D-11 are agent
defaults applied after repeated server interruptions prevented user confirmation — verify
before executing.*