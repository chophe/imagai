<!-- SUPERSEDED DRAFT — NOT the active roadmap. See .planning/ROADMAP.md. -->
<!-- A concurrent roadmapper invocation wrote this to .planning/ROADMAP.md at 14:54 on
     2026-10-02. It differs from the active roadmap in one mapping: CFG-03 (requires-python
     floor) -> Phase 5 here, -> Phase 1 in the active roadmap. Both map all 16 v1 requirements. -->

# Roadmap: Imagai

## Overview

Six phases that take imagai from a codebase that cannot run to a runnable, correct, and
maintainable tool. Nothing here adds a user-facing capability. Phase 1 unblocks the toolchain
and is a hard prerequisite for every other phase, because no test can be written or run until
the environment works. Phases 2–6 each fix one category of defect or structural issue found
during codebase mapping, and each lands with passing tests. The `core.py` provider registry
lands last because it shares a file with the path-safety fix, and doing the smaller, riskier
correction first keeps the refactor honest.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Toolchain Migration** — Move rye to uv so the project actually runs
- [ ] **Phase 2: Output Path Safety** — Eliminate the arbitrary file write
- [ ] **Phase 3: Web Server Correctness** — Fix the NameError and add request validation
- [ ] **Phase 4: Error Surfacing** — Stop swallowing failures
- [ ] **Phase 5: Configuration Cleanup** — Remove the duplicative env loop and import side effect
- [ ] **Phase 6: Provider Registry** — Decouple providers from `core.py`

## Phase Details

### Phase 1: Toolchain Migration
**Goal**: The project installs, imports, and tests on macOS with uv, replacing the rye configuration that cannot be executed on this host
**Mode**: mvp
**Depends on**: Nothing (first phase)
**Requirements**: ENV-01, ENV-02, ENV-03, ENV-04, CFG-03
**Success Criteria** (what must be TRUE):
  1. On a machine with no pre-existing venv, `uv sync` followed by `uv run pytest` completes the test run with no import errors
  2. `uv run python -V` reports 3.12.9, and the same version appears in `.python-version` and `pyproject.toml`
  3. `README.md` contains no `rye` commands; `requirements.lock` and `requirements-dev.lock` are removed; no `[tool.rye]` block remains in `pyproject.toml`
  4. After deleting the venv and re-syncing, `uv run python -c "import imagai.cli"` succeeds without `typing_extensions` being pulled in only as a transitive dependency
  5. `requires-python` in `pyproject.toml` states `>=3.9`, consistent with pydantic 2.11's floor rather than the current incorrect `>=3.8`
**Plans**: 4 plans

Plans:
- [ ] 01-01: Add uv configuration and Python pin to `pyproject.toml`; remove `[tool.rye]` and the rye lockfiles; declare or drop `typing_extensions`
- [ ] 01-02: Correct `requires-python` to `>=3.9` in the same `pyproject.toml` edit
- [ ] 01-03: Rewrite the README's setup/run/test commands from rye to uv
- [ ] 01-04: Verify from a clean venv that sync, import, and the test suite all succeed; pin the resolved Python version

### Phase 2: Output Path Safety
**Goal**: A caller-supplied output filename can never cause a write outside `generated_images/`
**Mode**: mvp
**Depends on**: Phase 1 (needs a working environment to write and run the verification tests)
**Requirements**: SEC-01, SEC-02
**Success Criteria** (what must be TRUE):
  1. Requesting `--output /tmp/evil.png` returns an error and creates no file at `/tmp/evil.png`
  2. A test asserts that every path handed to the image-save functions resolves under `settings.output_dir`, rejecting both absolute paths and `..` traversal
  3. Ordinary filenames still work unchanged, including nested paths that stay inside the output directory
**Plans**: 2 plans

Plans:
- [ ] 02-01: Add path validation at the `core.py:76` join, rejecting absolute paths and `..` segments with a clear error
- [ ] 02-02: Add tests covering absolute paths, `..` traversal, and legitimate nested filenames

### Phase 3: Web Server Correctness
**Goal**: The Flask server starts correctly when run directly, and rejects malformed input before spending a provider call
**Mode**: mvp
**Depends on**: Phase 1
**Requirements**: SEC-03, SEC-04
**Success Criteria** (what must be TRUE):
  1. `python src/imagai/web_server.py` starts the server and serves the UI without raising `NameError`
  2. `POST /api/generate` with a missing or malformed prompt returns a 4xx with a JSON error body and never reaches the provider
  3. The `imagai-web` console script continues to work unchanged
**Plans**: 2 plans

Plans:
- [ ] 03-01: Move the `main()` definition above its `__main__` call in `web_server.py`
- [ ] 03-02: Add request-payload validation at the HTTP boundary returning 4xx

### Phase 4: Error Surfacing
**Goal**: Generation failures reach the caller instead of being logged and swallowed
**Mode**: mvp
**Depends on**: Phase 1
**Requirements**: ERR-01, ERR-02
**Success Criteria** (what must be TRUE):
  1. When a provider raises, `imagai generate` exits with a non-zero status and prints the failure to stderr
  2. The HTTP path returns a 5xx with an error body rather than a 200 with an empty result
  3. A provider exception produces a populated `ImageGenerationResponse.error` and no unhandled traceback escapes the orchestration boundary
**Plans**: 2 plans

Plans:
- [ ] 04-01: Catch provider exceptions at the `core.py` boundary and populate `ImageGenerationResponse.error`
- [ ] 04-02: Propagate that error to a non-zero CLI exit and a 5xx HTTP response; add tests for both surfaces

### Phase 5: Configuration Cleanup
**Goal**: Engine configuration comes solely from pydantic-settings, and importing config has no side effects
**Mode**: mvp
**Depends on**: Phase 1
**Requirements**: CFG-01, CFG-02, CFG-03
**Success Criteria** (what must be TRUE):
  1. Removing the manual `os.environ` loop changes no behavior for any documented `IMAGAI__*` variable, covered by a test over nested engine configuration
  2. `python -c "import imagai.config"` creates no directories and leaves no module-level state mutated
  3. `requires-python` states `>=3.9`, consistent with pydantic 2.11's floor, and `uv` still resolves the project on 3.12.9
**Plans**: 3 plans

Plans:
- [ ] 05-01: Add characterization tests pinning current env-var parsing behavior before removing anything
- [ ] 05-02: Delete the manual `os.environ` loop and the `mkdir` import side effect; move directory creation to an explicit call site
- [ ] 05-03: Correct `requires-python` and re-verify the full suite passes

### Phase 6: Provider Registry
**Goal**: A new provider can be added without editing orchestration code
**Mode**: mvp
**Depends on**: Phase 2 (shares `core.py`; landing the path-safety fix first keeps this refactor honest)
**Requirements**: ARCH-01, ARCH-02, ARCH-03
**Success Criteria** (what must be TRUE):
  1. Adding a provider requires only a new module plus a registry entry — `core.py` is not modified, verifiable by diffing it across the phase
  2. An engine name configured in `Settings.engines` resolves to the correct provider instance
  3. A regression test proves CLI and web generation behavior is unchanged by this refactor
**Plans**: 3 plans

Plans:
- [ ] 06-01: Add a regression test capturing current CLI and web generation behavior
- [ ] 06-02: Introduce the registry mapping engine name to provider instance; remove the hardcoded class from `core.py`
- [ ] 06-03: Prove `core.py` is untouched by provider addition and confirm the regression suite passes

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Toolchain Migration | 0/3 | Not started | - |
| 2. Output Path Safety | 0/2 | Not started | - |
| 3. Web Server Correctness | 0/2 | Not started | - |
| 4. Error Surfacing | 0/2 | Not started | - |
| 5. Configuration Cleanup | 0/3 | Not started | - |
| 6. Provider Registry | 0/3 | Not started | - |
