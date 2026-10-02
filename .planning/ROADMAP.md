# Roadmap: Imagai

## Overview

Imagai is a working prompt-to-image tool that cannot currently be run. This cycle changes no
user-facing behavior: it makes the tool installable and runnable, closes a path-traversal
correctness hole in the save pipeline, stops swallowing failures, untangles the configuration
module, and replaces the hardcoded provider in the orchestration seam with a real registry.
The order is forced by dependency, not preference — nothing can be tested or verified until the
uv toolchain exists, and nothing can be refactored safely until the error and configuration
contracts are pinned by tests.

**Milestone:** v1.0 — Harden the existing tool
**Phases:** 6 · **Requirements:** 16 v1 (100% mapped) · **Mode:** mvp

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Runnable Toolchain** - Migrate rye to uv so a clean checkout installs, runs, and tests
- [ ] **Phase 2: Path Containment** - The save pipeline can only write inside the configured output directory
- [ ] **Phase 3: HTTP Boundary** - The web server starts from source and rejects malformed payloads with 4xx
- [ ] **Phase 4: Error Propagation** - A failed generation reaches the user instead of being logged and dropped
- [ ] **Phase 5: Configuration Cleanup** - `imagai.config` becomes a pure settings declaration
- [ ] **Phase 6: Provider Registry** - A new backend is registered, not written into orchestration code

## Phase Details

### Phase 1: Runnable Toolchain
**Goal**: A developer on macOS installs, runs, and tests the project from a clean checkout using only uv
**Mode**: mvp
**Depends on**: Nothing (first phase)
**Requirements**: ENV-01, ENV-02, ENV-03, ENV-04, CFG-03
**Success Criteria** (what must be TRUE):
  1. From a clean checkout on macOS, the single uv command documented in the README produces a working `imagai` console script and a passing test suite (ENV-01)
  2. `.python-version`, `requires-python`, and the uv lockfile all name the same Python version, and `uv run python -V` reports that version (ENV-02)
  3. `README.md`, `pyproject.toml`, and the repository root contain no rye commands, no `[tool.rye]` section, and no rye lockfile — verifiable by a single `rg -n rye` returning nothing outside historical planning notes (ENV-03)
  4. A clean install's `python -c "import typing_extensions"` succeeds because `typing_extensions` is declared in `pyproject.toml` or the import was removed — not because a transitive package happened to pull it in (ENV-04)
  5. `requires-python` reads `>=3.9`, and every import declared in `pyproject.toml` is satisfiable on the pinned 3.12.9 interpreter (CFG-03) — *revised 2026-10-02: this criterion previously required resolving on a 3.9 interpreter. Python 3.9 reached EOL in October 2025, and `requirements.lock` carries zero `python_version` markers, so 3.9 was never actually verified. See `01-CONTEXT.md` D-05/D-06.*
**Plans**: 3 plans (TBD at planning)

Notes:
- One phase, not three: `pyproject.toml`, both rye lockfiles, and the README's rye commands must land together or the documented workflow stays broken.
- `requires-python` and the `typing_extensions` declaration both live in `pyproject.toml`, the same file this phase already rewrites. PROJECT.md groups this constraint with the toolchain work, not with config cleanup.
- `.venv/` on this machine is a Windows build (`home = C:\Users\aliah\...`, `Scripts/` not `bin/`). It is gitignored (`.gitignore:26`) with zero tracked files, so it only needs local deletion before the first `uv sync` — it is not a version-control issue.

### Phase 2: Path Containment
**Goal**: The image save pipeline can only write inside the configured output directory
**Mode**: mvp
**Depends on**: Phase 1
**Requirements**: SEC-01, SEC-02
**Success Criteria** (what must be TRUE):
  1. A request with `output` set to `/tmp/evil.png` is rejected with an error, and no file exists at `/tmp/evil.png` after the call — verified by a test (SEC-01)
  2. A request with `output` set to `../../escape.png` is rejected with an error, and nothing is written outside the output directory — verified by a test (SEC-01)
  3. Every path handed to `save_image_from_url` / `save_image_from_b64` resolves under `settings.output_dir`, proven by a test that exercises all four filename strategies (manual, LLM-generated, random, prompt-derived) and asserts containment for each (SEC-02)
  4. A legitimate filename such as `my_image.png` is still written to `settings.output_dir/my_image.png`, and the `n > 1` numbered variants still land in the output directory — the fix does not break the happy path (SEC-02)
**Plans**: 2 plans (TBD at planning)

Notes:
- The hole is `core.py:76` doing `Path(settings.output_dir) / current_filename` on a never-sanitized `output_filename` (`models.py:8`); an absolute path silently replaces the base. Containment belongs in the save pipeline (`utils.py`), which both front ends share, so the CLI is protected by the same check.
- Rejection must produce a populated `ImageGenerationResponse.error` (the existing contract), not an exception — Phase 4 formalizes how that error reaches the user.
- Flag for planning: `web_server.py:33` hardcodes `UPLOAD_FOLDER = Path("generated_images")` instead of reading `settings.output_dir`. SEC-02 names `settings.output_dir` as the containment root, so this divergence needs a decision in this phase.

### Phase 3: HTTP Boundary
**Goal**: The web server starts from source and rejects malformed requests before they reach the generation path
**Mode**: mvp
**Depends on**: Phase 1, Phase 2
**Requirements**: SEC-03, SEC-04
**Success Criteria** (what must be TRUE):
  1. `python src/imagai/web_server.py` starts the server and prints its startup banner without raising `NameError` (SEC-03)
  2. `POST /api/generate` with a missing `prompt` returns HTTP 400 with a JSON error body (SEC-04)
  3. `POST /api/generate` with an out-of-enum `size` such as `"10x10"`, or a non-integer `n`, returns HTTP 400 and never reaches the provider (SEC-04)
  4. `POST /api/generate` with an `output` value that escapes the output directory returns HTTP 400 and writes nothing outside it, consistent with Phase 2's rejection (SEC-04)
  5. A valid request still returns HTTP 200 with the same `results` shape as before this phase (SEC-04)
**Plans**: 3 plans (TBD at planning)

Notes:
- `web_server.py:365` calls `main()` four lines before its definition at `:368`.
- The request object is built twice in `generate_image()` (`:127-140` and `:157-170`); the second build silently discards the first's `input_image` handling order. Worth collapsing while touching this function.
- `int(data.get("n", 1))` and `float(data["strength"])` raise bare `ValueError` today, which the blanket `except Exception` at `:221` converts to a 500. SEC-04 is what turns those into 400s.
- No visual/UI design work in this phase — it is API request validation only.

### Phase 4: Error Propagation
**Goal**: A failed generation reaches the user instead of being logged and dropped
**Mode**: mvp
**Depends on**: Phase 1, Phase 2, Phase 3
**Requirements**: ERR-01, ERR-02
**Success Criteria** (what must be TRUE):
  1. When the provider raises, `imagai generate` prints a failure message and exits with a non-zero status — verified by a test asserting the exit code and the message (ERR-01)
  2. `POST /api/generate` whose generation fails returns a non-2xx status with a JSON body containing the error, rather than `"success": true` with a buried per-result `error` (ERR-01)
  3. Any exception raised inside the provider is caught at the `generate_image_core` boundary and returned as `ImageGenerationResponse(error=<message>)` — verified by a test that makes a provider raise an arbitrary exception (ERR-02)
  4. The failure path still records the traceback locally (e.g. via `logger.exception`) while the caller receives the readable message, so no information is lost (ERR-02)
  5. A failed *filename rejection* (Phase 2) and a failed *provider call* produce distinguishable, non-empty error messages (ERR-01)
**Plans**: 2 plans (TBD at planning)

Notes:
- `cli.py:225-229` prints the error but never sets a non-zero exit code — today every failure exits 0.
- `web_server.py:179-219` always returns `"success": true`; per-result `error` is set but the top-level status is 200.
- Phase 4 fixes the contracts that Phase 6's refactor must then preserve. Doing it before ARCH means `ARCH-03`'s regression test can assert a known-good error behavior instead of freezing today's swallowing.

### Phase 5: Configuration Cleanup
**Goal**: `imagai.config` becomes a pure declaration of settings, with pydantic-settings as the only source
**Mode**: mvp
**Depends on**: Phase 1
**Requirements**: CFG-01, CFG-02
**Success Criteria** (what must be TRUE):
  1. Importing `imagai.config` in a fresh process, with `output_dir` pointing at a path that does not exist, leaves that path uncreated — verified by a test that asserts the directory is absent after import (CFG-02)
  2. `config.py` contains no manual iteration over `os.environ`; every `IMAGAI__ENGINES__*` variable listed in the repository's `.env.example` produces the same parsed `settings.engines` as it does today, proven by a parametrized test covering `api_key`, `base_url`, and `model` (CFG-01)
  3. Any configuration form that pydantic-settings alone cannot express — notably engine names containing the `__` nested delimiter — is either made to work or explicitly documented as unsupported, with a test pinning the decision either way (CFG-01)
  4. No other module depends on a side effect `config.py` used to provide: `core.py`, `utils.py`, `cli.py`, and `web_server.py` still create the output directory at the moment of writing, not at import time (CFG-02)
  5. `settings.engines` remains the single source of engine configuration read by `core.py`, `cli.py`, `utils.py`, and `web_server.py` (CFG-01)
**Plans**: 2 plans (TBD at planning)

Notes:
- Blast radius exceeds `config.py`: four modules import the `settings` singleton, so this is a cross-module change, not a file-local cleanup.
- The removal target is `config.py:38-54` (the `os.environ` loop) and `config.py:56-58` (the `mkdir` at import).
- Known risk: the loop exists to work around nested-delimiter parsing for multi-word engine names like `openai_dalle3`. Removing it without the parametrized test risks a silent config regression that no existing test would catch. Success criterion 3 makes that risk explicit rather than assumed away.
- `web_server.py:34` performs its own `mkdir` at module import — a second import side effect this phase should address.

### Phase 6: Provider Registry
**Goal**: Adding an image backend is a registration, not an edit to orchestration code
**Mode**: mvp
**Depends on**: Phase 1, Phase 2, Phase 4, Phase 5
**Requirements**: ARCH-01, ARCH-02, ARCH-03
**Success Criteria** (what must be TRUE):
  1. A test-only provider class is registered by name and invoked by `generate_image_core` with no change to `core.py` — proven by a test that registers the fake provider and asserts it is called (ARCH-01)
  2. `core.py` contains no import of, or reference to, any concrete provider class — verifiable with a single `rg 'Provider' core.py` returning only registry-internal symbols (ARCH-01)
  3. The registry maps each name in `Settings.engines` to a provider instance built from that engine's `EngineConfig`, and an engine name absent from configuration still returns the existing "not configured" error (ARCH-02)
  4. A behavior snapshot captured before this phase still matches: the same request driven through the CLI and the web path produces the same saved path, filename, and injected metadata after the refactor (ARCH-03)
**Plans**: 2 plans (TBD at planning)

Notes:
- `core.py:3` imports `OpenAISDKProvider` and `core.py:28` hardcodes it. This contradicts the multi-backend framing in the README.
- `ARCH-03`'s "unchanged behavior" is scoped to *generation* outcomes (saved path, filename, metadata, error shape) — not to exit codes or HTTP status, which Phase 4 intentionally changed.
- Shares `core.py` with Phase 2 (`core.py:76`) and Phase 4 (the boundary `try`/`except`). Sequential execution keeps these edits from colliding; the path-construction code and the provider-instantiation line are separate regions.

## Cross-Phase Notes

- **Sequencing is load-bearing.** Phase 1 gates everything: no runnable test suite means no phase after it can prove its own success criteria. Phase 4 gates Phase 6, and Phase 5 gates Phase 6 (`ARCH-02` resolves through `Settings.engines`).
- **Test infrastructure is a Phase 1 output, not a Phase 1 prerequisite.** `tests/test_cli.py` contains two `assert True` placeholders — the regression tests that later phases depend on (SEC-02, ERR-01/02, ARCH-03) have to be written, not merely run.
- **Three import-time side effects** are addressed across phases: `config.py:58` mkdir (Phase 5), `web_server.py:34` mkdir (Phase 5), and the `os.environ` loop's global mutation (Phase 5).
- **Deferred, deliberately out of this milestone:** linter/formatter and CI (TOOL-01, TOOL-02), coverage beyond `tests/test_cli.py` as a standalone goal (QUAL-01, now partly absorbed by the per-phase tests above), logging configuration (QUAL-02), new backends (FEAT-01), moving Rich rendering out of the provider data layer (FEAT-02), and replacing blocking sync clients inside `async def` (FEAT-03). ARCH-01/ARCH-02 are the enablers for FEAT-01, not FEAT-01 itself.
- **`AGENTS.md` and `GEMINI.md` claim the repo is indexed in `graft/`.** A `graft/` directory does exist, but its content is unverified — do not let a missing or stale index block an implementation step.

## Progress

**Execution Order:**
Phases execute in numeric order: 1 → 2 → 3 → 4 → 5 → 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Runnable Toolchain | 0/3 | Not started | - |
| 2. Path Containment | 0/2 | Not started | - |
| 3. HTTP Boundary | 0/3 | Not started | - |
| 4. Error Propagation | 0/2 | Not started | - |
| 5. Configuration Cleanup | 0/2 | Not started | - |
| 6. Provider Registry | 0/2 | Not started | - |

Plan counts are estimates; `plan-phase` sets the real count.
