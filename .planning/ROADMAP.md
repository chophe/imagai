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

- [x] **Phase 1: Runnable Toolchain** - Migrate rye to uv so a clean checkout installs, runs, and tests (completed 2026-10-04)
- [x] **Phase 2: Path Containment** - The save pipeline can only write inside the configured output directory (completed 2026-10-07)
- [ ] **Phase 2.5: Web Server Safety** (INSERTED) - The dev server starts safely and cannot execute attacker-chosen shell
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

**Plans**: 3/3 plans executed

Plans:
**Wave 1**

- [x] 01-01-PLAN.md — Core toolchain migration: rewrite pyproject.toml, replace rye lockfiles with uv.lock, verify clean install

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 01-02-PLAN.md — Import fix + rye reference cleanup: fix typing_extensions import, update README/docs/web files to uv

**Wave 3** *(blocked on Wave 2 completion)*

- [x] 01-03-PLAN.md — End-to-end verification: grep verification + clean checkout simulation

**Cross-cutting constraints:**

- uv run python -V reports 3.12.9
- requires-python reads >=3.9

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

**Plans**: 2 plans

Plans:
**Wave 1**

- [x] 02-01-PLAN.md — Containment helper in utils.py + wire into both save functions + SEC-01 rejection tests

**Wave 2** *(blocked on Wave 1 completion)*

- [x] 02-02-PLAN.md — Unify web_server.py UPLOAD_FOLDER to settings.output_dir + SEC-02 comprehensive tests

Notes:

- The hole is `core.py:76` doing `Path(settings.output_dir) / current_filename` on a never-sanitized `output_filename` (`models.py:8`); an absolute path silently replaces the base. Containment belongs in the save pipeline (`utils.py`), which both front ends share, so the CLI is protected by the same check.
- Rejection must produce a populated `ImageGenerationResponse.error` (the existing contract), not an exception — Phase 4 formalizes how that error reaches the user.
- `web_server.py:33` hardcodes `UPLOAD_FOLDER = Path("generated_images")` — unified to `settings.output_dir` in Plan 02 (D-05).

### Phase 2.5: Web Server Safety

> **INSERTED 2026-10-07** between Phase 2 and Phase 3. Owns SEC-03, SEC-05, SEC-06.

**Goal**: The dev server starts safely from source, binds only to localhost, and cannot execute a caller-supplied shell string
**Mode**: mvp
**Depends on**: Phase 1, Phase 2
**Requirements**: SEC-03, SEC-05, SEC-06, SEC-07, SEC-08
**Success Criteria** (what must be TRUE):

  1. `POST /api/generate-cli` with `{"command": "imagai; touch /tmp/pwned"}` returns an error, no subprocess runs, and `/tmp/pwned` does not exist — verified by a test (SEC-05)
  2. `subprocess.run` in `web_server.py` is called with `shell=False` and an argv list; no call site in the repo passes `shell=True` — verified by grep (SEC-05)
  3. The allow-list is enforced on the argv tokens, so a command whose *first tokens* match an allowed prefix is rejected when later tokens contain shell metacharacters (SEC-05)
  4. `main()` defaults to `host="127.0.0.1"` and `debug=False`, so a bare `imagai-web` binds loopback only — verified by a test asserting the defaults (SEC-06)
  5. `imagai generate --help` and `POST /api/generate` still work unchanged (SEC-05)
  6. `python src/imagai/web_server.py` starts the server and prints its startup banner without raising `NameError` (SEC-03)
  7. No `CORS(app)` call remains and `flask-cors` is absent from `pyproject.toml`; a cross-origin `POST /api/generate-cli` receives no CORS grant — verified by a test (SEC-07)
  8. `POST /api/generate-cli` does not return an image that already existed in `settings.output_dir` before the call — verified by a test that seeds a file, calls the endpoint, and asserts it is absent from the response (SEC-08)

**Plans**: 3 plans

Plans:
**Wave 1**

- [ ] 02.5-01-PLAN.md — Tracer: `/api/generate-cli` runs an allow-listed argv with `shell=False`; before/after image diff; SEC-05 + SEC-08 tests
**Wave 2** *(blocked on Wave 1 — same `web_server.py`)*

- [ ] 02.5-02-PLAN.md — `main()` defined above the `__main__` guard with loopback/no-debug defaults; `CORS(app)` and the `flask_cors` import removed
**Wave 3** *(blocked on Wave 2 — its repo-wide grep gates depend on the code changes)*

- [ ] 02.5-03-PLAN.md — `flask-cors` dropped from `pyproject.toml` + `uv.lock`; the five falsified doc claims corrected

Notes:

- **Why this is urgent.** `imagai-web` resolves to `web_server:main`, whose defaults are
  `host="0.0.0.0", debug=True` (`web_server.py:368-377`). The server therefore listens on every
  interface by default, and `/api/generate-cli` (`:227`) runs `subprocess.run(command, shell=True)`
  behind a `startswith` guard that `imagai; <cmd>` bypasses. That is a network-reachable RCE by
  default, not a local correctness bug.
- **This invalidates a PROJECT.md premise.** "The web server is localhost-only, so security work
  targets correctness rather than remote exposure" was false as written. Fixing the bind address
  restores the premise, which is what makes auth-out-of-scope defensible again.
- **Why `shell=False` and not a stricter allow-list.** Enforcing the allow-list on argv tokens
  removes the shell's parsing of `;`, `|`, `&&`, and backticks entirely. A stricter regex over the
  raw string would keep re-introducing bypasses.
- **Endpoint fate — decided in discuss-phase:** `/api/generate-cli` is **kept**, argv-only. The
  bundled UI never calls it, but the curl-driven path is preserved deliberately rather than deleted
  by omission. See `02.5-CONTEXT.md` D-01/D-02.
- **CORS — decided in discuss-phase:** `GET /` serves `web_interface.html` same-origin
  (`web_server.py:39-49`) and that UI consumes only `/api/generate` and `/api/engines`, so wide-open
  CORS was never needed. `CORS(app)` and the `flask-cors` dependency are removed (SEC-07). This also
  corrects stale claims in `docs/architecture.md:6,36` and `docs/dependencies.md:21`.
- **Cross-request image leak — decided in discuss-phase:** the endpoint currently returns every image
  in `settings.output_dir` modified in the last 300s, so concurrent requests see each other's files.
  Fixed here with a before/after directory diff (SEC-08). Accepted residual: the diff still races
  under truly concurrent calls; a `threading.Lock` is the upgrade path if concurrency is ever reported.
- Werkzeug resolves to 3.1.9, so the debugger PIN-bypass CVE is patched. `debug=True` is treated
  here as information-disclosure and DoS surface, not as a second RCE.
- No visual/UI design work in this phase — it is server-execution safety only.

### Phase 3: HTTP Boundary

**Goal**: The web server rejects malformed requests before they reach the generation path
**Mode**: mvp
**Depends on**: Phase 1, Phase 2, Phase 2.5
**Requirements**: SEC-04
**Success Criteria** (what must be TRUE):

  1. `POST /api/generate` with a missing `prompt` returns HTTP 400 with a JSON error body (SEC-04)
  2. `POST /api/generate` with an out-of-enum `size` such as `"10x10"`, or a non-integer `n`, returns HTTP 400 and never reaches the provider (SEC-04)
  3. `POST /api/generate` with an `output` value that escapes the output directory returns HTTP 400 and writes nothing outside it, consistent with Phase 2's rejection (SEC-04)
  4. A valid request still returns HTTP 200 with the same `results` shape as before this phase (SEC-04)
  5. `python src/imagai/web_server.py` starts the server and prints its startup banner without raising `NameError` (inherited from Phase 2.5; regression guard only) (SEC-04)

**Plans**: 3 plans (TBD at planning)

Notes:

- `web_server.py:365` calls `main()` four lines before its definition at `:368`. (SEC-03 was moved
  to Phase 2.5, which fixes the same line as part of making the server start safely.)
- **By this phase the shell-execution hole (SEC-05) and the loopback default (SEC-06) are already
  closed** — Phase 2.5 lands first. This phase adds request-shape validation on top of a corrected
  server, so its planning reads a file whose dangerous path is already gone.
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
Phases execute in numeric order: 1 → 2 → 2.5 → 3 → 4 → 5 → 6

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Runnable Toolchain | 3/3 | Complete    | 2026-10-04 |
| 2. Path Containment | 2/2 | Complete    | 2026-10-07 |
| 2.5. Web Server Safety | 0/3 | Not started | - |
| 3. HTTP Boundary | 0/3 | Not started | - |
| 4. Error Propagation | 0/2 | Not started | - |
| 5. Configuration Cleanup | 0/2 | Not started | - |
| 6. Provider Registry | 0/2 | Not started | - |

Plan counts are estimates; `plan-phase` sets the real count.
