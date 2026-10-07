# Requirements: Imagai

**Defined:** 2026-10-02
**Core Value:** Fast prompt-to-image. If everything else fails, turning a prompt into an image must still work.

## v1 Requirements

This cycle hardens the existing tool rather than extending it. Nothing here adds a user-facing
capability; everything here makes the tool runnable, correct, and maintainable.

### Environment & Toolchain

- [x] **ENV-01**: A developer on macOS can install all dependencies and run the test suite with a single documented uv command
- [x] **ENV-02**: The Python version is consistent across `.python-version`, `pyproject.toml`, and the lockfile, so a clean install resolves the same interpreter the project targets
- [x] **ENV-03**: `README.md` documents the uv workflow; no rye commands, rye lockfiles, or `[tool.rye]` configuration remain
- [x] **ENV-04**: `typing_extensions` is declared in `pyproject.toml` or its import is removed, so a clean install does not depend on a transitive package

### Security & Correctness

- [ ] **SEC-01**: An `output` filename that is absolute, contains `..`, or otherwise escapes the output directory is rejected with an error rather than written
- [ ] **SEC-02**: Every path written by the image save pipeline resolves inside `settings.output_dir`, verified by a test
- [ ] **SEC-03**: `python src/imagai/web_server.py` starts the server without raising `NameError`
- [ ] **SEC-04**: HTTP endpoints validate request payloads before dispatching to core, returning 4xx on invalid input

### Error Handling

- [ ] **ERR-01**: A failed generation surfaces to the caller — non-zero CLI exit with a message, or HTTP 5xx with an error body — instead of being logged and swallowed
- [ ] **ERR-02**: Provider exceptions are caught at the orchestration boundary and converted into a populated `ImageGenerationResponse.error`

### Configuration

- [ ] **CFG-01**: Engine configuration is derived solely from pydantic-settings; the manual `os.environ` parsing loop in `config.py` is removed with no loss of supported configuration
- [ ] **CFG-02**: Importing `imagai.config` creates no directories and mutates no global state as a side effect
- [x] **CFG-03**: `requires-python` in `pyproject.toml` states the real floor (>=3.9, per pydantic 2.11) rather than the current incorrect `>=3.8`

### Architecture

- [ ] **ARCH-01**: A new provider can be registered without editing `core.py`
- [ ] **ARCH-02**: The provider registry resolves engine name to provider instance using the existing `Settings.engines` configuration
- [ ] **ARCH-03**: Existing CLI and web generation behavior is unchanged by ARCH-01 and ARCH-02, proven by a regression test

## v2 Requirements

Deferred. Tracked, not in the current roadmap.

### Tooling

- **TOOL-01**: A linter and formatter (ruff or equivalent) are configured, catching the unused imports currently present in `providers/openai_sdk_provider.py` and the undeclared `typing_extensions` import
- **TOOL-02**: CI runs the test suite on commit

### Quality

- **QUAL-01**: Test coverage extends beyond `tests/test_cli.py` to the provider, config, and utils layers
- **QUAL-02**: Three module loggers have a real logging configuration instead of none

### Features

- **FEAT-01**: Additional image backends beyond the single OpenAI-compatible provider
- **FEAT-02**: Rich rendering is moved out of the provider data layer (`openai_sdk_provider.py:249-293`) into the presentation layer
- **FEAT-03**: Blocking sync clients inside `async def` are replaced with async equivalents (`openai_sdk_provider.py:44`, `:101`)

## Out of Scope

| Feature | Reason |
|---------|--------|
| Authentication / user accounts | Web server is a localhost dev tool, not a product surface. Auth against a non-existent threat is speculative overhead |
| Multi-user or public web exposure | Would reopen the auth question and change the security model entirely |
| New image backends or models | This cycle hardens the existing surface. `ARCH-01`/`ARCH-02` are the enablers; adding backends is a separate cycle |
| Lockfile hashing / supply-chain audit | `STACK.md` flags the rye lockfiles as unhashed, but a single-developer local tool does not consume them in an automated pipeline |
| CI/CD pipeline and deployment target | Same reasoning as lockfile hashing — no consumer for it today |
| Dockerfile / containerization | The tool runs locally; there is no deployment to containerize |

## Traceability

Populated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 | Phase 1 | Complete |
| ENV-02 | Phase 1 | Complete |
| ENV-03 | Phase 1 | Complete |
| ENV-04 | Phase 1 | Complete |
| SEC-01 | Phase 2 | Pending |
| SEC-02 | Phase 2 | Pending |
| SEC-03 | Phase 3 | Pending |
| SEC-04 | Phase 3 | Pending |
| ERR-01 | Phase 4 | Pending |
| ERR-02 | Phase 4 | Pending |
| CFG-01 | Phase 5 | Pending |
| CFG-02 | Phase 5 | Pending |
| CFG-03 | Phase 1 | Complete |
| ARCH-01 | Phase 6 | Pending |
| ARCH-02 | Phase 6 | Pending |
| ARCH-03 | Phase 6 | Pending |

**Coverage:**

- v1 requirements: 16 total
- Mapped to phases: 16
- Unmapped: 0

**Phase assignments:**

- Phase 1 — Runnable Toolchain: ENV-01, ENV-02, ENV-03, ENV-04, CFG-03
- Phase 2 — Path Containment: SEC-01, SEC-02
- Phase 3 — HTTP Boundary: SEC-03, SEC-04
- Phase 4 — Error Propagation: ERR-01, ERR-02
- Phase 5 — Configuration Cleanup: CFG-01, CFG-02
- Phase 6 — Provider Registry: ARCH-01, ARCH-02, ARCH-03

> **Count correction (2026-10-02):** the initial definition of this file stated "17 total"
> v1 requirements, but only 16 REQ-IDs were defined. All 16 are mapped above — the earlier
> count was an overcount, not a missing requirement.

---
*Requirements defined: 2026-10-02*
*Last updated: 2026-10-02 after roadmap creation (traceability populated; count corrected to 16)*
