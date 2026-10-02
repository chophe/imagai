# Imagai

## What This Is

A CLI tool to generate images using various AI APIs, including OpenAI (DALL-E) and other
OpenAI-compatible services for models like Stable Diffusion, Gemini, Imagen, etc. A Flask web
server exposes the same generation path through a browser UI, and both front ends call a single
orchestration function in `core.py`.

## Core Value

Fast prompt-to-image. If everything else fails, turning a prompt into an image must still work.

## Requirements

### Validated

- ✓ CLI image generation via `imagai generate` with 18 configurable options — existing
- ✓ Web UI + JSON API over the same core, including image-to-image upload — existing
- ✓ Engine selection via env-driven config, multi-engine registry in `Settings.engines` — existing
- ✓ Four filename strategies (manual, LLM-generated, random, prompt-derived) with precedence — existing
- ✓ Image save pipeline with EXIF/PNG metadata injection (prompt + model) — existing
- ✓ `list-engines` command with per-engine `/models` probe — existing

### Active

- [ ] Migrate toolchain from rye to uv so the project actually runs (`rye` is not installed;
      `.venv/` is a non-functional Windows build; ambient Python is 3.13.11 vs pinned 3.12.9)
- [ ] Fix arbitrary file write — `output_filename` is never sanitized (`models.py:8`) and
      `core.py:76` does `Path(output_dir) / filename`, so an absolute path replaces the base
- [ ] Fix `NameError` — `web_server.py:364` calls `main()` four lines before its definition at `:368`
- [ ] Add request/response validation at the HTTP boundary
- [ ] Surface failures instead of swallowing them (broad `except Exception` paths)
- [ ] Untangle `config.py` — drop the hand-rolled `os.environ` loop that duplicates
      pydantic-settings, and remove directory creation as an import side effect
- [ ] Introduce a real provider registry so `core.py` stops hardcoding the concrete class

### Out of Scope

- **Authentication / multi-user support** — the web server is a localhost dev tool. Adding auth
  would be speculative hardening against a threat that does not exist here.
- **New image backends or models** — this cycle hardens the existing surface. Feature work
  waits until the toolchain and defects are settled.
- **Public internet exposure of the web server** — would reopen the auth question above.
- **CI/CD pipeline and lockfile hashing** — noted as debt in `STACK.md`, not needed for a
  personal dev tool.

## Context

Small, mature Python codebase: 8 source files in `src/imagai/`, 1 test file, ~2,741 lines of
codebase map in `.planning/codebase/`.

**State of the world right now: the tool does not run.** `rye` is absent from `PATH`, the
checked-in `.venv/` is a Windows build (`home = C:\Users\aliah\...`, `Scripts/` not `bin/`), and
neither `imagai` nor `flask` is importable from ambient Python. This is the Phase 1 blocker.

**Architecture is sound.** Clean layering — presentation (`cli.py`, `web_server.py`) →
orchestration (`core.py`) → provider (`providers/`) → I/O (`utils.py`), with Pydantic DTOs
crossing each boundary. The `core.py` seam genuinely is shared by both front ends, which is a
real strength worth preserving during refactoring.

**Known defects and debt** (detail in `.planning/codebase/CONCERNS.md`, 775 lines):
- The path-traversal write above, reachable via `POST /api/generate`
- The `NameError` above, masked in normal use because the `imagai-web` console script calls
  `main()` correctly
- No provider registry — `core.py` hardcodes `OpenAISDKProvider`, so a new backend means editing
  orchestration code. This contradicts the multi-backend framing in the README.
- Rich rendering leaks into the provider data layer (`openai_sdk_provider.py:249-293`)
- Blocking sync client inside `async def` (`openai_sdk_provider.py:44`, `:101`)
- Three module loggers with zero logging configuration
- `typing_extensions` imported at `cli.py:2` but undeclared in `pyproject.toml`
- No linter or formatter configured; unused imports go unnoticed
- Test coverage is one file (`tests/test_cli.py`)

**Prior work:** `docs/architecture.md`, `docs/testing.md`, `docs/dependencies.md`, and
`docs/code-quality.md` were written in an earlier session. Two claims in them were found wrong
during mapping and are corrected here: uploads *are* capped by `MAX_CONTENT_LENGTH`
(`web_server.py:36`), and `config.py`'s env loop is duplicative rather than supplementary.

## Constraints

- **Security scope**: the web server is localhost-only, so security work targets correctness
  (don't write outside the output directory) rather than remote exposure — why auth is out of scope
- **Toolchain**: migrating to uv means `pyproject.toml`, both rye lockfiles, and the README's rye
  commands all need updating together — why they're one phase, not three
- **Python version**: pinned to 3.12.9, but `pyproject.toml` declares `requires-python = ">=3.8"`
  while `pydantic 2.11` needs >=3.9 — the declared floor is wrong either way
- **Compatibility**: `pydantic-settings` and Pydantic v2 APIs are assumed throughout; the
  project is already v2-only
- **Single developer**: no CI, no deployment target, no multi-user requirement

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Migrate rye → uv | rye is not installed, so the documented workflow is unrunnable; uv is already present and much faster | — Pending |
| Harden before adding features | Core value is fast prompt-to-image; a broken runtime and a file-write bug undermine it more than a new backend would help | — Pending |
| No auth on the web server | Localhost dev tool, not a product surface. The path traversal gets fixed as a correctness bug, not framed as a remote vuln | — Pending |
| Keep the `core.py` orchestration seam | Both front ends already share it; refactoring should preserve it rather than dissolve it | — Pending |
| Add a real provider registry | `core.py` hardcodes the provider class, capping how cheaply new backends can be added and contradicting the README's multi-backend claim | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-10-02 after initialization*
