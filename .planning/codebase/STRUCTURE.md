---
last_mapped_commit: 69cf5754b8d8021364ed445b02914b299360f67b
last_mapped_at: 2026-10-02
---
# Codebase Structure

**Analysis Date:** 2026-10-02

## Directory Layout

```
imagai/
├── src/
│   └── imagai/                  # The shipped Python package (src-layout)
│       ├── __init__.py          # __version__ = "0.1.0" + hello() stub
│       ├── cli.py               # Typer app: `generate`, `list-engines`
│       ├── core.py              # generate_image_core — the orchestration seam
│       ├── config.py            # Settings/EngineConfig + `settings` singleton
│       ├── models.py            # ImageGenerationRequest / ImageGenerationResponse
│       ├── utils.py             # Filename strategies + Pillow image saving
│       ├── web_server.py        # Flask app + `imagai-web` launcher
│       └── providers/
│           ├── __init__.py      # Empty (0 bytes) — no provider registry
│           ├── base_provider.py       # BaseImageProvider ABC
│           └── openai_sdk_provider.py # OpenAISDKProvider — only implementation
├── tests/
│   ├── __init__.py              # Empty, makes tests a package
│   └── test_cli.py              # 2 placeholder tests (assert True)
├── docs/                        # Prior hand-written analysis
│   ├── architecture.md
│   ├── code-quality.md
│   ├── dependencies.md
│   └── testing.md
├── generated_images/            # Default output dir; created at import; gitignored
├── web_interface.html           # 33 KB single-file browser UI (no build step)
├── pyproject.toml               # Hatchling build + [project.scripts] entry points
├── requirements.lock            # Rye lock (runtime)
├── requirements-dev.lock        # Rye lock (dev)
├── .python-version              # 3.12.9
├── .env                         # Real credentials — gitignored, never read
├── .env.example                 # Documented variable names (safe to read)
├── .gitignore
├── .venv/                       # Rye venv (note: Windows-style Scripts/ layout)
└── .planning/
    └── codebase/                # These GSD analysis documents
```

## Directory Purposes

**`src/imagai/`:**

- Purpose: The entire installable package; the only directory shipped in the wheel.
- Contains: 6 flat modules + the `providers/` subpackage.
- Key files: `cli.py:15` (Typer entry), `core.py:19` (orchestration),
  `config.py:36` (`settings` singleton), `models.py:5`/`:24` (DTOs).
- Note: `[tool.hatch.build.targets.wheel] packages = ["src/imagai"]` (`pyproject.toml:37-38`).

**`src/imagai/providers/`:**

- Purpose: Pluggable generation backends behind `BaseImageProvider`.
- Contains: one ABC and one implementation.
- Key files: `base_provider.py:6`, `openai_sdk_provider.py:17`.
- Note: `providers/__init__.py` is empty — nothing is re-exported, so consumers
  import full paths (`from imagai.providers.openai_sdk_provider import OpenAISDKProvider`,
  `core.py:3`).

**`tests/`:**

- Purpose: pytest suite.
- Contains: `__init__.py` and `test_cli.py`.
- Note: no `conftest.py`, no `[tool.pytest.ini_options]` in `pyproject.toml`,
  no fixtures directory.

**`docs/`:**

- Purpose: Human-authored design/analysis notes, written before this map.
- Contains: `architecture.md` (claims verified — see below), `code-quality.md`,
  `dependencies.md`, `testing.md`.
- These are advisory; `README.md` at the root is the user-facing doc.

**`generated_images/`:**

- Purpose: Default image output directory (48 files present locally).
- Contains: PNG/JPEG outputs with injected prompt+model metadata.
- Created at import time by `config.py:58`; gitignored (`.gitignore` final line).

**`.planning/codebase/`:**

- Purpose: GSD-generated codebase maps consumed by plan/execute phases.
- Contains: `ARCHITECTURE.md`, `STRUCTURE.md`, and sibling focus docs.

## Key File Locations

**Entry Points:**

- `src/imagai/cli.py:15` — Typer `app`, bound as `imagai` (`pyproject.toml:41`)
- `src/imagai/cli.py:46` — `generate` command (the main user action)
- `src/imagai/cli.py:260` — `list-engines` command
- `src/imagai/web_server.py:29` — Flask `app` instance
- `src/imagai/web_server.py:368` — `main()`, bound as `imagai-web` (`pyproject.toml:42`)
- `web_interface.html` — browser client; calls only `/api/generate` and `/api/engines`

**Configuration:**

- `pyproject.toml` — deps, hatchling build, console scripts, rye dev-deps
- `src/imagai/config.py:17` — `Settings` (env prefix `IMAGAI__`, nested `__`)
- `src/imagai/config.py:7` — `EngineConfig` (`api_key`, `base_url`, `model`)
- `src/imagai/config.py:38-54` — redundant manual env-var scan (verified duplicative)
- `.env.example` — engine variable names: `IMAGAI__ENGINES__<NAME>__{API_KEY,BASE_URL,MODEL}`
- `.python-version` — pins 3.12.9 (contradicts `requires-python = ">=3.8"`, `pyproject.toml:22`)
- `.gitignore` — excludes `.env`, `.venv`, `generated_images/`, `__pycache__/`

**Core Logic:**

- `src/imagai/core.py:19` — `generate_image_core`; engine lookup (`:23`), provider
  construction (`:28`), filename precedence (`:45-75`), save dispatch (`:83-93`),
  cleanup (`:106-108`)
- `src/imagai/providers/openai_sdk_provider.py:27` — `generate_image`;
  OpenRouter chat branch (`:43-176`), `images.generate` branch (`:178-295`)
- `src/imagai/utils.py:26` — `generate_filename_from_prompt_llm` (engine priority
  `filename_generation` → `default_engine` → first engine matching `"openai"`)
- `src/imagai/utils.py:126` / `:119` — `generate_filename` / `generate_random_filename`
- `src/imagai/utils.py:155` / `:191` — `save_image_from_url` / `save_image_from_b64`
- `src/imagai/utils.py:137` — `_inject_metadata` (EXIF tags 270/305 for JPEG; `PngInfo` for PNG)
- `src/imagai/utils.py:218` — `get_image_extension` allowlist: jpg, jpeg, png, gif, webp
- `src/imagai/web_server.py:77` — `POST /api/generate`

**Testing:**

- `tests/test_cli.py` — only test file; 6 lines, both bodies are `assert True`
- No configured coverage tooling, no lint/format config (no `ruff`, `black`, `mypy`, `isort`)

**Documentation:**

- `README.md` — setup, Rye workflow, all four filename modes, filename-engine config
- `docs/architecture.md` — verified accurate; see "Verified prior analysis" below

## Naming Conventions

**Files:**

- `snake_case.py`: `openai_sdk_provider.py`, `base_provider.py`, `web_server.py`
- Flat module names inside the package — no subdirectories beyond `providers/`
- Tests mirror the module under test: `tests/test_cli.py` ← `src/imagai/cli.py`
- Docs are lowercase topic slugs: `docs/code-quality.md`, `docs/dependencies.md`

**Functions and variables:**

- `snake_case` for all functions: `generate_image_core`, `save_image_from_url`
- Private helpers take a single leading underscore, defined next to their caller:
  `_inject_metadata` (`utils.py:137`), `_is_image_model` (`cli.py:310`),
  `_generate` (`cli.py:220`, `web_server.py:173`)
- Async coroutines take **no** `_async`/`_coro` suffix — `generate_image`,
  `save_image_from_b64`, `close`
- No trailing-underscore name mangling anywhere in the codebase

**Classes:**

- `PascalCase`; Pydantic models use bare noun phrases with **no** `Model`/`DTO`/`Schema` suffix:
  `ImageGenerationRequest`, `ImageGenerationResponse`, `EngineConfig`, `Settings`,
  `OpenAISDKProvider`, `BaseImageProvider`
- ABCs are prefixed `Base`: `BaseImageProvider` (`base_provider.py:6`)
- Module-level constants are `UPPER_SNAKE`: `UPLOAD_FOLDER`, `MAX_CONTENT_LENGTH` (`web_server.py:33-36`)
- **Exception:** the two framework app objects are lowercase module globals —
  `app` (`cli.py:15`, `web_server.py:29`) and `console` (`cli.py:20`)

**CLI and HTTP naming:**

- Typer commands use kebab-case: `@app.command(name="list-engines")` (`cli.py:260`);
  short flags double letters only where mnemonic (`-n` for `--num-images`)
- Option names are `snake_case` long-form (`--negative-prompt`, `--aspect-ratio`,
  `--auto-filename`) — these map 1:1 to `extra_params` keys
- Flask routes are `/api/<plural-noun>`: `/api/engines`, `/api/generate`, `/api/images`

**Environment variables:**

- `UPPER_SNAKE` with `IMAGAI__` prefix and `__` nesting:
  `IMAGAI__ENGINES__<ENGINE_NAME>__{API_KEY,BASE_URL,MODEL}`
- Engine names are matched **case-insensitively** and lowercased at load
  (`config.py:42`), so `OPENAI_GPT` in `.env` becomes `openai_gpt` in `settings.engines`
- Single underscores inside engine names are safe; the `__` delimiter means
  `OPENAI_GPT` splits cleanly as `['IMAGAI','ENGINES','OPENAI_GPT','API_KEY']`
  (verified against `pydantic-settings`)

## Where to Add New Code

**New Feature (end-to-end vertical slice):**

- Primary code: `src/imagai/core.py` (orchestration) — the seam both entry points share
- Interface: `src/imagai/cli.py` (add `@app.command()` or an option to
  `generate` at `cli.py:46`) **and/or** `src/imagai/web_server.py` (add `@app.route`)
- Contract: `src/imagai/models.py` (add the field to `ImageGenerationRequest`,
  or pass it through the untyped `extra_params` at `models.py:18` if provider-specific)
- Tests: `tests/test_core.py` (matches `src/imagai/core.py`)

**New CLI command:**

- Add to `src/imagai/cli.py` with `@app.command()`; name it kebab-case
  (`@app.command(name="list-engines")`, `cli.py:260`). No registration file to
  update — Typer discovers decorators on the `app` at `cli.py:15`.

**New provider / backend:**

- Create `src/imagai/providers/<name>_provider.py`, subclass `BaseImageProvider`,
  implement `async def generate_image(request) -> List[ImageGenerationResponse]`
  (`base_provider.py:8-10`). Return `ImageGenerationResponse(error=...)` rather
  than raising (`openai_sdk_provider.py:296-298`).
- **You must also edit `core.py`:** import at `core.py:3` and instantiate at
  `core.py:28`. There is no registry, factory, or entry-point lookup — a new
  provider is invisible until `core.py` is changed.
- Optional: add `async def close(self)` — `core.py:107` duck-types it with
  `hasattr`, so it is not required but avoids leaked clients.

**New engine (a model/vendor, not code):**

- Configuration only. Add `IMAGAI__ENGINES__<NAME>__API_KEY` /
  `__BASE_URL` / `__MODEL` to `.env` (document in `.env.example`). No code change
  required if the endpoint is OpenAI-compatible — `OpenAISDKProvider` handles
  DALL-E 3, Stability, and Imagen by branching on `self.config.model`
  (`openai_sdk_provider.py:186`, `:190`) and `base_url` (`:36`).
- Add a branch in `openai_sdk_provider.py` only if the vendor needs
  request-shaping (`extra_body`, header injection) that no existing branch covers.

**New provider-specific request parameter:**

- Add a CLI option in `src/imagai/cli.py` (pattern at `cli.py:103-151`), then pack
  it into `extra_params` in the dict comprehension at `cli.py:202-214`.
- Consume it in `src/imagai/providers/openai_sdk_provider.py` via
  `request.extra_params.get("<key>")` (pattern at `:209-212` or `:59-60`).
- Mirror it in the Flask `param_mapping` dict (`web_server.py:108-115`) and in
  the two `ImageGenerationRequest` blocks (`web_server.py:127-140`, `:157-170`).
- If it should be typed rather than free-form, add it to
  `ImageGenerationRequest` in `src/imagai/models.py` instead.

**New filename strategy:**

- Add the generator to `src/imagai/utils.py` (`generate_filename`,
  `generate_random_filename`, `generate_filename_from_prompt_llm` are at
  `:126`, `:119`, `:26`), then add a branch to the precedence chain in
  `core.py:45-75` — order is `--output` → `--auto-filename` →
  `--random-filename` → default.

**New image-save path (e.g. raw bytes, streaming, S3):**

- Add the saver to `src/imagai/utils.py` following the
  `save_image_from_url` (`:155`) / `save_image_from_b64` (`:191`) shape:
  return `Optional[Path]`, `logger.error` and return `None` on every failure.
- Dispatch it from `core.py:83-93`, which currently tests `image_url` then
  `image_b64_json`.
- Reuse `_inject_metadata` (`utils.py:137`) rather than writing metadata logic again.

**New configuration field:**

- Per-engine field: add to `EngineConfig` in `src/imagai/config.py:7-14`.
- Global field: add to `Settings` in `src/imagai/config.py:17-33` — it is picked
  up from `IMAGAI__<FIELD>` automatically.
- Caveat: the manual env loop at `config.py:38-54` only forwards keys that already
  exist on `EngineConfig` (`hasattr` check at `config.py:46`); new fields work
  through pydantic-settings regardless.

**New HTTP endpoint:**

- Add to `src/imagai/web_server.py` with `@app.route`. For any endpoint that
  generates an image, build one `ImageGenerationRequest` and call
  `asyncio.run(generate_image_core(...))` (`web_server.py:173-176`) — do **not**
  shell out (see the anti-pattern in `ARCHITECTURE.md`).
- Return the `{"success": bool, ...}` envelope used by every existing route
  (`web_server.py:66-74`, `:179-183`).
- Wire the browser side by editing `web_interface.html` directly — it is a
  single unbuilt file, so no bundler or asset pipeline is involved.

**Utilities / shared helpers:**

- `src/imagai/utils.py` is the only shared-helper module. Put new pure helpers
  there and prefix with `_` if module-private.
- Do not import from `cli.py` or `web_server.py` — both are top-layer entry
  points and neither is a library module.

**Tests:**

- `tests/test_<module>.py`, mirroring the module name (`test_cli.py` ← `cli.py`).
  Add `tests/__init__.py`-compatible imports; there is no `conftest.py`, so
  shared fixtures must be defined in a new `tests/conftest.py`.
- Highest-value targets (pure, no network): `sanitize_filename` (`utils.py:18`),
  `generate_filename` (`utils.py:126`), `get_image_extension` (`utils.py:218`),
  and the `_inject_metadata` behavior (`utils.py:137`).

**Documentation:**

- `docs/<topic>.md` for analysis notes; `README.md` for anything a user needs.
- `web_interface.html` for any UI change.

## Special Directories

**`src/imagai/__pycache__/` and `src/imagai/providers/__pycache__/`:**

- Purpose: CPython bytecode cache.
- Generated: Yes (automatically)
- Committed: No — gitignored (`.gitignore` line 2) and verified absent from
  `git ls-files`. Present on disk from local runs.

**`generated_images/`:**

- Purpose: Default output directory for generated images; also the Flask
  `UPLOAD_FOLDER` (`web_server.py:33`).
- Generated: Yes — created at import time by `config.py:58`
  (`Path(settings.output_dir).mkdir(parents=True, exist_ok=True)`).
- Committed: No — final line of `.gitignore`. 48 files present locally.

**`.venv/`:**

- Purpose: Rye-managed virtual environment.
- Generated: Yes (`rye sync`).
- Committed: No. **Note the layout:** it uses `Scripts/` and `Lib/site-packages/`
  (Windows convention) rather than `bin/` and `lib/`, and `.venv/bin/python`
  does not exist. `rye` is not currently on `PATH`. Use `uv` or a system
  interpreter when scripting against this environment.

**`.env`:**

- Purpose: Real per-engine credentials, loaded by `Settings` via
  `env_file=".env"` (`config.py:19`). Also drives the `os.environ` scan at
  `config.py:38-54`.
- Generated: Manually, by copying `.env.example`.
- Committed: No — gitignored (`.gitignore` under `# Environment`).
- **Never read or quote this file.** Note its existence only; use
  `.env.example` (variable *names* only) for documentation.

**`.planning/`:**

- Purpose: GSD planning state and codebase maps.
- Contains: `codebase/ARCHITECTURE.md`, `codebase/STRUCTURE.md`.
- Generated: Yes (by GSD commands). Committed: Yes.

### Verified prior analysis

`docs/architecture.md` was checked against source line-by-line and is **accurate**:
the directory tree, component descriptions, entry points
(`imagai` → `imagai.cli:app`, `imagai-web` → `imagai.web_server:main`),
filename precedence order (`--output` > `--auto-filename` > `--random-filename` >
default, `core.py:45-75`), the `filename_generation` → `default_engine` →
first-`openai`-named-engine fallback (`utils.py:35-48`), and the shared-core
claim for CLI and web all hold.

Two refinements: it describes the `config.py` manual env scan as supplementing
pydantic-settings (it duplicates it — verified redundant), and it does not
mention that `web_interface.html` only exercises `/api/generate` and
`/api/engines`, leaving three Flask routes unused by the shipped UI.

---

*Structure analysis: 2026-10-02*
