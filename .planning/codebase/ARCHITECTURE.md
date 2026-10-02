---
last_mapped_commit: 69cf5754b8d8021364ed445b02914b299360f67b
last_mapped_at: 2026-10-02
---
<!-- refreshed: 2026-10-02 -->

# Architecture

**Analysis Date:** 2026-10-02

## System Overview

```text
┌─────────────────────────────────────────────────────────────────┐
│                     Entry / Interface Layer                      │
├──────────────────────────────┬──────────────────────────────────┤
│   Typer CLI                  │   Flask Web Server                │
│   `src/imagai/cli.py`        │   `src/imagai/web_server.py`      │
│   `imagai` script            │   `imagai-web` script             │
└──────────────┬───────────────┴───────────────┬──────────────────┘
               │ ImageGenerationRequest        │ ImageGenerationRequest
               │ (models.py:5)                 │ (models.py:5)
               ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Application / Orchestration                     │
│                 `src/imagai/core.py`                            │
│           `generate_image_core(request) -> List[Response]`       │
│   engine lookup → filename resolution → save dispatch            │
└──────┬──────────────────────────────────────────┬───────────────┘
       │                                          │
       ▼                                          ▼
┌─────────────────────────────┐      ┌──────────────────────────────┐
│   Domain / Provider Layer    │      │      Support Layer           │
│ `src/imagai/providers/`     │      │ `src/imagai/utils.py`        │
│  base_provider.py:6 (ABC)   │      │  filenames + Pillow I/O      │
│  openai_sdk_provider.py:17  │      └───────────────┬──────────────┘
└──────────────┬──────────────┘                      │
               │ AsyncOpenAI / OpenAI SDK            │
               ▼                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│        External APIs + Filesystem Output                         │
│   OpenAI-compatible endpoints · `generated_images/`              │
└─────────────────────────────────────────────────────────────────┘

        Cross-cutting: `src/imagai/config.py` (`settings` singleton,
        imported by cli, core, utils, web_server, providers)
```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `app` (Typer) | Console-script entry; argument parsing, engine validation, Rich rendering | `src/imagai/cli.py:15` |
| `generate` | Builds `ImageGenerationRequest` from 18 CLI options; calls core | `src/imagai/cli.py:46` |
| `list_engines_command` | Renders engine table; probes `{base_url}/models` per engine | `src/imagai/cli.py:260` |
| `app` (Flask) | HTTP entry; serves UI + JSON API; CORS | `src/imagai/web_server.py:29` |
| `main` | Flask dev-server launcher (`imagai-web`) | `src/imagai/web_server.py:368` |
| `generate_image_core` | The single orchestration seam shared by CLI and web | `src/imagai/core.py:19` |
| `Settings` / `EngineConfig` | Env-driven engine registry + output dir | `src/imagai/config.py:17`, `src/imagai/config.py:7` |
| `ImageGenerationRequest` | Input DTO crossing every layer boundary | `src/imagai/models.py:5` |
| `ImageGenerationResponse` | Output DTO; carries `saved_path`/`error`/`usage` | `src/imagai/models.py:24` |
| `BaseImageProvider` | One-method async ABC for generation adapters | `src/imagai/providers/base_provider.py:6` |
| `OpenAISDKProvider` | Only concrete provider; OpenRouter chat + `images.generate` paths | `src/imagai/providers/openai_sdk_provider.py:17` |
| Filename helpers | Four filename strategies + `sanitize_filename` | `src/imagai/utils.py:18-134` |
| Image savers | Fetch/decode → Pillow → metadata injection → write | `src/imagai/utils.py:155`, `src/imagai/utils.py:191` |

## Pattern Overview

**Overall:** Layered (presentation → orchestration → provider → I/O), with a
ports-and-adapters seam at the provider boundary and Pydantic DTOs as the
contract between layers.

**Key Characteristics:**

- **One orchestration seam.** `generate_image_core` (`core.py:19`) is the *only*
  place both entry points converge. The CLI (`cli.py:221`) and the Flask API
  (`web_server.py:174`) are thin argument-translators around it.
- **Async core, sync entry.** The provider and savers are `async def`, but both
  entry points bridge with `asyncio.run` (`cli.py:224`, `web_server.py:176`),
  creating and tearing down an event loop per invocation.
- **Configuration as data, not code.** "Engines" are dictionary entries, not
  classes — a new backend is `.env` config, not a new subclass.
- **Errors are values.** Every layer boundary converts exceptions into an
  `ImageGenerationResponse.error` string; no layer raises across a boundary.
- **No registry / no factory.** `core.py:3` imports the concrete
  `OpenAISDKProvider` directly and `core.py:28` hardcodes it, so the ABC at
  `base_provider.py:6` is documentation, not dispatch.

## Layers

**Interface Layer:**

- Purpose: Translate an external request format into a domain DTO and render the result.
- Location: `src/imagai/cli.py`, `src/imagai/web_server.py`
- Contains: Typer commands, Flask routes, Rich tables/panels, JSON envelopes
- Depends on: `core.generate_image_core`, `models.ImageGenerationRequest`, `config.settings`
- Used by: end users (shell) and `web_interface.html` (browser)

**Orchestration Layer:**

- Purpose: Resolve engine → filename → output path → save, and assemble the response list.
- Location: `src/imagai/core.py`
- Contains: `generate_image_core` (109 lines, single function)
- Depends on: `config.settings`, `models`, `providers.openai_sdk_provider`, `utils`
- Used by: both interface layers only

**Provider Layer:**

- Purpose: Translate a domain request into vendor API calls and back.
- Location: `src/imagai/providers/`
- Contains: `BaseImageProvider` (ABC), `OpenAISDKProvider`
- Depends on: `models`, `config.EngineConfig`, `openai` SDK
- Used by: `core.py:28`

**Support Layer:**

- Purpose: Filename strategies, image decode/encode, metadata injection, settings.
- Location: `src/imagai/utils.py`, `src/imagai/config.py`, `src/imagai/models.py`
- Contains: pure functions plus the `settings` singleton
- Depends on: `pydantic`, `pydantic-settings`, `Pillow`, `httpx`, `openai`
- Used by: every other layer

## Data Flow

### Primary Request Path

CLI `imagai generate` — the path both entry points share from `core` onward:

1. Console script `imagai` resolves to `imagai.cli:app` (`pyproject.toml:41`).
2. Typer dispatches to `generate()` and parses 18 `Annotated` options (`cli.py:46-160`).
3. Engine resolved as `--engine` or `settings.default_engine`; missing → `typer.Exit(code=1)` (`cli.py:161-171`).
4. If `--prompt` is absent, `sys.stdin.read()` strips ASCII control chars incl. Windows Ctrl+Z (`cli.py:173-179`).
5. `ImageGenerationRequest` built; provider-specific keys packed into `extra_params`, dropping `None` (`cli.py:193-218`).
6. `asyncio.run(_generate())` invokes `generate_image_core(request)` (`cli.py:223-224`).
7. Engine name validated against `settings.engines`; miss returns a single error response (`core.py:23-26`).
8. `OpenAISDKProvider(engine_config)` constructed — creates `AsyncOpenAI` in `__init__` (`core.py:28`, `openai_sdk_provider.py:25`).
9. `await provider.generate_image(request)` (`core.py:31`) branches on provider (`openai_sdk_provider.py:33`):
   - **OpenRouter + Gemini** → sync `client.chat.completions.create` with `extra_body.modalities=["image","text"]` (`openai_sdk_provider.py:43-106`); data-URL content parsed into `image_b64_json` (`:132`) or `text_content` (`:156`).
   - **Otherwise** → `await self.async_client.images.generate(**kwargs)` (`openai_sdk_provider.py:234`); DALL-E 3 adds `quality`/`style` (`:186-188`); Stability strips `n`/`response_format` and forwards `extra_body` (`:190-224`).
10. Filename chosen by precedence: `--output` → `--auto-filename` (LLM) → `--random-filename` → prompt-truncated default (`core.py:45-75`). When `n > 1`, a `_{i+1}` suffix is appended in each branch.
11. Output path built as `Path(settings.output_dir) / current_filename` (`core.py:76`).
12. Save dispatched on payload type: URL first, then b64 (`core.py:83-93`) → Pillow open → `_inject_metadata` (EXIF 270/305 for JPEG, `PngInfo` for PNG) → write (`utils.py:163-173`, `utils.py:200-207`).
13. `saved_path` set, or `error` synthesized as `"Failed to save image to …"` (`core.py:94-99`).
14. `finally` closes the provider via `await provider.close()` (`core.py:106-108`).
15. CLI renders one `Panel` per result, preferring `saved_path` → `image_url` → `image_b64_json` (`cli.py:225-257`).

### Web Request Path

1. `imagai-web` resolves to `imagai.web_server:main` (`pyproject.toml:42`).
2. Flask dev server binds `0.0.0.0:5000` with `debug=True` default (`web_server.py:368-377`).
3. `POST /api/generate` validates `prompt` and `engine` (`web_server.py:83-104`); extra params coerced with bare `int()`/`float()` (`:119-124`).
4. `ImageGenerationRequest` constructed **twice** — the second overwrites the first after `input_image` is decoded into `extra_params` (`web_server.py:127-140`, `web_server.py:157-170`).
5. `asyncio.run(generate_image_core(image_request))` — identical to CLI step 6 onward (`web_server.py:173-176`).
6. Each result is serialized and the saved file re-encoded as an inline `data:` URI for preview (`web_server.py:185-217`).

`web_interface.html` calls only `/api/generate` and `/api/engines`. The routes
`/api/generate-cli` (`web_server.py:227`), `/api/images` (`:317`) and
`/api/images/<filename>` (`:307`) are not reachable from the shipped UI.

### Model-Listing Path

`imagai list-engines` reads only `settings.engines` locally, then per engine
tries the `OpenAI` client `models.list()` and falls back to a plain
`requests.get("{base_url}/models")` (`cli.py:329-388`), filtering to
image-like model IDs by substring match unless `--all` (`cli.py:310-326`).

**State Management:**

- No database, no ORM, no persistence layer.
- The engine registry *is* the state: `settings.engines` (`config.py:31`),
  built once at import from `.env` + `os.environ`.
- Mutable module-level singletons: `settings` (`config.py:36`), `console`
  (`cli.py:20`), `app` (`cli.py:15`), `app`/`UPLOAD_FOLDER` (`web_server.py:29`, `:33`).
- Per-request state is passed explicitly as the `ImageGenerationRequest` DTO.

## Key Abstractions

**`ImageGenerationRequest` / `ImageGenerationResponse`:**

- Purpose: The transport-neutral contract between interface, orchestration, and provider layers.
- Examples: `src/imagai/models.py:5`, `src/imagai/models.py:24`
- Pattern: Anemic Pydantic `BaseModel` DTO pair. Validation lives here
  (`Literal` sizes at `:9-11`, `n` bounds `ge=1, le=10` at `:13-15`). The
  response is *mutated in place* by core — `core.py:95` sets `saved_path` and
  `core.py:97` sets `error` on an object the provider already returned.
- Extension point: `extra_params: Optional[dict]` (`models.py:18`) is the
  untyped escape hatch carrying every provider-specific key.

**`BaseImageProvider`:**

- Purpose: Formal contract for a generation backend.
- Examples: `src/imagai/providers/base_provider.py:6`
- Pattern: Single-method ABC —
  `async def generate_image(request) -> List[ImageGenerationResponse]`
  (`base_provider.py:8-10`). One implementation returns `List` because a single
  API call yields `n` images.
- Note: `close()` (`:300`) is called via `hasattr` duck-typing in `core.py:107`
  because it is **not** declared on the ABC — a second provider without `close`
  still works, but the contract is implicit.

**`EngineConfig` / `Settings`:**

- Purpose: Declarative engine registry — the primary extension mechanism.
- Examples: `src/imagai/config.py:7`, `src/imagai/config.py:17`
- Pattern: `Dict[str, EngineConfig]` with `env_prefix="IMAGAI__"` and
  `env_nested_delimiter="__"` (`config.py:20-21`). Engine names are lowercased
  (`config.py:42`). Verified: `pydantic-settings` alone resolves
  `IMAGAI__ENGINES__OPENAI_GPT__API_KEY` to `engines["openai_gpt"].api_key`.

## Entry Points

**`imagai` (console script):**

- Location: `src/imagai/cli.py:15` (declared `pyproject.toml:41`)
- Triggers: shell invocation `imagai generate|list-engines|--version`
- Responsibilities: parse args, resolve engine, build request, drive the event
  loop, render Rich output, set exit codes (`cli.py:171`, `cli.py:188`).

**`imagai-web` (console script):**

- Location: `src/imagai/web_server.py:368` (declared `pyproject.toml:42`)
- Triggers: `imagai-web` → `app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)` (`web_server.py:377`)
- Responsibilities: serve `web_interface.html` at `/`, expose 5 JSON routes, run core per request.

**Flask dev server (indirect):**

- Location: `src/imagai/web_server.py:364-365`
- Note: `if __name__ == "__main__": main()` sits **above** the `main` definition
  (`web_server.py:368`), so `python -m imagai.web_server` raises `NameError`.
  Only the `imagai-web` console script works.

## Architectural Constraints

- **Threading / event loop:** Single-threaded per invocation. Async everywhere
  below the interface layer, but each entry point calls `asyncio.run`
  (`cli.py:224`, `web_server.py:176`), so there is no ambient/reusable loop and
  no nested-loop protection. Flask's `threaded=True` (`web_server.py:377`) means
  each request thread builds its own loop.
- **Global state:** `settings` (`config.py:36`), `console` (`cli.py:20`),
  Typer `app` (`cli.py:15`), Flask `app` (`web_server.py:29`), `UPLOAD_FOLDER`
  (`web_server.py:33`). `settings` is read (never written) by cli, core, utils,
  web_server, and the provider — it is the de-facto shared context object.
- **Import-time side effects:** `Path(settings.output_dir).mkdir(parents=True,
  exist_ok=True)` runs on *any* import of `config` (`config.py:58`), including
  under pytest; `UPLOAD_FOLDER.mkdir(exist_ok=True)` on import of
  `web_server` (`web_server.py:34`); `sys.path.insert(0, ...)` at
  `web_server.py:23` mutates `sys.path` before the `imagai` imports at `:25-27`.
- **Circular imports:** None. The import graph is a strict DAG —
  `config` → `models` → `base_provider` → `openai_sdk_provider` → `utils` →
  `core` → `{cli, web_server}`. Verified by full-repo grep; all intra-package
  imports are absolute (`from imagai.…`), never relative.
- **CWD dependence:** Three separate CWD-relative resolutions —
  `env_file=".env"` (`config.py:19`), `open("web_interface.html")`
  (`web_server.py:43`), and `Path("generated_images")` (`web_server.py:33`) —
  while the writer uses `settings.output_dir` (`core.py:76`). If those ever
  diverge, the web server saves where CLI does not.
- **Provider count is fixed at compile time:** no registry or entry-point
  lookup. Adding a non-OpenAI backend requires a code edit at `core.py:3` and
  `core.py:28`.

## Anti-Patterns

### Shell-Executed User Command Over HTTP

**What happens:** `POST /api/generate-cli` accepts an arbitrary `command` string
and runs `subprocess.run(command, shell=True, capture_output=True, text=True,
timeout=300, cwd=os.getcwd())` (`web_server.py:248-255`). The only guard is a
string-prefix allowlist (`web_server.py:239-241`).

**Why it's wrong here:** `startswith` does not neutralize shell metacharacters,
so `imagai; <anything>` or `imagai && curl …` passes the check and executes.
The endpoint also duplicates `/api/generate` and is never called by
`web_interface.html`.

**Do this instead:** Delete the endpoint. If it must stay, use
`shlex.split(command)` and pass the resulting argv list with `shell=False`,
re-validating each token against an allowlist of subcommands.

### HTTP Transport Concerns Inside the Domain Layer

**What happens:** `OpenAISDKProvider.generate_image` builds a Rich `Console`
and `Table` and prints an "API Usage & Cost Info" report
(`openai_sdk_provider.py:249-293`) from inside the provider.

**Why it's wrong here:** The provider is shared by the JSON API path
(`web_server.py:176`), which never wants Rich markup on stdout, and by any
future library consumer. Presentation is bound to the async generation call.

**Do this instead:** Return the data — `usage` and `estimated_cost` are already
fields on `ImageGenerationResponse` (`models.py:29-30`) and are already
populated at `openai_sdk_provider.py:245-246`. Render them in `cli.py` only.

### Blocking Client Inside `async def`

**What happens:** On the OpenRouter/Gemini path a **synchronous** `OpenAI(...)`
client is constructed (`openai_sdk_provider.py:44`) and
`client.chat.completions.create(...)` is called without `await`
(`openai_sdk_provider.py:101`) inside `async def generate_image`.

**Why it's wrong here:** The whole request blocks the event loop. `self.async_client`
exists (`openai_sdk_provider.py:25`) but is unused on this branch.

**Do this instead:** `completion = await self.async_client.chat.completions.create(...)`
and drop the second client entirely.

### Hand-Rolled Env Parsing Beside a Settings Library

**What happens:** `config.py:38-54` re-scans `os.environ` for
`IMAGAI__ENGINES__*` and `setattr`s fields — duplicating what
`env_nested_delimiter="__"` (`config.py:21`) already does. Verified that
pydantic-settings alone resolves the same keys correctly.

**Why it's wrong here:** Two sources of truth for one setting. Engine names are
silently `lower()`ed (`config.py:42`), `HttpUrl` coercion is wrapped in
`try/except: pass` so a bad URL leaves a raw `str` in a typed field
(`config.py:49-51`), and `config.py:53-54` is unreachable — `api_key` already
satisfies the `hasattr` branch at `config.py:46`.

**Do this instead:** Delete `config.py:38-54` and rely on
`pydantic-settings`. Move the `mkdir` at `config.py:58` into `cli.main_callback`
and `web_server.main`.

### Duplicated Request Construction

**What happens:** `web_server.py:127-140` and `web_server.py:157-170` build
byte-identical `ImageGenerationRequest` objects; the second assignment silently
discards the first.

**Why it's wrong here:** Two 14-line blocks that must be edited in lockstep
whenever a field is added — a guaranteed source of drift.

**Do this instead:** Finish populating `extra_params` (including `input_image`),
then construct `ImageGenerationRequest` exactly once.

## Error Handling

**Strategy:** Errors are converted to data at every boundary; no exception
crosses a layer boundary. The root cause is logged and then frequently dropped
from the response.

**Patterns:**

- **Provider boundary** — broad catch returns a one-element error list:
  `return [ImageGenerationResponse(error=str(e))]` (`openai_sdk_provider.py:296-298`).
- **Orchestration** — per-response `error` short-circuits (`core.py:33-35`);
  outer catch returns one error response (`core.py:101-105`); `finally` closes
  the provider (`core.py:106-108`).
- **Save failures** — `save_image_from_url` / `save_image_from_b64` return
  `None` on every failure path after `logger.error` (`utils.py:180`, `:188`,
  `:212`, `:215`). Core then discards the root cause and reports only
  `f"Failed to save image to {output_file_path}"` (`core.py:97-99`).
- **CLI** — `raise typer.Exit(code=1)` for config errors (`cli.py:171`, `:188`);
  per-image errors printed inline and the loop continues (`cli.py:226-229`).
- **Web** — `(jsonify({...}), status)` pairs throughout; global 404/500 handlers
  (`web_server.py:354-361`); the generic `except Exception` returns `str(e)`
  and `type(e).__name__` to the client (`web_server.py:221-224`), leaking internals.
- **No logging configuration anywhere.** `logger = logging.getLogger(__name__)`
  exists at `core.py:16`, `utils.py:15`, `openai_sdk_provider.py:14`, but no
  `logging.basicConfig` is ever called — every `logger.error` is invisible
  unless the embedding application configures logging.
- **Silent-drop validation:** `extra_params` is untyped (`models.py:18`), so a
  key like `aspect-ratio` (hyphen) is accepted and ignored by
  `openai_sdk_provider.py:209-212`.

## Cross-Cutting Concerns

**Logging:** Mixed and inconsistent. Stdlib `logging` module loggers for real
errors (`core.py:16`, `utils.py:15`, `openai_sdk_provider.py:14`) — never
configured. Bare `print()` for `--verbose` payloads (`utils.py:84-90`,
`utils.py:99-105`, `openai_sdk_provider.py:84-99`, `openai_sdk_provider.py:116-122`)
and Rich tables for usage output. Note `utils.py` uses `json` at `:101` while
the only `import json` is function-local at `utils.py:87` — the non-verbose
response-printing path would raise `NameError`.

**Validation:** Pydantic owns it. `ImageGenerationRequest` constrains `size` to
DALL-E literals (`models.py:9-11`), `quality`/`style`/`response_format`
(`:12`, `:16`, `:17`), and `n` to 1–10 (`:13-15`). `EngineConfig` types
`base_url` as `HttpUrl` (`config.py:9-11`). The `size` Literal is
DALL-E-specific, which is why the Stability path deletes `size` when
`aspect_ratio` is present (`openai_sdk_provider.py:220-221`).
`response_format` defaults disagree: `"url"` in `models.py:17` vs `"b64_json"`
in `cli.py:86` and `web_server.py:135`.

**Authentication:** No user identity or session concept. Auth is per-engine
static credentials: `EngineConfig.api_key` (`config.py:8`), sourced from
`IMAGAI__ENGINES__<NAME>__API_KEY`. Credential presence is reported as
`"✅ Set"` / `"⚠️ Not Set / Default"` by `cli.py:285-289`. A `.env` file exists
at the repo root (gitignored, contents never read by tooling); `.env.example`
documents the variable names. No secret ever reaches a log or response — but
`str(e)` in the Flask handlers (`web_server.py:222-224`) can surface an
SDK exception that embeds a request URL.

---

*Architecture analysis: 2026-10-02*
