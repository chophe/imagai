# Code Quality Review

Sampled: `src/imagai/cli.py`, `core.py`, `config.py`, `models.py`, `utils.py`,
`web_server.py`, `providers/openai_sdk_provider.py`, `providers/base_provider.py`, `tests/test_cli.py`.

## Strengths

- Clear layering: CLI (`cli.py`) → orchestration (`core.py`) → provider (`providers/`) → I/O helpers (`utils.py`). Easy to trace a request end to end.
- Pydantic models (`models.py`, `config.py:EngineConfig`) give real validation (e.g. `n` bounds, literals for `size`/`quality`/`style`).
- Provider abstraction (`providers/base_provider.py`) is minimal and correct — one method, no speculative interface.
- CLI error messages are user-friendly (missing engine, unconfigured engine with available-engine hint, `cli.py:161-188`).

## Issues

### 1. Oversized functions doing 3+ jobs
- `cli.py:list_engines_command` (`cli.py:260-417`, ~160 lines): table rendering + OpenAI client construction + HTTP fallback + model filtering + error panels all in one command function.
- `providers/openai_sdk_provider.py:generate_image` (`:27-301`, ~270 lines): OpenRouter chat path *and* `images.generate` path plus Rich table rendering in the data layer (`:249-293`). Presentation (`Console`/`Table`) does not belong in a provider.
- `cli.py:generate` (`:46-257`): 115-line command — arg parsing, engine validation, stdin prompt handling, request building (inline dict-comprehension `:202-214`), `asyncio.run`, result rendering.

### 2. Blocking sync I/O inside async code
- `providers/openai_sdk_provider.py:44,101`: sync `OpenAI(...)` client and blocking `chat.completions.create` called inside `async def generate_image` — blocks the event loop. Use `AsyncOpenAI` consistently.
- Same pattern in `web_server.py:176`: `asyncio.run()` inside a Flask sync handler; acceptable but each request pays full loop setup/teardown.

### 3. Duplication
- `utils.py:save_image_from_url` (`:155-188`) vs `save_image_from_b64` (`:191-215`): ~25 lines of identical Pillow open/inject-metadata/save logic, differing only in byte source. Extract one `_save_pil_image(bytes, path, prompt, model)`.
- `core.py:49-71`: the `f"{name}_{i+1}{ext}"` suffix block is copy-pasted 3× (output/auto/random branches). One helper fixes all three.
- `web_server.py:127-140` vs `:157-170`: `ImageGenerationRequest(...)` constructed twice; second overwrites the first after `input_image` handling.
- Two filename sanitizers that disagree: `sanitize_filename` (regex, `utils.py:18-23`) vs inline `isalnum` loop in `generate_filename` (`utils.py:129-131`).

### 4. Dead / inconsistent code in `utils.py:generate_filename_from_prompt_llm`
- `request_json` (`:77-83`, `max_tokens=30`, `temperature=0.7`) is built for verbose printing but the actual call (`:92-97`) uses `max_tokens=20`, `temperature=0.2`. The printed request is not the sent request.
- `import json` inside the function body (`:87`); verbose path uses `print` instead of `logging` like the rest of the module.

### 5. `config.py` re-implements what pydantic-settings already does
- Manual `os.environ` scan (`config.py:38-54`) duplicates `env_nested_delimiter="__"` parsing; fragile (`split("__")`, silent `lower()`, `try/except: pass` on `HttpUrl` coercion leaving a raw string in a typed field).
- Module-level side effect `Path(settings.output_dir).mkdir(...)` (`:58`) runs on every import, including tests.

### 6. Security / robustness (web_server.py)
- `subprocess.run(command, shell=True, ...)` (`:248-255`) on user-supplied `command` with only a string-prefix allowlist — bypassable (`imagai; evil...` starts with the prefix). Prefer an arg-list without `shell=True`, or drop `/api/generate-cli` since `/api/generate` already covers it.
- `debug=True` default in `main()` (`:369`); broad `except Exception → jsonify(str(e))` (`:222-224`) leaks internals to clients.
- `UPLOAD_FOLDER = Path("generated_images")` (`:33`) ignores `settings.output_dir`; `sys.path.insert` hack (`:23`) instead of relying on the installed package.

### 7. Model / default mismatches
- `response_format` default is `"url"` in `models.py:17` but `"b64_json"` in `cli.py:87` and `web_server.py:135`. One default should win.
- `size` Literal (`models.py:9-11`) only allows DALL-E sizes; Stability path works around it by deleting `size` when `aspect_ratio` is set (`openai_sdk_provider.py:220-221`). Validate per-provider instead of one global literal.
- `extra_params: Optional[dict]` (`models.py:18`) is untyped; typos (e.g. `aspect-ratio` vs `aspect_ratio`) fail silently.

### 8. Weak error propagation
- `utils.save_*` return `None` on *all* failures (`utils.py:180,188,212,215`); `core.py:97-99` then synthesizes `"Failed to save image to ..."`, discarding the logged root cause from the API response object.
- `web_server.py:117-124` silently drops unknown params and does unguarded `int()`/`float()` casts on user input (500 on bad `seed`).

### 9. Tests and style tooling
- `tests/test_cli.py` is two `assert True` placeholders — zero coverage of filename logic, the most testable pure functions in the repo.
- No linter/formatter config (`ruff`/`black`/`mypy`) in `pyproject.toml`; unused imports left around (e.g. `httpx`, `Image` in `openai_sdk_provider.py:1,4`; `Dict/Any/List` in `web_server.py:13`).

## Prioritized recommendations

- **P0 — Remove `shell=True` in `/api/generate-cli`** (`web_server.py:248`) or delete the endpoint; fix `debug=True` default (`:369`). Injection + debug server is the only security-grade finding.
- **P0 — Unify `response_format` default** (`models.py` vs `cli.py` vs `web_server.py`) and add a real test pinning it. Mismatched defaults cause silent behavior differences between CLI and web.
- **P1 — Split `OpenAISDKProvider.generate_image`**: extract `_generate_via_chat()` (OpenRouter path, async client) and `_render_usage_table()` out; fixes the blocking-sync-call bug at the same time.
- **P1 — Deduplicate image saving** (`utils.py`): one `_save_pil_image()` + thin `from_url`/`from_b64` wrappers; deduplicate the `_{i+1}` suffix in `core.py` into a helper. Small diffs, kill three copy-paste sites.
- **P1 — Delete the manual env loop in `config.py:38-54`** and rely on pydantic-settings; move `mkdir` out of import time into CLI/web startup.
- **P2 — Type `extra_params`** (TypedDict or per-provider models) and validate `size` per provider instead of one global Literal.
- **P2 — Add `ruff` + pytest coverage for pure functions** (`sanitize_filename`, `generate_filename`, `get_image_extension`, suffix helper): highest value-per-line tests in the repo; start there before touching network code.
