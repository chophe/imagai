---
last_mapped_commit: 69cf5754b8d8021364ed445b02914b299360f67b
last_mapped_at: 2026-10-02
---
# Coding Conventions

**Analysis Date:** 2026-10-02

> No formatter/linter is configured in `pyproject.toml` (no `ruff`, `black`, `flake8`, `mypy`,
> `isort`, `pre-commit`). The rules below are **derived from the code as written** and are
> prescriptive for new code. Where the codebase is inconsistent, the rule states the
> direction to standardize on, not what the outlier currently does.

## Naming Patterns

**Files:**

- Use `snake_case.py`, matching the module's single responsibility: `src/imagai/utils.py`,
  `src/imagai/core.py`, `src/imagai/models.py`, `src/imagai/config.py`,
  `src/imagai/cli.py`, `src/imagai/web_server.py`.
- Test files are `test_<module_under_test>.py` — `tests/test_cli.py` tests
  `src/imagai/cli.py`. Add `tests/test_utils.py` for `src/imagai/utils.py`; do not
  create `tests/utils_test.py`.
- `tests/__init__.py` exists and is empty — `tests/` is a package. Keep it that way so
  test module names cannot collide.

**Functions:**

- Use `snake_case` for functions and methods. Examples: `generate_image_core`
  (`src/imagai/core.py:19`), `save_image_from_b64` (`src/imagai/utils.py:191`),
  `list_engines_command` (`src/imagai/cli.py:261`).
- Prefix module-private helpers with a single underscore: `_is_image_model`
  (`src/imagai/cli.py:310`), `_inject_metadata` (`src/imagai/utils.py:137`).
- CLI entry points keep the verb-only name when the command name is derived
  automatically (`generate` at `src/imagai/cli.py:46` → `imagai generate`). When the
  function name would collide or is not a clean CLI noun, use the
  `<verb>_command` suffix plus an explicit `name=` — see
  `list_engines_command` with `@app.command(name="list-engines")` at
  `src/imagai/cli.py:260-261`. Follow that pattern rather than renaming the
  function.
- Closures used purely to adapt async → sync get a leading underscore:
  `_generate()` at `src/imagai/cli.py:220` and `src/imagai/web_server.py:173`.
- Prefix network calls to a specific vendor with the vendor name, not a generic verb:
  `generate_filename_from_prompt_llm` (`src/imagai/utils.py:26`).

**Variables:**

- Use `snake_case`: `selected_engine`, `base_filename`, `output_ext`, `saved_path`
  (`src/imagai/core.py:45-77`).
- Loop variables follow the collection's semantics: `for engine_name, engine_config in
  settings.engines.items()` (`src/imagai/web_server.py:57`); `for i, result in
  enumerate(results)` (`src/imagai/cli.py:225`).
- Module-level constants are `UPPER_SNAKE_CASE`: `UPLOAD_FOLDER`
  (`src/imagai/web_server.py:33`).
- The version string lives in exactly one place: `__version__ = "0.1.0"` at
  `src/imagai/__init__.py:1`, read by the CLI at `src/imagai/cli.py:10,25`. Do not
  hardcode the version elsewhere.

**Types:**

- Class names are `PascalCase`: `ImageGenerationRequest`, `ImageGenerationResponse`
  (`src/imagai/models.py:5,24`), `Settings`, `EngineConfig` (`src/imagai/config.py:17,7`),
  `BaseImageProvider` (`src/imagai/providers/base_provider.py:6`),
  `OpenAISDKProvider` (`src/imagai/providers/openai_sdk_provider.py:17`).
- Prefer `typing.Optional[X]` over bare `X | None` for library modules, matching
  `src/imagai/models.py` and `src/imagai/utils.py:8`. The `X | None` syntax in
  `src/imagai/cli.py:48` is an outlier.
- Use `typing.List[X]` (not `list[X]`) in library modules — `List` is imported in
  `src/imagai/core.py:14`, `src/imagai/providers/base_provider.py:3`,
  `src/imagai/providers/openai_sdk_provider.py:8`. The lowercase
  `list[str]` at `src/imagai/cli.py:337` is an outlier.
- Restrict string choices with `typing.Literal`, as in `src/imagai/models.py:9-17`
  (`quality: Optional[Literal["standard", "hd"]]`, `response_format:
  Optional[Literal["url", "b64_json"]]`). New request fields should follow this.
- Bound numeric fields with `Field(..., ge=1, le=10)` rather than validating in the
  caller — see `n` at `src/imagai/models.py:13-15`.

## Code Style

**Formatting:**

- No formatter is configured. The existing code is Black-shaped: 4-space indent, double
  quotes, trailing commas in multi-line call sites, ~88-column target. Examples:
  `src/imagai/core.py:20-21`, `src/imagai/utils.py:166-171`.
- **New rule:** when you touch a file, keep its existing wrap style. Do not
  reformat untouched lines — the diff cost outweighs the consistency gain.
- Prefer ternary expressions over `if/else` assignment blocks. This is the dominant
  idiom, e.g. `src/imagai/cli.py:285-292` (`api_key_status`, `base_url_str`) and
  `src/imagai/providers/openai_sdk_provider.py:168-172` (`resp.usage`).

**Linting:**

- None configured. Until one exists, self-enforce these rules the codebase follows:
  - No unused imports. Remove them when touching a file — `httpx` and
    `openai.types.images_response.Image` are unused in
    `src/imagai/providers/openai_sdk_provider.py:1,4`; `Dict`/`Any` are unused in
    `src/imagai/web_server.py:13`.
  - No bare `except:`. Always `except Exception as e:` and bind the exception,
    matching `src/imagai/providers/openai_sdk_provider.py:296`.
  - No `# type: ignore` without a code or explanation — the one existing use is
    `src/imagai/cli.py:306`.
- All I/O-facing functions must carry a timeout. `resp = _requests.get(url, headers=headers,
  timeout=20)` (`src/imagai/cli.py:367`) and `timeout=300` on
  `subprocess.run` (`src/imagai/web_server.py:253`) are the pattern.

## Import Organization

**Order:**

1. Third-party library imports (`typer`, `rich`, `openai`, `flask`, `PIL`).
2. First-party `imagai.*` imports.
3. Stdlib imports (`os`, `sys`, `re`, `asyncio`, `logging`, `pathlib`, `typing`) — this
   project places stdlib **last**, e.g. `src/imagai/cli.py:6-8` and
   `src/imagai/core.py:12-14`.
4. Module side-effect statements (`settings = Settings()` at
   `src/imagai/config.py:36`).

Group imports with blank lines between groups, but keep each group unsorted where the
file already is. Reproduce the local order rather than imposing a new alphabetization.

**Path Aliases:**

- None. Always import via the installed package: `from imagai.config import settings`
  (`src/imagai/core.py:1`). Do **not** add `sys.path` manipulation to make imports
  work — `src/imagai/web_server.py:23` (`sys.path.insert(0, str(Path(__file__).parent
  / "src"))`) is a bug that points at a non-existent `web_server/src` directory; the
  package is declared in `pyproject.toml` under
  `[tool.hatch.build.targets.wheel] packages = ["src/imagai"]`.
- Import the specific names you need (`from imagai.utils import generate_filename` at
  `src/imagai/core.py:4-11`) rather than module objects.

**Lazy imports:**

- Defer optional-dependency imports into the function body with a comment saying why.
  Both existing cases follow this shape: `from openai import OpenAI  # lazy import to
  avoid unnecessary import on normal runs` (`src/imagai/cli.py:300`) and
  `import requests as _requests  # type: ignore` (`src/imagai/cli.py:306`). Alias
  the fallback import (`_requests`) so it cannot shadow a real name.
- **Do not** put an import inside a function purely for convenience. The one bad case is
  `import json` inside the body of `generate_filename_from_prompt_llm`
  (`src/imagai/utils.py:87`) where `json` is already imported at module top in the
  sibling module. Move it to the module header.

## Error Handling

**Patterns:**

Three distinct, layer-specific strategies are in use. Match the one that belongs to the
layer you are writing in.

**CLI layer (`src/imagai/cli.py`)** — print a user-facing message, then exit:

```python
console.print("[bold red]Error:[/bold red] No engine specified and no default engine configured. Use --engine or set IMAGAI__DEFAULT_ENGINE.")
raise typer.Exit(code=1)
```

Use Rich markup (`[bold red]`, `[green]`, `[yellow]`, `[dim]`) and always append an
actionable hint — see the "Available configured engines: ..." fallback at
`src/imagai/cli.py:166-170`. Non-fatal notices use `console.print(...)` and `return`
(`src/imagai/cli.py:274-278`) instead of exiting. Never let a traceback reach the user:
wrap anything that can raise, as done at `src/imagai/cli.py:299-311`.

**Core/provider layer (`src/imagai/core.py`, `src/imagai/providers/`)** — never raise;
return a typed error object:

```python
except Exception as e:
    logger.error(f"Error generating image with engine {request.engine}: {e}")
    return [ImageGenerationResponse(error=str(e))]
```

(`src/imagai/providers/openai_sdk_provider.py:296-298`). Validation failures short-circuit
before any I/O with `return [ImageGenerationResponse(error=error_msg)]`
(`src/imagai/core.py:23-26`). **This is the key invariant of these layers: the returned
list always has one element per requested image, and `ImageGenerationResponse.error`
being non-`None` is the only failure signal.**

**Web layer (`src/imagai/web_server.py`)** — return a `(jsonify(...), status)` tuple:

```python
return jsonify({"success": False, "error": "Prompt is required"}), 400
```

Use 400 for missing/invalid input, 404 for unknown endpoint or missing file, 408 for
timeout, 500 for unexpected failure. Every JSON response body includes a `success`
boolean — keep it, the frontend reads it. Register handlers for error classes with
`@app.errorhandler` (`src/imagai/web_server.py:354,359`) rather than repeating the
tuple inline.

**Utilities (`src/imagai/utils.py`)** — log and return `None`:

```python
except Exception as e:
    logger.error(f"Error decoding or saving base64 image: {e}")
    return None
```

Return `Optional[Path]` on success/failure and keep the root cause in the log line.
Note the known weakness: `save_image_from_url` and `save_image_from_b64` return `None`
for every failure mode, so `src/imagai/core.py:97-99` has to synthesize
`f"Failed to save image to {output_file_path}"` and loses the original message. When you
touch these, prefer raising or returning a result object with an `error` field.

**Guard clauses:**

- Validate at the top of a function and return early, rather than nesting the happy
  path. `src/imagai/core.py:23-26` and `src/imagai/cli.py:162-171` are the models.
- Use `getattr(obj, "attr", None)` when reading a field that may be absent on a
  provider-specific response object — `getattr(request, "extra_params", None)`
  (`src/imagai/providers/openai_sdk_provider.py:59`), `getattr(api_response, "usage",
  None)` (`src/imagai/providers/openai_sdk_provider.py:235`). This is the accepted
  substitute for `hasattr` + direct access.

**Never swallow exceptions silently.** No bare `except Exception: pass` in new code.
The two existing violations are `src/imagai/config.py:49-51` (HttpUrl coercion failure
leaves a raw string in a typed field) and
`src/imagai/providers/openai_sdk_provider.py:97-98` / `:123-127` (broad catch that
degrades to a print).

## Logging

**Framework:** stdlib `logging`. `print()` is the dominant output mechanism today
(≈50 `print(` calls vs 12 `logger.` calls across `src/imagai/`) — this is a debt, not a
convention to copy.

**Patterns:**

- Every library module that logs declares a module logger immediately after imports:
  `logger = logging.getLogger(__name__)` at `src/imagai/utils.py:15`,
  `src/imagai/core.py:16`, and `src/imagai/providers/openai_sdk_provider.py:14`.
  **Add this line whenever you create a new module that logs.**
- `src/imagai/cli.py` and `src/imagai/web_server.py` intentionally have no logger — they
  are presentation layers and use Rich `console.print` / raw `print`. Keep it that way.
- Log at `ERROR` for swallowed failures and `WARNING` for degraded-but-recoverable paths.
  Examples: `logger.warning("OpenAI API key for filename generation not configured. Using
  default filename.")` (`src/imagai/utils.py:55-57`);
  `logger.error(f"Error generating filename with LLM: {e}. Using default method.")`
  (`src/imagai/utils.py:112`).
- Log success at `INFO` with the resolved path: `logger.info(f"Image saved to
  {output_path}")` (`src/imagai/utils.py:174`).
- Use `logger.exception(...)` only where the traceback is genuinely useful and you are
  at the outermost boundary of the operation — `src/imagai/core.py:102-104`.
- Verbose output goes through `print`, gated on `request.verbose`
  (`src/imagai/providers/openai_sdk_provider.py:83-99,226-232`;
  `src/imagai/utils.py:83-90`). If you add verbose output, gate it on `verbose` and
  print a `--- Label ---` header with a matching `---...---` footer.
- **Never** put an API key, prompt body, or full response payload in a log line at a
  level enabled by default. `src/imagai/utils.py:84-89` prints the request body under
  `verbose`; follow that gating.

## Comments

**When to Comment:**

- Comment the *why*, not the *what*. `src/imagai/cli.py:298` (`# If requested, also
  fetch models from each engine and display them`) and `src/imagai/utils.py:31` (priority
  chain for filename-generation engine selection) explain decisions.
- Number the alternative branches when several paths exist, so the reader can navigate:
  `# 1) Try via OpenAI client when available` / `# 2) Fallback via plain HTTP`
  (`src/imagai/cli.py:340,357`).
- Cite the external constraint that produced a workaround:
  `src/imagai/providers/openai_sdk_provider.py:191-192` (Stability ignores `n`).
- Comment non-obvious platform behavior: `# Remove ASCII control characters (including
  \x1a from Windows Ctrl+Z) and trim` (`src/imagai/cli.py:178`).
- **Do not** add comments restating the next line. `# Load the images` above a
  `get_images()` call adds noise.

**JSDoc/TSDoc:**

- One-line docstrings on public functions and route handlers, in the imperative
  third-person-singular style already used: `"""Sanitizes a string to be a valid
  filename."""` (`src/imagai/utils.py:19`), `"""Get available engines from
  configuration"""` (`src/imagai/web_server.py:54`),
  `"""Generates a random filename."""` (`src/imagai/utils.py:120`).
- Use a multi-line docstring when documenting the contract of an interface, as in
  `BaseImageProvider.generate_image` (`src/imagai/providers/base_provider.py:11-14`).
- Module docstring only where the module needs orientation: `src/imagai/web_server.py:2-5`.
- Pydantic fields get `description=` rather than a `#` comment —
  `src/imagai/config.py:8-14`, `src/imagai/models.py:14`.

## Function Design

**Size:**

- Target 30 lines or fewer per function; absolute ceiling ~60. Every current violation
  is a known debt item: `generate_image_core` is 90 lines (`src/imagai/core.py:19-109`),
  `generate` is 210 lines including its Typer signature (`src/imagai/cli.py:46-257`),
  `OpenAISDKProvider.generate_image` is 270 lines
  (`src/imagai/providers/openai_sdk_provider.py:27-295`), `list_engines_command` is 157
  lines (`src/imagai/cli.py:261-417`).
- **Rule: new functions must not exceed ~60 lines.** If a task pushes a function past
  that, extract helpers instead — `_inject_metadata` (`src/imagai/utils.py:137`) is the
  model of a correctly extracted helper.
- Typer signatures inflate line count artificially because every parameter carries an
  `Annotated[..., typer.Option(...)]` block. Do not extract parameter groups into a
  dataclass to shorten them; keep the flat signature so `--help` renders correctly.

**Parameters:**

- Type every parameter and give optional ones a default. Annotate defaults as
  `Optional[X] = None` (`src/imagai/core.py:19-21`, `src/imagai/models.py:8`).
- Do **not** use non-`Optional` parameters with a `None` default. Two existing bugs:
  `prompt: str = None` and `model: str = None` at `src/imagai/utils.py:156` and `:192`.
  Write `prompt: Optional[str] = None`.
- Keep parameter counts low by taking a Pydantic model where a function handles a
  request payload — `generate_image_core(request: ImageGenerationRequest)`
  (`src/imagai/core.py:20`).
- Do not mutate the caller's input object. `src/imagai/web_server.py:157-170` rebuilds
  `ImageGenerationRequest` to smuggle in `extra_params["input_image"]` after having
  already built it at `:127-140` — build one object with the final dict instead.

**Return Values:**

- Return the domain type; do not return tuples. Return
  `List[ImageGenerationResponse]` (`src/imagai/core.py:21`) or `Optional[Path]`
  (`src/imagai/utils.py:157`).
- Make failure representable in the return type. In provider/core code that means an
  object with `.error` set, never a raised exception.
- Async all the way down. `generate_image_core`, `generate_filename_from_prompt_llm`,
  `save_image_from_*`, and `provider.generate_image` are all `async def`. Mark new I/O
  functions `async def` unless a caller is strictly synchronous.
- **Always** `await client.close()` on an async OpenAI client. Use `try/finally` with a
  `locals()` guard (`src/imagai/utils.py:114-116`) or a `close()` method on the provider
  invoked in `finally` (`src/imagai/core.py:106-108`).
- Use `AsyncOpenAI` inside `async def` — never the blocking `OpenAI`. The exception is
  `OpenAI(...)` at `src/imagai/providers/openai_sdk_provider.py:44` followed by a blocking
  `client.chat.completions.create(...)` at `:101` inside an `async def`. Fix with
  `await AsyncOpenAI(...).chat.completions.create(...)`.

## Module Design

**Exports:**

- One primary responsibility per module, and the name states it. `utils.py` holds I/O
  and naming helpers; `core.py` is the single orchestration entry point; `models.py` is
  types only; `providers/` is vendor adapters.
- `src/imagai/__init__.py` exports only `__version__` (plus a leftover `hello()` helper
  at `:4` that no module calls — delete it rather than keeping dead package surface).
- `src/imagai/providers/__init__.py` is intentionally empty: consumers import the
  concrete class directly (`from imagai.providers.openai_sdk_provider import
  OpenAISDKProvider`, `src/imagai/core.py:3`). Keep it empty; do not add re-exports.

**Barrel Files:**

- None, and none needed. The `from imagai.utils import (...)` multi-name form at
  `src/imagai/core.py:4-11` serves the purpose. Do not add `__all__` to modules or
  create a `utils/__init__.py` re-export surface.

**Layering:**

- Enforce the dependency direction: `cli.py` / `web_server.py` → `core.py` →
  `providers/*` → `models.py`, with `config.py` and `utils.py` as leaves. Providers
  must never import from `cli.py`, `web_server.py`, or `core.py`.
- **Presentation does not belong in a data layer.** `src/imagai/providers/openai_sdk_provider.py:249-293`
  builds a `rich` `Console` and `Table` and prints usage/cost from inside the provider;
  `Console`/`Table`/`Panel` belong only in `src/imagai/cli.py` and
  `src/imagai/web_server.py`. New provider code must return data (e.g. populate
  `ImageGenerationResponse.usage` / `.estimated_cost`, which already exist at
  `src/imagai/models.py:29-30`) and let the caller render it.
- No import-time side effects. `src/imagai/config.py:58` runs
  `Path(settings.output_dir).mkdir(parents=True, exist_ok=True)` on every import —
  verified: importing `imagai.config` in an empty temp dir creates
  `generated_images/`. Move directory creation into CLI/web startup (`src/imagai/cli.py`
  and `src/imagai/web_server.py:34` are the correct places) so tests can import the
  package without touching the filesystem.

**Configuration:**

- Read configuration only through the module-level `settings` singleton
  (`src/imagai/config.py:36`), imported as `from imagai.config import settings`
  (`src/imagai/core.py:1`, `src/imagai/cli.py:13`). Never call `Settings()` again.
- Env vars use the `IMAGAI__` prefix with `__` as the nested delimiter
  (`src/imagai/config.py:18-23`), so an engine is configured as
  `IMAGAI__ENGINES__<NAME>__API_KEY`.
- Every new setting needs a `Field(..., description="...")`
  (`src/imagai/config.py:25-30`).
- The placeholder `"YOUR_OPENAI_API_KEY"` is the sentinel for "unset" and is compared
  literally in 8 places (`src/imagai/config.py:32,45`, `src/imagai/cli.py:287,331`,
  `src/imagai/utils.py:53`). Reference it as a single module-level constant rather than
  repeating the literal.
- Consolidate duplicate defaults. `response_format` defaults to `"url"` in
  `src/imagai/models.py:17` but `"b64_json"` in `src/imagai/cli.py:86` and
  `src/imagai/web_server.py:135,165`; pick one and have every layer import it.

---

*Convention analysis: 2026-10-02*
