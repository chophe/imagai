---
last_mapped_commit: 69cf5754b8d8021364ed445b02914b299360f67b
last_mapped_at: 2026-10-02
---
# Testing Patterns

**Analysis Date:** 2026-10-02

> **Read this first.** The project has **no real test suite**. There is exactly one test
> file, `tests/test_cli.py`, containing two `assert True` placeholders. Everything below
> distinguishes (a) the tiny set of conventions that *do* exist and (b) the patterns this
> project *should* adopt, which are labeled as recommendations, not observations. Do not
> assume a pattern exists here just because it is written down — sections marked
> "Establish" are work to be done.

## Test Framework

**Runner:**

- `pytest` (declared `pytest>=7.0.0` as a dev dependency under `[tool.rye]` in
  `pyproject.toml`; `requirements-dev.lock` pins `pytest==8.3.5`).
- Config: **none**. There is no `[tool.pytest.ini_options]` section in `pyproject.toml`,
  no `pytest.ini`, no `tox.ini`, no `setup.cfg`, and no `tests/conftest.py`. Everything
  runs on pytest's defaults — rootdir auto-detection, no `testpaths`, no markers, no
  `addopts`.
- No `tests/__init__.py`-related import mode config; `tests/__init__.py` exists and is
  empty.

**Assertion Library:**

- Built-in `assert`. No `pytest-mock`, no `pytest-asyncio`, no `pytest-cov`, no
  `hypothesis` in `requirements-dev.lock`.

**Verified current state:**

```
$ python3 -m pytest -q
..                                                                       [100%]
2 passed in 0.05s
```

Two tests, zero application code imported, zero network calls, ~0.05s runtime.

**Run Commands:**

```bash
rye sync                        # install deps (required first — see Environment below)
rye run pytest -q               # run all tests
rye run pytest tests/test_cli.py::test_app_version   # single test
rye run pytest -q -k "filename"                      # filter by name
```

Not yet available (add the dependency first with `rye add --dev pytest-cov`):

```bash
rye run pytest --cov=src/imagai --cov-report=term-missing   # coverage
rye run pytest --cov=src/imagai --cov-fail-under=50        # gate
```

**Environment caveat (verify before writing tests):**

- The package is **not installed** into the default interpreter —
  `python3 -c "import imagai"` fails with `ModuleNotFoundError`. Tests currently pass only
  because they import nothing.
- `.venv/` in the repo is a **Windows-layout** venv (`Lib/`, `Scripts/`, no `bin/`), so it
  is not usable on macOS/Linux. Use `rye sync` to create a working environment.
- **Any test that imports `imagai` needs the environment installed** (`rye sync`) or
  `PYTHONPATH=src` set. Verified: `PYTHONPATH=src python3 -m pytest -q` passes, and
  `PYTHONPATH=src python3 -c "from imagai.config import settings"` works.
- `flask` is not installed in the ambient interpreter, so `src/imagai/web_server.py`
  cannot be imported or tested until `rye sync` runs.
- Importing `imagai.config` has a filesystem side effect: `src/imagai/config.py:58`
  creates `settings.output_dir` (`generated_images/` by default) on import. Verified in a
  temp dir — the directory appears after import. Any test importing `config` currently
  writes to the CWD.

## Test File Organization

**Location:**

- All tests live in `tests/` at the repository root, mirroring `src/imagai/` — they are
  **not** co-located with source. `tests/test_cli.py` ↔ `src/imagai/cli.py`.

**Naming:**

- `test_<module_under_test>.py`. Only `tests/test_cli.py` exists.
- Test functions are `test_<behavior>()`, lowercase snake_case:
  `test_app_version`, `test_generate_help` (`tests/test_cli.py:1,5`).
- **Establish** the rest of the mapping when you add files:
  `tests/test_utils.py` for `src/imagai/utils.py`, `tests/test_core.py` for
  `src/imagai/core.py`, `tests/test_models.py`, `tests/test_config.py`,
  `tests/test_openai_sdk_provider.py` for
  `src/imagai/providers/openai_sdk_provider.py`, `tests/test_web_server.py`.

**Structure:**

```
tests/
├── __init__.py          # empty, present
├── conftest.py          # DOES NOT EXIST — create it
└── test_cli.py          # 2 placeholder tests
```

## Test Structure

**Suite Organization:**
Current state, verbatim (`tests/test_cli.py`, all 6 lines):

```python
def test_app_version():
    assert True

def test_generate_help():
    assert True
```

There are **no** classes, no fixtures, no setup/teardown, no docstrings, and no imports.
The names promise coverage of `cli.py` (`app_version`, `generate_help`) that the bodies
do not deliver.

**Establish this pattern** — one module-level `test_*` function per behavior, no test
classes, import the code under test at module top:

```python
from typer.testing import CliRunner

from imagai.cli import app

runner = CliRunner()

def test_app_version_prints_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "Imagai Version" in result.stdout
```

`typer.testing.CliRunner` is available (typer 0.24.2 installed; `CliRunner` import
verified working). It is the right tool for `src/imagai/cli.py` because it needs no
network and no API key.

**Patterns:**

- **Setup:** none currently. Prefer plain function-local setup, or module-level constants
  for immutable values. Introduce fixtures only once real setup exists (see Fixtures).
- **Teardown:** none. Not needed — no test touches the filesystem today. When tests do,
  use pytest's built-in `tmp_path` and `monkeypatch` fixtures rather than manual cleanup.
- **Assertions:** use bare `assert`. For exceptions, use
  `with pytest.raises(ValueError):` — `pytest.raises` is already available, no extra
  dependency needed.
- **Async:** **no async support is installed.** `pytest-asyncio` is absent, so `async def`
  tests will be silently skipped-or-error. The application under test is async-heavy
  (`generate_image_core`, `save_image_from_*`, `provider.generate_image`). **Any test of
  async code requires `rye add --dev pytest-asyncio` and
  `@pytest.mark.asyncio`** (or `asyncio.run(...)` inside a sync test as a stopgap, which is
  what production code does at `src/imagai/cli.py:224` and `src/imagai/web_server.py:176`).

## Mocking

**Framework:** none installed. Use stdlib `unittest.mock` (`unittest.mock.patch`,
`MagicMock`, `AsyncMock`) — it ships with Python and needs no new dependency.
`pytest-mock` is optional; prefer stdlib until a reason exists to add it.

**What to Mock:**

This is the critical part. **Every real call in this codebase hits the network and
requires a paid API key.** Mock at these seams:

| Seam | Target | Where |
|------|--------|-------|
| Provider boundary | `imagai.core.OpenAISDKProvider` | `src/imagai/core.py:3,28` — patch the *name in `core`*, not the class |
| Provider client | `OpenAISDKProvider.async_client` | `src/imagai/providers/openai_sdk_provider.py:25` — `AsyncOpenAI` instance |
| Sync chat client | `imagai.providers.openai_sdk_provider.OpenAI` | `src/imagai/providers/openai_sdk_provider.py:3,44` |
| HTTP download | `httpx.AsyncClient` | `src/imagai/utils.py:159-160` |
| Models list HTTP | `requests.get` (as `_requests`) | `src/imagai/cli.py:306,367` |
| Settings | `imagai.config.settings` / `imagai.core.settings` | imported at `src/imagai/core.py:1`, `src/imagai/utils.py:11`, `src/imagai/cli.py:13` |
| Image saving | `imagai.core.save_image_from_url` / `save_image_from_b64` | `src/imagai/core.py:84,88` |
| `asyncio.run` | `imagai.cli.asyncio.run` | `src/imagai/cli.py:224` |

Patch **where the name is used**, not where it is defined. `core.py` does
`from imagai.providers.openai_sdk_provider import OpenAISDKProvider`, so
`@patch("imagai.core.OpenAISDKProvider")` works and
`@patch("imagai.providers.openai_sdk_provider.OpenAISDKProvider")` does not.

**Mock `settings.engines` for every test that constructs a request.** `settings` is a
module-level singleton (`src/imagai/config.py:36`) populated from the real `.env` and
`os.environ` (`src/imagai/config.py:38-54`). Tests must not depend on the developer's
local keys. Replace the whole object with a `Settings`-shaped stub, or `monkeypatch` the
specific attributes.

**What NOT to Mock:**

- **Do not mock `imagai.models`.** The Pydantic validation in `src/imagai/models.py` —
  the `Literal` constraints on `size`/`quality`/`style`/`response_format`
  (`:9-17`) and the `ge=1, le=10` bound on `n` (`:13-15`) — is pure, fast logic and is
  exactly what needs testing. Let real `ValidationError` be raised.
- **Do not mock `imagai.utils` pure helpers.** `sanitize_filename`
  (`src/imagai/utils.py:18`), `generate_filename` (`:126`),
  `generate_random_filename` (`:119`), and `get_image_extension` (`:218`) are
  deterministic, side-effect-free, and network-free. They are the highest
  value-per-line tests available and need zero mocks.
- **Do not mock `asyncio` itself.** Mock the coroutine you are calling, not the loop.
- **Do not let tests touch real keys or the real `generated_images/`.** Use `tmp_path`
  and `monkeypatch.setenv`.

## Fixtures and Factories

**Test Data:**
No fixtures, factories, or sample data exist in the repo today.

**Establish `tests/conftest.py`** with these fixtures, in this order of value:

```python
import pytest
from pathlib import Path

@pytest.fixture
def stub_settings(monkeypatch, tmp_path):
    """Settings with deterministic engines and an isolated output dir."""
    # Build a Settings-like object; override imagai.config.settings.output_dir
    # to tmp_path so no test writes into the repo's generated_images/.
    ...

@pytest.fixture
def fake_provider(monkeypatch):
    """OpenAISDKProvider stub returning ImageGenerationResponse objects."""
    ...

@pytest.fixture
def image_bytes():
    """A tiny valid PNG produced by PIL, for save_image_* tests."""
    ...
```

Key rules for fixtures you write:

- Scope `output_dir` to `tmp_path` in **every** fixture that leads to a save. Note that
  `src/imagai/config.py:58` already creates `generated_images/` at import time, so an
  import-time side effect you cannot avoid without fixing that file.
- Use `monkeypatch.setenv("IMAGAI__...", ...)` for env-dependent tests rather than
  mutating `settings.engines` directly, so teardown is automatic.
- Generate image bytes with Pillow (`PIL.Image.new("RGB", (2, 2)).save(buf, "PNG")`) —
  Pillow is already a runtime dependency (`pyproject.toml`), no new dependency needed.
- Build valid `ImageGenerationResponse` objects by real construction
  (`ImageGenerationResponse(image_url="https://...")`), not `MagicMock`, so field names
  stay honest.

**Location:** all fixtures in `tests/conftest.py`. Do not define fixtures in individual
test modules unless they are single-use.

## Coverage

**Requirements:** **none enforced.** `pytest-cov` is not a declared dependency
(confirmed absent from `requirements-dev.lock` and not importable). There is no
`--cov` invocation, no `fail_under`, no coverage badge, and no CI to enforce anything.

**View Coverage:**
Not available. To enable:

```bash
rye add --dev pytest-cov
rye run pytest --cov=src/imagai --cov-report=term-missing
```

Coverage should be measured against `src/imagai` (the packaged path per
`[tool.hatch.build.targets.wheel] packages = ["src/imagai"]` in `pyproject.toml`).

**Practical baseline:** real coverage is effectively **0%**. The two tests import nothing
from `imagai`, so no application line executes. The suite under test is ~1515 lines of
source (`cli.py` 421, `web_server.py` 377, `providers/openai_sdk_provider.py` 301,
`utils.py` 222, `core.py` 109, `config.py` 58, `models.py` 32, `base_provider.py` 15,
`__init__.py` 5).

**Do not set a high `fail-under` gate yet.** Start at 50% once real tests land, raise
later. A strict 80% gate on a 2-test suite fails immediately and gets disabled.

## Test Types

**Unit Tests:**

- Approach: direct function calls on pure helpers, real Pydantic construction for
  validation, `monkeypatch`/`patch` for anything touching `settings` or a client.
- Highest-value targets, in order — all pure, all network-free, all currently untested:
  1. `get_image_extension` (`src/imagai/utils.py:218-222`) — known-extension whitelist
     vs `"png"` fallback.
  2. `sanitize_filename` (`src/imagai/utils.py:18-23`) — note the regex character class
     contains an escaped `\\x00-\\x1F` inside a raw string, so control characters are
     likely **not** stripped as intended; verify the actual behavior before writing the
     assertion, then pin it.
  3. `generate_filename` (`src/imagai/utils.py:126-134`) — prompt with/without spaces,
     30-char truncation, no-prompt fallback. Note it uses a **different** sanitizer than
     `sanitize_filename` (inline `isalnum` loop at `:129-131`).
  4. `ImageGenerationRequest` validation (`src/imagai/models.py:5-21`) — reject bad
     `size`/`quality`/`style`/`response_format` literals; reject `n=0` and `n=11`.
  5. The `n > 1` filename suffixing in `generate_image_core`
     (`src/imagai/core.py:49-51,58-63,66-71`) — three copy-pasted branches that must
     agree; a parametrized test over all three is cheap.
  6. `save_image_from_url` / `save_image_from_b64` (`src/imagai/utils.py:155,191`) with a
     mocked `httpx` / real base64 bytes and `tmp_path` — assert the file exists, assert
     `None` is returned on bad input.
- Note the default-value mismatch these tests will surface: `response_format` defaults to
  `"url"` in `src/imagai/models.py:17` but `"b64_json"` in `src/imagai/cli.py:86` and
  `src/imagai/web_server.py:135`. **Write a test that pins whichever value is chosen** —
  this is the single highest-value regression test available.

**Integration Tests:**

- Scope: `generate_image_core` with a mocked provider and mocked save functions — i.e.
  orchestration logic only, no network. Verify: unknown engine returns a single
  `ImageGenerationResponse` with `.error` set (`src/imagai/core.py:23-26`); a text-only
  provider response is passed through without a save attempt (`src/imagai/core.py:38-44`);
  `provider.close()` is called in `finally` (`src/imagai/core.py:106-108`).
- Flask routes in `src/imagai/web_server.py` are testable with `app.test_client()`
  (Flask is a declared dependency; installed only after `rye sync`). Targets:
  `/api/engines` (`:52`), `/api/generate` missing-prompt 400 (`:83-84`) and
  unconfigured-engine 400 (`:101-104`), `/api/images` (`:317`), the 404 handler
  (`:354-356`).
- The provider itself (`generate_image`) is only integration-testable by mocking the
  OpenAI client — it branches heavily on config (`src/imagai/providers/openai_sdk_provider.py:43,186,190`)
  and needs cases for OpenRouter-chat, DALL·E, and Stability paths.

**E2E Tests:**

- **Not used, and not currently possible.** No Playwright/Cypress/Selenium dependency,
  no browser automation, and `web_interface.html` (33 KB, at repo root) has no test
  harness.
- `/api/generate-cli` (`src/imagai/web_server.py:227`) shells out via
  `subprocess.run(..., shell=True)` — an E2E test of it would require a real API key. Do
  not attempt; and see the security note below.
- If browser testing is ever wanted, note the repo already has a `dogfood` skill for
  manual exploratory QA of the running server (`rye run imagai-web`).

**CI:**

- **None.** No `.github/`, no GitLab CI, no Jenkinsfile, no pre-commit config, no tox.
  Tests are run manually (`rye run pytest -q`, documented in `README.md`).
- **Establish:** a single GitHub Actions job on push/PR doing `rye sync` +
  `rye run pytest -q`. Do not add a version matrix until multi-version support is real
  (`pyproject.toml` declares `requires-python = ">=3.8"` but the code uses `str | None`
  at `src/imagai/cli.py:48`, which is 3.10+ syntax — the declared floor is wrong and
  would fail on 3.8/3.9).

## Common Patterns

**Async Testing:**
Not currently possible — `pytest-asyncio` is not installed. Establish with:

```bash
rye add --dev pytest-asyncio
```

Then:

```python
import pytest

@pytest.mark.asyncio
async def test_saves_b64_image(tmp_path):
    out = tmp_path / "x.png"
    result = await save_image_from_b64(_valid_png_b64(), out, "p", "m")
    assert result == out
    assert out.exists()
```

Configure the mode in `pyproject.toml` once the dependency exists:

```toml
[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "-q"
```

Until then, use `asyncio.run(...)` inside a sync test as a stopgap — matching how
production code bridges the gap (`src/imagai/cli.py:220-224`,
`src/imagai/web_server.py:173-176`).

**Error Testing:**

```python
import pytest
from imagai.models import ImageGenerationRequest

def test_rejects_unknown_response_format():
    with pytest.raises(ValueError):   # pydantic ValidationError subclasses ValueError
        ImageGenerationRequest(prompt="p", engine="e", response_format="xml")

def test_n_upper_bound_enforced():
    with pytest.raises(ValueError):
        ImageGenerationRequest(prompt="p", engine="e", n=11)   # le=10
```

For functions that swallow errors into a `None` return
(`src/imagai/utils.py:180,188,212,215`), assert the sentinel:

```python
async def test_save_returns_none_on_bad_base64(tmp_path):
    assert await save_image_from_b64("not-base64!!", tmp_path / "x.png") is None
```

For core-layer failures, assert on `.error` rather than on an exception:

```python

# src/imagai/core.py:23-26

async def test_unknown_engine_returns_error_response(stub_settings):
    results = await generate_image_core(
        ImageGenerationRequest(prompt="p", engine="nope")
    )
    assert len(results) == 1
    assert "not configured" in results[0].error
```

**CLI Testing:**

```python
from typer.testing import CliRunner
from imagai.cli import app

runner = CliRunner()

def test_generate_help_lists_options():
    result = runner.invoke(app, ["generate", "--help"])
    assert result.exit_code == 0
    assert "--num-images" in result.stdout

def test_generate_without_engine_exits_1(stub_settings):
    result = runner.invoke(app, ["generate", "--prompt", "x"])
    assert result.exit_code == 1
    assert "No engine specified" in result.stdout
```

`CliRunner` needs no network and no key — this is the cheapest real coverage available
for `src/imagai/cli.py` (421 lines).

**Flask Testing:**

```python
import pytest
from imagai.web_server import app

@pytest.fixture
def client(monkeypatch, tmp_path):
    app.config["TESTING"] = True
    app.config["UPLOAD_FOLDER"] = str(tmp_path)
    return app.test_client()

def test_missing_prompt_returns_400(client):
    res = client.post("/api/generate", json={})
    assert res.status_code == 400
    assert res.get_json()["success"] is False
```

`src/imagai/web_server.py:29` creates the Flask app at module import, and `:33-34`
creates `generated_images/` at import time — override `UPLOAD_FOLDER` in the fixture to
keep tests hermetic.

## Coverage Gaps

Ranked by risk. Nothing below is covered today.

**Provider error handling — HIGH RISK:**

- What's not tested: `OpenAISDKProvider.generate_image` response parsing — the base64
  data-URL split (`src/imagai/providers/openai_sdk_provider.py:132-141`), the `images`
  field fallback (`:142-152`), usage/cost extraction (`:165-174,235-236`), the
  `"No image data found in API response."` branch (`:244`), and the catch-all
  (`:296-298`). This is the largest untested block (~270 lines) and the most
  provider-dependent.
- Files: `src/imagai/providers/openai_sdk_provider.py`
- Risk: silent wrong-format images reach users unnoticed; SDK response-shape changes
  break with no signal.
- Priority: High

**`generate_image_core` orchestration — HIGH RISK:**

- What's not tested: filename selection across all four branches
  (`src/imagai/core.py:45-75`), the `n>1` suffix duplicated three times, the
  text-content bypass (`:38-44`), the save-failure error synthesis (`:94-99`), and the
  `finally: provider.close()` guarantee (`:106-108`).
- Files: `src/imagai/core.py`
- Risk: files overwritten or misnamed; leaked provider connections.
- Priority: High

**Pure filename/extension helpers — HIGH VALUE, LOW COST:**

- What's not tested: `sanitize_filename`, `generate_filename`,
  `generate_random_filename`, `get_image_extension`
  (`src/imagai/utils.py:18,126,119,218`). Zero mocks required.
- Files: `src/imagai/utils.py`
- Risk: path traversal or overwriting from unsanitized filenames; these two sanitizers
  currently disagree.
- Priority: High

**Pydantic model validation — HIGH VALUE, LOW COST:**

- What's not tested: every `Literal` and the `n` bound in
  `src/imagai/models.py:9-17`; the `response_format` default mismatch between
  `models.py:17`, `cli.py:86`, and `web_server.py:135`.
- Files: `src/imagai/models.py`, `src/imagai/cli.py`, `src/imagai/web_server.py`
- Risk: silent behavior differences between CLI and web entry points.
- Priority: High

**CLI surface — MEDIUM:**

- What's not tested: `--version`/`--help` output, missing-engine exit codes
  (`src/imagai/cli.py:162-171,181-188`), stdin prompt path (`:173-179`), and
  `list_engines_command`'s model-fetch branches (`:329-417`).
- Files: `src/imagai/cli.py`
- Risk: broken entry points ship undetected — the placeholders were presumably named for
  exactly these two tests and never written.
- Priority: High

**Config loading — MEDIUM:**

- What's not tested: `IMAGAI__ENGINES__*` env parsing
  (`src/imagai/config.py:38-54`), the `base_url` `HttpUrl` coercion, and the import-time
  `mkdir` (`:58`). No test can be written for this file without first removing or
  neutralizing the import side effect.
- Files: `src/imagai/config.py`
- Risk: engines silently fail to configure; tests write into the repo directory.
- Priority: Medium

**Web routes — MEDIUM:**

- What's not tested: all six routes and both error handlers in
  `src/imagai/web_server.py`. Blocked today because Flask is not installed in the
  ambient interpreter and `web_server.py:23` contains a broken `sys.path` hack.
- Files: `src/imagai/web_server.py`
- Risk: the JSON contract the 33 KB `web_interface.html` depends on can break silently.
- Priority: Medium

**Image saving / metadata — MEDIUM:**

- What's not tested: `_inject_metadata` EXIF vs PNG branches
  (`src/imagai/utils.py:137-152`) and both save paths (`:155,191`) including the
  16 MB / format edge cases.
- Files: `src/imagai/utils.py`
- Risk: silently degraded output metadata; duplicated logic can diverge.
- Priority: Medium

**Image-payload saving is untestable offline as written** — `save_image_from_url` opens
a real `httpx.AsyncClient` (`src/imagai/utils.py:159`). Mock `httpx.AsyncClient.get` or
the module-level `httpx` before writing these tests.

## Security Note for Test Authors

`POST /api/generate-cli` runs a user-supplied string through
`subprocess.run(command, shell=True, ...)` (`src/imagai/web_server.py:248-255`) with a
prefix allowlist (`str.startswith` on `"imagai"`, `"rye run imagai"`,
`"python -m imagai"`, at `:239-241`). That check is bypassable
(`imagai; <arbitrary command>` passes it), so this is a command-injection endpoint.

- **Do not write a test that exercises `generate-cli` against a real shell.** Mock
  `subprocess.run` if you need to test the response shape at all, and only with benign
  arguments.
- `debug=True` is the default for the server (`src/imagai/web_server.py:369`) — do not
  bind this to a shared interface in any test setup.
- Never place a real API key in a test, a fixture, or an env default. Use the
  `"YOUR_OPENAI_API_KEY"` placeholder sentinel (`src/imagai/config.py:32`) if a key-shaped
  string is structurally required.

---

*Testing analysis: 2026-10-02*
