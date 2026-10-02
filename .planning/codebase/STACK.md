---
last_mapped_commit: 69cf5754b8d8021364ed445b02914b299360f67b
last_mapped_at: 2026-10-02
---
# Technology Stack

**Analysis Date:** 2026-10-02

## Languages

**Primary:**

- Python 3.12.9 — entire application. Version pinned in `.python-version` (`3.12.9`) and hard-linked into the venv at `.venv/pyvenv.cfg` (`version_info = 3.12.9`, `implementation = CPython`). All logic lives in `src/imagai/`.
- Note the declared floor contradicts the pin: `pyproject.toml` sets `requires-python = ">=3.8"`, but `pydantic 2.11` requires >=3.9 and the locked set targets 3.12. Treat 3.12.9 as the real target.

**Secondary:**

- HTML / CSS / vanilla JavaScript — single file, `web_interface.html` (870 lines). No build step, no bundler, no framework, and **no external CDN resources** (verified: zero `http://` / `https://` asset references). All assets are inline. Client state uses `localStorage` under key `imagai_settings` (`web_interface.html:535`).
- TOML — `pyproject.toml` (project + build + rye config).
- dotenv syntax — `.env` / `.env.example`.

## Runtime

**Environment:**

- CPython 3.12.9 (`.python-version`, `.venv/pyvenv.cfg`).
- **The checked-out `.venv` is non-functional on this host.** `.venv/pyvenv.cfg` points at `home = C:\Users\aliah\.rye\py\cpython@3.12.9` (Windows) and the bin dir is `Scripts/` rather than `bin/`. Run `rye sync` to recreate before executing anything. `.venv` is gitignored (`.gitignore`).
- No OS-level runtime services, no daemon, no container. The web server is an optional in-process Flask dev server.

**Package Manager:**

- [Rye](https://rye-up.com/) — `[tool.rye] managed = true` in `pyproject.toml`. `rye` is not currently on `PATH` in this environment.
- Lockfiles: `requirements.lock` (runtime) and `requirements-dev.lock` (with `pytest`). Both are rye-generated with `generate-hashes: false` and `universal: false`, so they are **not** suitable for CI installs or supply-chain auditing — add hashes before consuming them in an automated pipeline.
- Canonical commands (per `README.md:30-63`): `rye sync`, `rye add <pkg>`, `rye add --dev <pkg>`, `rye lock`, `rye build`, `rye run imagai ...`, `rye run pytest -q`.

## Frameworks

**Core:**

- Typer 0.16.0 — CLI framework. `app = typer.Typer(...)` at `src/imagai/cli.py:15-19`; commands `generate` (`cli.py:46`) and `list-engines` (`cli.py:260`).
- Pydantic 2.11.5 — request/response validation and config models. `src/imagai/models.py` (`ImageGenerationRequest`, `ImageGenerationResponse`), `EngineConfig` at `src/imagai/config.py:7`.
- pydantic-settings 2.9.1 — env-var-driven settings. `Settings` at `src/imagai/config.py:17-23`.
- OpenAI Python SDK 1.82.1 — the single transport for *all* engines. `AsyncOpenAI` at `src/imagai/providers/openai_sdk_provider.py:25`, `OpenAI` (sync) at `:44` and `cli.py:344`, `AsyncOpenAI` for LLM filenames at `src/imagai/utils.py:60`.
- httpx 0.28.1 — async download of provider-returned image URLs. `httpx.AsyncClient()` at `src/imagai/utils.py:159-160`.
- Pillow 11.2.1 — image decode/re-encode plus EXIF/PNG metadata injection. `Image.open` at `utils.py:164,198`; `_inject_metadata` at `utils.py:137-152`.
- Rich 14.0.0 — terminal rendering. `Console` at `cli.py:20`, `Table` at `cli.py:279`, `Panel` at `cli.py:244`; also `Console`/`Table` inside the provider at `openai_sdk_provider.py:249-293` (misplaced presentation layer — see `docs/code-quality.md` issue 1).

**Web:**

- Flask 3.1.2 — optional local web UI + REST API. `app = Flask(__name__)` at `src/imagai/web_server.py:29`.
- flask-cors 6.0.1 — `CORS(app)` at `web_server.py:30`, applied to **all** routes with no origin restriction.
- Werkzeug 3.1.3 — single symbol only: `secure_filename` at `web_server.py:20`, used at `web_server.py:311`. It is a transitive dependency of Flask anyway; the explicit pin in `pyproject.toml` is redundant.

**Testing:**

- pytest 8.3.5 — declared under `[tool.rye] dev-dependencies`, resolved into `requirements-dev.lock`. No `[tool.pytest.ini_options]`, no `pytest.ini`/`setup.cfg`, no markers, no coverage tooling. Only test file is `tests/test_cli.py` (6 lines, two `assert True` placeholders).

**Build/Dev:**

- hatchling — `[build-system] requires = ["hatchling"]`, `build-backend = "hatchling.build"`, wheel packages `["src/imagai"]` (`pyproject.toml`).
- **No linter, formatter, or type checker is configured.** No `ruff`, `black`, `flake8`, `mypy`, or `pre-commit` anywhere. Add one before growing the codebase — unused imports already sit in the source (e.g. `httpx` and `openai.types.images_response.Image` are imported but never used in `providers/openai_sdk_provider.py:1,4`; `render_template_string`, `Dict`, `Any`, `List` unused in `web_server.py:13,18`).

## Key Dependencies

**Critical:**

- `openai>=1.0.0` (locked 1.82.1) — the only path to every image backend. Do not swap for raw `httpx` without reimplementing the images and chat-completions call shapes in `providers/openai_sdk_provider.py`.
- `pydantic>=2.0.0` (2.11.5) + `pydantic-settings>=2.0.0` (2.9.1) — every request/response and all config. Changing pydantic majors is a breaking change here.
- `typer[all]>=0.9.0` (0.16.0) — CLI surface. The `[all]` extra buys nothing: the locked tree resolves only `click`, `rich`, `shellingham`, `typing-extensions`, and `add_completion=False` is set (`cli.py:18`). Drop the extra.
- `pillow>=10.0.0` (11.2.1) — required for metadata injection; the save path is Pillow-only, there is no raw-bytes passthrough.
- `httpx>=0.25.0` (0.28.1) — async HTTP for image download and the transitive base of the OpenAI SDK.

**Web stack:**

- `flask>=2.0.0` (3.1.2), `flask-cors>=3.0.10` (6.0.1), `werkzeug>=2.0.0` (3.1.3) — whole stack used only by `src/imagai/web_server.py`. Neither the CLI nor the core imports them, so this trio is removable at the cost of the web UI.

**Removal candidate:**

- `requests>=2.32.5` (2.32.5) — exactly one call site: the lazy `import requests as _requests` at `cli.py:306`, used for the `GET {base_url}/models` fallback at `cli.py:367` when the OpenAI client fails. `httpx` is already a direct dependency and does this in three lines. Removing it drops `urllib3` and `charset-normalizer` from the tree.

**Undeclared direct dependency:**

- `typing_extensions` is imported directly at `src/imagai/cli.py:2` (`from typing_extensions import Annotated`) but is **not** in `[project] dependencies`. It resolves today only as a transitive of `typer`/`pydantic`/`openai` (4.13.2 in the locks). Either add it explicitly or switch to `typing.Annotated` — do not leave it implicit.

**Runtime transitives of note:**

- `click` 8.2.1, `shellingham` 1.5.4 (typer), `markdown-it-py` 3.0.0, `pygments` 2.19.1, `mdurl` 0.1.2 (rich), `jinja2` 3.1.6, `markupsafe` 3.0.2, `itsdangerous` 2.2.0, `blinker` 1.9.0 (flask), `httpcore` 1.0.9 + `h11` 0.16.0 (httpx), `jiter` 0.10.0 + `distro` 1.9.0 + `tqdm` 4.67.1 + `sniffio` 1.3.1 (openai), `pydantic-core` 2.33.2 + `annotated-types` 0.7.0 + `typing-inspection` 0.4.1 (pydantic), `python-dotenv` 1.1.0 (pydantic-settings).

## Configuration

**Environment:**

- Configured entirely through pydantic-settings: `SettingsConfigDict(env_file=".env", env_prefix="IMAGAI__", env_nested_delimiter="__", extra="ignore")` at `src/imagai/config.py:18-23`.
- Recognised variables (see `.env.example` for the full shape):
  - `IMAGAI__OUTPUT_DIR` — output directory, default `generated_images` (`config.py:25-27`).
  - `IMAGAI__DEFAULT_ENGINE` — fallback engine name (`config.py:28-30`).
  - `IMAGAI__ENGINES__<NAME>__API_KEY`, `...__{BASE_URL,MODEL}` — one triple per engine. `<NAME>` is lowercased by the manual env scan (`config.py:42`).
- `config.py:38-54` adds a **manual** `os.environ` scan on top of pydantic-settings that re-parses `IMAGAI__ENGINES__*` with `split("__")`. It is the only way nested engines actually populate — the `env_nested_delimiter` alone does not cover this shape. Preserve or replace it deliberately; do not delete it assuming pydantic-settings covers the case.
- Two non-`IMAGAI__` variables are read directly in the provider: `OPENROUTER_HTTP_REFERER` and `OPENROUTER_X_TITLE` (`providers/openai_sdk_provider.py:47-48`), sent as `HTTP-Referer` / `X-Title` ranking headers.
- `.env` is present in the repo root and is gitignored (`.gitignore`). `.env.example` carries placeholders only. Never commit real values.
- Import-time side effect: `config.py:58` runs `Path(settings.output_dir).mkdir(parents=True, exist_ok=True)` on every import of `imagai.config`, including during tests. Any refactor must preserve or relocate that behaviour deliberately.

**Build:**

- `pyproject.toml` — single source of truth for deps, build backend, wheel packaging, console scripts (`imagai = imagai.cli:app`, `imagai-web = imagai.web_server:main`).
- `requirements.lock` / `requirements-dev.lock` — rye lockfiles, keep in sync via `rye lock`.
- `.python-version` — interpreter pin, change with `rye pin <version>`.
- `web_server.py:33-36` — Flask-local config: `UPLOAD_FOLDER` hardcoded to `Path("generated_images")` (ignores `settings.output_dir` and is relative to CWD) and `MAX_CONTENT_LENGTH = 16 * 1024 * 1024`. `main()` at `web_server.py:368-377` hardcodes `host="0.0.0.0"`, `port=5000`, `debug=True`.
- `web_server.py:23` inserts `src/` into `sys.path` at import time — a packaging workaround that is redundant given the installed `imagai` package.
- No `.github/`, no `Dockerfile`, no `Makefile`, no `docker-compose` — no CI or container config exists in this repo.

## Platform Requirements

**Development:**

- macOS / Linux (or WSL). Rye installed. Python 3.12.9 available through rye.
- `rye sync` before any run; the committed `.venv` is Windows-built and unusable here.
- A writable `.env` at the repo root, plus a CWD from which `generated_images/` and `web_interface.html` resolve (both `web_server.py:33` and `web_server.py:43` use relative paths).
- Network egress to the configured provider `base_url` hosts (OpenAI, OpenRouter, and/or third-party OpenAI-compatible gateways) is required for any non-filename operation.

**Production:**

- None. The package is a local developer tool: `rye build` produces a wheel/sdist (`[tool.hatch.build.targets.wheel] packages = ["src/imagai"]`) installed editable for development (`_imagai.pth` in site-packages).
- The web UI is explicitly a **development** server — `app.run(..., debug=True)` at `web_server.py:377` behind a wide-open `CORS(app)` and an unauthenticated `/api/generate-cli` endpoint that runs `subprocess.run(..., shell=True)` (`web_server.py:248-255`). Do not expose it on a shared or public interface as-is.

---

*Stack analysis: 2026-10-02*
