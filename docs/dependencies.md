# Dependencies Review

Source files inspected: `pyproject.toml`, `requirements.lock`, `requirements-dev.lock`, `.python-version` (3.12.9).
Import usage spot-checked under `src/imagai/` (`cli.py`, `web_server.py`, `utils.py`, `config.py`, `core.py`, `providers/openai_sdk_provider.py`).

## 1. Dependency overview

Direct dependencies (`pyproject.toml` `[project] dependencies`):

| Package | Declared | Locked (`requirements.lock`) | Used where | Verdict |
|---|---|---|---|---|
| `typer[all]` | `>=0.9.0` | 0.16.0 | `cli.py:1` | Keep, but drop `[all]` extra |
| `httpx` | `>=0.25.0` | 0.28.1 | `utils.py:1`, `providers/openai_sdk_provider.py:1` (+ via `openai`) | Keep |
| `pydantic` | `>=2.0.0` | 2.11.5 | `models.py`, `config.py:1` | Keep |
| `pydantic-settings` | `>=2.0.0` | 2.9.1 | `config.py:2` | Keep |
| `pillow` | `>=10.0.0` | 11.2.1 | `utils.py:5,13` | Keep |
| `openai` | `>=1.0.0` | 1.82.1 | `utils.py:12`, `providers/openai_sdk_provider.py:3`, `cli.py:300` | Keep |
| `rich` | `>=13.0.0` | 14.0.0 | `cli.py:3-5`, `providers/openai_sdk_provider.py:11-12` | Keep |
| `requests` | `>=2.32.5` | 2.32.5 | `cli.py:306` (lazy fallback only) | Remove candidate — replace with `httpx` |
| `flask` | `>=2.0.0` | 3.1.2 | `web_server.py:18` | Keep (or revisit, see below) |
| `flask-cors` | `>=3.0.10` | 6.0.1 | `web_server.py:19` | Keep |
| `werkzeug` | `>=2.0.0` | 3.1.3 | `web_server.py:20,311` (`secure_filename` only) | Remove candidate — redundant via Flask |
| Dev: `pytest` | `>=7.0.0` (`tool.rye dev-dependencies`) | 8.3.5 (`requirements-dev.lock`) | `tests/` | Keep |

Key transitives (from lockfiles): `click`, `shellingham`, `anyio`, `httpcore`, `h11`, `jiter`, `tqdm`, `distro`, `jinja2`, `markupsafe`, `itsdangerous`, `blinker`, `urllib3`, `certifi`, `charset-normalizer`, `idna`, `markdown-it-py`, `pygments`, `python-dotenv`, `typing-extensions`.

## 2. Outdated / risky items

1. **Lower bounds too loose, no upper bounds.** `typer>=0.9`, `httpx>=0.25`, `openai>=1.0`, `flask>=2.0`, `werkzeug>=2.0` all allow a fresh install to resolve to a much newer (possibly breaking) major than the locked version. Use compatible-release pins, e.g. `typer~=0.16`, `httpx~=0.28`, `openai~=1.82`, `flask~=3.1`, or at minimum raise floors to the locked majors.
2. **`requires-python = ">=3.8"` is stale.** Python 3.8 is EOL (Oct 2024), and the locked set (`pydantic 2.11` needs >=3.9, `rich 14` / `typer 0.16` target newer) plus `.python-version` 3.12.9 contradict it. Bump to `>=3.9` (better: `>=3.10`) and test accordingly.
3. **`flask>=2.0` / `werkzeug>=2.0` allow known-vulnerable 2.0.x.** Locked versions (3.1.2 / 3.1.3) are fine, but a fresh resolver could pick an old insecure release. Raise floors to `>=3.0` (matching the lock).
4. **`openai==1.82.1` likely stale.** The 1.x SDK moves fast; re-run `rye lock --update openai` (then full test) to pick up model/API fixes. Same for `httpx`, `pillow`, `pydantic`.
5. **Lockfiles have `generate-hashes: false`, `universal: false`.** Fine for local dev, but don't ship/supply-chain-audit on these; enable hashes if the lock is consumed in CI/deploy.
6. **`typer[all]` is heavier than needed.** The `[all]` extra pulls shell-completion helpers beyond what `cli.py` uses (`typer`, `Annotated`, `rich`). Locked tree only shows `shellingham`/`rich`/`click`, so the extra buys little here — switch to plain `typer`.

## 3. Potentially unnecessary items

1. **`requests` — top removal candidate.** Single use is a lazy `import requests` fallback for `GET {base_url}/models` in `cli.py:306`. `httpx` is already a direct dep and does the same call in ~3 lines (`httpx.get(...)`). Removing `requests` drops a whole parallel HTTP stack (`urllib3`, `charset-normalizer` as direct-via-requests edges) for zero feature loss.
2. **`werkzeug` explicit pin — redundant.** Only use is `secure_filename` (`web_server.py:20,311`), which is re-exported through the `Flask` dependency itself. Delete the line from `pyproject.toml` and rely on Flask's own `werkzeug` requirement (or keep only if you need to force `>=3.1` for a CVE floor — then add a comment saying so).
3. **`flask` + `flask-cors` + `werkzeug` as a trio — review, don't delete blindly.** The web server is a small internal UI (`web_server.py` ~377 lines, one `Flask` app, `CORS(app)` wide-open, one `subprocess.run(shell=True)` endpoint at `/api/generate-cli`). If the web UI stays, keep Flask. If you want fewer deps, the stdlib or the already-present `httpx`+`pydantic` stack (e.g. FastAPI) would fit better — but that's a rewrite, not a cleanup. At minimum, narrow `CORS(app)` to the needed origins.
