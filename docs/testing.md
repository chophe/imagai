# Testing Strategy

## Current setup

- Runner: `pytest >= 7` (dev dependency via `[tool.rye]` in `pyproject.toml`). No pytest config section, no `conftest.py`, no coverage tool (`pytest-cov` not declared).
- Tests: single file `tests/test_cli.py` with 2 placeholder tests (`test_app_version`, `test_generate_help`), both `assert True` — they exercise nothing.
- No CI config (no `.github/`, GitLab, Jenkins, or tox files found).
- Source under test: ~1540 lines across `cli.py` (421), `web_server.py` (377), `providers/openai_sdk_provider.py` (301), `utils.py` (222), `core.py` (109), `config.py` (58), `models.py` (32).

## What's covered / missing

- Covered: effectively nothing — the two tests pass trivially and import no app code.
- Missing: CLI commands (`generate`, version flag, error paths), `core.generate_image_core`, provider dispatch and API error handling (largest untested logic), `config` settings/env loading, `models` validation, `utils` helpers, `web_server` routes. No mocked-API tests, so any real test would hit network/keys.

## Recommendations

1. **Replace placeholders with real CLI tests using `typer.testing.CliRunner`.** Assert `--help` exit code, `--version` output, and `generate` missing-arg behavior. No network needed, covers `cli.py`.
2. **Add mocked provider tests.** Mock `httpx`/`openai` client at the `base_provider` boundary; test success, HTTP error, and auth-missing paths in `core.py` + `openai_sdk_provider.py`.
3. **Add `pytest-cov` with a fail-under gate (start at 50%, raise later).** Run `pytest --cov=src/imagai --cov-fail-under=50`. Skipped: strict 80%+ gate now, add when suite is real.
4. **Add minimal CI (GitHub Actions: `uv/pip install -e .` + `pytest`).** Single `python -m pytest` job on push/PR. Skipped: matrix builds, add when multi-version support matters.
5. **Add `tests/conftest.py` with fixtures for temp output dirs and fake settings.** Removes env-key dependence so tests run offline and hermetically.
