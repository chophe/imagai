# Architecture

## Overview
Imagai is a CLI-first image-generation tool targeting OpenAI-compatible image APIs (DALL-E, Stable Diffusion, Gemini/Imagen via OpenAI-compatible gateways). A thin Flask server (`imagai-web`) reuses the same core to expose the CLI over a web UI (`web_interface.html`).

Stack: Python, Typer (CLI), Pydantic + pydantic-settings (models/config), OpenAI SDK + httpx (providers), Pillow (image I/O), Flask + Flask-CORS (web server).

## Directory Structure
```
.
├── src/imagai/              # Main package (shipped as wheel via hatchling)
│   ├── cli.py               # Typer app: `generate`, `list-engines`
│   ├── core.py              # Orchestration: engine lookup → provider call → save file
│   ├── config.py            # Settings (IMAGAI__ prefix, .env) + EngineConfig
│   ├── models.py            # ImageGenerationRequest / ImageGenerationResponse
│   ├── providers/
│   │   ├── base_provider.py       # Provider interface
│   │   └── openai_sdk_provider.py # Single OpenAI-SDK implementation for all engines
│   ├── utils.py             # Filename strategies, URL/b64 saving, LLM filename gen
│   └── web_server.py        # Flask app wrapping generate_image_core
├── tests/test_cli.py
├── web_interface.html       # Static UI served by Flask at /
├── generated_images/        # Default output dir (created at startup by config.py)
├── pyproject.toml           # Entry points: imagai=imagai.cli:app, imagai-web=imagai.web_server:main
├── .env / .env.example      # Per-engine keys: IMAGAI__ENGINES__<NAME>__{API_KEY,BASE_URL,MODEL}
└── docs/architecture.md     # This file
```

## Main Components
- **CLI (`cli.py`):** Parses `generate --prompt/--engine/--output/-n/--size/--quality/--auto-filename/--random-filename/--verbose`, builds `ImageGenerationRequest`, calls `generate_image_core()` via asyncio, renders Rich output.
- **Config (`config.py`):** `Settings` with `env_prefix=IMAGAI__`, `env_nested_delimiter=__`; `engines: Dict[str, EngineConfig]` (api_key/base_url/model). Supplements pydantic-settings with manual `IMAGAI__ENGINES__` env scan + `Path(output_dir).mkdir`.
- **Models (`models.py`):** `ImageGenerationRequest` (prompt, engine, size, quality, n, style, response_format, filename flags) and `ImageGenerationResponse` (image_url/image_b64_json/saved_path/error/text_content for chat-based vision fallbacks).
- **Core (`core.py:generate_image_core`):** Engine lookup → `OpenAISDKProvider(engine_config).generate_image(request)` → filename resolution (`--output` > `--auto-filename` (LLM) > `--random-filename` > prompt-truncated default) → save to `settings.output_dir/` → return responses.
- **Providers (`providers/`):** `base_provider.py` interface; `openai_sdk_provider.py` is the only concrete provider — all engines go through the OpenAI SDK with per-engine `base_url`/`model`/`api_key`.
- **Utils (`utils.py`):** `sanitize_filename`, `generate_filename`, `generate_random_filename`, `generate_filename_from_prompt_llm` (prefers `filename_generation` engine → `default_engine` → first `*openai*` engine), `save_image_from_url` / `save_image_from_b64`.
- **Web server (`web_server.py`):** Flask + CORS; `GET /` serves `web_interface.html`, `GET /api/engines`, generation endpoints call the same `generate_image_core()`.

## Data Flow / Entry Points
Entry points (`pyproject.toml [project.scripts]`): `imagai` → `imagai.cli:app`; `imagai-web` → `imagai.web_server:main`.

CLI flow:
```
user → imagai generate -p "..." --engine X → ImageGenerationRequest
  → core.generate_image_core() → OpenAISDKProvider.generate_image()
  → image URL/b64 → utils.save_* → generated_images/<name>.png → Rich console output
```

Web flow: `web_interface.html → Flask (/api/*) → generate_image_core()` — same path as CLI from core onward.
