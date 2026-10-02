---
last_mapped_commit: 69cf5754b8d8021364ed445b02914b299360f67b
last_mapped_at: 2026-10-02
---
# External Integrations

**Analysis Date:** 2026-10-02

## APIs & External Services

**Image generation (all via one provider):**

- Every image backend is reached through the OpenAI SDK against a per-engine `base_url` — there is no native client for any vendor. Single call site: `await self.async_client.images.generate(**kwargs)` at `src/imagai/providers/openai_sdk_provider.py:234`.
  - Client: `openai` 1.82.1 (`AsyncOpenAI` constructed at `openai_sdk_provider.py:25`)
  - Auth: bearer API key per engine, from `IMAGAI__ENGINES__<NAME>__API_KEY`
  - OpenAI Images API — model `dall-e-3` is the hardcoded default (`openai_sdk_provider.py:40`); `.env.example:11-13` documents a `gpt-image-1` engine. `quality` and `style` are only sent when the model name contains `dall-e-3` (`openai_sdk_provider.py:186-188`).
  - Google Imagen (via an OpenAI-compatible gateway) — engines `IMAGEN4`, `IMAGEN4U`, `IMAGEN3F`, `IMAGEN31`, `IMAGEN32` in `.env.example:15-34`. No Imagen-specific code; it rides the generic images path.
  - Stability AI (via an OpenAI-compatible gateway, AvalAI named in comments at `openai_sdk_provider.py:192` and `cli.py:304`) — engines `STABILITY_v11U`, `STABILITY_v10U`, `STABILITY_v11C`, `STABILITY_v10C` in `.env.example:36-52`. Special-cased at `openai_sdk_provider.py:190-224`: drops `n` and `response_format`, forwards `negative_prompt`/`seed`/`strength`/`output_format`/`aspect_ratio`/`mode` through `extra_body`, defaults `mode` to `text-to-image` unless the model contains `sd3`, and prefers `aspect_ratio` over `size`.
  - Any other OpenAI-compatible third party — nothing vendor-specific is required; set `BASE_URL` and `MODEL` on any engine name.

**OpenRouter (special-cased chat path):**

- Detected by substring match `"openrouter.ai" in str(config.base_url)` at `openai_sdk_provider.py:36`. This is the only vendor detection in the codebase and it is string-sniffed, not configured.
- When OpenRouter **and** the model name contains `gemini` (`openai_sdk_provider.py:43`), generation switches from `images.generate` to `client.chat.completions.create` (`:101`) — a *sync* client (`OpenAI`, `:44`) called from an `async def`.
- Image models additionally get `extra_body={"modalities": ["image", "text"]}` (`:78-81`); responses are parsed from either a `data:image/...;base64,` prefix on `message.content` (`:132-141`) or a `message.images[0].image_url.url` data URL / plain URL (`:142-152`).
- Optional OpenRouter ranking headers read from env: `OPENROUTER_HTTP_REFERER` → `HTTP-Referer` and `OPENROUTER_X_TITLE` → `X-Title` (`openai_sdk_provider.py:47-52`). Absent, no headers are sent.
- Vision: an `extra_params["image_url"]` is appended as an `image_url` content part for chat-based Gemini requests (`:59-67`); the CLI surfaces it as `--image-url` (`cli.py:145-151`).

**Text/chat (filename generation only):**

- `AsyncOpenAI().chat.completions.create` at `src/imagai/utils.py:92-97` — asks an LLM for a filesystem-safe filename, `max_tokens=20`, `temperature=0.2`.
- Default model `gpt-4.1-mini` (`utils.py:66`); `.env.example:1-3` shows a dedicated `FILENAME_GENERATION` engine pinned to `gemini-2.0-flash`.
- Engine resolution order: dedicated `filename_generation` engine → `default_engine` → first engine whose name contains `openai` (`utils.py:35-48`). If the resolved key is missing or is the literal placeholder `YOUR_OPENAI_API_KEY`, it silently falls back to the non-LLM filename strategy with a warning (`utils.py:50-58`). Any change to engine-selection order must preserve that fallback.

**Model discovery:**

- `imagai list-engines` queries each configured engine for its model list, two ways (`src/imagai/cli.py:329-401`): first `client.models.list()` via the OpenAI SDK (`:348`), then a plain `GET {base_url}/models` with `Authorization: Bearer <key>` via `requests` and a 20s timeout (`:362-369`). Three payload shapes are tolerated: `{"data": [{"id"|"name"|"model"}]}`, `list[str]`, `list[dict]`.
- Engines without a real key are skipped (`:331`); unconfigured keys are the literal placeholder string (`:287`).

**Image download (egress to provider CDNs):**

- `httpx.AsyncClient().get(image_url)` at `src/imagai/utils.py:159-160`, `raise_for_status()`, then Pillow decode. This fetches the signed/temporary URL returned in the API response (typically an OpenAI or gateway CDN) — treat it as an outbound request to an untrusted host. `httpx.HTTPStatusError` is caught distinctly (`:181`).

## Data Storage

**Databases:**

- None. No SQL, no ORM, no migration tool, no connection strings anywhere in the repo.

**File Storage:**

- Local filesystem only. No S3/GCS/Azure Blob, no object-store SDK.
- Write target: `Path(settings.output_dir) / filename` at `src/imagai/core.py:76`. Default `generated_images` (`config.py:25-27`), created at **import time** by `config.py:58` with `parents=True, exist_ok=True`. This directory is gitignored.
- The Flask layer hardcodes a second, independent `UPLOAD_FOLDER = Path("generated_images")` (`web_server.py:33-34`) that does **not** read `settings.output_dir`. If you change the configured output dir, you must change this too or the web UI will list and serve the wrong directory.
- Image bytes are always re-encoded through Pillow rather than written verbatim (`utils.py:164-173`, `:198-207`); prompt/model metadata is embedded as EXIF tags 270/305 for JPEG and `PngInfo` text chunks for PNG (`utils.py:137-152`).
- Image-to-image uploads arrive as base64 data URLs in the JSON body, are decoded into raw bytes and stuffed into `extra_params["input_image"]` (`web_server.py:143-154`). They live in request memory only — nothing is persisted. Note `ImageGenerationRequest` is constructed twice in that handler (`web_server.py:127-140` then `:157-170`); the second overwrites the first so the first build is dead code.
- Read path: `GET /api/images` globs `UPLOAD_FOLDER` and returns stat metadata (`web_server.py:317-351`); `GET /api/images/<filename>` streams via `send_from_directory` after `secure_filename` (`web_server.py:311-312`).

**Caching:**

- None. No cache layer, no memoization, no HTTP cache headers. Every `generate` call hits the provider, and every `list-engines` run re-queries every engine.

## Authentication & Identity

**Auth Provider:**

- None. There are no user accounts, no sessions, no tokens, no OAuth.
- Provider credentials are static bearer API keys, one per engine, read from `.env` via `IMAGAI__ENGINES__<NAME>__API_KEY` and handed to `AsyncOpenAI(api_key=...)` (`openai_sdk_provider.py:20-25`, `utils.py:60-65`).
- "Not configured" is encoded as the literal sentinel string `YOUR_OPENAI_API_KEY` (`config.py:32`) and compared at `utils.py:53`, `cli.py:287`, `cli.py:331`. Preserve this exact sentinel if you refactor the default engine config.
- API keys are never logged. `list-engines` prints only a boolean "API Key Set" status (`cli.py:285-288`) — keep it that way.

**Web API authentication:**

- None. `CORS(app)` at `web_server.py:30` enables all origins, and no route performs authentication or authorization. `POST /api/generate-cli` (`web_server.py:227-304`) takes a caller-supplied `command` string and passes it to `subprocess.run(..., shell=True, timeout=300)` behind only a `startswith` prefix check (`web_server.py:239-244`) — a prefix check does not constrain a shell metacharacter after the prefix. Treat this server as strictly single-user/local; it must not be bound to a public interface without a rewrite.
- `main()` defaults to `host="0.0.0.0"` with `debug=True` (`web_server.py:369,377`), i.e. all interfaces plus the Werkzeug debugger.

**Secrets location:**

- `.env` at the repo root — **present** (existence noted only; contents not read). Gitignored via `.gitignore`.
- `.env.example` holds placeholders (`[key]`, `[url]`) only and is the safe template to follow when documenting a new engine.
- Never commit real values. Any new credential must go through the `IMAGAI__ENGINES__*` shape so pydantic-settings picks it up.

## Monitoring & Observability

**Error Tracking:**

- None. No Sentry/Datadog/Bugsnag/OTel. No structured error reporting of any kind.

**Logs:**

- `logging.getLogger(__name__)` is instantiated in `src/imagai/utils.py:15`, `src/imagai/core.py:16`, and `src/imagai/providers/openai_sdk_provider.py:14` — but **no handler or level is configured anywhere in the project**, and neither entry point calls `logging.basicConfig`. By default these records are dropped, so error messages in `utils.py` and `core.py` are effectively invisible at runtime. Add configuration in the CLI/web entry points if you need them.
- Verbose diagnostics are printed, not logged: `request.verbose` emits raw request/response bodies via `print()` in `utils.py:83-90,98-105` and `openai_sdk_provider.py:83-99,115-127,226-232`, and Rich `Console`/`Table` renders usage and estimated-cost rows at `openai_sdk_provider.py:249-293` and `cli.py:279-296`. Note `utils.py:99-104` references `json` in the verbose response branch, but `json` is only imported inside the earlier `if verbose` request block (`utils.py:87`) — a `NameError` path worth fixing when you touch this file.

## CI/CD & Deployment

**Hosting:**

- None. No `.github/`, no `Dockerfile`, no `docker-compose`, no `Makefile`, no infrastructure-as-code. Distribution is a locally built wheel/sdist via `rye build`; development installs are editable (`_imagai.pth` in site-packages).

**CI Pipeline:**

- None. Tests are run manually: `rye run pytest -q` (`README.md:43-46`). There is no test workflow, no lint gate, no coverage gate, and no release automation.

**Deploy command surface:**

- `imagai` → `imagai.cli:app` (Typer app) and `imagai-web` → `imagai.web_server:main` (`pyproject.toml` `[project.scripts]`).

## Environment Configuration

**Required env vars:**

- `IMAGAI__ENGINES__<NAME>__API_KEY` — at least one, to generate anything. **Required for real use.**
- `IMAGAI__ENGINES__<NAME>__MODEL` — recommended; falls back to `dall-e-3` (`openai_sdk_provider.py:40`).
- `IMAGAI__ENGINES__<NAME>__BASE_URL` — optional; omit for official OpenAI, set for any gateway/OpenRouter.
- `IMAGAI__DEFAULT_ENGINE` — optional; required only when `--engine` is not passed (`cli.py:161`).
- `IMAGAI__OUTPUT_DIR` — optional, default `generated_images`.
- `OPENROUTER_HTTP_REFERER`, `OPENROUTER_X_TITLE` — optional, OpenRouter ranking headers only.
- Nothing is truly *required* to import the package: `config.py:32` seeds a placeholder `openai_dalle3` engine so `import imagai.config` always succeeds.

**Secrets location:**

- `.env` in the repo root, gitignored. Environment variables override the file (pydantic-settings precedence), and the manual scan at `config.py:38-54` reads `os.environ` directly — so real-process env wins over `.env` for nested engine keys as well.

## Webhooks & Callbacks

**Incoming:**

- None. No webhook receivers, no signature verification, no provider-initiated callbacks. The only inbound HTTP surface is this tool's own local Flask API: `GET /` (`web_server.py:39`), `GET /api/engines` (`:52`), `POST /api/generate` (`:77`), `POST /api/generate-cli` (`:227`), `GET /api/images` (`:317`), `GET /api/images/<filename>` (`:307`), plus JSON 404/500 handlers (`:354-361`).
- Note `web_interface.html` only consumes `/api/generate` (`:654`) and `/api/engines` (`:748`); `/api/generate-cli` is reachable but not called by the bundled UI.

**Outgoing:**

- None. No callback URLs, no event publishing, no outbound webhooks. All outbound traffic is direct client→provider HTTPS.
- One indirect callback channel exists: OpenAI image URLs are temporary, so generated files must be fetched while the URL is valid — a provider-side expiry, not a callback, but the same class of time-sensitivity to respect.

---

*Integration audit: 2026-10-02*
