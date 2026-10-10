# Phase 3: HTTP Boundary - Context

**Gathered:** 2026-10-10
**Status:** Ready for planning

<domain>
## Phase Boundary

`POST /api/generate` rejects malformed requests with HTTP 400 *before* any provider call, while valid
requests keep returning the exact `results` shape they return today. The boundary becomes a real
validation gate rather than a coercion funnel: every rejected field fails the whole request with a
400, no image is generated, and nothing is written outside the configured output directory.

SEC-04 is the phase's only requirement. The SEC-03 `NameError` criterion is inherited from Phase 2.5
(which already fixed it) and is a regression guard only, not new work.

</domain>

<decisions>
## Implementation Decisions

### Validation mechanism
- Reuse the existing pydantic `ImageGenerationRequest` (models.py:5) — pydantic is already a direct
  dependency and this model already constrains every field Phase 3 cares about. No parallel schema.
- Pass **raw** request values into the model and let pydantic validate them. Remove the
  `.get()`-with-default coercion (`int(data.get("n", 1))`, the raw `size`/`output` reads), which is
  what currently converts would-be 400s into 500s.
- Collapse the duplicated request build in `generate_image()` (`web_server.py:130-143` and
  `:160-173` — byte-identical except the second picks up the decoded `input_image`) into a single
  build. This phase owns it.
- Error body carries field-level detail: `{"success": false, "error": {field: message}}` derived
  from pydantic's `ValidationError`.

### Output containment boundary
- Pre-check `output` against Phase 2's `_contained_path` (`utils.py:26-46`) at the web boundary and
  reject with 400 when it fails. This makes the boundary consistent with the save layer's rejection
  instead of letting a traversal attempt be silently absorbed.
- Status code for a rejected `output` is 400, same as every other malformed field.
- A rejected `output` fails the whole request — no provider call is made — matching the ROADMAP
  criterion "never reaches the provider".

### Scope of error handling
- The catch-all `except` at `web_server.py:224-227` that returns `{"error": str(e)}` with 500 is
  left untouched. Error propagation (ERR-01/ERR-02) is Phase 4's explicit territory.
- Every field rejection is a 400. No 422s — one error code for malformed input keeps the contract
  simple and matches the existing "Prompt is required" behavior.
- `n` bounds come from pydantic's existing `Field(1, ge=1, le=10)` (models.py:13-15). No new limit.
- The `size` enum comes from `ImageGenerationRequest`'s existing `Literal` (models.py:9-11) — one
  source of truth, no web-specific duplicate.

### Test strategy
- Add a shared `client` pytest fixture; the file currently does local imports in all 17 tests.
- One test per rejected field, plus a parametrized assertion that the provider is never reached,
  backed by a mock so no live API call is possible.
- Cover the duplicate-build collapse with a test asserting the built request is identical before
  and after the `input_image` decode.
- `test_api_generate_contract_unchanged` (`tests/test_web_server.py:468-476`, asserts 400 for
  `json={}`) must stay green. The new validation must not change what a valid request returns.

### the agent's Discretion
- Whether the `_contained_path` pre-check lives inline in the route or as a small helper beside the
  other validation.
- Whether field-level error rendering is one `__all__` map or a comprehension over `e.errors()`.

</decisions>

<code_context>
## Existing Code Insights

### Reusable Assets
- `ImageGenerationRequest` (`src/imagai/models.py:5`) — pydantic model already constraining `size`
  (Literal, models.py:9-11), `quality`/`style`/`response_format` (Literals), `n`
  (`Field(1, ge=1, le=10)`, models.py:13-15).
- `_contained_path` (`src/imagai/utils.py:26-46`) — Phase 2's containment check. Accepts a resolved
  parent == resolved `settings.output_dir` with no `..` parts; rejects traversal, absolute paths,
  and subdirectories; never raises, returns bool. Callers `save_image_from_url` (`:182-186`) and
  `save_image_from_b64` (`:223`) log and return None.
- `generate_image_core` (`src/imagai/core.py`) — the orchestration entry the validated request goes to.

### Established Patterns
- No schema library, decorator, or `before_request` hook exists — validation is hand-rolled per
  endpoint (web_server.py:86-107 for the prompt guard, `:283-296` for the `command` isinstance
  guard added in Phase 2.5).
- Existing error contract: `{"success": false, "error": "Prompt is required"}` with 400
  (web_server.py:86-107); success body is `{"success": true, "results": [...]}` (`:182-222`).
- Typer CLI does the same validation by `Annotated` constraints (cli.py:45-214) before pydantic
  sees it — so the CLI and the web boundary currently disagree on which layer validates.
- Response building reads the saved file back as a best-effort preview; a preview failure lands in
  `result_data["preview_error"]` (`:212`), not a request failure.

### Integration Points
- `web_server.py:127-179` — `generate_image()`, the function the whole phase rewrites its front half.
- `tests/test_web_server.py` — 17 tests, no fixtures, `from imagai.web_server import app;
  client = app.test_client()` per test (`:65, :93, :472`). Mocking via
  `patch.object(web_server, "UPLOAD_FOLDER", Path(tmpdir))` inside `tempfile.TemporaryDirectory()`
  (`:67, :313, :387`) and `patch.object(web_server.subprocess, "run", ...)` (`:68-72`).
- `tests/test_web_server.py:468-476` — the contract test that must stay green.

</code_context>

<specifics>
## Specific Ideas

- The ROADMAP criteria are explicit that a valid request must return HTTP 200 with the **same
  `results` shape as before this phase**. Preserve every existing response key: `index`, `success`,
  `error`, `saved_path`, `image_data` (data-URL), `image_url`, `image_b64_json`.
- Phase 2.5 left a documented residual: a concurrent `/api/generate` writing into
  `settings.output_dir` can land its file in a `/api/generate-cli` response. The `_GENERATE_CLI_LOCK`
  is non-reentrant and named to stay non-reentrant — if this phase makes `/api/generate` acquire
  it, keep the acquisition order consistent to avoid a deadlock. Whether to *close* that residual
  is the agent's call, but it must not regress it.

</specifics>

<deferred>
## Deferred Ideas

- The catch-all 500 error shape and error propagation — Phase 4 (ERR-01/ERR-02).
- The `test_cli.py` `assert True` placeholders — Phase 4/6, already tracked in STATE.md.
- Rate limiting / quota on `/api/generate-cli`'s unbounded args — note only, no quota system.

</deferred>
