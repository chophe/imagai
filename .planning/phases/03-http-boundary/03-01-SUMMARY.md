---
phase: 03-http-boundary
plan: 01
subsystem: api
tags: [flask, pydantic, validation, security, sec-04]

requires:
  - phase: 02.5
    provides: web server that starts safely, binds loopback, and cannot execute a shell string
  - phase: 02
    provides: _contained_path containment check and the bare-basename save contract
provides:
  - One pydantic-validated ImageGenerationRequest build in generate_image()
  - Field-level 400 rendering from pydantic ValidationError (no caller input echoed)
  - A web-boundary _contained_path pre-check that rejects escaping output with 400 before any provider call
  - generate_image() front half no longer coerces malformed values into 500s
affects: [phase-3, http-boundary, error-propagation, provider-registry]

actuals:
  tokens: ~3500
  tasks: 3
  commits: 2

tech-stack:
  added: []
  patterns:
    - pydantic ValidationError -> field-level 400 response body
    - only-present-keys splat so absent fields keep the model's defaults

key-files:
  created: []
  modified:
    - src/imagai/web_server.py

key-decisions:
  - "Reuse the existing ImageGenerationRequest rather than authoring a parallel request schema — pydantic was already a direct dependency and the model already constrained every field this phase validates"
  - "Pass only-present-keys via a dict comprehension over ('size','n','quality','style'); a .get() with a None fallback would silently override pydantic's default of n=1"
  - "Keep the prompt/engine presence guards returning their existing STRING error bodies rather than unifying them into the dict shape, because test_api_generate_contract_unchanged asserts the exact string 'Prompt is required'"
  - "Leave the generate_image catch-all except (line 250, 500 with str(e)) untouched — error propagation is Phase 4's explicit ERR-01/ERR-02 territory"

patterns-established:
  - "Field-level 400: render only {field: msg} from pydantic's ValidationError, omitting pydantic's input/url/ctx sub-keys so caller input and schema URLs are never echoed"
  - "Guard ordering in generate_image(): prompt presence -> engine presence -> engine configured -> pydantic gate -> output containment -> provider call"

requirements-completed: [SEC-04]

coverage:
  - id: D1
    description: Malformed size/n/quality/style/response_format/prompt are rejected with a field-level 400 instead of reaching the provider or raising a 500
    requirement: SEC-04
    verification:
      - kind: unit
        ref: "tests/test_web_server.py#test_api_generate_contract_unchanged (existing contract held)"
        status: pass
      - kind: manual_procedural
        ref: "Flask test-client matrix: 12 rejection cases (missing prompt, non-string prompt, n='abc', n=0, n=99, n=-5, size='10x10', quality='ultra', response_format='png', output traversal, output absolute, output subdir) all returned 400"
        status: pass
    human_judgment: false
  - id: D2
    description: An output value escaping settings.output_dir returns 400 with field-level detail and no provider call
    requirement: SEC-04
    verification:
      - kind: manual_procedural
        ref: "Flask test client with patched settings.engines and mocked generate_image_core: traversal/absolute/subdir output all 400 with core_called=False"
        status: pass
    human_judgment: false
  - id: D3
    description: A valid request still returns HTTP 200 with the same results shape (top-level keys command/results/success; per-result index/success/error plus optional saved_path/image_data/image_url/image_b64_json)
    requirement: SEC-04
    verification:
      - kind: manual_procedural
        ref: "Flask test client with mocked generate_image_core: status 200, top-level keys ['command','results','success'], response-building block verified untouched by git diff"
        status: pass
    human_judgment: false

duration: 4min
completed: 2026-10-10
status: complete
---

# Phase 3: HTTP Boundary Summary

**Pydantic-validated `/api/generate` returning field-level 400s for malformed fields, with an output-containment pre-check that rejects path escapes before any provider call**

## Performance

- **Duration:** 4 min (implementation; verification performed separately by the orchestrator)
- **Started:** 2026-10-10T10:42:00Z
- **Completed:** 2026-10-10T10:46:00Z
- **Tasks:** 3
- **Files modified:** 1 (`src/imagai/web_server.py`)

## Accomplishments

- Collapsed the duplicated `ImageGenerationRequest` build (the two byte-identical constructions at what were `:130-143` and `:160-173`) into a single build placed after a one-time `input_image` base64 decode, so the decode runs once and the build sees the decoded bytes.
- Removed the `int(data.get("n", 1))` coercion that converted a malformed `n` into an unhandled `ValueError` and therefore a 500. `n` now flows raw into pydantic, which owns its own default and bounds.
- Wrapped the single build in `try/except ValidationError`, rendering a field-level 400 body `{"success": false, "error": {field: message}}` with only field names and messages echoed — pydantic's `input`, `url`, and `ctx` sub-keys are deliberately omitted so caller input and schema URLs never appear in a response (threat T-03-03).
- Added a web-boundary `_contained_path` pre-check on `output_filename`, so a traversal attempt returns 400 with a field-level message *before* the provider is reached, instead of being silently absorbed by the save layer returning `None`.

## Task Commits

Each task was committed atomically:

1. **Task 1: Collapse the duplicate build into one and prove it end-to-end** - `5f7e741` (feat)
2. **Task 2: Wrap the build in a pydantic gate that returns a field-level 400** - `5f7e741` (feat, same commit — the build rewrite and its gate are one contiguous region)
3. **Task 3: Pre-check output against _contained_path and reject with 400** - `549b05d` (feat)

**Plan metadata:** (SUMMARY.md, written by the orchestrator after the interrupted dispatch)

## Files Created/Modified

- `src/imagai/web_server.py` — front half of `generate_image()` rewritten: prompt/engine guards preserved verbatim, one input-image decode, one pydantic build, ValidationError → 400, output containment pre-check. The response-building block after `asyncio.run(_generate())` and the catch-all `except` at line 250 are both untouched (confirmed by `git diff` hunk boundaries: `@@ -26`, `@@ -36`, `@@ -126,51 +143,60 @@`).

## Decisions Made

- Only-present-keys splat — `{key: data[key] for key in ("size","n","quality","style") if key in data}`. Absent fields keep pydantic's defaults (`n=1`, `size='1024x1024'`); present malformed ones reach pydantic to be validated. A `.get()` with a `None` fallback would have silently overridden those defaults, which was the seam the planner flagged before execution.
- The prompt-presence and engine-config guards keep returning their existing **string** error bodies. They are not unified into the new dict shape, because `tests/test_web_server.py:468-476` (`test_api_generate_contract_unchanged`) asserts the exact string `"Prompt is required"` for a missing prompt. A present-but-invalid prompt (e.g. a non-string) now surfaces through the pydantic gate as `{"prompt": <msg>}`, which is the new case the lock anticipated.
- The catch-all `except Exception` at line 250 (`{"success": false, "error": str(e), "type": ...}`, 500) is deliberately untouched. Error propagation is Phase 4's ERR-01/ERR-02 scope; changing it here would duplicate that work.

## Deviations from Plan

None - the plan executed exactly as written. The only orchestrator-side difference is that Task 1 and Task 2 landed in a single commit rather than two: the build rewrite and its immediate gate are one contiguous region of the same function, and separating them would have produced a commit that does not compile on its own.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- SEC-04's rejection contract is in place; plan 03-02 adds the authoritative test suite for it (one rejection test per field, a parametrized "provider never reached" mock, and the duplicate-build regression test).
- The catch-all 500 at `web_server.py:250` and its `str(e)` leak remain open and are explicitly Phase 4's to fix.
- A Phase 2.5 residual remains open and is Phase 3's discretion: a concurrent `/api/generate` writing into `settings.output_dir` (which is `UPLOAD_FOLDER` by Phase 2's D-05) can land its file in a `/api/generate-cli` response, because `_GENERATE_CLI_LOCK` only covers generate-cli. This plan neither closed nor regressed it. If plan 03-02 or Phase 3's remaining work touches it, the lock is non-reentrant by design — keep acquisition order consistent to avoid a deadlock.

---
*Phase: 03-http-boundary*
*Completed: 2026-10-10*
