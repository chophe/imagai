---
phase: 03-http-boundary
plan: 02
subsystem: testing
tags: [flask, pytest, pydantic, validation, security, sec-04]

requires:
  - phase: 03-http-boundary
    provides: the pydantic 400 gate, the _contained_path output pre-check, and the collapsed request build in web_server.py
  - phase: 02.5
    provides: 17 existing web-server safety tests, the shared test patterns, and the client-per-test style
  - phase: 02
    provides: _contained_path containment semantics the boundary pre-check now mirrors

provides:
  - The authoritative SEC-04 rejection suite: one test per rejected field, each asserting 400 plus the field key in the error body
  - A parametrized provider-never-reached proof backed by a mocked generate_image_core (no live API call is possible)
  - A shared `client` pytest fixture and an `engine_ready` fixture that resolves a configured engine past the engine guard
  - A frozen valid-path shape pin (index/success/error/saved_path/image_data/image_url/image_b64_json)
  - Static guards that no 422 exists and that the Phase 4 catch-all 500 body is untouched

affects: [phase-4, http-boundary, error-propagation, provider-registry]

actuals:
  tokens: ~5800
  tasks: 2
  commits: 2

tech-stack:
  added: []
  patterns:
    - shared pytest fixtures (client / engine_ready) replacing per-test app.test_client()
    - AsyncMock + patch.object(web_server, "generate_image_core") as the provider-never-reached seam
    - pydantic subclass with a recording __init__ to prove a single request construction

key-files:
  created: []
  modified:
    - tests/test_web_server.py

key-decisions:
  - "Added the shared `client` fixture and a second `engine_ready` fixture rather than patching settings in each test: the engine guard runs BEFORE the pydantic gate, so with default_engine unset every payload would 400 for the engine reason and a rejection test would pass for the wrong reason. The fixture asserts its own precondition (a configured engine exists) once, so a config variance cannot masquerade as a pass."
  - "Used AsyncMock for the provider seam rather than a plain async def, so `core.called is False` is available in the same style as the existing `m.called is False` assertions in this file."
  - "Pinned the only-present-keys splat with a happy-path assertion on the captured request (n == 1, size == 1024x1024) rather than a source grep, because the splat's failure mode is a silent None, which only shows on the built model."
  - "Kpt every existing test byte-for-byte unchanged; the new tests append below them."

patterns-established:
  - "Provider-never-reached: patch the module attribute (web_server.generate_image_core) with an AsyncMock and assert `.called is False`; the module imports the name at scope, so the patch point is the module attribute"
  - "Containment tests place the output dir one level inside a fresh tmp_path so `../../x` resolves to a path the test itself owns, making 'nothing was written outside output_dir' an assertion rather than an assumption"

requirements-completed: [SEC-04]

coverage:
  - id: D1
    description: One HTTP 400 rejection test per malformed field (size, n non-integer, n below bound, n above bound, quality, style, response_format, non-string prompt, output absolute/traversal/subdirectory), each carrying the field key in the error body
    requirement: SEC-04
    verification:
      - kind: unit
        ref: "tests/test_web_server.py#test_api_generate_rejects_size_outside_enum (and the 11 sibling rejection tests)"
        status: pass
    human_judgment: false
  - id: D2
    description: A parametrized assertion that no malformed payload ever crosses into the provider, backed by a mocked generate_image_core
    requirement: SEC-04
    verification:
      - kind: unit
        ref: "tests/test_web_server.py#test_api_generate_malformed_payload_never_reaches_provider (9 parametrized cases)"
        status: pass
    human_judgment: false
  - id: D3
    description: A valid request still returns HTTP 200 with exactly the frozen result keys, so a dropped or renamed key is a red build
    requirement: SEC-04
    verification:
      - kind: unit
        ref: "tests/test_web_server.py#test_api_generate_valid_request_returns_frozen_results_shape"
        status: pass
    human_judgment: false
  - id: D4
    description: The collapsed build is a single construction fed the decoded input_image bytes and is otherwise identical to the no-image build
    requirement: SEC-04
    verification:
      - kind: unit
        ref: "tests/test_web_server.py#test_api_generate_builds_the_request_once and #test_api_generate_build_is_identical_before_and_after_input_image_decode"
        status: pass
    human_judgment: false
  - id: D5
    description: The 400 error body never echoes caller input, pydantic schema URLs, or pydantic's input/ctx sub-keys (T-03-03)
    requirement: SEC-04
    verification:
      - kind: unit
        ref: "tests/test_web_server.py#test_api_generate_error_body_never_echoes_caller_input"
        status: pass
    human_judgment: false
  - id: D6
    description: Boundary-valid n=1 and n=10 are not schema-rejected, proving no new limit was introduced
    requirement: SEC-04
    verification:
      - kind: unit
        ref: "tests/test_web_server.py#test_api_generate_accepts_n_at_both_bounds"
        status: pass
    human_judgment: false
  - id: D7
    description: Static guards that the boundary emits no 422 and that the Phase 4 catch-all 500 body (str(e)) is untouched
    requirement: SEC-04
    verification:
      - kind: unit
        ref: "tests/test_web_server.py#test_no_422_for_malformed_input and #test_generate_catch_all_500_body_is_untouched"
        status: pass
    human_judgment: false

duration: 12min
completed: 2026-10-11
status: complete
---

# Phase 3: HTTP Boundary Summary

**One rejection test per malformed field plus a mock-backed provider-never-reached proof, a shared `client`/`engine_ready` fixture pair, and a frozen valid-path shape pin for `POST /api/generate`**

## Performance

- **Duration:** 12 min
- **Started:** 2026-10-11T00:05:00Z
- **Completed:** 2026-10-11T00:17:00Z
- **Tasks:** 2
- **Files modified:** 1 (`tests/test_web_server.py`, 17 → 45 tests)

## Accomplishments

- 28 new tests cover every field the sculptor model constrains: `size` out-of-enum, `n` non-integer / below bound / above bound, `quality`, `style`, `response_format`, a non-string `prompt`, and `output` in absolute, traversal, and subdirectory form. Each asserts 400, `success is False`, the field key in the error body, and that the mocked provider was never called.
- A single `@pytest.mark.parametrize` test iterates all nine malformed payloads and asserts `core.called is False` for every one — the "provider never reached" sentence from CONTEXT, backed by an `AsyncMock` so no live API call is possible (T-03-07).
- Added the shared `client` fixture the file lacked, plus an `engine_ready` fixture that patches `settings.default_engine` to a configured engine. Without it the engine guard fires before the schema gate and every rejection passes for the wrong reason; the fixture asserts the precondition once so a config variance cannot masquerade as a pass.
- Pinned the collapsed build twice over: a `pydantic` subclass with a recording `__init__` proves exactly one construction happens and that it receives the *decoded* input-image bytes, and a paired-request comparison proves the with-image build equals the no-image build on every other field.
- Pinned the frozen valid path: 200, top-level `{success, results, command}`, and each result carries exactly `index, success, error, saved_path, image_data, image_url, image_b64_json` (T-03-06).
- Static guards keep the surrounding contracts honest: `test_no_422_for_malformed_input` (no status other than 400 for malformed input) and `test_generate_catch_all_500_body_is_untouched` (Phase 3 did not start Phase 4's error propagation). `test_api_generate_contract_unchanged` and all 17 pre-existing tests remain untouched and green.

## Task Commits

Each task was committed atomically:

1. **Task 1: One rejection test per field + parametrized provider-never-reached** - `c83499a` (test)
2. **Task 2: Preserve the happy-path shape and pin every boundary regression guard** - `ed0ff35` (test)

## Files Created/Modified

- `tests/test_web_server.py` — added module-level imports and two fixtures (`client`, `engine_ready`), a `_dispatch` recorder helper, and 28 new tests below the existing 17. No existing test was edited, weakened, or removed.

## Decisions Made

- **`engine_ready` fixture instead of per-test patching.** The critical seam in this plan: `generate_image()` checks engine presence and configuration *before* the pydantic gate, and `settings.default_engine` is unset in this repo, so an unpatched payload 400s on the engine for the wrong reason. One fixture resolves a configured engine and asserts `name in settings.engines` — the plan's "assert this precondition once so a config-variance can't masquerade as a pass".
- **`AsyncMock` for the provider seam.** The endpoint does `asyncio.run(generate_image_core(req))`, so the stand-in must be awaitable-returning. `AsyncMock(return_value=[])` gives both the await and `core.called is False` in the file's existing assert style.
- **Behavioural pin over source grep for the splat.** `test_api_generate_absent_fields_keep_model_defaults` asserts `seen[0].n == 1` on the request the mocked core was handed, rather than grepping for the dict comprehension. The splat's failure mode is a silent `None`, which only shows on the built model.
- **`tmp_path` over `tempfile.TemporaryDirectory()` for the new tests.** The output-containment tests need the output dir nested one level inside the temp root so that `../../escape.png` resolves to a path the test itself owns — that makes "nothing was written outside `output_dir`" a real assertion instead of an assumption about a shared `/tmp`.

## Deviations from Plan

None - plan executed exactly as written. No source file was modified.

## Issues Encountered

- Two checkpoint-6 edits were made to `src/imagai/web_server.py` and reverted (see the RED-proof table below). Both were reverted with `git restore` and `git status --short src/` confirms the tree is clean at `ed0ff35`; the final run is 45 passed.

## Verification

Run from the repo root with `uv` (`/Users/ali/.local/bin/uv`):

| Check | Command | Result |
|---|---|---|
| 1 | `uv run pytest tests/test_web_server.py -q` | `45 passed` (baseline 17 — strictly more) |
| 2 | `uv run pytest tests/test_containment.py -q` | `13 passed` (unchanged) |
| 3 | `uv run pytest -k "not llm" -q` | `65 passed, 1 deselected` (baseline 37 — not dropped) |
| 4 | `rg -n "shell=True" src/ tests/` | empty |
| 5 | `rg -n "openai\|AsyncOpenAI\|httpx" tests/test_web_server.py` | empty |
| 6 | RED proofs below | 3 of 3 confirmed |

**Check 5 detail — live-API reachability of the new tests.** The grep is empty: no new test imports `openai`, `AsyncOpenAI`, or `httpx`, and no new test posts an unmocked valid payload. Every test that can reach the provider path patches `web_server.generate_image_core` (an `AsyncMock` returning `[]`, or a recorder returning `[]`). The remaining import dependency is transitive and unmodified: `imagai.web_server` → `imagai.core` → `imagai.providers.openai_sdk_provider` constructs an `AsyncOpenAI` *client object* but opens no socket, and no new test reaches it because the module attribute is patched before the request.

**Check 6 detail — the tests bite.** Each behaviour was reverted in a scratch edit to `web_server.py`, the relevant test observed RED, then the file was restored:

| Reverted behavior | Test(s) that caught it | Result |
|---|---|---|
| Re-added `int()` coercion on `n` | `test_api_generate_rejects_non_integer_n`, `test_api_generate_malformed_payload_never_reaches_provider[malformed1]`, `test_api_generate_error_body_never_echoes_caller_input` | 3 FAILED |
| Deleted the `_contained_path` output pre-check | `test_api_generate_rejects_absolute_output`, `test_api_generate_rejects_traversal_output`, `test_api_generate_rejects_subdirectory_output`, `test_api_generate_malformed_payload_never_reaches_provider[malformed7]`, `[malformed8]` | 5 FAILED |
| Replaced the only-present-keys splat with a `.get()` None fallback for `n` | `test_api_generate_absent_fields_keep_model_defaults` | 1 FAILED |
| Dropped `result_data["image_url"]` from the response block | `test_api_generate_valid_request_returns_frozen_results_shape` | 1 FAILED |

All three source edits were reverted with `git restore src/imagai/web_server.py`; the final state is 45 passed with `src/` clean. No source edit was ever committed.

## Self-Check: PASSED

- `tests/test_web_server.py` exists on disk with 45 collected tests (17 pre-existing + 28 new).
- Both task commits are on branch `gsd/phase-01-runnable-toolchain`: `c83499a`, `ed0ff35`.
- Every task `<verify>` command re-run after the final task: `45 passed`.
- Plan-level `<verification>`: `65 passed, 1 deselected` under the mandatory `-k "not llm"` filter.
- No source file modified — `git status --short src/` is empty.

## User Setup Required

None - no external service configuration required. The suite runs fully offline; the only pre-existing live-API test remains the deselected `test_llm_*` case, which is why the `-k "not llm"` filter is mandatory.

## Next Phase Readiness

- SEC-04's rejection contract is now regress-proof: any field that stops returning 400, any payload that reaches the provider, or any dropped response key turns the suite red.
- The Phase 4 catch-all 500 at `web_server.py:250` and its `str(e)` leak are pinned-but-unfixed, and `test_generate_catch_all_500_body_is_untouched` will need updating when Phase 4 (ERR-01/ERR-02) legitimately changes that body.
- The Phase 2.5 residual (a concurrent `/api/generate` writing into `settings.output_dir` landing its file in a `/api/generate-cli` response) is untouched and remains open; this plan neither closed nor regressed it.
- `.planning/STATE.md` and `.planning/ROADMAP.md` were deliberately not updated — the orchestrator commits those after the wave.

---

*Phase: 03-http-boundary*
*Completed: 2026-10-11*
