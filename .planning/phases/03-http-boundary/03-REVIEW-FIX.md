---
phase: 3
phase_name: HTTP Boundary
fixed_at: 2026-10-10T22:46:14Z
review_path: .planning/phases/03-http-boundary/03-REVIEW.md
iteration: 1
findings_in_scope: 5
fixed: 5
skipped: 0
status: all_fixed
---

# Phase 3: Code Review Fix Report

**Fixed at:** 2026-10-10T22:46:14Z
**Source review:** `.planning/phases/03-http-boundary/03-REVIEW.md`
**Iteration:** 1

**Summary:**
- Findings in scope: 5 (`CR-01`, `WR-01`, `WR-02`, `WR-03`, `WR-04` — critical + warning
  under the default `critical_warning` scope; `--all` was not passed, so `IN-05/06/07` are
  recorded, not fixed)
- Fixed: 5
- Skipped: 0

## Fixed Issues

### CR-01: malformed `seed`/`strength` bypass the new gate and 500 with the caller's value echoed

**Files modified:** `src/imagai/web_server.py`, `tests/test_web_server.py`
**Commit:** `b39287a` — fix(03): CR-01 reject a non-numeric seed/strength with a field-level 400
**Applied fix:** the `seed`/`strength` coercion in the `param_mapping` loop is now wrapped in
`try/except (TypeError, ValueError)` and returns a field-level 400 `{"seed": "seed must be a
number"}` / `{"strength": "strength must be a number"}` — the field name only, no caller value.

**Why validated in the param loop rather than moved under the gate:** `seed` and `strength` are
provider `extra_params` that `ImageGenerationRequest` does not model (`extra_params:
Optional[dict]` is a free-form dict, so pydantic would not check the numeric-ness of its
members), and adding them to the model would make every consumer of the shared request — CLI,
core, provider — carry two web-boundary extras it never passes. CONTEXT forbids a new request
schema and names "validate the type where the coercion lives" as an acceptable shape, so the
coercion stays where the phase put it and gains the same 400 contract the pydantic gate gives
the model fields. The alternative (model fields) would also change the error body's *shape*
for these two params — they would arrive on the request rather than in `extra_params`, which
the provider contract does not expect.

### WR-01: malformed `input_image` bypasses the gate and 500s

**Files modified:** `src/imagai/web_server.py`, `tests/test_web_server.py`
**Commit:** `a37a2a6` — fix(03): WR-01 reject a malformed input_image with a field-level 400
**Applied fix:** an `isinstance(input_image, str)` guard answers 400 before any decode, and the
decode itself is wrapped in `try/except (ValueError, IndexError)` — `binascii.Error` is a
`ValueError`, so bad base64, a comma-less data URL, and a non-string each become
`{"input_image": "input_image must be valid base64"}` or `{"input_image": "input_image must be a
string"}`. A valid decode is unchanged (pinned by its own control test).

### WR-02: unhashable `engine` 500s the guard chain one line above the new gate

**Files modified:** `src/imagai/web_server.py`, `tests/test_web_server.py`
**Commit:** `175f56a` — fix(03): WR-02 reject a non-string engine with a field-level 400
**Applied fix:** an `isinstance(engine, str)` guard was inserted immediately before the
`engine not in settings.engines` membership test, returning `{"engine": "engine must be a
string"}`. This also covers a *hashable* non-string engine (an int): the membership test
already answered 400 there, but with `Engine "1234567" is not configured`, which quoted the
caller's value — now it does not. Guard order is otherwise unchanged (presence → configured →
pydantic gate), and the string guard's own message is untouched.

### WR-03: request-body malformation still 500s both endpoints

**Files modified:** `src/imagai/web_server.py`, `tests/test_web_server.py`
**Commit:** `0996b3d` — fix(03): WR-03 return a 400 for an unreadable request body on both endpoints
**Applied fix:** `request.get_json()` → `request.get_json(silent=True)` on `/api/generate` and
`/api/generate-cli`, so Flask's own 400 (malformed JSON) and 415 (wrong Content-Type) become
`None` instead of an exception that the outer `except` answered with 500. The new
`if not isinstance(data, dict)` guard then returns a clean `"Request body must be a JSON
object"` 400. It also closes the layer below the one the review named: a JSON *array* body
passed the key test by membership and raised `TypeError` on `data["prompt"]` /
`data["command"]` into the same 500. The pre-existing `Prompt is required` / `Command is
required` guards and their exact strings are untouched (`json={}` still answers the contract
test's string).

### WR-04: no test pins `seed`/`strength`/`input_image`/`engine` malformation

**Files modified:** `tests/test_web_server.py`
**Commit:** `5a1b314` — test(03): WR-04 pin the non-model params the SEC-04 block never covered
**Applied fix:** test-only. Added the cross-cutting coverage the SEC-04 block lacked —
`test_api_generate_malformed_params_never_reaches_provider` (the same 4-case
provider-never-reached proof for the non-model fields) and
`test_api_generate_param_error_body_never_echoes_caller_input` (the T-03-03 sentinel across
`seed`/`strength`/`input_image`). No existing test was edited, weakened, deleted, or skipped.

The per-field tests for these same params were committed with their own fixes (see CR-01/WR-01/
WR-02 above), so this finding's RED proof is the source reverting to its whole pre-review state
rather than one hunk — see the verification table.

## Notes / not fixed

### IN-05: provider-specific params still cross into the provider unvalidated (by design)

`negative_prompt`, `mode`, `aspect_ratio`, `output_format` still ride `extra_params` raw into
the provider's `extra_body` (`openai_sdk_provider.py:198-212`). Not fixed: it is a pre-existing
pass-through, not a phase regression, and validating provider-specific vocabularies belongs to
the provider-registry phase, not the HTTP boundary. Note that this same review's verdict #4
("a 500 is still reachable with a malformed payload") was true *only* for these fields; after
CR-01/WR-01/WR-02 the first malformed **non-provider** field is now rejected at the boundary
before the gate, and the loop no longer coerces anything the provider could receive unprepared
except these four pass-throughs.

### IN-06: `output: ""` returns 200 and reaches the provider

`web_server.py:186`'s truthiness skip plus `core.py:47`'s `if base_filename:` treat `""` as
absent, so the bare-filename contract is not enforced for the empty string. Not fixed: the
review rates it harmless (auto-naming follows, no containment bypass), and this fix pass is
scoped to "malformed input must not 500". Worth a line in the Phase 4 error-propagation
cleanup rather than a behaviour change here.

### IN-07: engine-guard 400s echo the caller's engine string

`web_server.py:150` still answers `Engine "X" is not configured` for a *string* engine.
Confirmed by spec and left alone: the phase's guard-order spec assigns string 400s to that
guard, and it leaks no configuration. WR-02's fix added the non-string branch only, which
carries no caller value — so the by-design echo is unchanged and the new echo surface is nil.

## Verification

**Where verification ran (per #2825):** in the main checkout, not an isolated worktree. This
runtime has no `Agent()` dispatch tool, so the `gsd-code-fixer` role was executed in-process,
and the worktree step was deliberately skipped: the hand-rolled worktree has no `.venv` (gitignored)
and this environment has no network, so `uv run pytest` cannot run there at all; the phase
artifacts under review (03-REVIEW.md and friends) are also uncommitted in the main checkout.
The branch named by the orchestrator (`gsd/phase-01-runnable-toolchain`) was used directly, and
`workflow.use_worktrees` was left unset rather than opted out.

| Check | Command | Result |
|---|---|---|
| 1 | `uv run pytest -k "not llm" -q` | `90 passed, 1 deselected` (baseline 65 — strictly more) |
| 2 | `uv run pytest tests/test_web_server.py -q` | `70 passed` (baseline 45: 17 pre-existing + 28 from plan 03-02) |
| 3 | `git diff 75ef854 -- tests/test_web_server.py` | addition-only — zero deletions, all 45 pre-existing tests present |
| 4 | `git diff 75ef854 -- src/imagai/web_server.py` | hunks at 98/118/136/304 only; the `generate_image` catch-all at :320-323 and its `{"error": str(e), "type": type(e).__name__}` 500 body are byte-identical |
| 5 | `rg -n "openai\|AsyncOpenAI\|httpx" tests/test_web_server.py` | empty |
| 6 | Live-API reachability | see below |
| 7 | RED proofs | 6/6, 4/4, 3/3, 6/6, 5/5 — see below |

**Red proofs (each: write test → revert-behavior → observed RED → restore).** For CR-01, WR-01,
WR-02 and WR-03 the test was written against the unmodified source and failed there:

| Finding | Test(s) | Failure before the fix | After the fix |
|---|---|---|---|
| CR-01 | `test_api_generate_rejects_non_numeric_seed_or_strength` (6 params) | `6 failed` — `assert 500 == 400` | 6 passed |
| WR-01 | `test_api_generate_rejects_malformed_input_image` (4 params) | `4 failed` — `assert 500 == 400`; the two valid-path controls passed before and after | 4 passed, controls green |
| WR-02 | `test_api_generate_rejects_non_string_engine` (3 params) | `3 failed` — 500 for dict/list, quoted-value 400 for the int | 3 passed |
| WR-03 | `test_malformed_request_body_is_a_400_not_the_catch_all_500` (2 endpoints × 3 body kinds) | `6 failed` — `assert 500 == 400` | 6 passed |
| WR-04 | `test_api_generate_malformed_params_never_reaches_provider` (4 params), `test_api_generate_param_error_body_never_echoes_caller_input` | `5 failed` — with `src/imagai/web_server.py` restored to its whole pre-review state (`75ef854`), then re-restored to HEAD | 5 passed |

The WR-04 restore was `git restore --source=75ef854` followed by `git restore --staged
--worktree`; the tree is clean under `src/` and the final full-suite run (check 1) came after
the restore.

**Live-API reachability (check 6).** The grep is empty — no test file imports `openai`,
`AsyncOpenAI`, or `httpx`. An AST audit of every `client.post("/api/generate"` in the file
confirms each one sits inside a `patch.object(web_server, "generate_image_core", ...)` block,
so no new unmocked post can reach the provider. The single exception is the pre-existing
`test_api_generate_contract_unchanged` (`json={}`, stopped by the prompt guard, documented in
the original review). The transitive import chain
(`web_server → core → providers/openai_sdk_provider`) constructs an `AsyncOpenAI` *client
object* and opens no socket.

**Post-fix behaviour, re-probed with the provider mocked:** every payload above now answers 400
with a field-level, echo-free body (`{"error":{"seed":"seed must be a number"}}`,
`{"error":{"input_image":"input_image must be valid base64"}}`,
`{"error":{"engine":"engine must be a string"}}`, `{"error":"Request body must be a JSON
object"}`) and `generate_image_core` is never called. A valid request still returns 200, and a
*falsy* non-string engine (`{"engine": []}`, `{"engine": 0}`) still resolves to the default
engine via the pre-existing `or` fallback — unchanged, and not a 500, so it stays as it was.

---

_Fixed: 2026-10-10T22:46:14Z_
_Fixer: the agent (gsd-code-fixer), executed in-process — no `Agent()` dispatch in this runtime_
_Iteration: 1_
