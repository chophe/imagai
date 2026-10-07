---
phase: 02-path-containment
verified: 2026-10-06T00:00:00Z
status: passed
score: 7/7 must-haves verified
requirements:
  SEC-01: verified
  SEC-02: verified
plans_verified:
  - 02-01-PLAN.md
  - 02-02-PLAN.md
review_findings_resolved:
  CR-01: fixed (commit 558c314) — re-verified, fix holds
  CR-02: out of scope (pre-existing, Phase 3 SEC-04)
  CR-03: out of scope (pre-existing, Phase 3 SEC-03)
  WR-01: open (non-blocking, test naming/CWD fragility)
  WR-02: open (non-blocking, latent network dep)
  IN-01: open (non-blocking, symlink behavior verified manually, untested in suite)
  IN-02: open (non-blocking, test writes to real output dir)
  IN-03: open (non-blocking, LLM test makes live API call)
---

# Phase 2 Verification: Path Containment

**Goal:** The image save pipeline can only write inside the configured output directory
**Mode:** mvp
**Verified:** 2026-10-06

## Verdict

**PASSED — 7/7 must-haves verified.**

The CR-01 fix (commit 558c314) holds. `_contained_path` now correctly accepts an absolute
`settings.output_dir`, and still rejects absolute attack paths, `..` traversal, subdirectories,
and symlink escapes. All 12 non-LLM tests pass; the 13th (`test_llm_filename_contained`) is
excluded because it makes a live API call. Four review warnings/info items remain open but none
blocks the phase goal.

## Must-Have Verification

### Plan 02-01 (SEC-01)

| # | Must-have | Status | Evidence |
|---|-----------|--------|----------|
| 1 | `_contained_path` returns True for paths under `settings.output_dir` and False for absolute paths, `..`, and separator paths | **VERIFIED** | `src/imagai/utils.py:26-46`. Direct probe: `output_dir/test.png`→True, `/tmp/evil.png`→False, `../../escape.png`→False, `sub/img.png`→False, `sub/../x.png`→False |
| 2 | `save_image_from_url` / `save_image_from_b64` return `None` when `output_path` escapes | **VERIFIED** | `utils.py:182` and `:223` — check is the first statement in each `try` block. Probe: all 6 malicious variants (b64 + url × absolute/traversal/subdir) returned `None` |
| 3 | No file is written outside `settings.output_dir` for any malicious path | **VERIFIED** | After all 6 rejection probes, `/tmp/evil*.png`, `../../escape*.png`, and `sub/img.png` all confirmed absent. Symlink-file probe: `out/evil.png -> /tmp/target.png` rejected, target byte count unchanged |

### Plan 02-02 (SEC-02)

| # | Must-have | Status | Evidence |
|---|-----------|--------|----------|
| 4 | `web_server.py` `UPLOAD_FOLDER == Path(settings.output_dir)` | **VERIFIED** | `web_server.py:33`. Import check: both resolve to `generated_images`. Only line 33 changed in that file (diff confirmed) |
| 5 | All four filename strategies produce contained paths | **VERIFIED** | manual + `n>1` + prompt-derived + random by test; LLM strategy verified structurally — `generate_filename_from_prompt_llm` runs output through `sanitize_filename` (`utils.py:131`) then appends a timestamp. Adversarial inputs (`../../etc/passwd`, `a/b/c.png`, `/tmp/evil.png`, `sub/img.png`, `x\y.png`) all sanitize to bare basenames with `_contained_path` True |
| 6 | `n > 1` numbered variants pass `_contained_path` | **VERIFIED** | `my_image_2.png`, `photo_2.png`, `photo_3.png` all True. End-to-end via `generate_image_core` with `n=3`: `photo_1/2/3.png` written, all resolved under a temp output dir |
| 7 | `my_image.png` writes to `settings.output_dir/my_image.png` (happy path unbroken) | **VERIFIED** | `save_image_from_b64` returned a Path and the file exists under both the default relative `output_dir` and a patched absolute one |

## CR-01 Fix — Direct Behavior Verification

The fix replaced the blanket `is_absolute()` early-reject with
`canonical_output.parent != canonical_root`. Probed against a temp directory with an **absolute**
`output_dir` patched in:

| Input | Result | Correct? |
|-------|--------|----------|
| `<abs_dir>/my_image.png` | accepted | ✅ |
| `/tmp/evil.png` | rejected | ✅ |
| `<abs_dir>/sub/img.png` | rejected | ✅ (D-02 bare basename) |
| `<abs_dir>/../x.png` | rejected | ✅ |
| `<abs_dir>/evil.png` (symlink → outside) | rejected | ✅ (D-03) |
| `<abs_dir>/real.png` | accepted | ✅ |
| `save_image_from_b64(<abs_dir>/real.png)` | Path returned, file written | ✅ |
| `save_image_from_b64(/tmp/evil_abs.png)` | `None`, no file | ✅ |

Regression test `test_absolute_output_dir_still_contains_basename` (`tests/test_containment.py:119-131`)
locks this in and passes.

## Success Criteria Cross-Check (ROADMAP Phase 2)

| Criterion | Status |
|-----------|--------|
| 1. `output=/tmp/evil.png` rejected with an error, no file at `/tmp/evil.png` | ✅ verified by test |
| 2. `output=../../escape.png` rejected, nothing written outside output dir | ✅ verified by test |
| 3. All four filename strategies resolve under `settings.output_dir`, asserted per strategy | ✅ (LLM proven structurally — see caveat W3) |
| 4. Legitimate filename + `n>1` variants land in the output dir | ✅ verified, including through `generate_image_core` |

## Requirement Traceability

| ID | In PLAN frontmatter | In REQUIREMENTS.md | Status |
|----|---------------------|-------------------|--------|
| SEC-01 | 02-01-PLAN.md | line 20, mapped Phase 2 | **VERIFIED** |
| SEC-02 | 02-01-PLAN.md, 02-02-PLAN.md | line 21, mapped Phase 2 | **VERIFIED** |

Every requirement ID claimed by the plans is accounted for in REQUIREMENTS.md, and both are
assigned to Phase 2 in the traceability table. No orphans, no unmapped IDs. The `Pending` status
in the REQUIREMENTS.md traceability table is a bookkeeping field updated at milestone close, not a
verification gap.

## End-to-End Check Through `generate_image_core`

With the provider stubbed (no network), `generate_image_core` populated
`ImageGenerationResponse.error` and left `saved_path=None` for every escaping filename:

```
'/tmp/evil.png'     → error='Failed to save image to /tmp/evil.png'    /tmp/evil.png absent
'../../escape.png'  → error='Failed to save image to generated_images/../../escape.png'  absent
'sub/img.png'       → error='Failed to save image to generated_images/sub/img.png'      absent
```

Happy path with `n=3` into a temp dir wrote `photo_1.png`, `photo_2.png`, `photo_3.png`, all
resolving under that dir. This exercises the real seam at `core.py:76` →
`save_image_from_*`, confirming D-01 (enforcement in `utils.py` protects both front ends).

## D-01..D-05 Decision Compliance

| Decision | Honored | Evidence |
|----------|---------|----------|
| D-01 enforcement in `utils.py`, not `core.py` | ✅ | Check lives in the two save functions; `core.py` unchanged by this phase |
| D-02 bare basename only; absolute and `..` rejected | ✅ | Enforced via `..`-in-parts check + `parent == root` |
| D-03 resolve symlinks, reject escapes | ✅ | `resolve()` on both sides; symlink-file and symlink-dir escapes rejected |
| D-04 return `None`, no exception | ✅ | Both functions `return None`; the enclosing `except` is never reached. `core.py:97` sets the error |
| D-05 `UPLOAD_FOLDER` unified to `settings.output_dir` | ✅ | `web_server.py:33`; import-time `mkdir` left in place for Phase 5 per plan |

## Test Suite

```
uv run pytest tests/ -k "not llm" -q
→ 14 passed, 1 deselected in 7.02s
```

12 of 13 in `tests/test_containment.py` pass (5 SEC-01 + 7 SEC-02). The deselected test is
`test_llm_filename_contained`, which makes a live API call because an engine key **is** configured
in this environment (verified: `openai_dalle3.api_key` set, `filename_generation` engine present).
Its assertion is redundant given the structural proof in must-have #5, so excluding it costs no
coverage of the containment contract.

## Open Findings (non-blocking)

**W1 — WR-01, test name/CWD fragility.** `test_save_image_from_b64_rejects_subdirectory`
(`tests/test_containment.py:61-66`) asserts `Path("sub/img.png")` is rejected. Under D-02
(bare basename is the contract) this is the *intended* behavior, so the test is right and the
review's claim that it "describes behavior that doesn't exist" does not hold post-fix. It is
CWD-independent in practice: `parent` is `<cwd>/sub`, which only equals `output_dir` in the
degenerate case where the CWD's child directory happens to be named identically to `output_dir`.
The review's suggested rewrite (`_anchored_subdirectory_allowed` asserting
`output_dir/sub/img.png` is True) would be **wrong** under D-02 — do not apply it.

**W2 — WR-02, latent network dependency.** The two `save_image_from_url` rejection tests pass
`http://example.com/img.png`. Containment rejects before any HTTP call, so they do not hit the
network today, but a regression in the check would turn them into live requests. Worth pointing
at an unreachable address (`http://127.0.0.1:9/`) for fast failure.

**W3 — IN-03, live API call in the unit suite.** `test_llm_filename_contained` calls the real
provider when a key is configured — slow, costly, flaky. Fix by forcing the no-key fallback or
monkeypatching. Not blocking: the strategy's containment is structurally guaranteed by
`sanitize_filename`.

**W4 — IN-01, no symlink test in the suite.** No test creates a symlink inside `output_dir`
pointing outside. The behavior **is** correct (verified manually above for both a symlinked
directory and a symlinked file), but it is unpinned against regression. D-03's threat
mitigation would benefit from a test.

**W5 — IN-02, test writes to the real output directory.**
`test_happy_path_writes_to_output_dir` writes `my_image.png` into the user's actual
`settings.output_dir` and never cleans up. Fix with `tmp_path` + `monkeypatch` on
`settings.output_dir`. (I removed the artifact this run created.)

**W6 — pre-existing, not this phase's debt.** `web_server.py:250` `shell=True` (CR-02) and
`main()` called before definition (CR-03) remain open, both assigned to Phase 3.
`tests/test_cli.py:2,6` still contain two `assert True` placeholders — Phase 4/6 own those.

## Scope Boundary

Phase 2 delivers containment at the save pipeline only. It does not validate HTTP payloads
(Phase 3, SEC-04), propagate errors to the user (Phase 4), remove import-time side effects
(Phase 5, owns `config.py:58` and `web_server.py:34` mkdirs), or touch the provider seam
(Phase 6). No regressions in those areas from this phase.