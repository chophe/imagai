---
phase: 02-path-containment
plan: 02
subsystem: web_server
tags: [security, path-containment, sec-02]
dependency_graph:
  requires: [02-01]
  provides: [web_server UPLOAD_FOLDER unification, SEC-02 comprehensive tests]
  affects: [src/imagai/web_server.py, tests/test_containment.py]
tech_stack:
  added: []
  patterns: [single-source-of-truth, settings-driven-config]
key_files:
  created: []
  modified:
    - src/imagai/web_server.py
    - tests/test_containment.py
decisions:
  - "UPLOAD_FOLDER unified to Path(settings.output_dir) — one source of truth for read/serve and write paths"
  - "All four filename strategies proven to produce contained paths via _contained_path"
  - "n>1 numbered variants (photo_2.png, photo_3.png) proven contained"
  - "Happy path preserved: my_image.png writes to settings.output_dir/my_image.png"
metrics:
  duration: 3m
  completed: "2026-10-06T01:15:00Z"
  tasks: 2
  commits: 2
status: complete
actuals:
  tokens: 2800
  tasks: 2
  commits: 2
---

# Phase 2 Plan 2: Web Server Unification + SEC-02 Tests Summary

Unified web_server.py's hardcoded UPLOAD_FOLDER to settings.output_dir (D-05) and wrote comprehensive SEC-02 tests proving all four filename strategies plus n>1 variants produce contained paths, and the happy path still works.

## What Was Done

### Task 1: Unify web_server.py UPLOAD_FOLDER to settings.output_dir
- Changed line 33 from `UPLOAD_FOLDER = Path("generated_images")` to `UPLOAD_FOLDER = Path(settings.output_dir)`
- All downstream references (mkdir, app.config, glob, send_from_directory) automatically use the unified value
- Closes T-02-03 threat: web server read/serve path matches write path

### Task 2: Write SEC-02 comprehensive containment tests
- Added 7 new test functions to tests/test_containment.py
- Tests cover all four filename strategies: manual, prompt-derived, random, LLM-generated
- n>1 numbered variants (photo_2.png, photo_3.png) proven contained
- Happy path test: my_image.png writes to settings.output_dir/my_image.png
- Web server UPLOAD_FOLDER unification test
- All 12 tests pass (5 SEC-01 + 7 SEC-02)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed async call in test_llm_filename_contained**
- **Found during:** Task 2 verification
- **Issue:** `generate_filename_from_prompt_llm` is an async function; calling it without `await` returned a coroutine, causing `TypeError: unsupported operand type(s) for /: 'PosixPath' and 'coroutine'`
- **Fix:** Wrapped the call in `asyncio.run()` with proper `await`
- **Files modified:** tests/test_containment.py
- **Commit:** 0ee0272

## Self-Check: PASSED

- [x] src/imagai/web_server.py line 33 reads `UPLOAD_FOLDER = Path(settings.output_dir)`
- [x] tests/test_containment.py contains 12 test functions (5 from Plan 01 + 7 new)
- [x] All 12 tests pass under `uv run pytest tests/test_containment.py -v`
- [x] test_manual_filename_contained asserts _contained_path returns True for "my_image.png" and "my_image_2.png"
- [x] test_prompt_derived_filename_contained asserts _contained_path returns True for generate_filename output
- [x] test_random_filename_contained asserts _contained_path returns True for generate_random_filename output
- [x] test_llm_filename_contained asserts _contained_path returns True for generate_filename_from_prompt_llm output
- [x] test_n_greater_than_1_variants_contained asserts _contained_path returns True for "photo_2.png" and "photo_3.png"
- [x] test_happy_path_writes_to_output_dir asserts the file exists at settings.output_dir/my_image.png
- [x] test_web_server_upload_folder_unified asserts UPLOAD_FOLDER == Path(settings.output_dir)
- [x] Both commits exist: 9a2b053, 0ee0272
