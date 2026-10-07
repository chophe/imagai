---
phase: 02-path-containment
plan: 01
subsystem: utils
tags: [security, path-containment, sec-01]
dependency_graph:
  requires: []
  provides: [_contained_path helper, save_image_from_url containment, save_image_from_b64 containment]
  affects: [src/imagai/utils.py, tests/test_containment.py]
tech_stack:
  added: []
  patterns: [path-containment-check, resolve-canonicalization]
key_files:
  created:
    - tests/test_containment.py
  modified:
    - src/imagai/utils.py
decisions:
  - "_contained_path rejects absolute paths, .. traversal, and paths outside settings.output_dir"
  - "Both save functions return None (no exception) on containment rejection per D-04"
  - "Containment check is first statement in try block, before any I/O or network activity"
metrics:
  duration: 4m32s
  completed: "2026-10-06T01:03:50Z"
  tasks: 3
  commits: 3
status: complete
actuals:
  tokens: 1820
  tasks: 3
  commits: 3
---

# Phase 2 Plan 1: Path Containment Enforcement Summary

Added `_contained_path` helper to utils.py and wired it into both save functions so any output path escaping `settings.output_dir` is rejected with `None` (no exception), closing the path-traversal write hole for both CLI and web server.

## What Was Done

### Task 1: Add _contained_path helper to utils.py
- Added `_contained_path(output_path: Path) -> bool` helper after `sanitize_filename`
- Rejects absolute paths, `..` traversal, and paths that don't resolve under `settings.output_dir`
- Canonicalizes via `Path.resolve()` to prevent symlink escapes
- Returns `bool`, never raises

### Task 2: Wire _contained_path into both save functions
- Added containment check as first statement in `save_image_from_url` try block
- Added containment check as first statement in `save_image_from_b64` try block
- Both functions log warning and return `None` on rejection (D-04 contract)

### Task 3: Write SEC-01 rejection tests
- Created `tests/test_containment.py` with 5 pytest tests
- Tests cover absolute path, traversal, and subdirectory rejection for both save functions
- All 5 tests pass

## Deviations from Plan

None - plan executed exactly as written.

## Self-Check: PASSED

- [x] src/imagai/utils.py modified with _contained_path helper
- [x] tests/test_containment.py created with 5 tests
- [x] All 3 commits exist: bf616b5, 9c43b89, b70c9f0
- [x] All 5 tests pass under `uv run pytest tests/test_containment.py -v`
- [x] _contained_path helper works correctly per verify command
- [x] Both save functions reject escaping paths with None
