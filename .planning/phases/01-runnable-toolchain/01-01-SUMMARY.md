---
phase: 01-runnable-toolchain
plan: 01
subsystem: toolchain
tags: [uv, pyproject, migration, lockfile]
dependency_graph:
  requires: []
  provides: [uv-toolchain, uv-lock, python-3.9-floor]
  affects: [all-later-phases]
tech_stack:
  added: []
  patterns: [PEP-735-dependency-groups, stdlib-typing-Annotated]
key_files:
  created: [uv.lock]
  modified: [pyproject.toml, src/imagai/cli.py]
  deleted: [requirements.lock, requirements-dev.lock]
decisions:
  - "Remove [tool.rye] entirely per D-03"
  - "Use PEP 735 [dependency-groups] per D-01"
  - "requires-python >=3.9 per D-04"
  - "Remove requests per D-10"
  - "Keep werkzeug per D-09"
  - "stdlib typing.Annotated per D-07"
metrics:
  duration: 15m
  completed: "2026-10-03"
status: complete
actuals:
  tokens: 73661
  tasks: 3
  commits: 2
plan_head_before: 483bb4b274d3a3b19818a171896f9f1b40a8c6df
---

# Phase 01 Plan 01: Runnable Toolchain Summary

Migrate project toolchain from rye to uv: rewrite pyproject.toml, fix typing import, replace lockfiles, verify clean install.

## What Was Done

### Task 1: Rewrite pyproject.toml for uv and fix typing import
- Removed `[tool.rye]` section entirely
- Added `[dependency-groups]` with `dev = ["pytest>=7.0.0"]` (PEP 735)
- Changed `requires-python` from `>=3.8` to `>=3.9`
- Removed `requests>=2.32.5` from dependencies (unused — httpx handles all HTTP)
- Changed `src/imagai/cli.py` line 2 from `from typing_extensions import Annotated` to `from typing import Annotated`
- Preserved: `[build-system]`, `[tool.hatch.metadata]`, `[tool.hatch.build.targets.wheel]`, `[project.scripts]`

### Task 2: Replace rye lockfiles with uv.lock
- Deleted `requirements.lock` and `requirements-dev.lock` (rye-generated)
- Generated `uv.lock` via `uv lock` (61 packages resolved)

### Task 3: Clean install and verify toolchain
- Deleted local `.venv/` (Windows build)
- `uv sync` created fresh macOS venv with 40 packages
- `uv run pytest` — 2 passed
- `uv run python -V` — Python 3.12.9
- `uv run python -c "import imagai.cli"` — import OK

## Deviations from Plan

None — plan executed exactly as written.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | 4dd6d2e | feat(01-01): migrate pyproject.toml to uv and fix typing import |
| 2 | 8c95018 | feat(01-01): replace rye lockfiles with uv.lock |
| 3 | — | verification-only, no file changes |

## Self-Check: PASSED

- [x] pyproject.toml exists and is valid
- [x] src/imagai/cli.py exists with correct import
- [x] uv.lock exists and is non-empty (287KB)
- [x] requirements.lock deleted
- [x] requirements-dev.lock deleted
- [x] Commit 4dd6d2e exists
- [x] Commit 8c95018 exists
