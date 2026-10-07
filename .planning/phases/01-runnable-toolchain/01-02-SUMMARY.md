---
phase: 01-runnable-toolchain
plan: 02
subsystem: toolchain
tags: [uv, docs, migration, rye-removal]
dependency_graph:
  requires: [01-01]
  provides: [uv-native-docs, zero-rye-surface]
  affects: [all-later-phases]
tech_stack:
  added: []
  patterns: []
key_files:
  created: []
  modified: [src/imagai/cli.py, README.md, docs/testing.md, docs/dependencies.md, web_interface.html, src/imagai/web_server.py]
  deleted: []
decisions:
  - "Replace all rye references with uv equivalents per D-02"
  - "Use uv run imagai pattern for README commands"
  - "Use uv python pin for Python version switching"
  - "Use uv lock --update-package for lockfile updates"
metrics:
  duration: 96m
  completed: "2026-10-03"
status: complete
actuals:
  tokens: 1922
  tasks: 3
  commits: 3
plan_head_before: 34702bfcd0e376dfa7bacac250edd1b053022f5c
---

# Phase 01 Plan 02: Remove Rye References Summary

Remove all rye references from build/docs surface — cli.py error messages, README.md, docs/, web_interface.html, web_server.py — leaving the project fully uv-native.

## What Was Done

### Task 1: Update rye error messages in cli.py
- Line 361: `rye add requests && rye sync` → `uv add requests && uv sync`
- Line 416: `rye sync` → `uv sync`
- Lazy `import requests as _requests` at line 306 left unchanged per PATTERNS.md

### Task 2: Update README.md to uv commands
- Setup section: Rye → uv, `rye sync` → `uv sync`
- Quick start: `rye run` → `uv run` for all commands
- Common tasks: `rye run pytest` → `uv run pytest`, `rye add` → `uv add`, `rye lock` → `uv lock`, `rye build` → `uv build`
- Note section: `rye pin` → `uv python pin`

### Task 3: Remove rye references from docs/ and web files
- docs/testing.md: `[tool.rye]` → `[dependency-groups]`
- docs/dependencies.md: `tool.rye dev-dependencies` → `dependency-groups dev`, `requirements-dev.lock` → `uv.lock`, `rye lock --update` → `uv lock --update-package`
- web_interface.html: `rye run imagai generate` → `uv run imagai generate`
- web_server.py: `rye run imagai` → `uv run imagai`

## Deviations from Plan

None — plan executed exactly as written.

## Commits

| Task | Commit | Description |
|------|--------|-------------|
| 1 | 1b188e5 | fix(01-02): update rye error messages to uv in cli.py |
| 2 | d64a306 | docs(01-02): migrate README.md from rye to uv commands |
| 3 | 4c0c677 | docs(01-02): remove rye references from docs and web files |

## Self-Check: PASSED

- [x] src/imagai/cli.py has zero rye references
- [x] README.md has zero rye references
- [x] docs/ has zero rye references
- [x] web_interface.html has zero rye references
- [x] src/imagai/web_server.py has zero rye references
- [x] rg -n rye --glob '!.planning/**' returns no matches
- [x] Commit 1b188e5 exists
- [x] Commit d64a306 exists
- [x] Commit 4c0c677 exists
