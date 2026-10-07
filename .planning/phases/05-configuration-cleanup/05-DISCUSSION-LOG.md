# Phase 5: Configuration Cleanup - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-04
**Phase:** 5-configuration-cleanup
**Areas discussed:** Engine env-var parsing, Default-engine merge, Output-dir creation point, web_server mkdir side effect

---

## Engine env-var parsing

| Option | Description | Selected |
|--------|-------------|----------|
| Native pydantic-settings + dummy api_key | Delete the os.environ loop; rely on native parsing (already handles .env.example names); make api_key default to "dummy" for partial configs; document `__`-in-engine-names as unsupported with a test | ✓ |
| Native + a model_validator shim | Delete the loop but add a model_validator that lowercases keys and fills missing api_key | |
| Custom EnvSettingsSource | Keep a custom source fully replicating the loop | |

**User's choice:** Native pydantic-settings + dummy api_key
**Notes:** A live experiment showed native pydantic-settings already reproduces the loop for every `.env.example` name; the only gap is the `api_key="dummy"` leniency. `__`-in-engine-name fails in both parsers, so it is documented unsupported and test-pinned.

---

## Default-engine merge

| Option | Description | Selected |
|--------|-------------|----------|
| Preserve current behavior | Built-in `openai_dalle3` default for the no-env case; env engines replace the dict (today's behavior) | ✓ |
| Merge env onto the default | Always keep `openai_dalle3` unless explicitly overridden | |
| Drop the built-in default | `engines` starts empty, populated only from env | |

**User's choice:** Preserve current behavior
**Notes:** Verified by experiment that env replaces the whole dict in both the loop and native pydantic-settings, so this is not a regression.

---

## Output-dir creation point

| Option | Description | Selected |
|--------|-------------|----------|
| Rely on utils.py write-time mkdir | utils.py already does `output_path.parent.mkdir(parents=True, exist_ok=True)`; remove config.py:58 | ✓ |
| Add a lazy ensure helper | New `ensure_output_dir()` in utils.py called per save | |
| Create in core.py | Orchestration owns directory creation | |

**User's choice:** Rely on utils.py write-time mkdir
**Notes:** No new code; satisfies CFG-02.

---

## web_server mkdir side effect

| Option | Description | Selected |
|--------|-------------|----------|
| Remove it | Reads guard with `.exists()`; `send_from_directory` 404s when absent; writes create it | ✓ |
| Lazy ensure on first request | Keep serving robust without an import side effect | |
| Keep the import-time mkdir | Contradicts CFG-02 | |

**User's choice:** Remove it
**Notes:** Satisfies CFG-02 (import creates no directories).

---

## the agent's Discretion

- `api_key` default expressed as a Field default or a model_validator.
- Wording/placement of the `__`-unsupported documentation and its pinning test.
- Test file layout and naming.
- Whether the CFG-02 import check runs in a fresh subprocess (recommended).

## Deferred Ideas

None — discussion stayed within phase scope.
