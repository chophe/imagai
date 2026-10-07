# Phase 5: Configuration Cleanup - Context

**Gathered:** 2026-10-04
**Status:** Ready for planning

<domain>
## Phase Boundary

`imagai.config` becomes a pure declaration of settings: pydantic-settings is the only source of
engine configuration, and importing `imagai.config` creates no directories and mutates no global
state. The manual `os.environ` parsing loop (`config.py:38-54`) and the import-time `mkdir`
(`config.py:56-58`) are removed, and `web_server.py`'s own import-time `mkdir` is removed too.

Requirements: CFG-01, CFG-02.

Explicitly out of scope: the path-containment fix (Phase 2), HTTP schema validation (Phase 3),
error propagation (Phase 4), provider registry (Phase 6), and adding a linter/formatter
(TOOL-01, deferred to v2).

</domain>

<decisions>
## Implementation Decisions

### Engine configuration source
- **D-01:** Delete the manual `os.environ` loop (`config.py:38-54`) and rely on **native
  pydantic-settings**. Verified by experiment: pydantic-settings already reproduces the loop for
  every name in `.env.example` — `OPENAI_DALLE3` → `openai_dalle3` (keys lowercased),
  `STABILITY_v11U` → `stability_v11u`, `base_url` coerced via the `Optional[HttpUrl]` type,
  `model`/`api_key` set. The one behavioral gap is leniency: the loop seeded `api_key="dummy"`
  for an engine declared with only `model`/`base_url`, whereas native pydantic-settings raises
  "Field required". Preserve that leniency by giving `EngineConfig.api_key` a default of `"dummy"`
  (or an equivalent validator) so partial engine configs still load.
  — **Reversibility:** costly — removing the loop is the phase's core change; restoring the manual
  parsing would re-introduce the side-effectful config module every later phase assumes is gone.

### Unsupported configuration forms
- **D-02:** An engine name containing the `__` nested delimiter (e.g. `MY__ENGINE`) is
  **explicitly unsupported**. It fails in both the old loop (mis-splits the key) and native
  pydantic-settings (nests it wrongly). Document it as unsupported in `config.py` and pin the
  decision with a test, per success criterion 3. No `.env.example` name uses `__` inside an engine
  name, so nothing supported regresses.
  — **Reversibility:** reversible.

### Merge semantics
- **D-03:** Preserve current behavior exactly: when env vars provide engines, they **replace** the
  `engines` dict (the built-in `openai_dalle3` default does not survive alongside them); when no
  env engines are set, the built-in default applies. This is unchanged from today (verified by
  experiment) — no new merge logic, so the phase stays a pure cleanup.
  — **Reversibility:** reversible.

### Directory creation (no import side effects)
- **D-04:** Remove the import-time `mkdir` at `config.py:56-58`. Rely on the write-time
  `output_path.parent.mkdir(parents=True, exist_ok=True)` already present in `utils.py` (lines
  162, 196). No new helper is added.
  — **Reversibility:** reversible.
- **D-05:** Remove `web_server.py:34`'s import-time `mkdir`. Reads already guard with
  `UPLOAD_FOLDER.exists()` before globbing, and `send_from_directory` returns 404 when the directory
  is absent (correct — no images yet). Writes create the directory through the save pipeline.
  — **Reversibility:** reversible.

### the agent's Discretion
- Whether `api_key`'s `"dummy"` default is expressed as a Pydantic `Field(default="dummy")` or a
  `model_validator`; either satisfies D-01.
- Exact wording/placement of the `__`-unsupported documentation and the test that pins it.
- Test file layout and naming (pytest is the established runner).
- Whether to assert the CFG-02 import behavior by spawning a fresh subprocess (recommended, since
  import side effects are only observable in a clean process).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase definition
- `.planning/ROADMAP.md` § Phase 5: Configuration Cleanup — goal, CFG-01/CFG-02 success criteria,
  and the known-risk note about the nested delimiter.
- `.planning/REQUIREMENTS.md` — CFG-01 (line 32) and CFG-02 (line 33).
- `.planning/PROJECT.md` — "harden before adding features"; the toolchain now runs (Phase 1).

### Code under change
- `src/imagai/config.py` — the whole file. Remove `:38-54` (os.environ loop) and `:56-58`
  (import-time mkdir); `EngineConfig` (`:7-14`) gains the `api_key="dummy"` default; `Settings`
  (`:17-33`) keeps `env_prefix="IMAGAI__"`, `env_nested_delimiter="__"`.
- `src/imagai/web_server.py:34` — second import-time `mkdir` to remove (D-05); `:267-289`, `:312`
  read/serve the directory.
- `src/imagai/utils.py:162,196` — the write-time `mkdir` that becomes the sole creation point (D-04).
- `.env.example` — the authoritative list of `IMAGAI__ENGINES__*` names the parser must reproduce
  (criterion 2).

### Engine consumers (must keep working off `settings.engines`)
- `src/imagai/core.py:23-27` — engine lookup + "Available engines" error message.
- `src/imagai/cli.py:166,181,185` — `list-engines` and engine validation.
- `src/imagai/utils.py:35-44` — filename-generation engine selection.
- `src/imagai/web_server.py:57,101,373` — engine listing and validation.

### Context and evidence
- `.planning/codebase/CONCERNS.md` — the `config.py` env-loop / import-side-effect notes.
- `.planning/codebase/ARCHITECTURE.md` — module layering.

### Verification surface
- `tests/test_cli.py` — current placeholder test file; new config tests are added, not assumed.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `utils.py`'s write-time `mkdir(parents=True, exist_ok=True)` (lines 162, 196) — already the
  correct creation point; D-04 relies on it rather than adding code.
- `EngineConfig` — the Pydantic model the `api_key="dummy"` default belongs on (D-01).
- `settings.engines` — already the single source read by `core.py`, `cli.py`, `utils.py`, and
  `web_server.py`; this phase must not disturb that.

### Established Patterns
- pydantic-settings v2 with `env_prefix` + `env_nested_delimiter`; `extra="ignore"`.
- Pydantic DTOs cross every boundary; `Settings` is a module-level singleton.
- `pytest` is the runner (Phase 1 made `uv run pytest` work).

### Integration Points
- Four modules import the `settings` singleton — removing the loop and the import-time mkdir is a
  cross-module change, not file-local.
- `web_server.py`'s import-time mkdir is the second side effect this phase removes.

</code_context>

<specifics>
## Specific Ideas

Grounded in a live parser experiment (not assumed):

- Pure pydantic-settings parsed `IMAGAI__ENGINES__OPENAI_DALLE3__*` → `engines["openai_dalle3"]`
  correctly, and `IMAGAI__ENGINES__STABILITY_v11U__*` → `engines["stability_v11u"]`.
- Env-provided engines replaced the whole dict (built-in default did not survive) — the same as
  the current loop, so D-03 preserves behavior rather than changing it.
- `IMAGAI__ENGINES__MY__ENGINE__API_KEY` failed in both parsers — the basis for D-02.

Verification is expected to be test-driven and mechanical, matching the phase criteria:

- Fresh-process import with `output_dir` pointing at a nonexistent path leaves it absent (CFG-02).
- A parametrized test over `.env.example` names (`api_key`, `base_url`, `model`) yields the same
  `settings.engines` as today (CFG-01).
- A test pins the `__`-in-engine-name decision (CFG-01, criterion 3).
- `core.py`, `utils.py`, `cli.py`, `web_server.py` still create the output dir at write time, not
  import time (CFG-02, criterion 4).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope. (Logging configuration, QUAL-02, and the linter,
TOOL-01, remain deferred to v2 and were not raised.)

</deferred>

---

*Phase: 5-Configuration Cleanup*
*Context gathered: 2026-10-04*
