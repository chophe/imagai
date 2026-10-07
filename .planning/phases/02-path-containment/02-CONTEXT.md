# Phase 2: Path Containment - Context

**Gathered:** 2026-10-04
**Status:** Ready for planning

<domain>
## Phase Boundary

The image save pipeline can only write inside the configured output directory. Every path handed to
`save_image_from_url` / `save_image_from_b64` must resolve under `settings.output_dir`; a filename
that would escape it is rejected with a populated error rather than written.

Requirements: SEC-01, SEC-02.

Explicitly out of scope: HTTP request/response schema validation (Phase 3, SEC-04), error
propagation UX beyond the existing `ImageGenerationResponse.error` contract (Phase 4), the
`config.py` env-loop cleanup (Phase 5), provider registry work (Phase 6).

</domain>

<decisions>
## Implementation Decisions

### Enforcement point
- **D-01:** The containment check lives inside `utils.py`'s save functions
  (`save_image_from_url`, `save_image_from_b64`), not in `core.py`. Both the CLI and the web server
  call these functions, so one check protects both front ends. `core.py` keeps computing the
  candidate path as it does today; the pipeline is the enforcement boundary.
  — **Reversibility:** costly — moving the check later means re-auditing every caller of the save
  functions for the missing guarantee.

### Filename policy
- **D-02:** A filename must be a **bare basename**. Any path separator or subdirectory component
  (e.g. `sub/img.png`) is rejected — even one that would resolve under the output directory.
  Absolute paths and `..` are therefore rejected by the same rule. `my_image.png` still writes to
  `settings.output_dir/my_image.png`, and the `n > 1` numbered variants keep landing in the output
  directory.
  — **Reversibility:** costly — widening to contained subdirectories later changes the rejection
  contract every caller and test already relies on.
- **D-03:** Containment resolves symlinks. The final path is canonicalized (realpath) and asserted
  to be under the canonicalized `settings.output_dir`. A symlink inside the output directory that
  points elsewhere is rejected, not followed.
  — **Reversibility:** reversible.

### Rejection contract
- **D-04:** On a rejected path, the save function returns `None` and does **not** raise. `core.py`
  sets `ImageGenerationResponse.error` to a clear message naming the rejection (the path escapes
  the output directory), matching the existing error contract the CLI and web server already
  consume. No exception type crosses the boundary.
  — **Reversibility:** one-way — Phase 4 (Error Propagation) formalizes how this error reaches the
  user; changing the boundary to a raised exception later would break the published response
  contract both front ends depend on.

### Output-directory source of truth
- **D-05:** `web_server.py`'s hardcoded `UPLOAD_FOLDER = Path("generated_images")` is unified to
  `settings.output_dir` in this phase. The upload/serve path, the save pipeline, and the containment
  root become one value. SEC-02 names `settings.output_dir` as the containment root, so a second
  hardcoded directory would make the web UI read/serve a different directory than the pipeline
  writes to.
  — **Reversibility:** reversible — Phase 5 (Configuration Cleanup) still owns removing the
  import-time `mkdir()` side effect; only the *value* is unified here.

### the agent's Discretion
- Exact wording of the rejection error message, provided it names the rejection clearly enough for
  Phase 4 to surface.
- Name and location of the containment helper inside `utils.py` (e.g. a private `_contained_path`
  helper), and whether the two save functions share it.
- Test file layout and naming for the containment tests (pytest is the established runner).
- Whether rejection is additionally logged; logging configuration is a deferred item (QUAL-02).

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Phase definition
- `.planning/ROADMAP.md` § Phase 2: Path Containment — goal, SEC-01/SEC-02 success criteria, and the
  `web_server.py:33` divergence flag.
- `.planning/REQUIREMENTS.md` — SEC-01 and SEC-02 (lines 20–21); SEC-04 belongs to Phase 3.
- `.planning/PROJECT.md` — security scope (localhost correctness bug, not a remote vuln) and the
  "harden before adding features" decision.

### Code under change
- `src/imagai/utils.py` — `save_image_from_url` / `save_image_from_b64`; the enforcement boundary
  (D-01).
- `src/imagai/core.py:45-98` — filename-strategy logic and `output_file_path = Path(settings.output_dir) / current_filename` at `:76`; sets `api_response.error` on failure.
- `src/imagai/models.py:8` — `ImageGenerationRequest.output_filename` (the never-sanitized input).
- `src/imagai/web_server.py:33-35` — hardcoded `UPLOAD_FOLDER`; `:312` `send_from_directory`; `:267-289` generated-image listing; unify to `settings.output_dir` (D-05).
- `src/imagai/config.py` — `settings.output_dir` (the containment root).

### Context and evidence
- `.planning/codebase/CONCERNS.md` — the path-traversal write, reachable via `POST /api/generate`.
- `.planning/codebase/ARCHITECTURE.md` — layering: presentation → core → providers → I/O.

### Verification surface
- `tests/test_cli.py` — the current (placeholder) test file; new containment tests are added, not
  assumed.

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- `utils.py` `save_image_from_url` / `save_image_from_b64` — the shared write path; the single place
  a containment check protects both front ends.
- `core.py` filename-strategy block (`:45-75`) — manual / LLM-generated / random / prompt-derived
  strategies plus the `n > 1` numbered variants; the containment check must hold for every variant.
- `ImageGenerationResponse.error` — the existing error channel; a rejected path populates it rather
  than raising.

### Established Patterns
- Pydantic DTOs cross every boundary (`ImageGenerationRequest` → core → `ImageGenerationResponse`).
- The CLI (`cli.py`) and web server (`web_server.py`) both call the same `core.py` orchestration
  function — the reason containment in `utils.py` protects both.
- `pytest` is the test runner (Phase 1 made `uv run pytest` work).

### Integration Points
- `core.py:76` computes `output_file_path` and passes it to the save functions — the seam the check
  sits behind.
- `web_server.py` reads/serves `UPLOAD_FOLDER` (`:267-289`, `:312`) — must point at
  `settings.output_dir` after D-05 so read and write agree.

</code_context>

<specifics>
## Specific Ideas

Verification is expected to be test-driven and mechanical, matching the phase criteria:

- `/tmp/evil.png` → rejected, and no file exists at `/tmp/evil.png` after the call.
- `../../escape.png` → rejected, nothing written outside the output directory.
- All four filename strategies plus the `n > 1` variants → every path resolves under
  `settings.output_dir`.
- `my_image.png` → still written to `settings.output_dir/my_image.png` (happy path preserved).

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope.

</deferred>

---

*Phase: 2-Path Containment*
*Context gathered: 2026-10-04*
