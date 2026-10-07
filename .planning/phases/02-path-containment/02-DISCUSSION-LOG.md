# Phase 2: Path Containment - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-10-04
**Phase:** 2-path-containment
**Areas discussed:** Enforcement point, Filename policy, Rejection contract, UPLOAD_FOLDER divergence

---

## Enforcement point

| Option | Description | Selected |
|--------|-------------|----------|
| In utils.py save functions | Validate inside `save_image_from_url` / `save_image_from_b64`; both CLI and web are protected by one check | ✓ |
| In core.py call site | Validate in `core.py` before computing `output_file_path`; leaves the CLI unprotected unless it also validates | |
| Both (defense in depth) | Validate in both places; strongest, slightly redundant | |

**User's choice:** In utils.py save functions
**Notes:** Matches the ROADMAP note that containment belongs in the shared save pipeline (`utils.py`).

---

## Filename policy

| Option | Description | Selected |
|--------|-------------|----------|
| Bare basename only | Reject any separator / subdirectory component; filenames are a single basename | ✓ |
| Allow contained subdirectories | Allow `sub/img.png` when it resolves under output_dir | |
| Sanitize and write | Rewrite unsafe input to a safe basename and write anyway | |

**User's choice:** Bare basename only
**Notes:** Keeps the existing numbered-variant and happy-path behavior; avoids a larger test surface.

| Option | Description | Selected |
|--------|-------------|----------|
| Resolve symlinks (realpath) | Canonicalize the final path and assert it is under the canonicalized output_dir | ✓ |
| Lexical containment only | Reject absolute / `..` / separators only | |

**User's choice:** Resolve symlinks (realpath)
**Notes:** Catches a symlink inside output_dir that points elsewhere; strongest correctness for a modest amount of code.

---

## Rejection contract

| Option | Description | Selected |
|--------|-------------|----------|
| Return None + core sets error | `utils.py` returns None; `core.py` populates `ImageGenerationResponse.error` | ✓ |
| Raise typed exception, core catches | `utils.py` raises `PathContainmentError`; core catches it | |
| Raise and propagate | Exception crosses the boundary | |

**User's choice:** Return None + core sets error
**Notes:** Matches the ROADMAP's stated contract (populated error, not an exception) and keeps the existing CLI/web error handling intact.

---

## UPLOAD_FOLDER divergence

| Option | Description | Selected |
|--------|-------------|----------|
| Unify to settings.output_dir now | Point `web_server.py` at `settings.output_dir`; read/serve and write share one directory | ✓ |
| Defer to Phase 5 | Leave the hardcoded path; Phase 5 reconciles it | |
| Unify value, keep mkdir for Phase 5 | Unify the value now, leave the import-time `mkdir()` side effect for Phase 5 | |

**User's choice:** Unify to settings.output_dir now
**Notes:** SEC-02 names `settings.output_dir` as the containment root; a second hardcoded directory would make the web UI serve a different directory than the pipeline writes to.

---

## the agent's Discretion

- Exact wording of the rejection error message.
- Name/location of the containment helper inside `utils.py` and whether both save functions share it.
- Test file layout and naming (pytest is the established runner).
- Whether rejection is additionally logged (logging config is deferred, QUAL-02).

## Deferred Ideas

None — discussion stayed within phase scope.
