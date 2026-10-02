# Onboarding Summary

**Created:** 2026-10-02
**Project:** Imagai — CLI + web tool for generating images via OpenAI-compatible APIs

## Project State

- PROJECT.md: present
- REQUIREMENTS.md: present
- ROADMAP.md: present
- STATE.md: present

## Codebase Context

- Brownfield repo: yes
- Map readiness: complete
- Codebase map: `.planning/codebase/` (7 documents, 2,741 lines)
- Fast map available: yes

**Map contents:** `STACK.md`, `INTEGRATIONS.md`, `ARCHITECTURE.md`, `STRUCTURE.md`,
`CONVENTIONS.md`, `TESTING.md`, `CONCERNS.md` — stamped with a `last_mapped_commit`
drift baseline of `69cf575`.

## Docs Context

- Existing ADR/PRD/SPEC/RFC candidates: 0

The four `docs/*.md` files from a prior session (`architecture`, `testing`, `dependencies`,
`code-quality`) do not match GSD's ADR/PRD/SPEC/RFC patterns, so docs ingest was skipped.
Their content is superseded by the codebase map above, which verified their claims against
source and corrected two of them (uploads *are* capped by `MAX_CONTENT_LENGTH`;
`config.py`'s env loop is duplicative, not supplementary).

## Key Findings

Three issues from the map shaped the roadmap:

1. **The project does not currently run.** `rye` is not installed, the local `.venv/` is a
   Windows build, and ambient Python is 3.13.11 against a 3.12.9 pin. This is Phase 1.
2. **Arbitrary file write.** `output_filename` is unsanitized (`models.py:8`) and
   `core.py:76` joins it into the output path, so an absolute path replaces the base.
3. **`NameError` in `web_server.py:364`** — `main()` is called before its definition at `:368`.

## Scope Decisions

- Core value: **fast prompt-to-image** — drives all prioritization
- This cycle **hardens** rather than adds features
- The web server is a **localhost dev tool**, so authentication is explicitly out of scope
- Toolchain **migrates rye → uv**; uv is already installed, rye is not

## Recommended Next Step

- `/gsd-manager` — review the 6-phase roadmap and begin Phase 1 (Toolchain Migration)

## Open Items

- 3 commits are unpushed to `origin/main`
- `.planning/config.json` originally carried `mapper_model: anthropic/claude-haiku-4-5`,
  an unfunded tier that failed all four mapper agents. Superseded by `model_profile: inherit`,
  but worth setting explicitly if named tiers are ever funded
- `graft/` was installed mid-onboarding and its graph is gitignored via `/graft/`
