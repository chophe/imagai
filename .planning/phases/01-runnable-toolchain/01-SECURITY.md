---
phase: "01"
slug: runnable-toolchain
status: verified
threats_open: 0
asvs_level: 1
created: "2026-10-09"
---

# Phase 01 — Security

> Per-phase security contract: threat register, accepted risks, and audit trail.
> Produced by /gsd-secure-phase on 2026-10-09 (State B — no prior SECURITY.md; register reconstructed from PLAN threat models + implementation evidence).

---

## Trust Boundaries

| Boundary | Description | Data Crossing |
|----------|-------------|----------------|
| uv → PyPI | `uv sync` / `uv lock` resolve packages from PyPI | Public open-source packages; identical supply-chain profile to the prior rye workflow, dependency set unchanged minus unused `requests`, versions pinned in `uv.lock` |

Plans 01-02 and 01-03 cross no production trust boundaries (docs/string edits and verification-only command runs).

---

## Threat Register

| Threat ID | Category | Component | Severity | Disposition | Mitigation | Status |
|-----------|----------|-----------|----------|-------------|------------|--------|
| T-01-01 | Tampering | uv sync / PyPI installs | low | accept | `uv.lock` (287 KB, present at repo root) pins exact versions; dependency set is the rye set minus unused `requests`; verified `rg requests pyproject.toml` → no match | closed |
| T-01-02 | Denial of Service | uv lock resolution | low | accept | Resolution is a local bounded operation; failure mode is a clear error, not a hang — confirmed by successful `uv sync` / `uv lock --check` runs | closed |
| T-02-01 | Information Disclosure | cli.py error messages | low | accept | Messages now reference `uv add` / `uv sync` instead of rye; no sensitive data in either form | closed |
| T-02-02 | Tampering | docs/ content | low | accept | Documentation edits are cosmetic; no executable code paths affected | closed |
| T-03-01 | Tampering | uv sync (verification run) | low | accept | Same supply-chain profile as Plan 01 — pinned lockfile, no new packages | closed |
| T-03-02 | Information Disclosure | verification output | low | accept | Verification prints version strings and pass/fail status only | closed |

Accepted-risk claims were re-verified against the live tree on 2026-10-09 (lockfile present and in sync, `requests` absent from dependencies, error strings reference uv).

---

## Accepted Risks Log

| Risk ID | Threat Ref | Rationale | Accepted By | Date |
|---------|------------|-----------|-------------|------|
| AR-01 | T-01-01 | PyPI supply-chain trust retained; unchanged from pre-migration baseline and offset by version pinning | Phase 1 planning (D-01..D-12) | 2026-10-03 |
| AR-02 | T-01-02 | uv resolution DoS surface equivalent to rye; non-blocking, clear failure mode | Phase 1 planning | 2026-10-03 |
| AR-03 | T-02-01 | Error-message wording shift carries no disclosure risk either way | Phase 1 planning | 2026-10-03 |
| AR-04 | T-02-02 | Docs tampering out of trust boundary (repo contributors already trusted) | Phase 1 planning | 2026-10-03 |
| AR-05 | T-03-01 | Verification run repeats Plan 01 boundary with no new exposure | Phase 1 planning | 2026-10-03 |
| AR-06 | T-03-02 | Verification output is version/status text only | Phase 1 planning | 2026-10-03 |

---

## Security Audit Trail

| Audit Date | Threats Total | Closed | Open | Run By |
|------------|---------------|--------|------|--------|
| 2026-10-09 | 6 | 6 | 0 | gsd-secure-phase hook (verify:post) |

---

## Sign-Off

- [x] All threats have a disposition (mitigate / accept / transfer)
- [x] Accepted risks documented in Accepted Risks Log
- [x] `threats_open: 0` confirmed
- [x] `status: verified` set in frontmatter

**Approval:** verified 2026-10-09

## Security Audit 2026-10-09

| Metric | Count |
|---|---|
| Threats found | 6 |
| Closed | 6 |
| Open | 0 |
