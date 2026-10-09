---
phase: 01
review: 01-REVIEW.md
titles: json
findings:
  - id: CR-01
    severity: critical
    disposition: open
    title: "Command injection via `subprocess.run(shell=True)` in `/api/generate-cli`"
  - id: CR-02
    severity: critical
    disposition: open
    title: "`str | None` syntax requires Python 3.10+ but `requires-python = \">=3.9\"`"
  - id: CR-04
    severity: critical
    disposition: open
    title: "`main()` is called four lines before it is defined"
  - id: WR-01
    severity: warning
    disposition: open
    title: "`import requests as _requests` is dead code — `requests` is not a dependency"
  - id: WR-02
    severity: warning
    disposition: open
    title: "Misleading error message tells users to install `requests`"
  - id: WR-03
    severity: warning
    disposition: open
    title: "Relative path for `web_interface.html` breaks when server started outside project root"
  - id: WR-04
    severity: warning
    disposition: open
    title: "Duplicate `ImageGenerationRequest` creation — first instance is dead code"
  - id: WR-05
    severity: warning
    disposition: open
    title: "`debug=True` and `host=\"0.0.0.0\"` defaults in `main()`"
  - id: WR-06
    severity: warning
    disposition: open
    title: "README states `list-engines` command \"needs to be implemented\" — it is implemented"
  - id: WR-07
    severity: warning
    disposition: open
    title: "`docs/dependencies.md` references deleted lockfiles"
  - id: WR-08
    severity: warning
    disposition: open
    title: "`docs/dependencies.md` describes the pre-migration `requires-python`"
  - id: WR-09
    severity: warning
    disposition: open
    title: "`flask>=2.0.0` allows known-vulnerable 2.0.x releases"
  - id: WR-10
    severity: warning
    disposition: open
    title: "`buildCliCommand` function in `web_interface.html` is dead code"
  - id: WR-11
    severity: warning
    disposition: open
    title: "`sanitize_filename` strips every digit and every uppercase ASCII letter"
  - id: IN-01
    severity: info
    disposition: open
    title: "`_is_image_model` uses very short indicator `\"sd\"` causing false positives"
  - id: IN-02
    severity: info
    disposition: open
    title: "`web_server.py` uses `print` instead of proper logging"
  - id: IN-03
    severity: info
    disposition: open
    title: "`typer[all]` extra is heavier than needed"
open: 17
total: 17
recorded: 2026-10-09T14:49:23.584Z
---

# Phase 01: Code Review Disposition

| Finding | Severity | Disposition | Source |
|---------|----------|-------------|--------|
| CR-01 | critical | open | - |
| CR-02 | critical | open | - |
| CR-04 | critical | open | - |
| WR-01 | warning | open | - |
| WR-02 | warning | open | - |
| WR-03 | warning | open | - |
| WR-04 | warning | open | - |
| WR-05 | warning | open | - |
| WR-06 | warning | open | - |
| WR-07 | warning | open | - |
| WR-08 | warning | open | - |
| WR-09 | warning | open | - |
| WR-10 | warning | open | - |
| WR-11 | warning | open | - |
| IN-01 | info | open | - |
| IN-02 | info | open | - |
| IN-03 | info | open | - |

Dispositions: `open` (recorded, not yet triaged), `fixed`, `skipped`, `deferred`.
Set `deferred` by hand and put the reason in the Source cell; both are preserved. A `|` in the reason is kept as prose and escaped on the next run.
Re-running the gate keeps every row it can. A row the current review no longer reports is kept and its Source cell flagged, so a finding does not leave this record silently. ONE exception: when a finding id is REUSED by a different finding, the earlier decision cannot keep a row — the id is taken — and it is dropped. A RECORDED decision (anything but `open`) is named on the console when that happens; a row still at `open` is replaced silently, because `open` records no decision to lose.
