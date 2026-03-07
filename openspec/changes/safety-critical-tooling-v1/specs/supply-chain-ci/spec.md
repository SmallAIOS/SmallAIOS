## ADDED Requirements

### Requirement: cargo-deny dependency auditing in CI
Every PR SHALL be checked by cargo-deny for known security advisories, license violations, banned crates, and unauthorized dependency sources.

#### Scenario: PR with vulnerable dependency is blocked
- **WHEN** a PR adds or updates a dependency with a known RustSec advisory
- **THEN** the cargo-deny CI job SHALL fail and report the advisory ID and affected crate

#### Scenario: GPL dependency is rejected
- **WHEN** a PR introduces a transitive dependency with a GPL/LGPL/AGPL license
- **THEN** the cargo-deny CI job SHALL fail and report the license violation

### Requirement: cargo-geiger unsafe surface area tracking
CI SHALL run cargo-geiger to produce an unsafe usage report. The report SHALL be available as a CI artifact.

#### Scenario: Unsafe usage report generated
- **WHEN** CI runs on a push to develop or main
- **THEN** cargo-geiger SHALL produce a report listing unsafe usage counts per crate

#### Scenario: New unsafe block detected
- **WHEN** a PR introduces new `unsafe` code
- **THEN** the cargo-geiger diff SHALL highlight the increase in the PR comment or CI log
