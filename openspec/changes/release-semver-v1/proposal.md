## Why

The project has semver PR title enforcement and cargo-release tooling but lacks a documented end-to-end release process. Release cadence, changelog automation, version bump decisions, and the develop→main merge workflow are not formalized, leading to inconsistent releases and manual overhead.

## What Changes

- Define a release checklist and cadence (milestone-based, not time-based for pre-1.0)
- Automate CHANGELOG.md generation from conventional commit PR titles
- Formalize the develop→main squash-merge + sync-back workflow
- Add release candidate validation (CI must be green, coverage thresholds met)
- Document version bump rules: when to bump minor vs patch pre-1.0
- Add GitHub Release notes automation from changelog entries

## Capabilities

### New Capabilities
- `release-process`: End-to-end release workflow documentation and automation — covers release checklist, develop→main merge strategy, post-merge sync, tagging, and GitHub Release creation
- `changelog-automation`: Automated changelog generation from conventional commit PR titles — covers tooling selection, CI integration, and CHANGELOG.md format maintenance

### Modified Capabilities
None — this change adds process documentation and CI automation without modifying existing specs.

## Impact

- `.github/workflows/release.yml` — enhance with changelog generation
- `CHANGELOG.md` — switch from manual to automated entries
- `release.toml` — potential updates for pre/post-release hooks
- `CLAUDE.md` — update release documentation section
- New: release checklist document or runbook
