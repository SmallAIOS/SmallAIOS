## ADDED Requirements

### Requirement: Release Runbook
The project SHALL provide a release runbook at `docs/release-runbook.md` that documents the complete end-to-end release workflow from develop to published GitHub Release.

#### Scenario: Runbook covers pre-release checks
- **WHEN** a maintainer consults the runbook before a release
- **THEN** the runbook MUST list all pre-release checks: CI green on develop, no open blockers, changelog reviewed, bump level determined

#### Scenario: Runbook covers merge workflow
- **WHEN** a maintainer follows the runbook merge steps
- **THEN** the runbook MUST document: (1) create PR from develop to main, (2) squash-merge the PR, (3) immediately merge main back into develop to sync histories

#### Scenario: Runbook covers version bump and tagging
- **WHEN** a maintainer follows the runbook release steps
- **THEN** the runbook MUST document: (1) checkout main, (2) run `make release BUMP=<level>`, (3) review the commit and tag, (4) push with `--follow-tags` to trigger release.yml

#### Scenario: Runbook covers post-release verification
- **WHEN** a maintainer completes a release
- **THEN** the runbook MUST document verification steps: GitHub Release created, kernel artifacts uploaded, Docker image published to GHCR, develop synced with main

#### Scenario: Runbook covers rollback
- **WHEN** a release must be rolled back
- **THEN** the runbook MUST document: (1) delete the git tag locally and remotely, (2) revert the version bump commit on main, (3) sync main back to develop

### Requirement: Develop-to-Main Squash-Merge Workflow
The release process SHALL use GitHub squash-merge for the develop-to-main PR, followed by an immediate sync-back merge from main into develop.

#### Scenario: Squash-merge creates single commit on main
- **WHEN** a develop-to-main PR is squash-merged
- **THEN** main MUST contain exactly one new commit representing all changes since the last release
- **AND** the commit message MUST follow conventional commit format

#### Scenario: Sync-back merge prevents history divergence
- **WHEN** the squash-merge completes
- **THEN** main MUST be merged into develop with a regular merge commit
- **AND** after the sync-back, `git log develop..main` MUST return zero commits

#### Scenario: Post-release sync includes version bump
- **WHEN** cargo-release creates a version bump commit and tag on main
- **THEN** main MUST be merged into develop again
- **AND** develop MUST contain the version bump commit after the sync

### Requirement: Version Bump Decision Script
The project SHALL provide a script `scripts/suggest-release-bump.sh` that analyzes all merged PR titles since the last release tag and outputs the recommended semver bump level.

#### Scenario: Script reads git history since last tag
- **WHEN** `scripts/suggest-release-bump.sh` is executed
- **THEN** it MUST find the most recent `v*` tag in git history
- **AND** it MUST extract commit messages between that tag and HEAD

#### Scenario: Script applies pre-1.0 bump rules
- **WHEN** the workspace version major is 0
- **THEN** the script MUST apply pre-1.0 rules: breaking changes produce `minor`, feat produces `minor`, fix/perf/revert produce `patch`, all others produce `none`
- **AND** the output MUST be the highest bump level across all commits

#### Scenario: Script outputs actionable suggestion
- **WHEN** the script completes analysis
- **THEN** it MUST print the suggested bump level (`major`, `minor`, `patch`, or `none`) to stdout
- **AND** it MUST print the individual PR title classifications to stderr for review

#### Scenario: Script handles no changes
- **WHEN** there are no commits since the last tag
- **THEN** the script MUST output `none`
- **AND** it MUST print a message indicating no changes found

### Requirement: Pre-Release Hook Runs Tests and Changelog
The cargo-release pre-release hook SHALL run both the test suite and changelog generation before creating the version bump commit.

#### Scenario: Hook runs tests first
- **WHEN** `make release BUMP=<level>` is executed
- **THEN** `make test` MUST run before any version changes
- **AND** if tests fail, cargo-release MUST abort without modifying any files

#### Scenario: Hook generates changelog
- **WHEN** tests pass during the pre-release hook
- **THEN** the changelog generation tool MUST update CHANGELOG.md with entries since the last tag
- **AND** the updated CHANGELOG.md MUST be staged for inclusion in the version bump commit

#### Scenario: Hook failure is recoverable
- **WHEN** the pre-release hook fails at any step
- **THEN** no version bump commit or tag MUST be created
- **AND** the maintainer MUST be able to fix the issue and re-run `make release`

### Requirement: GitHub Release Notes from CHANGELOG
The release.yml workflow SHALL extract release notes from CHANGELOG.md for the version being released.

#### Scenario: Release notes extracted for tagged version
- **WHEN** a `v*` tag triggers release.yml
- **THEN** the workflow MUST extract the CHANGELOG.md section matching the tag version
- **AND** the extracted text MUST be used as the GitHub Release body

#### Scenario: Fallback for missing changelog section
- **WHEN** the tag version has no matching section in CHANGELOG.md
- **THEN** the workflow MUST use a fallback message containing the tag name
- **AND** the release MUST still be created successfully
