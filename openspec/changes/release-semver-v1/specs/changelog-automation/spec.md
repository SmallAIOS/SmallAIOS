## ADDED Requirements

### Requirement: git-cliff Configuration
The project SHALL include a `cliff.toml` configuration file at the repository root that defines how conventional commit messages are mapped to changelog entries in Keep a Changelog format.

#### Scenario: Configuration maps commit types to changelog groups
- **WHEN** git-cliff processes commit history
- **THEN** `feat` commits MUST appear under `### Added`
- **AND** `fix` commits MUST appear under `### Fixed`
- **AND** `perf` commits MUST appear under `### Performance`
- **AND** `refactor` commits MUST appear under `### Changed`
- **AND** `revert` commits MUST appear under `### Reverted`

#### Scenario: Non-functional commits are excluded
- **WHEN** git-cliff processes commit history
- **THEN** commits with types `chore`, `ci`, `test`, `style`, `build`, `docs` MUST be excluded from the generated changelog

#### Scenario: Output format matches existing CHANGELOG.md
- **WHEN** git-cliff generates changelog output
- **THEN** the output MUST conform to Keep a Changelog 1.1.0 format
- **AND** the output MUST include version headers as `## [<version>] - <date>`
- **AND** the output MUST preserve existing entries for previous releases

#### Scenario: Scoped commits include scope in entry
- **WHEN** a commit message includes a scope (e.g., `feat(net): add QUIC 0-RTT`)
- **THEN** the changelog entry MUST include the scope as a prefix (e.g., `**net:** add QUIC 0-RTT`)

### Requirement: Makefile Changelog Target
The project SHALL provide a `make changelog` target that invokes git-cliff to regenerate CHANGELOG.md.

#### Scenario: Target generates changelog from git history
- **WHEN** `make changelog` is executed
- **THEN** git-cliff MUST be invoked with the repository's `cliff.toml` configuration
- **AND** CHANGELOG.md MUST be updated in place with entries since the last `v*` tag in the `[Unreleased]` section

#### Scenario: Target works with no new commits
- **WHEN** `make changelog` is executed with no commits since the last tag
- **THEN** the `[Unreleased]` section MUST be empty or contain no entries
- **AND** existing versioned sections MUST be preserved unchanged

#### Scenario: Target prepends to existing changelog
- **WHEN** `make changelog` is executed
- **THEN** all previously released version sections in CHANGELOG.md MUST be preserved
- **AND** only the `[Unreleased]` section MUST be modified

### Requirement: Changelog Generation in Release Flow
The changelog generation MUST be integrated into the cargo-release pre-release hook so that CHANGELOG.md is updated and staged before the version bump commit.

#### Scenario: Unreleased section becomes versioned at release time
- **WHEN** `make release BUMP=<level>` is executed
- **THEN** the `[Unreleased]` section entries MUST be moved under a new `## [<new-version>] - <date>` header
- **AND** a fresh empty `[Unreleased]` section MUST be created above it
- **AND** the version comparison link at the bottom of the file MUST be updated

#### Scenario: Generated changelog is included in version bump commit
- **WHEN** the pre-release hook completes successfully
- **THEN** the updated CHANGELOG.md MUST be staged (`git add`)
- **AND** it MUST be included in the same commit as the version bump

#### Scenario: Release with only non-functional changes
- **WHEN** all commits since the last tag are types that are excluded from the changelog (chore, ci, test, etc.)
- **THEN** the new version section MUST still be created with the version header
- **AND** the section body MAY be empty or contain a note indicating internal changes only

### Requirement: CI Changelog Validation
The CI pipeline SHOULD validate that CHANGELOG.md is well-formed and consistent with the git history.

#### Scenario: CI detects missing changelog entries for feat/fix PRs
- **WHEN** a PR is opened against develop
- **AND** the PR contains commits of type `feat` or `fix`
- **THEN** CI SHOULD warn if the `[Unreleased]` section does not contain corresponding entries
- **AND** this check MUST NOT block the PR (warning only, since changelog is generated at release time)

#### Scenario: CI validates changelog format
- **WHEN** CI runs on a push to main or develop
- **THEN** CI SHOULD verify that CHANGELOG.md follows Keep a Changelog format
- **AND** CI SHOULD verify that version headers match existing git tags
