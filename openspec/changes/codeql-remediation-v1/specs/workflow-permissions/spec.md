## ADDED Requirements

### Requirement: CI workflow jobs declare explicit permissions
Every job in `.github/workflows/ci.yml` SHALL have an explicit `permissions:` block specifying only the permissions that job requires. Jobs that need no GitHub token access SHALL declare `permissions: {}`.

#### Scenario: Build job has minimal permissions
- **WHEN** the CI workflow runs a build job (e.g., Build x86-64 Kernel)
- **THEN** the job's `permissions:` block SHALL contain only `contents: read`

#### Scenario: Coverage job can write checks
- **WHEN** the CI workflow runs the Code Coverage job
- **THEN** the job's `permissions:` block SHALL contain `contents: read` and `checks: write`

#### Scenario: SonarCloud job can read PRs
- **WHEN** the CI workflow runs the SonarCloud Analysis job
- **THEN** the job's `permissions:` block SHALL contain `contents: read` and `pull-requests: read`

#### Scenario: Format and lint jobs are read-only
- **WHEN** the CI workflow runs the Format Check or Clippy Lint jobs
- **THEN** the job's `permissions:` block SHALL contain only `contents: read`

### Requirement: Release workflow jobs declare explicit permissions
Every job in `.github/workflows/release.yml` SHALL have an explicit `permissions:` block. The release creation job SHALL have `contents: write` for creating GitHub Releases and uploading assets.

#### Scenario: Release job can create releases
- **WHEN** the release workflow runs the build-and-release job
- **THEN** the job's `permissions:` block SHALL contain `contents: write`

#### Scenario: Docker publish job has package write
- **WHEN** the release workflow runs the Docker publish job
- **THEN** the job's `permissions:` block SHALL contain `packages: write` and `contents: read`
