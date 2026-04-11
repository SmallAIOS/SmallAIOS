## ADDED Requirements

### Requirement: Multi-arch Docker image push on release
The release workflow SHALL build and push multi-architecture Docker images to GHCR when a version tag is pushed.

#### Scenario: Release tag triggers Docker publish
- **GIVEN** a tag matching `v*` is pushed (e.g., `v0.1.0`)
- **WHEN** the release workflow runs
- **THEN** a `docker-publish` job SHALL build images for `linux/amd64` and `linux/arm64`
- **AND** the multi-arch manifest SHALL be pushed to `ghcr.io/smallaios/smallaios`

#### Scenario: Image tagging
- **GIVEN** the release tag is `v0.1.0`
- **THEN** the images SHALL be tagged as `ghcr.io/smallaios/smallaios:v0.1.0`
- **AND** if the tag does NOT contain `alpha`, `beta`, or `rc`, the images SHALL also be tagged as `ghcr.io/smallaios/smallaios:latest`

#### Scenario: Pre-release tagging
- **GIVEN** the release tag is `v0.1.0-alpha.1`
- **THEN** the images SHALL be tagged as `ghcr.io/smallaios/smallaios:v0.1.0-alpha.1`
- **AND** the images SHALL NOT be tagged as `latest`

#### Scenario: Authentication
- **WHEN** pushing to GHCR
- **THEN** the workflow SHALL authenticate using `GITHUB_TOKEN` with `packages: write` permission
- **AND** no additional secrets SHALL be required

#### Scenario: Image size validation
- **WHEN** the multi-arch build completes
- **THEN** each per-architecture image SHALL be verified to be under 15 MB
- **AND** the job SHALL fail if any image exceeds 15 MB

### Requirement: QEMU emulation for cross-architecture builds
The CI workflow SHALL use QEMU user-mode emulation to build arm64 images on amd64 runners.

#### Scenario: Buildx with QEMU
- **GIVEN** the CI runner is `ubuntu-latest` (amd64)
- **WHEN** building for `linux/arm64`
- **THEN** the workflow SHALL set up QEMU via `docker/setup-qemu-action`
- **AND** the workflow SHALL set up Docker Buildx via `docker/setup-buildx-action`

## UNCHANGED Requirements

### Requirement: CI Docker build on PRs
The existing `docker-build-local` CI job SHALL continue to build only the native (amd64) image on PRs and validate image size. Multi-arch builds are not required for PR validation.

### Requirement: GitHub Release artifacts
The existing `release` job SHALL continue to produce bare-metal kernel binaries as GitHub Release assets. The Docker publish job is additive and does not modify the release asset workflow.
