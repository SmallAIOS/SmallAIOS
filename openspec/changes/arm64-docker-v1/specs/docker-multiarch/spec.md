## MODIFIED Requirements

### Requirement: Multi-architecture Dockerfile
The `Dockerfile` SHALL support building for both `linux/amd64` and `linux/arm64` platforms via `docker buildx`.

#### Scenario: Build x86_64 container image
- **WHEN** `docker buildx build --platform linux/amd64 -t smallaios .` is run
- **THEN** the builder stage SHALL compile with `--target x86_64-unknown-linux-musl --release`
- **AND** the runtime stage SHALL contain the `smallaios-container` binary
- **AND** the final image size SHALL be less than 15 MB

#### Scenario: Build aarch64 container image
- **WHEN** `docker buildx build --platform linux/arm64 -t smallaios .` is run
- **THEN** the builder stage SHALL compile with `--target aarch64-unknown-linux-musl --release`
- **AND** the runtime stage SHALL contain the `smallaios-container` binary
- **AND** the final image size SHALL be less than 15 MB

#### Scenario: Build multi-arch manifest
- **WHEN** `docker buildx build --platform linux/amd64,linux/arm64 -t smallaios .` is run
- **THEN** buildx SHALL produce images for both architectures
- **AND** a multi-arch manifest SHALL be created linking both platform images

#### Scenario: Architecture mapping
- **GIVEN** `TARGETARCH` is injected by Docker buildx
- **WHEN** `TARGETARCH` is `amd64`
- **THEN** the Rust target triple SHALL be `x86_64-unknown-linux-musl`
- **WHEN** `TARGETARCH` is `arm64`
- **THEN** the Rust target triple SHALL be `aarch64-unknown-linux-musl`
- **WHEN** `TARGETARCH` is any other value
- **THEN** the build SHALL fail with a clear error message

#### Scenario: GPU feature flag with multi-arch
- **WHEN** `docker buildx build --platform linux/arm64 --build-arg ENABLE_GPU=1 .` is run
- **THEN** the build SHALL compile with `--features nvidia_gpu` for the arm64 target
- **AND** the GPU feature flag SHALL work identically on both architectures

#### Scenario: Entrypoint
- **WHEN** the container starts on either architecture
- **THEN** the entrypoint SHALL be `/smallaios`
- **AND** the binary SHALL be the correct architecture for the host

### Requirement: Runtime image uses scratch base
The runtime stage SHALL use `FROM scratch` to produce a minimal image containing only the statically-linked binary.

#### Scenario: Minimal image contents
- **GIVEN** the build completes for any supported architecture
- **THEN** the runtime image SHALL contain exactly one file: `/smallaios`
- **AND** the image SHALL have no shell, no package manager, and no libc

## UNCHANGED Requirements

### Requirement: Docker build targets in Makefile
The existing `docker-build` and `docker-push` Makefile targets already use `docker buildx build --platform linux/amd64,linux/arm64` and SHALL continue to work without modification.

### Requirement: docker-compose local development
The existing `docker-compose.yml` and local development targets (`docker-local`, `docker-local-gpu`) SHALL continue to work for x86_64 hosts. ARM64 local development via docker-compose is not in scope for this change.
