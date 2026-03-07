## ADDED Requirements

### Requirement: Multi-stage Dockerfile for container mode
The repository SHALL provide a `Dockerfile` at the repository root that builds the SmallAIOS container binary (`x86_64-unknown-linux-musl`) in a multi-stage build and produces a minimal runtime image.

#### Scenario: Build CPU-only container image
- **WHEN** `docker build -t smallaios:local .` is run
- **THEN** the builder stage SHALL compile the workspace with `--target x86_64-unknown-linux-musl --release`
- **AND** the runtime stage SHALL contain only the `smallaios-container` binary
- **AND** the final image size SHALL be less than 15 MB

#### Scenario: Build GPU-enabled container image
- **WHEN** `docker build --build-arg ENABLE_GPU=1 -t smallaios:local-gpu .` is run
- **THEN** the builder stage SHALL compile with `--target x86_64-unknown-linux-musl --release --features nvidia_gpu`
- **AND** the runtime stage SHALL contain the GPU-enabled `smallaios-container` binary

#### Scenario: Entrypoint
- **WHEN** the container starts
- **THEN** the entrypoint SHALL be `/smallaios` (the container binary)
- **AND** the binary SHALL accept command-line arguments for model path and configuration

### Requirement: docker-compose with NVIDIA runtime
The repository SHALL provide a `docker-compose.yml` at the repository root with service profiles for CPU-only and GPU-accelerated local testing.

#### Scenario: CPU-only service
- **WHEN** `docker compose up` is run (default profile)
- **THEN** the `smallaios` service SHALL build and start without GPU support
- **AND** port 8080 SHALL be forwarded from the host to the container
- **AND** the `./models/` directory SHALL be mounted at `/models` inside the container

#### Scenario: GPU service
- **WHEN** `docker compose --profile gpu up` is run
- **THEN** the `smallaios-gpu` service SHALL build with `ENABLE_GPU=1`
- **AND** the service SHALL use the `nvidia` runtime
- **AND** `NVIDIA_VISIBLE_DEVICES` SHALL be set to `all`
- **AND** port 8080 SHALL be forwarded and `./models/` SHALL be mounted

#### Scenario: Health check
- **WHEN** the container is running
- **THEN** docker-compose SHALL define a health check that polls the SmallAIOS health endpoint
- **AND** the container SHALL be marked unhealthy if the health check fails for 30 seconds

### Requirement: Makefile targets for Docker local testing
The Makefile SHALL provide convenience targets for local Docker workflows.

#### Scenario: CPU-only local run
- **WHEN** `make docker-local` is run
- **THEN** it SHALL execute `docker compose up --build`
- **AND** the container SHALL start in the foreground with serial output visible

#### Scenario: GPU local run
- **WHEN** `make docker-local-gpu` is run
- **THEN** it SHALL execute `docker compose --profile gpu up --build`
- **AND** the container SHALL start with NVIDIA GPU access

#### Scenario: Local image cleanup
- **WHEN** `make docker-local-clean` is run
- **THEN** it SHALL stop and remove the local containers and images

### Requirement: CI smoke test for Dockerfile
The CI pipeline SHALL verify that the Dockerfile builds successfully.

#### Scenario: Dockerfile build in CI
- **WHEN** a push or PR triggers the CI pipeline
- **THEN** a `docker-build-local` job SHALL build the Dockerfile (CPU-only, no GPU)
- **AND** the job SHALL verify the image size is under 15 MB
- **AND** the job SHALL NOT require a GPU runner
