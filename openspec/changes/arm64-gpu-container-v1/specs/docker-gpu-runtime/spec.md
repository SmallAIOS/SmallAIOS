## MODIFIED Requirements

### Requirement: Multi-stage Dockerfile for container mode
The repository SHALL provide Dockerfiles that build both CPU-only and GPU-enabled SmallAIOS container images, with ARM64 GPU support via NVIDIA CUDA base images.

#### Scenario: Build CPU-only container image
- **WHEN** `docker build -t smallaios:local .` is run
- **THEN** the builder stage SHALL compile the workspace with `--target x86_64-unknown-linux-musl --release`
- **AND** the runtime stage SHALL contain only the `smallaios-container` binary
- **AND** the final image size SHALL be less than 15 MB

#### Scenario: Build GPU-enabled container image
- **WHEN** `docker build --build-arg ENABLE_GPU=1 -t smallaios:local-gpu .` is run
- **THEN** the builder stage SHALL compile with `--target x86_64-unknown-linux-musl --release --features nvidia_gpu,cuda`
- **AND** the runtime stage SHALL contain the GPU-enabled `smallaios-container` binary

#### Scenario: Build ARM64 GPU-enabled container image
- **WHEN** `docker build -f Dockerfile.cuda --platform linux/arm64 -t smallaios:arm64-gpu .` is run
- **THEN** the builder stage SHALL compile with `--target aarch64-unknown-linux-musl --release --features nvidia_gpu,cuda`
- **AND** the runtime stage SHALL use the ARM64 variant of `nvcr.io/nvidia/cuda:12.x-runtime-ubuntu24.04`
- **AND** the image SHALL include `libcudart`, `libcublas`, and `libcudnn` for ARM64

#### Scenario: Entrypoint
- **WHEN** the container starts
- **THEN** the entrypoint SHALL be `/smallaios` (the container binary)
- **AND** the binary SHALL accept command-line arguments for model path and configuration

### Requirement: docker-compose with NVIDIA runtime
The repository SHALL provide a `docker-compose.yml` with service profiles for CPU-only and GPU-accelerated local testing, including ARM64 GPU variants.

#### Scenario: CPU-only service
- **WHEN** `docker compose up` is run (default profile)
- **THEN** the `smallaios` service SHALL build and start without GPU support
- **AND** port 8080 SHALL be forwarded from the host to the container
- **AND** the `./models/` directory SHALL be mounted at `/models` inside the container

#### Scenario: GPU service
- **WHEN** `docker compose --profile gpu up` is run
- **THEN** the `smallaios-gpu` service SHALL build with GPU features enabled
- **AND** the service SHALL use the `nvidia` runtime
- **AND** `NVIDIA_VISIBLE_DEVICES` SHALL be set to `all`
- **AND** port 8080 SHALL be forwarded and `./models/` SHALL be mounted

#### Scenario: Health check
- **WHEN** the container is running
- **THEN** docker-compose SHALL define a health check that polls the SmallAIOS health endpoint
- **AND** the container SHALL be marked unhealthy if the health check fails for 30 seconds

### Requirement: CI smoke test for Dockerfile
The CI pipeline SHALL verify that both CPU-only and GPU Dockerfiles build successfully.

#### Scenario: CPU Dockerfile build in CI
- **WHEN** a push or PR triggers the CI pipeline
- **THEN** a `docker-build-local` job SHALL build the CPU-only Dockerfile
- **AND** the job SHALL verify the image size is under 15 MB

#### Scenario: GPU Dockerfile build in CI
- **WHEN** a push or PR triggers the CI pipeline
- **THEN** a `docker-build-gpu` job SHALL build the GPU Dockerfile (without requiring a GPU runner)
- **AND** the job SHALL verify the image is a valid OCI image for the target architecture
