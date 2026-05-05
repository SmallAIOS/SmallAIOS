# docker-gpu-runtime Specification

## Purpose
TBD - created by archiving change jetson-orin-container-v1. Update Purpose after archive.
## Requirements
### Requirement: Jetson Orin container variants

The repository SHALL provide two Jetson-specific Dockerfiles, both producing images bootable on NVIDIA Jetson Orin (Nano / NX / AGX) with GPU acceleration: a recommended slim variant (`Dockerfile.jetson.slim`) and a full-JetPack convenience variant (`Dockerfile.jetson`).

#### Scenario: Image base is L4T-derived

- **GIVEN** a developer building `Dockerfile.jetson`
- **THEN** the base image SHALL be `nvcr.io/nvidia/l4t-jetpack:r36.4.0` or a documented L4T-derived alternative (e.g. `nvcr.io/nvidia/l4t-cuda:12.6.x-runtime` plus a manual cuDNN install)
- **AND** the image SHALL NOT use `nvcr.io/nvidia/cuda:13.0.0-*` because the Jetson driver (L4T R36.4.x) reports CUDA 12.6 capability and the CDI runtime requirements check rejects 13.0 images

#### Scenario: Build uses glibc, not musl

- **GIVEN** the build invocation in `Dockerfile.jetson`
- **THEN** the Rust target SHALL be `aarch64-unknown-linux-gnu`
- **AND** the binary SHALL dynamically link against the host-mounted `libcuda.so.1` (glibc-compatible) provided by the NVIDIA Container Toolkit
- **AND** the workspace `rust-toolchain.toml` SHALL list `aarch64-unknown-linux-gnu` so `rustup target add` resolves cleanly

#### Scenario: Build enables Orin compute capability

- **GIVEN** the build command in `Dockerfile.jetson`
- **THEN** the cargo invocation SHALL pass `--features cuda,nvidia_gpu,smallaios-arch-nvidia/tegra-orin` (the `tegra-orin` feature on `smallaios-arch-nvidia` selects `cc_87`)
- **AND** the resulting binary SHALL be tuned for compute capability 8.7 (Ampere on Tegra Orin)
- **AND** the build SHALL NOT enable `smallaios-arch-nvidia/tegra` (which selects the Tegra X1 / cc 5.3 bare-metal HAL — irrelevant to the container path)

#### Scenario: Compose profile separation

- **GIVEN** `docker-compose.yml`
- **THEN** the full-JetPack Jetson service SHALL live under a `jetson` profile and the slim Jetson service under a `jetson-slim` profile, both distinct from the existing `gpu` profile
- **AND** `docker compose --profile jetson up` and `docker compose --profile jetson-slim up` SHALL be the user-visible invocations
- **AND** the existing `gpu` profile (x86 + discrete GPU via `Dockerfile.cuda`) SHALL remain functional and unchanged

#### Scenario: Container runs as non-root user

- **GIVEN** the runtime stage of `Dockerfile.jetson` or `Dockerfile.jetson.slim`
- **THEN** the image SHALL declare a `USER` directive resolving to a dedicated non-root account (e.g. `smallaios` at UID 10001)
- **AND** that account SHALL own `/smallaios` and `/models` so a plain `docker run` works without manual permission fixups
- **AND** the `smallaios` binary SHALL launch with no Linux capabilities beyond what the unprivileged user inherits — listening on 8080 (>1024) and reading `/models` are the only privileged-resource interactions, neither of which requires root
- **AND** any future Dockerfile that exposes a network endpoint SHALL follow the same non-root pattern

#### Scenario: Slim variant runtime base

- **GIVEN** `Dockerfile.jetson.slim`
- **THEN** the runtime stage base SHALL be `nvcr.io/nvidia/l4t-cuda:12.6.x-runtime` (or a documented equivalent), NOT the full L4T JetPack image
- **AND** the cuDNN .so files (`libcudnn.so.9`, `libcudnn_cnn`, `libcudnn_ops`, `libcudnn_graph`, `libcudnn_adv`, `libcudnn_engines_precompiled`, `libcudnn_engines_runtime_compiled`, `libcudnn_heuristic`) SHALL be copied from the JetPack builder stage into `/usr/lib/aarch64-linux-gnu/` of the runtime image
- **AND** `ldconfig` SHALL be run after the copy so the runtime linker picks them up
- **AND** the resulting image SHALL boot with `compute 8.7` and pass the same smoke test as `Dockerfile.jetson`
- **AND** the resulting image SHALL be substantially smaller than the full-JetPack variant (target: at least 2× smaller; observed: 4.09 GB vs 9.83 GB on dev box)

### Requirement: Jetson runtime smoke test

The repository SHALL ship a `scripts/test-jetson-gpu.sh` smoke test that validates the Jetson container end-to-end on real hardware.

#### Scenario: GPU init confirmed in logs

- **GIVEN** the Jetson container started by the smoke test on a Jetson Orin NX / AGX / Nano (Super)
- **WHEN** the smoke test inspects the container logs after the readiness probe succeeds
- **THEN** the logs SHALL contain a line of the form `CUDA initialized: <name> (compute 8.7, ...)`
- **AND** the smoke test SHALL fail loudly if `compute 8.7` is missing (e.g. silent CPU fallback, or wrong device cc)

#### Scenario: SqueezeNet inference returns 200

- **GIVEN** the smoke test has booted the container with `models/squeezenet.onnx` loaded
- **WHEN** the smoke test issues a POST against `/v1/inference` for the `squeezenet` model with a properly-shaped float32 input tensor
- **THEN** the response SHALL be HTTP 200 with a JSON body containing the model's output tensor

#### Scenario: Always cleans up

- **GIVEN** any path through the smoke test (success, assertion failure, timeout, ctrl-c)
- **WHEN** the script exits
- **THEN** the smoke test SHALL run `docker compose --profile jetson down` so no Jetson container leaks across runs

### Requirement: Jetson advisory CI job

The CI workflow SHALL include a job that builds `Dockerfile.jetson` under QEMU emulation on every PR, marked `continue-on-error: true` until a self-hosted Jetson runner is available.

#### Scenario: Dockerfile rot caught in CI

- **GIVEN** a PR that breaks `Dockerfile.jetson` in a way that prevents image build (e.g. base image tag rename, missing apt package)
- **WHEN** the `jetson-image-build` advisory job runs
- **THEN** it SHALL surface the failure in the PR check status
- **AND** because it is advisory, it SHALL NOT block merge — gating waits until a self-hosted Jetson runner can also do a runtime smoke test

