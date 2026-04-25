## ADDED Requirements

### Requirement: Automatic GPU-to-CPU fallback
The container SHALL automatically fall back to CPU inference when GPU hardware or the NVIDIA runtime is unavailable.

#### Scenario: No GPU detected at startup
- **WHEN** the container starts with `SMALLAIOS_GPU_BACKEND=cuda` but no GPU is available (`cudaGetDeviceCount` returns 0)
- **THEN** the container SHALL log a warning indicating GPU was requested but not found
- **AND** the container SHALL fall back to CPU-only inference
- **AND** model loading and inference SHALL proceed without error

#### Scenario: Container run without --gpus flag
- **WHEN** the GPU-enabled container image is started without `--gpus all`
- **THEN** CUDA libraries SHALL fail to initialize (no device visible)
- **AND** the container SHALL fall back to CPU-only inference with a logged warning

#### Scenario: CUDA initialization failure
- **WHEN** `CudaProvider::new_from_runtime()` fails for any reason (driver mismatch, library not found, device error)
- **THEN** the container SHALL log the specific CUDA error
- **AND** the container SHALL fall back to `CpuFallback` provider
- **AND** the fallback SHALL be transparent to the inference API (same `Session::run()` interface)

#### Scenario: GPU backend env var unset
- **WHEN** the container starts without `SMALLAIOS_GPU_BACKEND` set (or set to `cpu`)
- **THEN** the container SHALL use CPU-only inference
- **AND** no CUDA initialization SHALL be attempted

### Requirement: Provider selection plumbing
The container boot flow SHALL wire `SMALLAIOS_GPU_BACKEND` through to `Session` construction via the existing provider architecture.

#### Scenario: End-to-end provider wiring
- **WHEN** `SMALLAIOS_GPU_BACKEND=cuda` and a GPU is available
- **THEN** the container SHALL create a `GpuBackend` via `GpuBackend::from_env("cuda")`
- **AND** pass it through `SessionConfig { gpu_backend: Some(backend) }` to `Session::initialize()`
- **AND** `dispatch_node()` SHALL check `supports_op()` and dispatch to GPU for supported operators

#### Scenario: Mixed GPU/CPU operator execution
- **WHEN** a model contains both GPU-supported ops (MatMul, Conv) and CPU-only ops (Relu, Reshape)
- **THEN** `dispatch_node()` SHALL dispatch each operator to the appropriate provider
- **AND** tensor data SHALL be transferred between host and device as needed between operators
