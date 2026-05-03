# cuda-container-runtime Specification

## Purpose
TBD - created by archiving changes safetensors-model-loader-v1 and jetson-orin-container-v1. Update Purpose after archive.
## Requirements
### Requirement: BF16 tensor I/O for GPU dispatch
GPU dispatch SHALL accept BF16 input tensors and produce BF16 output tensors directly, without requiring f32 conversion at the host boundary.

#### Scenario: BF16 GEMM input/output
- **WHEN** `gpu_gemm()` is invoked with BF16 input tensors
- **THEN** the device buffers SHALL be allocated with BF16 size (2 bytes per element)
- **AND** `cublasGemmEx` SHALL be called with `cudaDataType_t::CUDA_R_16BF` for input AND output types
- **AND** the result tensor SHALL have `DataType::BFloat16`

#### Scenario: Mixed precision compute with BF16 I/O
- **WHEN** the GPU precision mode is BF16 with BF16 input tensors
- **THEN** the cuBLAS call SHALL use `CUBLAS_COMPUTE_32F_FAST_16BF` (f32 accumulation, BF16 input)
- **AND** the output SHALL still be written as BF16 to match input dtype

### Requirement: GPU-resident model weight loading
GPU dispatch SHALL load model weight tensors from safetensors files directly into `DeviceBuffer`s without going through host-side `Tensor` objects.

#### Scenario: Direct mmap-to-VRAM weight transfer
- **WHEN** the safetensors loader is told to load weights into a GPU-resident graph
- **THEN** for each weight tensor, the loader SHALL allocate a `DeviceBuffer` of the correct size
- **AND** invoke `cudaMemcpy(HostToDevice)` directly from the safetensors mmap region to the device buffer
- **AND** SHALL NOT allocate a host-side `Vec<u8>` copy of the weight data

#### Scenario: VRAM accounting for weight load
- **WHEN** loading a model with N weight tensors totaling B bytes
- **THEN** total `DeviceBuffer` allocations SHALL be approximately B bytes (plus alignment overhead)
- **AND** the loader SHALL log total VRAM consumed by weights after load completes
- **AND** SHALL fail with a clear error if `cudaMalloc` returns out-of-memory

### Requirement: Per-session GPU resource lifecycle
The CUDA runtime SHALL allow multiple sessions to share a single GPU context while maintaining per-session resource ownership.

#### Scenario: Shared CudaRuntime across sessions
- **WHEN** multiple sessions are created with the same `Arc<CudaRuntime>`
- **THEN** each session SHALL share the same cuBLAS, cuBLASLt, and cuDNN handles
- **AND** allocate its own `DeviceBuffer`s for weights and KV cache
- **AND** each session's resources SHALL be freed independently when the session is dropped

#### Scenario: GPU OOM on session creation
- **WHEN** loading a session whose total weight + KV cache size exceeds available VRAM
- **THEN** the session creation SHALL return a clear error indicating VRAM exhaustion
- **AND** SHALL release any partially-allocated `DeviceBuffer`s before returning

### Requirement: Session is thread-safe under the cuda feature

When the `cuda` feature is enabled, `smallaios_onnx_rt::session::Session` SHALL be `Send + Sync` so it can be embedded in multi-threaded HTTP request handlers and `std::thread::spawn` closures.

#### Scenario: Session held across HTTP worker threads

- **GIVEN** the smallaios container compiled with `--features cuda,nvidia_gpu` is started with at least one model loaded
- **WHEN** the `HttpServer::route_fn` thread pool dispatches an inference request to a worker thread different from the one that constructed the `Session`
- **THEN** the worker SHALL be able to call `session.run(...)` without a Rust compile-time `Send`/`Sync` error
- **AND** the worker SHALL not race with concurrent requests on the same `Session` (writes to the GPU graph cache, stream pool, and device weight cache are serialized via `Mutex`)

#### Scenario: Static thread-safety assertion

- **GIVEN** any future change touches `Session` or any field reachable from it under the `cuda` feature
- **THEN** the workspace SHALL contain a `const _: fn() = || { fn assert_send_sync<T: Send + Sync>() {} assert_send_sync::<Session>(); };` (or equivalent static check)
- **AND** the assertion SHALL be unconditionally compiled when the `cuda` feature is on, so any regression is caught at `cargo check` time before reaching review

#### Scenario: Raw CUDA handles wrapped behind Send/Sync newtypes

- **GIVEN** any cached CUDA handle (`cudaGraphExec_t`, `cudaGraph_t`, `cudaStream_t`, `cublasHandle_t`, `cudnnHandle_t`) stored on `Session` or a type owned by `Session`
- **THEN** the handle SHALL be wrapped in a newtype with `unsafe impl Send + Sync` and a `// SAFETY:` comment naming the CUDA contract that justifies the impl
- **AND** the newtype SHALL be the only place those bounds are asserted unsafely (no scattered `unsafe impl Send` for `*mut c_void` etc.)

### Requirement: cargo check --features cuda is gated in CI

The repository SHALL enforce a CI gate that runs `cargo check --workspace --features cuda,nvidia_gpu` on every PR.

#### Scenario: cuda-only regression caught in CI

- **GIVEN** a PR that breaks `cargo check --features cuda` without breaking the default-feature build
- **WHEN** the PR pipeline runs
- **THEN** the `cuda-check` job SHALL fail
- **AND** the change-gates meta-job SHALL block merge until it is fixed
