## ADDED Requirements

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
