## ADDED Requirements

### Requirement: CUDA FFI bindings for GPU inference
The `onnx-rt` crate SHALL provide hand-rolled `extern "C"` FFI bindings to CUDA, cuBLAS, and cuDNN libraries behind the `cuda` feature flag.

#### Scenario: CUDA device discovery
- **WHEN** the `cuda` feature is enabled and the container runs with `--gpus all`
- **THEN** `CudaProvider::new_from_runtime()` SHALL call `cudaGetDeviceCount` and `cudaGetDeviceProperties`
- **AND** it SHALL populate `GpuInfo` with real device name, compute capability, and VRAM size

#### Scenario: CUDA memory management
- **WHEN** the GPU provider allocates device memory
- **THEN** it SHALL use `cudaMalloc` for allocation and `cudaFree` for deallocation
- **AND** host-to-device and device-to-host transfers SHALL use `cudaMemcpy` with appropriate direction flags
- **AND** allocation failures SHALL return an error, not panic

#### Scenario: cuBLAS GEMM dispatch (Tier 1)
- **WHEN** the executor encounters `MatMul`, `Gemm`, or `MatMulInteger` operators with the GPU provider active
- **THEN** the provider SHALL dispatch to `cublasSgemm` (f32) or `cublasGemmEx` (i8)
- **AND** the result SHALL match the CPU reference output within 1e-5 relative tolerance for f32

#### Scenario: cuDNN convolution dispatch (Tier 2)
- **WHEN** the executor encounters a `Conv` operator with the GPU provider active
- **THEN** the provider SHALL dispatch to `cudnnConvolutionForward`
- **AND** the result SHALL match the CPU reference output within 1e-5 relative tolerance for f32

#### Scenario: Element-wise ops remain on CPU
- **WHEN** the executor encounters element-wise operators (Relu, Add, Sigmoid, etc.) with the GPU provider active
- **THEN** the provider SHALL NOT transfer these operations to the GPU
- **AND** data SHALL be transferred back to the host for CPU execution of these operators

### Requirement: Multi-precision GPU compute
The CUDA provider SHALL support multiple numeric precisions via `cublasGemmEx` compute type selection, matching the GB10 Blackwell hardware capabilities.

#### Scenario: TF32 tensor core compute (default)
- **WHEN** the GPU precision mode is `tf32` (default)
- **THEN** GEMM operators SHALL use `CUBLAS_COMPUTE_32F_FAST_TF32` for automatic tensor core acceleration
- **AND** inputs and outputs SHALL remain f32
- **AND** results SHALL match f32 reference within 1e-3 tolerance

#### Scenario: FP16 tensor core compute
- **WHEN** the GPU precision mode is `fp16`
- **THEN** GEMM operators SHALL use `CUBLAS_COMPUTE_32F_FAST_16F` for FP16 tensor core accumulation
- **AND** inputs and outputs SHALL remain f32

#### Scenario: BF16 tensor core compute
- **WHEN** the GPU precision mode is `bf16`
- **THEN** GEMM operators SHALL use `CUBLAS_COMPUTE_32F_FAST_16BF` for BF16 tensor core accumulation
- **AND** inputs and outputs SHALL remain f32

#### Scenario: INT8 GEMM with 4-aligned dimensions
- **WHEN** the executor encounters `MatMulInteger` with i8 inputs and dimensions that are multiples of 4
- **THEN** the provider SHALL dispatch to `cublasGemmEx` with `CUBLAS_COMPUTE_32I`
- **AND** the output SHALL be INT32
- **AND** if dimensions are not 4-aligned, the provider SHALL fall back to CPU

#### Scenario: FP8 GEMM (Tier 3)
- **WHEN** the executor encounters a GEMM operator with FP8 (E4M3 or E5M2) input tensors
- **THEN** the provider SHALL dispatch to `cublasGemmEx` with FP8 input types and f32 accumulation
- **AND** the output SHALL be f32

#### Scenario: INT4 GEMM (Tier 4)
- **WHEN** the executor encounters a GEMM operator with INT4 packed input tensors
- **THEN** the provider SHALL dispatch to cuBLAS IMMA with INT4 inputs and INT32 accumulation
- **AND** dimensions SHALL be multiples of 8

#### Scenario: Precision selection via environment variable
- **WHEN** `SMALLAIOS_GPU_PRECISION` is set to `f32`, `tf32`, `fp16`, or `bf16`
- **THEN** the container SHALL select the corresponding `cublasComputeType_t` for all f32 GEMM operators
- **AND** the default SHALL be `tf32` for best automatic performance

### Requirement: Weight preloading to GPU VRAM
Model weights SHALL be transferred to GPU memory once at session initialization, not per-inference.

#### Scenario: Bulk weight transfer at model load
- **WHEN** `Session::initialize()` is called with a GPU backend
- **THEN** all model initializer tensors (weights, biases) SHALL be transferred to GPU VRAM
- **AND** subsequent `Session::run()` calls SHALL use the preloaded GPU tensors without re-transfer

#### Scenario: Activation tensors transferred per-inference
- **WHEN** `Session::run()` is called with input tensors
- **THEN** input activation tensors SHALL be transferred host-to-device for the current inference
- **AND** output tensors SHALL be transferred device-to-host after inference completes

### Requirement: CUDA version compatibility check
The CUDA provider SHALL verify runtime compatibility at initialization.

#### Scenario: CUDA version validation
- **WHEN** `CudaProvider::new_from_runtime()` initializes
- **THEN** it SHALL query the CUDA runtime version
- **AND** if the major version does not match the compiled FFI bindings, initialization SHALL fail with a descriptive error
- **AND** the detected CUDA version SHALL be logged

### Requirement: GPU-enabled Dockerfile variant
A separate Dockerfile SHALL produce a GPU-enabled container image using NVIDIA CUDA base images.

#### Scenario: Build GPU container image
- **WHEN** `docker build -f Dockerfile.cuda -t smallaios:gpu .` is run
- **THEN** the builder stage SHALL compile with `--features cuda,nvidia_gpu`
- **AND** the runtime stage SHALL use `nvcr.io/nvidia/cuda:12.x-runtime-ubuntu24.04` as the base
- **AND** the image SHALL contain `libcudart`, `libcublas`, and `libcudnn` from the base image

#### Scenario: GPU container image supports ARM64
- **WHEN** the GPU Dockerfile is built for `linux/arm64`
- **THEN** the image SHALL use the ARM64 variant of the NVIDIA CUDA base image
- **AND** the SmallAIOS binary SHALL be compiled for `aarch64-unknown-linux-musl`

#### Scenario: CPU-only image unchanged
- **WHEN** the existing `Dockerfile` is built without GPU flags
- **THEN** the image SHALL remain `scratch`-based and under 15 MB
- **AND** no CUDA dependencies SHALL be included
