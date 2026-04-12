## 1. ARM64 CPU Inference Validation (Phase 1)

- [x] 1.1 Verify `just build-container-arm` produces a working `aarch64-unknown-linux-musl` binary on the current codebase
- [x] 1.2 Build ARM64 Docker image and run it on DGX Spark (Grace CPU), confirm container boots and serves health endpoint
- [x] 1.3 Create validation model suite: download/generate ResNet-50, MobileNetV2, SqueezeNet, and simple MLP ONNX models for testing
- [x] 1.4 Run each validation model on ARM64 via CPU provider, record pass/fail and output tensors
- [x] 1.5 Compare ARM64 outputs against x86-64 reference outputs, verify 1e-5 relative tolerance for f32
- [x] 1.6 Produce operator gap report: list any operators that fail or are missing on ARM64, grouped by model
- [x] 1.7 Fix any ARM64-specific operator issues discovered (x86 intrinsics, alignment, endianness)
- [x] 1.8 Re-run validation suite and confirm all models pass on ARM64

## 2. ARM64 CI Integration

- [x] 2.1 Add QEMU-emulated ARM64 container build job to `.github/workflows/ci.yml`
- [x] 2.2 Configure the job to cross-compile via `just build-container-arm` and build the Docker image
- [x] 2.3 Add ARM64 smoke test step: run a minimal MLP model under QEMU emulation, verify non-error output
- [x] 2.4 Verify the CI job passes on a PR and does not block x86-64 jobs on failure (advisory initially)

## 3. CUDA FFI Bindings

- [x] 3.1 Create `onnx-rt/src/cuda/mod.rs` module behind `#[cfg(feature = "cuda")]`
- [x] 3.2 Create `onnx-rt/src/cuda/ffi.rs` with `extern "C"` declarations for CUDA runtime: `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaGetDeviceCount`, `cudaGetDeviceProperties`, `cudaRuntimeGetVersion`
- [x] 3.3 Add `extern "C"` declarations for cuBLAS: `cublasCreate`, `cublasDestroy`, `cublasSgemm`, `cublasGemmEx`
- [x] 3.4 Add `extern "C"` declarations for cuDNN: `cudnnCreate`, `cudnnDestroy`, `cudnnConvolutionForward`, `cudnnCreateTensorDescriptor`, `cudnnCreateFilterDescriptor`, `cudnnCreateConvolutionDescriptor`
- [x] 3.5 Create safe Rust wrappers in `onnx-rt/src/cuda/mod.rs` for each FFI function with error handling (CUDA status code → Result)
- [x] 3.6 Add `cuda` feature flag to `onnx-rt/Cargo.toml` with appropriate `#[link]` directives for `cudart`, `cublas`, `cudnn`
- [x] 3.7 Write unit tests for FFI wrapper error paths (mock/cfg-gated so they compile without CUDA)

## 4. CUDA Device Memory Manager

- [x] 4.1 Create `onnx-rt/src/cuda/memory.rs` with `DeviceBuffer` struct (pointer + size + Drop for `cudaFree`)
- [x] 4.2 Implement `DeviceBuffer::alloc(size)` → `Result<DeviceBuffer>` wrapping `cudaMalloc`
- [x] 4.3 Implement `DeviceBuffer::copy_from_host(&[u8])` and `DeviceBuffer::copy_to_host(&mut [u8])` wrapping `cudaMemcpy`
- [x] 4.4 Implement `DeviceWeightStore` for bulk-transferring model initializers to VRAM at session load
- [x] 4.5 Write tests for allocation, transfer, and deallocation lifecycle

## 5. CudaProvider Implementation

- [x] 5.1 Replace stub `CudaProvider::new()` in `arch/nvidia/src/cuda_provider.rs` with `new_from_runtime()` that calls `cudaGetDeviceCount` and `cudaGetDeviceProperties`
- [x] 5.2 Add CUDA version compatibility check: query runtime version, fail fast if major version mismatches compiled bindings
- [x] 5.3 Implement `ComputeProvider::supports_op()` returning true for MatMul, Gemm, MatMulInteger (Tier 1) and Conv (Tier 2)
- [x] 5.4 Implement GPU dispatch for MatMul/Gemm via `cublasSgemm`: handle transpose flags, alpha/beta, leading dimensions
- [x] 5.5 Implement GPU dispatch for MatMulInteger via `cublasGemmEx` with INT8 compute type
- [x] 5.6 Implement GPU dispatch for Conv via `cudnnConvolutionForward`: create descriptors, select algorithm, execute
- [x] 5.7 Implement host↔device tensor transfer around GPU-dispatched ops (copy input activations to device, copy output back)
- [x] 5.8 Write numerical correctness tests comparing GPU output against CPU reference (1e-5 tolerance for f32, 1e-3 for f16)

## 6. Provider Wiring (Container → Session → Executor)

- [x] 6.1 Implement `GpuBackend::from_env("cuda")` in `compute/src/lib.rs` that creates a `CudaProvider` via `new_from_runtime()`
- [x] 6.2 Update `container/src/main.rs` to read `SMALLAIOS_GPU_BACKEND` env var and create `GpuBackend` (currently ignored)
- [x] 6.3 Wire `GpuBackend` through `SessionConfig { gpu_backend: Some(backend) }` in session construction
- [x] 6.4 Update `Session::initialize()` to store the GPU backend and call `DeviceWeightStore` for weight preloading
- [x] 6.5 Update `execute_graph()` to pass `&GpuBackend` to `dispatch_node()`
- [x] 6.6 Update `dispatch_node()` in `onnx-rt/src/executor.rs` to check `supports_op()` and dispatch to GPU or fall through to CPU

## 7. GPU-to-CPU Fallback

- [x] 7.1 Implement fallback logic in container boot: if `CudaProvider::new_from_runtime()` fails, log warning and use `CpuFallback`
- [x] 7.2 Handle `cudaGetDeviceCount` returning 0: log "GPU requested but not found", fall back to CPU
- [x] 7.3 Handle missing CUDA libraries (dlopen failure): log specific error, fall back to CPU
- [x] 7.4 Ensure fallback is transparent: same `Session::run()` interface regardless of provider
- [x] 7.5 Write integration test: start with `SMALLAIOS_GPU_BACKEND=cuda` but no GPU available, verify CPU inference succeeds

## 8. GPU-Enabled Dockerfile

- [x] 8.1 Create `Dockerfile.cuda` with builder stage using Rust nightly + musl cross-compilation
- [x] 8.2 Set runtime stage to `nvcr.io/nvidia/cuda:13.0.0-runtime-ubuntu24.04` with ARM64 support
- [x] 8.3 Copy compiled binary into runtime stage, set entrypoint to `/smallaios`
- [x] 8.4 Add `SMALLAIOS_GPU_BACKEND=cuda` as default env var in the GPU Dockerfile
- [x] 8.5 Update `docker-compose.yml` GPU profile to use `Dockerfile.cuda` and `nvidia` runtime
- [x] 8.6 Verify existing CPU-only `Dockerfile` and `scratch` image are unchanged and still under 15 MB
- [x] 8.7 Add `just docker-build-gpu` recipe to `justfile` for building the GPU variant

## 9. GPU CI Integration

- [x] 9.1 Add `docker-build-gpu` CI job that builds `Dockerfile.cuda` (no GPU runner needed, build-only)
- [x] 9.2 Verify GPU Dockerfile builds for both `linux/amd64` and `linux/arm64` platforms
- [x] 9.3 Document self-hosted runner setup for DGX Spark GPU tests (manual until runner available)

## 10. Tier 2 Precision: TF32 / FP16 / BF16 Compute Modes

- [x] 10.1 Add `GpuPrecision` enum to `cuda/mod.rs` with variants: `F32`, `Tf32`, `Fp16`, `Bf16`, `Int8`
- [x] 10.2 Update `gpu_gemm()` to accept a `GpuPrecision` parameter, selecting the `cublasComputeType_t` accordingly
- [x] 10.3 Wire precision selection from `SMALLAIOS_GPU_PRECISION` env var (default: `tf32` for best auto performance)
- [x] 10.4 Add tests: TF32 GEMM (same inputs as f32, verify reduced precision is within 1e-3)
- [x] 10.5 Add tests: FP16 compute path (f32 I/O with FP16 tensor core accumulation)
- [x] 10.6 Add tests: BF16 compute path (f32 I/O with BF16 tensor core accumulation)

## 11. cuDNN Conv Dispatch (Tier 2 Operator)

- [x] 11.1 Create `cuda/conv.rs` with `gpu_conv2d()` function wrapping cuDNN convolution forward
- [x] 11.2 Implement cuDNN descriptor creation: tensor (NCHW), filter, convolution (pad, stride, dilation)
- [x] 11.3 Implement algorithm selection via `cudnnConvolutionForward` with `IMPLICIT_PRECOMP_GEMM`
- [x] 11.4 Wire `gpu_conv2d()` into `try_cuda_dispatch()` for the `Conv` operator
- [x] 11.5 Implement im2col + cuBLAS GEMM fallback for shapes cuDNN can't handle
- [x] 11.6 Add tests: 1x1 conv, 3x3 conv, strided conv, dilated conv — compare GPU vs CPU
- [x] 11.7 Add tests: Conv with multiple precision modes (TF32 default, FP16 optional)

## 12. Tier 3 Precision: FP8 (E4M3 / E5M2)

- [x] 12.1 Add `CUDA_R_8F_E4M3` (28) and `CUDA_R_8F_E5M2` (29) to `cudaDataType_t` enum in `ffi.rs`
- [x] 12.2 Implement `gpu_gemm_fp8()` in `dispatch.rs` using cuBLASLt with FP8 input types and f32 accumulation
- [ ] 12.3 Wire FP8 dispatch from executor for `MatMul` ops on FP8-typed tensors
- [ ] 12.4 Add FP8 tensor data type support to `tensor.rs` (`DataType::Float8E4M3`, `DataType::Float8E5M2`)
- [x] 12.5 Add tests: FP8 GEMM numerical correctness vs f32 reference (tolerance 1e-2 for E4M3, 1e-1 for E5M2)
- [ ] 12.6 Validate with a real FP8-quantized ONNX model (e.g. ORT FP8 export) if available

## 13. Tier 4 Precision: INT4 / FP4 (Blackwell-specific)

- [x] 13.1 Research cuBLAS INT4 packed format: 2 values per byte, confirm packing convention
- [x] 13.2 Add `CUDA_R_4I` and `CUDA_R_4F` to `cudaDataType_t` if supported by CUDA 13.0 headers
- [ ] 13.3 Implement `gpu_gemm_int4()` with packed INT4 input and INT32 accumulation
- [ ] 13.4 Add FP4 (E2M1) support if Blackwell-specific APIs are available in CUDA 13.0
- [ ] 13.5 Add INT4/FP4 tensor data types to `tensor.rs`
- [ ] 13.6 Add tests: INT4 GEMM with 8-aligned dimensions (INT4 likely needs stricter alignment)
- [ ] 13.7 Validate with a real 4-bit quantized ONNX model (e.g. GPTQ or AWQ export)

## 14. Phase 3 Validation and Benchmarks

- [ ] 14.1 Run end-to-end model benchmarks on DGX Spark: CPU vs GPU inference latency for ResNet-50, MobileNetV2, SqueezeNet, MLP
- [ ] 14.2 Profile GPU memory usage: verify weight preloading VRAM consumption stays within budget for each test model
- [x] 14.3 Audit GPU container image size, document the size trade-off vs CPU-only image
- [x] 14.4 Verify GPU results match CPU reference outputs within tolerance for all test models
- [x] 14.5 Create deployment examples: `docker-compose.yml` GPU profile, K8s manifest with `nvidia.com/gpu` resource request
- [x] 14.6 Document DGX Spark memory architecture findings (unified vs discrete VRAM) and any impact on allocation strategy
- [x] 14.7 Benchmark precision modes: FP32 vs TF32 vs FP16 vs INT8 throughput on GEMM-heavy models
- [x] 14.8 Document precision selection guidance: which mode for which model type
