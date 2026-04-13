## Context

SmallAIOS has a complete provider architecture for GPU inference — `ComputeProvider` trait in `compute/src/lib.rs`, `CudaProvider` stub in `arch/nvidia/src/cuda_provider.rs`, feature flags (`cuda`, `gpu`) in `onnx-rt/Cargo.toml`, and GPU backend plumbing in the executor (`onnx-rt/src/executor.rs`). However, none of it is wired to real hardware:

- `CudaProvider::new()` initializes fake GPU info (Tesla T4 / Tegra GM20B) and stub allocators
- `dispatch_node()` checks `_gpu_supported` but always falls through to CPU
- `Session` is always created with `gpu_backend: None`
- The container reads `SMALLAIOS_GPU_BACKEND` env var but ignores it

The target hardware is a DGX Spark: ARM64 Grace CPU + Blackwell GPU. The NVIDIA Container Toolkit (`nvidia-ctk`) provides GPU access inside containers via `--gpus all`, mounting the CUDA driver and libraries into the container filesystem. This is the standard approach for containerized GPU workloads — no kernel-level GPU drivers needed from SmallAIOS.

**Current file touchpoints:**
- `arch/nvidia/src/cuda_provider.rs:87-380` — CudaProvider stub + ComputeProvider impl
- `compute/src/lib.rs:25-276` — ComputeProvider trait, GpuBackend enum, CpuFallback
- `onnx-rt/src/executor.rs:268-399` — dispatch_node with unused GPU branch
- `onnx-rt/src/session.rs:159-349` — Session with gpu_backend: None
- `container/src/main.rs:40-164` — boot flow, load_sessions, ignored GPU env var

## Goals / Non-Goals

**Goals:**
- Validate ARM64 CPU inference with real ONNX models on DGX Spark hardware
- Add real CUDA dispatch via cuBLAS/cuDNN FFI, replacing the CudaProvider stub
- Wire the existing provider architecture end-to-end: container env var → CudaProvider → Session → executor → GPU dispatch
- Produce a GPU-enabled Docker image using NVIDIA base images with standard CUDA runtime
- Graceful CPU fallback when no GPU is available

**Non-Goals:**
- Bare-metal GPU HAL or register-level programming (no changes to `arch/nvidia` platform init)
- Custom CUDA kernels (`.cu` files) — use cuBLAS/cuDNN library calls only
- Multi-GPU or tensor parallelism
- TensorRT or ONNX Runtime delegation
- Performance parity with ORT/TensorRT
- LLM / generative model support (separate change: `generative-llm-v1`)

## Decisions

### 1. CUDA FFI via `extern "C"` bindings, not `cuda-sys` crate

**Decision:** Write minimal hand-rolled FFI declarations for ~15 CUDA/cuBLAS/cuDNN functions rather than depending on `cuda-sys` or `cudarc`.

**Rationale:**
- SmallAIOS is `#![no_std]` in the onnx-rt crate — existing CUDA binding crates assume `std`
- We need a very small surface: `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaGetDeviceCount`, `cudaGetDeviceProperties`, `cublasCreate`, `cublasSgemm`, `cublasGemmEx`, `cudnnCreate`, `cudnnConvolutionForward`, and a handful more
- Hand-rolled FFI avoids pulling in a crate that wraps 2000+ CUDA functions we don't need
- The `cuda` feature flag already exists; FFI module lives behind it

**Alternative considered:** `cudarc` crate — provides safe wrappers but requires `std`, adds ~50KB to binary, and wraps far more surface area than needed.

**Location:** New `onnx-rt/src/cuda/ffi.rs` with raw `extern "C"` blocks, wrapped by safe Rust APIs in `onnx-rt/src/cuda/mod.rs`.

### 2. GPU-enabled Dockerfile uses NVIDIA base image, not scratch

**Decision:** Add a separate build stage / Dockerfile variant (`Dockerfile.cuda`) that uses `nvcr.io/nvidia/cuda:12.8-runtime-ubuntu24.04` as the runtime base instead of `scratch`.

**Rationale:**
- The `scratch`-based image stays <15 MB for CPU-only deployments — this is a core SmallAIOS constraint
- CUDA runtime libraries (`libcudart.so`, `libcublas.so`, `libcudnn.so`) add ~200-500 MB — this is unavoidable and matches every other GPU inference container
- Using NVIDIA's official base image ensures driver compatibility with the Container Toolkit
- ARM64 variants of NVIDIA base images are available on NGC

**Alternative considered:** Mount CUDA libs from host via volume mounts — fragile, version-mismatch-prone, non-portable across CUDA versions.

### 3. Operator offload: cuBLAS for GEMM ops first, cuDNN for Conv second

**Decision:** Implement GPU dispatch in two tiers:
- **Tier 1:** `MatMul`, `Gemm`, `MatMulInteger` via cuBLAS (`cublasSgemm`, `cublasGemmEx` for int8)
- **Tier 2:** `Conv` via cuDNN (`cudnnConvolutionForward`)

Element-wise ops (Relu, Add, Sigmoid, etc.) stay on CPU — the host↔device transfer cost exceeds compute savings for typical tensor sizes until we have fused kernels.

**Rationale:**
- GEMM dominates inference compute (70-90% of wall time for CNNs and transformers)
- cuBLAS is the simplest FFI surface — one function call per GEMM
- cuDNN for Conv is the second biggest win and has a well-defined C API
- Element-wise GPU dispatch requires either custom kernels or cuDNN's activation API, with diminishing returns until fused op patterns are implemented

**Alternative considered:** Offload everything to GPU — requires custom CUDA kernels for every op, massive implementation scope, marginal benefit for non-GEMM ops.

### 4. Provider wiring: factory function in container, plumbed through Session

**Decision:** Add a provider factory in the container that reads `SMALLAIOS_GPU_BACKEND` and creates a `GpuBackend`, then plumb it through `SessionConfig` → `Session` → `execute_graph` → `dispatch_node`.

**Flow:**
```
container/main.rs: SMALLAIOS_GPU_BACKEND="cuda"
  → compute::GpuBackend::from_env("cuda")
  → CudaProvider::new_from_runtime()  // probes cudaGetDeviceCount
  → SessionConfig { gpu_backend: Some(backend) }
  → Session::initialize() stores backend
  → execute_graph() passes &GpuBackend to dispatch_node()
  → dispatch_node() checks supports_op(), dispatches to GPU or CPU
```

**Rationale:** This follows the existing architecture — `GpuBackend`, `ComputeProvider`, feature flags, and the `dispatch_node` GPU check all exist. The gap is wiring, not architecture. Minimal code changes to connect existing pieces.

### 5. Weight preloading: bulk transfer at model load, not per-inference

**Decision:** Transfer model weights (initializers) to GPU VRAM during `Session::initialize()`, not during each `Session::run()`. Activation tensors are transferred per-inference.

**Rationale:**
- Model weights are constant across inferences — transferring once amortizes PCIe/memory bus cost
- Activation tensors change each inference and must be transferred each time
- DGX Spark has 128 GB unified memory (Grace Hopper architecture uses NVLink-C2C between CPU and GPU), so transfer costs may be lower than discrete GPU, but preloading is still the right pattern

### 6. ARM64 CI: QEMU-emulated container build, native GPU tests manual

**Decision:** CI gets a QEMU-emulated ARM64 container build job (validates compilation + basic CPU inference). GPU tests on DGX Spark are manual until a self-hosted runner is available.

**Rationale:**
- GitHub Actions supports ARM64 via QEMU emulation — slow but works for build validation
- GPU testing requires physical hardware with NVIDIA drivers — can't be emulated
- A self-hosted runner on the DGX Spark is the eventual target but requires setup

## Risks / Trade-offs

**[Container size blowup]** → GPU image jumps from <15 MB to ~200-500 MB. **Mitigation:** CPU-only `scratch` image remains the default; GPU image is a separate variant. Document the size trade-off clearly. Users who need GPU accept the size.

**[CUDA version coupling]** → FFI bindings target a specific CUDA API version (12.x). Driver/runtime version mismatches can cause silent failures. **Mitigation:** Runtime version check in `CudaProvider::new_from_runtime()` — log CUDA version, fail fast if major version doesn't match compiled bindings.

**[ARM64 + CUDA edge cases]** → NVIDIA's ARM64 CUDA support is newer than x86. Some cuBLAS/cuDNN functions may behave differently or have different performance characteristics. **Mitigation:** Phase 1 validates CPU inference first, establishing a correctness baseline. Phase 2 GPU results compared against CPU reference outputs.

**[DGX Spark unified memory model]** → Grace-Blackwell uses NVLink-C2C unified memory, not discrete PCIe. Standard `cudaMalloc`/`cudaMemcpy` still works but `cudaMallocManaged` (unified virtual addressing) might be more efficient. **Mitigation:** Start with explicit transfers (`cudaMalloc` + `cudaMemcpy`), profile, then consider UVA in a follow-up.

**[no_std compatibility]** → CUDA FFI calls require linking against shared libraries, which works in container mode (`std` via musl) but not in bare-metal kernel mode. **Mitigation:** All CUDA code is behind `#[cfg(feature = "cuda")]` and the `cuda` feature implies `gpu` which is container-only. The kernel build never enables `cuda`.

**[Operator correctness divergence]** → GPU floating-point results may differ from CPU (different rounding, FMA behavior). **Mitigation:** Tolerance-based comparison (1e-5 relative error for f32, 1e-3 for f16) in validation tests, matching ONNX Runtime's conformance approach.

## Resolved Questions

1. **CUDA version target**: **CUDA 13.0** (installed on DGX Spark). FFI bindings target major version 13. `Dockerfile.cuda` uses `nvcr.io/nvidia/cuda:13.0.0-runtime-ubuntu24.04`.
2. **cuDNN vs cuBLAS-only for Conv**: cuDNN is installed (9.20.0.48). Using `cudnnConvolutionForward` for Conv dispatch.
3. **DGX Spark memory architecture**: **128 GB unified memory** shared between Grace CPU and GB10 GPU. `cudaMalloc`/`cudaMemcpy` work correctly. Unified virtual addressing (`cudaMallocManaged`) is a future optimization.
4. **DGX Spark GPU identity**: NVIDIA GB10, Blackwell architecture, **Compute Capability 12.1**, 48 SMs, SBSA platform (server-grade ARM64, not Tegra).

## Decisions (Addendum)

### 7. Multi-precision GPU dispatch via cublasGemmEx

**Decision:** Support all precision modes that the GB10 hardware provides and that map to ONNX quantized model formats. Implement in tiers by ONNX model prevalence:

**Validated on hardware (GB10, CC 12.1):**

| Precision | cuBLAS Compute Type | Input → Output | Status |
|-----------|-------------------|----------------|--------|
| FP32 | `CUBLAS_COMPUTE_32F` (68) | f32 → f32 | **Implemented** via `cublasSgemm` |
| TF32 | `CUBLAS_COMPUTE_32F_FAST_TF32` (77) | f32 → f32 | Available (auto tensor core) |
| FP16 | `CUBLAS_COMPUTE_32F_FAST_16F` (74) | f32 → f32 | Available via `cublasGemmEx` |
| BF16 | `CUBLAS_COMPUTE_32F_FAST_16BF` (75) | f32 → f32 | Available via `cublasGemmEx` |
| INT8 | `CUBLAS_COMPUTE_32I` (72) | i8 → i32 | **Implemented** via `cublasGemmEx` (4-aligned dims) |
| FP8 (E4M3) | `CUBLAS_COMPUTE_32F` with `CUDA_R_8F_E4M3` | fp8 → f32 | Hardware ready, needs FFI types |
| INT4 | IMMA tensor core | i4 → i32 | Hardware ready, needs packed format |
| FP4 (E2M1) | Blackwell-specific | fp4 → f32 | Hardware ready, experimental |

**Rationale:** The GB10 Blackwell GPU has the widest precision support of any NVIDIA GPU. Rather than limit to f32+i8, expose the full precision stack so SmallAIOS can dispatch ONNX quantized models at their native precision without dequantize→compute→requantize overhead.

**Implementation tiers:**
- **Tier 1 (done):** FP32 (`cublasSgemm`) + INT8 (`cublasGemmEx` COMPUTE_32I)
- **Tier 2:** TF32/FP16/BF16 via `cublasGemmEx` compute type selection — minimal code, just a different enum value on existing GEMM path
- **Tier 3:** FP8 (E4M3/E5M2) — needs `cudaDataType_t` additions (`CUDA_R_8F_E4M3 = 28`, `CUDA_R_8F_E5M2 = 29`) and ONNX FP8 quantized model support
- **Tier 4:** INT4/FP4 — packed formats, Blackwell-specific APIs, wait for ONNX ecosystem maturity

**Note:** FP64 is hardware-supported but not needed — no ONNX inference models use f64 compute.

### 8. Conv dispatch: cuDNN preferred, im2col+cuBLAS fallback

**Decision:** Primary Conv dispatch via `cudnnConvolutionForward`. If cuDNN init fails or descriptors can't be created for a particular Conv shape, fall back to im2col + cuBLAS GEMM.

**Rationale:** cuDNN selects the optimal algorithm (Winograd, FFT, direct, implicit GEMM) per conv shape automatically. im2col+GEMM is a correct fallback that covers all shapes but is ~15% slower for typical CNN layers.

## Open Questions

1. **FP8 ONNX ecosystem readiness**: Which ONNX model exporters produce FP8 quantized models today? ONNX opset 21 has `QuantizeLinear`/`DequantizeLinear` with FP8 types but real-world adoption is TBD. FP8 GEMM via cuBLASLt is validated and working on GB10.
### 9. GB10 GEMM Precision Benchmarks (measured)

Latency in milliseconds per GEMM call, averaged over 20 iterations on GB10 (CC 12.1, 48 SMs):

| Size | F32 (ms) | TF32 (ms) | FP16 (ms) | INT8 (ms) | TF32 speedup |
|------|----------|-----------|-----------|-----------|-------------|
| 64 | 0.006 | 0.007 | 0.007 | 0.004 | 0.9x |
| 128 | 0.009 | 0.004 | 0.004 | 0.008 | 2.3x |
| 256 | 0.012 | 0.006 | 0.006 | 0.012 | 2.0x |
| 512 | 0.029 | 0.012 | 0.013 | 0.019 | 2.4x |
| 1024 | 0.129 | 0.062 | 0.064 | 0.061 | 2.1x |
| 2048 | 0.992 | 0.451 | 0.383 | 0.342 | 2.2x |

**Key findings:**
- TF32 delivers **~2.2x speedup** over F32 at meaningful sizes (512+) with negligible accuracy loss
- FP16 is comparable to TF32 (slightly faster at 2048)
- INT8 is fastest at large sizes (**2.9x** over F32 at 2048)
- Small matrices (<256) are launch-latency dominated — precision mode doesn't matter
- **Default TF32 is the right choice** for general inference (2x faster, transparent to users)

## Open Questions

2. **INT4/FP4 not in CUDA 13.0 public headers**: `CUDA_R_4I` and `CUDA_R_4F` are not defined in CUDA 13.0 `cuda_runtime.h`. INT4 GEMM requires cuBLASLt IMMA with custom packing (2 values per byte, device-side memory layout TBD). FP4 (E2M1) is Blackwell-specific and may require a future CUDA toolkit update. Deferred to a follow-up change when ONNX INT4/FP4 ecosystem matures.
