## Why

SmallAIOS builds and runs as an ARM64 Docker container (`aarch64-unknown-linux-musl`) but has never been validated on real ARM64+NVIDIA hardware. The DGX Spark (Grace-Blackwell, ARM64 host + Blackwell GPU) is the first available target for end-to-end validation. The ONNX runtime has 29+ CPU operators covering CNNs, MLPs, and basic transformers — enough to run real models — but nobody has tested them on ARM64 hardware yet.

GPU acceleration in SmallAIOS today is a stub: the `nvidia_gpu` feature flag compiles but the `CudaProvider` never dispatches to real hardware. The archived `tegra-gpu-hal-v1` change targeted bare-metal Tegra X1 register programming, which doesn't apply to DGX Spark and isn't the right approach for container deployments anyway. Containers should use NVIDIA's standard toolchain: the NVIDIA Container Toolkit (`nvidia-ctk`), CUDA base images, and cuDNN/cuBLAS libraries — the same stack every other inference framework uses. Container-first; bare-metal GPU HAL is future work.

## What Changes

### Phase 1 — ARM64 CPU Inference Validation

- **Cross-compile and run** the existing `smallaios-container` Docker image on the DGX Spark (ARM64) using CPU-only inference
- **Test with standard ONNX models**: ResNet-50, MobileNetV2, SqueezeNet, BERT-base (if transformer ops are sufficient), simple MLP — exercising the 29+ implemented operators on real ARM64 hardware
- **Identify operator gaps**: document which models fail and which operators are missing
- **CI integration**: add an ARM64 container build+test job (QEMU-emulated or native runner if available)

### Phase 2 — NVIDIA Container Toolkit GPU Integration

- **GPU-enabled Dockerfile variant**: based on `nvcr.io/nvidia/cuda:12.x-runtime-ubuntu22.04` (ARM64) instead of `scratch`, providing `libcudart`, `libcublas`, `libcudnn` at runtime
- **CUDA FFI bindings** in `onnx-rt`: link against `libcudart`/`libcublas` via `extern "C"` FFI (behind `cuda` feature flag) for memory allocation (`cudaMalloc`/`cudaMemcpy`) and GEMM dispatch (`cublasSgemm`/`cublasHgemm`)
- **CudaProvider implementation**: replace the stub with real dispatch — host-to-device transfer, cuBLAS GEMM for `MatMul`/`Conv`/`Gemm`, device-to-host transfer
- **Operator offload strategy**: start with `MatMul` and `Gemm` (cuBLAS), then `Conv` (cuDNN), then batch remaining compute-heavy operators. Element-wise ops stay on CPU initially.
- **Runtime detection**: probe for GPU availability at startup (`cudaGetDeviceCount`), fall back to CPU gracefully if no GPU or if running without `--gpus`

### Phase 3 — Validation and Benchmarks

- **End-to-end model benchmarks**: CPU vs GPU inference latency on the DGX Spark for the test model suite
- **Memory profiling**: ensure GPU memory allocation stays within VRAM budget for target models
- **Container size audit**: GPU container will be larger than <15 MB target (CUDA runtime adds ~200-500 MB); document the trade-off and maintain CPU-only image as the slim variant
- **Deployment examples**: docker-compose and K8s manifests with GPU resource requests

## Capabilities

### New Capabilities
- `arm64-inference-validation`: ARM64 container image validated with real ONNX model inference on DGX Spark hardware, including test model suite and operator gap analysis
- `cuda-container-runtime`: GPU-accelerated ONNX inference via NVIDIA Container Toolkit, cuBLAS FFI bindings, and cuDNN — no bare-metal HAL required
- `gpu-cpu-fallback`: Automatic runtime fallback from GPU to CPU inference when NVIDIA runtime is unavailable

### Modified Capabilities
- `docker-multiarch`: Extended with GPU-enabled ARM64 variant using NVIDIA CUDA base images
- `onnx-cpu-execution`: Validated on ARM64 architecture (previously only tested on x86-64)

## Impact

- **`Dockerfile`**: Add GPU-enabled build stage using NVIDIA CUDA base image; existing `scratch`-based CPU image unchanged
- **`onnx-rt/src/cuda/`**: New module with CUDA FFI bindings (`ffi.rs`), device memory manager (`memory.rs`), cuBLAS dispatch (`blas.rs`)
- **`onnx-rt/src/providers.rs`**: `CudaProvider` gains real implementation behind `cuda` feature
- **`container/src/main.rs`**: GPU device detection and provider selection at startup
- **CI**: New ARM64 container build+test job; GPU test job if self-hosted runner available
- **Container size**: CPU image stays <15 MB; GPU image ~200-500 MB due to CUDA runtime (industry standard for GPU inference containers)
- **Dependencies**: `libcudart`, `libcublas`, `libcudnn` linked dynamically from the NVIDIA base image (no new Rust crate dependencies)
- **No changes to `arch/nvidia`**: All GPU access goes through CUDA runtime APIs from the container, not bare-metal HAL
