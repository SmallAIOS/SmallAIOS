## Why

The ONNX runtime needs to dispatch operators to GPU backends (NVIDIA CUDA, AMD ROCm, Intel Level Zero, Apple Metal) but there is no unified abstraction connecting them. Each GPU crate (`arch/nvidia`, `arch/amd`, `arch/intel_gpu`) independently implements compute engines, memory allocators, DMA engines, and kernel registries with the same conceptual interface but no shared trait. Adding Apple Metal support for container-mode macOS inference requires a fourth implementation. Without a unified `ComputeProvider` trait, the ONNX executor would need vendor-specific dispatch paths — making the runtime tightly coupled to every GPU vendor.

## What Changes

- Define a `ComputeProvider` trait in the kernel crate (or a new shared crate) with methods for: device init, memory allocation, kernel launch, synchronization, host↔device transfer
- Create `arch/apple` crate implementing Metal GPU support (container-mode only, macOS)
- Retrofit existing NVIDIA, AMD, and Intel GPU crates to implement the `ComputeProvider` trait
- Add device selection logic to the ONNX executor: enumerate available backends, select the best for each operator, fall back to CPU
- Wire GPU dispatch into the executor from `onnx-cpu-runtime-v1` so operators can route to GPU when available

## Capabilities

### New Capabilities
- `compute-provider`: Unified GPU compute abstraction trait with device lifecycle, memory management, kernel dispatch, and synchronization
- `metal-backend`: Apple Metal GPU backend for container-mode inference on macOS (Apple Silicon and Intel Macs with discrete GPUs)

### Modified Capabilities
- `onnx-runtime`: Add requirements for GPU execution provider selection and CPU fallback behavior

## Impact

- **Code:** New trait definition (likely `kernel/src/compute_provider.rs` or new `compute/` crate), new `arch/apple/` crate, trait impls in `arch/{nvidia,amd,intel_gpu}`, device selection in `onnx-rt/src/executor.rs`
- **Build:** New crate in workspace, new feature flags (`metal`, `cuda`, `rocm`, `level-zero`) on `onnx-rt`
- **Dependencies:** `arch/apple` will need Metal framework bindings — must be `#![no_std]` compatible with raw FFI to Metal API (or `metal-rs` behind `std` gate for container mode)
- **Platform:** Metal backend only compiles on macOS targets; behind `#[cfg(target_os = "macos")]` or feature flag
