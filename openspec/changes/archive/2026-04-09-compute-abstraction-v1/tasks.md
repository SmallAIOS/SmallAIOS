## 1. ComputeProvider Trait Definition

- [ ] 1.1 Create `compute/` crate in workspace: `Cargo.toml` (edition 2021, `#![no_std]`, `alloc`), add to workspace members
- [ ] 1.2 Define `ComputeProvider` trait with associated types (`Buffer`, `Kernel`, `Error`) and methods: `device_info`, `init`, `alloc`, `free`, `copy_host_to_device`, `copy_device_to_host`, `load_kernel`, `launch`, `synchronize`, `supports_op`
- [ ] 1.3 Define `DeviceInfo` struct: `name`, `memory_bytes`, `compute_units`, `backend_type` (enum: Cuda, Rocm, LevelZero, Metal, Cpu)
- [ ] 1.4 Define `GpuBackend` enum with feature-gated variants for each backend plus `Cpu` fallback
- [ ] 1.5 Implement `GpuBackend` dispatch methods that delegate to the inner provider for each trait method
- [ ] 1.6 Add `compute` crate to Layer 0 in `docs/architecture.md` and CLAUDE.md workspace description
- [ ] 1.7 Update Justfile/CI to include `compute` crate in build and test targets

## 2. Retrofit Existing GPU Crates

- [ ] 2.1 Add `compute` as a dependency to `arch/nvidia/Cargo.toml`; implement `ComputeProvider` for `CudaProvider` (delegating to existing `ComputeEngine`, `VramAllocator`, `DmaEngine`)
- [ ] 2.2 Add `compute` as a dependency to `arch/amd/Cargo.toml`; implement `ComputeProvider` for `RocmProvider` (delegating to existing components)
- [ ] 2.3 Add `compute` as a dependency to `arch/intel_gpu/Cargo.toml`; implement `ComputeProvider` for `LevelZeroProvider` (delegating to existing components)
- [ ] 2.4 Verify all three crates compile with trait impls; update any method signatures that don't align with the trait

## 3. Apple Metal Crate — Structure

- [ ] 3.1 Create `arch/apple/` crate: `Cargo.toml` (edition 2021, `std` for container mode), `src/lib.rs` with modules
- [ ] 3.2 Add Metal framework FFI bindings: `MTLDevice`, `MTLBuffer`, `MTLCommandQueue`, `MTLComputePipelineState`, `MTLCommandBuffer`, `MTLComputeCommandEncoder` — minimal raw `objc2` or manual `msg_send!` wrappers
- [ ] 3.3 Implement `MetalProvider` struct: holds device handle, command queue, kernel cache (`BTreeMap<String, PipelineState>`)
- [ ] 3.4 Implement `ComputeProvider` for `MetalProvider`: `init()` creates default MTLDevice and command queue
- [ ] 3.5 Implement `device_info()`: query device name, recommended max working set size, GPU family

## 4. Metal Memory Management

- [ ] 4.1 Implement `alloc()`: create `MTLBuffer` with `storageModeShared` (unified memory on Apple Silicon)
- [ ] 4.2 Implement `free()`: release `MTLBuffer` via Objective-C `release`
- [ ] 4.3 Implement `copy_host_to_device()`: `memcpy` into buffer's `contents()` pointer (shared memory — no DMA needed)
- [ ] 4.4 Implement `copy_device_to_host()`: `memcpy` from buffer's `contents()` pointer
- [ ] 4.5 Unit tests: alloc/free cycle, round-trip host→device→host data integrity

## 5. Metal Compute Pipeline

- [ ] 5.1 Implement `load_kernel()`: compile MSL source string via `newLibraryWithSource:`, extract function, create `MTLComputePipelineState`
- [ ] 5.2 Implement kernel caching: store compiled pipeline states by name, skip recompilation on subsequent loads
- [ ] 5.3 Implement `launch()`: create command buffer, create compute command encoder, set pipeline state, set buffer arguments, dispatch threads with grid/block config, end encoding, commit
- [ ] 5.4 Implement `synchronize()`: `waitUntilCompleted` on command buffer, check for errors
- [ ] 5.5 Unit tests: compile a trivial MSL kernel (vector add), launch, verify output

## 6. Metal Shader Kernels for ONNX Operators

- [ ] 6.1 Write MSL kernel for element-wise ops (add, sub, mul, div, relu, sigmoid, tanh) — single kernel with op-type parameter or separate per-op kernels
- [ ] 6.2 Write MSL kernel for MatMul/Gemm — tiled with SIMD group matrix multiply intrinsics (`simdgroup_matrix`)
- [ ] 6.3 Write MSL kernel for Conv — im2col buffer preparation + MatMul, or direct sliding window
- [ ] 6.4 Write MSL kernel for Softmax — parallel max reduction, subtract-and-exp, sum reduction, normalize
- [ ] 6.5 Implement `supports_op()` returning `true` for operators with MSL kernels, `false` for others
- [ ] 6.6 Unit tests for each MSL kernel: compare GPU output against CPU reference implementation within f32 epsilon

## 7. ONNX Executor GPU Integration

- [ ] 7.1 Add `compute` crate as dependency to `onnx-rt/Cargo.toml` with feature flags: `metal`, `cuda`, `rocm`, `level-zero`
- [ ] 7.2 Extend `Session` to hold an optional `GpuBackend` initialized at session creation
- [ ] 7.3 Modify `executor::dispatch_node()` to check `gpu_backend.supports_op()` first; if supported, dispatch to GPU; otherwise fall back to CPU path
- [ ] 7.4 Implement tensor transfer at CPU↔GPU boundaries: when an operator switches execution device, copy tensors between host and device memory
- [ ] 7.5 Integration test: run a MatMul → Relu graph on Metal backend, verify output matches CPU reference

## 8. Build and CI

- [ ] 8.1 Add `arch/apple` to workspace conditional compilation: `#[cfg(target_os = "macos")]` or feature-gated
- [ ] 8.2 Update CI to handle macOS-only crate (skip on Linux runners, test on macOS runners if available)
- [ ] 8.3 Verify `just test` passes on macOS with `--features metal`; verify it passes on Linux without Metal
- [ ] 8.4 Run `just clippy` and `just fmt-check` across all modified crates
