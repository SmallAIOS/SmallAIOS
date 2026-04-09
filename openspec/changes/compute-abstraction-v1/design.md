## Context

Four GPU backends exist or are planned: NVIDIA (`arch/nvidia`), AMD (`arch/amd`), Intel (`arch/intel_gpu`), and Apple Metal (new). Each has the same conceptual components:

| Component | NVIDIA | AMD | Intel | Apple (new) |
|-----------|--------|-----|-------|-------------|
| Provider | `CudaProvider` | `RocmProvider` | `LevelZeroProvider` | `MetalProvider` |
| Compute | `ComputeEngine` | `ComputeEngine` | `ComputeEngine` | `MetalComputeEngine` |
| Memory | `VramAllocator` | `VramAllocator` | `VramAllocator` | `MetalBufferAllocator` |
| DMA | `DmaEngine` | `SdmaEngine` | `DmaEngine` | (unified memory) |
| Kernels | `PtxRegistry` | `HipRegistry` | `SpirvRegistry` | `MetalShaderRegistry` |

The three existing crates are stubs — architecturally complete but no real hardware interaction. The Metal backend is the first one that can actually be tested on the developer's macOS hardware.

The ONNX runtime executor (from `onnx-cpu-runtime-v1`) has a CPU dispatch path. This change adds a GPU dispatch layer: check if a GPU backend is available, check if the operator has a GPU kernel, dispatch to GPU or fall back to CPU.

## Goals / Non-Goals

**Goals:**
- Define a `ComputeProvider` trait that captures the common interface across all GPU vendors
- Implement a real, testable Metal backend for macOS container mode
- Retrofit existing GPU crates to implement the trait (they stay as stubs)
- Add GPU-aware operator dispatch to the ONNX executor with CPU fallback
- Support `#![no_std]` for the trait definition; Metal impl uses `std` (container-only)

**Non-Goals:**
- Real CUDA/ROCm/Level Zero hardware interaction (stubs only)
- Multi-GPU support (single device per session)
- Async kernel launch / compute-transfer overlap (synchronous initially)
- Automatic operator partitioning between CPU and GPU (explicit provider selection)
- Performance parity with vendor-native runtimes (correctness first)

## Decisions

### D1: Trait Location — New `compute` Crate at Layer 0

Create a new `compute/` crate at Layer 0 (Foundation) alongside `kernel` and `security`. This crate defines the `ComputeProvider` trait and associated types (`DeviceInfo`, `Buffer`, `KernelHandle`). GPU arch crates at Layer 2 implement it.

**Why not in `kernel`:** The kernel crate is `#![no_std]` bare-metal. The trait must be usable from both kernel mode and container mode. A separate crate keeps concerns clean.

**Why not in `onnx-rt`:** The compute abstraction is not ONNX-specific. IPC, networking, or future workloads might use GPU compute.

```
Layer 2 — HAL:       arch/{nvidia,amd,intel_gpu,apple} impl ComputeProvider
Layer 1 — Services:  onnx-rt uses ComputeProvider
Layer 0 — Foundation: compute (trait definition)
```

### D2: ComputeProvider Trait Design

```rust
pub trait ComputeProvider {
    type Buffer;
    type Kernel;
    type Error;

    // Lifecycle
    fn device_info(&self) -> DeviceInfo;
    fn init(&mut self) -> Result<(), Self::Error>;

    // Memory
    fn alloc(&mut self, size: usize) -> Result<Self::Buffer, Self::Error>;
    fn free(&mut self, buf: Self::Buffer) -> Result<(), Self::Error>;
    fn copy_host_to_device(&mut self, src: &[u8], dst: &Self::Buffer) -> Result<(), Self::Error>;
    fn copy_device_to_host(&self, src: &Self::Buffer, dst: &mut [u8]) -> Result<(), Self::Error>;

    // Dispatch
    fn load_kernel(&mut self, name: &str, source: &[u8]) -> Result<Self::Kernel, Self::Error>;
    fn launch(&mut self, kernel: &Self::Kernel, grid: [u32; 3], block: [u32; 3], args: &[&Self::Buffer]) -> Result<(), Self::Error>;
    fn synchronize(&mut self) -> Result<(), Self::Error>;

    // Operator-level (optional convenience)
    fn supports_op(&self, op: &str) -> bool;
}
```

**Why associated types over trait objects:** Each backend has different buffer and kernel handle types. Associated types give zero-cost abstraction. The ONNX executor can be generic over `P: ComputeProvider` or use an enum-dispatch pattern for runtime selection.

### D3: Metal Backend — Container-Only with `objc2` FFI

The Metal backend will use raw Objective-C FFI via `objc2` (or manual `msg_send!`) to access the Metal framework. This runs only in container mode on macOS — gated behind `#[cfg(target_os = "macos")]` and `feature = "metal"`.

**Key Metal concepts mapped:**
- `MTLDevice` → `MetalProvider.device`
- `MTLBuffer` → `ComputeProvider::Buffer`
- `MTLComputePipelineState` → `ComputeProvider::Kernel`
- `MTLCommandBuffer` + `MTLComputeCommandEncoder` → `launch()` + `synchronize()`
- Metal Shading Language (MSL) kernels compiled at init time

**Why Metal over MPS (Metal Performance Shaders):** MPS provides high-level ops (matmul, conv) but as opaque Obj-C objects. Using raw compute shaders gives full control over kernels and matches the pattern of PTX/HIP/SPIR-V in other backends. MPS can be added later as an optimization for specific ops.

**Why not `metal-rs` crate:** It's a good wrapper but pulls in `std` and has broader API surface than needed. We need only: device creation, buffer allocation, compute pipeline creation, command buffer submission. Raw FFI keeps the dependency surface minimal.

### D4: Enum Dispatch for Runtime Backend Selection

Rather than dynamic dispatch (`dyn ComputeProvider`), use an enum:

```rust
pub enum GpuBackend {
    #[cfg(feature = "cuda")]
    Cuda(CudaProvider),
    #[cfg(feature = "metal")]
    Metal(MetalProvider),
    #[cfg(feature = "rocm")]
    Rocm(RocmProvider),
    #[cfg(feature = "level-zero")]
    LevelZero(LevelZeroProvider),
    Cpu, // fallback — no GPU
}
```

This avoids trait objects in `no_std`, compiles away unused backends via feature flags, and allows the executor to match once at session creation.

### D5: Metal Kernel Strategy — MSL for Core Ops

Implement Metal Shading Language kernels for the most compute-intensive operators first:
1. MatMul/Gemm (SIMD group matrix multiply on Apple Silicon)
2. Conv (im2col + MatMul or direct convolution)
3. Softmax (parallel reduction)
4. Element-wise ops (Add, Mul, Relu — trivially parallel)

Remaining operators fall back to CPU. This covers ~80% of inference compute for typical models.

## Risks / Trade-offs

**[Risk] Metal FFI complexity in no_std context** — Metal requires Objective-C runtime. Mitigation: The Metal backend is container-only (has `std` and libc). The trait definition in `compute/` stays `no_std`; only the Metal impl requires `std`.

**[Risk] Existing GPU crate refactor scope** — Retrofitting 3 crates to implement the trait is significant surface area. Mitigation: Since they're all stubs, the impl is mechanical: map existing methods to trait methods, adjust signatures. No behavioral changes.

**[Risk] Apple Silicon vs. Intel Mac differences** — Apple GPU (M1/M2/M3) and Intel Macs with AMD discrete GPUs have different Metal capabilities. Mitigation: Target Apple Silicon first (unified memory simplifies buffer management). Intel Mac support is stretch goal.

**[Trade-off] Synchronous dispatch** — Initial Metal integration blocks on `waitUntilCompleted`. Real performance requires async command buffers. Acceptable for correctness-first; async is a future optimization.

## Open Questions

- **Q1:** Should the `compute` crate live at Layer 0 or be a sub-module of `kernel`? Separate crate gives cleaner layering but adds workspace complexity.
- **Q2:** For Metal unified memory, should `copy_host_to_device` be a no-op (shared buffer) or should the trait have a `supports_unified_memory()` hint?
- **Q3:** License considerations for Metal SDK headers and `objc2` FFI — need to verify Apple's framework usage terms for this context.
