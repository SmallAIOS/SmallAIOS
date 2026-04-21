## Context

SmallAIOS runs on Apple Silicon Macs (M1-M4) both in container mode and
during development. The CPU operator surface is complete (136 ops, all
major model families load), but inference speed is CPU-bound. Apple
Silicon GPUs have 5-10x the throughput of their CPU cores for dense
linear algebra (matmul, attention) and element-wise operations. The
Metal backend (`arch/apple/`) already has a working `MetalProvider`
with device init, buffer management, kernel compilation, and dispatch
— plus 11 pre-written MSL shaders. But the ONNX executor never
actually calls them because the GPU dispatch path in `dispatch_node`
is a TODO.

This change wires the existing shaders, adds new high-value shaders,
and implements the tensor-transfer lifecycle so operators can execute
on Metal with automatic CPU fallback for unsupported ops.

## Decisions

### D1: GPU-first dispatch with transparent CPU fallback

**Decision.** When the `gpu` feature is enabled and a `MetalProvider`
is available, `dispatch_node` checks `gpu_backend.supports_op(op_type)`.
If true, the operator executes on GPU: inputs are transferred to device
buffers, the kernel is launched, results are transferred back. If false,
the existing CPU implementation runs unchanged.

**Rationale.** No model should fail to run because of incomplete GPU
coverage. The fallback is transparent — the same `execute_graph` call
works whether 0%, 50%, or 100% of operators have GPU kernels.

### D2: Tensor transfer via a per-graph MetalTensorCache

**Decision.** Add a `MetalTensorCache` struct that:
1. Allocates device `MTLBuffer`s lazily on first use per tensor name
2. Caches them across operator calls within a single `execute_graph`
   invocation (tensors produced by one GPU op and consumed by the next
   stay on-device without round-tripping through host memory)
3. Copies to host only when: (a) the tensor is a graph output, or
   (b) the next consumer is a CPU-only op

This is the "minimal transfer" optimization. Without it, every GPU
op would copy in + copy out, negating the speedup.

**Rationale.** Full lazy-transfer requires tracking the "device-dirty"
bit per tensor — doable but adds complexity. The cache approach is
simpler: if a tensor is in the cache, it's on-device and current.
When a CPU op needs it, the cache copies it to host and evicts it.

### D3: Operator tiering — wire existing shaders first

**Decision.** Implement in two tiers:

**Tier 1 — wire existing shaders (11 ops):** Add, Sub, Mul, Div,
Relu, Sigmoid, Tanh, MatMul (naive), MatMul (tiled), Softmax, Conv2D.
These shaders already exist in `arch/apple/src/shaders.rs` and are
tested. Wiring them requires only the dispatch plumbing + tensor
transfer, no new MSL code.

**Tier 2 — new high-value shaders (~8 ops):**
- `scaled_dot_product_attention` — the SDPA helper, fused QK^T
  scaling + causal mask + softmax + V multiply. Single kernel launch
  instead of 4 separate ops.
- `group_query_attention` — full GQA with fused RoPE, KV concat,
  grouped dispatch. The single highest-value kernel for LLM inference.
- `layer_normalization` — reduction-heavy; GPU wins via parallel
  reduction in shared memory.
- `rms_normalization` — same pattern, no mean subtraction.
- `gemm_tiled_f16` — f16 matmul for models using half precision.
- `gemm_i8_simdgroup` — int8 matmul using `simdgroup_matrix` on M3+,
  with software fallback for M1/M2.
- `rotary_embedding` — standalone fused RoPE for DeepSeek-style exports.
- `batch_normalization` — standard BatchNorm via parallel reduction.

### D4: MSL kernel conventions

**Decision.** All MSL kernels follow these conventions:

```metal
kernel void op_name(
    device const float* input0  [[buffer(0)]],
    device const float* input1  [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant uint& N            [[buffer(3)]],  // element count or shape param
    uint tid [[thread_position_in_grid]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    if (tid >= N) return;  // bounds guard
    // ...
}
```

- Inputs are `device const float*`, outputs are `device float*`
- Shape parameters are `constant uint&` in the last buffer slots
- Every kernel has a `tid >= N` bounds guard
- Thread group size: 256 for element-wise, 16x16 for 2D (matmul tiles)
- Shared memory (`threadgroup float[]`) used only for reductions

**Rationale.** Consistent conventions make it easy to add new kernels
and simplify the dispatch helper that maps `Tensor` → `MTLBuffer`.

### D5: M1/M2 vs M3+ hardware feature detection

**Decision.** The `MetalProvider` detects GPU family at init time via
`MTLDevice::supportsFamily(.apple9)` (Apple9 = M3+). Kernels that
use `simdgroup_matrix` (hardware 8x8 matmul, M3+ only) have a
software fallback path selected at kernel-compile time via a
preprocessor `#define HAS_SIMDGROUP_MATRIX`.

**Rationale.** M1/M2 are the majority of developer machines today.
Not supporting them would make Metal inference unusable for most
contributors. The software fallback uses shared-memory tiling which
is ~2x slower than `simdgroup_matrix` but still 3-5x faster than CPU.

### D6: Session-level GPU opt-in

**Decision.** GPU inference is opt-in via `Session` configuration:

```rust
let session = Session::builder()
    .with_gpu(GpuConfig::metal())  // enable Metal
    .build_from_bytes(&model_bytes)?;
```

If `.with_gpu()` is not called, the session runs pure CPU as before.
The `GpuConfig::metal()` constructor initializes a `MetalProvider`
and passes it to `execute_graph`. If Metal is unavailable (Linux,
no GPU), it falls back to CPU silently.

**Rationale.** Inference results may differ slightly between CPU and
GPU (f32 rounding) so users should explicitly opt in. The builder
pattern matches the existing `Session` API.

### D7: Correctness testing strategy

**Decision.** Every GPU kernel is tested via a "CPU reference"
pattern:

1. Run the operator on CPU (the existing implementation)
2. Run the same operator on GPU
3. Assert the results match within tolerance (±1e-5 relative for
   f32, ±1 quantized step for int8)

These tests run in the `arch/apple` crate's test module, gated by
`#[cfg(target_os = "macos")]`. They are NOT run in CI (no macOS
runner), but `just test-metal` invokes them locally.

### D8: File layout

```
arch/apple/src/
├── lib.rs
├── metal_provider.rs   (extended: MetalTensorCache, GPU family detect)
├── shaders.rs          (extended: new MSL kernels)
├── dispatch.rs         (NEW: maps OpKind → kernel launch config)
└── stub.rs             (unchanged: non-macOS stubs)

onnx-rt/src/
├── executor.rs         (modified: GPU dispatch path in dispatch_node)
└── session.rs          (modified: GpuConfig builder)

compute/src/
└── lib.rs              (modified: ComputeProvider tensor transfer traits)
```

## Alternatives considered

### A1: Write a Metal Performance Shaders (MPS) backend

Use Apple's pre-built MPS kernels (MPSMatrixMultiplication, etc.)
instead of hand-written MSL. **Rejected:** MPS requires Objective-C
bridging and the `objc2` crate ecosystem, adding ~5 dependencies.
Hand-written MSL gives us control over fusion (SDPA, GQA) that MPS
doesn't support, and keeps the dependency count at zero.

### A2: Use WebGPU/wgpu as the abstraction layer

Target wgpu/WGSL instead of Metal directly, gaining portability to
Vulkan/DX12. **Rejected:** wgpu is `std`-only and adds ~30
dependencies. SmallAIOS is `no_std` first. Metal-native is the
right choice for the Apple Silicon target; Vulkan can be added as
a separate `arch/amd` or `arch/nvidia` backend later following the
same pattern.

### A3: Implement GPU dispatch inside the ops/ modules

Have each `op_matmul` check for GPU availability internally.
**Rejected:** scatters GPU logic across 136+ op files. Centralizing
GPU dispatch in `executor.rs` + `arch/apple/src/dispatch.rs` keeps
the separation clean and makes it easy to add new GPU backends.

## Open questions

**Q1:** Should the `MetalTensorCache` persist across multiple
`Session::run()` calls for the same session? Currently scoped to
one `execute_graph` invocation. Persisting would help autoregressive
generation (KV cache stays on device between tokens). Recommend yes
but defer to implementation.

**Q2:** Should we add f16 storage on-device and convert only at
the host boundary? Apple GPUs are ~2x faster at f16 than f32.
Recommend yes for Tier 2 but not blocking Tier 1.
