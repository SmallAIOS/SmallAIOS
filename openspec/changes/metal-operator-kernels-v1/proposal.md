## Why

SmallAIOS now loads every major LLM family end-to-end (BERT, ViT, GPT-2, Llama-3.2, DeepSeek-R1, Gemma 3, MobileNetV2) — but all 136 operators execute on CPU. The Apple Metal GPU backend in `arch/apple/` has a working `MetalProvider` (device init, buffer alloc, kernel compile, dispatch — 9 hardware-verified tests) and 11 pre-written MSL shaders (`shaders.rs`), but the executor's `dispatch_node` **always falls through to the CPU path** even when `gpu_backend.supports_op()` returns true. Wiring the existing shaders through the dispatch path and adding kernels for the highest-leverage transformer ops (SDPA, GroupQueryAttention, RoPE) is the single most impactful performance improvement available — MatMul and attention together account for ~80% of LLM inference time, and Apple Silicon's GPU is 5-10x faster than its CPU cores for these workloads.

## What Changes

- **Wire the existing 11 MSL shaders** through the GPU dispatch path in `executor.rs` so operators actually execute on Metal when the `gpu` feature is enabled and a `MetalProvider` is available. This is the "unblock" step — the shaders exist, they just need to be called.
- **Add new MSL shaders** for the high-value transformer ops not yet covered:
  - `scaled_dot_product_attention` (the SDPA helper from `ops/microsoft.rs`)
  - `group_query_attention` (the full GQA kernel with fused RoPE + causal mask)
  - `layer_normalization` / `rms_normalization` (reduction-heavy, big GPU win)
  - `gemm_i8` (int8 matmul on GPU using Metal's `simdgroup_matrix` on M3+)
- **Implement host↔device tensor transfer** in the dispatch path: copy input tensors to Metal buffers before launch, copy results back after synchronization.
- **Add a `MetalTensorCache`** that reuses device buffers across operator calls within a single `execute_graph` invocation, avoiding per-op alloc/free overhead.
- **Update `supports_op`** to reflect only the ops that have real, tested Metal kernels (currently it claims 11 but none are wired).
- **Feature-gate** all Metal dispatch behind `#[cfg(feature = "gpu")]` + a runtime check for Metal device availability.

## Capabilities

### New Capabilities
- `metal-gpu-inference`: Metal GPU execution path for ONNX operators on Apple Silicon, including tensor transfer, kernel dispatch, buffer lifecycle, and operator coverage for the hot-path transformer ops.

### Modified Capabilities
- `onnx-cpu-execution`: The executor's dispatch path gains a GPU-first-then-CPU-fallback pattern. When the `gpu` feature is enabled and a `MetalProvider` is available, supported operators dispatch to Metal; unsupported operators fall through to the existing CPU implementations transparently. No CPU behavior changes.

## Impact

**Affected code:**
- `onnx-rt/src/executor.rs` — GPU dispatch wiring in `dispatch_node` (currently a TODO)
- `arch/apple/src/shaders.rs` — new MSL kernels for SDPA, GQA, LayerNorm, RMSNorm, int8 GEMM
- `arch/apple/src/metal_provider.rs` — `MetalTensorCache`, extended `supports_op`, tensor transfer helpers
- `compute/src/lib.rs` — `ComputeProvider` trait may need minor extensions for tensor transfer
- `onnx-rt/src/session.rs` — session creation optionally initializes a `MetalProvider`

**Affected features:**
- `onnx-rt` `gpu` feature flag (already exists, currently unused in practice)
- `arch/apple` default feature (currently empty; may add `metal-inference`)

**Dependencies:**
- `metal = "0.33"` (already in `arch/apple/Cargo.toml`, macOS-only)
- No new external dependencies

**Risks:**
- Metal shader correctness must match CPU reference within tolerance (±1e-5 relative for f32, ±1 quantized step for int8). Each kernel needs a CPU-vs-GPU comparison test.
- Memory pressure: copying tensors to/from device doubles peak memory during transfer. The `MetalTensorCache` mitigates this but doesn't eliminate it for the first/last transfer.
- Apple Silicon generational differences: `simdgroup_matrix` (hardware matmul) is M3+ only. Kernels must have a fallback path for M1/M2.
