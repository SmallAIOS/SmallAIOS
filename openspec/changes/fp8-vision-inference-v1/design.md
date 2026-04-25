## Context

We have FP8 GEMM working at the kernel level (`gpu_gemm_fp8` from
`arm64-gpu-container-v1` passes correctness on simple shapes). The
ResNet-50 hybrid path runs all 53 Convs through cuDNN at TF32
precision today, delivering 33 ms inference. Blackwell tensor cores
support FP8 (E4M3 and E5M2) at roughly 2× the throughput of TF32
for the same compute volume — meaning ResNet-50's Conv-dominated
graph could plausibly reach the high teens of ms with FP8 dispatch.

The cuDNN 9 path for FP8 Conv uses the **backend API** rather than
the legacy frontend functions (`cudnnConvolutionForward` etc.). The
backend API constructs a graph-of-descriptors representation and
executes it via `cudnnBackendExecute`. This is a bigger surface than
our existing FFI but is necessary for FP8.

For weight conversion, FP8 E4M3 is "1 sign + 4 exponent + 3 mantissa"
with bias 7, max representable 448. E5M2 is "1+5+2", bias 15, max
57344. Per-tensor max-abs scaling is the standard naïve approach:

```rust
let s = max(abs(weights)) / fp8_max;
weights_fp8 = clamp(round(weights / s), fp8_min, fp8_max);
// At dequant: weights_f32 = weights_fp8 * s
```

For ResNet-50 Conv weights this is typically accurate to ~1% on
single channels. Per-channel scaling (one scale per output channel)
would be more accurate but is a follow-up.

## Goals / Non-Goals

**Goals:**

- ResNet-50 v2 hybrid + FP8 E4M3 mode hits **≥1.5× speedup** over
  the TF32 hybrid baseline (~33 ms → ~22 ms or better).
- `max_abs_diff` between FP8 and TF32 outputs stays under `5e-2`
  for E4M3, `1e-1` for E5M2 (realistic FP8 quantization tolerance
  on a deep CNN).
- Top-1 / top-5 ImageNet accuracy on a real validation set drops
  by ≤1% absolute when switching from TF32 to FP8 E4M3 (proven
  via a separate accuracy script run, not in the bench harness).
- Default `compute_precision = Tf32` keeps existing behavior.

**Non-Goals:**

- Per-channel weight quantization. Per-tensor max-abs is good
  enough for v1; per-channel is a follow-up if accuracy regresses
  too much.
- Calibration with a validation dataset. The user is expected to
  hand the runtime f32 weights and let `compute_precision` quantize
  on the fly; sophisticated calibration belongs in a tooling layer.
- Loading already-FP8 ONNX models. That's `arm64-gpu-container-v1`'s
  task 12.6.
- NVFP4 / MXFP4 block-scaled matmul. Different (much more complex)
  scaling regime; separate change.
- FP8 for BatchNorm, Pool, Relu, Add. These ops are cheap and the
  conversion overhead would dominate. Activations remain f32 between
  Convs; the FP8 kernel itself takes f32 input and quantizes
  internally (cuDNN's FP8 Conv API supports this).

## Decisions

### 1. Use cuDNN backend API for FP8 Conv

**Decision:** Add FFI bindings for:
- `cudnnBackendCreateDescriptor`, `cudnnBackendDestroyDescriptor`
- `cudnnBackendSetAttribute`, `cudnnBackendGetAttribute`
- `cudnnBackendFinalize`, `cudnnBackendExecute`

Plus the relevant attribute enums and descriptor types
(`CUDNN_BACKEND_TENSOR_DESCRIPTOR`,
`CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR`,
`CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR`,
`CUDNN_BACKEND_EXECUTION_PLAN_DESCRIPTOR`,
`CUDNN_BACKEND_VARIANT_PACK_DESCRIPTOR`).

Wrap with RAII in a new `onnx-rt/src/cuda/backend.rs`.

**Rationale:** This is the supported path for FP8 Conv in cuDNN 9.
The legacy `cudnnConvolutionForward` doesn't accept FP8 tensor
types.

**Alternative considered:** use cuBLASLt for matmul-as-Conv
(im2col + GEMM). Rejected — adds an im2col pass and loses the
direct-conv optimizations cuDNN does. The backend API is the
documented path.

### 2. Runtime per-tensor weight quantization at session init

**Decision:** When `SessionConfig::compute_precision` is
`Fp8E4M3` or `Fp8E5M2`, and a `Session::run` is called for the
first time, the runtime traverses the model's initializers and
quantizes every f32 Conv/Gemm/MatMul weight tensor to FP8 once.
The quantized tensors land in the existing
`device_initializer_cache` from `gpu-resident-vision-hybrid-v1`,
so all subsequent inferences use the cached FP8 weights.

For each weight tensor:

```rust
let abs_max = weights.iter().map(|w| w.abs()).max();
let scale = abs_max / FP8_MAX;
let fp8_bytes: Vec<u8> = weights.iter()
    .map(|w| f32_to_fp8(w / scale, mode))
    .collect();
// Store both the FP8 tensor and the scale per init name
```

Scales live in a parallel `BTreeMap<String, f32>` on the cache.

**Rationale:** One-shot conversion at session init is cheaper than
on-the-fly. Scale persistence lets the cuDNN kernel apply scale
correctly during convolution.

### 3. Activation handling: leave f32 between ops, quantize inside the kernel

**Decision:** Activations between ops remain f32 in our value map.
When a Conv runs in FP8 mode, the cuDNN backend descriptor is
configured with FP8 input + FP8 weight + f32 output (with internal
on-the-fly quantization of the input). Output stays f32 so the next
op (BN / Relu / etc.) uses its existing f32 path.

**Rationale:** Avoids inserting explicit Quantize/Dequantize ops
into the hybrid executor. The cuDNN kernel handles the
input-side quantization; output dequant is implicit in the kernel
configuration.

**Trade-off:** loses some quantization-aware accuracy benefits
(e.g. cumulative error builds up across layers because each layer
re-quantizes input from f32). Acceptable for v1; QAT (quantization-
aware training) deserves a separate sweep.

### 4. New `compute_precision` field, not overload existing config

**Decision:**

```rust
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub enum ComputePrecision {
    /// Default — TF32 on Blackwell, FP32 elsewhere.
    #[default]
    Tf32,
    /// FP16 with FP32 accumulate (existing path; passes via
    /// `GpuPrecision::F16` if already wired).
    Fp16,
    /// BF16 with FP32 accumulate.
    Bf16,
    /// FP8 E4M3 with FP32 accumulate.
    Fp8E4M3,
    /// FP8 E5M2 with FP32 accumulate.
    Fp8E5M2,
}

pub struct SessionConfig {
    // ...
    pub compute_precision: ComputePrecision,
}
```

This unifies / supersedes the existing `GpuPrecision` enum used by
`CudaRuntime::init_with_precision` (today: `F32` / `F16` / `Tf32`).
We add an internal mapping in `Session::run` from
`compute_precision` to the runtime's existing precision knob, plus
the new FP8 dispatch routing.

**Rationale:** A single user-facing precision knob is easier to
reason about than splitting "GPU precision" and "FP8 mode" across
two flags.

### 5. FP8 dispatch routing in the hybrid executor

**Decision:** In `executor_hybrid.rs::try_gpu_dispatch`, the
"Conv" branch checks `Session.config.compute_precision`. If
`Fp8E4M3` / `Fp8E5M2`, route to `gpu_conv2d_device_fp8` instead of
`gpu_conv2d_device`. Same for "Gemm" / "MatMul" → `gpu_gemm_device_fp8`.

The fp8 dispatch fns get the cached FP8 weight tensor + scale
from the device-initializer cache.

If the FP8 dispatch fails (e.g. cuDNN backend rejects the shape),
fall back to the TF32 dispatch path within the same op call —
graceful degradation, log a warning the first time it happens
per Session.

### 6. Quantization helper module

**Decision:** New `onnx-rt/src/quantize.rs` module with:

```rust
pub fn fp32_to_fp8_e4m3(x: f32) -> u8 { ... }
pub fn fp32_to_fp8_e5m2(x: f32) -> u8 { ... }
pub fn quantize_tensor_per_tensor_e4m3(t: &Tensor) -> (Tensor, f32);
pub fn quantize_tensor_per_tensor_e5m2(t: &Tensor) -> (Tensor, f32);
```

Implementations follow the OCP FP8 (E4M3FN) spec exactly — the same
encoding used by NVIDIA, Intel, AMD, and ONNX's `FLOAT8E4M3FN`
type. Reference: OCP "Microscaling Formats Specification" v1.0.

Bit packing: produces a `Tensor` with `data_type =
DataType::Float8E4M3` (already in the enum from
`arm64-gpu-container-v1`).

### 7. Test strategy

Three layers:

1. **Quantization unit tests** in `quantize.rs::tests`: round-trip
   error bounds (E4M3 max ~5%, E5M2 max ~12%), scale-zero handling,
   subnormals (E4M3 has them, E5M2 doesn't), nan/inf rejection.
2. **Per-op FP8 correctness tests** in `test_cuda.rs`: ResNet-style
   `[1, 64, 56, 56]` Conv, FP8 vs TF32, `max_abs_diff < 5e-2` for
   E4M3, `< 1e-1` for E5M2.
3. **End-to-end ResNet-50 FP8 bench**:
   `bench_resnet50_cpu_vs_gpu_hybrid_fp8e4m3`. Speedup target ≥1.5×
   over TF32 hybrid; output diff `< 5e-2` vs CPU.

## Risks / Trade-offs

- [**Risk**: cuDNN backend API is verbose and easy to misconfigure
  (descriptor attribute enums, finalize step, variant packs)] →
  Mitigation: copy the structure from a known-good cuDNN sample
  (the `cuDNN_FrontEnd` repo on GitHub has reference patterns).
  Wrap with RAII to make leaks impossible.
- [**Risk**: per-tensor max-abs scaling produces large accuracy
  loss on certain Conv layers (those with very wide weight ranges)]
  → Mitigation: log + measure on ResNet-50 first; if accuracy drops
  too much, switch a follow-up to per-output-channel scaling.
- [**Risk**: FP8 Conv on small spatial dims (1×1 Convs in
  bottleneck blocks) is launch-overhead-bound, so the FP8 speedup
  doesn't materialize] → Mitigation: should be addressed by
  `cuda-graphs-v1` capturing the launches once. Cross-reference
  in the design.
- [**Risk**: cuDNN's FP8 backend doesn't support some Conv shape
  (e.g. specific stride/dilation combinations)] → Mitigation: the
  fall-back to TF32 dispatch within the same op (decision 5)
  handles this gracefully.
- [**Trade-off**: f32 activations between ops (decision 3) means
  cumulative quantization error builds up] → Acceptable for v1.
  Per-layer fully-quantized graphs require QAT (training-time
  awareness), which is out of scope.
- [**Trade-off**: ComputePrecision enum supersedes GpuPrecision] →
  Internal API churn; mitigated by keeping `GpuPrecision` for
  backward compatibility and just mapping at the Session layer.

## Migration Plan

Purely additive. Default `compute_precision = Tf32` keeps every
existing caller's behavior byte-for-byte. Users opt into FP8 by
setting `SessionConfig::compute_precision = ComputePrecision::Fp8E4M3`.

Internal: the existing `GpuPrecision` enum remains; `compute_precision`
maps to it for the non-FP8 modes, and gates the new FP8 path for the
FP8 modes.

Rollback: revert; the runtime falls back to TF32.

## Open Questions

- Should we expose a `Session::quantization_stats() ->
  QuantizationStats` introspection method (per-layer scale, max
  observed error)? Useful for debugging accuracy regressions.
  Defer; can add when first user complains about FP8 accuracy.
- Per-channel scaling: how much code growth? Probably another
  N values per Conv weight + cuDNN backend attribute config.
  Defer to v2 unless v1 accuracy is unacceptable.
- Should `compute_precision` apply to the CPU path too? Currently
  CPU is f32-only. FP8 on CPU is a different beast (no tensor
  cores; would just be a software emulation). Skip — keep
  `compute_precision` GPU-only for now.
