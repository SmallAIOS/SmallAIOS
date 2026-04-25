## Context

Single-request latency on the hybrid executor is excellent (33 ms
ResNet-50 v2 on DGX Spark). Throughput per second at batch=1 is
~30 inferences/s. Blackwell tensor cores can in principle deliver
hundreds of TFLOPS — they're underutilized when each cuDNN kernel
launches with only ~600 KB of activation work. At batch=64,
activations grow to ~38 MB per Conv layer, the GPU saturates, and
per-image cost drops near-linearly until kernel/memory limits kick
in. Industry-standard serving frameworks expect ~30 inferences/sec at
batch=1 vs ~500–1500 inferences/sec at batch=64 on equivalent
hardware.

The runtime is already capable of arbitrary batch sizes — every
existing CUDA op (`gpu_conv2d_device`, `gpu_batchnorm`,
`gpu_gemm_device`, etc.) takes the leading dim from the input shape
and runs whatever batch is supplied. The CPU operators likewise
respect their input rank. Nothing in the kernels is hard-coded to
N=1.

What's missing:

1. The session API (`Session::run`) accepts a single `InferenceInput`
   per name. To run 64 images, callers would have to make 64 calls
   today.
2. There is no host-side "stack N inputs along axis 0" helper to
   build a batched tensor from a list of N batch-1 tensors.
3. There is no batched-output unstacking helper either.
4. `SessionConfig` has a `max_batch_size: usize` field already (set
   to 1 by default) but it's currently unused — meant as a
   placeholder for this work.

The hybrid executor's per-Session graph cache (from
`gpu-resident-vision-hybrid-v1`) is already shape-keyed, so it'll
naturally cache one graph per batch size used. Combined with
`cuda-graphs-v1` (capture/replay) the batched throughput benefit
compounds — a single `cudaGraphLaunch` on a B=64 graph runs 64
images.

## Goals / Non-Goals

**Goals:**

- Add `Session::run_batched(&[InferenceInput])` taking N inputs of
  identical shape (apart from the batch dim) and returning N outputs.
- Internally compose N inputs into a single batched tensor with
  leading dim = N, dispatch through the existing executor exactly
  once, decompose the batched output back into N tensors.
- Hit ≥10× throughput improvement on ResNet-50 v2 at B=64 vs B=1
  on DGX Spark. (Industry expectation; we'll calibrate the exact
  bound during implementation.)
- Single-request `run` latency unchanged (it's a degenerate B=1
  call through the same code path).
- Clear error reporting on shape mismatch, dtype mismatch, or
  exceeding `BatchPolicy` limits.

**Non-Goals:**

- Request-level scheduling across HTTP/RPC boundaries — that's the
  container/server layer's concern (`container` crate, separate
  change).
- Dynamic per-request padding to common batch buckets — initial
  scope offers fixed `Static(N)` and capped `Dynamic { max, pad
  }`. Smart bucketing is a follow-up.
- KV-cache batching for LLM workloads. The Gemma path uses its own
  device executor (`execute_graph_gpu`); LLM batching has its own
  set of concerns (variable seq lengths, KV cache fragmentation).
- Concurrent multi-Session inference. Each Session is single-
  threaded; dispatching multiple Sessions in parallel from
  different threads is allowed but not coordinated.
- Pipelined inference (overlap prefill/decode). Separate work.

## Decisions

### 1. New API: `Session::run_batched`, not overload `run`

**Decision:** Introduce a new method:

```rust
pub fn run_batched(
    &self,
    inputs: &[InferenceInput],   // N inputs total, grouped by name
    batch_size: usize,           // explicit; must agree with len(inputs) per name
) -> Result<Vec<InferenceOutput>, SessionError>;
```

Keep `Session::run` as the single-input entry point. Internally
`run` is a one-line shim that calls `run_batched` with N=1.

**Rationale:** Overloading `run` to detect "is this a list of B=1
tensors or a single B=N tensor?" creates ambiguity for users. A
distinct method makes the contract explicit.

**Alternatives considered:**

- Single API that accepts arbitrary leading-dim tensors. **Rejected**
  because it forces callers to do the stacking themselves;
  `run_batched` makes the runtime do it once correctly.
- Iterator-based API. **Rejected** as over-engineered for v1; can
  be added later.

### 2. Stacking happens at the executor boundary, not in the kernels

**Decision:** Add a `stack_along_batch_axis(&[Tensor]) -> Tensor`
helper in a new `onnx-rt/src/batch.rs` module. The helper validates
ranks/shapes/dtypes and produces a single tensor with leading dim N.
Symmetric `unstack_along_batch_axis(&Tensor, n: usize) -> Vec<Tensor>`
splits an output back into N tensors.

`run_batched` stacks → calls existing `execute_graph` /
`execute_graph_hybrid` once → unstacks. Kernels see exactly the
shapes they already handle.

**Rationale:** Zero kernel changes. The N-image work happens in
exactly one stacking pass on the host (one `Vec<u8>::with_capacity`
+ `copy_from_slice` per input), already very fast.

**Memory cost:** 2× the activation memory during a single inference
(stacked input + unstacked output). For typical vision sizes
(~600 KB × 64 = 38 MB), negligible on DGX Spark with 128 GB.

### 3. `BatchPolicy` enum on `SessionConfig`

**Decision:**

```rust
pub enum BatchPolicy {
    /// Reject `run_batched`. Default.
    Disabled,
    /// Always require exactly N inputs.
    Static(usize),
    /// Accept 1..=max; if `pad`, the runtime repeats the last
    /// input to reach `max` and discards the resulting padded
    /// outputs.
    Dynamic { max: usize, pad: bool },
}

impl Default for BatchPolicy {
    fn default() -> Self { BatchPolicy::Disabled }
}
```

`SessionConfig` gains `pub batch_policy: BatchPolicy`. The existing
`max_batch_size: usize` field is repurposed — when `BatchPolicy` is
`Disabled` and `max_batch_size > 1`, treat it as
`Dynamic { max: max_batch_size, pad: false }` for backward
compatibility. Document the migration path.

**Rationale:** Three policy variants cover the realistic deployment
shapes: fixed-batch (training-style), variable-batch-with-padding
(production serving with bucketing), and disabled (current behavior).

### 4. Validation rules

**Decision:** `run_batched` enforces:

1. All inputs of the same name MUST have identical shape and dtype.
2. The number of inputs per name MUST be the same (no
   ragged batches).
3. If `BatchPolicy::Static(N)`, the count MUST equal N.
4. If `BatchPolicy::Dynamic { max, pad }`, the count MUST be in
   `1..=max`. With `pad: true`, fewer than `max` triggers padding;
   with `pad: false`, the runtime runs the actual batch size as-is
   (which may incur a graph rebuild if cuda-graphs-v1 is also on).

Returns:
- `SessionError::BatchPolicyViolation` for policy mismatches
- `SessionError::BatchShapeMismatch` for input-shape mismatches
- `SessionError::BatchEmpty` for zero-input call

### 5. Output unstacking preserves the original output names

**Decision:** Each name in the output set produces N output tensors,
returned in input order. The returned `Vec<InferenceOutput>` has
length `N * num_output_names` (e.g., for a single-output classifier
with N=64, returns 64 `InferenceOutput`s, one per image).

**Rationale:** Matches what users expect — "I asked for 64
inferences, give me 64 outputs."

**Alternative considered:** return `Vec<Vec<InferenceOutput>>`
(grouped by image). Rejected — flat list is simpler, names already
disambiguate.

### 6. Padding semantics

**Decision:** When `BatchPolicy::Dynamic { pad: true }` and the
caller supplies `K < max` inputs, the runtime repeats the last input
to reach `max`. After running, only the first `K` outputs are
returned to the caller. Padded outputs are computed and discarded.

**Rationale:** Padding lets serving systems hit a fixed-shape graph
cache hit even when natural request count varies. Cost is at most
1 additional inference's worth of compute, which is fine for
ResNet-50 (~33 ms / 64 ≈ 0.5 ms per image; padding costs ~1 image
of unused work).

**Alternative:** zero-pad inputs. Rejected — produces meaningless
outputs that callers might inadvertently use.

### 7. Throughput benchmarks

**Decision:** Add 4 throughput benches in `bench_vision_models.rs`:

```rust
#[test] #[ignore] fn bench_resnet50_throughput_b1();
#[test] #[ignore] fn bench_resnet50_throughput_b4();
#[test] #[ignore] fn bench_resnet50_throughput_b16();
#[test] #[ignore] fn bench_resnet50_throughput_b64();
```

Each runs 100 inferences at the given batch size and reports
images/sec. Targets:

- B=1: same as `bench_resnet50_cpu_vs_gpu_hybrid` (~30 img/s
  baseline)
- B=4: ≥3.5× B=1 throughput
- B=16: ≥10× B=1 throughput
- B=64: ≥20× B=1 throughput (or saturates DGX Spark's compute
  ceiling earlier)

Single-request latency (`run`) MUST NOT regress.

### 8. Test strategy

Three layers:

1. **Stack/unstack unit tests** in `onnx-rt/src/batch.rs`'s test
   module: round-trip identity, shape-mismatch errors, dtype
   handling (f32, bf16, int64 for shape tensors).
2. **`run_batched` integration tests** in
   `onnx-rt/tests/test_real_model.rs` (or a new file): N=1, N=4
   on the synthetic MLP, comparing stacked-batch output to N
   single-call outputs within `1e-4`.
3. **Throughput benchmarks** in `bench_vision_models.rs` (Section 7).

## Risks / Trade-offs

- [**Risk**: graph cache from `cuda-graphs-v1` thrashes when batch
  sizes vary widely] → Mitigation: the disable-on-thrashing logic
  from cuda-graphs-v1 already handles this. With
  `BatchPolicy::Dynamic { pad: true }` the cache sees only one
  shape (= `max`), so this is a non-issue for serving workloads.
- [**Risk**: stacking N tensors of size 600 KB takes ~1 ms of
  host-side memcpy, eroding small-batch throughput] → Mitigation:
  measured. ResNet-50 inference is ~33 ms; 1 ms of stacking on B=4
  is negligible. For very small graphs (MLP at 0.16 ms inference)
  stacking might dominate; document this.
- [**Risk**: padded inputs produce garbage outputs that callers
  might read by accident if the API is misused] → Mitigation: the
  runtime returns only `K` outputs, never `max`. Padding is
  invisible to the caller.
- [**Risk**: changing the meaning of `max_batch_size` for
  back-compat is confusing] → Mitigation: document the migration
  inline in the field's doc-comment. New code should set
  `BatchPolicy` directly.
- [**Trade-off**: a separate `run_batched` API instead of unifying
  on a single method] → Worth it: explicit batch entry-point is
  easier to use correctly.

## Migration Plan

Purely additive on the public API. `Session::run` continues to work
exactly as before. Users opt into batching by setting
`SessionConfig::batch_policy = BatchPolicy::Static(N)` (or
`Dynamic`) and calling the new `run_batched`. Default
`BatchPolicy::Disabled` rejects `run_batched` — no risk of
accidental use.

Internal change: `Session::run` becomes a thin shim that calls
`run_batched(inputs, 1)`. The actual batched dispatch lives in
`execute_graph_hybrid` / `execute_graph` which already accept
arbitrary batch sizes.

## Open Questions

- Should we expose a `run_batched_owned(Vec<InferenceInput>)`
  variant that consumes its inputs to avoid the per-input
  `tensor.clone()` in `Session::run`'s current implementation?
  Defer until a benchmark identifies clone cost as a real issue.
- For `BatchPolicy::Dynamic { pad: true }`, should we surface the
  pad count in the result (e.g. via a new `InferenceMetadata`
  return)? Defer to a follow-up; v1 returns just outputs.
- Should batched runs respect a per-request budget (`time per
  image` × N) or only a per-batch budget? Probably per-image is
  correct for fairness; document.
