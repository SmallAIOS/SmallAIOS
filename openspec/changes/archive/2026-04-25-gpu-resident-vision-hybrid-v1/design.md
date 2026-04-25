## Context

Current CUDA dispatch in `onnx-rt/src/executor.rs::try_cuda_dispatch`
takes one host `Tensor` per op, runs GPU dispatch via cuBLAS/cuDNN,
and returns a host `Tensor`. This means every GPU-eligible op does:

1. `cudaMemcpy` input host→device.
2. cuDNN/cuBLAS kernel launch.
3. `cudaMemcpy` output device→host.

For ResNet-50 v2 with 53 Convs that's 106 memcpys *just for Convs*,
plus ~16 s of CPU compute on the 117 intermediate BatchNorm/Relu/Add
ops that aren't dispatched. Net speedup observed: 1.17×.

Meanwhile, `onnx-rt/src/cuda/gpu_executor.rs` contains:

- `DeviceTensor { buffer: DeviceBuffer, shape: Vec<i64>, dtype: DataType, name: String }`.
- `execute_graph_gpu(&ExecutionGraph, &[HostInput], &[HostInitializer], &CudaRuntime) -> Result<Vec<HostOutput>>`.
- Device-side `gpu_gemm_device`, per-op dispatch on `DeviceTensor`s.
- Used by the safetensors-backed `Session::run_safetensors` path for
  LLM inference (Gemma, etc.).

What's missing for vision models:

- Device-side `gpu_batchnorm`, `gpu_relu`/`gpu_activation`,
  `gpu_maxpool`/`gpu_avgpool`/`gpu_gap`, `gpu_add`.
- `execute_graph_gpu` isn't wired into `Session::run` for
  ONNX-decoded models — only the safetensors path uses it.
- No mechanism in the standard `execute_graph` to keep activations
  device-resident.

This change fills those gaps with a hybrid executor that extends the
standard `execute_graph` rather than forking a new path, so ONNX and
safetensors sessions share the same device-tensor runtime.

## Goals / Non-Goals

**Goals:**

- Introduce a per-tensor residency signal (`HostOrDevice`) so the
  executor tracks whether each named value lives on host or device.
- Route ops with all-device inputs to device kernels when the op is
  GPU-supported, producing device outputs.
- Route ops with any host input to the existing CPU path, inserting
  device→host copies for the device-resident inputs.
- Insert host→device copies only at entry to a GPU-supported op when
  upstream output is host-resident (typically graph inputs or CPU-op
  outputs).
- End-to-end target: ResNet-50 v2 hybrid speedup ≥5×, SqueezeNet and
  MobileNetV2 ≥3× on DGX Spark.
- Zero regression on the CPU-only test suite and existing single-op
  GPU tests.

**Non-Goals:**

- Batch-dimension support across the inference API. Single-request
  latency is the target here; throughput via batching is a separate
  change.
- Multi-stream / async DMA overlap. Complementary but not required to
  hit the 5× target on ResNet-50 (bottleneck is CPU scalar compute,
  not PCIe bandwidth).
- Op fusion (Conv+BN+Relu, Relu+Add). Performance refinement, lives
  on top of this change.
- GPU kernels for shape-path ops (Gather, Reshape, Concat, Cast,
  Shape, Unsqueeze). Their outputs are small — often shape vectors of
  tens of bytes — so the boundary memcpy costs nothing.
- Precision-mode tuning (FP16 / BF16 / INT8). TF32 default is fine
  for correctness here.
- Making hybrid residency the default. Initial rollout is opt-in via
  `SessionConfig::gpu_residency`.
- Persistent weight preloading across inferences. Weights get copied
  to device on first use and cached for the session lifetime (see
  decision 4 below).

## Decisions

### 1. Extend `execute_graph`, don't fork it

**Decision:** Add residency tracking inside the existing
`execute_graph` function. Replace the `BTreeMap<String, Tensor>`
value map with a `BTreeMap<String, ValueLocation>` where
`ValueLocation` is either `Host(Tensor)` or `Device(Arc<DeviceTensor>)`.

**Rationale:** Forking would duplicate the control-flow handling (If,
Loop, Scan), attribute routing, and operator dispatch. The existing
`execute_graph` is ~1000 lines; cloning it is a maintenance tax we'd
immediately regret. In-place extension keeps the graph-traversal
loop in one place and makes it obvious which ops have GPU residency
support.

**Alternatives considered:**

- Graph-level pre-pass that partitions into GPU and CPU subgraphs and
  emits explicit copy nodes. **Rejected** — requires rewriting
  `ExecutionGraph`, caching per-session, and doesn't compose well
  with control-flow ops. Keeps the run-time path simple.
- Forking to a new `execute_graph_hybrid` function. **Rejected** —
  code duplication as described above.
- Splitting the value map by type (`host_map: BTreeMap<String,
  Tensor>`, `device_map: BTreeMap<String, Arc<DeviceTensor>>`).
  **Rejected** — duplicates lookup logic and makes it easy to
  accidentally have the same name in both maps. Single map with
  a `ValueLocation` variant is the cleanest invariant.

### 2. `ValueLocation` enum, `Arc<DeviceTensor>` for branching

**Decision:**

```rust
enum ValueLocation {
    Host(Tensor),
    Device(Arc<DeviceTensor>),
}
```

Use `Arc<DeviceTensor>` so branching graphs (one tensor feeding
multiple consumers) share a single device buffer without copying.

**Rationale:** All ONNX forward ops in our runtime treat their inputs
as read-only (`&Tensor`); sharing the same `DeviceBuffer` across
readers is always correct. Cloning a `DeviceBuffer` would require a
device→device `cudaMemcpy`, which is fast but wasteful.

**Risk:** if we ever add an in-place op that mutates its input, `Arc`
sharing breaks. Mitigation: audit all GPU ops to ensure
non-mutating input semantics (they already are — cuDNN forward APIs
take `const void*` for inputs).

### 3. Op dispatch decision table

**Decision:** For each op, determine GPU-supportedness at dispatch
time:

```rust
fn gpu_op_supported(op_type: &str, input_dtype: DataType) -> bool {
    matches!(
        (op_type, input_dtype),
        ("Conv", DataType::Float | DataType::BFloat16)
        | ("Gemm", DataType::Float | DataType::BFloat16)
        | ("MatMul", DataType::Float | DataType::BFloat16)
        | ("BatchNormalization", DataType::Float | DataType::BFloat16)
        | ("Relu" | "Clip" | "LeakyRelu", DataType::Float | DataType::BFloat16)
        | ("MaxPool" | "AveragePool" | "GlobalAveragePool", DataType::Float | DataType::BFloat16)
        | ("Add", DataType::Float | DataType::BFloat16)
    )
}
```

The dispatcher reads the inputs' current `ValueLocation`, checks
`gpu_op_supported`, and picks CPU or GPU route.

**Rationale:** Explicit op table is greppable, easy to extend, and
avoids a trait-dispatch layer we don't need yet.

### 4. Weight + BN-parameter device caching

**Decision:** Add a device-side cache keyed by tensor name, lazily
populated on first GPU op that needs an initializer. Lives on
`CudaRuntime` (or a new `DeviceWeightCache` next to it), cleared on
session drop. Supports Conv weights, BN scale/bias/mean/variance,
Gemm weights + bias — anything that appears as an initializer.

**Rationale:** Initializers don't change during inference. Copying
them once and keeping them device-resident is strictly better than
re-copying every call. This is already how
`crate::cuda::initializers_to_gpu` works for the safetensors path;
we reuse that.

**Alternatives considered:**

- Eagerly copy every initializer to device at `Session::initialize`
  time. **Rejected** for CPU-only sessions — we'd waste VRAM on
  tensors never used on GPU.
- Keep a host-side cache only. **Rejected** — defeats the point of
  the change.

### 5. CPU fallback path, mid-graph

**Decision:** When an op has no GPU implementation for its input
dtype, or a required input isn't GPU-eligible (e.g. a `Shape → Gather
→ Reshape` sequence needing int64 shape vectors), copy any
device-resident inputs back to host and run the existing CPU path.
The output lives on host. The *next* GPU-supported op will copy its
inputs back to device if needed.

**Rationale:** This is the whole point of "hybrid". A single CPU op
in the middle of a vision graph must not collapse the rest of the
graph to CPU.

**Worked example (ResNet-50 inner block):**

```
x(host) → Conv(dispatch GPU) → y1(device)
       → BatchNorm(dispatch GPU) → y2(device)
       → Relu(dispatch GPU) → y3(device)
       → Conv(dispatch GPU) → y4(device)
       → BatchNorm(dispatch GPU) → y5(device)
       → Add(dispatch GPU; residual x copied to device) → y6(device)
       → Relu(dispatch GPU) → y7(device)
```

Only the first graph input (x) and the final output incur memcpy;
all intermediates stay on device.

### 6. BatchNorm input/parameter layout

**Decision:** cuDNN's `cudnnBatchNormalizationForwardInference`
expects activations in `NCHW` (or `NHWC`) tensor descriptors and
takes four per-channel parameter buffers (scale γ, bias β, running
mean μ, running variance σ²) as `cudnnTensorDescriptor_t` of shape
`[1, C, 1, 1]`. We mirror that: the GPU op takes five
`DeviceTensor` inputs matching the ONNX operator signature.

**Rationale:** Direct 1:1 mapping to the cuDNN API. No shape
massaging needed for the typical case.

### 7. Pool attribute semantics

**Decision:** Introduce `PoolAttrs { kernel_shape: [i32; 2], pads:
[i32; 4], strides: [i32; 2], ceil_mode: bool, count_include_pad: bool }`
in `operators.rs`, parsed via `PoolAttrs::from_attributes`. Shared
between CPU and GPU dispatch, mirroring the `ConvAttrs` pattern.

**Rationale:** Same proven pattern as `conv-attribute-coverage-v1`.
Avoids drift between CPU and GPU pools.

### 8. `gpu_add` implementation

**Decision:** Start with cuDNN `cudnnOpTensor(OP_TENSOR_ADD, ...)`.
It supports element-wise add with tensor broadcasting against
per-channel or scalar parameters. For shapes cuDNN doesn't accept
(e.g. arbitrary broadcasting), fall back to CPU (which means copying
device inputs back to host — still correct, just slower).

**Rationale:** ResNet residual connections are always identical
shapes (`[N, C, H, W] + [N, C, H, W]`) which cuDNN handles
natively. Minority cases fall through cleanly.

**Alternative:** ship a small custom PTX kernel. **Rejected for now**
— cuDNN covers the common cases at zero additional code.

### 9. `SessionConfig::gpu_residency` opt-in

**Decision:**

```rust
pub enum GpuResidency {
    /// Existing per-op dispatch with host-resident intermediates.
    OpByOp,
    /// Track tensor location; keep activations device-resident across
    /// adjacent GPU-supported ops.
    Hybrid,
}

impl Default for GpuResidency {
    fn default() -> Self {
        GpuResidency::OpByOp
    }
}

pub struct SessionConfig {
    // ...existing fields...
    pub gpu_residency: GpuResidency,
}
```

Default is `OpByOp`. Users opt in with
`SessionConfig { gpu_residency: GpuResidency::Hybrid, ... }`.

**Rationale:** Zero-risk rollout. We validate on the bench suite,
flip the default in a follow-up change once confident.

### 10. Testing strategy

**Decision:** Three layers.

1. **Per-op GPU tests** in `test_cuda.rs`: `test_gpu_batchnorm_matches_cpu`,
   `test_gpu_relu_matches_cpu`, `test_gpu_maxpool_*`,
   `test_gpu_avgpool_*`, `test_gpu_gap_*`, `test_gpu_add_*`. Use the
   same `conv_gpu_vs_cpu_max_abs` pattern already in `test_cuda.rs`.
2. **Hybrid-path integration** test: a 3-node graph
   (Conv → BatchNorm → Relu) that exercises device-residency across
   op boundaries. Assert no memcpy events between the two ops via a
   CUDA-event counter stub (or by checking device pointer equality
   across the intermediate).
3. **End-to-end vision benchmarks**: the existing
   `bench_vision_models.rs` tests flip their residency mode to
   `Hybrid` and hard-assert output shape + diff.

### 11. Instrumentation

**Decision:** Add a `gpu-profile` Cargo feature that, when enabled,
records per-op wall-clock time and memcpy byte counts into an
in-memory ring buffer, dumped at session drop. Off by default so
production builds have zero overhead.

**Rationale:** We'll need per-op attribution to prove the speedup
came from residency vs other factors during code review.

## Risks / Trade-offs

- [**Risk**: a new in-place GPU op is added later and breaks `Arc`
  sharing of device buffers] → Mitigation: document the read-only
  contract on `Device(Arc<DeviceTensor>)` and add a debug-mode
  `Arc::strong_count == 1` check before any op that wants to write in
  place.
- [**Risk**: cuDNN `cudnnBatchNormalizationForwardInference`
  numerical behavior differs subtly from our CPU implementation]
  → Mitigation: strict CPU-vs-GPU test at `max_abs_diff < 1e-3` and a
  dedicated small-input fixture in the unit test so divergence shows
  up immediately.
- [**Risk**: graph output goes device→host but a caller never uses it,
  wasting a memcpy] → Mitigation: output tensors have no special
  handling; assume they're consumed. A future optimization could lazy
  the copy, but the test suite reads outputs so lazy copy adds
  complexity without measured benefit here.
- [**Risk**: workspace allocation for BN/pool scales with input size
  and VRAM is not unlimited] → Mitigation: cuDNN workspace queries are
  already used for Conv. Extend to new ops; if allocation fails, fall
  back to CPU path for that op (and report via `OpError`).
- [**Risk**: the hybrid path has latent bugs that only show up on
  real vision graphs, not the unit tests] → Mitigation: the benchmark
  suite on DGX Spark exercises three different topologies
  (SqueezeNet, MobileNetV2, ResNet-50). All three passing is the
  acceptance gate.
- [**Trade-off**: `Arc<DeviceTensor>` means we can't trivially free
  device buffers as soon as a single consumer finishes. For
  short-lived intermediates that's fine (the Arc drops when the last
  consumer runs); for outputs feeding many consumers it holds the
  buffer longer than strictly needed] → Acceptable on DGX Spark with
  128 GB unified memory. Would revisit on smaller accelerators.
- [**Risk**: behavior changes under the `OpByOp` default because of
  shared attribute parsers] → Mitigation: `OpByOp` path keeps calling
  the same CPU operators; only the GPU-route code sees the shared
  parsers. Every existing test keeps running under default config.

## Migration Plan

Purely additive. Default `GpuResidency::OpByOp` means no existing
caller changes behavior. Users flip to `Hybrid` to try the new path.
Once ResNet-50 benchmark hits ≥5× and the bench suite is green for
several iterations, a tiny follow-up change flips the default.

If we ever need to roll back, the change is contained to
`executor.rs`, `session.rs`, and the new `cuda/{batchnorm,activation,
pool,add}.rs` files — reverting the commit restores prior behavior.

## Open Questions

- Should the residency-tracking code also power the safetensors path
  (collapsing `execute_graph` and `execute_graph_gpu` into one)? Long
  term yes — but this change keeps them separate to keep the blast
  radius small.
- Do we expose VRAM-budget diagnostics in `InferenceProfile`? The
  existing profile type tracks per-op wall clock; adding "peak VRAM
  usage during inference" feels natural but isn't strictly needed for
  this change. Propose: add alongside the `gpu-profile` feature flag.
- Should `GpuResidency::FullDevice` (fail if any op can't run on GPU)
  ship in this change? Leaning no — it's a different failure mode
  (errors where today the model just runs slower). Ship as a
  follow-up once hybrid is stable.
