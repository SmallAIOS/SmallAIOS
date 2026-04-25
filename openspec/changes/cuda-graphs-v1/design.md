## Context

`gpu-resident-vision-hybrid-v1` shipped a hybrid executor that keeps
ResNet-50 v2 activations device-resident across 174 ops. Per-op profile
data:

```
Conv               total=387009 us  count=371  avg=1043 us
Relu               total= 44615 us  count=350  avg=  127 us
BatchNormalization total= 23951 us  count=357  avg=   67 us
Add                total=  7040 us  count=112  avg=   62 us
```

For Relu (127 µs) and BatchNorm (67 µs) the actual GPU compute on a
~200 KB activation is microseconds; the rest is CPU-side kernel-launch
overhead from `cudaLaunchKernel` / `cudnnPoolingForward` /
`cudnnActivationForward` / `cudnnOpTensor`. cuDNN 9 quotes 10–50 µs
per launch on Blackwell.

CUDA Graphs (introduced in CUDA 10.0, hardened through CUDA 12) let
the runtime capture a sequence of kernel launches once and replay
them as a single graph submission. The host-side cost of launching
N kernels collapses to one `cudaGraphLaunch` call. For our 170-op
ResNet-50, that's the difference between ~5 ms of launch overhead
and ~30 µs.

The hybrid executor's existing structure is well-suited to capture
because:

1. The op sequence is fixed for a given input shape (no
   data-dependent control flow in the vision benchmarks).
2. Initializers are already device-resident via the lazy weight
   cache from `gpu-resident-vision-hybrid-v1`.
3. All cuDNN / cuBLAS handles already accept a stream parameter
   via `cudnnSetStream` / `cublasSetStream`.

What's missing: the FFI bindings for `cudaStream` and the
`cudaGraph` family, a per-Session graph cache, and the flow control
to capture once and replay thereafter.

## Goals / Non-Goals

**Goals:**

- ResNet-50 v2 hybrid + graph-capture mode hits **≥1.5× speedup**
  over hybrid alone (~33 ms → ~22 ms or better on DGX Spark).
- MLP / SqueezeNet / MobileNetV2 each see **≥1.2× speedup** over
  their hybrid baselines.
- Output `max_abs_diff` between capture mode and hybrid-no-capture
  matches within `1e-4` (same compute, just different launch
  mechanism).
- Opt-in via `SessionConfig::cuda_graph = CudaGraphMode::Capture`.
  Default (`Off`) is byte-for-byte identical to today's hybrid path.
- Graceful fallback when capture fails for any reason — never crash.

**Non-Goals:**

- Async multi-stream / DMA overlap. Complementary; a separate
  change with its own proposal.
- Variable-input-shape via `cudaGraphExecUpdate`. The initial scope
  rebuilds the graph on shape change (cheap for batch=1 vision
  benches; potentially expensive for diverse-shape serving — that's
  a follow-up).
- Capturing across If/Loop/Scan control-flow ops. The CPU dispatch
  path handles those today.
- Cross-Session graph sharing. Per-Session cache is sufficient for
  the immediate target.
- TensorRT integration. We keep the clean-room ONNX runtime
  principle; this change is pure cuDNN/cuBLAS reuse.
- INT8 / FP8 / NVFP4 quantization paths. Orthogonal.

## Decisions

### 1. Stream capture, not explicit graph construction

**Decision:** Use `cudaStreamBeginCapture(stream, CUDA_STREAM_CAPTURE_MODE_THREAD_LOCAL)`
around the existing hybrid dispatch loop. End with
`cudaStreamEndCapture(stream, &graph)`. Then build a
`cudaGraphExec_t` via `cudaGraphInstantiate`. Replay with
`cudaGraphLaunch(graph_exec, stream)`.

**Rationale:** Stream capture is a thin overlay on the existing
op-by-op code path. Every cuDNN / cuBLAS / cudaMemcpyAsync call
made on the captured stream gets recorded automatically. Explicit
construction (`cudaGraphAddKernelNode` / `cudaGraphAddMemcpyNode`)
would require refactoring every op call site to emit graph nodes
instead of immediate launches — significantly more code, more
fragile to add new ops.

**Alternatives considered:**

- `CUDA_STREAM_CAPTURE_MODE_GLOBAL` — simpler but interferes with
  any concurrent CUDA activity in the same process. Inappropriate
  for a library that may run alongside other CUDA users.
- `CUDA_STREAM_CAPTURE_MODE_RELAXED` — allows a few unsafe APIs
  during capture; rejected because we want strict failure on
  unsupported ops.

### 2. Dedicated per-Session capture stream

**Decision:** Add `capture_stream: cudaStream_t` to a new
`CudaGraphCache` (per-Session, owned by `Session`). Created via
`cudaStreamCreate` lazily on first capture. cuDNN / cuBLAS handles
are bound to this stream via `cudnnSetStream` /
`cublasSetStream` for the duration of capture. Reset to default
stream after capture ends so non-graph paths are unaffected.

**Rationale:** Capture requires a non-default stream. Per-Session
isolation (vs. process-global) prevents stream contention when
multiple sessions exist. cuDNN and cuBLAS already support stream
binding per handle.

**Risk:** Setting + resetting handle streams adds a small
serialization cost on every inference. Acceptable since it happens
once per `run()`, not per op.

### 3. Per-Session graph cache keyed by `(shape, dtype)`

**Decision:**

```rust
struct CudaGraphCache {
    capture_stream: ffi::cudaStream_t,
    entries: BTreeMap<GraphKey, GraphEntry>,
    rebuild_count: u32,
    inference_count: u32,
    disabled: bool,
}

struct GraphKey {
    input_shapes: Vec<Vec<i64>>,
    input_dtypes: Vec<DataType>,
}

struct GraphEntry {
    graph: ffi::cudaGraph_t,
    graph_exec: ffi::cudaGraphExec_t,
    input_buffers: Vec<DeviceBuffer>,   // pinned for graph lifetime
    output_buffers: Vec<DeviceBuffer>,  // pinned for graph lifetime
}
```

On `Session::run`, look up `GraphKey`; replay if hit, capture +
build new `GraphEntry` if miss. After 32 inferences, if
`rebuild_count > inference_count / 100` (1% threshold), set
`disabled = true` and fall back to per-op execution permanently
for this Session (with a single warning log).

**Rationale:** Caching by shape lets serving workloads with batch
sizes 1, 4, 16, 64 keep four graphs alive simultaneously without
churn. The 1% rebuild threshold disables capture when the workload
is so dynamic that caching is counterproductive.

### 4. Pinned input/output buffers

**Decision:** During capture, allocate a `DeviceBuffer` for each
graph input and output of the captured graph. Inputs are filled
via `cudaMemcpyAsync(stream)` from the user's host tensor; outputs
are read via `cudaMemcpyAsync(stream)` after `cudaGraphLaunch`.
The buffers persist for the lifetime of the `GraphEntry`.

**Rationale:** A captured graph hard-codes device pointers. The
input pointer must be stable across replays. Allocating fresh
buffers per inference would require `cudaGraphExecUpdate` to swap
pointers — added complexity. Pinning input/output buffers in the
cache is simpler.

**Memory cost:** Each cached graph holds its input + output device
memory for the Session lifetime. On ResNet-50 (input ~600 KB,
output 4 KB), this is negligible. On large LLMs with many
shape variants it could matter — defer to a follow-up.

### 5. Capture happens inside the existing hybrid dispatch loop

**Decision:** Don't build a parallel "capture executor". Wrap the
existing `execute_graph_hybrid`'s op-loop in
`cudaStreamBeginCapture` / `cudaStreamEndCapture` when capture
mode is on and no cached graph exists for the input shape.

**Rationale:** The hybrid loop already handles all the dispatch
edge cases (CPU fallback for unsupported ops, ensure_device,
ensure_host). Reusing that code path means capture sees exactly
the same op sequence we'd run normally, so the captured graph is
correct by construction.

**Subtle issue:** `ensure_host` (device→host memcpy for CPU
fallback) breaks capture — `cudaMemcpyAsync` from device to host
is allowed, but the subsequent CPU op runs on the host outside of
CUDA. If a CPU op is encountered during capture, we abort the
capture, log a warning ("graph capture aborted: CPU fallback for
op X"), and fall back to per-op execution for this Session. The
1%-rebuild threshold catches this.

### 6. Graceful fallback on capture / replay failure

**Decision:** Three failure modes:

1. **Capture aborts during the warm-up inference** (e.g. CPU
   fallback occurred, or some FFI call isn't capture-safe). Log
   `"graph capture aborted: <reason>"` once, complete the warm-up
   on the per-op path, mark `disabled = true` for this Session.
2. **`cudaGraphInstantiate` fails** (e.g. resource exhaustion).
   Same fallback: warn, disable, continue per-op.
3. **`cudaGraphLaunch` fails on a subsequent inference** (e.g.
   sudden CUDA OOM). Discard the cached `GraphEntry`, attempt
   re-capture next inference; if that fails too, disable.

Errors NEVER propagate out of `Session::run` for capture-related
issues — the inference always completes (potentially via per-op
path). Only actual op failures (e.g. `cudnnConvolutionForward`
returning `CUDNN_STATUS_BAD_PARAM` for an unsupported shape)
propagate as before.

### 7. New cuDNN / CUDA FFI bindings

**Decision:** Add to `onnx-rt/src/cuda/ffi.rs`:

```rust
pub type cudaStream_t = *mut core::ffi::c_void;
pub type cudaGraph_t = *mut core::ffi::c_void;
pub type cudaGraphExec_t = *mut core::ffi::c_void;

#[repr(i32)]
pub enum cudaStreamCaptureMode {
    Global = 0,
    ThreadLocal = 1,
    Relaxed = 2,
}

extern "C" {
    pub fn cudaStreamCreate(stream: *mut cudaStream_t) -> cudaError_t;
    pub fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamBeginCapture(
        stream: cudaStream_t,
        mode: cudaStreamCaptureMode,
    ) -> cudaError_t;
    pub fn cudaStreamEndCapture(
        stream: cudaStream_t,
        graph: *mut cudaGraph_t,
    ) -> cudaError_t;
    pub fn cudaGraphDestroy(graph: cudaGraph_t) -> cudaError_t;
    pub fn cudaGraphInstantiate(
        graph_exec: *mut cudaGraphExec_t,
        graph: cudaGraph_t,
        error_node: *mut *mut core::ffi::c_void,
        log_buffer: *mut u8,
        buffer_size: usize,
    ) -> cudaError_t;
    pub fn cudaGraphExecDestroy(graph_exec: cudaGraphExec_t) -> cudaError_t;
    pub fn cudaGraphLaunch(
        graph_exec: cudaGraphExec_t,
        stream: cudaStream_t,
    ) -> cudaError_t;
}
```

Plus stream-binding additions for cuDNN/cuBLAS:

```rust
extern "C" {
    pub fn cudnnSetStream(handle: cudnnHandle_t, stream: cudaStream_t) -> cudnnStatus_t;
    pub fn cublasSetStream_v2(handle: cublasHandle_t, stream: cudaStream_t) -> cublasStatus_t;
}
```

Wrapped with RAII in a new `onnx-rt/src/cuda/graph.rs` module.

### 8. Profiling integration

**Decision:** Extend the `gpu-profile` feature with three new event
kinds:

```rust
pub enum EventKind {
    Op,
    HostToDevice,
    DeviceToHost,
    GraphCapture,   // NEW: capture wall time
    GraphLaunch,    // NEW: replay wall time
    GraphRebuild,   // NEW: shape change triggered rebuild
}
```

The `dump_to_stderr` summary adds rows for graph events when any
exist. Useful for proving the speedup actually comes from launch
elimination.

### 9. Test strategy

**Decision:** Three layers, mirroring the existing pattern.

1. **Per-op capture-vs-no-capture sanity test** in
   `onnx-rt/tests/test_cuda.rs`. Capture a Conv → BN → Relu chain,
   replay it, assert output matches the per-op path within
   `1e-4`.
2. **Per-bench capture variants** in `bench_vision_models.rs`. Each
   existing `bench_*_cpu_vs_gpu_hybrid` test gets a sibling
   `_with_graph` variant that flips `cuda_graph = Capture`. Latency
   delta and output diff are reported.
3. **Capture-failure fallback test**: deliberately inject a CPU
   fallback into a captured graph (via a small synthetic ONNX
   model with a Reshape node), assert capture aborts cleanly and
   the inference still completes via per-op fallback.

## Risks / Trade-offs

- [**Risk**: cuDNN heuristic algo selection picks differently after
  capture (different shapes wouldn't reach this code path because
  we re-capture on shape change, but defensive)] → Mitigation: the
  existing `gpu_conv2d_device` algo fallback list is deterministic
  for fixed shapes. Document the determinism.
- [**Risk**: a future op uses a CUDA API that's not capture-safe
  (e.g. `cudaMallocAsync` from inside an op)] → Mitigation: the
  capture-failure fallback (decision 6) covers this. New ops that
  need such APIs should explicitly opt out of capture by checking
  the runtime mode and bypassing the device path.
- [**Risk**: pinned input/output buffers in the cache leak device
  memory if `Session` is dropped without `Drop` running on the
  cache] → Mitigation: standard Rust RAII via `Drop` impls on
  `GraphEntry` and `CudaGraphCache`. Audit on review.
- [**Risk**: graph capture fails silently and the user thinks they're
  using capture mode but get the per-op path] → Mitigation: log
  exactly once per Session when capture is disabled, reason
  included. Optionally surface a `Session::cuda_graph_status() ->
  CudaGraphStatus` introspection method.
- [**Trade-off**: per-Session cache means each Session pays the
  capture cost once, even for identical model + shape] → Acceptable.
  Cross-Session sharing is a follow-up if benchmarks show it
  matters.
- [**Trade-off**: rebuilding on every shape change makes diverse-
  batch-size serving slow on the first inference of each new shape]
  → Acceptable for v1. `cudaGraphExecUpdate` for in-place pointer
  swaps is a follow-up.

## Migration Plan

Purely additive. Default `CudaGraphMode::Off` keeps every existing
caller's behavior byte-for-byte. Users opt in by setting
`gpu_residency = Hybrid` and `cuda_graph = Capture` on their
`SessionConfig`. No API breakage.

Rollback: revert the commit; the executor reverts to the per-op
hybrid path.

## Open Questions

- Should we expose `Session::cuda_graph_stats() -> CudaGraphStats`
  with capture/launch counts and rebuild count? Useful for
  observability in production. Proposing yes, kept opt-in.
- Should `CudaGraphMode` have a `CaptureWithUpdate` variant that
  uses `cudaGraphExecUpdate` for in-place pointer rebinding on
  shape changes? Defer to a follow-up; v1 keeps it simple.
- Do we need to handle a session that mixes hybrid and op-by-op
  ops in the same graph? In practice the hybrid executor runs all
  GPU-eligible ops on GPU; CPU fallback for shape-path ops triggers
  capture abort. So no — same-graph mixed mode isn't a goal.
