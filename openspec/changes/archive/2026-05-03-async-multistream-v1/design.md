## Context

Today the hybrid executor uses a single (default or capture) CUDA
stream. Every operation — host→device memcpy of inputs, all the cuDNN
kernel launches, device→host memcpy of outputs — runs in strict
order on that one stream. After `cuda-graphs-v1`, the captured graph
launches in one shot but still on a single stream, and inputs/outputs
still memcpy serially around the launch.

CUDA streams provide independent in-order command queues that the GPU
schedules in parallel. Two streams can:

- run kernels concurrently (subject to GPU SM occupancy)
- overlap host↔device memcpy on a copy engine with kernel execution
  on the SMs

For our workload, the highest-value pattern is **transfer/compute
overlap**: one inference's H2D copy runs on a dedicated transfer
stream while the previous inference's compute is still finishing on
the main stream. The same pattern applies to D2H of the output.
CUDA events (`cudaEventRecord` / `cudaStreamWaitEvent`) provide the
dependency edges between streams.

For DGX Spark's GB10, the unified-memory architecture changes the
calculus a bit (host and device memory share the same pool with
cache coherence), but the API model is unchanged — the runtime still
needs to express dependencies even if the hardware can fulfill them
without a literal copy.

## Goals / Non-Goals

**Goals:**

- Add a `StreamConfig::Overlap { transfer_streams }` mode that
  dedicates separate streams for H2D and D2H transfers, gated by
  CUDA events for correctness.
- Compose cleanly with `cuda-graphs-v1`: the compute stream is the
  capture/replay stream; transfer streams are external to the
  captured graph.
- Compose cleanly with `dynamic-batching-v1`: stream config is
  per-Session, so batched runs benefit too.
- Hit ≥1.3× throughput improvement at B=64 with 2-stream overlap
  vs single-stream baseline.
- Preserve correctness: outputs MUST be byte-for-byte identical to
  the single-stream baseline. Events MUST gate every cross-stream
  dependency.

**Non-Goals:**

- Cross-Session stream sharing. Each Session owns its streams.
- Out-of-order inference (running multiple `run()` calls in
  parallel from a single thread). Still synchronizes at the
  `Session::run` boundary.
- HTTP/RPC request scheduling that batches requests across clients
  — that's the container/server layer's concern.
- Pipeline parallelism within a single inference (e.g. streaming
  the first half of the network on stream A while the second half
  starts on stream B). Way more complex; defer to a future change
  if profiling shows it matters.
- GPU-direct storage (`GDS`) or RDMA paths.

## Decisions

### 1. Streams in a per-Session pool

**Decision:** Add a `StreamPool` to `Session`:

```rust
pub struct StreamPool {
    compute: Stream,   // main compute stream (or graph-capture stream)
    h2d: Vec<Stream>,  // configurable count of H2D streams
    d2h: Vec<Stream>,  // configurable count of D2H streams
    events: Vec<Event>, // pool of reusable events for cross-stream deps
}
```

`StreamConfig::Overlap { transfer_streams: 1 }` allocates one h2d
and one d2h stream. `transfer_streams: 2` allocates two of each
(overlap-while-overlapping for higher batch sizes).

**Rationale:** Per-Session ownership keeps stream destruction safe
(`Drop` impl). A pool of events avoids constant `cudaEventCreate` /
`cudaEventDestroy` churn.

### 2. Event-gated dependencies, not implicit synchronization

**Decision:** Every cross-stream dependency uses `cudaEventRecord`
+ `cudaStreamWaitEvent` explicitly. Never rely on
`cudaStreamSynchronize` or `cudaDeviceSynchronize` mid-pipeline —
those flush the entire stream / device and kill the overlap we're
trying to create.

Pipeline outline for a single inference with 2 streams:

1. Record event `e_h2d_done` on the H2D stream after the input
   memcpy.
2. Have the compute stream wait on `e_h2d_done` before launching
   the first kernel.
3. After the last kernel (or `cudaGraphLaunch` if combined with
   `cuda-graphs-v1`), record event `e_compute_done` on the compute
   stream.
4. Have the D2H stream wait on `e_compute_done` before issuing the
   output memcpy.
5. Synchronize the D2H stream before returning to the caller (this
   is the only `cudaStreamSynchronize` in the inner loop).

**Rationale:** Explicit event dependencies are the supported,
deterministic way to coordinate streams. Implicit ordering is a
foot-gun.

### 3. Output handed back synchronously

**Decision:** `Session::run` still returns when the output is on
host. Only the D2H stream needs to be synchronized at the end; the
compute stream can keep running cleanup or warming up the next
inference.

**Rationale:** Users expect `run()` to return a usable output. We
can revisit this with an async API in a separate change if there's
demand for it.

### 4. cuDNN/cuBLAS handle stream binding

**Decision:** During an inference, bind cuDNN and cuBLAS handles to
`pool.compute` via `cudnnSetStream` / `cublasSetStream_v2`. Reset
to default stream after the inference completes (so any subsequent
non-overlap inference sees the original behavior).

This is the same dance `cuda-graphs-v1` does. When both
`cuda-graphs-v1` and `async-multistream-v1` are active, the compute
stream IS the capture stream — no conflict.

**Rationale:** cuDNN/cuBLAS stream binding is per-handle, and we
have one set of handles per `CudaRuntime`. Binding at run-time is
the only correct option.

### 5. Compose with `cuda-graphs-v1`

**Decision:** When both `CudaGraphMode::Capture` and
`StreamConfig::Overlap` are active:

- The captured stream IS `pool.compute`.
- Capture happens during the warm-up inference exactly as today.
- During replay, the runtime issues `cudaMemcpyAsync` to
  `pool.h2d[0]` to put inputs in the cached input buffer, records
  an event, has the compute stream wait on it, calls
  `cudaGraphLaunch(graph_exec, pool.compute)`, records another
  event, has the D2H stream wait on it, copies outputs to host,
  syncs D2H stream.

This way the captured graph stays internally consistent (single
stream during execution) while transfers overlap externally.

### 6. `StreamConfig` opt-in

**Decision:**

```rust
pub enum StreamConfig {
    /// Existing single-stream behavior. Default.
    SingleStream,
    /// Dedicate `transfer_streams` parallel streams each for H2D
    /// and D2H. Compute runs on a separate dedicated stream.
    /// `transfer_streams = 1` is the typical setting.
    Overlap { transfer_streams: usize },
}

impl Default for StreamConfig {
    fn default() -> Self { StreamConfig::SingleStream }
}
```

`SessionConfig` gains `pub stream_config: StreamConfig`.

**Rationale:** Default-off matches the rest of our perf knobs.
Users opt in via `SessionConfig`.

### 7. Throughput target measurement

**Decision:** Add three new benchmarks:

- `bench_resnet50_throughput_b64_singlestream` — baseline, equivalent
  to the existing B=64 throughput bench from `dynamic-batching-v1`.
- `bench_resnet50_throughput_b64_2stream` — same workload with
  `StreamConfig::Overlap { transfer_streams: 1 }`.
- `bench_resnet50_throughput_b64_2stream_with_graph` — same but
  composes `cuda-graphs-v1` + `async-multistream-v1`.

Targets:
- 2-stream ≥ 1.3× single-stream throughput
- 2-stream + graph ≥ 1.5× single-stream + graph throughput

### 8. Test strategy

Three layers:

1. **Event ordering correctness test**: a Conv → BN → Conv chain,
   manually intercepting the inference to assert that
   `cudaStreamWaitEvent` is called before the compute stream
   accesses the input buffer (use a debug-instrumented FFI shim
   under a test-only feature).
2. **Output equivalence test**: same model, same input, run once
   single-stream and once 2-stream. Outputs MUST be byte-for-byte
   identical.
3. **Throughput benchmarks** as in Section 7.

## Risks / Trade-offs

- [**Risk**: missed event dependency causes a silent
  use-before-write race; outputs are wrong sporadically] →
  Mitigation: every cross-stream access goes through a single
  helper that records or waits on an event; never bare async
  memcpys without event guarding. Audit on review. Consider a
  debug-mode "strict" check that issues `cudaDeviceSynchronize`
  between streams to catch ordering bugs early.
- [**Risk**: stream count > GPU's available copy engines bottlenecks
  on PCIe rather than parallelizing] → Mitigation: 2 streams (1 H2D,
  1 D2H) maps to most GPUs' 2 copy engines. Higher counts won't
  help and we cap `transfer_streams` at 2 by validation in v1.
- [**Risk**: per-Session stream pool leaks if Session is dropped
  during an in-flight inference (race between Drop and CUDA work)]
  → Mitigation: `Session::Drop` calls `cudaStreamSynchronize` on
  every stream before destroying. Audit for Drop-order safety.
- [**Risk**: cuDNN handles bound to a custom stream during overlap
  but a non-overlap inference happens on the same Session next] →
  Mitigation: always reset to default stream at the end of every
  inference. Add a test that mixes `Overlap` and `SingleStream`
  configs across sessions to ensure no leakage.
- [**Trade-off**: per-Session stream pool means short-lived sessions
  pay constant stream-creation cost] → Acceptable; 2 streams is
  microseconds. Pool pre-allocation handled in `Session::new`.

## Migration Plan

Purely additive. Default `StreamConfig::SingleStream` is the
existing behavior. Users opt in by setting
`SessionConfig::stream_config = StreamConfig::Overlap { transfer_streams: 1 }`.

Composition order with other changes:
- `async-multistream-v1` MUST land after `cuda-graphs-v1` because
  the multi-stream pipeline needs to know how to compose with
  graph capture/replay.
- `async-multistream-v1` is independent of `dynamic-batching-v1` —
  either order works; combining gives the biggest payoff.

Rollback: revert; sessions go back to single-stream.

## Open Questions

- Does a 3-stream config (split H2D into 2 parallel streams) help
  for very large inputs (e.g. high-batch ResNet)? Probably not —
  the single H2D channel is already saturating PCIe in benchmarks.
  Defer to measurement.
- Should we expose a `Session::stream_stats()` introspection method
  that reports actual overlap percentage? Useful for tuning. Defer
  unless first user asks.
- Async result API (`run` returns a `Future` instead of a `Vec`)?
  Out of scope — the synchronous API is what the rest of the
  runtime expects. Async would be a separate change with a
  Tokio-or-equivalent dependency decision.
