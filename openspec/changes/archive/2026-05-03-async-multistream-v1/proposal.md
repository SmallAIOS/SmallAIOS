## Why

The hybrid GPU executor and `cuda-graphs-v1` capture path both run on
a single CUDA stream today. Every host↔device memcpy and every kernel
launch is sequenced strictly. For multi-request serving workloads —
where one inference's H2D transfer could overlap with the previous
inference's compute, and the next inference's D2H copy could start
before the current one finishes — single-stream execution leaves
substantial parallelism on the table. A 2-stream pipeline (compute on
one stream, transfers on another) typically delivers 1.3-1.5× extra
throughput at modest complexity. For batched workloads it
compounds with `dynamic-batching-v1`: B=64 batched + 2-stream
overlap could deliver 25-40× throughput vs B=1 single-stream.

## What Changes

- Add a `StreamConfig` to `SessionConfig` controlling stream count
  and overlap mode:
  - `StreamConfig::SingleStream` — current behavior, default.
  - `StreamConfig::Overlap { transfer_streams: usize }` — dedicate
    N streams to host↔device transfers, run compute on the main
    stream, use CUDA events to gate dependencies.
- Issue input H2D `cudaMemcpyAsync` on a dedicated transfer stream
  while the previous inference is still computing on the compute
  stream. Use a CUDA event to make the compute stream wait on the
  transfer.
- Issue output D2H `cudaMemcpyAsync` on a dedicated transfer stream
  after recording an event on the compute stream. The host wait on
  the inference output is a `cudaStreamSynchronize` on the transfer
  stream rather than the whole compute stream.
- Compose with `cuda-graphs-v1`: the compute stream is the captured
  stream; transfers happen on the side streams via async memcpys
  + events. The captured graph still wraps purely the compute
  portion.
- Compose with `dynamic-batching-v1`: a future serving layer can
  issue overlapping `run_batched` calls; each one threads through
  the stream pool.
- Add throughput benchmarks comparing single-stream vs
  2-stream / 3-stream modes.
- Document the streaming model in `docs/architecture.md`.

## Capabilities

### New Capabilities

- _None_ — multi-stream is an extension of existing GPU execution.

### Modified Capabilities

- `onnx-runtime`: extend the CUDA Execution Provider with
  multi-stream execution scenarios (transfer/compute overlap, event
  synchronization, ordering invariants).

## Impact

- **Code:** new `onnx-rt/src/cuda/streams.rs` module (RAII
  `Stream` + `Event` wrappers), changes to `executor_hybrid.rs`
  (route H2D/D2H to transfer streams, insert events), new
  `StreamConfig` enum + field on `SessionConfig`. cuDNN handle
  binding may need to flip between streams when capture mode is on.
- **Tests:** ordering correctness tests (assert no read-before-write
  hazards), throughput benchmarks at B=1 / B=16 / B=64 with
  single-stream vs 2-stream comparison.
- **Downstream:** ~1.3-1.5× extra throughput on serving workloads
  beyond what `dynamic-batching-v1` delivers. Latency for a single
  request is unchanged or marginally improved.
- **Dependencies:** none new — CUDA streams + events have been in
  CUDA since before our minimum-supported version. Composes with
  `cuda-graphs-v1` (must run AFTER cuda-graphs-v1 lands so the
  capture stream and the transfer streams interact correctly) and
  `dynamic-batching-v1`.
- **Out of scope (flagged):** truly concurrent multi-request inference
  across Sessions (still synchronizes at the Session level), HTTP
  request scheduling / batching that's the container layer's job,
  GPU-side kernel-level parallelism (CUDA streams are about
  inter-stream overlap; intra-kernel concurrency is cuDNN's job).
