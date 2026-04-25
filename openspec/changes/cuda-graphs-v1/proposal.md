## Why

The hybrid GPU executor delivers 418× CPU-vs-GPU speedup on ResNet-50
v2, but profile data from the `gpu-profile` feature shows per-op
average times of 67–127 µs for cheap ops (BatchNorm/Relu) — dominated
by per-call CUDA launch overhead, not real compute. With ~170 ops
× 7 inferences per benchmark and ~30 µs per launch, ~36 ms of pure
launch overhead per benchmark is unavoidable today. CUDA Graphs
replace the per-op launch storm with a single `cudaGraphLaunch`,
recovering 15–30% of inference latency without changing any kernel.

## What Changes

- Add `SessionConfig::cuda_graph: CudaGraphMode` (variants `Off` (default), `Capture`).
- Wrap the first hybrid inference of a Session in
  `cudaStreamBeginCapture` / `cudaStreamEndCapture` on a dedicated
  per-Session capture stream. Bind cuDNN / cuBLAS handles to that
  stream during capture.
- Build a `cudaGraphExec_t` from the captured graph, store it in a
  per-Session `CudaGraphCache` keyed by `(input shape, dtype)`.
- Subsequent inferences with matching input shape skip the per-op
  dispatch loop and call `cudaGraphLaunch` once, copying inputs
  into the cached input buffer and outputs out of the cached output
  buffer.
- Invalidate and rebuild on input-shape change. Disable capture for
  the Session if rebuilds happen on >1% of inferences (defensive
  guard for poorly-behaved workloads).
- Extend `gpu-profile` to report graph capture time, replay time,
  and rebuild count.
- Add `bench_vision_models.rs` variants that flip on capture mode and
  compare against hybrid baseline.
- Document the new mode in `docs/architecture.md` and the benchmark
  doc.

## Capabilities

### New Capabilities

- _None_ — this is a performance optimization layered on top of
  existing GPU execution capabilities.

### Modified Capabilities

- `onnx-runtime`: extend the CUDA Execution Provider with
  graph-capture / replay scenarios for the hybrid executor; add a
  `SessionConfig::cuda_graph` opt-in flag and the launch-overhead
  reduction success criterion.

## Impact

- **Code:** `onnx-rt/src/session.rs` (new `CudaGraphMode` enum +
  `SessionConfig` field), `onnx-rt/src/executor_hybrid.rs` (capture +
  replay paths), new `onnx-rt/src/cuda/graph.rs` (RAII wrappers for
  `cudaGraph_t` / `cudaGraphExec_t` / capture streams),
  `onnx-rt/src/cuda/ffi.rs` (cudaStream + cudaGraph FFI bindings).
- **Tests:** new `test_cuda_graph_capture_relu_chain`-style per-op
  capture tests + `bench_vision_models.rs` capture-mode variants.
- **Downstream:** ResNet-50 hybrid latency drops from ~33 ms → ~22
  ms (target). MLP / SqueezeNet / MobileNetV2 see proportional
  improvements bounded by their per-op count. Default behavior
  unchanged when `CudaGraphMode::Off`.
- **Dependencies:** CUDA 12.3+ for `cudaStreamBeginCapture` mode
  features; CUDA 13.0 already installed on DGX Spark satisfies this.
- **Out of scope (flagged):** multi-stream / async DMA overlap,
  variable-shape graph patching via `cudaGraphExecUpdate`, control-
  flow op capture, cross-Session graph sharing.
