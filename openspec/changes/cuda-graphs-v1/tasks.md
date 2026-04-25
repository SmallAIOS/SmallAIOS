## 1. CUDA + cuDNN/cuBLAS FFI bindings

- [ ] 1.1 Add `cudaStream_t`, `cudaGraph_t`, `cudaGraphExec_t` type aliases in `onnx-rt/src/cuda/ffi.rs`
- [ ] 1.2 Add `cudaStreamCaptureMode` enum (`Global = 0`, `ThreadLocal = 1`, `Relaxed = 2`)
- [ ] 1.3 Add extern declarations: `cudaStreamCreate`, `cudaStreamDestroy`, `cudaStreamSynchronize`, `cudaStreamBeginCapture`, `cudaStreamEndCapture`
- [ ] 1.4 Add extern declarations: `cudaGraphInstantiate`, `cudaGraphLaunch`, `cudaGraphExecDestroy`, `cudaGraphDestroy`
- [ ] 1.5 Add `cudnnSetStream` extern in the cuDNN block; add `cublasSetStream_v2` in the cuBLAS block

## 2. RAII wrappers for graph + stream resources

- [ ] 2.1 Create `onnx-rt/src/cuda/graph.rs` module
- [ ] 2.2 Add `pub struct CaptureStream { stream: cudaStream_t }` with `new()` (calls `cudaStreamCreate`) and `Drop` (calls `cudaStreamDestroy`)
- [ ] 2.3 Add `pub struct CudaGraph { graph: cudaGraph_t }` with `Drop` calling `cudaGraphDestroy`
- [ ] 2.4 Add `pub struct CudaGraphExec { graph_exec: cudaGraphExec_t }` with `Drop` calling `cudaGraphExecDestroy`
- [ ] 2.5 Add `pub mod graph;` in `onnx-rt/src/cuda/mod.rs`

## 3. SessionConfig + CudaGraphMode

- [ ] 3.1 Add `pub enum CudaGraphMode { Off, Capture }` with `#[derive(Default)] #[default] Off` in `onnx-rt/src/session.rs`
- [ ] 3.2 Add `pub cuda_graph: CudaGraphMode` field to `SessionConfig`
- [ ] 3.3 Update `Default for SessionConfig` to include `cuda_graph: CudaGraphMode::default()`
- [ ] 3.4 Update any test SessionConfig literals (`tests/integration_inference.rs`) to include the new field

## 4. CudaGraphCache + GraphEntry

- [ ] 4.1 Define `GraphKey { input_shapes: Vec<Vec<i64>>, input_dtypes: Vec<DataType> }` (in `executor_hybrid.rs` or a new submodule)
- [ ] 4.2 Define `GraphEntry { graph: CudaGraph, graph_exec: CudaGraphExec, input_buffers: Vec<DeviceBuffer>, output_buffers: Vec<DeviceBuffer> }`
- [ ] 4.3 Define `pub struct CudaGraphCache { capture_stream: CaptureStream, entries: BTreeMap<GraphKey, GraphEntry>, rebuild_count: u32, inference_count: u32, disabled: bool }`
- [ ] 4.4 Add `Session::cuda_graph_cache: RefCell<Option<CudaGraphCache>>` field, gated on `cfg(feature = "cuda")`, lazy-init on first hybrid+capture run
- [ ] 4.5 Add a type alias `CudaGraphCacheSlot = Option<CudaGraphCache>` to avoid `clippy::type_complexity`

## 5. Capture path in execute_graph_hybrid

- [ ] 5.1 Extend `execute_graph_hybrid` signature with an optional `&mut CudaGraphCache` and `cuda_graph_mode: CudaGraphMode`
- [ ] 5.2 Build a `GraphKey` from the inference inputs at the top of `execute_graph_hybrid`
- [ ] 5.3 If mode is `Capture`, cache is not disabled, and a `GraphEntry` exists for the key → take the replay path (Section 6); otherwise fall through to capture or per-op
- [ ] 5.4 If mode is `Capture` and no entry exists → bind cuDNN/cuBLAS handles to `cache.capture_stream`, call `cudaStreamBeginCapture(stream, ThreadLocal)`, run the existing op-dispatch loop, call `cudaStreamEndCapture(stream, &graph)`
- [ ] 5.5 On capture success, instantiate via `cudaGraphInstantiate`, allocate pinned input/output `DeviceBuffer`s sized to match the graph inputs/outputs, store the `GraphEntry` keyed by `GraphKey`
- [ ] 5.6 On any capture failure (returns non-`SUCCESS` from `cudaStreamEndCapture` or `cudaGraphInstantiate`, or a CPU-fallback occurred mid-capture) → log warning, set `cache.disabled = true`, leave the warm-up inference's outputs intact (it already ran via the per-op path), return them
- [ ] 5.7 Always reset cuDNN/cuBLAS handles to the default stream (`null_mut()`) after capture or replay

## 6. Replay path

- [ ] 6.1 On replay, `cudaMemcpyAsync` user input host bytes into `entry.input_buffers` on `cache.capture_stream`
- [ ] 6.2 Call `cudaGraphLaunch(entry.graph_exec, cache.capture_stream)`
- [ ] 6.3 `cudaMemcpyAsync` the contents of `entry.output_buffers` back to fresh host `Tensor`s
- [ ] 6.4 `cudaStreamSynchronize(cache.capture_stream)` before returning
- [ ] 6.5 If any of the above returns non-success → discard `entry`, log warning, fall back to per-op execution for this inference

## 7. Disable-on-thrashing logic

- [ ] 7.1 Increment `cache.inference_count` on every successful `Session::run` that reaches the cache
- [ ] 7.2 Increment `cache.rebuild_count` whenever a new `GraphEntry` is captured
- [ ] 7.3 After `inference_count >= 32`, if `rebuild_count * 100 > inference_count` (>1%), set `cache.disabled = true` and log a single info message
- [ ] 7.4 Once disabled, all subsequent runs use the per-op path; the cache state persists for diagnostics until Session drop

## 8. Profile feature integration

- [ ] 8.1 Extend `EventKind` in `onnx-rt/src/cuda/profile.rs` with `GraphCapture`, `GraphLaunch`, `GraphRebuild`
- [ ] 8.2 Wrap `cudaGraphInstantiate` with `record_op("graph_capture", start)` (under feature gate)
- [ ] 8.3 Wrap `cudaGraphLaunch` with `record_op("graph_launch", start)` (under feature gate)
- [ ] 8.4 Update `dump_to_stderr` to print rebuild count and capture/launch timings as a separate "graph" section

## 9. Per-op capture sanity test

- [ ] 9.1 Add `test_cuda_graph_capture_relu_chain` in `onnx-rt/tests/test_cuda.rs`: capture a `Conv → BatchNorm → Relu` chain on a tiny `[1, 16, 8, 8]` input
- [ ] 9.2 Replay the captured graph; assert output matches the per-op (no-capture) path within `1e-4`
- [ ] 9.3 Add `test_cuda_graph_capture_aborts_on_cpu_fallback`: capture a chain that includes a `Reshape` (CPU-only) and assert capture gracefully aborts and the per-op path still produces correct output

## 10. End-to-end vision benchmarks

- [ ] 10.1 Add a `BenchMode::HybridGraph` variant in `onnx-rt/tests/bench_vision_models.rs`
- [ ] 10.2 Add 4 new bench tests: `bench_{mlp,squeezenet,mobilenet_v2,resnet50}_cpu_vs_gpu_hybrid_with_graph`
- [ ] 10.3 Each bench compares latency vs the hybrid-no-graph baseline and reports the ratio
- [ ] 10.4 Run all 4 on DGX Spark; record results in `docs/benchmarks/arm64-gpu-cpu-vs-gpu.md`
- [ ] 10.5 Verify the `≥1.5× ResNet-50 / ≥1.2× others` targets are met
- [ ] 10.6 Verify `max_abs_diff` between capture-mode and per-op outputs ≤ `1e-4`

## 11. Documentation

- [ ] 11.1 Add a "CUDA Graph Capture" subsection to the GPU Residency section in `docs/architecture.md`
- [ ] 11.2 Update `docs/benchmarks/arm64-gpu-cpu-vs-gpu.md` with the new capture-mode results and a "How it works" sidebar
- [ ] 11.3 Document `CudaGraphMode` and the `cuda_graph` field in the `SessionConfig` doc comment

## 12. Final verification

- [ ] 12.1 `cargo fmt -p smallaios-onnx-rt`
- [ ] 12.2 `cargo clippy -p smallaios-onnx-rt --no-default-features --features cpu -- -D warnings` clean
- [ ] 12.3 `cargo clippy -p smallaios-onnx-rt --no-default-features --features cuda -- -D warnings` clean
- [ ] 12.4 `cargo clippy -p smallaios-onnx-rt --no-default-features --features gpu-profile -- -D warnings` clean
- [ ] 12.5 `cargo test -p smallaios-onnx-rt --no-default-features --features cpu` — full suite green
- [ ] 12.6 `cargo test -p smallaios-onnx-rt --lib --no-default-features --features cuda` — full suite green
- [ ] 12.7 `cargo test -p smallaios-onnx-rt --release --test bench_vision_models --features cuda -- --ignored --nocapture` — all hybrid + hybrid_with_graph benches pass
- [ ] 12.8 `openspec validate cuda-graphs-v1 --strict` passes
