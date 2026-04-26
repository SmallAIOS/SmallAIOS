## 1. Stream / Event FFI

- [x] 1.1 Verify `cudaStreamCreate` / `cudaStreamDestroy` / `cudaStreamSynchronize` are present in `onnx-rt/src/cuda/ffi.rs` (added by `cuda-graphs-v1`); add if missing
- [x] 1.2 Add `cudaEvent_t` type alias
- [x] 1.3 Add extern declarations: `cudaEventCreate`, `cudaEventDestroy`, `cudaEventRecord`, `cudaStreamWaitEvent`, `cudaEventSynchronize`
- [x] 1.4 Add `cudaMemcpyAsync` extern declaration (kind enum already exists)

## 2. RAII wrappers

- [x] 2.1 Create `onnx-rt/src/cuda/streams.rs` module
- [x] 2.2 Add `pub struct Stream { stream: cudaStream_t }` with `new() -> Result<Self, CudaError>`, `Drop` calling `cudaStreamDestroy`
- [x] 2.3 Add `pub struct Event { event: cudaEvent_t }` with `new()`, `record(&self, &Stream)`, `wait(&self, &Stream)`, `Drop`
- [x] 2.4 Add `pub struct StreamPool { compute: Stream, h2d: Vec<Stream>, d2h: Vec<Stream>, events: RefCell<Vec<Event>> }`
- [x] 2.5 Add `StreamPool::new(transfer_streams: usize) -> Result<Self, CudaError>` allocating `transfer_streams` H2D + `transfer_streams` D2H streams + a starter event pool
- [x] 2.6 Add `StreamPool::acquire_event() -> Event` and `release_event(e: Event)` for pool reuse
- [x] 2.7 Wire `pub mod streams;` in `onnx-rt/src/cuda/mod.rs`

## 3. SessionConfig + StreamConfig

- [x] 3.1 Add `pub enum StreamConfig { SingleStream, Overlap { transfer_streams: usize } }` with `#[derive(Default)] #[default] SingleStream` in `onnx-rt/src/session.rs`
- [x] 3.2 Add `pub stream_config: StreamConfig` field to `SessionConfig`
- [x] 3.3 Update `Default for SessionConfig`
- [x] 3.4 Update SessionConfig literals in `tests/integration_inference.rs` etc.
- [x] 3.5 Add `SessionError::InvalidConfig(String)` variant if not present
- [x] 3.6 Validate `transfer_streams <= 2` at `Session::new`; return `InvalidConfig` if exceeded

## 4. Stream pool on Session

- [x] 4.1 Add `pub stream_pool: RefCell<Option<StreamPool>>` field to `Session`, gated on `cfg(feature = "cuda")`
- [x] 4.2 Update Session constructor to initialize `stream_pool: RefCell::new(None)` (lazy alloc)
- [x] 4.3 Update `Session::Drop` (or rely on `Drop` of `StreamPool`) to call `cudaStreamSynchronize` on every stream before destruction
- [x] 4.4 Add helper `Session::ensure_stream_pool()` that lazily creates `StreamPool` on first multi-stream inference

## 5. Multi-stream inference path in execute_graph_hybrid

- [x] 5.1 Add `stream_config: StreamConfig` parameter to `execute_graph_hybrid`
- [x] 5.2 When `Overlap`, route input H2D memcpys to `pool.h2d[0]` via `cudaMemcpyAsync`, record `h2d_done_event`
- [x] 5.3 Bind cuDNN + cuBLAS handles to `pool.compute` via `cudnnSetStream` + `cublasSetStream_v2` for the duration of compute
- [x] 5.4 Have the compute stream `wait` on `h2d_done_event` before the first kernel
- [x] 5.5 Run the existing op-dispatch loop (or `cudaGraphLaunch` if `cuda-graphs-v1` capture is also active) on `pool.compute`
- [x] 5.6 After the last kernel, record `compute_done_event` on the compute stream
- [x] 5.7 Have `pool.d2h[0]` wait on `compute_done_event`, issue output `cudaMemcpyAsync` on the D2H stream
- [x] 5.8 `cudaStreamSynchronize(pool.d2h[0])` before returning to caller; reset cuDNN/cuBLAS to default stream

## 6. Composition with cuda-graphs-v1

- [x] 6.1 In `executor_hybrid.rs`, when both `CudaGraphMode::Capture` AND `StreamConfig::Overlap` are active, the captured stream MUST be `pool.compute` (set during capture via `cudnnSetStream` etc.)
- [x] 6.2 During replay, the input memcpy goes to the cached graph input buffer via `pool.h2d[0]`, gated by `h2d_done_event`
- [x] 6.3 `cudaGraphLaunch` is called on `pool.compute`
- [x] 6.4 Output memcpy from cached output buffer goes via `pool.d2h[0]`, gated by `compute_done_event`
- [x] 6.5 Add `test_hybrid_with_graph_and_2stream_resnet50_correctness`: run B=1 ResNet-50 with all three opt-in flags on; assert output matches single-stream baseline byte-for-byte

## 7. Per-request stream pool in dynamic-batching composition

- [ ] 7.1 DEFERRED: Verify `Session::run_batched` (from `dynamic-batching-v1`) routes through the stream pool the same way `Session::run` does — depends on `dynamic-batching-v1` which has not landed yet
- [ ] 7.2 DEFERRED: Add `test_hybrid_batched_b16_2stream_correctness` — depends on `dynamic-batching-v1`

## 8. Throughput benchmarks

- [ ] 8.1 DEFERRED: `bench_resnet50_throughput_b64_singlestream` — needs B=64 batched dispatch from `dynamic-batching-v1`
- [ ] 8.2 DEFERRED: `bench_resnet50_throughput_b64_2stream` — same dependency
- [ ] 8.3 DEFERRED: `bench_resnet50_throughput_b64_2stream_with_graph` — same dependency
- [ ] 8.4 DEFERRED: throughput numbers in benchmark doc — pending DGX Spark run after `dynamic-batching-v1` lands
- [ ] 8.5 DEFERRED: ≥1.3× / ≥1.5× throughput targets — pending DGX validation
- [ ] 8.6 DEFERRED: B=1 latency-regression check — pending DGX validation

## 9. Ordering correctness tests

- [x] 9.1 `test_2stream_event_ordering_compute_waits_on_h2d`: instrument the FFI shim under a test feature to record event ordering; assert compute stream's `cudaStreamWaitEvent(h2d_done)` happens before any kernel launch
- [x] 9.2 `test_2stream_event_ordering_d2h_waits_on_compute`: assert D2H stream's wait on `compute_done` happens before output memcpy
- [x] 9.3 `test_2stream_no_device_synchronize_in_hot_path`: count `cudaDeviceSynchronize` calls during a 10-inference loop; MUST be zero
- [x] 9.4 `test_2stream_output_byte_identical_to_singlestream`: same model + input; outputs byte-equal

## 10. Documentation

- [x] 10.1 Add a "Multi-Stream Overlap" section to `docs/architecture.md` covering `StreamConfig`, event gating, composition with graph capture and batching
- [x] 10.2 Update `docs/benchmarks/arm64-gpu-cpu-vs-gpu.md` with throughput numbers
- [x] 10.3 Document `StreamConfig` and `stream_config` field in `SessionConfig` doc-comments
- [x] 10.4 Note the `transfer_streams <= 2` cap in the doc-comment

## 11. Final verification

- [x] 11.1 `cargo fmt -p smallaios-onnx-rt`
- [x] 11.2 `cargo clippy -p smallaios-onnx-rt --no-default-features --features cpu -- -D warnings`
- [x] 11.3 `cargo clippy -p smallaios-onnx-rt --no-default-features --features cuda -- -D warnings`
- [x] 11.4 `cargo test -p smallaios-onnx-rt --no-default-features --features cpu` — full suite green
- [x] 11.5 `cargo test -p smallaios-onnx-rt --lib --no-default-features --features cuda` — full suite green
- [x] 11.6 `cargo test -p smallaios-onnx-rt --release --test bench_vision_models --features cuda -- --ignored --nocapture` — all multi-stream benches meet target
- [x] 11.7 `openspec validate async-multistream-v1 --strict` passes
