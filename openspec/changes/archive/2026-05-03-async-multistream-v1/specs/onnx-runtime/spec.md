## ADDED Requirements

### Requirement: Multi-Stream Overlap Mode
The runtime SHALL support a multi-stream execution mode, selectable via `SessionConfig::stream_config = StreamConfig::Overlap { transfer_streams }`, in which host↔device memcpys run on dedicated transfer streams while compute runs on a separate dedicated stream. Cross-stream dependencies SHALL be expressed via CUDA events (`cudaEventRecord` + `cudaStreamWaitEvent`) — never via stream / device synchronization mid-pipeline. Default behavior (`StreamConfig::SingleStream`) MUST be byte-for-byte identical to the pre-`async-multistream-v1` execution.

#### Scenario: Default mode preserves existing behavior
- **WHEN** `SessionConfig::stream_config` is not set (defaults to `StreamConfig::SingleStream`)
- **THEN** all CUDA work MUST run on a single stream as before
- **AND** no multi-stream resources MUST be allocated

#### Scenario: Overlap mode allocates a stream pool
- **WHEN** `StreamConfig::Overlap { transfer_streams: 1 }` is configured
- **THEN** `Session::new` (or first `run()` call, lazy) MUST allocate one compute stream, one H2D stream, and one D2H stream
- **AND** MUST allocate a pool of CUDA events sufficient to gate every cross-stream dependency (at least 2 events per inference)
- **AND** all stream and event resources MUST be released via `Drop` when the Session is dropped

#### Scenario: Input H2D overlaps with prior inference compute
- **WHEN** `Session::run` is called twice in sequence on a session with `Overlap` mode
- **THEN** the second call's input H2D memcpy MUST be issued on the H2D transfer stream
- **AND** the H2D memcpy MUST NOT block on the first call's compute stream completing
- **AND** the second call's first kernel launch MUST wait on the H2D-done event before reading the input buffer

#### Scenario: Output D2H overlaps with subsequent compute
- **WHEN** the current inference finishes its compute and a subsequent inference begins
- **THEN** the current call's output D2H memcpy MUST be issued on the D2H transfer stream
- **AND** the D2H memcpy MUST be gated on a `compute_done` event recorded on the compute stream
- **AND** the host MUST wait via `cudaStreamSynchronize(d2h_stream)` before returning, NOT on the compute stream

#### Scenario: Output is byte-for-byte identical to single-stream
- **WHEN** the same model is run on the same input once with `StreamConfig::SingleStream` and once with `StreamConfig::Overlap`
- **THEN** the two output tensors MUST have identical shapes and identical raw bytes
- **AND** any divergence MUST cause the multi-stream test to fail

#### Scenario: cuDNN handles rebound around overlap inference
- **WHEN** a multi-stream inference runs
- **THEN** cuDNN and cuBLAS handles MUST be bound to the compute stream via `cudnnSetStream` / `cublasSetStream_v2` for the duration of the inference
- **AND** MUST be reset to the default stream after the inference completes
- **AND** a subsequent single-stream inference on the same Session (after the user changes `stream_config`) MUST work correctly without leaked stream binding

### Requirement: Multi-Stream Throughput Target
On DGX Spark with `GpuResidency::Hybrid` and `BatchPolicy::Static(64)` (composing with `dynamic-batching-v1`), enabling `StreamConfig::Overlap { transfer_streams: 1 }` SHALL deliver a measurable throughput improvement over the single-stream baseline.

#### Scenario: 2-stream B=64 hits target
- **WHEN** the `bench_resnet50_throughput_b64_2stream` benchmark is run
- **THEN** the measured images-per-second MUST be at least 1.3× the `bench_resnet50_throughput_b64_singlestream` baseline

#### Scenario: 2-stream + graph capture composes correctly
- **WHEN** the `bench_resnet50_throughput_b64_2stream_with_graph` benchmark is run (combining `async-multistream-v1` + `cuda-graphs-v1`)
- **THEN** the measured images-per-second MUST be at least 1.5× the single-stream + graph baseline
- **AND** the captured graph MUST run on the compute stream while transfers happen on the H2D/D2H streams

#### Scenario: Single-request latency does not regress in Overlap mode
- **WHEN** `Session::run` is called on a session with `StreamConfig::Overlap` and a single input
- **THEN** the measured latency MUST NOT exceed the `StreamConfig::SingleStream` latency by more than 5%

### Requirement: Cross-Stream Dependency Safety
The runtime SHALL guarantee that no GPU operator on any stream reads a buffer until every prior write to that buffer has completed, via explicit CUDA event dependencies. There SHALL be no use of `cudaDeviceSynchronize` or device-wide barrier calls in the inference hot path.

#### Scenario: H2D-then-compute uses event gating
- **WHEN** the runtime issues an `cudaMemcpyAsync(h2d_stream, ...)` for an input
- **THEN** the runtime MUST call `cudaEventRecord(h2d_done_event, h2d_stream)` immediately after the memcpy
- **AND** MUST call `cudaStreamWaitEvent(compute_stream, h2d_done_event, 0)` before any kernel that consumes the input

#### Scenario: Compute-then-D2H uses event gating
- **WHEN** the runtime finishes the last kernel of an inference on the compute stream
- **THEN** the runtime MUST call `cudaEventRecord(compute_done_event, compute_stream)`
- **AND** MUST call `cudaStreamWaitEvent(d2h_stream, compute_done_event, 0)` before issuing the output `cudaMemcpyAsync(d2h_stream, ...)`

#### Scenario: No device-wide synchronization in the inner loop
- **WHEN** the multi-stream pipeline runs an inference
- **THEN** the runtime MUST NOT call `cudaDeviceSynchronize` between the input H2D and the output D2H
- **AND** the only synchronization permitted is `cudaStreamSynchronize(d2h_stream)` at the end (to wait for the output to land on host)

### Requirement: Transfer Stream Count Cap
The runtime SHALL cap `StreamConfig::Overlap::transfer_streams` at 2 (one H2D + one D2H per "stream" — so the cap is 2 total slots). Configurations with `transfer_streams > 2` SHALL return `SessionError::InvalidConfig` at session creation.

#### Scenario: transfer_streams = 1 is accepted
- **WHEN** `StreamConfig::Overlap { transfer_streams: 1 }` is configured
- **THEN** the session MUST initialize one H2D stream and one D2H stream

#### Scenario: transfer_streams = 2 is accepted
- **WHEN** `StreamConfig::Overlap { transfer_streams: 2 }` is configured
- **THEN** the session MUST initialize two H2D streams and two D2H streams (parallel-overlap)

#### Scenario: transfer_streams = 3 is rejected
- **WHEN** `StreamConfig::Overlap { transfer_streams: 3 }` is configured
- **THEN** `Session::new` (or the first `run()` call) MUST return `SessionError::InvalidConfig` with a message naming the cap
