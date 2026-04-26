## ADDED Requirements

### Requirement: CUDA Graph Capture Mode
The runtime SHALL support a CUDA graph capture mode, selectable via `SessionConfig::cuda_graph = CudaGraphMode::Capture`, in which the first hybrid inference for a given input shape is captured into a `cudaGraph_t` and subsequent inferences with matching shape replay the cached graph via a single `cudaGraphLaunch`. Default behavior (`CudaGraphMode::Off`) SHALL be byte-for-byte identical to the current per-op hybrid execution path.

#### Scenario: First inference captures the graph
- **WHEN** a session is configured with `gpu_residency = GpuResidency::Hybrid` and `cuda_graph = CudaGraphMode::Capture` and `Session::run` is called for the first time
- **THEN** the runtime MUST allocate a per-Session capture stream via `cudaStreamCreate`
- **AND** MUST bind the cuDNN and cuBLAS handles to the capture stream via `cudnnSetStream` and `cublasSetStream_v2` for the duration of capture
- **AND** MUST wrap the existing hybrid op-dispatch loop in `cudaStreamBeginCapture(stream, CUDA_STREAM_CAPTURE_MODE_THREAD_LOCAL)` and `cudaStreamEndCapture(stream, &graph)`
- **AND** MUST instantiate a `cudaGraphExec_t` via `cudaGraphInstantiate`
- **AND** MUST cache the resulting `GraphEntry` in the per-Session `CudaGraphCache`, keyed by `(input shapes, input dtypes)`

#### Scenario: Subsequent inferences replay the cached graph
- **WHEN** a subsequent `Session::run` call has input shapes and dtypes matching a cached `GraphEntry`
- **THEN** the runtime MUST skip the per-op dispatch loop
- **AND** MUST copy the user's input host tensors into the cache's pre-allocated input device buffers via `cudaMemcpyAsync` on the capture stream
- **AND** MUST call `cudaGraphLaunch(graph_exec, capture_stream)` exactly once
- **AND** MUST copy the cache's pre-allocated output device buffers back to host tensors via `cudaMemcpyAsync` on the capture stream
- **AND** MUST synchronize the stream before returning to the caller

#### Scenario: Shape change triggers graph rebuild
- **WHEN** `Session::run` is called with input shapes that do not match any cached `GraphEntry`
- **THEN** the runtime MUST treat the call as a fresh capture (capture + instantiate a new graph for the new shape key)
- **AND** MUST keep prior `GraphEntry` values cached (multiple shapes can coexist)
- **AND** MUST increment the per-Session `rebuild_count`

#### Scenario: Capture aborts on CPU fallback during warm-up
- **WHEN** during graph capture an operator falls back to the CPU path (for example because its dtype is not GPU-supported, or the op kind is not in `gpu_op_supported`)
- **THEN** the runtime MUST abort the in-progress capture
- **AND** MUST log a single warning message including the offending op kind
- **AND** MUST complete the warm-up inference via the per-op hybrid path
- **AND** MUST set `CudaGraphCache::disabled = true` for the remainder of the Session
- **AND** MUST NOT propagate the capture failure as an error from `Session::run`

#### Scenario: Replay failure falls back without crashing
- **WHEN** `cudaGraphLaunch` returns a non-success status (for example CUDA OOM)
- **THEN** the runtime MUST discard the failing `GraphEntry` from the cache
- **AND** MUST attempt a fresh capture on the next inference
- **AND** MUST fall back to per-op execution for the current inference so the call still produces a valid output

#### Scenario: Excessive rebuilds disable capture for the Session
- **WHEN** the per-Session `rebuild_count` exceeds 1% of `inference_count` after at least 32 inferences
- **THEN** the runtime MUST set `CudaGraphCache::disabled = true`
- **AND** MUST log a single informational message indicating the workload is not graph-capture-friendly
- **AND** all subsequent inferences MUST run on the per-op hybrid path

#### Scenario: Output equivalence between capture and per-op modes
- **WHEN** the same model is run on the same inputs once with `CudaGraphMode::Off` and once with `CudaGraphMode::Capture`
- **THEN** the two output tensors MUST have identical shapes
- **AND** the element-wise `max_abs_diff` MUST be less than `1e-4`
- **AND** any divergence beyond `1e-4` MUST cause the capture-mode test to fail

#### Scenario: Default mode preserves existing behavior
- **WHEN** `SessionConfig::cuda_graph` is not set (defaults to `CudaGraphMode::Off`)
- **THEN** the runtime MUST run the existing per-op hybrid (or op-by-op) path with no capture stream allocation, no `cudnnSetStream` rebinding, and no graph cache state

### Requirement: CUDA Graph Capture Latency Reduction
On DGX Spark, the runtime SHALL deliver a measurable latency reduction when `CudaGraphMode::Capture` is enabled on a session that already uses `GpuResidency::Hybrid`. Specifically the median end-to-end inference latency for ResNet-50 v2 SHALL drop by at least 1.5× relative to the hybrid-no-capture baseline.

#### Scenario: ResNet-50 capture-mode benchmark hits target
- **WHEN** the `bench_resnet50_cpu_vs_gpu_hybrid_with_graph` benchmark is run on DGX Spark
- **THEN** the GPU mean latency MUST be at least 1.5× lower than the same benchmark without graph capture
- **AND** the reported `max_abs_diff` versus the CPU reference MUST remain below `1e-2`

#### Scenario: Smaller models also improve
- **WHEN** the MLP / SqueezeNet / MobileNetV2 capture-mode benches are run
- **THEN** each MUST report a GPU mean latency at least 1.2× lower than its hybrid baseline

### Requirement: Graph Cache Lifetime Tied to Session
The per-Session graph cache (`CudaGraphCache`) SHALL own all `cudaGraph_t` and `cudaGraphExec_t` handles, all pinned device input/output buffers, and the capture stream. The cache SHALL release these resources via `Drop` impls when the Session is dropped, and MUST NOT leak GPU memory across Session lifetimes.

#### Scenario: Session drop releases graph resources
- **WHEN** a `Session` configured with `CudaGraphMode::Capture` is dropped after at least one inference
- **THEN** all cached `cudaGraphExec_t` handles MUST be released via `cudaGraphExecDestroy`
- **AND** all cached `cudaGraph_t` handles MUST be released via `cudaGraphDestroy`
- **AND** the capture stream MUST be released via `cudaStreamDestroy`
- **AND** all pinned input/output `DeviceBuffer`s MUST be freed via `cudaFree`
