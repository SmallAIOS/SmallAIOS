# onnx-runtime Specification

## Purpose
TBD - created by archiving change smallaios-kernel-v1. Update Purpose after archive.
## Requirements
### Requirement: Model Loading via Protobuf Parsing
The ONNX runtime SHALL parse ONNX model files using a minimal protobuf parser code-generated from the onnx.proto3 schema.

#### Scenario: Load a valid ONNX model
- WHEN the runtime receives a valid .onnx protobuf-serialized model file
- THEN it MUST parse the ModelProto structure including graph, nodes, initializers, and metadata
- AND MUST support ONNX IR version 10 (ONNX 1.16+) and opset version 21

#### Scenario: Reject a malformed protobuf
- WHEN the runtime receives a corrupted or truncated protobuf file
- THEN it MUST return a descriptive OnnxError without panicking
- AND MUST NOT allocate unbounded memory during parsing

### Requirement: Model Validation
The runtime SHALL validate loaded models before execution to ensure all operators, shapes, and data types are supported.

#### Scenario: Validate operator support
- WHEN a model contains only Tier 1 operators (MatMul, Conv, Relu, Softmax, etc.)
- THEN validation MUST succeed
- AND the runtime MUST report the list of operators used

#### Scenario: Reject unsupported opset version
- WHEN a model specifies an opset version greater than 21
- THEN validation MUST fail with an UnsupportedOpset error
- AND MUST report the requested vs. supported opset versions

### Requirement: Graph Optimization — Operator Fusion
The runtime SHALL implement operator fusion passes to combine sequences of operators into single fused kernels.

#### Scenario: Fuse Conv-BatchNorm-Relu
- WHEN the graph contains a Conv node followed by BatchNormalization followed by Relu
- THEN the optimizer MUST fuse them into a single FusedConvBNRelu node
- AND the fused node MUST produce numerically equivalent output (within f32 epsilon)

#### Scenario: Fuse MatMul-Add into FusedLinear
- WHEN the graph contains a MatMul node followed by an Add with a constant bias
- THEN the optimizer MUST fuse them into a single FusedLinear (GEMM) node

### Requirement: Graph Optimization — Constant Folding
The runtime SHALL pre-compute subgraphs composed entirely of constant inputs at model load time.

#### Scenario: Fold static reshape
- WHEN a Reshape operator has all-constant inputs (data and shape)
- THEN the optimizer MUST evaluate the operator at load time
- AND replace it with the computed constant tensor in the graph

### Requirement: Graph Optimization — Memory Planning
The runtime SHALL analyze tensor lifetimes and plan buffer reuse to minimize peak memory consumption.

#### Scenario: Reuse dead tensor buffers
- WHEN tensor A's lifetime ends at operator 3 and tensor C begins at operator 4
- THEN the memory planner MUST assign tensor C to the same buffer as tensor A
- AND peak memory MUST be reduced compared to naive allocation

### Requirement: CPU Execution Provider — x86-64 SIMD
The CPU execution provider SHALL use runtime CPUID detection to select optimal SIMD kernels for x86-64.

#### Scenario: Select AVX-512 GEMM on capable hardware
- WHEN the CPU supports AVX-512F and AVX-512VL
- THEN the execution provider MUST select the AVX-512 GEMM kernel for MatMul operations
- AND MUST fall back to AVX2 kernels if AVX-512 is unavailable

#### Scenario: Execute quantized inference with VNNI
- WHEN the CPU supports AVX-512 VNNI and the model uses INT8 quantized operators
- THEN the execution provider MUST use VNNI dot-product instructions for INT8 MatMul

### Requirement: CPU Execution Provider — ARM64 SIMD
The CPU execution provider SHALL use runtime feature detection to select optimal SIMD kernels for ARM64.

#### Scenario: Select SVE kernels on capable hardware
- WHEN the CPU supports SVE/SVE2 extensions
- THEN the execution provider MUST select SVE GEMM kernels for MatMul operations
- AND MUST fall back to NEON kernels if SVE is unavailable

### Requirement: CUDA Execution Provider
The runtime SHALL support GPU inference via a CUDA execution provider that launches custom PTX kernels and cuDNN/cuBLAS primitives on NVIDIA GPUs. When the provider is enabled, operators whose input dtype is in the GPU-supported set SHALL be eligible for GPU dispatch; when `SessionConfig::gpu_residency = GpuResidency::Hybrid` the runtime SHALL additionally keep intermediate tensors device-resident across adjacent GPU-eligible operators rather than memcpying each intermediate back to host.

#### Scenario: GPU kernel launch for MatMul
- WHEN an inference session is configured with the CUDA execution provider
- AND the graph contains a MatMul operator
- THEN the runtime MUST launch a GPU kernel using tensor core HMMA instructions for fp16/bf16
- AND MUST use the GPU memory pool for intermediate tensor allocation

#### Scenario: Async DMA transfers
- WHEN input tensors reside in host memory and the session uses GPU execution
- THEN the runtime MUST transfer inputs to GPU via async DMA
- AND MUST overlap DMA transfers with computation where possible

#### Scenario: Hybrid residency benchmark speedup
- WHEN `SessionConfig::gpu_residency = GpuResidency::Hybrid` is set and the ResNet-50 v2 vision benchmark is run on DGX Spark
- THEN the CPU-vs-GPU mean-latency speedup MUST be at least 5×
- AND the GPU output MUST match the CPU reference within `max_abs_diff < 1e-2`

### Requirement: Device-Resident Hybrid Execution Mode
The runtime SHALL support a GPU-resident hybrid execution mode, selectable via `SessionConfig::gpu_residency = GpuResidency::Hybrid`, in which intermediate tensors produced by GPU-supported operators remain in device memory across consecutive GPU-supported operators. The runtime SHALL track the residency of every named graph value (host or device) and SHALL insert host↔device memcpys only at boundaries where residency must change — i.e. when a GPU op consumes a host-resident input, when a CPU op consumes a device-resident input, and when a graph output is produced on device.

#### Scenario: Adjacent GPU ops reuse device buffers
- **WHEN** a session is configured with `GpuResidency::Hybrid` and the graph contains two adjacent GPU-supported operators (e.g. `Conv → BatchNormalization`) whose input and output dtypes are both GPU-eligible
- **THEN** the runtime MUST dispatch both operators to GPU
- **AND** the intermediate tensor produced by the first operator MUST remain on device for the duration of the second operator's execution
- **AND** the runtime MUST NOT perform a device→host memcpy on the intermediate tensor

#### Scenario: CPU op between two GPU ops copies back correctly
- **WHEN** a GPU-supported operator produces a device-resident tensor that feeds a CPU-only operator whose output is consumed by another GPU-supported operator
- **THEN** the runtime MUST copy the device-resident intermediate to host before invoking the CPU operator
- **AND** the CPU operator's output MUST be eligible for host→device copy when the next GPU operator needs it
- **AND** the hybrid executor MUST NOT degrade to all-CPU execution after the first CPU op — subsequent GPU-eligible ops MUST still dispatch to GPU

#### Scenario: Graph output copied device→host
- **WHEN** a graph output value is produced by a GPU-supported operator in hybrid mode
- **THEN** the runtime MUST copy the output tensor device→host before returning it to the caller
- **AND** the returned `Tensor` MUST have its canonical data on host

#### Scenario: Default residency is unchanged
- **WHEN** a session is constructed without explicitly setting `SessionConfig::gpu_residency`
- **THEN** the runtime MUST use `GpuResidency::OpByOp`
- **AND** the execution behavior MUST be byte-for-byte identical to the pre-change op-by-op dispatch path

#### Scenario: Hybrid mode gracefully handles unsupported op mid-graph
- **WHEN** hybrid mode is active and an operator has no GPU implementation for its input dtype or shape
- **THEN** the runtime MUST copy all device-resident inputs of that operator back to host
- **AND** MUST execute the operator on the CPU path
- **AND** MUST NOT return an error for the unsupported-on-GPU condition itself (the op executed correctly, just on CPU)

### Requirement: CUDA BatchNormalization Support
The CUDA execution provider SHALL support the ONNX `BatchNormalization` operator in inference mode via `cudnnBatchNormalizationForwardInference`. The implementation SHALL operate on `DeviceTensor` inputs (input X plus four per-channel parameter tensors — scale γ, bias β, running mean μ, running variance σ²) and SHALL produce a `DeviceTensor` output of the same shape and dtype as X.

#### Scenario: BatchNorm dispatches correctly in hybrid mode
- **WHEN** hybrid mode is active and a `BatchNormalization` node has a device-resident input X and device-resident or initializer-sourced parameter tensors
- **THEN** the runtime MUST call `cudnnBatchNormalizationForwardInference` with mode `CUDNN_BATCHNORM_SPATIAL`
- **AND** MUST pass the epsilon attribute (default `1e-5`) to the cuDNN call
- **AND** MUST produce a device-resident output tensor that the next GPU operator can consume without memcpy

#### Scenario: BatchNorm GPU output matches CPU output
- **WHEN** the same `BatchNormalization` node is executed on CPU and on GPU with identical inputs and parameters
- **THEN** the element-wise `max_abs_diff` MUST be less than `1e-3` under the runtime's default TF32 precision mode

#### Scenario: BatchNorm with non-float dtype falls back
- **WHEN** a `BatchNormalization` node has an input dtype the GPU path does not support (e.g. int32)
- **THEN** the hybrid executor MUST route the operator to the CPU path
- **AND** MUST NOT return a GPU dispatch error

### Requirement: CUDA Activation Support (Relu, Clip, LeakyRelu)
The CUDA execution provider SHALL support the ONNX `Relu`, `Clip`, and `LeakyRelu` operators via a single cuDNN `cudnnActivationForward` code path, selecting the appropriate `cudnnActivationMode_t` per operator. The implementation SHALL operate on `DeviceTensor` inputs and produce a `DeviceTensor` output of the same shape and dtype as the input.

#### Scenario: Relu dispatches in hybrid mode
- **WHEN** hybrid mode is active and a `Relu` node has a device-resident input
- **THEN** the runtime MUST call `cudnnActivationForward` with mode `CUDNN_ACTIVATION_RELU`
- **AND** MUST produce a device-resident output tensor

#### Scenario: Clip uses CLIPPED_RELU with correct bounds
- **WHEN** a `Clip` node specifies `min = 0` and a positive `max`
- **THEN** the runtime MUST call `cudnnActivationForward` with mode `CUDNN_ACTIVATION_CLIPPED_RELU` and the `max` value as the coefficient
- **AND** the result MUST match the CPU path within `max_abs_diff < 1e-3`

#### Scenario: Unsupported Clip bounds fall back
- **WHEN** a `Clip` node specifies bounds that the cuDNN activation modes cannot express (for example a negative lower bound)
- **THEN** the hybrid executor MUST route the operator to the CPU path

#### Scenario: LeakyRelu alpha honored
- **WHEN** a `LeakyRelu` node specifies `alpha`
- **THEN** the runtime MUST call `cudnnActivationForward` with mode `CUDNN_ACTIVATION_ELU` or an equivalent signal-preserving mode with `alpha` as the coefficient, OR fall back to CPU if no cuDNN mode matches
- **AND** MUST NOT silently use a different activation function

### Requirement: CUDA Pooling Support (MaxPool, AveragePool, GlobalAveragePool)
The CUDA execution provider SHALL support the ONNX `MaxPool`, `AveragePool`, and `GlobalAveragePool` operators via `cudnnPoolingForward`. `GlobalAveragePool` SHALL be implemented as an `AveragePool` with `kernel_shape` equal to the spatial input dimensions. The implementation SHALL share attribute parsing with the CPU path via a new `PoolAttrs` type.

#### Scenario: MaxPool with explicit attributes dispatches to GPU
- **WHEN** hybrid mode is active and a `MaxPool` node has a device-resident input with `kernel_shape = [3, 3]`, `strides = [2, 2]`, and explicit `pads`
- **THEN** the runtime MUST construct a `cudnnPoolingDescriptor_t` with mode `CUDNN_POOLING_MAX`
- **AND** MUST call `cudnnPoolingForward` with the device input and output descriptors
- **AND** the resulting output shape MUST match the ONNX pooling output formula

#### Scenario: GlobalAveragePool reduces to spatial dims
- **WHEN** hybrid mode is active and a `GlobalAveragePool` node has a device-resident input of shape `[N, C, H, W]`
- **THEN** the runtime MUST call `cudnnPoolingForward` with `CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING`, `kernel_shape = [H, W]`, and stride `[1, 1]`
- **AND** MUST produce a device-resident output of shape `[N, C, 1, 1]`

#### Scenario: Pool GPU output matches CPU output
- **WHEN** a pooling operator is executed on both paths with identical inputs
- **THEN** `max_abs_diff` MUST be less than `1e-3` under default precision

### Requirement: CUDA Broadcast Add
The CUDA execution provider SHALL support the ONNX `Add` operator for the residual-connection pattern where both operands share the same shape, via `cudnnOpTensor(OP_TENSOR_ADD, …)` or an equivalent device-side element-wise add. The implementation SHALL operate on `DeviceTensor` inputs and produce a `DeviceTensor` output.

#### Scenario: Same-shape Add dispatches to GPU in hybrid mode
- **WHEN** hybrid mode is active and an `Add` node has two device-resident inputs of identical shape and dtype
- **THEN** the runtime MUST dispatch the operator to GPU
- **AND** MUST produce a device-resident output of the same shape

#### Scenario: Unsupported Add broadcast patterns fall back
- **WHEN** an `Add` node has inputs whose shapes require a broadcast pattern cuDNN `OpTensor` does not support
- **THEN** the hybrid executor MUST route the operator to the CPU path

#### Scenario: Add GPU output matches CPU output
- **WHEN** a same-shape `Add` is executed on both paths
- **THEN** the element-wise `max_abs_diff` MUST be less than `1e-3` under default precision

### Requirement: CUDA Grouped Convolution Support
The CUDA execution provider SHALL support grouped convolutions (including depthwise, where `group == input_channels`) by forwarding the `group` attribute from the dispatched ONNX Conv node to cuDNN. The forwarding SHALL call `cudnnSetConvolutionGroupCount` on the convolution descriptor after creation and before algorithm selection; when `group == 1` the runtime MAY skip the call. The `group` value SHALL come from the shared `ConvAttrs::from_attributes` parser so the CPU and CUDA dispatch paths agree on attribute semantics.

#### Scenario: Depthwise Conv dispatches correctly to cuDNN
- **WHEN** a Conv node has input shape `[1, C, H, W]`, weight shape `[C, 1, KH, KW]`, and `group = C`
- **AND** the session is configured with the CUDA execution provider
- **THEN** `try_cuda_dispatch` MUST parse `group` via `ConvAttrs::from_attributes`
- **AND** MUST forward `group` to `cuda::conv::gpu_conv2d`
- **AND** `gpu_conv2d` MUST call `cudnnSetConvolutionGroupCount(conv_desc, group)` after `cudnnCreateConvolutionDescriptor` and before the algorithm / workspace query
- **AND** the resulting output shape MUST be `[1, C, OH, OW]`

#### Scenario: Plain group=1 convolution takes the unchanged fast path
- **WHEN** a Conv node has `group = 1` (the ONNX default)
- **AND** the session is configured with the CUDA execution provider
- **THEN** the runtime MAY skip the `cudnnSetConvolutionGroupCount` call
- **AND** the resulting byte-for-byte output MUST match the pre-change behavior of `gpu_conv2d`

#### Scenario: Grouped conv uses a correctly-sized cuDNN workspace
- **WHEN** cuDNN selects a convolution algorithm for a grouped Conv and reports a workspace size via `cudnnGetConvolutionForwardWorkspaceSize`
- **THEN** the runtime MUST allocate a device-side workspace of at least that many bytes
- **AND** MUST pass the workspace pointer and size to `cudnnConvolutionForward`
- **AND** MUST reuse or release the workspace cleanly across inferences without leaking device memory

#### Scenario: Grouped conv output matches CPU output
- **WHEN** the same Conv node is dispatched first through `op_conv` (CPU) and then through `gpu_conv2d` (CUDA) with identical inputs, weights, biases, and `ConvAttrs`
- **THEN** the two output tensors MUST have identical shapes
- **AND** the element-wise `max_abs_diff` MUST be less than `1e-3` when the CUDA runtime is configured in its default TF32 precision mode

### Requirement: Operator-Level Scheduler Integration
The runtime SHALL insert mandatory scheduler yield points between every operator in the execution graph and support per-operator time budgets.

#### Scenario: Yield between operators
- WHEN the runtime executes an inference graph
- THEN it MUST yield to the scheduler after each operator completes
- AND the yield MUST allow higher-priority tasks (SYSTEM, IPC) to execute before inference resumes

#### Scenario: Per-operator timing
- WHEN an operator executes during inference
- THEN the runtime MUST measure its wall-clock execution time
- AND MUST compare the measured time against the operator's configured budget

#### Scenario: Operator budget exceeded
- WHEN an operator's execution time exceeds its soft budget
- THEN the runtime MUST log a warning with operator name, actual time, and budget to syslog
- AND MUST continue inference normally

#### Scenario: Operator hard timeout
- WHEN an operator's execution time exceeds its hard limit (default: 10x budget)
- THEN the runtime MUST abort the inference and return OnnxError::OperatorTimeout

### Requirement: WCET Calibration
The runtime SHALL support optional worst-case execution time calibration during session creation for edge deployment targets.

#### Scenario: Calibration run during session creation
- WHEN a session is created with calibrate_wcet enabled
- THEN the runtime MUST execute each operator once with representative input data
- AND MUST compute WCET estimates as measured_time × wcet_safety_factor
- AND MUST assign calibrated estimates as operator budgets for the session

#### Scenario: Calibration on constrained hardware
- WHEN calibration runs on a constrained target (Jetson Nano, RPi)
- THEN the recommended wcet_safety_factor MUST be >= 3.0 to account for thermal throttling and clock variability

### Requirement: Session API
The runtime SHALL expose a Session API with load, create_session, run, and metadata operations.

#### Scenario: Create and run an inference session
- WHEN a client calls load_model with valid ONNX bytes followed by create_session
- THEN the runtime MUST return a ready Session handle
- AND calling run with correctly shaped input tensors MUST return output tensors matching the model's output specification

#### Scenario: Query model metadata
- WHEN a client calls get_metadata on a loaded model
- THEN the runtime MUST return the model's input names/shapes, output names/shapes, opset version, and producer name

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

