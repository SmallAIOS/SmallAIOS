## ADDED Requirements

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

## MODIFIED Requirements

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
