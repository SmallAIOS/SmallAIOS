## ADDED Requirements

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
