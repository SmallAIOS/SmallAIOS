## ADDED Requirements

### Requirement: Pooling Attribute Parser Is the Single Source of Truth
The runtime SHALL expose `PoolAttrs::from_attributes(&[AttributeProto]) -> Result<PoolAttrs, OpError>` and call it from both the CPU dispatch path (`dispatch_node` for `MaxPool` / `AveragePool` / `GlobalAveragePool`) and the CUDA dispatch path. The CPU and CUDA paths SHALL NOT contain independent pool attribute parsers.

#### Scenario: CPU and CUDA pool dispatch parse attributes identically
- **WHEN** a `MaxPool` or `AveragePool` node is dispatched first on the CPU path, then with an identical `AttributeProto` list on the CUDA path
- **THEN** both dispatch sites MUST call `PoolAttrs::from_attributes`
- **AND** MUST produce the same `PoolAttrs` value

#### Scenario: Pool parser defaults match ONNX
- **WHEN** `PoolAttrs::from_attributes` receives an attribute list with no pool attributes
- **THEN** it MUST populate `strides = [1, 1]`, `pads = [0, 0, 0, 0]`, `ceil_mode = false`, `count_include_pad = false`
- **AND** MUST treat a missing `kernel_shape` as an error for `MaxPool` and `AveragePool` (required attribute)

#### Scenario: GlobalAveragePool requires no kernel_shape
- **WHEN** a `GlobalAveragePool` node is dispatched
- **THEN** the executor MUST derive the effective kernel from the input spatial dimensions instead of requiring `kernel_shape`
- **AND** MUST pass `strides = [1, 1]`, `pads = [0, 0, 0, 0]` to the underlying pool kernel

### Requirement: BatchNormalization Attribute Parser Is the Single Source of Truth
The runtime SHALL expose `BatchNormAttrs::from_attributes(&[AttributeProto]) -> Result<BatchNormAttrs, OpError>` carrying at minimum `epsilon: f32` and `momentum: f32`, with ONNX-default values (`epsilon = 1e-5`, `momentum = 0.9`). Both CPU and CUDA dispatch sites for `BatchNormalization` SHALL call this parser.

#### Scenario: BatchNorm parser returns defaults for empty attr list
- **WHEN** `BatchNormAttrs::from_attributes` receives an empty attribute list
- **THEN** it MUST return `epsilon = 1e-5` and `momentum = 0.9`

#### Scenario: BatchNorm parser populates epsilon and momentum
- **WHEN** a `BatchNormalization` node specifies `epsilon = 1e-3` and `momentum = 0.5`
- **THEN** `BatchNormAttrs::from_attributes` MUST return those values verbatim

## MODIFIED Requirements

### Requirement: Node Attribute Propagation
The ONNX runtime SHALL propagate operator attributes from the ONNX protobuf `NodeProto` through to operator dispatch. For the Conv, BatchNormalization, pooling (`MaxPool` / `AveragePool` / `GlobalAveragePool`), and activation (`Relu` / `Clip` / `LeakyRelu`) operators, the runtime SHALL parse attributes once via a shared parser (`ConvAttrs::from_attributes`, `BatchNormAttrs::from_attributes`, `PoolAttrs::from_attributes`, or direct attribute lookup for activations) and pass the parsed result to both the CPU and CUDA dispatch paths. Unknown or unsupported attribute values SHALL surface as `OpError::InvalidAttribute` rather than being silently dropped.

#### Scenario: Conv operator receives padding, stride, dilation, and group attributes
- **WHEN** a Conv node specifies any combination of `pads`, `strides`, `dilations`, and `group` attributes
- **THEN** the executor MUST construct a `ConvAttrs` value via `ConvAttrs::from_attributes`
- **AND** MUST pass this `ConvAttrs` to the `op_conv` function (CPU dispatch) or `cuda::conv::gpu_conv2d` (CUDA dispatch)
- **AND** the convolution output spatial dimensions MUST match the ONNX Conv-11 formula `out = (in + pad_begin + pad_end - (kernel - 1) * dilation - 1) / stride + 1`

#### Scenario: BatchNormalization operator receives epsilon and momentum attributes
- **WHEN** a `BatchNormalization` node specifies `epsilon` or `momentum`
- **THEN** the executor MUST construct a `BatchNormAttrs` value via `BatchNormAttrs::from_attributes`
- **AND** MUST pass this `BatchNormAttrs` to both the CPU and CUDA dispatch paths

#### Scenario: Pool operators receive kernel_shape / pads / strides attributes
- **WHEN** a `MaxPool` or `AveragePool` node specifies `kernel_shape`, `pads`, or `strides`
- **THEN** the executor MUST construct a `PoolAttrs` value via `PoolAttrs::from_attributes`
- **AND** MUST pass this `PoolAttrs` to both the CPU and CUDA dispatch paths

#### Scenario: Softmax operator receives axis attribute
- **WHEN** a Softmax node specifies an `axis` attribute
- **THEN** the executor MUST pass the axis value to `op_softmax`
- **AND** softmax normalization MUST be computed along the specified axis

#### Scenario: Unsupported attribute value surfaces a loud error
- **WHEN** a Conv, BatchNormalization, or pooling node carries an attribute value the runtime does not yet support
- **THEN** the attribute parser MUST return `OpError::InvalidAttribute` with the offending attribute name and value
- **AND** MUST NOT silently fall back to a default
