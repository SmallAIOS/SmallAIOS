## MODIFIED Requirements

### Requirement: Node Attribute Propagation
The ONNX runtime SHALL propagate operator attributes from the ONNX protobuf `NodeProto` through to operator dispatch. For the Conv operator specifically, the runtime SHALL parse `strides`, `pads`, `dilations`, and `group` with ONNX-default fallbacks (`strides = [1, 1]`, `pads = [0, 0, 0, 0]`, `dilations = [1, 1]`, `group = 1`), expose them to both the CPU and CUDA dispatch paths via a single shared `ConvAttrs` value, and honor every attribute in the executed kernel. Unknown or unsupported attribute values (for example an `auto_pad` mode the runtime does not handle) SHALL surface as `OpError::InvalidAttribute` rather than being silently dropped.

#### Scenario: Conv operator receives padding, stride, dilation, and group attributes
- **WHEN** a Conv node specifies any combination of `pads`, `strides`, `dilations`, and `group` attributes
- **THEN** the executor MUST construct a `ConvAttrs` value via `ConvAttrs::from_attributes`
- **AND** MUST pass this `ConvAttrs` to the `op_conv` function (CPU dispatch) or `cuda::conv::gpu_conv2d` (CUDA dispatch)
- **AND** the convolution output spatial dimensions MUST match the ONNX Conv-11 formula `out = (in + pad_begin + pad_end - (kernel - 1) * dilation - 1) / stride + 1`
- **AND** when `group > 1`, output channel `c_out` in group `g` MUST read only from input channels in the range `[g * C_in/group, (g + 1) * C_in/group)`

#### Scenario: Depthwise convolution with group = input channels
- **WHEN** a Conv node has input shape `[1, C, H, W]`, weight shape `[C, 1, KH, KW]`, and `group = C`
- **THEN** `validate_conv_inputs` MUST accept the input-weight channel relationship because `input.dims[1] == weight.dims[1] * group`
- **AND** `op_conv` (CPU) and `gpu_conv2d` (GPU) MUST each produce output shape `[1, C, OH, OW]`
- **AND** the CPU and GPU outputs MUST agree element-wise within `max_abs_diff < 1e-3`

#### Scenario: Strided convolution reduces spatial dimensions
- **WHEN** a Conv node has `strides = [2, 2]` and otherwise default attributes on a `[1, C, 224, 224]` input with a `3×3` kernel and `pads = [1, 1, 1, 1]`
- **THEN** the output shape MUST be `[1, C_out, 112, 112]`
- **AND** this output MUST be usable as input to downstream operators without shape-mismatch errors

#### Scenario: Attribute defaults preserve legacy call sites
- **WHEN** a Conv node specifies no attributes (or only `kernel_shape`)
- **THEN** `ConvAttrs::from_attributes` MUST produce `group = 1`, `strides = [1, 1]`, `pads = [0, 0, 0, 0]`, `dilations = [1, 1]`
- **AND** the resulting output tensor MUST be byte-identical to the pre-change behavior of `op_conv`

#### Scenario: Unsupported attribute surfaces a loud error
- **WHEN** a Conv node carries an attribute the runtime does not yet support (for example `auto_pad = "SAME_UPPER"` when explicit `pads` is expected)
- **THEN** `ConvAttrs::from_attributes` or the operator MUST return `OpError::InvalidAttribute` including the offending attribute name and value
- **AND** MUST NOT silently fall back to a default

#### Scenario: Softmax operator receives axis attribute
- **WHEN** a Softmax node specifies an `axis` attribute
- **THEN** the executor MUST pass the axis value to `op_softmax`
- **AND** softmax normalization MUST be computed along the specified axis

## ADDED Requirements

### Requirement: Conv Weight Shape Validation Honors Group
The CPU `validate_conv_inputs` check SHALL verify that `input.dims[1] == weight.dims[1] * group` and that `C_out % group == 0`. It SHALL NOT compare `input.dims[1]` directly to `weight.dims[1]`. Violations SHALL return `OpError::ShapeMismatch` with a message naming the three offending dimensions.

#### Scenario: Depthwise weight passes validation
- **WHEN** `input.shape = [1, 32, 56, 56]`, `weight.shape = [32, 1, 3, 3]`, and `group = 32`
- **THEN** `validate_conv_inputs` MUST succeed
- **AND** `op_conv` MUST proceed to the kernel invocation

#### Scenario: Grouped weight with non-divisible C_out is rejected
- **WHEN** `input.shape = [1, 16, 8, 8]`, `weight.shape = [15, 8, 3, 3]`, and `group = 2` (so `C_out == 15` is not divisible by `group`)
- **THEN** `validate_conv_inputs` MUST return `OpError::ShapeMismatch`
- **AND** the error message MUST include `C_out`, `group`, and the divisibility constraint

#### Scenario: Plain group=1 conv passes validation unchanged
- **WHEN** `input.shape = [1, 3, 224, 224]`, `weight.shape = [64, 3, 7, 7]`, and `group` is absent (defaults to 1)
- **THEN** `validate_conv_inputs` MUST succeed
- **AND** the validation logic MUST be byte-equivalent to the pre-change behavior for the `group == 1` case

### Requirement: Conv Attribute Parser Is the Single Source of Truth
The runtime SHALL expose `ConvAttrs::from_attributes(&[AttributeProto]) -> Result<ConvAttrs, OpError>` and call it from both `dispatch_convolution` (CPU path) and `try_cuda_dispatch` (CUDA path). The CPU and CUDA paths SHALL NOT contain independent attribute parsers.

#### Scenario: CPU and CUDA Conv dispatch parse attributes identically
- **WHEN** a Conv node is dispatched first on the CPU path, then with an identical `AttributeProto` list on the CUDA path
- **THEN** both dispatch sites MUST call `ConvAttrs::from_attributes`
- **AND** MUST produce the same `ConvAttrs` value

#### Scenario: Attribute parser supports all four Conv attributes
- **WHEN** `ConvAttrs::from_attributes` receives an attribute list containing any combination of `strides`, `pads`, `dilations`, and `group`
- **THEN** it MUST populate the corresponding field from the attribute value
- **AND** MUST leave any absent field at the ONNX default (`[1, 1]` / `[0, 0, 0, 0]` / `1`)

#### Scenario: Attribute parser rejects unrelated attributes without failure
- **WHEN** the Conv node carries attributes the parser does not consume (for example `kernel_shape`)
- **THEN** `ConvAttrs::from_attributes` MUST ignore them and return a valid `ConvAttrs` value
- **AND** MUST NOT return an error for these benign extras
