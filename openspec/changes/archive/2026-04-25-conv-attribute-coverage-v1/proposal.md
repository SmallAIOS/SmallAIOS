## Why

The CPU `op_conv` implementation ignores all Conv attributes (`strides`,
`pads`, `dilations`, `group`) and the GPU dispatch path only forwards
three of them, skipping `group`. This silently blocks MobileNetV2
(depthwise-separable convolutions with `group = input_channels`) and
ResNet-50 v2 (stem `strides = [2, 2]` produces 2× too-large feature
maps, breaking downstream residual `Add` shape checks). Both models
already fail at warmup in the `arm64-gpu-container-v1` vision benchmark
harness; the existing `onnx-cpu-execution` spec already requires Conv
attribute pass-through, but the implementation does not comply.

## What Changes

- Change the CPU `op_conv` signature to accept a `ConvAttrs` struct
  carrying `strides`, `pads`, `dilations`, and `group`.
- Rewrite `conv_compute` to honor strides, padding, dilation, and
  grouped output-channel blocking.
- Update `validate_conv_inputs` to accept depthwise weight shapes — the
  ONNX spec requires `weight.dims[1] == input.dims[1] / group`, not
  `input.dims[1] == weight.dims[1]`.
- Replace the `_attrs: &[AttributeProto]` drop in
  `dispatch_convolution` with real attribute parsing and plumb the
  resulting `ConvAttrs` into `op_conv`.
- Extend the CUDA dispatch path in `try_cuda_dispatch` to parse `group`
  and forward it to `cuda::conv::gpu_conv2d`, which calls
  `cudnnSetConvolutionGroupCount` before algorithm selection.
- Introduce a shared `ConvAttrs::from_attributes(&[AttributeProto])`
  helper so the CPU and GPU paths parse attributes identically.
- Add unit tests for depthwise, group-of-2, strided, padded, and
  dilated conv on both CPU and GPU.
- Unblock the end-to-end `bench_mobilenet_v2_cpu_vs_gpu` and
  `bench_resnet50_cpu_vs_gpu` tests in
  `onnx-rt/tests/bench_vision_models.rs`.

## Capabilities

### New Capabilities

- _None_ — no new capability boundary is introduced.

### Modified Capabilities

- `onnx-cpu-execution`: extend the Conv-attribute scenario to cover
  `group` and `dilations`, add a grouped/depthwise conv scenario, and
  add a validation-rule requirement aligning with
  `weight.dims[1] == input.dims[1] / group`.
- `onnx-runtime`: add a CUDA grouped-convolution requirement under the
  CUDA Execution Provider — `group` must be forwarded to cuDNN via
  `cudnnSetConvolutionGroupCount` so depthwise convolutions dispatch to
  GPU correctly.

## Impact

- **Code:** `onnx-rt/src/operators.rs` (op_conv, validate_conv_inputs,
  conv_compute, new ConvAttrs), `onnx-rt/src/executor.rs`
  (dispatch_convolution, try_cuda_dispatch Conv branch),
  `onnx-rt/src/cuda/conv.rs` (gpu_conv2d grouped conv + cuDNN
  group count), potentially `onnx-rt/src/cuda/ffi.rs` if
  `cudnnSetConvolutionGroupCount` is not yet bound.
- **Tests:** new depthwise / strided / padded / dilated / group-2 Conv
  unit tests in `onnx-rt/src/operators.rs` and
  `onnx-rt/tests/test_cuda.rs`. The existing benchmark tests
  `bench_mobilenet_v2_cpu_vs_gpu` and `bench_resnet50_cpu_vs_gpu` flip
  from failure to success.
- **Downstream:** enables at-least-correct inference for all
  grouped/depthwise-conv models (MobileNet family, EfficientNet,
  ShuffleNet) and any ResNet-style model that uses stride-2 stem
  convolutions.
- **Out of scope (flagged so readers don't expect them):**
  `ConvTranspose` attribute coverage, 3D (`NCDHW`) convolutions,
  `QLinearConv` attribute coverage, performance tuning of the grouped
  inner loop, and any new auto-pad modes beyond what the GPU path
  already handles.
