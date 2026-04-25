## ADDED Requirements

### Requirement: CUDA Execution Provider Accepts Arbitrary Batch
The CUDA execution provider's device-side operators (Conv, Gemm, MatMul, BatchNormalization, Activation, Pool, Add) SHALL accept input tensors with arbitrary leading-dimension batch size `N >= 1` and produce outputs whose leading dimension matches. No op SHALL hard-code or assume `N = 1`.

#### Scenario: Conv accepts batched input
- **WHEN** `gpu_conv2d_device` is invoked with an input of shape `[16, 3, 224, 224]`
- **THEN** the cuDNN convolution descriptor MUST be configured for `n = 16`
- **AND** the resulting output MUST have shape `[16, K, OH, OW]`
- **AND** the per-image computation MUST be byte-for-byte equivalent to running the same op 16 times at `N = 1` (within TF32 rounding tolerance of `1e-3` max-abs-diff)

#### Scenario: BatchNorm accepts batched input
- **WHEN** `gpu_batchnorm` is invoked with an input of shape `[B, C, H, W]` for any `B >= 1`
- **THEN** the cuDNN batchnorm descriptor MUST be configured for `n = B`
- **AND** the per-channel parameters (scale, bias, mean, variance) MUST be broadcast across all `B` images
- **AND** the resulting output MUST have shape `[B, C, H, W]`

#### Scenario: Hybrid executor preserves batch dim end-to-end
- **WHEN** `execute_graph_hybrid` is invoked with a graph input of shape `[B, ...]` for any `B >= 1`
- **THEN** every intermediate tensor in the value map MUST have leading dim `B` (or whatever the graph reshapes it to)
- **AND** the final graph output MUST have leading dim `B`
