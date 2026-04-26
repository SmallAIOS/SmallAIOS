## ADDED Requirements

### Requirement: GPU RMS normalization
The runtime SHALL provide a GPU implementation of `RMSNormalization` that normalizes along the last axis using warp-level reductions, supporting BF16 and F32 inputs with F32 accumulation.

#### Scenario: RMS norm on F32 tensor
- **WHEN** `RMSNormalization` is dispatched with an F32 device tensor of shape `[batch, seq_len, hidden_size]` and an F32 weight vector of shape `[hidden_size]`
- **THEN** the kernel SHALL compute `y[b, s, i] = x[b, s, i] * rsqrt(mean(x[b, s, :]^2) + eps) * weight[i]`
- **AND** the mean of squares SHALL be computed in F32 regardless of input dtype
- **AND** the result SHALL match a CPU reference within 1e-3 absolute tolerance

#### Scenario: RMS norm on BF16 tensor
- **WHEN** `RMSNormalization` is dispatched with a BF16 device tensor and a BF16 weight vector
- **THEN** each element SHALL be converted to F32 on load, accumulated in F32, multiplied by the BF16 weight converted to F32, and written back as BF16
- **AND** the result SHALL match the CPU reference in `ops/microsoft.rs::rms_normalization` within 1e-2 absolute tolerance

#### Scenario: Warp-level reduction for mean-of-squares
- **WHEN** the RMSNorm kernel is launched with `hidden_size >= 32`
- **THEN** the per-thread partial sums SHALL be reduced within each warp via `__shfl_down_sync`
- **AND** per-warp partials SHALL be combined via shared memory
- **AND** thread 0 of each block SHALL write the final `inv_rms` value to shared memory for all threads to read

#### Scenario: One thread block per outer element
- **WHEN** the kernel is launched on a `[batch, seq_len, hidden_size]` input
- **THEN** the grid SHALL have `batch * seq_len` blocks
- **AND** each block SHALL process exactly one hidden-dimension slice

### Requirement: Gemma `1 + weight` convention via load-time bake
The GPU RMSNorm kernel SHALL implement the standard formula and rely on the safetensors loader to pre-adjust weights for Gemma's `(1 + weight)` convention.

#### Scenario: Gemma weights loaded via safetensors path
- **WHEN** a Gemma model is loaded via `Session::from_safetensors`
- **THEN** the safetensors loader SHALL pre-add 1.0 to each RMSNorm weight element before writing it to the `DeviceBuffer`
- **AND** the GPU RMSNorm kernel SHALL apply the plain `y = x * rsqrt(mean + eps) * weight` formula without a conditional on model family
- **AND** the kernel SHALL NOT carry a per-op `plus_one` attribute

### Requirement: RMSNorm fails fast on unsupported shapes or dtypes
The RMSNorm kernel SHALL reject inputs it cannot handle.

#### Scenario: Unsupported dtype
- **WHEN** `RMSNormalization` is dispatched with a dtype other than BF16 or F32
- **THEN** the dispatcher SHALL return a `CudaError` naming the rejected dtype
- **AND** SHALL NOT launch any kernel

#### Scenario: Hidden size larger than one block
- **WHEN** the last-axis size exceeds the maximum threads per block for the target device
- **THEN** the kernel SHALL use a grid-stride loop within the block to cover the full hidden dimension
- **AND** the warp reductions SHALL still produce the correct mean-of-squares
