# gpu-elementwise-ops Specification

## Purpose
TBD - created by archiving change transformer-gpu-kernels-v1. Update Purpose after archive.
## Requirements
### Requirement: GPU element-wise Add
The runtime SHALL provide a GPU implementation of `Add` for BF16 and F32 tensors, callable from `dispatch_gpu_node` with device-resident inputs and outputs.

#### Scenario: Add with matching shapes
- **WHEN** `Add` is dispatched with two device tensors of identical shape and dtype
- **THEN** the kernel SHALL compute `c[i] = a[i] + b[i]` for every element
- **AND** the output `DeviceTensor` SHALL have the same shape and dtype as the inputs
- **AND** SHALL NOT transfer any tensor to host memory during execution

#### Scenario: Add with broadcasting
- **WHEN** `Add` is dispatched with two device tensors of compatible broadcast shapes (NumPy rules)
- **THEN** the kernel SHALL produce an output whose shape is the broadcast shape
- **AND** per-input strides SHALL be precomputed on the host and passed as kernel arguments

#### Scenario: Add with BF16 inputs
- **WHEN** `Add` is dispatched with two BF16 device tensors
- **THEN** the BF16 kernel variant SHALL be launched
- **AND** the result SHALL match a CPU reference implementation within 1e-2 absolute tolerance

#### Scenario: Add with F32 inputs
- **WHEN** `Add` is dispatched with two F32 device tensors
- **THEN** the F32 kernel variant SHALL be launched
- **AND** the result SHALL match a CPU reference implementation within 1e-3 absolute tolerance

### Requirement: GPU element-wise Mul
The runtime SHALL provide a GPU implementation of `Mul` for BF16 and F32 tensors with broadcasting support.

#### Scenario: Mul with matching shapes
- **WHEN** `Mul` is dispatched with two device tensors of identical shape and dtype
- **THEN** the kernel SHALL compute `c[i] = a[i] * b[i]` for every element
- **AND** the result SHALL match the CPU reference within the dtype's documented tolerance band

#### Scenario: Mul with broadcasting
- **WHEN** `Mul` is dispatched with broadcast-compatible shapes (e.g. `[1, 4096]` against `[32, 4096]`)
- **THEN** the kernel SHALL produce the broadcast output shape
- **AND** the element-wise product SHALL be correct at every output position

#### Scenario: Mul in SwiGLU path
- **WHEN** `Mul` is dispatched between the output of `Silu` and an `up_proj` tensor in a Gemma MLP layer
- **THEN** the result SHALL be the SwiGLU gate output `silu(gate) * up`
- **AND** the output dtype SHALL match the input dtype

### Requirement: GPU Silu activation
The runtime SHALL provide a GPU implementation of `Silu` (Swish, `x * sigmoid(x)`) for BF16 and F32 tensors.

#### Scenario: Silu on F32 tensor
- **WHEN** `Silu` is dispatched with an F32 device tensor
- **THEN** the kernel SHALL compute `y[i] = x[i] / (1 + expf(-x[i]))` for every element
- **AND** the result SHALL match a CPU reference within 1e-3 absolute tolerance

#### Scenario: Silu on BF16 tensor
- **WHEN** `Silu` is dispatched with a BF16 device tensor
- **THEN** the kernel SHALL convert each element to F32, apply sigmoid, multiply, and convert back to BF16
- **AND** the result SHALL match the CPU reference within 1e-2 absolute tolerance

### Requirement: Element-wise ops fail fast on unsupported dtypes
Element-wise GPU kernels SHALL reject tensor dtypes they do not support.

#### Scenario: Unsupported dtype
- **WHEN** `Add`, `Mul`, or `Silu` is dispatched with a tensor dtype other than BF16 or F32 (e.g. I32, I64, FP16)
- **THEN** the dispatcher SHALL return a `CudaError` identifying the operator and the rejected dtype
- **AND** SHALL NOT launch any kernel
- **AND** SHALL NOT silently transfer the tensor to the host

#### Scenario: Mismatched input dtypes
- **WHEN** `Add` or `Mul` is dispatched with two inputs of different dtypes (e.g. BF16 and F32)
- **THEN** the dispatcher SHALL return a `CudaError` identifying the mismatch
- **AND** SHALL NOT launch any kernel

