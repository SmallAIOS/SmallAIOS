# metal-gpu-inference Specification

## Purpose
TBD - created by archiving change metal-operator-kernels-v1. Update Purpose after archive.
## Requirements
### Requirement: Metal GPU operator dispatch
The ONNX runtime SHALL dispatch supported operators to the Apple Metal GPU when the `gpu` feature is enabled and a `MetalProvider` is available, with transparent fallback to the CPU implementation for unsupported operators.

#### Scenario: GPU dispatch for a supported operator
- **WHEN** the `gpu` feature is enabled and a `MetalProvider` is initialized
- **AND** a graph node uses an operator in the GPU-supported set (e.g., MatMul)
- **THEN** the operator MUST execute on the Metal GPU
- **AND** the result MUST match the CPU reference within ±1e-5 relative tolerance for f32

#### Scenario: CPU fallback for an unsupported operator
- **WHEN** the `gpu` feature is enabled and a `MetalProvider` is initialized
- **AND** a graph node uses an operator NOT in the GPU-supported set
- **THEN** the operator MUST execute on the CPU path unchanged
- **AND** the graph MUST complete successfully mixing GPU and CPU operators

#### Scenario: No GPU available
- **WHEN** the `gpu` feature is enabled but no Metal device is detected (e.g., Linux)
- **THEN** all operators MUST fall back to the CPU path silently
- **AND** no error SHALL be returned

### Requirement: Metal tensor transfer and caching
The runtime SHALL manage host↔device tensor transfer with a per-graph `MetalTensorCache` that minimizes unnecessary copies.

#### Scenario: Tensor stays on-device between consecutive GPU ops
- **WHEN** operator A produces a tensor on the GPU
- **AND** operator B (also GPU-supported) consumes that tensor
- **THEN** the tensor MUST remain in device memory without a host round-trip

#### Scenario: Tensor copied to host for CPU consumer
- **WHEN** a GPU operator produces a tensor
- **AND** the next consumer is a CPU-only operator
- **THEN** the tensor MUST be copied to host memory before the CPU op executes

### Requirement: Tier 1 Metal shaders — existing operators
The runtime SHALL provide tested Metal shaders for: Add, Sub, Mul, Div, Relu, Sigmoid, Tanh, MatMul, Softmax, Conv2D.

#### Scenario: Element-wise Add on GPU matches CPU
- **WHEN** `op_add` executes on Metal with two f32 tensors
- **THEN** every output element MUST match the CPU result within ±1e-5

#### Scenario: Tiled MatMul on GPU matches CPU
- **WHEN** `op_matmul` executes on Metal with two 2D f32 tensors
- **THEN** the output MUST match the CPU result within ±1e-4 (relaxed for accumulated FMA error)

### Requirement: Tier 2 Metal shaders — transformer ops
The runtime SHALL provide tested Metal shaders for: scaled_dot_product_attention, group_query_attention, layer_normalization, rms_normalization, rotary_embedding.

#### Scenario: SDPA on GPU matches CPU
- **WHEN** `scaled_dot_product_attention` executes on Metal
- **THEN** the attention output MUST match the CPU reference within ±1e-4

#### Scenario: GroupQueryAttention on GPU matches CPU
- **WHEN** `op_group_query_attention` executes on Metal with KV cache
- **THEN** the attention output and updated KV cache MUST match CPU within ±1e-4

### Requirement: M1/M2 hardware compatibility
All Metal shaders SHALL execute correctly on Apple Silicon M1 and M2, with optional acceleration via `simdgroup_matrix` on M3+.

#### Scenario: MatMul runs on M1
- **WHEN** the Metal device does not support Apple GPU family 9 (M3+)
- **THEN** the tiled MatMul shader MUST use shared-memory tiling instead of `simdgroup_matrix`
- **AND** the result MUST still match the CPU reference within tolerance

### Requirement: Session-level GPU opt-in
GPU inference SHALL be opt-in via the `Session` builder API.

#### Scenario: Default session runs on CPU
- **WHEN** a `Session` is created without `.with_gpu()`
- **THEN** all operators MUST execute on CPU regardless of Metal availability

#### Scenario: GPU-enabled session uses Metal
- **WHEN** a `Session` is created with `.with_gpu(GpuConfig::metal())`
- **AND** a Metal device is available
- **THEN** supported operators MUST dispatch to Metal

