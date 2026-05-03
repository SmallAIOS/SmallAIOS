## ADDED Requirements

### Requirement: Construct ExecutionGraph from architecture and weights
The runtime SHALL provide a `GraphBuilder` API for constructing an `ExecutionGraph` programmatically without an ONNX intermediate.

#### Scenario: Build empty graph
- **WHEN** a new `GraphBuilder` is created
- **THEN** it SHALL produce an empty `ExecutionGraph` with no nodes
- **AND** assign sequential output tensor names (`tensor_0`, `tensor_1`, ...) automatically

#### Scenario: Append operator node
- **WHEN** the caller invokes `builder.matmul(input_a, input_b)` (or any other op helper)
- **THEN** the builder SHALL emit an `ExecutionNode` with the correct `op_type` and inputs
- **AND** generate a fresh output tensor name
- **AND** return that name for chaining into subsequent operators

#### Scenario: Bind initializer tensors
- **WHEN** the builder is given a weight tensor name and a `TensorView` from a `SafetensorsFile`
- **THEN** it SHALL register the weight as a graph initializer with the correct shape and dtype
- **AND** make the initializer available as an input to subsequent operators

### Requirement: GPU-resident weight loading
For GPU-targeted graphs, the builder SHALL transfer weights directly from safetensors mmap to GPU `DeviceBuffer` without host-side intermediate copies.

#### Scenario: Direct GPU weight load
- **WHEN** building a graph with `gpu_resident: true` and a `CudaRuntime` is available
- **THEN** each weight tensor SHALL be allocated as a `DeviceBuffer` and populated via `cudaMemcpy` from the mmap region
- **AND** the resulting `Tensor` SHALL reference the `DeviceBuffer` instead of host bytes
- **AND** total host RAM usage during weight loading SHALL be bounded (only the active mmap pages, not the full model)

### Requirement: Graph builder helpers for transformer operations
The builder SHALL provide high-level helpers for common transformer building blocks.

#### Scenario: RMSNorm helper
- **WHEN** the caller invokes `builder.rms_norm(input, weight_tensor_name, epsilon)`
- **THEN** the builder SHALL emit an `RMSNormalization` node with the correct attributes

#### Scenario: Rotary embedding helper
- **WHEN** the caller invokes `builder.rotary_embedding(input, theta, p_rope)`
- **THEN** the builder SHALL emit a `RotaryEmbedding` node with the rotary parameters

#### Scenario: Attention helper
- **WHEN** the caller invokes `builder.attention(q, k, v, num_heads, num_kv_heads, sliding_window)`
- **THEN** the builder SHALL emit a `GroupQueryAttention` node with the correct parameters
- **AND** include sliding-window mask attribute when applicable

#### Scenario: SwiGLU helper
- **WHEN** the caller invokes `builder.swiglu(gate, up)`
- **THEN** the builder SHALL emit operators that compute `silu(gate) * up`

### Requirement: Graph validation after construction
The builder SHALL validate the constructed graph before returning it for execution.

#### Scenario: Topological order check
- **WHEN** a graph is finalized via `builder.build()`
- **THEN** the builder SHALL verify the nodes form a valid DAG with no cycles
- **AND** verify every node input is either an initializer, a graph input, or a previous node's output

#### Scenario: Missing weight error
- **WHEN** a node references a weight tensor name not present in the safetensors file
- **THEN** `builder.build()` SHALL return an error identifying the missing weight name
