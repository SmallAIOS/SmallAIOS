## ADDED Requirements

### Requirement: GPU rotary position embedding
The runtime SHALL provide a GPU implementation of `RotaryEmbedding` that applies rotary position embeddings to Q and K tensors using precomputed cos/sin tables, supporting BF16 and F32.

#### Scenario: Standard RoPE on Q tensor (F32)
- **WHEN** `RotaryEmbedding` is dispatched with an F32 Q tensor of shape `[batch, num_heads, seq_len, head_dim]` and precomputed `cos`/`sin` tables of shape `[max_seq_len, head_dim / 2]`
- **THEN** the kernel SHALL for each pair `(x[2i], x[2i+1])` compute
    - `out[2i] = x[2i] * cos[pos, i] - x[2i+1] * sin[pos, i]`
    - `out[2i+1] = x[2i] * sin[pos, i] + x[2i+1] * cos[pos, i]`
- **AND** the result SHALL match a CPU reference within 1e-3 absolute tolerance

#### Scenario: Standard RoPE on K tensor (BF16)
- **WHEN** `RotaryEmbedding` is dispatched with a BF16 K tensor of shape `[batch, num_kv_heads, seq_len, head_dim]` and BF16 cos/sin tables
- **THEN** the BF16 kernel variant SHALL be launched
- **AND** each rotation pair SHALL be computed in F32 intermediate precision and written back as BF16
- **AND** the result SHALL match the CPU reference in `ops/microsoft.rs::rotary_embedding` within 1e-2 absolute tolerance

#### Scenario: Head dimension must be even
- **WHEN** `RotaryEmbedding` is dispatched with a `head_dim` that is not even
- **THEN** the dispatcher SHALL return a `CudaError` naming the rejected head dimension
- **AND** SHALL NOT launch the kernel

#### Scenario: Cos/sin tables as graph initializers
- **WHEN** a Gemma graph is built via `build_gemma_graph`
- **THEN** the cos/sin tables SHALL be precomputed at graph build time from `rope_theta` and `max_position_embeddings`
- **AND** SHALL be emitted as constant initializers copied into `DeviceBuffer`s at session load time
- **AND** the GPU kernel SHALL consume these initializers directly without recomputation

### Requirement: p-RoPE baked into precomputed tables
The GPU RoPE kernel SHALL apply the standard rotation formula, with proportional-RoPE variants handled at table precomputation time.

#### Scenario: Gemma 4 proportional RoPE
- **WHEN** a Gemma 4 graph is built and the p-RoPE flag is set
- **THEN** the graph builder SHALL scale the RoPE frequencies according to the Gemma 4 proportional formula when precomputing the cos/sin tables
- **AND** the GPU kernel SHALL apply the same standard formula regardless of whether p-RoPE was used
- **AND** the kernel SHALL NOT carry a `p_rope` attribute

### Requirement: RoPE fails fast on unsupported configurations
The RoPE kernel SHALL reject inputs it cannot handle.

#### Scenario: Unsupported dtype
- **WHEN** `RotaryEmbedding` is dispatched with a dtype other than BF16 or F32
- **THEN** the dispatcher SHALL return a `CudaError` naming the rejected dtype
- **AND** SHALL NOT launch the kernel

#### Scenario: Missing cos/sin tables
- **WHEN** `RotaryEmbedding` is dispatched without the required cos/sin initializer inputs
- **THEN** the dispatcher SHALL return a `CudaError` indicating the missing initializers
- **AND** SHALL NOT launch the kernel
