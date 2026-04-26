# gpu-embedding-lookup Specification

## Purpose
TBD - created by archiving change transformer-gpu-kernels-v1. Update Purpose after archive.
## Requirements
### Requirement: GPU Gather for embedding lookup
The runtime SHALL provide a GPU implementation of `Gather` with `axis=0` for indexed embedding lookup, producing device-resident output tensors without host transfer.

#### Scenario: Int64 token indices with BF16 embedding table
- **WHEN** `Gather` is dispatched with a BF16 embedding table of shape `[vocab_size, hidden_size]` and an Int64 index tensor of shape `[batch, seq_len]`
- **THEN** the kernel SHALL produce a BF16 output tensor of shape `[batch, seq_len, hidden_size]`
- **AND** each output row `[b, s, :]` SHALL equal `embedding[indices[b, s], :]`
- **AND** the output SHALL remain a `DeviceTensor` with no host copy

#### Scenario: Int64 token indices with F32 embedding table
- **WHEN** `Gather` is dispatched with an F32 embedding table and Int64 indices
- **THEN** the F32 kernel variant SHALL be launched
- **AND** the output dtype SHALL be F32
- **AND** the result SHALL match a CPU reference within 1e-6 absolute tolerance (exact copy semantics)

#### Scenario: One thread block per output row
- **WHEN** the Gather kernel is launched
- **THEN** the grid SHALL have one thread block per output row (`batch * seq_len` blocks)
- **AND** each block SHALL cooperatively copy `hidden_size` elements from the selected source row to the destination row

#### Scenario: Batched lookup shape transformation
- **WHEN** the first operator in a Gemma forward pass (embedding lookup) is dispatched with a `[1, seq_len]` token ID tensor
- **THEN** the output SHALL be a `[1, seq_len, hidden_size]` device tensor matching the model's hidden size
- **AND** subsequent operators in the forward pass SHALL consume this tensor without host conversion

### Requirement: Gather fails fast on unsupported inputs
The Gather kernel SHALL reject inputs it does not support.

#### Scenario: Axis other than 0
- **WHEN** `Gather` is dispatched with an `axis` attribute not equal to 0
- **THEN** the dispatcher SHALL return a `CudaError` indicating only `axis=0` is supported in v1
- **AND** SHALL NOT launch the kernel

#### Scenario: Non-Int64 index tensor
- **WHEN** `Gather` is dispatched with an index tensor whose dtype is not Int64 (e.g. Int32)
- **THEN** the dispatcher SHALL return a `CudaError` identifying the rejected dtype
- **AND** SHALL NOT launch the kernel

#### Scenario: Unsupported embedding dtype
- **WHEN** `Gather` is dispatched with an embedding dtype other than BF16 or F32
- **THEN** the dispatcher SHALL return a `CudaError` identifying the rejected dtype
- **AND** SHALL NOT launch the kernel

