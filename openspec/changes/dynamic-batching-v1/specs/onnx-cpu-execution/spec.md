## ADDED Requirements

### Requirement: CPU Executor Accepts Arbitrary Leading Batch Dim
The CPU graph executor (`execute_graph`) and every Tier 1 CPU operator SHALL accept inputs with arbitrary leading-dimension batch size `N >= 1` and propagate that dimension through to the outputs without modification. No CPU operator SHALL contain a `N == 1` shortcut that breaks for `N > 1`.

#### Scenario: CPU MatMul handles batched input
- **WHEN** `op_matmul` is called with operands `[B, M, K]` and `[K, N]` for any `B >= 1`
- **THEN** the output MUST have shape `[B, M, N]`
- **AND** each per-batch slice MUST equal the result of running `op_matmul` separately at `B=1` (within bit-exact equality for f32)

#### Scenario: CPU Conv handles batched input
- **WHEN** `op_conv` is called with input `[B, C, H, W]` for any `B >= 1`
- **THEN** the output MUST have shape `[B, K, OH, OW]`
- **AND** the per-batch convolution MUST be independent (no leakage between batch indices)

#### Scenario: Stack helper produces a valid batched tensor
- **WHEN** `batch::stack_along_batch_axis` is called with `N` tensors of identical shape `[D1, D2, ...]` and identical dtype
- **THEN** the result MUST have shape `[N, D1, D2, ...]`
- **AND** the byte content of slice `[i, ...]` MUST equal the byte content of input tensor `i`

#### Scenario: Unstack helper round-trips with stack
- **WHEN** a tensor of shape `[N, D1, D2, ...]` is produced by `stack_along_batch_axis` and then passed to `unstack_along_batch_axis`
- **THEN** the unstacker MUST return `N` tensors each of shape `[D1, D2, ...]` with byte content matching the original `N` inputs

#### Scenario: Stack rejects shape mismatch
- **WHEN** `stack_along_batch_axis` is called with two tensors of differing shape (apart from a missing leading dim)
- **THEN** the helper MUST return `OpError::ShapeMismatch` naming the offending input index
