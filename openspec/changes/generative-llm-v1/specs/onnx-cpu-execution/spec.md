## ADDED Requirements

### Requirement: Sub-Graph Executor
The ONNX runtime SHALL support recursive execution of inner graphs embedded inside `If`, `Loop`, and `Scan` operators via a sub-graph executor with isolated value scope and shared initializer scope.

#### Scenario: Inner graph compiled once at session load time
- **WHEN** a model containing a `Loop` operator is loaded
- **THEN** the graph builder MUST compile the inner `GraphProto` into a standalone `ExecutionGraph` during `Session::new()`
- **AND** the compiled inner graph MUST be cached on the parent `ExecutionNode` and reused across all iterations at dispatch time
- **AND** the inner graph MUST NOT be rebuilt on each iteration

#### Scenario: Isolated value scope with shared initializers
- **WHEN** a sub-graph is executed from inside a `Loop` or `Scan` body
- **THEN** the sub-executor MUST create a fresh `value_map` seeded with loop-carried values and outer-referenced names
- **AND** model initializer tensors MUST be visible by name inside the sub-graph without being copied per iteration
- **AND** writes made inside the body MUST NOT mutate the outer `value_map`
- **AND** on sub-graph exit only the body's declared output tensors MUST be propagated back to the parent

#### Scenario: Nested If inside a Loop body
- **WHEN** a model contains a `Loop` whose body contains an `If` node that itself contains MatMul and Softmax nodes
- **THEN** the sub-graph executor MUST recursively execute the `If` branch selected by the runtime condition for each iteration
- **AND** the inner results MUST be routed correctly back through the `Loop`'s carried-value slots
- **AND** final output values MUST match a hand-computed reference

### Requirement: Loop Operator
The ONNX runtime SHALL implement the `Loop` operator with full ONNX termination semantics, supporting all three stop signals (`M`, `cond`, `cond_out`) in combination.

#### Scenario: Max trip count `M` bounds iterations
- **WHEN** a `Loop` node is dispatched with `M = 64` and a body that always emits `cond_out = true`
- **THEN** the loop MUST execute exactly 64 iterations
- **AND** the outputs MUST be the carried values from iteration 63

#### Scenario: Body-emitted `cond_out` stops early
- **WHEN** a `Loop` node is dispatched with `M = 64` and a body whose `cond_out` returns `false` at iteration 32
- **THEN** the loop MUST stop at the end of iteration 32
- **AND** the outputs MUST be iteration 32's carried values
- **AND** iterations 33..64 MUST NOT execute

#### Scenario: External `cond = false` skips the loop entirely
- **WHEN** a `Loop` node is dispatched with `cond = false`
- **THEN** the loop MUST execute zero iterations
- **AND** the outputs MUST equal the initial carried values (`v_initial`)

#### Scenario: Loop-carried values thread through iterations
- **WHEN** a `Loop` body emits a new hidden-state tensor at each iteration as a carried output
- **THEN** iteration N+1 MUST receive iteration N's emitted tensor as its input for that slot
- **AND** the final output MUST be iteration last's emitted tensor

### Requirement: If Operator
The ONNX runtime SHALL implement the `If` operator with both `then` and `else` branches compiled at graph build time.

#### Scenario: Select the then-branch on true
- **WHEN** an `If` node receives a condition tensor containing `true`
- **THEN** the sub-graph executor MUST evaluate only the `then` branch
- **AND** MUST return the `then` branch outputs
- **AND** MUST NOT evaluate the `else` branch

#### Scenario: Branches with different output shapes
- **WHEN** an `If` node's then-branch produces a shape `[1, 768]` and its else-branch produces `[1, 1024]`
- **THEN** the dispatcher MUST return the shape matching the selected branch
- **AND** downstream operators MUST see the branch-specific shape

### Requirement: Scan Operator
The ONNX runtime SHALL implement the `Scan` operator for the simple sequence case where the body is applied element-by-element to a sequence-dimensional input.

#### Scenario: Scan applies a constant-add body to each element
- **WHEN** a `Scan` node is configured with a body `body(x_in) = x_in + 1` and given a sequence input `[0, 1, 2, 3, 4]`
- **THEN** the output sequence MUST be `[1, 2, 3, 4, 5]`
- **AND** the body MUST be invoked exactly 5 times
- **AND** each invocation MUST receive the corresponding sequence element as `x_in`

### Requirement: Sub-Graph WCET Budget Integration
The sub-graph executor SHALL participate in the existing operator budget enforcement system so that `Loop`, `If`, and `Scan` are accounted for as single atomic units at the parent level, and inner hard-limit failures bubble up as parent hard-limit failures.

#### Scenario: Loop is a single budget accounting unit
- **WHEN** a `Loop` with 100 iterations executes inside a profiled session
- **THEN** the `InferenceProfile.operators` list MUST contain exactly one entry for the `Loop` op
- **AND** its `actual_us` MUST equal the wall-clock sum across all iterations plus sub-dispatch overhead
- **AND** the entry MUST be compared against the `OperatorBudget` for the `Loop` class, not against 100 separate per-iteration budgets

#### Scenario: Inner hard-limit aborts the whole loop
- **WHEN** an inner operator inside a `Loop` body exceeds its own hard budget limit
- **THEN** the sub-executor MUST return `SessionError::ExecutionFailed` from inside the sub-graph
- **AND** the parent `Loop` MUST stop iterating immediately
- **AND** the error MUST bubble up to `Session::run()` unchanged

### Requirement: Generative and Normalization Operator Completeness
The ONNX runtime SHALL implement the following generative, normalization, and reduction operators for f32 CPU execution: `RMSNormalization`, `MatMulInteger`, `DynamicQuantizeLinear`, `RandomNormal`, `RandomNormalLike`, `RandomUniform`, `RandomUniformLike`, `Multinomial`, `Bernoulli`, `Dropout`, `EyeLike`, `ReduceL1`, `ReduceL2`, `ReduceLogSum`, `ReduceLogSumExp`, `ReduceSumSquare`, `LpNormalization`, `MeanVarianceNormalization`, `Softplus`.

#### Scenario: RMSNormalization matches PyTorch reference
- **WHEN** an `RMSNormalization` node is dispatched on an input tensor with `epsilon = 1e-6`
- **THEN** the output MUST match a PyTorch `nn.RMSNorm` reference within 1e-5 relative tolerance

#### Scenario: DynamicQuantizeLinear produces valid per-tensor scale
- **WHEN** a `DynamicQuantizeLinear` node is dispatched on an f32 tensor
- **THEN** the output MUST be a quantized `u8` tensor, an f32 scale, and a `u8` zero-point
- **AND** dequantizing the output MUST reconstruct the input within quantization error

#### Scenario: RandomUniform is reproducible given a seed
- **WHEN** two `RandomUniform` nodes are dispatched with identical `seed`, `shape`, `low`, and `high` attributes
- **THEN** both outputs MUST be bit-identical

#### Scenario: Sampling via Multinomial produces index in distribution support
- **WHEN** a `Multinomial` node is dispatched on a probability vector and a seed
- **THEN** the output MUST be a tensor of indices each in `[0, vocab_size)`

### Requirement: Phase 2 Inventory Flip
The `SUPPORTED_OPS_INVENTORY` table added by Phase 1 SHALL be updated such that the 21 Phase 2 operators (3 control-flow plus 18 generative/norm) are marked `OperatorStatus::Implemented` upon completion of this change.

#### Scenario: Inventory reflects Phase 2 completion
- **WHEN** the Phase 2 change is fully implemented
- **THEN** the inventory MUST contain `(OpKind::Loop, OperatorStatus::Implemented)`
- **AND** MUST contain `(OpKind::If, OperatorStatus::Implemented)`
- **AND** MUST contain `(OpKind::Scan, OperatorStatus::Implemented)`
- **AND** MUST contain `OperatorStatus::Implemented` entries for each of the 18 generative/norm operators
- **AND** no Phase 2 operator SHALL remain as `Planned(Phase::P2)`
