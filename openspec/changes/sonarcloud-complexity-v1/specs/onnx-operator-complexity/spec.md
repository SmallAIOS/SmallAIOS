## ADDED Requirements

### Requirement: op_conv cognitive complexity ≤ 15
The `op_conv()` function in `onnx-rt/src/operators.rs` SHALL have cognitive complexity ≤ 15 as measured by SonarCloud rule `rust:S3776`. The inner convolution kernel (input_channels × kernel_h × kernel_w summation) SHALL be extracted into a `convolve_at()` helper function. All existing convolution tests SHALL continue to pass.

#### Scenario: op_conv refactored below threshold
- **WHEN** SonarCloud analyzes `op_conv()` after refactoring
- **THEN** the cognitive complexity score SHALL be ≤ 15 (currently 40)

#### Scenario: op_conv behavior preserved
- **WHEN** existing ONNX convolution tests execute against the refactored code
- **THEN** all tests SHALL pass with identical output values

### Requirement: op_add cognitive complexity ≤ 15
The `op_add()` function in `onnx-rt/src/operators.rs` SHALL have cognitive complexity ≤ 15. The broadcast coordinate iteration and index computation SHALL be extracted into a shared `BroadcastIter` utility. All existing addition/broadcast tests SHALL continue to pass.

#### Scenario: op_add refactored below threshold
- **WHEN** SonarCloud analyzes `op_add()` after refactoring
- **THEN** the cognitive complexity score SHALL be ≤ 15 (currently 27)

#### Scenario: Broadcast iteration utility is shared
- **WHEN** `op_add()`, `op_softmax()`, and `op_reshape()` are refactored
- **THEN** all three SHALL use the same `BroadcastIter` utility for coordinate iteration

### Requirement: op_softmax cognitive complexity ≤ 15
The `op_softmax()` function in `onnx-rt/src/operators.rs` SHALL have cognitive complexity ≤ 15. Broadcast iteration SHALL use the shared `BroadcastIter` utility. All existing softmax tests SHALL continue to pass.

#### Scenario: op_softmax refactored below threshold
- **WHEN** SonarCloud analyzes `op_softmax()` after refactoring
- **THEN** the cognitive complexity score SHALL be ≤ 15 (currently 27)

### Requirement: op_reshape cognitive complexity ≤ 15
The `op_reshape()` function in `onnx-rt/src/operators.rs` SHALL have cognitive complexity ≤ 15. All existing reshape tests SHALL continue to pass.

#### Scenario: op_reshape refactored below threshold
- **WHEN** SonarCloud analyzes `op_reshape()` after refactoring
- **THEN** the cognitive complexity score SHALL be ≤ 15 (currently 19)

### Requirement: validate_model cognitive complexity ≤ 15
The `validate_model()` function in `onnx-rt/src/session.rs` SHALL have cognitive complexity ≤ 15. Validation sub-checks SHALL be extracted into named helper functions. All existing model validation tests SHALL continue to pass.

#### Scenario: validate_model refactored below threshold
- **WHEN** SonarCloud analyzes `validate_model()` after refactoring
- **THEN** the cognitive complexity score SHALL be ≤ 15 (currently 23)

### Requirement: build_execution_graph cognitive complexity ≤ 15
The `build_execution_graph()` function in `onnx-rt/src/graph.rs` SHALL have cognitive complexity ≤ 15. Graph construction phases SHALL be extracted into helper functions. All existing graph tests SHALL continue to pass.

#### Scenario: build_execution_graph refactored below threshold
- **WHEN** SonarCloud analyzes `build_execution_graph()` after refactoring
- **THEN** the cognitive complexity score SHALL be ≤ 15 (currently 23)

### Requirement: plan_memory cognitive complexity ≤ 15
The `plan_memory()` function in `onnx-rt/src/memory_planner.rs` SHALL have cognitive complexity ≤ 15. All existing memory planning tests SHALL continue to pass.

#### Scenario: plan_memory refactored below threshold
- **WHEN** SonarCloud analyzes `plan_memory()` after refactoring
- **THEN** the cognitive complexity score SHALL be ≤ 15 (currently 18)

### Requirement: topological_sort cognitive complexity ≤ 15
The `topological_sort()` function in `onnx-rt/src/graph.rs` SHALL have cognitive complexity ≤ 15. All existing graph ordering tests SHALL continue to pass.

#### Scenario: topological_sort refactored below threshold
- **WHEN** SonarCloud analyzes `topological_sort()` after refactoring
- **THEN** the cognitive complexity score SHALL be ≤ 15 (currently 16)

### Requirement: No public API changes in onnx-rt
All extracted helper functions SHALL be private (`fn` or `pub(crate)`). No existing public types, traits, or function signatures in the `onnx-rt` crate SHALL change.

#### Scenario: Public API unchanged
- **WHEN** downstream crates (container, onnx-rt consumers) compile against the refactored code
- **THEN** compilation SHALL succeed without modification
