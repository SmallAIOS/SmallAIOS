## ADDED Requirements

### Requirement: TimeSource Trait
The ONNX runtime SHALL provide a `TimeSource` trait abstracting wall-clock time measurement so the crate remains `no_std` compatible.

#### Scenario: NullTimeSource returns zero
- **WHEN** `NullTimeSource::now_us()` is called
- **THEN** it MUST return 0
- **AND** MUST NOT require any allocation or syscall

#### Scenario: StdTimeSource measures wall-clock time
- **WHEN** `StdTimeSource::now_us()` is called at times T1 and T2 with T2 > T1
- **THEN** the returned values MUST be monotonic (later call >= earlier call)
- **AND** the difference MUST approximate the real elapsed time in microseconds

### Requirement: Operator Classification
The ONNX runtime SHALL classify operators into budget categories for WCET enforcement.

#### Scenario: Elementwise operators
- **WHEN** `classify_op()` is called with Add, Sub, Mul, Div, Relu, Sigmoid, Tanh, Clip, Cast, Reshape, Flatten, Squeeze, Unsqueeze, Transpose, Concat, Slice, Pad, Gather
- **THEN** it MUST return `OperatorClass::Elementwise`

#### Scenario: Reduction operators
- **WHEN** `classify_op()` is called with Softmax, LayerNormalization, BatchNormalization, MaxPool, AveragePool, GlobalAveragePool, ReduceMean, ReduceSum
- **THEN** it MUST return `OperatorClass::Reduction`

#### Scenario: GEMM operators
- **WHEN** `classify_op()` is called with MatMul, Gemm, Conv
- **THEN** it MUST return `OperatorClass::Gemm`

### Requirement: Per-Operator Budget Enforcement
The ONNX runtime SHALL check each operator's execution time against its budget and act on the result.

#### Scenario: Within budget
- **WHEN** an operator completes within its configured budget
- **THEN** execution MUST continue normally
- **AND** no log entries MUST be emitted

#### Scenario: Warning threshold
- **WHEN** an operator exceeds 1x its budget but less than the soft multiplier
- **THEN** a warning MUST be logged
- **AND** execution MUST continue
- **AND** the profile's `warnings_count` MUST increment

#### Scenario: Soft limit
- **WHEN** an operator exceeds the soft multiplier threshold
- **THEN** a soft-limit event MUST be logged
- **AND** the profile's `soft_limit_count` MUST increment
- **AND** execution MUST continue

#### Scenario: Hard limit
- **WHEN** an operator exceeds the hard multiplier threshold (default 10x budget)
- **THEN** execution MUST abort with `SessionError::ExecutionFailed`
- **AND** the error message MUST include the operator name and measured time
- **AND** the profile's `hard_limit_aborted` flag MUST be set to true

### Requirement: Inference Profile Report
The ONNX runtime SHALL produce per-operator timing reports when profiling is enabled.

#### Scenario: run_with_profile returns profile
- **WHEN** `Session::run_with_profile()` is called on a successful inference
- **THEN** it MUST return an `(Vec<InferenceOutput>, InferenceProfile)` tuple
- **AND** the profile MUST contain per-operator measurements
- **AND** the profile's `total_us` MUST equal the sum of per-operator times

#### Scenario: Standard run does not profile
- **WHEN** `Session::run()` is called
- **THEN** no profile MUST be computed
- **AND** no time measurement overhead MUST be incurred per operator
