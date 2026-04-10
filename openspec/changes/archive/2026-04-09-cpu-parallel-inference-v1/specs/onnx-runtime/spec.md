## MODIFIED Requirements

### Requirement: Session API
The runtime SHALL expose a Session API with load, create_session, run, and metadata operations, including parallelism configuration.

#### Scenario: Create and run an inference session
- **WHEN** a client calls load_model with valid ONNX bytes followed by create_session
- **THEN** the runtime MUST return a ready Session handle
- **AND** calling run with correctly shaped input tensors MUST return output tensors matching the model's output specification

#### Scenario: Configure parallelism
- **WHEN** a session is created with `max_threads` and optional `parallel_thresholds`
- **THEN** the runtime MUST limit operator parallelism to the specified thread count
- **AND** MUST use custom thresholds if provided, or defaults otherwise
- **AND** `max_threads = 1` MUST disable all parallelism with zero overhead

### Requirement: Operator-Level Scheduler Integration
The runtime SHALL insert mandatory scheduler yield points between every operator in the execution graph and support per-operator time budgets, with parallel execution reflected in timing.

#### Scenario: Per-operator timing with parallel execution
- **WHEN** an operator executes in parallel across multiple cores and profiling is enabled
- **THEN** the runtime MUST measure wall-clock execution time (not CPU time)
- **AND** MUST report `parallel_efficiency` as `estimated_serial_time / (wall_time × cores_used)`
- **AND** wall-clock time MUST be compared against the operator's configured budget
