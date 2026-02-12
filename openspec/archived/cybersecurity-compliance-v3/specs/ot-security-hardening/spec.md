# Delta for OT Security Hardening

## ADDED Requirements

### Requirement: WCET Analysis Framework
The system SHALL provide a WCET (Worst-Case Execution Time) analysis framework for all kernel critical paths: syscall dispatch, capability check, memory allocation, task scheduling, and interrupt handling.

#### Scenario: WCET analysis for syscall dispatch
- WHEN the WCET analysis framework is applied to the syscall dispatch path
- THEN it MUST produce a documented upper bound on execution time for every syscall entry point
- AND the analysis MUST cover the full path from syscall invocation through capability validation to handler return
- AND the result MUST specify the target hardware platform, clock frequency, and cache configuration under which the bound was derived

#### Scenario: WCET analysis for capability check
- WHEN the WCET analysis framework is applied to the capability check path
- THEN it MUST produce a documented upper bound on the time to validate a capability token against the capability table
- AND MUST account for the worst-case capability table size (maximum concurrent capabilities per task)

#### Scenario: WCET analysis for memory allocation
- WHEN the WCET analysis framework is applied to the memory allocation path (buddy allocator and slab allocator)
- THEN it MUST produce a documented upper bound on allocation time for each allocator
- AND MUST account for the worst-case fragmentation scenario (buddy) and worst-case slab cache miss (slab)
- AND MUST document the maximum allocation size that can be serviced within the WCET bound

#### Scenario: WCET analysis for task scheduling
- WHEN the WCET analysis framework is applied to the task scheduling path
- THEN it MUST produce a documented upper bound on the time from scheduling decision to context switch completion
- AND MUST account for all three priority classes (SYSTEM > IPC > INFERENCE) and worst-case queue depths

#### Scenario: WCET analysis for interrupt handling
- WHEN the WCET analysis framework is applied to the interrupt handling path
- THEN it MUST produce a documented upper bound on interrupt latency from hardware assertion to handler entry
- AND MUST account for interrupt nesting, priority arbitration, and any critical sections that disable interrupts
- AND MUST document per-architecture bounds for both x86-64 and AArch64

### Requirement: Combined Static and Empirical WCET Methodology
WCET analysis SHALL combine static bounds (no recursion, bounded loops) with empirical measurement (N>=10000 samples per path on target hardware, reporting p99.9).

#### Scenario: Static analysis of bounded execution
- WHEN static WCET analysis is performed on a kernel critical path
- THEN the analysis MUST verify the absence of recursion in the path
- AND MUST verify that all loops have statically determinable upper bounds
- AND MUST document the loop bound for each loop in the path
- AND the static bound MUST be computed by summing worst-case instruction costs along the longest execution path

#### Scenario: Empirical measurement campaign
- WHEN empirical WCET measurement is performed on a kernel critical path
- THEN the measurement campaign MUST collect at least 10,000 samples per path on each target hardware platform
- AND MUST report the p99.9 (99.9th percentile) execution time
- AND MUST report the observed minimum, maximum, mean, and standard deviation
- AND the measurement MUST be performed under realistic load conditions (concurrent tasks, interrupt activity, memory pressure)

#### Scenario: Static and empirical bound comparison
- WHEN both static and empirical WCET results are available for a critical path
- THEN the documentation MUST present both bounds side by side
- AND MUST flag any case where the empirical p99.9 exceeds 80% of the static bound as requiring investigation
- AND the published WCET bound MUST be the static bound (conservative) unless formal analysis confirms a tighter bound is safe

#### Scenario: WCET regression in CI
- WHEN a code change modifies a kernel critical path
- THEN the CI pipeline MUST re-run the empirical WCET measurement for the affected path
- AND MUST fail the build if the measured p99.9 exceeds the previously established static bound
- AND MUST generate a report comparing the new measurements against the baseline

### Requirement: Fail-Safe State Definitions
The system SHALL define fail-safe states for all failure modes: inference timeout returns error code (no partial results), memory exhaustion rejects new allocations (preserve existing), watchdog timeout triggers controlled reset with audit log flush, and capability violation denies and logs (no crash).

#### Scenario: Inference timeout fail-safe
- WHEN an ONNX inference operation exceeds its configured timeout
- THEN the system MUST terminate the inference and return an error code to the caller
- AND MUST NOT return partial inference results
- AND MUST release all tensor memory and GPU resources allocated for the timed-out inference
- AND MUST log the timeout event with the model identifier, elapsed time, and configured timeout value

#### Scenario: Memory exhaustion fail-safe
- WHEN a memory allocation request cannot be satisfied due to exhaustion of the buddy or slab allocator
- THEN the allocator MUST reject the new allocation and return an out-of-memory error
- AND MUST preserve all existing allocations without corruption
- AND MUST NOT trigger a panic, crash, or undefined behavior
- AND MUST log the exhaustion event with the requested size, allocator state, and requesting task ID

#### Scenario: Watchdog timeout fail-safe
- WHEN the hardware or software watchdog timer expires without receiving a heartbeat
- THEN the system MUST initiate a controlled reset sequence
- AND MUST flush all pending audit log entries to persistent storage before reset
- AND MUST revoke all outstanding capabilities
- AND MUST log the watchdog timeout event (to the extent possible before reset)
- AND the total time from watchdog expiry to reset completion MUST be bounded (configurable, default 100ms)

#### Scenario: Capability violation fail-safe
- WHEN a task attempts an operation without holding the required capability
- THEN the capability system MUST deny the operation and return a permission-denied error to the caller
- AND MUST NOT crash, panic, or terminate the offending task (unless configured for strict mode)
- AND MUST log the violation with the task ID, requested operation, missing capability, and timestamp
- AND the denied task MUST continue executing with its remaining valid capabilities

### Requirement: Fail-Safe State Documentation
Each fail-safe state SHALL document: trigger condition, system behavior during failure, recovery procedure, and maximum time to reach safe state.

#### Scenario: Inference timeout fail-safe documentation
- WHEN the fail-safe state documentation for inference timeout is reviewed
- THEN it MUST specify the trigger condition as inference execution time exceeding the per-model configurable timeout
- AND MUST specify the system behavior during failure as: terminate inference, deallocate resources, return error code
- AND MUST specify the recovery procedure as: caller retries with same or different input, or caller reports failure upstream
- AND MUST specify the maximum time to reach safe state as the configured timeout value plus a bounded cleanup overhead (documented per platform)

#### Scenario: Memory exhaustion fail-safe documentation
- WHEN the fail-safe state documentation for memory exhaustion is reviewed
- THEN it MUST specify the trigger condition as allocation failure from the buddy or slab allocator
- AND MUST specify the system behavior during failure as: reject allocation, preserve existing state, continue operation
- AND MUST specify the recovery procedure as: caller frees unused resources or waits for other tasks to release memory
- AND MUST specify the maximum time to reach safe state as zero (immediate rejection, no transition delay)

#### Scenario: Watchdog timeout fail-safe documentation
- WHEN the fail-safe state documentation for watchdog timeout is reviewed
- THEN it MUST specify the trigger condition as watchdog counter expiry without heartbeat refresh
- AND MUST specify the system behavior during failure as: flush audit logs, revoke capabilities, terminate tasks, trigger reset
- AND MUST specify the recovery procedure as: system reboots into known-good configuration, reloads models from persistent storage
- AND MUST specify the maximum time to reach safe state as the configured watchdog reset bound (default 100ms)

#### Scenario: Capability violation fail-safe documentation
- WHEN the fail-safe state documentation for capability violation is reviewed
- THEN it MUST specify the trigger condition as a capability check failure on any operation
- AND MUST specify the system behavior during failure as: deny operation, log event, continue task execution
- AND MUST specify the recovery procedure as: task requests the missing capability through authorized grant mechanism, or operator investigates the denial
- AND MUST specify the maximum time to reach safe state as zero (immediate denial, no transition delay)

### Requirement: OT-Specific Anomaly Detection
The system SHALL provide OT-specific anomaly detection: model output range validation (configurable bounds per output tensor), inference timing bounds (configurable per model), and input data validation (NaN/Inf detection, shape verification).

#### Scenario: Model output range validation
- WHEN an ONNX inference operation completes and produces output tensors
- THEN the anomaly detection system MUST check each output tensor element against configurable upper and lower bounds
- AND MUST flag any output element outside the configured range as an anomaly
- AND MUST report the anomaly with the model identifier, output tensor index, offending element index, actual value, and configured bounds
- AND the operator MUST be able to configure per-output-tensor bounds via the model manifest

#### Scenario: Inference timing bounds validation
- WHEN an ONNX inference operation completes
- THEN the anomaly detection system MUST compare the actual inference duration against configurable per-model timing bounds (minimum and maximum)
- AND MUST flag any inference that completes outside the configured timing window as an anomaly
- AND MUST report the anomaly with the model identifier, actual duration, and configured bounds
- AND the timing bounds MUST be configurable independently for each loaded model

#### Scenario: Input data NaN and Inf detection
- WHEN input data is submitted for ONNX inference
- THEN the anomaly detection system MUST scan all floating-point input tensor elements for NaN (Not a Number) and Inf (Infinity) values
- AND MUST reject the input and return a validation error if any NaN or Inf value is detected
- AND MUST log the rejection with the model identifier, input tensor index, and offending element index

#### Scenario: Input data shape verification
- WHEN input data is submitted for ONNX inference
- THEN the anomaly detection system MUST verify that the input tensor dimensions match the model's expected input shape
- AND MUST reject the input and return a shape mismatch error if the dimensions do not match
- AND MUST support both fixed-shape and dynamic-shape models (for dynamic axes, verify the dimension falls within configured minimum and maximum bounds)

#### Scenario: Anomaly detection configuration
- WHEN an ONNX model is loaded with its model manifest
- THEN the anomaly detection system MUST read the output range bounds, timing bounds, and input validation rules from the manifest
- AND MUST apply default anomaly detection settings (NaN/Inf rejection, shape verification) if no explicit configuration is provided
- AND MUST log the active anomaly detection configuration for the model at load time

### Requirement: Functional Safety Standards Cross-Reference
The system SHALL cross-reference functional safety standards: IEC 61508 (general), ISO 26262 ASIL D (automotive), and DO-178C DAL A (aviation) -- documenting which requirements satisfy which standard.

#### Scenario: IEC 61508 cross-reference
- WHEN the functional safety cross-reference documentation is reviewed for IEC 61508
- THEN it MUST map each OT security hardening requirement to the relevant IEC 61508 SIL (Safety Integrity Level) requirements
- AND MUST document the SIL level claimed for each mapped requirement
- AND MUST identify any gaps where OT security requirements do not fully satisfy IEC 61508 requirements

#### Scenario: ISO 26262 ASIL D cross-reference
- WHEN the functional safety cross-reference documentation is reviewed for ISO 26262
- THEN it MUST map each OT security hardening requirement to the relevant ISO 26262 ASIL D requirements
- AND MUST document which requirements satisfy ASIL D decomposition criteria
- AND MUST identify any requirements that satisfy a lower ASIL level and document the rationale

#### Scenario: DO-178C DAL A cross-reference
- WHEN the functional safety cross-reference documentation is reviewed for DO-178C
- THEN it MUST map each OT security hardening requirement to the relevant DO-178C DAL A objectives
- AND MUST document MC/DC (Modified Condition/Decision Coverage) requirements for each mapped component
- AND MUST trace each requirement to the DO-178C objective table (A-1 through A-10) with satisfaction evidence

#### Scenario: Multi-standard traceability matrix
- WHEN the full functional safety cross-reference is reviewed
- THEN it MUST include a traceability matrix with rows for each OT security hardening requirement and columns for IEC 61508, ISO 26262, and DO-178C
- AND each cell MUST indicate the specific clause, SIL/ASIL/DAL level, and satisfaction status (satisfied, partial, not applicable)
- AND the matrix MUST be maintained as a machine-readable artifact (CSV or structured table) for automated compliance reporting

### Requirement: Safe Shutdown Procedure
The safe shutdown procedure SHALL flush audit logs, revoke all capabilities, terminate all tasks, and trigger watchdog reset within bounded time (configurable, default 100ms).

#### Scenario: Audit log flush during shutdown
- WHEN the safe shutdown procedure is initiated
- THEN the system MUST flush all pending audit log entries to persistent storage as the first shutdown action
- AND MUST complete the flush within a bounded time (configurable, default 25ms of the total shutdown budget)
- AND MUST cryptographically sign the final log batch with ML-DSA-65 before flushing
- AND if the flush cannot complete within the time bound, the system MUST proceed with remaining shutdown steps and record the incomplete flush in the next boot log

#### Scenario: Capability revocation during shutdown
- WHEN the audit log flush completes or times out during safe shutdown
- THEN the system MUST revoke all outstanding capabilities for all tasks
- AND MUST complete the revocation within a bounded time (configurable, default 25ms of the total shutdown budget)
- AND after revocation, no task MUST be able to perform any capability-protected operation

#### Scenario: Task termination during shutdown
- WHEN all capabilities have been revoked during safe shutdown
- THEN the system MUST terminate all running tasks (SYSTEM, IPC, and INFERENCE priority classes)
- AND MUST release all memory and GPU resources held by terminated tasks
- AND MUST complete task termination within a bounded time (configurable, default 25ms of the total shutdown budget)

#### Scenario: Watchdog reset trigger
- WHEN all tasks have been terminated during safe shutdown
- THEN the system MUST trigger the hardware watchdog reset
- AND the total elapsed time from shutdown initiation to watchdog reset trigger MUST NOT exceed the configured shutdown time bound (default 100ms)
- AND if any shutdown phase exceeds its time bound, the system MUST skip remaining phases and trigger the watchdog reset immediately

#### Scenario: Safe shutdown triggered by operator command
- WHEN an operator issues a shutdown command via the management API
- THEN the system MUST execute the full safe shutdown procedure (flush, revoke, terminate, reset)
- AND MUST acknowledge the shutdown command before beginning the procedure
- AND MUST log the operator-initiated shutdown as an audit event before flushing

#### Scenario: Safe shutdown triggered by critical failure
- WHEN a critical failure is detected (unrecoverable hardware error, kernel integrity violation, or repeated watchdog near-misses)
- THEN the system MUST initiate the safe shutdown procedure automatically
- AND MUST log the triggering failure condition as the first audit event in the shutdown sequence
- AND MUST complete the shutdown within the configured time bound regardless of the failure type
