## ADDED Requirements

### Requirement: Extra UB Detection via cargo-careful
The CI pipeline SHALL run the test suite with cargo-careful to detect undefined behavior beyond what standard tests and Miri catch.

#### Scenario: Clean careful test run
- **WHEN** `cargo careful test` runs on all host-testable crates
- **AND** no extra UB is detected
- **THEN** the check MUST pass

#### Scenario: UB detected by careful
- **WHEN** `cargo careful test` detects undefined behavior (debug assertion failure, alignment violation, etc.)
- **THEN** the check MUST report the failure with the specific assertion and test name
- **AND** the check result MUST be advisory (continue-on-error) until promoted to a gate

#### Scenario: Runs only on host-testable crates
- **WHEN** the cargo-careful job executes
- **THEN** it MUST test only host-testable crates (kernel, security, onnx-rt, ipc, net, posix, container, bus, bench)
- **AND** MUST NOT attempt to run on bare-metal targets (x86_64-unknown-none, aarch64-unknown-none)
