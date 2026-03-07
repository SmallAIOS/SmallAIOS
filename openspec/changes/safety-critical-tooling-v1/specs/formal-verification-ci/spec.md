## ADDED Requirements

### Requirement: Kani proofs for critical unsafe code
All `unsafe` blocks in memory management (buddy allocator, slab allocator, tensor pool), kernel state, and constant-time crypto SHALL have Kani proof harnesses that verify absence of panics, out-of-bounds access, and arithmetic overflow.

#### Scenario: Kani verifies buddy allocator safety
- **WHEN** Kani runs on `kernel/src/mem/buddy.rs` proof harnesses
- **THEN** verification SHALL succeed proving no panics, no out-of-bounds, and no integer overflow for all inputs within the specified bounds

#### Scenario: Kani CI job fails on regression
- **WHEN** a code change introduces a potential panic in a verified function
- **THEN** the Kani CI job SHALL fail and report the counterexample

### Requirement: Miri UB detection in CI
The full test suite SHALL be run under Miri on a weekly schedule to detect undefined behavior in `unsafe` code.

#### Scenario: Miri detects use-after-free
- **WHEN** Miri runs the test suite and encounters a use-after-free
- **THEN** the CI job SHALL fail with a diagnostic pointing to the offending code

#### Scenario: Weekly Miri run on nightly
- **WHEN** the weekly CI schedule triggers
- **THEN** Miri SHALL run all host-testable crate tests and report results
