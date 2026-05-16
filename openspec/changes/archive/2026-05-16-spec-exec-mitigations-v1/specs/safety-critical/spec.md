## ADDED Requirements

### Requirement: Speculative-execution mitigation evidence in the safety case

The DO-178C safety case SHALL cite the `kernel-security` speculative-execution mitigation matrix as the canonical evidence for transient-execution side-channel coverage, rather than restating per-architecture mitigation detail in the safety-critical spec.

#### Scenario: Safety case references the canonical matrix

- **GIVEN** a DO-178C reviewer assembling the transient-execution side-channel argument
- **WHEN** they request evidence of Spectre/Meltdown/Retbleed coverage
- **THEN** the safety case SHALL cite the `kernel-security` capability spec and `docs/spec-exec-audit.md` as the single canonical trust-boundary × architecture × attack-class matrix
- **AND** the safety case SHALL cite the unikernel single-address-space model (`docs/architecture.md`) as the structural evidence for Meltdown immunity
- **AND** the safety-critical spec SHALL NOT duplicate the matrix — duplication risks divergence of the audit trail

#### Scenario: New speculative-execution CVE triggers safety-case re-review

- **GIVEN** a new CVE in the speculative-execution attack class
- **WHEN** the `kernel-security` re-audit OpenSpec change lands (per its Review Trigger)
- **THEN** the safety case owner SHALL re-review whether the safety-critical argument is still discharged
- **AND** SHALL record the re-review outcome alongside the updated `docs/spec-exec-audit.md`
