## ADDED Requirements

### Requirement: Speculative-execution mitigations cross-reference

The `security` capability SHALL cross-reference the `kernel-security` speculative-execution mitigation matrix so that a reviewer auditing the security posture finds a single canonical pointer rather than scattered per-arch detail.

#### Scenario: Security review locates the speculation mitigations

- **GIVEN** a reviewer auditing the SmallAIOS security posture
- **WHEN** they look for speculative-execution side-channel coverage
- **THEN** the `security` spec SHALL point to the `kernel-security` capability and `docs/spec-exec-audit.md` as the authoritative source
- **AND** the cross-reference SHALL note that the constant-time cryptographic discipline (`pqc-crypto`) is a *separate* concern tracked independently — speculative-execution mitigations do not subsume side-channel-resistant crypto and vice versa

#### Scenario: Mitigation opt-outs are documented with their residual risk

- **GIVEN** a deployment that enables a performance opt-out (e.g. `spec-exec-ibpb-off`)
- **WHEN** the security posture is assessed
- **THEN** the residual Spectre v2 risk accepted by that opt-out SHALL be documented in `docs/spec-exec-audit.md`
- **AND** the `security` spec SHALL require that any such opt-out be an explicit, non-default Cargo feature
