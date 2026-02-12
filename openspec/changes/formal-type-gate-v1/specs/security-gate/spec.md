# Delta for Security Gate

## ADDED Requirements

### Requirement: Security Gate at Trust Boundaries
The system SHALL provide a `SecurityGate` that enforces layered verification on all data crossing trust boundaries: (1) capability check, (2) classification level check, (3) integrity level check (Biba model), (4) verified message type check, (5) enforcement mode resolution.

#### Scenario: Gate checks all layers in order
- GIVEN a `BoundaryCrossing` with a `SecurityLabel` and task context
- WHEN `SecurityGate::check()` is called
- THEN it MUST evaluate layers 1 through 4 in order
- AND MUST stop at the first failing layer
- AND MUST resolve the enforcement mode (Enforcing/Permissive) for the final verdict
- AND MUST emit an audit event recording the boundary, direction, task, label, verdict, and failing layer (if any)

#### Scenario: Gate returns Allowed when all layers pass
- GIVEN a crossing where the task holds required capability, classification permits flow, integrity direction is valid, and message type matches a registered VerifiedMessageType with all invariants satisfied
- WHEN `SecurityGate::check()` is called
- THEN it MUST return `GateVerdict::Allowed`

#### Scenario: Gate returns Denied in Enforcing mode
- GIVEN a crossing that fails any verification layer
- AND the resolved enforcement mode is `Enforcing`
- WHEN `SecurityGate::check()` is called
- THEN it MUST return `GateVerdict::Denied` with the failing layer index and deny reason
- AND the data MUST NOT proceed past the trust boundary

#### Scenario: Gate returns PermissivePass in Permissive mode
- GIVEN a crossing that fails any verification layer
- AND the resolved enforcement mode is `Permissive`
- WHEN `SecurityGate::check()` is called
- THEN it MUST return `GateVerdict::PermissivePass` with the failing layer index and deny reason
- AND the data MUST be allowed to proceed past the trust boundary
- AND the violation MUST be logged to the audit chain

#### Scenario: Gate tracks statistics
- GIVEN a running `SecurityGate`
- THEN it MUST maintain counters for: total checks, total denials, total permissive passes
- AND these counters MUST be queryable for metrics export

### Requirement: Gate Compiles to No-Op Without Feature Flag
The `SecurityGate` SHALL be gated behind the `formal-gate` feature flag on the security crate. When the feature is disabled, `SecurityGate::check()` MUST compile to a no-op that unconditionally returns `GateVerdict::Allowed` with zero runtime overhead.

#### Scenario: Feature flag disabled
- GIVEN a build with `formal-gate` feature disabled
- WHEN `SecurityGate::check()` is called
- THEN it MUST return `GateVerdict::Allowed` unconditionally
- AND MUST NOT perform any layer checks
- AND `SecurityLabel` MUST be a zero-size type

#### Scenario: Feature flag enabled
- GIVEN a build with `formal-gate` feature enabled
- WHEN `SecurityGate::check()` is called
- THEN it MUST perform all layer checks as specified

### Requirement: Integrity Level Check (Biba Model)
The security gate SHALL enforce the Biba integrity model: data at a lower integrity level MUST NOT flow to a higher-integrity destination without passing through an explicit integrity promotion gate.

#### Scenario: Low integrity to Medium destination allowed
- GIVEN data labeled `IntegrityLevel::Low` crossing to a `Medium`-integrity destination
- WHEN the integrity layer is evaluated
- THEN it MUST return allowed (Low can flow to Medium or Low)

#### Scenario: Medium integrity to High destination blocked
- GIVEN data labeled `IntegrityLevel::Medium` crossing to a `High`-integrity destination
- WHEN the integrity layer is evaluated
- THEN it MUST return denied with `DenyReason::IntegrityViolation`

#### Scenario: Integrity promotion gate
- GIVEN data that has passed through an explicit validation and promotion step (range check, rate limit, authorized task signature)
- WHEN the data's integrity is promoted from Medium to High
- THEN the promotion MUST be logged as an audit event
- AND the data MUST carry the promoted label for subsequent checks

## MODIFIED Requirements

### Requirement: DataFlow Gains Message Type Reference
The existing `DataFlow` struct in `boundary/data_flow_auth.rs` SHALL be extended with an optional `expected_message_type: Option<MessageTypeId>` field linking each cross-boundary flow to its verified message type.

#### Scenario: Existing flows retain behavior
- GIVEN the 8 existing `CROSS_BOUNDARY_FLOWS` entries
- WHEN the `expected_message_type` field is added
- THEN existing flows MAY have `expected_message_type: None` (backward compatible)
- AND all existing tests MUST continue to pass

### Requirement: BoundaryDefinition Gains Enforcement Mode
The existing `BoundaryDefinition` struct in `boundary/trust_boundaries.rs` SHALL be extended with a `default_mode: EnforcementMode` field specifying the default enforcement mode for that boundary.

#### Scenario: Default enforcement modes
- GIVEN the 5 existing `BOUNDARY_DEFINITIONS` entries
- WHEN the `default_mode` field is added
- THEN the default value MUST be `EnforcementMode::Permissive` for backward compatibility
