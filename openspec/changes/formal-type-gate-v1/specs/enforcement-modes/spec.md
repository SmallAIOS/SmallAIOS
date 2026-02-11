# Delta for Enforcement Modes

## ADDED Requirements

### Requirement: Enforcement Mode Definition
The system SHALL define an `EnforcementMode` enum with two variants: `Enforcing` (hard reject on violation) and `Permissive` (log and allow on violation).

#### Scenario: Enforcing mode behavior
- GIVEN `EnforcementMode::Enforcing`
- WHEN a gate check layer fails
- THEN the gate MUST return `GateVerdict::Denied`
- AND the data MUST NOT cross the trust boundary

#### Scenario: Permissive mode behavior
- GIVEN `EnforcementMode::Permissive`
- WHEN a gate check layer fails
- THEN the gate MUST return `GateVerdict::PermissivePass`
- AND the violation MUST be logged to the audit chain with full context (boundary, direction, task, label, failing layer, deny reason)
- AND the data MUST be allowed to cross the trust boundary

### Requirement: Hierarchical Mode Resolution
The enforcement mode for a given gate check SHALL be resolved hierarchically: per-message-type mode takes priority over per-boundary mode, which takes priority over global mode. This allows fine-grained control.

#### Scenario: Per-type mode overrides per-boundary mode
- GIVEN a message type with `mode: Enforcing`
- AND the boundary has `default_mode: Permissive`
- AND the global mode is `Permissive`
- WHEN the enforcement mode is resolved
- THEN the resolved mode MUST be `Enforcing` (type takes priority)

#### Scenario: Per-boundary mode overrides global when type has no explicit mode
- GIVEN a message type with no explicit mode (uses default)
- AND the boundary has `default_mode: Enforcing`
- AND the global mode is `Permissive`
- WHEN the enforcement mode is resolved
- THEN the resolved mode MUST be `Enforcing` (boundary overrides global)

#### Scenario: Global mode used as fallback
- GIVEN a message with an unknown type (not in registry)
- AND the boundary has no explicit mode override
- AND the global mode is `Permissive`
- WHEN the enforcement mode is resolved
- THEN the resolved mode MUST be `Permissive` (global fallback)

### Requirement: Graduation Lifecycle
The system SHALL support a message type graduation lifecycle: Untyped → Permissive → Enforcing. Transitions MUST be unidirectional (Permissive→Enforcing only, never Enforcing→Permissive via remote update).

#### Scenario: Untyped message (type not in registry)
- GIVEN a message with a `MessageTypeId` not present in the registry
- WHEN the gate evaluates the message type layer
- THEN it MUST treat the message as having failed the type check
- AND the enforcement mode resolution determines whether it's denied or permissive-passed

#### Scenario: Promotion from Permissive to Enforcing
- GIVEN a policy update that changes a message type's mode from `Permissive` to `Enforcing`
- WHEN the update is applied
- THEN the new mode MUST take effect immediately for subsequent gate checks
- AND the transition MUST be logged as an audit event

#### Scenario: Demotion from Enforcing to Permissive blocked
- GIVEN a policy update that attempts to change a message type's mode from `Enforcing` to `Permissive`
- WHEN the policy update is validated
- THEN the update MUST be rejected
- AND the rejection MUST be logged with the reason "enforcement mode demotion not permitted"
- AND the existing policy MUST remain unchanged

### Requirement: Mode Configuration in Policy
The `SecurityPolicy` SHALL support specifying enforcement modes at three levels: global default, per-boundary default (array of 5 modes, one per `TrustBoundary`), and per-message-type (stored in `VerifiedMessageType`).

#### Scenario: Policy contains all mode levels
- GIVEN a loaded `SecurityPolicy`
- THEN it MUST contain a `global_mode: EnforcementMode`
- AND it MUST contain a `boundary_modes: [EnforcementMode; 5]` array indexed by `TrustBoundary` discriminant
- AND each `VerifiedMessageType` in the registry MUST contain a `mode: EnforcementMode`
