# Delta for Verified Message Types

## ADDED Requirements

### Requirement: Verified Message Type Definition
The system SHALL define a `VerifiedMessageType` struct containing: a unique `MessageTypeId`, a human-readable name, the trust boundary it crosses, the data flow direction, a `SchemaHash` (SHA-3-256 of the formal proof artifact), an enforcement mode, and a static array of runtime-checkable invariants.

#### Scenario: Message type with complete proof
- GIVEN a `VerifiedMessageType` with a non-zero `schema_hash`
- THEN the type MUST be considered formally verified
- AND the `schema_hash` MUST correspond to a specific version of a Lean 4 or TLA+ proof artifact

#### Scenario: Message type without proof (in development)
- GIVEN a `VerifiedMessageType` with an all-zero `schema_hash`
- THEN the type MUST be considered unverified
- AND its enforcement mode SHOULD be `Permissive`

### Requirement: Message Type Registry
The system SHALL provide a `MessageTypeRegistry` with fixed-size storage (no heap allocation) that maps `MessageTypeId` to `VerifiedMessageType`. The registry MUST support lookup by ID and iteration over all registered types.

#### Scenario: Registry capacity
- GIVEN a `MessageTypeRegistry`
- THEN it MUST support at least 64 registered message types
- AND registration beyond capacity MUST return an error

#### Scenario: Lookup by ID
- GIVEN a registered message type with `MessageTypeId(0x0001)`
- WHEN `registry.lookup(MessageTypeId(0x0001))` is called
- THEN it MUST return the corresponding `VerifiedMessageType`

#### Scenario: Lookup of unregistered type
- GIVEN a `MessageTypeId` not present in the registry
- WHEN `registry.lookup()` is called
- THEN it MUST return `None`

### Requirement: Runtime Invariant Checking
Each `VerifiedMessageType` SHALL carry a static array of `Invariant` values that represent the runtime-checkable subset of properties proven by the formal verification artifact. The system MUST support the following invariant kinds:

- `MaxRank(u8)`: tensor rank upper bound
- `MinRank(u8)`: tensor rank lower bound
- `AllowedDtype(TensorDataType)`: permitted element type (multiple allowed via multiple invariants)
- `MaxElements(u32)`: total element count bound
- `MaxPayloadBytes(u32)`: wire-level size bound
- `ValueRange { min: i64, max: i64 }`: element value bounds
- `EnumMembership(u8)`: value must be a valid variant of the expected enum
- `NonZeroDimensions`: no zero-length dimensions in tensor shape
- `MonotonicTimestamp`: timestamp must be greater than previously seen (stateful)
- `RateLimit { max_per_sec: u32 }`: temporal rate bound (stateful)

#### Scenario: All invariants for a type must pass
- GIVEN a `VerifiedMessageType` with 3 invariants
- WHEN the type's invariants are checked against incoming data
- THEN ALL 3 invariants MUST pass for the check to succeed
- AND the first failing invariant's index MUST be reported in the deny reason

#### Scenario: Stateless invariant checking
- GIVEN an invariant of kind `MaxRank(4)`
- AND incoming tensor data with rank 3
- WHEN the invariant is checked
- THEN it MUST pass

#### Scenario: Stateful invariant checking (monotonic timestamp)
- GIVEN an invariant of kind `MonotonicTimestamp`
- AND the gate has recorded the last timestamp as T=100
- WHEN a message arrives with timestamp T=99
- THEN the invariant MUST fail

### Requirement: Schema Hash Lifecycle
When a formal proof artifact changes, the `SchemaHash` for any message type referencing that proof MUST change. A message type whose `schema_hash` does not match the currently loaded policy MUST be treated as unverified.

#### Scenario: Schema hash matches
- GIVEN a `VerifiedMessageType` in the policy with `schema_hash = H`
- AND the corresponding Lean 4 proof artifact hashes to H
- THEN the type is considered verified

#### Scenario: Schema hash mismatch after proof update
- GIVEN a Lean 4 proof artifact that has been updated (hash changed from H to H')
- AND the loaded policy still references hash H
- THEN the type MUST be treated as unverified until the policy is updated with hash H'

### Requirement: Initial Message Type Catalog
The compiled-in default policy SHALL include verified message types for all existing cross-boundary data flows:

| Type ID | Name | Boundary | Direction |
|---------|------|----------|-----------|
| 0x0001 | InferenceTensorInput | Network → Kernel | Inbound |
| 0x0002 | InferenceTensorOutput | Kernel → Network | Outbound |
| 0x0003 | InferenceRequest | Network → Kernel | Inbound |
| 0x0004 | InferenceResponse | Kernel → Network | Outbound |
| 0x0010 | BusSensorFrame | BusProtocol → Kernel | Inbound |
| 0x0011 | BusActuatorCommand | Kernel → BusProtocol | Outbound |
| 0x0020 | GpuTensorTransfer | Kernel → GPU | Outbound |
| 0x0021 | GpuInferenceResult | GPU → Kernel | Inbound |
| 0x0030 | K8sPodCommand | Kubernetes → Kernel | Inbound |
| 0x0031 | K8sHealthMetrics | Kernel → Kubernetes | Outbound |
| 0x0040 | IpcPubSubMessage | Kernel internal | Bidirectional |

#### Scenario: Default types registered at boot
- GIVEN a system booted with formal-gate enabled and no external policy blob
- THEN the compiled-in default policy MUST contain all types listed above
- AND each type MUST have at least one invariant defined
