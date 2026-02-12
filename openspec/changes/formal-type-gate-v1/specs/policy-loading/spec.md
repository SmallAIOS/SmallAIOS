# Delta for Policy Loading

## ADDED Requirements

### Requirement: Compiled-In Default Policy
The system SHALL include a compiled-in default `SecurityPolicy` as a `const` value containing the initial message type catalog, all enforcement modes set to `Permissive`, and an all-zero policy signature (indicating self-signed/built-in).

#### Scenario: Boot without external policy
- GIVEN no external policy blob is available at boot
- WHEN the `SecurityReady` phase initializes
- THEN the compiled-in default policy MUST be loaded
- AND all message types in the default catalog MUST be registered
- AND the global enforcement mode MUST be `Permissive`

#### Scenario: Default policy is valid
- GIVEN the compiled-in default policy
- THEN it MUST pass internal consistency validation (no duplicate type IDs, all invariants are well-formed)

### Requirement: Signed Policy Blob Format
External policy blobs SHALL follow a binary format containing: a 4-byte magic number (`0x53504F4C` = "SPOL"), a 4-byte version, the ML-DSA-65 signature over the payload, and the serialized policy payload (type registry, mode configuration, model whitelist).

#### Scenario: Valid blob parsing
- GIVEN a byte slice starting with magic `0x53504F4C` and version 1
- WHEN `SecurityPolicy::load_from_blob()` is called
- THEN it MUST parse the signature and payload
- AND verify the ML-DSA-65 signature against the system's policy verification key
- AND deserialize the policy payload into a `SecurityPolicy`

#### Scenario: Invalid magic rejected
- GIVEN a byte slice with incorrect magic number
- WHEN `SecurityPolicy::load_from_blob()` is called
- THEN it MUST return an error without attempting signature verification

#### Scenario: Signature verification failure
- GIVEN a validly formatted blob with an invalid signature
- WHEN `SecurityPolicy::load_from_blob()` is called
- THEN it MUST return a signature verification error
- AND MUST NOT load any data from the blob

### Requirement: Policy Loading at Boot (SecurityReady Phase)
During the `SecurityReady` boot phase, the system SHALL: (1) load the compiled-in default policy, (2) check for an external policy blob (memory-mapped region or environment-specified location), (3) if present, verify signature and swap in the external policy, (4) initialize the `SecurityGate` with the active policy.

#### Scenario: Boot with external policy blob
- GIVEN an external policy blob available at a known memory address
- WHEN the `SecurityReady` phase executes
- THEN the system MUST verify the blob's ML-DSA-65 signature
- AND IF valid, swap in the external policy replacing the default
- AND IF invalid, log a warning, retain the default policy, and continue boot

#### Scenario: Boot timing
- GIVEN the policy loading substep within `SecurityReady`
- THEN it MUST complete within the existing boot time budget
- AND MUST NOT add more than 5ms to the SecurityReady phase

### Requirement: Remote Policy Update
The system SHALL accept signed policy blobs over the mTLS-authenticated network boundary for runtime policy updates without reboot.

#### Scenario: Remote update accepted
- GIVEN a valid signed policy blob received over mTLS
- WHEN `SecurityPolicy::remote_update()` is called
- THEN the system MUST verify the ML-DSA-65 signature
- AND validate internal consistency (no duplicate type IDs, no enforcement demotion)
- AND atomically swap the active policy
- AND re-validate all loaded ONNX models against the new policy
- AND log the policy swap as an audit event with old and new policy hashes

#### Scenario: Remote update rejected — signature invalid
- GIVEN a policy blob with an invalid ML-DSA-65 signature received over mTLS
- WHEN `SecurityPolicy::remote_update()` is called
- THEN the update MUST be rejected
- AND the existing policy MUST remain unchanged
- AND the rejection MUST be logged as a security audit event

#### Scenario: Remote update rejected — enforcement demotion
- GIVEN a policy blob that demotes any message type from Enforcing to Permissive
- WHEN `SecurityPolicy::remote_update()` is called
- THEN the update MUST be rejected with reason "enforcement demotion not permitted"
- AND the existing policy MUST remain unchanged

#### Scenario: Model re-validation after policy update
- GIVEN a loaded ONNX model with hash H
- AND a new policy that does not include H in its model whitelist
- WHEN the policy is swapped
- THEN the model MUST be unloaded
- AND the unloading MUST be logged as an audit event

#### Scenario: Rollback on re-validation failure
- GIVEN a new policy that would cause all loaded models to be unloaded
- WHEN re-validation fails for critical models
- THEN the system MUST support rollback to the previous policy via management command
- AND the previous policy MUST be retained in memory (not discarded on swap)

### Requirement: Model Whitelist in Policy
The `SecurityPolicy` SHALL contain a `ModelWhitelist` — a fixed-size list of allowed model hashes (SHA-3-256). Only models whose hash appears in the whitelist may be loaded when the formal-gate feature is enabled.

#### Scenario: Model in whitelist
- GIVEN a model whose SHA-3-256 hash matches an entry in the policy's whitelist
- WHEN model loading is attempted at `ModelsLoaded` phase
- THEN the gate MUST allow model loading to proceed

#### Scenario: Model not in whitelist
- GIVEN a model whose SHA-3-256 hash does not match any whitelist entry
- WHEN model loading is attempted
- THEN the gate MUST block model loading
- AND MUST log the rejection with the model hash and the reason "model not whitelisted"

#### Scenario: Whitelist capacity
- GIVEN the `ModelWhitelist`
- THEN it MUST support at least 32 model hashes
- AND MUST use fixed-size storage (no heap allocation)

## MODIFIED Requirements

### Requirement: Boot Sequence Gains Policy Substep
The `SecurityReady` boot phase SHALL include a policy loading substep that executes after capability registry initialization and CSPRNG seeding.

#### Scenario: Phase ordering preserved
- GIVEN the 9-phase boot sequence
- WHEN the policy substep is added to SecurityReady
- THEN the phase ordering MUST remain: ConfigLoaded → MemoryReady → SchedulerReady → SecurityReady → NetworkReady → IpcReady → RuntimeReady → ModelsLoaded → Ready
- AND the policy substep MUST execute within SecurityReady (not as a separate phase)

### Requirement: ModelsLoaded Phase Gains Policy Validation
The `ModelsLoaded` boot phase SHALL validate each model against the loaded `SecurityPolicy` before permitting execution.

#### Scenario: Model validated at load
- GIVEN the `ModelsLoaded` phase with formal-gate enabled
- WHEN each ONNX model is loaded
- THEN the system MUST check the model's hash against the policy whitelist
- AND MUST verify the model's declared input/output shapes are compatible with registered `VerifiedMessageType` invariants
- AND MUST grant model execution capability only if validation passes
