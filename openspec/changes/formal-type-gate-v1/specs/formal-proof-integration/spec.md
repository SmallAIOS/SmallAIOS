# Delta for Formal Proof Integration

## ADDED Requirements

### Requirement: Schema Hash Links Types to Proofs
Each `VerifiedMessageType` SHALL carry a `SchemaHash` (32-byte SHA-3-256 digest) that uniquely identifies the formal proof artifact (Lean 4 theorem file or TLA+ specification) that verified the type's properties. The hash MUST be computed over the complete proof source file.

#### Scenario: Hash computed from Lean 4 proof
- GIVEN a Lean 4 proof file `formal/lean4/TensorTypeInvariants.lean`
- WHEN its SHA-3-256 hash is computed
- THEN the resulting 32-byte hash MUST be stored as the `schema_hash` of the corresponding `VerifiedMessageType`

#### Scenario: Hash computed from TLA+ specification
- GIVEN a TLA+ specification file `formal/tla/SecurityGate.tla`
- WHEN its SHA-3-256 hash is computed
- THEN the resulting 32-byte hash MUST be stored as the `schema_hash` of the corresponding `VerifiedMessageType`

### Requirement: Lean 4 Integrity Lattice Proof
A Lean 4 proof file `formal/lean4/IntegrityLattice.lean` SHALL prove that the three `IntegrityLevel` values (Low, Medium, High) form a valid lattice, and that the Biba no-write-up property holds: for all data flow from source to destination, `source.integrity >= destination.integrity` is required (or explicit promotion).

#### Scenario: Lattice ordering is total
- GIVEN the three integrity levels
- THEN the proof MUST show `Low ≤ Medium ≤ High` forms a total order

#### Scenario: Biba property holds
- GIVEN a data flow function parameterized by source and destination integrity
- THEN the proof MUST show that `allow(src, dst)` implies `src.integrity ≥ dst.integrity` OR the flow passes through a promotion gate

### Requirement: Lean 4 Message Type Properties Proof
A Lean 4 proof file `formal/lean4/MessageTypeProperties.lean` SHALL prove that the `MessageTypeRegistry` is well-formed: no duplicate type IDs, invariant check functions are total (always terminate), and the invariant check is deterministic (same input always gives same result).

#### Scenario: No duplicate IDs
- GIVEN a registry with N types
- THEN the proof MUST show all N type IDs are distinct

#### Scenario: Invariant totality
- GIVEN any invariant and any input data
- THEN the proof MUST show the invariant check terminates with either pass or fail

### Requirement: TLA+ Security Gate State Machine
A TLA+ specification `formal/tla/SecurityGate.tla` SHALL model the gate state machine and verify:
1. **Safety**: No data crosses a boundary without a gate check
2. **Monotonicity**: Enforcement mode transitions are one-directional (Permissive→Enforcing)
3. **Atomicity**: Policy swap is atomic — no gate check executes against a partially loaded policy
4. **Liveness**: Gate checks always terminate (no deadlock in check sequence)

#### Scenario: Model check passes
- WHEN TLC is run on `SecurityGate.tla`
- THEN all safety and liveness properties MUST be verified with no counterexamples

### Requirement: TLA+ Policy Update Protocol
A TLA+ specification `formal/tla/PolicyUpdate.tla` SHALL model the remote policy update protocol and verify:
1. **Authentication**: Policy blob is accepted only after ML-DSA-65 signature verification
2. **Atomicity**: Old policy remains active during validation; swap is instantaneous
3. **Rollback**: If new policy causes model re-validation failure, old policy is restored
4. **Monotonicity**: Enforcing types cannot be demoted to Permissive

#### Scenario: Model check passes
- WHEN TLC is run on `PolicyUpdate.tla`
- THEN all safety properties MUST be verified with no counterexamples
