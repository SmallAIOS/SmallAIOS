## 1. Foundation Types and Feature Flag

- [x] 1.1 Add `formal-gate` feature flag to `security/Cargo.toml` (default off)
- [x] 1.2 Implement `IntegrityLevel` enum (Low/Medium/High) with `PartialOrd`, `Ord`, `from_u8`, `as_str` in `security/src/labels.rs`
- [x] 1.3 Implement `MessageTypeId(u32)` newtype in `security/src/labels.rs`
- [x] 1.4 Implement `SecurityLabel` struct (classification + integrity + message_type) in `security/src/labels.rs`; compiles to ZST when `formal-gate` disabled
- [x] 1.5 Implement `EnforcementMode` enum (Enforcing/Permissive) in `security/src/enforcement.rs`
- [x] 1.6 Implement `GateVerdict` enum (Allowed/Denied/PermissivePass) and `DenyReason` enum in `security/src/enforcement.rs`
- [x] 1.7 Export new modules from `security/src/lib.rs`; gate all new modules behind `#[cfg(feature = "formal-gate")]`
- [x] 1.8 Unit tests for all foundation types: construction, comparison, ordering, serialization, feature-flag conditional compilation

## 2. Verified Message Types and Invariant System

- [x] 2.1 Implement `Invariant` enum (MaxRank, MinRank, AllowedDtype, MaxElements, MaxPayloadBytes, ValueRange, EnumMembership, NonZeroDimensions, MonotonicTimestamp, RateLimit) in `security/src/message_types.rs`
- [x] 2.2 Implement `SchemaHash` type alias (`[u8; 32]`) and `VerifiedMessageType` struct in `security/src/message_types.rs`
- [x] 2.3 Implement `InvariantChecker` — stateless checks (rank, dtype, elements, payload size, value range, enum, dimensions) and stateful checks (monotonic timestamp, rate limit) with per-type state tracking
- [x] 2.4 Implement `MessageTypeRegistry` — fixed-size array (`[Option<VerifiedMessageType>; 64]`), register, lookup by ID, iterate, count
- [x] 2.5 Implement the default message type catalog (11 types: InferenceTensorInput, InferenceTensorOutput, InferenceRequest, InferenceResponse, BusSensorFrame, BusActuatorCommand, GpuTensorTransfer, GpuInferenceResult, K8sPodCommand, K8sHealthMetrics, IpcPubSubMessage) as `const` entries
- [x] 2.6 Unit tests: registry CRUD, invariant checking (all 10 invariant kinds), stateful invariant state tracking, catalog completeness

## 3. Security Gate

- [x] 3.1 Implement `BoundaryCrossing` struct (boundary, direction, task_id, label, data reference) in `security/src/gate.rs`
- [x] 3.2 Implement `SecurityGate` struct with policy reference, default mode, and statistics counters
- [x] 3.3 Implement `SecurityGate::check()` — 5-layer verification pipeline: (1) capability, (2) classification, (3) integrity (Biba), (4) message type + invariants, (5) mode resolution → verdict
- [x] 3.4 Implement hierarchical mode resolution: type mode → boundary mode → global mode
- [x] 3.5 Implement audit emission on every gate decision (integrate with `security/src/audit/`)
- [x] 3.6 Implement no-op `SecurityGate` when `formal-gate` disabled (unconditional `Allowed`, ZST)
- [x] 3.7 Unit tests: all 5 layers independently, layer composition, mode resolution hierarchy, Enforcing vs Permissive verdicts, audit emission, no-op path, statistics counters

## 4. Security Policy and Loading

- [x] 4.1 Implement `ModelWhitelist` — fixed-size array (`[[u8; 32]; 32]`), add, contains, count
- [x] 4.2 Implement `SecurityPolicy` struct (type_registry, model_whitelist, boundary_modes, global_mode, policy_hash, policy_signature, version)
- [x] 4.3 Implement compiled-in default `SecurityPolicy` as `const` (all default types, Permissive modes, zero signature)
- [x] 4.4 Implement `SecurityPolicy::load_from_blob()` — parse magic (0x53504F4C), version, ML-DSA-65 signature, payload; verify signature; deserialize
- [x] 4.5 Implement `SecurityPolicy::remote_update()` — verify signature, validate consistency (no enforcement demotion, no duplicate type IDs), atomic swap, retain old policy for rollback
- [x] 4.6 Implement model re-validation after policy swap: check all loaded models against new whitelist, unload non-compliant models
- [x] 4.7 Unit tests: blob parsing (valid, invalid magic, bad signature), remote update (accept, reject demotion, reject bad signature), model whitelist CRUD, rollback

## 5. Boundary Integration

- [x] 5.1 Extend `DataFlow` struct with `expected_message_type: Option<MessageTypeId>` field; update `CROSS_BOUNDARY_FLOWS` entries; ensure existing tests pass
- [x] 5.2 Extend `BoundaryDefinition` struct with `default_mode: EnforcementMode` field; update `BOUNDARY_DEFINITIONS` entries (default to Permissive); ensure existing tests pass
- [x] 5.3 Add `SecurityLabel` field to IPC `Message` struct in `ipc/src/pubsub.rs`; conditionally compiled (`formal-gate`)
- [x] 5.4 Add gate check call site at network boundary (inbound inference requests in `ipc/src/inference_proto.rs` or `net/` ingress path)
- [x] 5.5 Add gate check call site at model loading (`onnx-rt/src/session.rs` — `Session::initialize()` validates model hash against policy whitelist and I/O shapes against registered types)
- [x] 5.6 Integration tests: end-to-end gate check across Network→Kernel flow, Bus→Kernel flow, Kernel→GPU flow; Enforcing rejection; Permissive pass-through

## 6. Boot Sequence Integration

- [x] 6.1 Add policy loading substep within `SecurityReady` phase in `container/src/boot.rs`: load default policy, check for external blob, verify and swap if present, initialize SecurityGate
- [x] 6.2 Add model-vs-policy validation in `ModelsLoaded` phase: hash check, I/O shape conformance, capability grant gated on validation
- [x] 6.3 Update `ContainerConfig` in `container/src/config.rs` with policy-related fields: `policy_blob_address`, `policy_verification_key`, `formal_gate_enabled`
- [x] 6.4 Integration tests: boot with default policy, boot with external blob (valid and invalid), model rejection on whitelist miss

## 7. Formal Verification Artifacts

- [x] 7.1 Write `formal/lean4/IntegrityLattice.lean` — prove Low ≤ Medium ≤ High forms total order; prove Biba no-write-up property
- [x] 7.2 Write `formal/lean4/MessageTypeProperties.lean` — prove registry well-formedness (no duplicate IDs); prove invariant check totality and determinism
- [x] 7.3 Write `formal/lean4/TensorTypeInvariants.lean` — prove tensor invariants (rank bounds, dtype membership) are sound with respect to ONNX type system
- [x] 7.4 Write `formal/lean4/LabelComposition.lean` — prove SecurityLabel comparison composes correctly with ClassificationLevel and IntegrityLevel orderings
- [x] 7.5 Write `formal/tla/SecurityGate.tla` + `.cfg` — model gate state machine; verify safety (no unchecked crossing), monotonicity (mode transitions), atomicity (policy swap), liveness (checks terminate)
- [x] 7.6 Write `formal/tla/PolicyUpdate.tla` + `.cfg` — model remote update protocol; verify authentication-before-swap, atomicity, rollback, monotonicity
- [x] 7.7 Run TLC on both TLA+ models; verify all properties pass with no counterexamples
- [x] 7.8 Compute SHA-3-256 hashes of all proof files; update default message type catalog with correct `schema_hash` values

## 8. Testing and CI

- [x] 8.1 Achieve 100% MC/DC coverage on gate.rs, labels.rs, enforcement.rs, message_types.rs, policy.rs
- [x] 8.2 Fuzz invariant checking with random tensor metadata (shape, dtype, element count, payload sizes)
- [x] 8.3 Fuzz policy blob parsing with random byte sequences — verify no panics
- [x] 8.4 Add `formal-gate` feature to CI matrix: run full test suite with feature enabled and disabled
- [x] 8.5 Add TLA+ SecurityGate and PolicyUpdate models to CI verification job
- [x] 8.6 Verify binary size impact: measure kernel binary with and without `formal-gate`, ensure < 15 MB
