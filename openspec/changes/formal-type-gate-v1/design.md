## Context

SmallAIOS enforces *who* can access *what* via capability tokens (`CapRegistry`) and validates *how* data flows across 5 trust boundaries (Kernel, Kubernetes, Network, BusProtocol, GPU) with authentication, integrity, and confidentiality checks. Data classification (`ClassificationLevel::Public | Internal | Restricted`) prevents sensitive data from leaking to lower-trust destinations.

What's missing is enforcement of *what shape* data must be when crossing a boundary. An authenticated, integrity-protected message can still carry malformed tensors, ill-typed bus frames, or unexpected IPC payloads. The `DataFlow` struct in `boundary/data_flow_auth.rs` describes the pipe but not the data it carries.

The ONNX inference protocol (`ipc/src/inference_proto.rs`) already defines wire types (`TensorData`, `InferenceRequest`, `InferenceResponse`) but these are just parsing types — they don't carry invariant guarantees. A `TensorData` with `shape: [0, 0, 0, 0]` parses successfully but is semantically invalid.

This change introduces a **formal methods firewall**: a type-checking gate at each trust boundary that validates data against formally verified message type schemas, with configurable enforcement modes. The security policy — including the verified type registry and enforcement configuration — loads independently from the ONNX model and can be updated remotely via signed policy blobs.

## Goals / Non-Goals

**Goals:**
- Every message crossing a trust boundary must match a registered `VerifiedMessageType` with runtime-checkable invariants derived from Lean 4 / TLA+ proofs
- Security policy loads at `SecurityReady` (boot phase 4), independently of models at `ModelsLoaded` (phase 8); model loading is gated by policy
- Permissive/Enforcing modes configurable per-message-type, enabling a graduation lifecycle: untyped → permissive → enforcing
- MAC security labels (`ClassificationLevel` + `IntegrityLevel` + `MessageTypeId`) attach to all boundary-crossing data, augmenting capability checks
- Remote policy update: signed policy blobs received over mTLS, verified with ML-DSA-65, hot-swapped without reboot
- All gate decisions logged to tamper-evident audit chain
- Zero overhead when formal verification feature flag is disabled
- Fixed-size, no-heap data structures consistent with existing `#![no_std]` patterns

**Non-Goals:**
- Replacing the capability system (labels augment it, don't replace it)
- Runtime re-execution of Lean 4 / TLA+ proofs (only the extracted runtime invariants are checked)
- Dynamic policy generation (policies are authored offline, signed, deployed)
- Per-packet crypto verification at GPU DMA rates (gate validates at session/transfer setup, not per-DMA-descriptor)
- Full dependent type system at runtime (invariants are a pragmatic subset)

## Decisions

### Decision 1: Labels augment capabilities (Option B from exploration)

**Choice**: MAC security labels are a separate, orthogonal enforcement layer. Every data item gets a label. Every boundary crossing checks both the task's capability AND the data's label.

**Rationale**: The capability system (`CapRegistry`, 4096 slots, bitmap revocation) is well-tested and proven via TLA+. Labels add information flow control (confidentiality + integrity direction) and type verification without disturbing capability semantics. The two systems compose: capability says "this task may access this resource", label says "this data may flow in this direction at this classification level with this verified type."

**Alternatives considered**:
- Option A (Labels replace capabilities): Would discard proven capability infrastructure and TLA+ proofs. No benefit over augmentation.
- Option C (Labels as capability metadata): Couples two orthogonal concerns. Makes capability revocation interact with classification changes.

### Decision 2: Security policy loaded independently from ONNX models

**Choice**: The security policy (verified type registry, enforcement modes, model hash whitelist) is a distinct artifact that loads at `SecurityReady` phase. ONNX models load later at `ModelsLoaded` and are validated against the already-loaded policy. Models carry no security metadata of their own.

**Rationale**: Models are untrusted data. They must not self-declare their security posture — that would be equivalent to letting a binary set its own SELinux context. Independent loading means: (a) policy can be audited without the model, (b) policy survives model updates, (c) one policy can govern multiple models, (d) policy signing uses a different key than model signing.

**Risk**: Policy and model can drift (policy expects types the model doesn't produce) → Mitigated by gate validation at model load time, which rejects the model if its declared I/O doesn't match registered message types.

### Decision 3: Per-message-type enforcement modes with graduation lifecycle

**Choice**: Each `VerifiedMessageType` carries its own `EnforcementMode` (Enforcing/Permissive). A global default applies to types without explicit mode. This enables graduation: new types start Permissive while proofs are developed, then promote to Enforcing once the Lean 4 proof is complete and the schema hash is locked.

**Rationale**: Global-only modes (all Enforcing or all Permissive) are too coarse. Per-boundary modes don't account for multiple message types crossing the same boundary. Per-type is the natural granularity because the formal proof is per-type.

**Lifecycle**:
```
Untyped (rejected)  →  Permissive (logged, allowed)  →  Enforcing (hard gate)
```
- Untyped: message has no type tag, or type tag not in registry → rejected in Enforcing mode, logged in Permissive
- Permissive: type is registered but proof is incomplete → invariant violations logged but data flows through
- Enforcing: type has completed proof (schema_hash set) → invariant violations cause hard rejection

### Decision 4: Remote policy update via signed blob over mTLS

**Choice**: Policy can be updated at runtime by delivering a signed policy blob over the mTLS-authenticated network boundary. The blob is verified using ML-DSA-65 (the system's existing PQC signature scheme) before being swapped in. No reboot required.

**Rationale**: In production deployments, security teams need to update policy (add types, promote Permissive→Enforcing, revoke model hashes) without redeploying the kernel. Memory-mapped compiled-in policy is the default, but remote update is essential for fleet management.

**Security properties**:
- Policy blob signed with ML-DSA-65 by an offline signing key
- Delivered over existing mTLS channel (dual auth: transport + payload signature)
- Atomic swap: old policy remains active until new policy is fully validated
- Audit log records every policy swap with old/new hashes
- Rollback: previous policy retained in memory, restorable via management command

**Alternatives considered**:
- Compile-in only: Too rigid for production fleet management.
- Filesystem-based: SmallAIOS has no general-purpose filesystem. Virtual FS is read-only.
- Unsigned update: Violates trust model — policy is a security-critical artifact.

### Decision 5: Integrity levels follow Biba model (no write-up, no read-down)

**Choice**: Three integrity levels (Low, Medium, High) following the Biba integrity model. Data at a lower integrity level cannot flow to a higher-integrity destination. Combined with Bell-LaPadula confidentiality (no read-up, no write-down from classification levels), this creates a lattice.

**Rationale**: The existing `ClassificationLevel` handles confidentiality (prevents data leaking down). But it doesn't prevent untrusted data flowing *up* — a Low-integrity network packet shouldn't directly modify High-integrity actuator commands. Biba complements Bell-LaPadula to create bidirectional flow control.

**Mapping to boundaries**:
- Network inbound: Low integrity (untrusted external source)
- Bus sensor data: Medium integrity (authenticated but not cryptographically verified)
- Kernel internal: High integrity (generated by trusted code)
- Inference output: Medium integrity (computed from possibly Low-integrity input)
- Actuator commands: High integrity required (safety-critical)

This means inference output (Medium) cannot directly drive actuators (High) without passing through a verified validation gate that promotes integrity — a deliberate chokepoint.

### Decision 6: Feature-flag gated — zero cost when disabled

**Choice**: The entire formal type gate is behind a feature flag (`formal-gate` on the security crate). When disabled, `SecurityGate` compiles away to a no-op, `SecurityLabel` is a zero-size type, and the boot sequence skips policy loading.

**Rationale**: Development builds need fast iteration. The gate adds latency at every boundary crossing. For dev/test without safety-critical requirements, it should be invisible.

**Implementation**: `#[cfg(feature = "formal-gate")]` on gate types and enforcement code. `SecurityLabel` is `()` when disabled. `BoundaryCrossing::check()` returns `GateVerdict::Allowed` unconditionally.

## Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        security crate                                   │
│                                                                         │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────┐    │
│  │ capability.rs │   │ labels.rs    │   │ message_types.rs         │    │
│  │              │   │              │   │                          │    │
│  │ CapRegistry  │   │ SecurityLabel│   │ VerifiedMessageType      │    │
│  │ Capability   │   │ IntegrityLvl │   │ MessageTypeRegistry      │    │
│  │ Permissions  │   │ MessageTypeId│   │ Invariant / InvariantChk │    │
│  │              │   │              │   │ SchemaHash               │    │
│  └──────┬───────┘   └──────┬───────┘   └──────────┬───────────────┘    │
│         │                  │                       │                    │
│         └──────────┬───────┴───────────────────────┘                    │
│                    ▼                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                      gate.rs                                    │    │
│  │                                                                 │    │
│  │  SecurityGate                                                   │    │
│  │    ├─ check(BoundaryCrossing) → GateVerdict                    │    │
│  │    │    Layer 1: Capability check (existing)                    │    │
│  │    │    Layer 2: Classification check (existing)                │    │
│  │    │    Layer 3: Integrity level check (Biba) ← NEW            │    │
│  │    │    Layer 4: Message type verification ← NEW                │    │
│  │    │    Layer 5: Enforcement mode → verdict                     │    │
│  │    ├─ mode resolution (global → boundary → type)                │    │
│  │    └─ audit emission on every decision                          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                    │                                                    │
│  ┌─────────────────▼───────────────────────────────────────────────┐    │
│  │                   policy.rs                                     │    │
│  │                                                                 │    │
│  │  SecurityPolicy                                                 │    │
│  │    ├─ compiled-in default (const)                               │    │
│  │    ├─ load_from_blob(signed_bytes) → Result                     │    │
│  │    ├─ remote_update(signed_bytes) → Result                      │    │
│  │    │    verify ML-DSA-65 signature                              │    │
│  │    │    validate internal consistency                            │    │
│  │    │    atomic swap with rollback support                        │    │
│  │    ├─ model_allowed(model_hash) → bool                          │    │
│  │    └─ type_registry() → &MessageTypeRegistry                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌────────────────┐   ┌──────────────────┐                             │
│  │ enforcement.rs │   │ boundary/ (mod)  │                             │
│  │                │   │                  │                             │
│  │ EnforcementMode│   │ DataFlow + type  │                             │
│  │ ModeResolution │   │ BoundaryDef +    │                             │
│  │ GateVerdict    │   │   enforcement    │                             │
│  └────────────────┘   └──────────────────┘                             │
└─────────────────────────────────────────────────────────────────────────┘
```

### Boot Sequence Integration

```
Phase 4: SecurityReady
  ├─ [existing] Initialize CapRegistry, seed CSPRNG
  ├─ [NEW] Load compiled-in SecurityPolicy (default types, default modes)
  ├─ [NEW] If policy blob available (env/memory-mapped): verify signature, swap in
  └─ [NEW] Initialize SecurityGate with policy reference

Phase 5-7: Network, IPC, Runtime Ready
  └─ [existing, unchanged]

Phase 8: ModelsLoaded
  ├─ [existing] Parse ONNX model
  ├─ [NEW] SecurityGate validates model:
  │    ├─ Model hash in policy's allowed set?
  │    ├─ Model's input shapes match registered VerifiedMessageType invariants?
  │    └─ Model's output shapes match registered VerifiedMessageType invariants?
  ├─ [existing] Build execution graph, optimize
  └─ [NEW] Grant model execution capability only if gate passes

Runtime: Remote policy update
  ├─ Signed blob arrives over mTLS network boundary
  ├─ SecurityGate validates blob:
  │    ├─ ML-DSA-65 signature verification
  │    ├─ Internal consistency (all referenced types have valid invariants)
  │    └─ No removal of Enforcing types (safety: can only add or promote)
  ├─ Atomic swap: new policy replaces old, old retained for rollback
  ├─ Re-validate loaded models against new policy
  │    └─ Models failing new policy are unloaded (logged, audited)
  └─ Audit log: policy swap event with old/new hashes
```

### Data Flow with Labels

```
External inference request (network boundary):
  ┌───────────────────────────────────────────────────────┐
  │ Raw bytes from TLS channel                            │
  │                                                       │
  │ SecurityGate.check(BoundaryCrossing {                │
  │   boundary: Network,                                  │
  │   direction: Inbound,                                 │
  │   label: SecurityLabel {                              │
  │     classification: Internal,    // inference I/O     │
  │     integrity: Low,              // untrusted source  │
  │     message_type: Some(0x0001),  // InferenceTensorIn │
  │   },                                                  │
  │   data_ref: &raw_bytes,                               │
  │ })                                                    │
  │                                                       │
  │ Layer 1: Task has NetworkSocket READ cap? ✓           │
  │ Layer 2: Internal ≤ dest max (KernelInternal)? ✓      │
  │ Layer 3: Low integrity → High dest? BLOCKED           │
  │          Low integrity → Medium dest? ✓ (inference)   │
  │ Layer 4: Type 0x0001 registered?                      │
  │          Invariants: rank ∈ [1,4]? dtype ∈ {f32,f16}? │
  │          total_elements ≤ 1M? ✓                       │
  │ Layer 5: Mode for 0x0001 = Enforcing → hard verdict   │
  │                                                       │
  │ → GateVerdict::Allowed (all layers passed)            │
  └───────────────────────────────────────────────────────┘

Internal actuator command (kernel → bus boundary):
  ┌───────────────────────────────────────────────────────┐
  │ Inference result flowing to actuator                  │
  │                                                       │
  │ Label: {                                              │
  │   classification: Internal,                           │
  │   integrity: Medium,     // inference output          │
  │   message_type: 0x0020,  // ActuatorCommand           │
  │ }                                                     │
  │                                                       │
  │ Layer 3: Medium → High (actuator bus)? BLOCKED        │
  │                                                       │
  │ → Must pass through integrity promotion gate:         │
  │   range-check, rate-limit, authorized-task signature  │
  │   If valid: promote integrity to High, re-check       │
  │                                                       │
  │ This is the safety chokepoint by design.              │
  └───────────────────────────────────────────────────────┘
```

### Key Type Definitions

```rust
// --- labels.rs ---

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum IntegrityLevel {
    Low = 0,       // Untrusted external source
    Medium = 1,    // Authenticated but not fully verified
    High = 2,      // Kernel-generated or promoted through verification
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MessageTypeId(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SecurityLabel {
    pub classification: ClassificationLevel,
    pub integrity: IntegrityLevel,
    pub message_type: Option<MessageTypeId>,
}

// --- message_types.rs ---

pub type SchemaHash = [u8; 32];  // SHA-3-256 of formal proof artifact

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerifiedMessageType {
    pub type_id: MessageTypeId,
    pub name: &'static str,
    pub boundary: TrustBoundary,
    pub direction: DataFlowDirection,
    pub schema_hash: SchemaHash,        // links to Lean4/TLA+ proof
    pub mode: EnforcementMode,
    pub invariants: &'static [Invariant],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Invariant {
    MaxRank(u8),                        // tensor rank upper bound
    MinRank(u8),                        // tensor rank lower bound
    AllowedDtype(TensorDataType),       // permitted element types (one per)
    MaxElements(u32),                   // total element count bound
    MaxPayloadBytes(u32),               // wire-level size bound
    ValueRange { min: i64, max: i64 },  // element value bounds
    EnumMembership(u8),                 // value must be valid enum variant
    NonZeroDimensions,                  // no zero-length dimensions
    MonotonicTimestamp,                 // requires gate state tracking
    RateLimit { max_per_sec: u32 },     // temporal invariant
}

// --- enforcement.rs ---

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EnforcementMode {
    Enforcing = 0,   // Hard reject on violation
    Permissive = 1,  // Log and allow on violation
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateVerdict {
    Allowed,
    Denied { layer: u8, reason: DenyReason },
    PermissivePass { layer: u8, reason: DenyReason },  // would deny, but mode is Permissive
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DenyReason {
    MissingCapability,
    ClassificationViolation,
    IntegrityViolation,
    UnknownMessageType,
    InvariantFailed(u8),  // index into type's invariant array
    ModelNotWhitelisted,
    PolicySignatureInvalid,
}

// --- gate.rs ---

pub struct SecurityGate {
    policy: &'static SecurityPolicy,  // or &mut for remote update
    default_mode: EnforcementMode,
    gate_checks: u64,
    gate_denials: u64,
    gate_permissive_passes: u64,
}

pub struct BoundaryCrossing<'a> {
    pub boundary: TrustBoundary,
    pub direction: DataFlowDirection,
    pub task_id: TaskId,
    pub label: SecurityLabel,
    pub data: &'a [u8],
}

// --- policy.rs ---

pub struct SecurityPolicy {
    pub type_registry: MessageTypeRegistry,
    pub model_whitelist: ModelWhitelist,
    pub boundary_modes: [EnforcementMode; 5],  // per TrustBoundary
    pub global_mode: EnforcementMode,
    pub policy_hash: SchemaHash,
    pub policy_signature: [u8; ML_DSA_65_SIG_SIZE],
    pub version: u32,
}
```

### Formal Verification Artifacts

| Artifact | Tool | Proves |
|----------|------|--------|
| `formal/lean4/IntegrityLattice.lean` | Lean 4 | Integrity levels form a valid lattice; Biba no-write-up property holds |
| `formal/lean4/MessageTypeProperties.lean` | Lean 4 | Type registry is well-formed; invariant checks are total and deterministic |
| `formal/lean4/TensorTypeInvariants.lean` | Lean 4 | Tensor message type invariants (rank bounds, dtype membership) are sound |
| `formal/lean4/LabelComposition.lean` | Lean 4 | SecurityLabel comparison composes correctly with ClassificationLevel ordering |
| `formal/tla/SecurityGate.tla` | TLA+ | Gate state machine: no data crosses boundary without check; enforcement mode transitions are monotonic (Permissive→Enforcing only); policy swap is atomic |
| `formal/tla/PolicyUpdate.tla` | TLA+ | Remote policy update: signature verification before swap; old policy retained; re-validation of loaded models |
