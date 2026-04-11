## Why

The existing security model enforces *who* can access *what* (capability-based access control) and *how* data flows across trust boundaries (authentication + integrity + confidentiality). But it does not enforce *what shape the data must be*. Any data that passes auth and integrity checks flows through — there is no gate verifying that the data conforms to a formally verified type before it enters the trusted compute path.

For safety-critical and high-assurance deployments (autonomous vehicles, medical devices, industrial control), this is insufficient. A malformed tensor, an unexpected bus frame structure, or an ill-typed IPC message can cause undefined behavior even if it came from an authenticated source. The system needs a **formal methods firewall**: a type-checking gate at every trust boundary that rejects data not conforming to verified message type schemas.

Additionally, the security policy must be **independent of the ONNX model**. Models are data — they must not declare their own security posture. The policy is loaded separately, either compiled-in or via a signed, remotely-updateable policy blob, and the model is validated against it. This separation ensures that security guarantees survive model updates and that policy can be audited independently.

Finally, different environments need different enforcement levels. Development needs fast iteration (permissive mode). Integration testing needs visibility into violations (permissive + logging). Production needs hard rejection of non-conforming data (enforcing mode). This must be configurable per boundary, per message type, or globally.

## What Changes

- Introduces a **verified message type registry** — each message type crossing a trust boundary has a unique ID, schema hash (linking to a Lean 4 / TLA+ formal proof), and runtime-checkable invariants
- Adds a **SecurityGate** at each trust boundary that layers type verification on top of existing auth/integrity/confidentiality checks
- Implements **enforcement modes** (Enforcing / Permissive) configurable globally, per-boundary, or per-message-type, with a graduation lifecycle from untyped → permissive → enforcing
- Introduces **MAC security labels** combining classification level (confidentiality), integrity level (Biba model), and message type ID — augmenting, not replacing, the existing capability system
- Separates **security policy loading** from ONNX model loading: policy loads at SecurityReady (boot phase 4), models load at ModelsLoaded (boot phase 8), gate validates models against policy
- Adds **remote policy update** support: signed policy blobs can be received over the network (mTLS + ML-DSA-65 signature verification) and hot-swapped without reboot
- Adds **audit integration**: all gate decisions (accept/reject/permissive-pass) are logged to the tamper-evident audit chain

## Capabilities

### New Capabilities
- `security-gate`: SecurityGate type-checking enforcement at trust boundaries with layered verification (auth → classification → integrity → message type)
- `verified-message-types`: Message type registry with schema hashes linking to formal proofs, runtime invariant checking (range, enum membership, length bounds, monotonicity)
- `enforcement-modes`: Permissive/Enforcing modes configurable at global, per-boundary, and per-message-type granularity with graduation lifecycle
- `security-labels`: MAC labels combining ClassificationLevel + IntegrityLevel + MessageTypeId, attached to all data crossing boundaries
- `policy-loading`: Independent security policy loading — compiled-in defaults, signed blob override at boot, remote update via mTLS with ML-DSA-65 verification
- `formal-proof-integration`: Schema hash lifecycle linking runtime type IDs to formal verification artifacts (Lean 4 proofs, TLA+ models)

### Modified Capabilities
- `security-model`: Augmented with MAC labels; capability checks remain unchanged but SecurityGate wraps them as an additional enforcement layer
- `container-interface`: Boot sequence gains policy-load substep within SecurityReady phase; ModelsLoaded phase gains model-vs-policy validation gate
- `ipc-messaging`: IPC messages carry SecurityLabel; pub/sub routing respects classification and integrity levels
- `networking`: Network boundary crossings validated against verified message types for inbound inference requests
- `onnx-runtime`: Model loading gated by policy (model hash whitelist, I/O shape conformance to registered message types)

## Impact

- `security/src/` — New modules: `gate.rs`, `labels.rs`, `message_types.rs`, `enforcement.rs`, `policy.rs`; modified: `boundary/`, `lib.rs`
- `security/src/boundary/` — Modified: `data_flow_auth.rs` gains message type field; `trust_boundaries.rs` gains enforcement mode per boundary
- `container/src/` — Modified: `boot.rs` (policy load substep), `config.rs` (policy config fields)
- `ipc/src/` — Modified: messages carry SecurityLabel
- `net/src/` — Modified: inbound data validated at network boundary gate
- `onnx-rt/src/` — Modified: `session.rs` model loading gated by policy
- `formal/lean4/` — New: `MessageTypeProperties.lean`, `TensorTypeInvariants.lean`, `IntegrityLattice.lean`
- `formal/tla/` — New: `SecurityGate.tla` (gate state machine verification)
- `openspec/changes/formal-type-gate-v1/specs/` — Delta specs for all new requirements
