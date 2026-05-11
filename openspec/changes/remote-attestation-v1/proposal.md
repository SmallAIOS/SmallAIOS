# remote-attestation-v1

## Summary

A SmallAIOS deployment today cannot answer the question "**is the kernel and model bundle that's serving my inference traffic the one I signed?**" to a remote verifier. The local `BootMeasurementLog` is a debug artifact; there is no protocol for a remote client to request a freshness-bound, signature-bound proof of what's running. This change adds a remote attestation surface: a verifier (e.g. an inference-API client, a fleet operator, an auditor) sends a nonce and an attestation policy; the SmallAIOS instance responds with a signed quote covering the kernel hash, the loaded ONNX models, the SmallAIOS configuration, and the verifier-supplied nonce.

Two attestation backends are specified, selected at build time:

- **x86-64 TPM 2.0 backend**, depending on `boot-root-of-trust-v1` Phase 1. The hardware quote is produced via `TPM2_Quote` over PCRs 11-14; the wire format is the `HybridQuote` CBOR defined in that change (TPM-AK-signed bundle + ML-DSA-65 + Ed25519 SmallAIOS counter-signature).
- **AArch64 PSA Initial Attestation backend**, depending on `op-tee-bridge-v1`. The Arm PSA Initial Attestation API (PSA-IA, PSA-AT 1.0) is implemented as an OP-TEE Trusted Application called via the bridge. The PSA-IA report is a CBOR Web Token (CWT) signed in Secure World with a Hardware-Unique Key derived ECDSA-P256 key; SmallAIOS counter-signs the CWT with ML-DSA-65 + Ed25519 in Normal World to match the PQC-default stance.

Both backends share a **single Normal-World protocol surface** (`AttestRequest` / `AttestResponse` CBOR over the kernel's existing IPC / HTTP / QUIC transports — caller's choice) and a **single verifier library shape** so an off-host verifier accepts either backend's output through one API. The capability `security-attestation` (new) defines the protocol, the report format, the policy semantics, and the verifier-side interfaces.

A reference verifier crate (`smallaios-attest-verifier`, in `tools/attest-verifier/`) ships with the change. It is build-host-side (`std`-using, runs on any developer workstation), produces a human-readable verification result, and integrates with the existing `cargo-vet`-style audit trail story by emitting a signed `attest-record.cbor` per verification (useful for fleet-operator audit logs).

## Why

- **Without remote attestation, "verified boot" is unverifiable in production.** A SmallAIOS instance can know it booted with the expected kernel hash (via `boot-root-of-trust-v1`'s measurement log), but only the instance itself knows. The inference client three hops away on the network has no way to confirm. Remote attestation closes that loop — verifier sends nonce, instance returns signed quote, verifier checks quote against the expected kernel+model+config hash bundle from the release record. This is the standard cloud / fleet pattern (Google's Project Borg, Microsoft's Azure Attestation Service, AWS Nitro Enclaves) that SmallAIOS lacks.
- **AI inference is a "confidential by intent" workload.** Customers running inference against a model they paid to license, or feeding sensitive inputs to a SmallAIOS-served model, need attestation as a precondition to trusting either the model output or the input handling. The `confidential-compute-v1` change (separate) addresses *memory* confidentiality; this change addresses *identity* confidentiality — even before any hardware confidential-compute is enabled, a customer can verify the inference endpoint is running a known SmallAIOS + known model bundle.
- **PQC-default is structurally enforced via hybrid signing.** Both backends produce a TPM-side or PSA-side hardware classical signature (RSA-2048 or ECDSA-P256 — whatever the hardware supports) AND a SmallAIOS-counter-signed ML-DSA-65 + Ed25519 hybrid. A verifier can require the PQC half, the classical half, or both, depending on policy. Today's hardware mandates the classical half; tomorrow's deployment can mandate PQC-only without re-spec.
- **DO-178C DAL A claim: "Field instances are running certified artifacts."** Remote attestation is the cryptographic primitive an auditor uses to *check* that claim across a fleet. The verifier emits a verifiable audit record per-instance per-time-window; the records are themselves signed and storable for compliance retention.
- **The two backends share 90% of the wire format.** The differentiator is the inner hardware signature (TPM AK vs. PSA HUK-derived key). Both go inside the same outer CBOR envelope with the same SmallAIOS counter-signature, so verifier code is mostly shared. Designing this from the start means the AArch64 backend lands without re-doing the protocol design — only the inner signature changes.
- **Aligned with industry standards.** The wire format follows IETF RATS (Remote ATtestation procedureS) WG conventions: EAT (Entity Attestation Token, draft-ietf-rats-eat) for the report shape, COSE_Sign1 / COSE_Sign for signature encoding, CBOR throughout. SmallAIOS gets ecosystem-compatible output by default. A standard EAT verifier (e.g. the Veraison project's tooling) can consume the classical half of our reports unmodified.

## What ships

### Protocol surface

```
AttestRequest := CBOR-Map {
  1: Nonce ([u8; 32]),                   // verifier-supplied freshness nonce
  2: PolicyId (string),                   // identifies the expected hash bundle (e.g. "release-0.3.0")
  3: ReportFormat ("HybridQuote" | "EAT"), // selects wire format
  4: PqcMode ("hybrid" | "classical-only" | "pqc-only"), // signature requirement
  5: Extensions (optional CBOR map),      // policy-specific extensions (e.g. session-binding nonce for QUIC)
}

AttestResponse := CBOR-Map {
  1: Backend ("tpm2" | "psa-ia"),
  2: Report (HybridQuote per boot-root-of-trust-v1 OR EAT-CWT per PSA-IA),
  3: SignatureMode (echoes PqcMode if honored, or error code if request rejected),
  4: ServerTime (uint),
  5: SmallAiosVersion (string),
}
```

### x86-64 backend (depends on `boot-root-of-trust-v1` Phase 1)

`security/src/attest/tpm2_backend.rs` (new). On request: gather the current `BootMeasurementLog` snapshot, call `produce_hybrid_quote(nonce)` from the dependency change, wrap in `AttestResponse`, return. Latency target: < 50 ms (TPM2_Quote takes ~10-30 ms on modern fTPMs; SmallAIOS counter-signing adds <5 ms for ML-DSA-65).

### AArch64 backend (depends on `op-tee-bridge-v1`)

`security/src/attest/psa_ia_backend.rs` (new). The PSA-IA Trusted Application UUID is `f0b13b9b-8b8a-4f57-9b95-79c83e3b09cd` (defined by Arm's PSA test suite reference TA, ships with upstream OP-TEE OS). On request:

1. Compute the SmallAIOS-side measurement bundle (kernel hash + model hashes + config hash, all SHA-3-256) — same shape as the TPM-backend bundle, just without PCR encoding.
2. Open a session to the PSA-IA TA via the OP-TEE bridge.
3. Invoke `PSA_INITIAL_ATTEST_GET_TOKEN` with the verifier-supplied nonce and the SmallAIOS-side measurement bundle as the `psa_arg_t` payload.
4. Receive the PSA EAT-CWT (a CBOR Web Token signed in Secure World by an HUK-derived ECDSA-P256 key).
5. ML-DSA-65 + Ed25519 counter-sign the CWT bundle.
6. Wrap and return as `AttestResponse` with `Backend = "psa-ia"`.

Latency target: < 100 ms (OP-TEE round-trip + PSA-IA TA work + SmallAIOS counter-sign). The TA-side work is the dominant term.

### Wire transports

The attestation surface is transport-agnostic. SmallAIOS exposes it over:

- **HTTP**: `POST /v1/attest` with `Content-Type: application/cbor` (existing HTTP server in `container/`).
- **QUIC / HTTP3**: same endpoint, leveraging the existing QUIC stack in `net/quic/`.
- **IPC**: an `Attest` capability call for in-host verifiers (e.g. a sidecar workload audit agent).

Each transport carries the same `AttestRequest` / `AttestResponse` CBOR. The transport layer enforces TLS / QUIC PQC-hybrid where it's already enforced for other endpoints — no new transport security work.

### Reference verifier

`tools/attest-verifier/` (new crate, `std`-using, build-host-only — not part of the kernel workspace). Capabilities:

- `attest-verifier verify --release-record release-0.3.0.json --endpoint https://host:8080/v1/attest --pqc-mode hybrid` — issues an attestation request, verifies the response, prints human-readable pass/fail, emits `attest-record-<timestamp>.cbor` as the audit trail entry.
- `attest-verifier inspect <attest-record.cbor>` — pretty-prints a previously-saved record for offline audit.
- Verifier-side trust anchor management: keeps a directory of TPM EK CA roots (Intel/AMD/Infineon/STM…), PSA HUK-derivation public keys, and SmallAIOS counter-signing public keys. Verifies each layer.

### Documentation

- `docs/remote-attestation-protocol.md` — wire format spec, policy semantics, verifier integration guide.
- `docs/attest-verifier-usage.md` — operator-side how-to.
- `docs/release-attestation-records.md` — release-engineering side: how to publish a `release-X.Y.Z.json` hash bundle, how it ties to `boot-root-of-trust-v1` Phase 4 signing.

## Out of scope

- **RISC-V backend.** Defer — no widely-deployed RISC-V hardware attestation primitive exists (Keystone / Penglai are research-stage, not production). RISC-V instances can opt into the *software-counter-signature half* of the protocol only; classical-half is `null`. Documented as a known limitation in the matrix doc update.
- **Continuous / streaming attestation.** Each `AttestRequest` is one-shot, request/response. Streaming "attestation events" (per-cap-flip, per-model-load) is a future enhancement; the existing audit log handles this surface for now.
- **Attestation-bound key release.** Some confidential-compute models bind decryption keys to attestation results (verifier checks attest, then unwraps a workload-supplied key). That's `confidential-compute-v1` territory. This change ships the verifier-side primitive; the key-release flow is built on top.
- **Decentralized / on-chain attestation records.** Audit records are local files; bridging to Sigstore / on-chain transparency logs is out of scope.
- **Vendor-specific attestation services (e.g. AWS Nitro NSM, Azure HSM).** SmallAIOS supports the *standard* TPM 2.0 and PSA-IA interfaces; cloud-vendor wrappers can adapt by speaking those standards through their own glue layers.

## Sequencing

This change has two phases internally, mapped to the dependencies:

- **Sub-phase A: protocol design + x86-64 TPM backend + reference verifier** — lands once `boot-root-of-trust-v1` Phase 1 is merged. ~2-3 weeks.
- **Sub-phase B: AArch64 PSA-IA backend** — lands once `op-tee-bridge-v1` is merged. ~2 weeks.

A and B can land as separate PRs against this single OpenSpec change, or as a single PR if the dependencies are sequenced tightly. The protocol surface is defined in A; B is a backend addition with no protocol design surface.

If schedule pressure forces a split, Sub-phase A alone is a complete, useful deliverable for x86-64 deployments — it covers the largest production deployment surface (cloud / datacenter x86-64). Sub-phase B is the followup that extends coverage to ARM datacenter and embedded.

## Effort estimate

| Sub-phase | Scope | Estimate |
|-----------|-------|----------|
| A | Protocol + CBOR + TPM2 backend wiring + transport bindings + verifier | ~2-3 weeks |
| B | PSA-IA TA invocation + CWT counter-signing + backend wiring | ~2 weeks |
| **Total** | | **~4-5 weeks** (after dependencies merge) |

## Dependencies

- `boot-root-of-trust-v1` Phase 1 (TPM 2.0 driver + `HybridQuote` format) — hard dependency for sub-phase A.
- `op-tee-bridge-v1` — hard dependency for sub-phase B.
- No dependency between A and B; they can land in either order once their respective dependencies are in.

## DO-178C alignment

Remote attestation gives DAL A its **field verification primitive**: the auditor's standing question "are the instances in the field running the certified binary?" gets a cryptographic answer per-instance per-poll, with a signed audit record for retention. The verifier-emitted `attest-record-<timestamp>.cbor` is the artifact that fits into the certification evidence chain.

Specifically, the certification claim **"the kernel binary serving inference for tenant T at time T1 is artifact X.Y.Z signed by SmallAIOS Engineering"** is provable by combining:

1. The hybrid quote covering PCR 11 (kernel hash) signed by the TPM AK + SmallAIOS counter-sig.
2. The release record mapping kernel hash → release version X.Y.Z + Engineering signature.
3. The freshness nonce in (1) proving the quote was produced at time T1.

All three are CBOR documents, all three are signed, all three are storable for the certification retention period (10+ years for DAL A). The verifier crate emits them in a single bundle per verification.
