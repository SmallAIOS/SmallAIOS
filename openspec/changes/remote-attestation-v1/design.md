# Design — remote-attestation-v1

## Goal

A verifier-driven, hardware-anchored, hybrid-signed attestation primitive observable as:

1. Verifier on a developer workstation runs `attest-verifier verify --release-record release-0.3.0.json --endpoint https://target:8080/v1/attest`.
2. Target SmallAIOS instance produces a hybrid quote in < 100 ms.
3. Verifier validates the inner hardware signature (TPM AK or PSA-IA HUK-derived key), the SmallAIOS counter-signature, the nonce freshness, and the measurement bundle against the expected release record.
4. Verifier emits `attest-record-2026-05-10T14:23:00Z.cbor` containing the request, response, verification trace, and verifier signature.

## Alternatives considered

### 1. Reuse the EAT (Entity Attestation Token) standard verbatim, no hybrid signing

Rejected. EAT is COSE-based, COSE supports only the classical signature algorithm suite (RSA, ECDSA, EdDSA). SmallAIOS's PQC-default stance requires ML-DSA support, which COSE has draft proposals for but no standard. We use EAT as the *inner* report format on the AArch64 path (PSA-IA is already EAT-shaped); the *outer* envelope is SmallAIOS-defined CBOR carrying the EAT plus the PQC counter-signature. Pure EAT verifiers can still consume the inner report, ignoring the outer counter-sig — degrades gracefully.

### 2. Use TLS client-certificate-based attestation

Considered. Could embed a TPM-backed cert in the TLS handshake, let the verifier check it as part of TLS. Rejected because (a) it limits attestation to TLS-fronted endpoints, no IPC story; (b) the verifier has no nonce control — the TLS-time cert is static, not request-bound; (c) it doesn't carry the measurement log, just identity; (d) PQC-hybrid TLS already exists in SmallAIOS's `net/quic/` — adding attestation as TLS extensions would conflate concerns. Keep attestation as an explicit application-layer protocol.

### 3. Skip the hybrid signing, use TPM / PSA classical only

Rejected. The classical-only path is supported (`PqcMode: "classical-only"` in the request), but the default and the recommendation is hybrid. Reasoning is the same as `boot-root-of-trust-v1`'s hybrid quote: future PQC-only deployments need the counter-sig pre-baked, and the cost of adding it (a few hundred microseconds) is negligible relative to the TPM/PSA round-trip.

### 4. Implement a vendor-specific attestation backend (AWS Nitro NSM, Azure HSM)

Rejected for this change. The cloud vendor attestation services have their own private signatures, key-distribution flows, and (frankly) lock-in concerns. SmallAIOS sticks to the open standards (TPM 2.0, PSA-IA) and lets cloud customers wrap our output in their vendor flow if they want. We don't ship vendor adapters.

### 5. RISC-V backend via Keystone enclave

Considered. Keystone is research-stage as of 2026-05; no production RISC-V hardware ships with Keystone enabled, and the protocol is still in flux. Deferred to a future change once Keystone (or its successor) stabilizes. RISC-V SmallAIOS instances can opt into the SmallAIOS counter-signature half only, marking the hardware half as `null` in the response — verifiers configured for `PqcMode: "pqc-only"` accept it; verifiers requiring classical hardware reject it. The matrix doc update will be explicit about this gap.

## Wire format details

### Report content (`HybridQuote` for TPM2, `EAT-CWT-wrapped` for PSA-IA)

Both backends produce a CBOR structure with the following normalized fields under the outer envelope:

```
NormalizedReport := CBOR-Map {
  1: "SmallAIOS-Attest-v1",       // version magic
  2: Nonce ([u8; 32]),
  3: Timestamp (uint, monotonic SmallAIOS time),
  4: MeasurementBundle ({
       kernel_hash: SHA-3-256,
       boot_config_hash: SHA-3-256,
       model_hashes: [SHA-3-256, ...],
       config_hash: SHA-3-256,
       counter_pub_digest: SHA-3-256,
     }),
  5: HardwareQuote (TpmQuote OR PsaCwt),  // per backend
  6: HybridSignature (ML-DSA-65 || Ed25519 over fields 1-5),
}
```

The verifier checks:

- field 1 magic;
- field 2 matches its sent nonce;
- field 5 verifies against the appropriate trust anchor (TPM EK cert chain → AK pub for TPM; PSA-IA spec key → HUK-derived pub for PSA);
- field 6 verifies against the SmallAIOS counter-signing public key, which is itself measured into the chain (PCR 14 on x86-64, included in the PSA-IA `psa_arg_t` payload on AArch64);
- field 4 matches the expected release-record bundle for the policy ID.

If all five pass, the verifier emits a PASS audit record. Any failure → FAIL with a documented error code.

### Transport encoding

`AttestRequest` and `AttestResponse` are CBOR over the wire. HTTP/QUIC carry them as `application/cbor`; IPC carries them as opaque byte buffers through the existing capability-call path.

The `AttestRequest`'s `Extensions` field (key 5) is a map of CBOR. Documented extension keys:

- `session_binding_nonce` (`[u8; 32]`): for QUIC channel-binding, lets the verifier prove the attestation is bound to the specific transport session.
- `policy_max_age_seconds` (uint): if set, the SmallAIOS server rejects the request if the measurement log was sealed (locked from further extends) more than N seconds ago. Defaults to "no age check" if absent.

Unknown extension keys are ignored by the server (forward-compatible).

## Backend selection

```rust
// security/src/attest/mod.rs

pub enum AttestBackend {
    #[cfg(feature = "tpm-attest")]    Tpm2(Tpm2Backend),
    #[cfg(feature = "op-tee")]        PsaIa(PsaIaBackend),
    SoftwareOnly(SwBackend),  // counter-sig only; verifier policy decides if acceptable
}

impl AttestBackend {
    pub fn select() -> Self {
        #[cfg(feature = "tpm-attest")] {
            if let Ok(t) = Tpm2Backend::initialize() { return Self::Tpm2(t); }
        }
        #[cfg(feature = "op-tee")] {
            if let Ok(p) = PsaIaBackend::initialize() { return Self::PsaIa(p); }
        }
        Self::SoftwareOnly(SwBackend::new())
    }
}
```

Backend selection is a one-time runtime probe at attest-server startup. The selected backend is logged into the boot measurement log so the choice is itself attested.

The `SoftwareOnly` backend produces a report with `HardwareQuote: null`; verifiers configured for `pqc-only` accept it, others reject. This is the RISC-V path and the no-TPM x86-64 path. It is documented as "weakest" tier — useful for development but not for production.

## Verifier crate

Lives at `tools/attest-verifier/`, **outside** the no-std workspace. Uses `std`, `tokio` (for HTTP/QUIC client work), `serde_cbor`, our own clean-room PQC libs from `security/` (re-exported for `std`).

CLI subcommands:

- `verify` — issue a request, validate the response, emit an audit record.
- `inspect` — pretty-print a saved audit record.
- `verify-batch` — repeat `verify` against a fleet manifest, emit a per-instance audit record set.

Trust anchor management: the verifier expects a directory layout under `~/.config/smallaios-attest/`:

```
trust-anchors/
  tpm-ek-roots/        # PEM-format EK CA certificates from TPM manufacturers
  psa-ia-pub-keys/     # PSA-IA HUK-derived public keys per Tegra Orin SKU / generation
  smallaios-counter-pub-keys/  # SmallAIOS Engineering's PQC counter-signing pub keys per release
release-records/
  release-0.3.0.json
  release-0.3.1.json
  ...
```

Release-record JSON shape:

```json
{
  "version": "0.3.0",
  "kernel_hash_sha3_256": "...",
  "boot_config_hash_sha3_256": "...",
  "model_hashes_sha3_256": ["..."],
  "config_hash_sha3_256": "...",
  "counter_pub_digest_sha3_256": "...",
  "released_at": "2026-04-15T12:00:00Z",
  "vendor_signature_ml_dsa_65": "..."
}
```

The release record is itself signed by SmallAIOS Engineering's release key (covered by `boot-root-of-trust-v1` Phase 4). The verifier checks the record's own signature before using it as ground truth.

## Latency / capacity

| Backend | Latency target | Throughput (single-server) |
|---------|----------------|----------------------------|
| TPM 2.0 (x86-64) | < 50 ms p99 | ~20 attests/sec (TPM bottleneck) |
| PSA-IA (AArch64) | < 100 ms p99 | ~10 attests/sec (TA bottleneck) |
| SoftwareOnly | < 5 ms p99 | ~200 attests/sec (CPU only) |

The attestation server is not on the inference fast-path. Verifiers poll on the order of once-per-hour-per-instance, not per-request. The throughput targets are deliberately modest — there is no need for thousands of attests/sec.

## Anti-replay

Two layers:

1. **Nonce-driven freshness.** Every request supplies a 32-byte verifier-generated nonce. The response embeds the nonce in the signed report. A replayed response carries a stale nonce, which the verifier detects.
2. **Server-side anti-replay window** (defense in depth). The attest server keeps a small bloom filter of recently-seen nonces (1-minute window). Repeats are rejected with an `AttestError::ReplayedNonce`. Protects against an attacker replaying a verifier's old request against the same server within the window.

The bloom filter is conservatively sized for ~10k req/min — orders of magnitude above the expected throughput. False positives are acceptable (verifier retries with a fresh nonce).

## Build / CI surface

- New crate-internal module: `security/src/attest/` (Layer 0).
- New backend modules: `security/src/attest/tpm2_backend.rs` (gated `tpm-attest`), `security/src/attest/psa_ia_backend.rs` (gated `op-tee`).
- New transport adapter: `container/src/attest_handler.rs` (HTTP/QUIC `POST /v1/attest`).
- New `std` crate: `tools/attest-verifier/`.
- New CI job: `attest-verifier-smoke` — boots SmallAIOS with `tpm-attest` (under QEMU+swtpm), runs the verifier against it, asserts PASS. Advisory initially.
- New docs: `docs/remote-attestation-protocol.md`, `docs/attest-verifier-usage.md`, `docs/release-attestation-records.md`.

## What this change explicitly does NOT do

- Does not change the inference data plane. Attestation is its own endpoint, separate from `/v1/inference`.
- Does not modify the existing audit log. Attestation records are a separate artifact type emitted by the verifier, not the server.
- Does not require continuous attestation. One-shot request/response only.
- Does not require any particular key-distribution infrastructure. Trust anchors are flat-file in this change; PKI integration is a follow-up.
- Does not implement Veraison / IETF-RATS conformance testing. Wire format follows EAT conventions but isn't certified against RATS reference suites; that's a future activity.
