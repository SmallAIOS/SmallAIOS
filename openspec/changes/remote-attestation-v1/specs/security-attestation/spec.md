## ADDED Requirements

### Requirement: Remote attestation protocol surface

The `smallaios-security` crate SHALL provide a remote attestation protocol producing hybrid-signed, nonce-bound, measurement-bundle reports verifiable by an off-host verifier. The protocol SHALL be transport-agnostic and SHALL be exposed over HTTP, QUIC/HTTP3, and capability-gated IPC.

#### Scenario: AttestRequest / AttestResponse CBOR shape

- **GIVEN** a verifier posting `application/cbor` to `POST /v1/attest`
- **WHEN** the request body is a CBOR map matching the documented `AttestRequest` shape (Nonce, PolicyId, ReportFormat, PqcMode, optional Extensions)
- **THEN** the server SHALL respond with a CBOR map matching the documented `AttestResponse` shape (Backend, Report, SignatureMode, ServerTime, SmallAiosVersion)
- **AND** the response Content-Type SHALL be `application/cbor`
- **AND** both shapes SHALL be documented in `docs/remote-attestation-protocol.md` with their full CBOR schema

#### Scenario: Three transports carry the same payload

- **GIVEN** a verifier with HTTP, QUIC/HTTP3, and IPC clients
- **WHEN** the same `AttestRequest` bytes are sent over each transport
- **THEN** the resulting `AttestResponse` bytes SHALL be byte-identical (the wire payload is transport-independent)
- **AND** the IPC variant SHALL require the caller to hold an `Attest` capability granted via `security::capability`

#### Scenario: Anti-replay bloom filter rejects nonce reuse within the window

- **GIVEN** a server that has handled an attestation request with nonce N within the last 60 seconds
- **WHEN** a second request arrives with the same nonce N
- **THEN** the server SHALL respond with `AttestResponse { Backend: …, Report: null, SignatureMode: "error", error: "ReplayedNonce" }` (or an equivalent CBOR error envelope)
- **AND** the server SHALL log the rejection with the source transport (HTTP/QUIC/IPC) and timestamp for audit

### Requirement: x86-64 TPM 2.0 backend

When built with `--features tpm-attest` on x86-64, the attestation server SHALL produce a hardware-rooted report using the `HybridQuote` format defined in `boot-root-of-trust-v1`.

#### Scenario: Round-trip with hybrid mode

- **GIVEN** an x86-64 SmallAIOS instance with `tpm-attest` enabled, a TPM 2.0 device present, and PCRs 11-14 extended per `boot-root-of-trust-v1`
- **WHEN** a verifier sends `AttestRequest { Nonce: N, PolicyId: "release-0.3.0", ReportFormat: "HybridQuote", PqcMode: "hybrid" }`
- **THEN** the server SHALL produce a `HybridQuote` containing the TPM2_Quote over PCRs 11-14 (with N as the qualifying nonce), the BootMeasurementLog entries, the ML-DSA-65 + Ed25519 counter-signature, the counter-public-key digest, the nonce, and the server timestamp
- **AND** the response SHALL be wrapped in `AttestResponse { Backend: "tpm2", Report: <HybridQuote>, SignatureMode: "hybrid", ServerTime: <t>, SmallAiosVersion: <ver> }`
- **AND** end-to-end latency from request receipt to response send SHALL be < 50 ms p99 under nominal load

#### Scenario: PqcMode mode selection honored

- **GIVEN** the same instance
- **WHEN** the verifier requests `PqcMode: "classical-only"`
- **THEN** the response SHALL include the TPM-side signature but SHALL omit the ML-DSA-65 + Ed25519 counter-signature (or include it as `null`)
- **AND** `SignatureMode` SHALL echo `"classical-only"` so the verifier knows the omission was honored

- **GIVEN** the same instance
- **WHEN** the verifier requests `PqcMode: "pqc-only"`
- **THEN** the response SHALL include the counter-signature but SHALL omit the TPM-side raw signature bytes (still including the PCR digest the TPM signed, since that's the measurement, not the signature)
- **AND** `SignatureMode` SHALL echo `"pqc-only"`

#### Scenario: Verifier validates all four layers

- **GIVEN** a captured `AttestResponse` from the TPM2 backend
- **WHEN** `attest-verifier verify --release-record release-0.3.0.json --pqc-mode hybrid` validates the response
- **THEN** the verifier SHALL check (a) the TPM AK signature against the TPM EK CA root, (b) the SmallAIOS counter-signature against the configured counter-public-key, (c) the nonce matches the originally-sent nonce, (d) the measurement bundle matches the release record's expected hashes
- **AND** all four checks SHALL succeed for a valid response
- **AND** any one failing check SHALL cause the verifier to report FAIL with a documented error code

### Requirement: AArch64 PSA-IA backend

When built with `--features op-tee` on AArch64 and the PSA-IA Trusted Application is loadable, the attestation server SHALL produce a hardware-rooted report using the PSA Initial Attestation API (EAT-CWT format) wrapped with a SmallAIOS counter-signature.

#### Scenario: PSA-IA TA invocation via OP-TEE bridge

- **GIVEN** an AArch64 SmallAIOS instance with the OP-TEE bridge available and the PSA-IA TA (UUID `f0b13b9b-8b8a-4f57-9b95-79c83e3b09cd`) loaded
- **WHEN** the attestation server initializes
- **THEN** the server SHALL open a long-lived OP-TEE session to the PSA-IA TA
- **AND** the boot measurement log SHALL record `AttestBackend::PsaIa { ta_uuid: …, op_tee_version: … }`

#### Scenario: EAT-CWT round-trip with hybrid counter-signature

- **GIVEN** an active PSA-IA session and a verifier-supplied nonce N
- **WHEN** the server invokes `PSA_INITIAL_ATTEST_GET_TOKEN` with N + the SmallAIOS measurement bundle
- **THEN** the server SHALL receive an EAT-CWT signed by an HUK-derived ECDSA-P256 key in Secure World
- **AND** the server SHALL ML-DSA-65 + Ed25519 counter-sign (CWT || measurement_bundle || nonce || timestamp)
- **AND** the response SHALL be wrapped in `AttestResponse { Backend: "psa-ia", Report: <EAT-CWT-with-counter-sig-envelope>, SignatureMode: "hybrid", … }`
- **AND** end-to-end latency from request receipt to response send SHALL be < 100 ms p99 under nominal load

#### Scenario: TA failure causes session re-open, not server crash

- **GIVEN** a long-lived PSA-IA session that returns an error mid-request (e.g. OP-TEE-side reset)
- **WHEN** the server detects the failure
- **THEN** the server SHALL close and re-open the session
- **AND** the affected request SHALL fail with `AttestError::BackendUnavailable` (the verifier retries with a fresh nonce)
- **AND** subsequent requests SHALL succeed against the new session

### Requirement: SoftwareOnly fallback backend

When no hardware backend is available (no TPM on x86-64, no PSA-IA TA on AArch64, RISC-V build), the attestation server SHALL produce a report with the ML-DSA-65 + Ed25519 counter-signature only, with the hardware-quote field set to null.

#### Scenario: SoftwareOnly response is valid but weakest tier

- **GIVEN** a SmallAIOS instance with no hardware attestation backend
- **WHEN** a verifier sends an `AttestRequest`
- **THEN** the response SHALL be `AttestResponse { Backend: "software-only", Report: { HardwareQuote: null, … }, SignatureMode: "pqc-only-fallback", … }`
- **AND** the verifier SHALL accept the response only if it is configured with `--allow-software-only` or `PqcMode: "pqc-only"`
- **AND** the verifier SHALL emit an audit record explicitly marking the verification as "software-only — weakest tier" so downstream auditors see the policy decision

### Requirement: Reference verifier crate

The repository SHALL provide a `tools/attest-verifier/` crate (std + tokio) implementing the verifier side of the protocol, with `verify`, `inspect`, and `verify-batch` subcommands.

#### Scenario: Verify subcommand emits an audit record per verification

- **GIVEN** a developer running `attest-verifier verify --release-record release-0.3.0.json --endpoint https://target:8080/v1/attest --pqc-mode hybrid`
- **WHEN** the verification succeeds
- **THEN** the tool SHALL print human-readable PASS output
- **AND** the tool SHALL write `attest-record-<rfc3339-timestamp>.cbor` to the current working directory (or a `--output-dir`-specified location)
- **AND** the audit record SHALL contain the request, the response, the verification trace (which checks passed/failed), the verifier's local time, and the verifier's own signature over the record

#### Scenario: Verify-batch handles a fleet

- **GIVEN** a fleet manifest listing N endpoints with their expected policy IDs
- **WHEN** `attest-verifier verify-batch --manifest fleet.toml` runs
- **THEN** the tool SHALL attempt to verify each endpoint in parallel (bounded by a `--concurrency` flag, default 8)
- **AND** the tool SHALL emit one audit record per endpoint into `--output-dir/`
- **AND** the tool SHALL exit with code 0 if all endpoints verify, or 1 if any fails (with a summary line listing failures)

#### Scenario: Trust anchor layout is documented

- **THEN** the verifier SHALL read trust anchors from `~/.config/smallaios-attest/trust-anchors/`:
  - `tpm-ek-roots/` for TPM EK CA certificates
  - `psa-ia-pub-keys/` for PSA-IA HUK-derived public keys
  - `smallaios-counter-pub-keys/` for SmallAIOS Engineering's PQC counter-signing pub keys
- **AND** the layout SHALL be documented in `docs/attest-verifier-usage.md`
- **AND** the `--trust-anchor-dir` flag SHALL allow overriding the default location

### Requirement: Release records as verifier ground truth

A SmallAIOS release SHALL publish a signed `release-X.Y.Z.json` record listing the canonical kernel hash, boot-config hash, model hashes, and config hash for that release. The verifier consults the record to determine the expected measurement bundle for a `PolicyId`.

#### Scenario: Release record JSON shape

- **THEN** a release record SHALL contain `version`, `kernel_hash_sha3_256`, `boot_config_hash_sha3_256`, `model_hashes_sha3_256` (array), `config_hash_sha3_256`, `counter_pub_digest_sha3_256`, `released_at` (RFC 3339), `vendor_signature_ml_dsa_65` (the SmallAIOS Engineering release-signing key's signature over the preceding fields)
- **AND** the shape SHALL be documented in `docs/release-attestation-records.md`
- **AND** the release pipeline (`just release` plus `boot-root-of-trust-v1` Phase 4 signing) SHALL produce one record per tagged release

#### Scenario: Verifier validates the record's own signature

- **GIVEN** a release record loaded by `attest-verifier verify`
- **WHEN** the verifier reads the file
- **THEN** the verifier SHALL first validate `vendor_signature_ml_dsa_65` against the SmallAIOS Engineering release-signing pub key
- **AND** if the record's own signature fails, the verifier SHALL print "Record signature invalid; refusing to trust as ground truth" and exit with code 3 (record-untrusted)
- **AND** only after the record's signature passes SHALL the verifier compare the measurement bundle from the attest response against the record's hashes
