# Tasks — remote-attestation-v1

## 0. Prerequisites

- [ ] 0.1 Confirm `boot-root-of-trust-v1` Phase 1 has merged and the `HybridQuote` CBOR format is stable. Pin to the develop SHA at the start of Sub-phase A; rebase if the format moves.
- [ ] 0.2 Confirm `op-tee-bridge-v1` has merged and the GP TEE Client API surface is stable. Pin to develop SHA at start of Sub-phase B.
- [ ] 0.3 Confirm the upstream OP-TEE PSA-IA TA UUID (`f0b13b9b-8b8a-4f57-9b95-79c83e3b09cd`) is loadable on the Tegra Orin reference firmware. If not, document the alternative (custom-built PSA-IA TA per the Trusted Firmware-M reference) in design.md.
- [ ] 0.4 Pin the EAT (Entity Attestation Token) draft version we target — currently `draft-ietf-rats-eat-25`. Re-pin if a newer draft alters wire format incompatibly.

## 1. Sub-phase A — Protocol + x86-64 TPM backend

### 1a. Protocol module

- [ ] 1.1 Create `security/src/attest/mod.rs` exposing `AttestBackend`, `AttestRequest`, `AttestResponse`, `AttestError`. Public types are `no_std`-friendly CBOR-derive structs.
- [ ] 1.2 Define `NormalizedReport` CBOR structure per design.md.
- [ ] 1.3 Implement `AttestBackend::select()` probe: try TPM2 first, then PSA-IA, then SoftwareOnly fallback.
- [ ] 1.4 Log the selected backend into `BootMeasurementLog` so the choice is itself attested.

### 1b. TPM 2.0 backend wiring

- [ ] 1.5 Create `security/src/attest/tpm2_backend.rs` gated `cfg(feature = "tpm-attest")`. Initialization: probe `boot-root-of-trust-v1`'s TPM driver, fail with `AttestError::TpmAbsent` if not available.
- [ ] 1.6 Implement `Tpm2Backend::handle_request(req: &AttestRequest) -> Result<AttestResponse>`:
  - validate nonce length;
  - call `produce_hybrid_quote(req.nonce)` from `boot-root-of-trust-v1`;
  - wrap in `NormalizedReport` shape, emit as `AttestResponse`.
- [ ] 1.7 Honor `PqcMode`: if `classical-only`, omit the ML-DSA-65 + Ed25519 counter-sig from the response (still compute it but mark omitted); if `pqc-only`, omit the TPM quote (but include its measurement). If `hybrid`, include both.
- [ ] 1.8 Enforce server-side anti-replay bloom filter (1-minute window, sized for 10k req/min).

### 1c. SoftwareOnly backend

- [ ] 1.9 Create `security/src/attest/sw_backend.rs` (no feature gate — always available as a fallback).
- [ ] 1.10 Implement `SwBackend::handle_request`: assembles the measurement bundle from `BootMeasurementLog`, signs with ML-DSA-65 + Ed25519 counter-sig only, emits with `HardwareQuote: null`.
- [ ] 1.11 Document this is the weakest tier — verifier policies should explicitly opt-in to accept SoftwareOnly responses.

### 1d. Transport adapters

- [ ] 1.12 Add `POST /v1/attest` route to the existing container HTTP server (`container/src/main.rs` or equivalent). Content-Type `application/cbor`, body is `AttestRequest`, response is `AttestResponse`.
- [ ] 1.13 Add the same route on the QUIC/HTTP3 side (existing `net/quic/` server).
- [ ] 1.14 Add an `Attest` capability call for in-host IPC verifiers. Defined in `security/src/capability.rs`, requires explicit cap-grant.
- [ ] 1.15 Document the three transports in `docs/remote-attestation-protocol.md`.

### 1e. Reference verifier crate

- [ ] 1.16 Create `tools/attest-verifier/Cargo.toml` and `tools/attest-verifier/src/main.rs`. `std` + `tokio` based, build-host-only.
- [ ] 1.17 Implement `verify` subcommand: load release record, generate nonce, build request, POST to endpoint, parse response, verify all four layers (hardware sig, counter-sig, nonce, measurement match), emit audit record.
- [ ] 1.18 Implement `inspect` subcommand: pretty-print a saved `attest-record-*.cbor`.
- [ ] 1.19 Implement `verify-batch` subcommand: take a fleet manifest (list of endpoints), verify each, emit per-instance records.
- [ ] 1.20 Define trust-anchor directory layout (`~/.config/smallaios-attest/`), implement loader.
- [ ] 1.21 Document the release-record JSON shape and how it's produced/signed in `docs/release-attestation-records.md`.

### 1f. CI smoke (x86-64 + swtpm)

- [ ] 1.22 Add `attest-tpm2-swtpm-smoke` CI job: boots kernel under QEMU + swtpm (re-uses the fixture from `boot-root-of-trust-v1`'s Phase 1 CI), runs `attest-verifier verify` against the booted instance, asserts PASS.
- [ ] 1.23 Same job tests `PqcMode: "classical-only"` and `PqcMode: "hybrid"` separately. `pqc-only` returns ECDSA-AK signature omitted but PCR digest still bound via the counter-sig — verify the verifier accepts it.
- [ ] 1.24 Advisory at land. Promote to gate after one stable week.

### 1g. Sub-phase A close-out

- [ ] 1.25 Update `docs/boot-security-matrix.md`: add a new row "Remote attestation" with **Yes (TPM2 + counter-sig)** for x86-64, **Yes (PSA-IA + counter-sig)** for AArch64 (pending Sub-phase B), **Software-only counter-sig** for RISC-V.
- [ ] 1.26 Update `CLAUDE.md` "Current state" with the attestation capability.
- [ ] 1.27 PR title: `feat(security/attest): remote-attestation-v1 sub-phase A — protocol + x86-64 TPM backend`. Target `develop`.

## 2. Sub-phase B — AArch64 PSA-IA backend

### 2a. PSA-IA TA invocation

- [ ] 2.1 Create `security/src/attest/psa_ia_backend.rs` gated `cfg(feature = "op-tee")`. Initialization: open a session to the PSA-IA TA via the `op-tee-bridge-v1` bridge.
- [ ] 2.2 Implement `PsaIaBackend::handle_request`:
  - assemble the SmallAIOS-side measurement bundle (SHA-3-256 of kernel + boot config + each model + config);
  - open session to PSA-IA TA UUID `f0b13b9b-8b8a-4f57-9b95-79c83e3b09cd`;
  - invoke `PSA_INITIAL_ATTEST_GET_TOKEN` with the verifier nonce + measurement bundle as the `psa_arg_t`;
  - receive the EAT-CWT;
  - ML-DSA-65 + Ed25519 counter-sign the CWT and the measurement bundle;
  - wrap as `AttestResponse` with `Backend = "psa-ia"`.
- [ ] 2.3 Cache the session across requests (avoid the ~10ms open-session cost per request); reopen on TA-side errors.
- [ ] 2.4 Honor `PqcMode` (same semantics as TPM2 backend).
- [ ] 2.5 Enforce server-side anti-replay bloom filter (shared with the TPM2 backend, one bloom per server).

### 2b. Verifier crate updates

- [ ] 2.6 Extend verifier to accept `Backend: "psa-ia"` responses. Verify the inner EAT-CWT signature against the PSA-IA HUK-derived public key (loaded from `~/.config/smallaios-attest/trust-anchors/psa-ia-pub-keys/`).
- [ ] 2.7 Add PSA-IA HUK-derived public-key loading documentation to `docs/attest-verifier-usage.md`. The keys come from Arm's PSA test suite reference TA or from per-SoC Arm-provided publications.
- [ ] 2.8 Add `--backend-hint psa-ia` flag for verifier-side testing against AArch64 endpoints.

### 2c. CI smoke (AArch64 + OP-TEE + PSA-IA)

- [ ] 2.9 Add `attest-psa-ia-qemu-smoke` CI job: re-uses the `op-tee-qemu-smoke` fixture from `op-tee-bridge-v1`, boots SmallAIOS with `--features op-tee` as BL33, runs `attest-verifier verify` against the booted AArch64 instance, asserts PASS.
- [ ] 2.10 Verify the PSA-IA TA UUID resolves and the CWT round-trip works against the upstream OP-TEE PSA-IA TA.
- [ ] 2.11 Advisory at land; promote with the same one-stable-week rule.

### 2d. Sub-phase B close-out

- [ ] 2.12 Update `docs/boot-security-matrix.md` AArch64 row: "Remote attestation" cell drops "(pending Sub-phase B)" and is finalized.
- [ ] 2.13 PR title: `feat(security/attest): remote-attestation-v1 sub-phase B — AArch64 PSA-IA backend`. Target `develop`.

## 3. Cross-phase verification

- [ ] 3.1 Cross-platform: verifier runs against a fleet manifest with mixed x86-64-TPM, AArch64-PSA-IA, and RISC-V-software-only endpoints, emits per-instance records, all valid per their respective policy modes.
- [ ] 3.2 Negative test: tamper with the kernel hash on the server, verifier reports FAIL with `MeasurementMismatch`.
- [ ] 3.3 Negative test: send a stale nonce (re-use one), verifier reports FAIL with `NonceMismatch` (when checking client-side) AND server reports `ReplayedNonce` (when re-issued within the window).
- [ ] 3.4 Negative test: `PqcMode: "pqc-only"` against a verifier configured for `PqcMode: "classical-only"` (mode mismatch on the verifier side, not the server). Verifier reports FAIL with `SignatureModeMismatch`.
- [ ] 3.5 `openspec validate remote-attestation-v1` returns valid.

## 4. Docs

- [ ] 4.1 `docs/remote-attestation-protocol.md`: wire format spec (CBOR schema for `AttestRequest`, `AttestResponse`, `NormalizedReport`), policy semantics, transport bindings, error codes.
- [ ] 4.2 `docs/attest-verifier-usage.md`: operator-side how-to, trust anchor management, fleet-manifest format, audit record retention guidance.
- [ ] 4.3 `docs/release-attestation-records.md`: release-engineering side, how to publish a `release-X.Y.Z.json`, how it ties to `boot-root-of-trust-v1` Phase 4 signing.
- [ ] 4.4 Update `CLAUDE.md` "Container Environment Variables" section if any attestation-related env vars are added (e.g. `SMALLAIOS_ATTEST_PORT`).
