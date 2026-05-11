# Tasks — confidential-compute-v1

## 0. Phase 1 prerequisites + design

- [ ] 0.1 Confirm `op-tee-bridge-v1` has merged (gives us the `arch/aarch64/src/smc.rs` infrastructure that RSI calls reuse).
- [ ] 0.2 Confirm `remote-attestation-v1` Sub-phase A has merged (gives us the `HybridQuote` envelope and the verifier crate to extend).
- [ ] 0.3 Pin `tf-rmm` upstream commit SHA to use for CI. Document in `docs/cca-realm-deployment.md` after that doc is created.
- [ ] 0.4 Pin the QEMU version that supports `+rme=on` (needs QEMU 8.2+ — confirm CI runner image carries it).
- [ ] 0.5 Pin the Arm CCA spec version + RSI spec version that Phase 1 targets. Document.
- [ ] 0.6 Capture and write the threat model in `docs/confidential-compute-threat-model.md` covering all six adversary classes per the design table. Independent review (security-aware engineer, not the implementer) before implementation starts.

## 1. Phase 1 — ARM CCA Realm bring-up

### 1a. Cross-platform abstraction layer

- [ ] 1.1 Create `security/src/confidential/mod.rs` defining `SealedRegion`, `AttestableEnclave`, `SharedWindow` traits per the design.
- [ ] 1.2 Add `ConfidentialError` enum covering platform-agnostic failure modes (`AllocationFailed`, `StateTransitionDenied`, `AttestationUnavailable`, `BackendNotInitialized`).
- [ ] 1.3 Add doc comments explaining each trait's responsibilities and noting the three intended backends (CCA, TDX, SNP).

### 1b. CCA RSI client

- [ ] 1.4 Create `security/src/confidential/cca_rsi_ids.rs` with `const u32` for each RSI FID per the Arm RSI spec.
- [ ] 1.5 Create `arch/aarch64/src/cca/mod.rs` exposing `rsi_realm_config()`, `rsi_ipa_state_get(ipa)`, `rsi_ipa_state_set(ipa, count, state)`, `rsi_attest_token_init(challenge)`, `rsi_attest_token_continue(buf)`, `rsi_host_call(args)`. Each wraps an `smc_call` to the appropriate FID.
- [ ] 1.6 Unit-test the RSI client against a mock SMC dispatcher (cfg(test) shim — same pattern as `op-tee-bridge-v1`'s tests).

### 1c. Granule state management + Sealed/SharedRegion

- [ ] 1.7 Create `arch/aarch64/src/cca/granule.rs` with `transition_to_shared(ipa, count)` and `transition_to_private(ipa, count)` wrappers.
- [ ] 1.8 Implement `SealedRegion` and `SharedWindow` traits in `arch/aarch64/src/cca/region.rs`. RAII drops transition pages back to the default state and zero them.
- [ ] 1.9 Implement type-state transitions: `SealedRegion::into_shared(self) -> SharedWindow` consumes the sealed region, zeroes it, transitions to shared. Reverse direction also provided.
- [ ] 1.10 Unit-test the type-state transitions (zero-on-transition behavior, no leakage across the boundary).

### 1d. Realm attestation backend

- [ ] 1.11 Create `security/src/confidential/cca_backend.rs` implementing `AttestableEnclave` for CCA. `Report = Vec<u8>` (CBOR-encoded RAT).
- [ ] 1.12 Implement `CcaBackend::produce_report(claims)`: call `rsi_attest_token_init(claims_hash)`, loop on `rsi_attest_token_continue` until EOF, return the assembled CBOR.
- [ ] 1.13 Wire into `remote-attestation-v1`'s backend selection: when running as a Realm, `AttestBackend::select()` returns `Cca(CcaBackend::initialize())`. The `AttestResponse::Backend` field carries `"cca-realm"`.

### 1e. Verifier crate CCA support

- [ ] 1.14 Extend `tools/attest-verifier/` with CCA RAT parsing: pull the CBOR, validate the platform token signature against the configured Arm CCA Platform Attestation Key root, validate the Realm token signature against the platform-token-supplied RMM key, extract Realm measurements.
- [ ] 1.15 Add `trust-anchors/cca-platform-roots/` documented directory layout.
- [ ] 1.16 Add `attest-verifier release-key-to-attested-target --target https://realm-endpoint --release-record release-X.Y.Z.json --key-to-release encrypted-key.bin` subcommand implementing the attested key release pattern.
- [ ] 1.17 Document the verifier-side CCA flow in `docs/cca-attestation-key-release.md`.

### 1f. Confidential ONNX loader

- [ ] 1.18 Create `onnx-rt/src/confidential.rs` exposing `ConfidentialOnnxRuntime::load_encrypted_model(blob: &[u8], wrapped_key: &[u8])`.
- [ ] 1.19 Implement the wrapped-key unwrap: use `RSI_REALM_CONFIG`-supplied attestation key pair to unwrap `K_model`. (The exact key-wrap scheme is documented in the design — likely ECDH-P384 + HKDF + AES-256-GCM.)
- [ ] 1.20 Decrypt model bytes from a shared-memory input window into a `SealedRegion`. The decrypted model never appears in shared memory.
- [ ] 1.21 Add a `cca-realm` Cargo feature on `smallaios-onnx-rt` that links the confidential loader. Without the feature, no symbol changes.
- [ ] 1.22 Unit-test against a fixture (encrypted dummy ONNX model + valid wrapped key + matching Realm measurement).

### 1g. Realm build pipeline

- [ ] 1.23 Create `arch/aarch64/linker-cca-realm.ld` setting the Realm image base + section layout per the tf-rmm reference implementation's expected Realm entry point.
- [ ] 1.24 Add `[[bin]] smallaios-cca-realm` to `arch/aarch64/Cargo.toml` with `required-features = ["cca-realm"]`.
- [ ] 1.25 Add `just build-cca-realm` recipe: `cargo build --release --target aarch64-unknown-none -p smallaios-arch-aarch64 --bin smallaios-cca-realm --features cca-realm`.
- [ ] 1.26 Document the produced artifact's expected boot-time layout in `docs/cca-realm-deployment.md`.

### 1h. CI smoke

- [ ] 1.27 Pre-bake tf-rmm into the CI runner image (or pre-cache its build artifacts). Pin tf-rmm SHA.
- [ ] 1.28 Add `cca-realm-qemu-smoke` CI job:
  - `cargo build --features cca-realm`;
  - `qemu-system-aarch64 -cpu max,rme=on -machine virt,gic-version=3 -bios tf-a.bin -kernel smallaios-cca-realm`;
  - run `attest-verifier verify --backend cca-realm --endpoint <Realm's exposed endpoint>` against the booted Realm;
  - assert PASS.
- [ ] 1.29 Add a smoke test for the confidential ONNX path: provision an encrypted model + wrapped key fixture, run the loader, assert decryption succeeds + a tiny inference returns expected output.
- [ ] 1.30 Advisory (`continue-on-error: true`) at land. Promotion to gate happens when CCA-capable hardware is available for additional validation.

### 1i. Docs

- [ ] 1.31 `docs/cca-realm-deployment.md`: architecture diagram, boot flow, build command, deployment notes for prospective customer environments.
- [ ] 1.32 `docs/confidential-compute-threat-model.md`: full threat model per the design's adversary table; Phase 1 covers the CCA column, Phases 2/3 fill the others.
- [ ] 1.33 `docs/cca-attestation-key-release.md`: model-owner-side and Realm-side flows for attested key release. References the verifier crate's `release-key-to-attested-target` subcommand.
- [ ] 1.34 Update `docs/boot-security-matrix.md` with a new "Confidential compute" section detailing Phase 1's CCA coverage.
- [ ] 1.35 Update `CLAUDE.md` "Current state" with the CCA Realm capability.

### 1j. Phase 1 close-out

- [ ] 1.36 PR title: `feat(security/confidential): confidential-compute-v1 phase 1 — ARM CCA Realm bring-up`. Target `develop`.
- [ ] 1.37 PR description includes a captured RAT round-trip log from the QEMU+tf-rmm smoke, plus a confidential-inference output trace.

## 2. Phase 2 — Intel TDX

### 2a. TDX abstraction backend

- [ ] 2.1 Create `arch/x86_64/src/tdx/mod.rs` exposing `tdcall(leaf, args)` wrapping the `TDCALL` instruction.
- [ ] 2.2 Create `security/src/confidential/tdx_backend.rs` implementing `SealedRegion`, `AttestableEnclave`, `SharedWindow` for TDX. Page-state transitions use `TDG.MEM.PAGE.ACCEPT` and `TDG.VP.VMCALL` flavors.
- [ ] 2.3 Implement `AttestableEnclave::produce_report` via `TDG.MR.REPORT` plus `TDG.MR.RTMR.EXTEND` for measurement-log extends mid-execution.
- [ ] 2.4 Pin the TDX-module spec version and the Intel SGX DCAP attestation-key shape SmallAIOS targets.

### 2b. TDX build target

- [ ] 2.5 Add `intel-tdx` Cargo feature on `smallaios-arch-x86_64`. Mutually exclusive with `cca-realm` and `amd-sev-snp` (compile_error! on conflict).
- [ ] 2.6 Add `[[bin]] smallaios-tdx` with `required-features = ["intel-tdx"]`.
- [ ] 2.7 Linker / build wiring per Intel TDX guest specs.

### 2c. Verifier crate TDX support

- [ ] 2.8 Extend the verifier with TDX TD Report parsing (Intel-defined binary format).
- [ ] 2.9 Add Intel SGX DCAP attestation-collateral fetching (the verifier downloads the platform's TCB info + QE identity from Intel's PCS endpoint or a local mirror).
- [ ] 2.10 Document the TDX-side trust anchors in `docs/intel-tdx-deployment.md`.

### 2d. CI

- [ ] 2.11 Add `intel-tdx-qemu-smoke` CI job using `qemu-system-x86_64 -cpu host,+tdx` (Intel TDX QEMU support stabilized in QEMU 8.x with the Intel TDX-module artifact loaded). Boot SmallAIOS as a TD, attest, verify.
- [ ] 2.12 Advisory at land; gate-promote criteria same as Phase 1.

### 2e. Phase 2 close-out

- [ ] 2.13 Update `docs/confidential-compute-threat-model.md` TDX column.
- [ ] 2.14 Update `docs/boot-security-matrix.md` x86-64 confidential-compute cell.
- [ ] 2.15 PR title: `feat(security/confidential): confidential-compute-v1 phase 2 — Intel TDX`. Target `develop`.

## 3. Phase 3 — AMD SEV-SNP

### 3a. SNP abstraction backend

- [ ] 3.1 Create `arch/x86_64/src/sev_snp/mod.rs` exposing the GHCB (Guest-Hypervisor Communication Block) interface for hypervisor calls.
- [ ] 3.2 Create `security/src/confidential/sev_snp_backend.rs` implementing the three traits. Page-state transitions use `PSMASH` and `RMP_ADJUST` (or rather, request via GHCB).
- [ ] 3.3 Implement `produce_report` via `SNP_GUEST_REQUEST` `MSG_REPORT_REQ`.

### 3b. SNP build target

- [ ] 3.4 Add `amd-sev-snp` Cargo feature on `smallaios-arch-x86_64`, mutually exclusive with `intel-tdx` and `cca-realm`.
- [ ] 3.5 Add `[[bin]] smallaios-sev-snp`.
- [ ] 3.6 Linker / build wiring per AMD SEV-SNP guest specs.

### 3c. Verifier crate SNP support

- [ ] 3.7 Extend the verifier with SNP attestation report parsing (AMD-defined binary format).
- [ ] 3.8 Add AMD attestation-collateral fetching (vcek certificate, AMD root keys).
- [ ] 3.9 Document trust anchors in `docs/amd-sev-snp-deployment.md`.

### 3d. CI

- [ ] 3.10 Add `amd-sev-snp-qemu-smoke` CI job using `qemu-system-x86_64 -cpu host,+sev-snp` (AMD SEV-SNP QEMU support is available with the SEV firmware loaded).
- [ ] 3.11 Advisory at land.

### 3e. Phase 3 close-out

- [ ] 3.12 Update `docs/confidential-compute-threat-model.md` SEV-SNP column.
- [ ] 3.13 Update `docs/boot-security-matrix.md` x86-64 confidential-compute cell to reflect both TDX and SNP coverage.
- [ ] 3.14 PR title: `feat(security/confidential): confidential-compute-v1 phase 3 — AMD SEV-SNP`. Target `develop`.

## 4. Cross-phase verification

- [ ] 4.1 The same `attest-verifier` binary verifies CCA, TDX, and SNP responses transparently (selecting the correct trust-anchor / report-parser per `AttestResponse::Backend`).
- [ ] 4.2 Confidential ONNX inference smoke runs on all three platforms (QEMU-emulated) with the same model bundle and the same encryption key wrap shape.
- [ ] 4.3 `openspec validate confidential-compute-v1` returns valid.
- [ ] 4.4 The threat-model document is reviewed by an independent reviewer before the Phase 3 PR merges.

## 5. Long-term housekeeping

- [ ] 5.1 Track CCA silicon GA dates and update `docs/cca-realm-deployment.md` with on-hardware verification results when hardware is available.
- [ ] 5.2 Track confidential GPU compute (`confidential-gpu-v1`) as a follow-up change once Phase 1-3 land.
- [ ] 5.3 Re-evaluate PCIe TEE-IO (TDISP) scope when production silicon ships (estimated 2027+).
