## ADDED Requirements

### Requirement: Cross-platform confidential-compute abstraction layer

The `smallaios-security` crate SHALL provide platform-agnostic abstraction traits (`SealedRegion`, `AttestableEnclave`, `SharedWindow`) that all confidential-compute backends (ARM CCA Realms, Intel TDX, AMD SEV-SNP) implement, so the inference runtime and the attestation surface depend only on the traits, not on platform-specific backends.

#### Scenario: SealedRegion zeroes on transition to shared

- **GIVEN** a `SealedRegion` containing decrypted model weights inside a confidential enclave
- **WHEN** the runtime consumes the region via `region.into_shared()` to expose it as a `SharedWindow`
- **THEN** the underlying pages SHALL be zeroed before the platform-level state transition completes
- **AND** the resulting `SharedWindow`'s contents SHALL be observable by the host as all-zero bytes (no leakage of the previously-private content)
- **AND** the type-state design SHALL make leakage by accidental in-place mutation impossible — only `into_shared()` produces a `SharedWindow` from a `SealedRegion`

#### Scenario: AttestableEnclave produces platform-tagged reports

- **GIVEN** a SmallAIOS build with exactly one confidential-compute backend feature enabled (e.g. `cca-realm`)
- **WHEN** the runtime calls `AttestableEnclave::produce_report(claims)`
- **THEN** the returned report SHALL be the platform-native attestation evidence (Realm Attestation Token for CCA, TD Report for TDX, SNP Attestation Report for SEV-SNP)
- **AND** the runtime SHALL surface the platform tag via `AttestableEnclave::backend_tag()` returning one of `"cca-realm"`, `"intel-tdx"`, `"amd-sev-snp"`
- **AND** the tag SHALL appear in the `AttestResponse::Backend` field per the `remote-attestation-v1` protocol

#### Scenario: Mutually-exclusive backend features

- **GIVEN** a build attempting to enable two confidential-compute backend features simultaneously (e.g. `cca-realm` + `intel-tdx`)
- **WHEN** Cargo compiles
- **THEN** a `compile_error!` SHALL halt the build with a documented message naming the conflicting features and the rationale (one enclave per kernel image)

### Requirement: ARM CCA Realm bring-up (Phase 1)

When built with `--features cca-realm` and run as a Realm under a CCA-RME-capable host, SmallAIOS SHALL execute correctly, produce a Realm Attestation Token via the Realm Services Interface, and run a confidential ONNX inference path.

#### Scenario: Realm boots under QEMU + tf-rmm

- **GIVEN** the `cca-realm` build (`smallaios-cca-realm` bin) and a pinned `tf-rmm` build
- **WHEN** `qemu-system-aarch64 -cpu max,rme=on -machine virt,gic-version=3 -bios tf-a.bin -kernel smallaios-cca-realm` runs
- **THEN** the Realm SHALL initialize, print a boot banner to the documented Realm-side console, and reach its scheduler idle loop without exception
- **AND** the Realm's first action SHALL be `RSI_REALM_CONFIG` to discover its IPA size, hash algorithm, and attestation algorithm — recorded in the boot measurement log
- **AND** the smoke job SHALL complete within 60 seconds wall-clock

#### Scenario: Attestation token round-trip via RSI_ATTEST_TOKEN_*

- **GIVEN** a booted Realm and a verifier-supplied 32-byte nonce N
- **WHEN** the Realm receives an attestation request, it SHALL call `RSI_ATTEST_TOKEN_INIT(N)`, then loop `RSI_ATTEST_TOKEN_CONTINUE` until the full token is fetched
- **THEN** the assembled token SHALL be a valid CBOR-encoded Realm Attestation Token (RAT) containing a Platform Token (signed by the CCA Platform Attestation Key) and a Realm Token (signed by an RMM-derived key)
- **AND** the token SHALL be wrapped in the `remote-attestation-v1` `HybridQuote` envelope with the `cca-realm` backend tag
- **AND** the verifier crate SHALL validate the RAT against the configured `cca-platform-roots/` trust anchors and assert PASS

#### Scenario: Confidential ONNX model loading via attested key release

- **GIVEN** an encrypted ONNX model bundle + a wrapped decryption key released by the model owner only after attestation verification
- **WHEN** the Realm receives the wrapped key
- **THEN** the runtime SHALL unwrap `K_model` using the Realm's RSI-supplied attestation key pair
- **AND** the runtime SHALL decrypt the model bytes from a shared-memory input window into a `SealedRegion` allocated in Realm-private memory
- **AND** the decrypted model bytes SHALL never appear in shared memory at any point during loading or inference
- **AND** the CI smoke SHALL run a vector-add inference and assert correct output

#### Scenario: Granule state transitions follow the documented RSI flow

- **WHEN** the runtime allocates a new `SealedRegion`
- **THEN** the implementation SHALL call `RSI_IPA_STATE_SET` with the target page IPA and the `RAM_PRIVATE` state
- **AND** the call SHALL succeed for any page within the Realm's permitted IPA range
- **AND** subsequent reads/writes from the Realm SHALL succeed
- **AND** reads from the host (outside the Realm) SHALL see ciphertext or platform-defined abort behavior

### Requirement: Intel TDX bring-up (Phase 2)

When built with `--features intel-tdx` and run as a Trust Domain under a TDX-capable host, SmallAIOS SHALL execute correctly, produce a TD Report via `TDG.MR.REPORT`, and run the same confidential ONNX inference path as Phase 1.

#### Scenario: TD boots and round-trips a TD Report

- **GIVEN** the `intel-tdx` build (`smallaios-tdx` bin) and a TDX-capable QEMU + TDX-module artifact pinning
- **WHEN** `qemu-system-x86_64 -cpu host,+tdx` boots the TD
- **THEN** the TD SHALL initialize, reach its idle loop, and accept attestation requests
- **AND** an attestation request SHALL produce a TD Report via `TDG.MR.REPORT(nonce)` wrapped in the existing `HybridQuote` envelope with `Backend: "intel-tdx"`
- **AND** the verifier crate SHALL validate the TD Report against Intel's DCAP attestation collateral and assert PASS

#### Scenario: Cross-platform abstraction is honored

- **GIVEN** the TDX backend
- **THEN** `SealedRegion::allocate(pages)` SHALL succeed using TDX `TDG.MEM.PAGE.ACCEPT` semantics
- **AND** `AttestableEnclave::backend_tag()` SHALL return `"intel-tdx"`
- **AND** the same `onnx-rt/src/confidential.rs` loader code path SHALL function unchanged when built with `--features intel-tdx`

### Requirement: AMD SEV-SNP bring-up (Phase 3)

When built with `--features amd-sev-snp` and run as an SNP VM under an SNP-capable host, SmallAIOS SHALL execute correctly, produce an SNP Attestation Report via `SNP_GUEST_REQUEST MSG_REPORT_REQ`, and run the same confidential ONNX inference path.

#### Scenario: SNP VM boots and round-trips an Attestation Report

- **GIVEN** the `amd-sev-snp` build and an SEV-SNP-capable QEMU + SEV firmware pinning
- **WHEN** `qemu-system-x86_64 -cpu host,+sev-snp` boots the VM
- **THEN** the VM SHALL initialize, reach its idle loop, and accept attestation requests
- **AND** an attestation request SHALL produce an SNP Attestation Report wrapped in the existing `HybridQuote` envelope with `Backend: "amd-sev-snp"`
- **AND** the verifier crate SHALL validate the report against AMD's attestation collateral (vcek cert + AMD root) and assert PASS

### Requirement: Documented threat model

The repository SHALL maintain a `docs/confidential-compute-threat-model.md` covering the six standard adversary classes (co-tenant VM, hypervisor, datacenter operator with physical access, compromised firmware, compromised CPU, side-channel timing), with a per-phase defense column for ARM CCA, Intel TDX, and AMD SEV-SNP.

#### Scenario: Threat model exists, is reviewed, and is updated per phase

- **GIVEN** the change at Phase 1 close-out
- **THEN** `docs/confidential-compute-threat-model.md` SHALL exist with all six adversary rows populated for the CCA column
- **AND** the document SHALL note that the TDX and SEV-SNP columns are pending Phases 2 and 3
- **AND** the document SHALL be reviewed by at least one engineer who did not implement Phase 1 (independent review)

- **GIVEN** the change at Phase 2 close-out
- **THEN** the TDX column SHALL be filled
- **AND** the residual-risks section SHALL be updated with any TDX-specific findings

- **GIVEN** the change at Phase 3 close-out
- **THEN** the SEV-SNP column SHALL be filled
- **AND** the residual-risks section SHALL be finalized for the change's archival

### Requirement: Verifier crate accepts all three platform reports

The `tools/attest-verifier/` crate SHALL accept `AttestResponse` payloads with `Backend` values `"cca-realm"`, `"intel-tdx"`, or `"amd-sev-snp"` and SHALL dispatch to the appropriate platform-specific report-parsing logic + trust-anchor directory automatically.

#### Scenario: Single verifier binary covers all platforms

- **GIVEN** a developer with the `attest-verifier` binary and a populated `~/.config/smallaios-attest/trust-anchors/` containing CCA roots, TDX collateral, and SNP collateral
- **WHEN** the developer runs `attest-verifier verify` against any of the three platform backends
- **THEN** the verifier SHALL detect the backend from `AttestResponse::Backend`
- **AND** the verifier SHALL load the matching trust anchor set automatically
- **AND** the audit record SHALL identify the platform clearly so downstream auditors see which confidential-compute path was used

#### Scenario: Attested key release subcommand

- **GIVEN** an attested-target endpoint and an encrypted key the operator wants to release only to a verified-attested target
- **WHEN** the operator runs `attest-verifier release-key-to-attested-target --target <endpoint> --release-record release-X.Y.Z.json --key-to-release encrypted-key.bin`
- **THEN** the verifier SHALL first issue an attestation request and validate the response
- **AND** only on PASS SHALL the verifier transmit the key (encrypted to the target's attestation-supplied wrap public key) to the target
- **AND** the verifier SHALL emit an audit record of the release decision
