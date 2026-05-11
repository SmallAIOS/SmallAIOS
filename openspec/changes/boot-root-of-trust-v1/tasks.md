# Tasks — boot-root-of-trust-v1

## 0. Prerequisites

- [ ] 0.1 Confirm the existing `verified-boot` Cargo feature (`security/Cargo.toml:16`) + the `BootMeasurementLog` API (`kernel/src/boot_integrity.rs`) are stable. Pin to the develop SHA at change start; any breaking change to either surface during the change forces a re-base of this work.
- [ ] 0.2 Confirm `swtpm` (software TPM 2.0 emulator) is available in CI runners (`apt install swtpm swtpm-tools`); if not, add to the CI image bootstrap step before Phase 1.
- [ ] 0.3 Capture and pin the JetPack 6.2.1 TF-A build's event-log location (DTB reserved-memory node name + offset) on a representative Orin NX host. Paste in the PR description for Phase 2 ground truth.
- [ ] 0.4 Capture the current `cargo-vet` audit status for any newly-touched dependencies (none expected — clean-room — but verify).

## 1. Phase 1 — TPM 2.0 driver + hardware-extended measurement log (x86-64)

### 1a. ACPI TPM2 table parser

- [ ] 1.1 Extend `kernel/src/acpi/` (existing) with a `tpm2_table.rs` module parsing the TPM2 ACPI table (TCG spec, table ID `TPM2`). Returns `StartMethod`, `ControlAreaPhysicalAddress`, `LAML/LASA` (event log pointers — also useful for Phase 2 / future fTPM).
- [ ] 1.2 Add detection logic: if no `TPM2` table, return `TpmAbsent`; expose as a probe API.

### 1b. TIS + CRB transports

- [ ] 1.3 Create `security/src/tpm2/mod.rs` exposing a `Tpm2Device` trait with `send_command(&[u8]) -> Result<Vec<u8>>` and `bank() -> PcrBank`. Two implementations: `tis.rs` (MMIO at `0xFED40000`), `crb.rs` (MMIO at ACPI-supplied base).
- [ ] 1.4 Implement `Tpm2Device::send_command` for TIS: STS / DATA_FIFO / locality 0 sequencing per TCG TIS spec rev 1.3.
- [ ] 1.5 Implement `Tpm2Device::send_command` for CRB: command/response buffer write + doorbell + status poll.
- [ ] 1.6 Add a software shim `swtpm.rs` (cfg-gated to `test` + `feature = "tpm-swtpm-shim"`) that proxies to `swtpm` over a socket for in-CI testing — used by the `tpm2-swtpm-smoke` CI job.

### 1c. Minimal TPM 2.0 command set

- [ ] 1.7 Implement `TPM2_Startup(Clear)` — required at every boot.
- [ ] 1.8 Implement `TPM2_GetCapability(TPM_CAP_PCRS)` to enumerate PCR banks and confirm SHA-256 is present.
- [ ] 1.9 Implement `TPM2_PCR_Extend(pcr_index, sha256_digest)` — the core extend primitive.
- [ ] 1.10 Implement `TPM2_PCR_Read(pcr_indices)` — return the current PCR state, used by `quote` and by audit dumps.
- [ ] 1.11 Implement `TPM2_CreatePrimary` + `TPM2_LoadExternal` to materialize a transient AK in the endorsement hierarchy (when no provisioned AK exists).
- [ ] 1.12 Implement `TPM2_Quote(ak_handle, pcr_selection, nonce)` returning `TPMS_ATTEST` + `TPMT_SIGNATURE`.

### 1d. BootMeasurementLog hardware-extend wiring

- [ ] 1.13 Add a `tpm-attest` Cargo feature to `smallaios-security`. Gates the `tpm2` module.
- [ ] 1.14 Modify `kernel/src/boot_integrity.rs::BootMeasurementLog::add_entry` so that when `tpm-attest` is on AND a TPM is present, the entry's digest is also extended into the configured PCR (PCR 11-14 per the proposal mapping).
- [ ] 1.15 Document the PCR mapping in `docs/attestation-quote-format.md` (new). PCR 11 = kernel image, 12 = boot config, 13 = ONNX models, 14 = SmallAIOS configuration.
- [ ] 1.16 Add the SHA-3→SHA-256 bridge: SmallAIOS-side measurements are SHA-3-256; the PCR-extend input is SHA-256 of the SHA-3-256 digest. Document the bridge in the same file.

### 1e. Hybrid quote format

- [ ] 1.17 Create `security/src/tpm2_quote.rs` defining the `HybridQuote` CBOR structure (proposal: TpmQuote, MeasurementLog, HybridSignature, HybridPubKeyDigest, Nonce, Timestamp).
- [ ] 1.18 Implement `produce_hybrid_quote(nonce: &[u8]) -> HybridQuote`: gather measurement log, call `TPM2_Quote`, ML-DSA-65 + Ed25519 hybrid sign the bundle.
- [ ] 1.19 Implement `verify_hybrid_quote(quote: &HybridQuote, ak_pub, counter_pub, expected_root) -> Result<()>` — used by `remote-attestation-v1` and by the CI smoke job.
- [ ] 1.20 Extend PCR 14 with `SHA-3-256(counter_pub)` at boot, so the counter-signing public key is in the chain of trust.

### 1f. CI smoke (swtpm)

- [ ] 1.21 Add `tpm2-swtpm-smoke` job to `.github/workflows/ci.yml`: launches `swtpm socket --tpmstate dir=/tmp/swtpm --ctrl type=unixio,path=/tmp/swtpm.sock --tpm2`, boots kernel under QEMU with `-chardev socket,id=chrtpm,path=/tmp/swtpm.sock -tpmdev emulator,id=tpm0,chardev=chrtpm -device tpm-tis,tpmdev=tpm0`, asserts PCR-extend path executes, dumps PCRs via `swtpm_ioctl`.
- [ ] 1.22 In the same CI job, request a quote, verify it externally (using `swtpm`'s exported AK), assert success. Advisory (`continue-on-error: true`) initially; promote to gate after one stable week.

### 1g. Phase 1 close-out

- [ ] 1.23 Update `docs/boot-security-matrix.md` x86-64 row: TPM Measured Boot column flips from **No** to **Yes (Phase 1)** with a link to this change.
- [ ] 1.24 Update `CLAUDE.md` "Current state" with the TPM 2.0 driver capability.
- [ ] 1.25 PR title: `feat(security/tpm2): boot-root-of-trust-v1 phase 1 — TPM 2.0 measured boot on x86-64`. Target `develop`.

## 2. Phase 2 — TF-A BL31 measurement chain consumption (AArch64)

### 2a. DTB parser improvements (also closes unikernel-orin-bringup-v1 task 2.12)

- [ ] 2.1 Extend `kernel::mem::phys::parse_dtb` to handle multi-cell `reg = <hi lo hi lo>` (Tegra234 uses 64-bit addresses split across two 32-bit cells).
- [ ] 2.2 Extend the parser to walk `/reserved-memory/` and expose its subnodes by name.
- [ ] 2.3 Unit-test against (a) the existing QEMU-virt DTB fixture, (b) a captured Tegra234 DTB snippet (added as a test fixture).

### 2b. TF-A event log reader

- [ ] 2.4 Create `arch/aarch64/src/tf_a_event_log.rs` parsing the TCG event log format (TCG_PCClientPCREvent v2). Validate header magic, walk entries, return a `Vec<EventLogEntry>` (no-alloc with `heapless::Vec` at compile-time-known max-entry count).
- [ ] 2.5 Wire `parse_dtb` discovery → `tf_a_event_log` parser: at AArch64 boot, look up `/reserved-memory/tf-a-event-log`, parse the contents, fail gracefully if absent.
- [ ] 2.6 Capture a real Tegra Orin TF-A event log via JTAG / serial dump, save as a CI fixture for replay testing.

### 2c. Chain reconciliation

- [ ] 2.7 Extend `BootMeasurementLog` with a `prepend_chain(entries: &[EventLogEntry])` API. Order: TF-A entries (oldest first), SmallAIOS entries appended.
- [ ] 2.8 Add a `merged_root_hash() -> [u8; 32]` API computing the SHA-256 replay hash over the merged log.
- [ ] 2.9 If a TPM is present (rare on AArch64 but possible on Neoverse server boards), reuse Phase 1's PCR-extend wiring against the merged chain.

### 2d. CI replay test

- [ ] 2.10 Add `tf-a-event-log-replay` CI job: feeds the captured fixture into the parser + reconciler, asserts the merged-root hash matches the expected value.

### 2e. Docs

- [ ] 2.11 Create `docs/aarch64-measured-boot.md` covering: TF-A BL31 measurement model, where the event log lives, how SmallAIOS reads it, what happens when it's absent.
- [ ] 2.12 Update `docs/boot-security-matrix.md` AArch64 row: TF-A event log column flips from **Partial** to **Yes (Phase 2)**.

### 2f. Phase 2 close-out

- [ ] 2.13 PR title: `feat(arch/aarch64): boot-root-of-trust-v1 phase 2 — TF-A event log consumption`. Target `develop`.

## 3. Phase 3 — RISC-V PMP via SBI vendor extension (best-effort)

- [ ] 3.1 Create `arch/riscv64/src/sbi/smallaios_ext.rs` defining `SBI_EXT_SMALLAIOS = 0x09534149`, `SBI_FN_PMP_REQUEST = 0x00`, `SBI_FN_PROBE = 0x01`.
- [ ] 3.2 Implement a `pmp_request(base: u64, length: u64, perms: PmpPerms) -> SbiResult` client that issues the ecall and returns `Success`, `NotSupported`, or `InvalidParam`.
- [ ] 3.3 Wire into the RISC-V boot path: at S-mode entry, request a PMP region matching the kernel's text/rodata, record the result in `BootMeasurementLog`.
- [ ] 3.4 Verify the no-op path: with stock OpenSBI, the call returns `NotSupported` and boot succeeds. Capture serial output proving this.
- [ ] 3.5 Document the OpenSBI patch needed to honor the extension in `docs/riscv-opensbi-pmp.md`. The patch lives in the doc, not in the repo (it's a downstream firmware change).
- [ ] 3.6 Update `docs/boot-security-matrix.md` RISC-V row: PMP column flips from **No** to **Best-effort (Phase 3)** with a link.
- [ ] 3.7 PR title: `feat(arch/riscv64): boot-root-of-trust-v1 phase 3 — best-effort SBI PMP request`. Target `develop`.

## 4. Phase 4 — Signed UEFI kernel image

### 4a. Vendor signing keys

- [ ] 4.1 Generate development Ed25519 + ML-DSA-65 hybrid signing keys, document storage layout in `docs/release-runbook.md` (new section: "Signing ceremony").
- [ ] 4.2 Generate ECDSA-P256 + RSA-2048 UEFI-compatible classical sibling keys for `sbsign` consumption.
- [ ] 4.3 Document HSM-backed production layout (YubiHSM 2 PIV slot allocation) in the runbook.

### 4b. Signing pipeline

- [ ] 4.4 Add a `just sign-release ARTIFACT` recipe that wraps `sbsign --cert <cert> --key <key> --output <artifact>.signed <artifact>`.
- [ ] 4.5 Extend the same recipe to ML-DSA-65 counter-sign the signed artifact, embedding the PQC signature in a `.note.signing` ELF / PE section so verifiers can find it.
- [ ] 4.6 Add `just verify-release-signature ARTIFACT` calling `sbverify --cert <cert>` + the PQC verification side.
- [ ] 4.7 Update CI release pipeline to invoke `just sign-release` on tagged releases.

### 4c. UEFI enrollment docs

- [ ] 4.8 Create `docs/uefi-secure-boot-enrollment.md` covering Microsoft default-key removal, SmallAIOS vendor PK/KEK/db enrollment, signature-verification logging on platforms that expose it.
- [ ] 4.9 Update `unikernel-orin-bringup-v1`'s `docs/jetson-orin-uefi-boot.md` to add a "Enrolling the SmallAIOS vendor key" section as an alternative to "Disable Secure Boot".
- [ ] 4.10 Update `docs/boot-security-matrix.md`: x86-64 + AArch64 "Secure Boot" cells flip from **No** to **Yes (Phase 4)** with a link.

### 4d. Phase 4 close-out

- [ ] 4.11 PR title: `feat(release): boot-root-of-trust-v1 phase 4 — signed UEFI kernel image`. Target `develop`.

## 5. Cross-phase verification

- [ ] 5.1 End-to-end smoke: x86-64 build with `tpm-attest` + Phase 4 signing, boots under QEMU + swtpm, produces a hybrid quote that verifies against the embedded counter-signing pub key. Captured PCRs 11-14 reflect kernel + boot config + (no models loaded in smoke) + capability config.
- [ ] 5.2 AArch64 smoke: Tegra Orin boot (extending `unikernel-orin-bringup-v1`'s capture), TF-A event log read, SmallAIOS chain appended, merged root hash matches expected.
- [ ] 5.3 RISC-V smoke: QEMU `virt` with stock OpenSBI, SBI vendor probe returns `NotSupported`, boot proceeds, measurement log entry records the no-op.
- [ ] 5.4 `openspec validate boot-root-of-trust-v1` returns valid.
