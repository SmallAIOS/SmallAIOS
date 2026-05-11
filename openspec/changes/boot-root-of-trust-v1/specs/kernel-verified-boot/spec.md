## ADDED Requirements

### Requirement: TPM 2.0 driver with TIS and CRB transports

The `smallaios-security` crate SHALL provide a clean-room `#![no_std]` TPM 2.0 driver supporting both TIS (TPM Interface Specification rev. 1.3) and CRB (Command Response Buffer) MMIO transports, gated by a new `tpm-attest` Cargo feature.

#### Scenario: ACPI-driven transport detection

- **GIVEN** an x86-64 system with a TPM 2.0 device described in the `TPM2` ACPI table
- **WHEN** SmallAIOS boots with the `tpm-attest` Cargo feature enabled
- **THEN** the kernel SHALL parse the `TPM2` ACPI table (`StartMethod`, `ControlAreaPhysicalAddress`, `LAML`, `LASA`) to determine whether to use the TIS or CRB transport
- **AND** the driver SHALL successfully issue `TPM2_Startup(Clear)` and `TPM2_GetCapability(TPM_CAP_PCRS)` to verify the SHA-256 PCR bank is present
- **AND** the boot measurement log SHALL record the detected transport (`tis` or `crb`) and the TPM manufacturer / firmware version returned by `TPM2_GetCapability`

#### Scenario: Graceful fallback when no TPM is present

- **GIVEN** an x86-64 system with no `TPM2` ACPI table
- **WHEN** SmallAIOS boots with `tpm-attest` enabled
- **THEN** the driver SHALL return `TpmAbsent` from its probe step
- **AND** `BootMeasurementLog` SHALL continue to operate in software-only mode (matching today's `verified-boot` behavior)
- **AND** subsequent calls to `produce_hybrid_quote()` SHALL return `Err(TpmAbsent)` rather than panicking

#### Scenario: Build without tpm-attest is byte-identical to today

- **GIVEN** a build with `--features verified-boot` but without `--features tpm-attest`
- **THEN** the resulting kernel binary SHALL NOT link any `tpm2` module symbols
- **AND** the `BootMeasurementLog` behavior SHALL be unchanged from the pre-change `verified-boot` behavior
- **AND** the `tpm2-swtpm-smoke` CI job SHALL skip cleanly (advisory failure exemption documented inline)

### Requirement: Hardware-extended boot measurement log

When the `tpm-attest` feature is enabled and a TPM 2.0 device is present, every `BootMeasurementLog::add_entry` call SHALL extend the corresponding PCR using `TPM2_PCR_Extend`, per a documented PCR-to-measurement-category mapping.

#### Scenario: Documented PCR mapping

- **GIVEN** the `tpm-attest` feature is enabled and a TPM is present
- **THEN** the kernel SHALL extend PCR 11 with the SHA-256 of the SHA-3-256 digest of the kernel image (text + rodata sections)
- **AND** the kernel SHALL extend PCR 12 with the SHA-256 of the SHA-3-256 digest of the boot configuration (Multiboot2 info structure, kernel cmdline, or DTB blob)
- **AND** the kernel SHALL extend PCR 13 with the SHA-256 of the SHA-3-256 digest of each loaded ONNX model, in load order
- **AND** the kernel SHALL extend PCR 14 with the SHA-256 of the SHA-3-256 digest of the SmallAIOS configuration (capability config, policy bundle) plus the public key half of the counter-signing key used for hybrid quotes
- **AND** the PCR mapping SHALL be documented in `docs/attestation-quote-format.md`

#### Scenario: SHA-3 to SHA-256 bridge is observable

- **GIVEN** a SmallAIOS-computed SHA-3-256 digest `D3`
- **WHEN** the digest is extended into PCR `N`
- **THEN** the PCR-extend input SHALL be `SHA-256(D3)` (the SHA-256 of the SHA-3-256 digest, not `D3` directly)
- **AND** the bridge step SHALL be logged as a `BridgeDigest` entry in `BootMeasurementLog` so an off-host verifier can reconcile the SmallAIOS-side and TPM-side digests

#### Scenario: PCR-extend failure is fail-loud

- **GIVEN** a TPM that returns an error from `TPM2_PCR_Extend` mid-boot
- **WHEN** SmallAIOS attempts to extend a measurement
- **THEN** the kernel SHALL log the error to the boot measurement log with a `TpmExtendFailed { pcr, tpm_rc }` entry
- **AND** the kernel SHALL NOT silently proceed — the entry SHALL be marked as "unsealed" (no hardware anchor) and `produce_hybrid_quote` SHALL refuse to issue a quote covering any unsealed entry unless the caller passes an explicit `allow_unsealed: true` flag (default false)

### Requirement: Hybrid PQC-friendly attestation quote format

The `smallaios-security` crate SHALL produce a hybrid attestation quote (`HybridQuote`) combining a hardware TPM 2.0 quote with an ML-DSA-65 + Ed25519 software counter-signature, encoded in CBOR.

#### Scenario: Hybrid quote structure

- **GIVEN** the `tpm-attest` feature is enabled, a TPM is present, and a verifier-supplied nonce
- **WHEN** the kernel produces a hybrid quote via `produce_hybrid_quote(nonce)`
- **THEN** the returned `HybridQuote` SHALL be a CBOR map containing:
  - key 1: raw `TPM2_Quote` output (`TPMS_ATTEST` + `TPMT_SIGNATURE` from `TPM2_Sign`-style signature)
  - key 2: the full `BootMeasurementLog` entries (CBOR array)
  - key 3: an ML-DSA-65 + Ed25519 hybrid signature over the concatenation of (1) || (2) || nonce || timestamp
  - key 4: SHA-3-256 of the public counter-signing key used in (3)
  - key 5: the verifier-supplied nonce
  - key 6: a SmallAIOS monotonic timestamp at quote generation time
- **AND** the wire format SHALL be documented in `docs/attestation-quote-format.md`
- **AND** the counter-signing key's public half SHALL be measured into PCR 14 at boot, so the value at key 4 is verifiable against the PCR chain

#### Scenario: Verifier can require either signature

- **GIVEN** a `HybridQuote` and a verifier configured for classical-only checking
- **WHEN** the verifier calls `verify_hybrid_quote(quote, ak_pub, None, expected_root)`
- **THEN** verification SHALL succeed if the TPM-side signature (key 1) validates against `ak_pub` and the measurement log replays to `expected_root`
- **AND** the ML-DSA-65 counter-signature SHALL be ignored in classical-only mode

- **GIVEN** the same quote and a verifier configured for PQC-only checking
- **WHEN** the verifier calls `verify_hybrid_quote(quote, None, counter_pub, expected_root)`
- **THEN** verification SHALL succeed if the counter-signature validates against `counter_pub` AND `counter_pub`'s SHA-3-256 digest appears in PCR 14 of the measurement log
- **AND** the TPM-side signature SHALL be ignored in PQC-only mode

- **GIVEN** the same quote and a verifier configured for hybrid (both) checking
- **WHEN** the verifier calls `verify_hybrid_quote(quote, ak_pub, counter_pub, expected_root)`
- **THEN** both signatures SHALL be required to validate; if either fails, verification SHALL fail

### Requirement: TF-A measurement chain consumption (AArch64)

On AArch64 platforms where Arm Trusted Firmware (TF-A) BL31 has emitted a TCG-format event log (via `MEASURED_BOOT=1` in the TF-A build), SmallAIOS SHALL read and merge that event log into its own `BootMeasurementLog` as prefix entries.

#### Scenario: DTB-described event log is consumed

- **GIVEN** a Tegra Orin (or other AArch64) boot where the DTB exposes `/reserved-memory/tf-a-event-log { reg = <...>; };`
- **WHEN** SmallAIOS boots
- **THEN** the kernel SHALL parse the reserved-memory region as a TCG_PCClientPCREvent v2 event log
- **AND** the kernel SHALL merge the TF-A entries (BL1, BL2, BL31) as prefix entries in `BootMeasurementLog`, ahead of SmallAIOS's own entries
- **AND** a `merged_root_hash()` API SHALL return the SHA-256 replay over the merged log

#### Scenario: Missing or malformed event log is non-fatal

- **GIVEN** a boot where the `/reserved-memory/tf-a-event-log` node is absent or contains malformed data
- **WHEN** SmallAIOS attempts to read it
- **THEN** the kernel SHALL log a `TfAEventLogAbsent` or `TfAEventLogMalformed` entry and continue with SmallAIOS-only measurements
- **AND** the boot SHALL NOT fail solely because TF-A did not emit a log (e.g. on a TF-A build without `MEASURED_BOOT=1`)

#### Scenario: DTB parser handles Tegra234 reserved-memory layout

- **GIVEN** a Tegra234 DTB with multi-cell `reg = <hi lo hi lo>` and `/reserved-memory/` subnodes
- **WHEN** `kernel::mem::phys::parse_dtb` parses the blob
- **THEN** the parser SHALL correctly resolve 64-bit addresses split across two 32-bit cells
- **AND** the parser SHALL expose `/reserved-memory/` subnodes by name to callers
- **AND** the parser SHALL satisfy the gap called out in `unikernel-orin-bringup-v1` task 2.12

### Requirement: Best-effort RISC-V PMP request via SBI vendor extension

On RISC-V, SmallAIOS SHALL issue an SBI vendor-extension call requesting that M-mode firmware install a PMP region matching the kernel's desired memory layout, and SHALL record the firmware's response in the boot measurement log. The boot SHALL succeed regardless of whether the request is honored.

#### Scenario: Vendor extension identifiers are documented

- **THEN** SmallAIOS SHALL use vendor extension ID `0x09534149` (ASCII `\x09SAI`) in the RISC-V SBI vendor-extension space (`0x09000000`-`0x09FFFFFF`)
- **AND** function ID `0x00` SHALL be `SBI_EXT_SMALLAIOS_PMP_REQUEST(base, length, permissions)`
- **AND** function ID `0x01` SHALL be `SBI_EXT_SMALLAIOS_PROBE()` returning the extension version or `NotSupported`
- **AND** the identifiers SHALL be documented in `docs/riscv-opensbi-pmp.md`

#### Scenario: Stock OpenSBI returns NotSupported, boot continues

- **GIVEN** a RISC-V boot under unpatched OpenSBI (which does not implement the SmallAIOS vendor extension)
- **WHEN** SmallAIOS issues `SBI_EXT_SMALLAIOS_PROBE`
- **THEN** the call SHALL return `NotSupported`
- **AND** SmallAIOS SHALL record a `SbiPmpRequestSkipped { reason: NotSupported }` entry in `BootMeasurementLog`
- **AND** boot SHALL proceed normally

#### Scenario: Patched OpenSBI honors the request

- **GIVEN** a RISC-V boot under a SmallAIOS-patched OpenSBI build (per `docs/riscv-opensbi-pmp.md`)
- **WHEN** SmallAIOS issues `SBI_EXT_SMALLAIOS_PMP_REQUEST(kernel_text_base, kernel_text_len, RX)`
- **THEN** the call SHALL return `Success`
- **AND** SmallAIOS SHALL record a `SbiPmpRequestHonored { base, length, perms }` entry in `BootMeasurementLog`
- **AND** the requested PMP region SHALL be installed in the firmware's PMP CSRs (verifiable by reading the relevant CSRs from M-mode debug output)

### Requirement: Signed UEFI kernel image with PQC counter-signature

The release build pipeline SHALL produce a UEFI-signed kernel artifact for both x86-64 (`smallaios-x86_64.efi`) and AArch64 (`smallaios.efi`), with both a classical UEFI Secure Boot signature (ECDSA-P256 or RSA-2048 via `sbsign`) and an ML-DSA-65 PQC counter-signature embedded in a documented binary section.

#### Scenario: Classical signature is verifiable by sbverify

- **GIVEN** a release-pipeline build of `smallaios.efi` invoked via `just sign-release smallaios.efi`
- **WHEN** an off-host verifier runs `sbverify --cert <SmallAIOS-vendor.crt> smallaios.efi`
- **THEN** `sbverify` SHALL report `Signature verification OK`
- **AND** the firmware on a UEFI Secure Boot-enabled system with the SmallAIOS vendor key enrolled in `db` SHALL load the artifact without prompting
- **AND** on Tegra Orin specifically, the `unikernel-orin-bringup-v1` "disable Secure Boot" workflow SHALL have a documented "enroll the SmallAIOS vendor key" alternative in `docs/jetson-orin-uefi-boot.md`

#### Scenario: PQC counter-signature is verifiable by SmallAIOS tooling

- **GIVEN** the same signed artifact
- **WHEN** an off-host verifier runs `just verify-release-signature smallaios.efi`
- **THEN** the recipe SHALL extract the ML-DSA-65 signature from the `.note.signing` PE section
- **AND** the recipe SHALL verify the signature against the SmallAIOS PQC vendor public key
- **AND** the verification SHALL succeed for any binary signed by the canonical release pipeline
- **AND** any tampered byte in the artifact (other than the signature section itself) SHALL cause verification to fail

#### Scenario: Development builds carry a clear non-release marker

- **GIVEN** a non-release build signed with a development key (not the release HSM-backed key)
- **THEN** the `.note.signing` section SHALL contain a `DEV — DO NOT USE FOR RELEASE` ASCII string preceding the signature bytes
- **AND** `just verify-release-signature` SHALL print a prominent warning and exit with code 2 (verified-but-dev), not code 0 (verified-release)
