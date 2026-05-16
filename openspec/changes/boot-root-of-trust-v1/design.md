# Design — boot-root-of-trust-v1

## Goal

Four sequenced milestones, each with an observable boot-time or off-host verification artifact:

1. **Phase 1 — x86-64 TPM 2.0 hardware-extended measurement log.** Success = `swtpm`-emulated PCRs 11-14 match the SmallAIOS in-DRAM `BootMeasurementLog` after boot, and a `TPM2_Quote` over those PCRs verifies against the AK public key.
2. **Phase 2 — AArch64 TF-A event log consumption.** Success = SmallAIOS's `BootMeasurementLog` begins with TF-A-supplied entries (BL1/BL2/BL31 hashes) ahead of SmallAIOS's own entries, and the merged log replays into a consistent root hash.
3. **Phase 3 — RISC-V PMP request SBI extension.** Success = SmallAIOS issues `SBI_EXT_SMALLAIOS_PMP_REQUEST` and records the response (`Success` on patched OpenSBI, `NotSupported` on stock); boot succeeds either way.
4. **Phase 4 — Signed UEFI kernel image.** Success = `sbverify --cert <vendor.crt> smallaios-x86_64.efi` returns OK; same for `smallaios.efi` on AArch64; firmware logs (where exposed) record signature-verified load.

## Phase 1 — TPM 2.0 driver

### Why a clean-room driver, not `tpm2-tss`

The well-known `tpm2-tss` C library is GPL/BSD, depends on POSIX I/O and dynamic memory, and pulls in OpenSSL. SmallAIOS is `#![no_std]`, single-binary, no dynamic allocation in the hot path, and uses its own crypto (`security/src/crypto/`). A clean-room driver also keeps `cargo-vet` happy — no `external_audit` line item, no supply-chain review pending an upstream maintainer — and aligns with the clean-room ONNX runtime and clean-room PQC stacks that are already in the workspace.

The downside is implementation surface: the TPM 2.0 command set is large (~120 commands per the spec). Phase 1 implements only the seven commands listed in the proposal. Future phases (sealing for `op-tee-bridge-v1`, NV ram for credential storage) add more.

### Transport detection

TPM 2.0 hardware exposes one of two transports:

- **TIS** (TPM Interface Specification rev. 1.3): MMIO at `0xFED40000`-`0xFED44FFF`. Used by legacy / discrete TPMs and most firmware-emulated TPMs on Windows-target platforms.
- **CRB** (Command Response Buffer): MMIO at a base discovered via ACPI. Used by modern fTPMs (AMD fTPM, Intel PTT) and newer dTPMs.

The `TPM2` ACPI table (TCG-defined, type ID `TPM2`) discloses both: the table's `StartMethod` field indicates which transport, and the `ControlAreaPhysicalAddress` field gives the CRB base when applicable. The existing `kernel/src/acpi/` ACPI table walker is extended with a TPM2 parser; no new RSDP / XSDT code.

If no `TPM2` ACPI table is present, the driver returns `TpmAbsent` and Phase 1 falls back gracefully: `BootMeasurementLog` continues to operate in software-only mode (today's behavior) and `tpm2_quote` returns `Err(TpmAbsent)`. CI tests both paths.

### PCR bank selection

TPM 2.0 supports multiple PCR banks (SHA-1, SHA-256, SHA-384, SHA-512, SM3). Phase 1 uses **SHA-256** because:

- SHA-256 is mandatory per the TCG PC Client spec (every TPM 2.0 supports it).
- SHA-1 is deprecated (collision attacks practical since 2017).
- SHA-3-256 (which SmallAIOS uses internally for its own measurements) is **not** a standard TPM PCR bank; we hash inputs to SHA-3-256 in software, then re-hash the SHA-3-256 digest into a SHA-256 PCR for hardware extension. The hybrid bridges between SmallAIOS's PQC-default choice (SHA-3) and the TPM's hardware-fixed choice (SHA-256).

A `#[cfg(feature = "tpm-sha384-bank")]` opt-in for SHA-384 PCRs is documented for high-security deployments, but Phase 1 ships SHA-256 only.

### Hybrid quote format

The PQC-default constraint means the TPM-signed quote alone is insufficient — a future PQC-only deployment can't accept an RSA-2048 TPM signature as the sole proof of integrity. The hybrid quote format addresses this:

```
HybridQuote := CBOR-Map {
  1: TpmQuote,           // raw TPM2_Quote output (TPMS_ATTEST + TPMT_SIGNATURE)
  2: MeasurementLog,     // SmallAIOS BootMeasurementLog entries (CBOR array)
  3: HybridSignature,    // ML-DSA-65 + Ed25519 over (1 || 2 || nonce || timestamp)
  4: HybridPubKeyDigest, // SHA-3-256 of the public counter-signing key (also extended into PCR 14)
  5: Nonce,              // verifier-supplied freshness nonce
  6: Timestamp,          // SmallAIOS monotonic timestamp at quote time
}
```

A verifier can require:

- Classical only (today's TPM ecosystem): verify (1) against the AK pub.
- PQC only (future): verify (3) against the counter-signing pub, check (4) was extended into PCR 11-14 chain.
- Both: belt-and-braces, the recommended default.

The CBOR encoding is `security/src/cbor/` (already exists for audit log records). No new wire-format crate.

### Alternatives considered

- **`tpm2-tss` as a vendored static C dependency.** Rejected for `no_std` incompatibility, OpenSSL dep, supply-chain surface. Also: clean-room is a stated DAL A advantage; bringing in C would require certification of the C build environment.
- **Skip the hybrid quote, ship classical-only.** Rejected — contradicts the PQC-default project stance and forces a wire-format break later. The hybrid format degrades gracefully (verifier can check one signature or both).
- **PCR 0-7 instead of 11-14.** Rejected — PCR 0-7 are owned by firmware/bootloader by TCG convention. Extending them from SmallAIOS would clobber firmware measurements and confuse any out-of-band verifier (e.g. an `tpm2_pcrread` from a recovery shell). PCR 11-14 mirror the IMA / LinuxKit convention, which keeps SmallAIOS audit tools interoperable with the wider ecosystem.
- **Use TPM `NV` storage for the kernel hash.** Out of scope for Phase 1 — NV storage has its own ownership / locking story and isn't needed for the measurement-log use case. Considered for `op-tee-bridge-v1` for sealing.

## Phase 2 — TF-A event log

### Why consumption-only, not driver

TF-A BL1/BL2 measurements are produced by the firmware vendor (NVIDIA on Tegra Orin, Arm reference firmware on Neoverse boards, etc.). SmallAIOS has no ability — and no need — to *produce* TF-A-side measurements. Our job is to *consume* the event log TF-A leaves behind, validate its TCG-spec format, and merge it as a prefix into our own log.

### Event log location

Two TF-A conventions:

- **Reserved DRAM region described in DTB**: `/reserved-memory/tf-a-event-log { reg = <0x... 0x...>; };`. Used by NVIDIA Tegra234 BSP and Arm reference platforms.
- **SMC vendor call**: TF-A exposes an SMC handler returning a pointer to the log. Less common, used by some Marvell platforms.

Phase 2 supports the DTB path first (matches Tegra Orin, which is SmallAIOS's primary AArch64 target). The SMC path is a follow-up. Detection is straightforward: walk the DTB, look for `/reserved-memory/tf-a-event-log`; if absent, log "no TF-A event log" and continue with SmallAIOS-only measurements.

### DTB parser dependency

`unikernel-orin-bringup-v1` task 2.12 calls out a real DTB-parser gap on Tegra234: `kernel::mem::phys::parse_dtb` doesn't recognize Tegra234's memory-node layout. Phase 2 must fix that gap — extending the parser to handle multi-cell `reg = <…>` values and `/reserved-memory/` subnodes. The Phase 2 deliverable includes this DTB-parser improvement as a sub-task.

### Reconciliation

After parsing TF-A's event log entries and SmallAIOS's measurement log, the kernel computes a merged-replay root hash (SHA-256, matching the TPM PCR algorithm even though no hardware PCR is present on most AArch64 boards). The merged root is the value SmallAIOS quotes when it later signs the quote with its ML-DSA-65 hybrid key. If the AArch64 board *does* have a TPM (rare — usually only on AArch64 servers), the same merged-replay can extend a hardware PCR; that path reuses the Phase 1 driver.

## Phase 3 — RISC-V SBI vendor extension

### Calling convention

Per the RISC-V SBI specification (v2.0+), vendor extensions use extension IDs in `0x09000000 - 0x09FFFFFF`. SmallAIOS picks `0x09534149` (ASCII `\x09SAI` for SmallAIOS) as its vendor ID. Function IDs:

- `0x00 SBI_EXT_SMALLAIOS_PMP_REQUEST(base, length, permissions)` — request a PMP region. Returns `Success`, `NotSupported`, `InvalidParam`.
- `0x01 SBI_EXT_SMALLAIOS_PROBE()` — probe for extension presence. Returns extension version or `NotSupported`.

### Why best-effort

Vendor SBI extensions are not portable. SiFive Freedom U boots stock OpenSBI; T-Head TH1520 boots a SiFive-derived OpenSBI; future RISC-V SoCs ship vendor-customized OpenSBI. SmallAIOS can't require a specific OpenSBI build because it would fragment the deployment matrix.

Phase 3's value is therefore the **request** itself, recorded in the measurement log: an auditor can see SmallAIOS asked for PMP region `X-Y` with permissions `Z`, and whether OpenSBI honored it. If a fleet operator wants the request honored, they deploy SmallAIOS-patched OpenSBI (the patch is documented but not part of the SmallAIOS repo).

### Alternative: skip RISC-V entirely

Considered. Rejected because (a) RISC-V is a documented SmallAIOS target, (b) the boot-security-matrix doc calls out RISC-V's gap explicitly so a no-op pretending to be a feature would be worse than nothing, (c) a documented best-effort with audit-log capture is cheap (~1 week) and keeps the matrix's bottom row honest. The graceful-fallback shape (best-effort SBI call, ignore-on-failure, log either way) becomes a reusable pattern for any future SBI extensions.

## Phase 4 — Signed kernel image

### Key management

The release vendor key lives in two halves:

- Classical: Ed25519 (existing `security/src/crypto/ed25519.rs`). UEFI Secure Boot accepts ECDSA-P256 and RSA-2048; we add an ECDSA-P256 sibling under the same release flow because UEFI doesn't yet accept Ed25519 in `sbsign`.
- PQC: ML-DSA-65 (existing `security/src/crypto/ml_dsa.rs`). No firmware accepts ML-DSA today, so the counter-signature is recorded in the SmallAIOS audit log and verifiable off-host, not by firmware. When firmware-side PQC arrives, the same key promotes.

Key storage: `docs/release-runbook.md` is extended with a signing-ceremony section. The release-engineering surface is HSM-backed in production (the runbook covers the YubiHSM 2 layout). Development builds use file-backed keys with a clear "DEV — DO NOT USE FOR RELEASE" warning baked into the signed artifact's `.note.signing` section.

### sbsigntools dependency

`sbsigntools` is the standard UEFI signing toolchain (Debian / Fedora packaged). Phase 4 wraps it in a `just sign-release` recipe that consumes a `release.toml`-listed key path. Verification on the verifier side uses `sbverify`. No new Rust dependencies are required — signing is a build-system step, not a runtime concern.

### UEFI enrollment

`docs/uefi-secure-boot-enrollment.md` (new) walks through:

1. Disabling Microsoft's default keys (PK/KEK) on production hardware.
2. Enrolling the SmallAIOS vendor PK / KEK / db using `efi-updatevar` or platform firmware setup UI.
3. Verifying the firmware load logs (where exposed via `efivar` or platform-specific tooling) show signature verification.

For Tegra Orin specifically, the Phase 2 (`unikernel-orin-bringup-v1`) "disable Secure Boot in firmware menu" instruction is replaced by an "enroll the SmallAIOS vendor key" instruction once Phase 4 lands. The old instruction stays as a fallback.

## Build / CI surface

### Phase 1

- New module: `security/src/tpm2/` (TIS + CRB transports, command serialization, response parsing).
- New module: `security/src/tpm2_quote.rs` (CBOR hybrid quote format).
- New module: `kernel/src/acpi/tpm2_table.rs` (TPM2 ACPI table parser).
- New cargo feature: `tpm-attest` on `smallaios-security`.
- Extended: `kernel/src/boot_integrity.rs` — `BootMeasurementLog::add_entry()` extends a hardware PCR when `tpm-attest` is enabled.
- New CI job: `tpm2-swtpm-smoke` — boots kernel under QEMU + swtpm, asserts PCR-extend path, verifies a quote.
- New docs: `docs/attestation-quote-format.md`, `docs/tpm2-driver.md`.

### Phase 2

- Extended: `kernel/src/mem/phys::parse_dtb` — multi-cell `reg`, `/reserved-memory/` subnodes (closes the gap from `unikernel-orin-bringup-v1` task 2.12).
- New module: `arch/aarch64/src/tf_a_event_log.rs` (TCG event log parser).
- Extended: `kernel/src/boot_integrity.rs` — chain reconciliation API.
- New CI job: `tf-a-event-log-replay` — synthetic event log fixture, asserts replay against expected root hash.
- New docs: `docs/aarch64-measured-boot.md`.

### Phase 3

- New module: `arch/riscv64/src/sbi/smallaios_ext.rs` (vendor extension client).
- Extended: `kernel/src/boot_integrity.rs` — records request + response.
- New CI job: none (best-effort, no observable change on stock OpenSBI).
- New docs: `docs/riscv-opensbi-pmp.md` (documented OpenSBI patch, not part of repo).

### Phase 4

- New `just` recipe: `sign-release`.
- New release-runbook section: "Signing ceremony".
- New docs: `docs/uefi-secure-boot-enrollment.md`.
- No new Rust modules — signing is post-build.

## What this change does NOT do

- Does not add any *new* attestation protocol surface. `remote-attestation-v1` consumes the Phase 1 hybrid quote format and adds the network-side protocol; this change defines the format only.
- Does not enable OP-TEE-side fTPM. `op-tee-bridge-v1` is a parallel change; SmallAIOS will read from an OP-TEE-hosted fTPM through that bridge in a future change.
- Does not change the existing `verified-boot` software-only path. Builds without `tpm-attest` behave exactly as today; this is purely additive.
- Does not change the AArch64 / RISC-V boot ELF format. Phase 4 signs the *UEFI* binary (PE/COFF); bare-metal `aarch64-unknown-none` builds remain unsigned (they're loaded by a trusted U-Boot anyway).
