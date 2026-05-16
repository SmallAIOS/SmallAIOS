# boot-root-of-trust-v1

## Summary

`docs/boot-security-matrix.md` documents that SmallAIOS today has a **software-only** integrity story: the `verified-boot` Cargo feature (defined in `security/Cargo.toml`, re-exported by `kernel/Cargo.toml` and `onnx-rt/Cargo.toml`) computes SHA-3-256 over the kernel image, the Multiboot2 / DTB info, and ONNX model files, and records them in an in-memory `BootMeasurementLog` (`kernel/src/boot_integrity.rs`). That log is **not anchored to any hardware root of trust** on any of the three target architectures (x86-64, AArch64, RISC-V), and SmallAIOS itself is **not** loaded by a signature-verified firmware chain: x86-64 boot uses Multiboot2 direct-load with no UEFI Secure Boot signature, AArch64 boot uses U-Boot `booti` without FIT signature verification, and RISC-V boot trusts OpenSBI implicitly with no PMP request from the kernel side.

This change closes the boot-side gap in **four sequenced phases**, each independently valuable, each with a serial-console or quote-output observable success criterion. Phase 1 introduces a clean-room `#![no_std]` TPM 2.0 driver (TIS / CRB transport) on x86-64 and extends the existing `BootMeasurementLog` to drive **hardware** PCR extends — turning the software measurement log into the canonical software record of what was also extended into the TPM, plus a `tpm2_quote` API that signs a PCR digest with the TPM's AK. Phase 2 brings AArch64 onto an equivalent footing: SmallAIOS becomes aware of the TF-A BL31 hand-off, reads the optional TCG-format event log left by BL1/BL2 in DRAM (TPM Event Log structure, `TCG_PCClientPCRSelection`), and reconciles its own measurements against the firmware chain. Phase 3 requests PMP regions via an OpenSBI-side SBI vendor extension for RISC-V (vendor-specific, best-effort), giving M-mode a kernel-asserted memory layout to enforce. Phase 4 signs the kernel binary (`smallaios.efi` for AArch64 UEFI; `smallaios-x86_64.efi` for x86-64 UEFI-loadable builds) with a vendor key and produces a documented enrollment procedure for both UEFI Secure Boot (PK/KEK/db) and platform-fused boot guards (Tegra fuse-burn, Intel Boot Guard — documented, not automated, since they're OEM-provisioning operations).

Capability: `kernel-verified-boot` (new). The existing `verified-boot` Cargo feature stays — this change extends it from "software self-hash" to "hardware-anchored measured boot + signed image". A new `tpm-attest` Cargo feature on `smallaios-security` gates the TPM driver; a `verified-boot` build that lacks `tpm-attest` keeps today's software-only behavior so the QEMU / dev-loop path doesn't regress.

## Why

- **`verified-boot` is software-only and easily defeated.** A `BootMeasurementLog` that lives in DRAM, computed by the kernel that's measuring itself, is a TOCTOU joke without a hardware anchor. An attacker who can patch the kernel binary on disk patches the measurement code at the same time. The measurement log is useful as a *post-hoc audit record*, but it cannot answer "is this kernel I'm talking to the one I signed?" without TPM PCRs (x86-64), TF-A event log + signed image (AArch64), or PMP-enforced layout (RISC-V). The boot-security-matrix doc says exactly this: every "SmallAIOS Integration" cell under "Hardware Root of Trust" reads **No**.
- **DO-178C DAL A coverage requires evidence the running binary is the certified binary.** The certification claim "the kernel binary executing in the field is bit-identical to the kernel binary tested against the requirements" needs a verifiable artifact, not a developer's assertion. A TPM 2.0 quote signed by an attestation key (AK) certified by an endorsement key (EK) gives an auditor a cryptographic chain back to a TPM manufacturer's root certificate. The same evidence shape (a quote over a measurement) is what the `remote-attestation-v1` change consumes — Phase 1 here is the prerequisite for that change's x86-64 backend.
- **PQC-default stance demands the quote be hybrid-signable.** SmallAIOS already has ML-DSA-65 + Ed25519 hybrid signatures in `security/src/crypto/`. TPM 2.0 hardware signs with RSA-2048/3072 or ECDSA-P256 (no PQC support in current TPM generations). The Phase 1 quote MUST therefore be a *hybrid*: the TPM signs the raw quote with its hardware AK (classical), and SmallAIOS counter-signs the same quote-plus-context with an ML-DSA-65 software key whose public half is itself measured into a PCR. Downstream verifiers get both signatures and can require either or both — supporting today's TPM hardware while not blocking a future PQC-only deployment.
- **Phase 1 is independently valuable.** Even without Phases 2-4, a TPM-extended `BootMeasurementLog` + a working `tpm2_quote` on x86-64 lets us claim hardware-anchored measured boot on the most common deployment target (x86-64 datacenter / cloud / desktop). Phase 1 is the deliverable that unblocks `remote-attestation-v1` Phase 1 and gives DAL A its first piece of cryptographic boot-integrity evidence.
- **Tegra Orin already provides the firmware half of what Phase 2 needs.** TF-A BL31 ships with JetPack 6, runs at EL3, and emits an optional TCG event log if `MEASURED_BOOT=1` is set in the TF-A build. Existing NVIDIA documentation (`docs/orin-secure-boot-flow.md` in the L4T 36.4 release notes) confirms TF-A's measured-boot mode is enabled in the NVIDIA-shipped firmware. SmallAIOS's job in Phase 2 is therefore *consumption*, not driver work: read the event log from the documented memory location, parse it against the TCG spec, and reconcile against our own measurements. No new EL3-side code is required.
- **RISC-V is the least mature platform and the matrix calls it out explicitly.** OpenSBI doesn't define a standard "request PMP region" extension, so Phase 3 is documented as best-effort and vendor-scoped (SiFive Freedom U, T-Head TH1520) with a graceful no-op fallback. The value is small but the fallback shape (best-effort SBI ecall, ignore on failure) is reusable for future SBI extensions and keeps the matrix's bottom row honest.

## Phase 1 — TPM 2.0 driver + hardware-extended measurement log (x86-64, ~2 weeks)

A clean-room `#![no_std]` TPM 2.0 driver lives in `security/src/tpm2/` (new module). Two transports: TIS (TPM Interface Specification, MMIO at `0xFED40000`+) for legacy / discrete TPMs and CRB (Command Response Buffer, also MMIO) for modern fTPM / dTPM 2.0 in firmware-Tier mode. Detection by reading the `TPM2` ACPI table from the RSDP chain (existing `kernel/src/acpi/` shape extended). The driver implements the minimum command set Phase 1 needs: `TPM2_Startup`, `TPM2_GetCapability` (probe for PCR bank layout — SHA-256 mandatory), `TPM2_PCR_Extend` (extend each measurement event), `TPM2_PCR_Read` (read final PCR state), `TPM2_CreatePrimary` + `TPM2_LoadExternal` for a transient AK if no AK is provisioned, `TPM2_Quote` (sign a PCR digest with the AK).

The `BootMeasurementLog::add_entry()` path (currently a pure software append) gains a `cfg(feature = "tpm-attest")` extension: when a TPM is present, every entry also drives `TPM2_PCR_Extend` against a documented PCR mapping (Phase 1 uses PCR 11-14 to follow the convention used by Linux IMA without colliding with PCR 0-7 / 8-10 owned by firmware and shim/grub):

| PCR | Content | Measured by |
|-----|---------|-------------|
| 11 | Kernel binary (SHA-3-256 of `smallaios.efi` text + rodata) | SmallAIOS at boot |
| 12 | Boot config (Multiboot2 / DTB info, kernel cmdline) | SmallAIOS at boot |
| 13 | ONNX model binaries (concatenated SHA-3-256 of each `.onnx` measured at load) | SmallAIOS on model load |
| 14 | SmallAIOS configuration (capability config, policy bundle) | SmallAIOS after `init` |

Phase 1 also defines the **hybrid quote format**: a CBOR document containing the raw TPM2_Quote output (TPM-AK-signed), the SmallAIOS software measurement log entries that fed the PCRs, and an ML-DSA-65 + Ed25519 hybrid signature over the whole bundle by a key whose public half is itself measured into PCR 14. The wire shape is documented in `docs/attestation-quote-format.md` (new) so `remote-attestation-v1` can consume it without re-design.

CI gains a `tpm2-swtpm-smoke` job running the kernel under QEMU with `swtpm` providing a software TPM 2.0 socket, asserting the PCR-extend path executes end-to-end and a quote verifies via the openssl `tss2-engine` (advisory until a self-hosted TPM-equipped runner exists).

## Phase 2 — TF-A BL31 measurement chain consumption (AArch64, ~2 weeks)

SmallAIOS's AArch64 boot path (currently `arch/aarch64/src/boot.rs` for the bare-metal path and `arch/aarch64/src/boot_uefi.rs` from `unikernel-orin-bringup-v1` for the UEFI path) gains a TF-A event-log reader. Per Arm DEN 0028 (SMC Calling Convention) and TBBR specs, BL31 either places the event log in a documented DRAM region (described by a TF-A-installed `EFI_TCG2_FINAL_EVENTS_TABLE`-equivalent configuration entry) or exposes it via an SMC vendor call. SmallAIOS reads it, validates the TCG event log structure, and merges entries 0-N (firmware) into its own `BootMeasurementLog` as prefix entries — so the final log covers the whole chain (BL1 → BL2 → BL31 → SmallAIOS → models) with a single audit shape.

On Jetson Orin specifically, the JetPack 6.2.1 TF-A build emits the event log at the location described in the device tree under `/reserved-memory/tf-a-event-log`. Phase 2's DTB parser (extending `kernel/src/mem/phys::parse_dtb` — note the existing Tegra234 DTB parsing gap called out in `unikernel-orin-bringup-v1` task 2.12) reads that reserved region and exposes it.

If a TPM is not present (the common AArch64 server case — most ARM SoCs don't ship a discrete TPM and the fTPM-in-OP-TEE path is the future story), the measurement chain is **software-only but multi-stage**: TF-A's measurements + SmallAIOS's measurements, all signed at the SmallAIOS layer with ML-DSA-65 hybrid. When OP-TEE is present (the `op-tee-bridge-v1` follow-up change), the same chain can be sealed inside the TEE — that's covered in the dependent change.

## Phase 3 — RISC-V PMP via SBI vendor extension (best-effort, ~1 week)

The OpenSBI SBI spec defines vendor-extension space (`0x09000000 - 0x09FFFFFF`). Phase 3 specifies a SmallAIOS-defined vendor function `SBI_EXT_SMALLAIOS_PMP_REQUEST` that takes (base, length, permissions) and asks the M-mode firmware to install a PMP region matching the kernel's desired layout. Whether the firmware honors the request is implementation-defined — on stock OpenSBI it's a no-op (which is fine), on a SmallAIOS-friendly OpenSBI fork (provided in `docs/riscv-opensbi-pmp.md` as a documented patch series) it installs the requested region. Either way, SmallAIOS records the request and the firmware's response into the measurement log so an auditor sees whether PMP was actually configured.

Phase 3 explicitly does **not** require a custom OpenSBI build to work — the no-op fallback is intentional. Boot succeeds whether the extension is honored or not; only the measurement log reflects the difference.

## Phase 4 — Signed kernel image (UEFI Secure Boot enrollment, ~1 week)

Phase 4 produces a UEFI-signed kernel artifact for both x86-64 and AArch64 UEFI builds. The vendor signing key is an Ed25519 / ML-DSA-65 hybrid key managed in `docs/release-runbook.md` (extended in this change) — release artifacts are signed by both halves. The x86-64 build produces `smallaios-x86_64.efi` signed with `sbsign`; the AArch64 build extends the existing `unikernel-orin-bringup-v1` Phase 2 UEFI image to a signed `smallaios.efi`. Documentation covers UEFI Secure Boot key enrollment (`efi-updatevar` / firmware setup), and references — but does not automate — the platform-fused secure boot procedures (Intel Boot Guard fuse provisioning, Tegra secure boot fuse-burn) as those are OEM operations that vary per device.

## Out of scope

- **Intel Boot Guard fuse provisioning**: documented as OEM-coordinated, not automated. Same for Tegra secure boot fuses.
- **OP-TEE-hosted fTPM**: a natural future story but blocked on `op-tee-bridge-v1`. Phase 2 reads TF-A event logs without OP-TEE; fTPM-backed PCRs land in a later change.
- **Boot integrity via DRTM (Intel TXT / AMD SKINIT)**: skipped — DRTM requires SINIT-ACM provisioning and a different threat model. Out of scope here.
- **Confidential compute integration**: Phase 1-4 don't depend on or block `confidential-compute-v1`. Memory encryption at boot is a separate concern.
- **Live PCR re-extends post-boot**: PCRs 11-14 are extended at boot and on model-load (PCR 13). Continuous runtime PCR extension (e.g. on every cap-flip) is out of scope; we use signed audit records (`security/src/audit/`) for that.

## Sequencing

Phase 1 (x86-64 TPM) first, independently mergeable. Phase 2 (AArch64 TF-A event log) and Phase 3 (RISC-V PMP request) can land in parallel after Phase 1 — they share no code with Phase 1 and only consume Phase 1's measurement-log shape. Phase 4 (signed image) lands last because it's a release-engineering surface change and benefits from Phases 1-3 being in for the signature-verification chain to anchor against. If schedule pressure forces a split, Phase 1 alone is a complete deliverable; Phase 2 alone is also a complete deliverable for AArch64 deployments without TPMs.

## Effort estimate

| Phase | Scope | Estimate |
|-------|-------|----------|
| 1 | TPM 2.0 driver (TIS + CRB), PCR extend wiring, `TPM2_Quote`, hybrid CBOR quote, swtpm CI | ~2 weeks |
| 2 | TF-A event log reader, DTB reserved-memory parse, chain reconciliation | ~2 weeks |
| 3 | SBI vendor extension, no-op fallback, documented OpenSBI patch | ~1 week |
| 4 | Vendor signing key in release-runbook, sbsign / sbsigntools wiring, UEFI enrollment docs | ~1 week |
| **Total** | | **~6-8 weeks** |

## DO-178C alignment

| Phase | Certification claim it supports |
|-------|--------------------------------|
| 1 | "The x86-64 kernel binary in the field is bit-identical to the certified binary." Evidence: TPM2 quote over PCR 11 verified against the certified-build hash artifact in the release record. |
| 2 | "The AArch64 boot chain (TF-A through SmallAIOS) is unmodified relative to certification." Evidence: TCG event log replay against the certified release-record event-log snapshot. |
| 3 | "The RISC-V kernel runs with the certified memory layout." Evidence: SBI vendor-extension request log + (when honored) firmware-side PMP state. |
| 4 | "The kernel artifact loaded by firmware was signed by SmallAIOS Engineering." Evidence: UEFI Secure Boot signature verification record in firmware logs (where firmware exposes them) + ML-DSA-65 counter-signature audit record. |

Together, Phases 1-4 close the "boot integrity" objective row in the DAL A traceability matrix maintained in `openspec/changes/archive/2026-02-27-boot-security-comparison-v10/` and referenced by `docs/boot-security-matrix.md`.
