# Boot Security Comparison Matrix

Cross-platform comparison of boot security capabilities for SmallAIOS target architectures.

## Platform Boot Security Analysis

### x86-64

**Firmware Trust Chain:** UEFI Secure Boot → shim → GRUB2 → kernel

| Capability | Description | SmallAIOS Integration |
|---|---|---|
| UEFI Secure Boot | Firmware verifies signed bootloader using PK/KEK/db key hierarchy. Only signed EFI binaries execute. | **No** — SmallAIOS uses Multiboot2/QEMU `-kernel` direct load. Future: sign as EFI stub. |
| Intel Boot Guard | Hardware-rooted verified boot using ACM (Authenticated Code Module) in CPU microcode. OEM-fused keys. | **No** — Requires OEM key provisioning at manufacture time. Not available in VM/QEMU. |
| TPM Measured Boot | PCR extend chain records boot measurements. Remote attestation via TPM2.0 quotes. | **No** — No TPM driver. SmallAIOS boot measurement log provides equivalent software-level recording. |
| GRUB Shim Chain | shim.efi (Microsoft-signed) loads GRUB (distro-signed) loads kernel (distro-signed). | **No** — Would require signing SmallAIOS kernel with a shim-trusted key. |
| Multiboot2 | Bootloader passes info structure to kernel. No integrity verification. | **Partial** — SmallAIOS measures Multiboot2 info structure with SHA-3-256 (verified-boot feature). |

**Hardware Root of Trust:** Intel Boot Guard fuse bits + UEFI PK (Platform Key)
**Measured Boot:** TPM PCR extend chain (PCR0-7 cover firmware, bootloader, kernel)
**Runtime Integrity:** Intel TXT (Trusted Execution Technology), SGX enclaves

### AArch64

**Firmware Trust Chain:** BootROM → BL1 (TF-A) → BL2 → BL31 (EL3 runtime) → BL33 (U-Boot) → kernel

| Capability | Description | SmallAIOS Integration |
|---|---|---|
| ARM Trusted Firmware (TF-A) | BL1/BL2 verified boot: each stage hash-verifies the next using certificates (TBBR). Runs at EL3. | **No** — TF-A runs before SmallAIOS. SmallAIOS trusts EL1 entry from TF-A/U-Boot. |
| TrustZone | Hardware isolation between Secure World (EL3/S-EL1) and Normal World (EL1/EL0). | **No** — SmallAIOS runs in Normal World. Secure World services (OP-TEE) inaccessible without SMC interface. |
| U-Boot Verified Boot | FIT (Flattened Image Tree) with SHA-256 + RSA-2048/4096 signature verification. | **No** — Would require signing SmallAIOS Image with U-Boot FIT format. Future: sign FIT image. |
| OP-TEE | Trusted Application execution in Secure World. Key storage, crypto acceleration. | **No** — No TEE client driver. Could use for key storage in future. |
| DTB Integrity | Device tree blob passed from firmware. No standard verification. | **Partial** — SmallAIOS measures DTB with SHA-3-256 (verified-boot feature). |

**Hardware Root of Trust:** SoC-fused BootROM keys (vendor-specific, e.g., Tegra secure boot fuses)
**Measured Boot:** TF-A BL1 can extend measurements to event log (optional, vendor-dependent)
**Runtime Integrity:** TrustZone memory partitioning, TZASC (TrustZone Address Space Controller)

### RISC-V

**Firmware Trust Chain:** ZSBL (Zero Stage Boot Loader) → FSBL/U-Boot SPL → OpenSBI (M-mode) → kernel (S-mode)

| Capability | Description | SmallAIOS Integration |
|---|---|---|
| OpenSBI | Supervisor Binary Interface: M-mode firmware providing SBI ecalls. No built-in verified boot. | **No** — OpenSBI is trusted implicitly. SmallAIOS enters at S-mode. |
| PMP (Physical Memory Protection) | Hardware memory access control: configurable read/write/execute regions. Set by M-mode firmware. | **No** — PMP configured by OpenSBI before SmallAIOS starts. Could request PMP via SBI extension (non-standard). |
| Vendor Secure Boot | SiFive: BootROM → signed FSBL. T-Head: proprietary secure boot. | **No** — Vendor-specific, not standardized. |
| RISC-V TEE (Keystone/Penglai) | Enclave-based trusted execution using PMP isolation. Research-stage. | **No** — Not production-ready. |
| Hart ID + DTB | OpenSBI passes hart ID (a0) and DTB pointer (a1) to S-mode. No integrity verification. | **Partial** — SmallAIOS measures DTB with SHA-3-256 (verified-boot feature). |

**Hardware Root of Trust:** Vendor-specific (SiFive secure boot fuses, T-Head TEE)
**Measured Boot:** No standard mechanism. Vendor-specific extensions possible.
**Runtime Integrity:** PMP (coarse-grained), proposed Smmpt extension (page-level)

## Comparison Matrix

| Feature | x86-64 | AArch64 | RISC-V |
|---|---|---|---|
| **Firmware trust chain** | UEFI → shim → GRUB → kernel | TF-A BL1 → BL2 → BL31 → U-Boot → kernel | ZSBL → FSBL → OpenSBI → kernel |
| **Hardware root of trust** | Intel Boot Guard (fused) | SoC BootROM keys (fused) | Vendor-specific (not standardized) |
| **Measured boot** | TPM PCR extend chain | TF-A event log (optional) | No standard mechanism |
| **Secure boot** | UEFI Secure Boot (PK/KEK/db) | TF-A TBBR + U-Boot FIT RSA | Vendor-specific |
| **Runtime integrity** | Intel TXT, SGX | TrustZone, TZASC | PMP (M-mode configured) |
| **SmallAIOS integration** | Multiboot2 measurement | DTB measurement | DTB measurement |
| **SmallAIOS `verified-boot`** | Kernel self-hash + model signatures | Kernel self-hash + model signatures | Kernel self-hash + model signatures |

## Trust Boundaries

### Boot Stage Classification

| Boot Stage | x86-64 | AArch64 | RISC-V |
|---|---|---|---|
| Firmware/BootROM | Firmware-verified (UEFI) | Firmware-verified (TF-A) | Firmware-verified (BootROM) |
| Bootloader | Firmware-verified (Secure Boot) | Firmware-verified (TBBR) | **Unverified** (no standard) |
| Kernel load | **Unverified** (Multiboot2) | **Unverified** (U-Boot booti) | **Unverified** (OpenSBI jump) |
| Kernel self-check | SmallAIOS-verified (`verified-boot`) | SmallAIOS-verified (`verified-boot`) | SmallAIOS-verified (`verified-boot`) |
| ONNX model load | SmallAIOS-verified (`verified-boot`) | SmallAIOS-verified (`verified-boot`) | SmallAIOS-verified (`verified-boot`) |

### Accepted Risks

| Risk | Rationale |
|---|---|
| Firmware compromise (pre-kernel) | Outside SmallAIOS control. Mitigated by platform Secure Boot when enabled. |
| Bootloader tampering | Mitigated by UEFI Secure Boot (x86) or TF-A TBBR (ARM). No standard for RISC-V. |
| Kernel load without signature | Mitigated by SmallAIOS `verified-boot` self-integrity check post-load. |
| Hardware root of trust absent | Acceptable for development/QEMU. Production deploys should enable platform Secure Boot. |

## Platform-Specific Deployment Recommendations

### x86-64

1. **Enable UEFI Secure Boot** in firmware settings
2. **Enroll SmallAIOS signing key** in the UEFI db (Signature Database):
   - Generate a DER-encoded X.509 certificate from the Ed25519 build key
   - Use `efi-updatevar` or firmware setup to add to db
3. **Sign the kernel binary** with `sbsign` or equivalent before deploying
4. **Enable `verified-boot` feature** for kernel self-integrity and model signature verification
5. **Consider TPM-based attestation** for remote integrity verification (future work)

### AArch64

1. **Configure TF-A TBBR** (Trusted Board Boot Requirements) on the target SoC
2. **Sign SmallAIOS kernel** as a FIT image using U-Boot mkimage + RSA key:
   ```
   mkimage -f auto -A arm64 -T kernel -C none -a 0x40080000 -e 0x40080000 \
     -k /path/to/keys -K u-boot.dtb -d smallaios-aarch64 kernel.itb
   ```
3. **Enable U-Boot verified boot** (`CONFIG_FIT_SIGNATURE=y`)
4. **Enable `verified-boot` feature** for SmallAIOS-level verification
5. **On Tegra X1:** Jetson Nano supports secure boot via fuse programming (requires NVIDIA tools)

### RISC-V

1. **Enable vendor-specific secure boot** if available (SiFive Freedom U, T-Head C906)
2. **Configure OpenSBI PMP** to protect M-mode firmware memory from S-mode access:
   ```
   PMP region 0: 0x80000000-0x80200000 (OpenSBI) → no S-mode access
   PMP region 1: 0x80200000+ (kernel) → RWX for S-mode
   ```
3. **Enable `verified-boot` feature** for SmallAIOS-level verification
4. **Note:** RISC-V secure boot is least mature of the three platforms. The SmallAIOS `verified-boot` feature provides the primary integrity guarantee.
