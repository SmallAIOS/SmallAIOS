## Why

SmallAIOS currently treats the boot chain as trusted ("assume verified boot chain" is an accepted risk in spec 06). All three architecture boot paths — x86-64 Multiboot2, AArch64 EL2→EL1, RISC-V OpenSBI — jump straight to `kernel_main` without verifying kernel integrity, measuring boot state, or checking ONNX model signatures. This leaves a gap: even though SmallAIOS has post-quantum crypto (ML-DSA-65, Ed25519, SHA-3) available in the `security` crate, none of it is used at boot time.

This change adds software-level boot integrity verification across all three architectures: kernel image hash checking, ONNX model signature verification at load time, and a boot measurement log. It also produces a cross-platform comparison of boot security features (UEFI Secure Boot, ARM TrustZone/ATF, RISC-V OpenSBI PMP) to document what hardware-assisted protections exist per platform.

## What Changes

- Add boot measurement log: record SHA-3 hashes of loaded components (kernel sections, DTB/config, ONNX models) into a kernel ring buffer at boot time
- Add ONNX model signature verification using Ed25519 (from existing `security::crypto`) at model load time, with opt-in enforcement via config flag
- Add kernel image self-integrity check: embed expected SHA-3 hash at build time, verify at early boot
- Create cross-platform boot security comparison matrix documenting UEFI Secure Boot (x86), ARM Trusted Firmware (aarch64), and OpenSBI PMP (riscv64) capabilities, trust boundaries, and what SmallAIOS can leverage per platform
- Add `verified-boot` feature flag to `kernel` and `security` crates gating the new verification code

## Capabilities

### New Capabilities
- `boot-integrity`: Boot-time integrity verification — kernel self-hash, model signature checking, boot measurement log
- `boot-security-matrix`: Cross-platform comparison of boot security features across x86-64, AArch64, RISC-V

### Modified Capabilities
- `06-security-model`: Move "verified boot chain" from out-of-scope to partially-mitigated; add boot measurement and model signing requirements

## Impact

- **Crates affected**: `kernel` (boot measurement log, self-hash), `security` (signature verification API, hash utilities), `onnx-rt` (model signature check at load), `arch/x86_64`, `arch/aarch64`, `arch/riscv64` (per-arch measurement hooks)
- **New feature flag**: `verified-boot` on `kernel` and `security` crates (default OFF to avoid breaking existing boot flow)
- **Build change**: Build script to embed SHA-3 hash of kernel binary (post-link step)
- **No breaking changes**: All new functionality is behind feature flag; default boot path unchanged
