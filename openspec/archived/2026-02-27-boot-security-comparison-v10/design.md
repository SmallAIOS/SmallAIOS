## Context

SmallAIOS has three architecture boot paths that jump directly to `kernel_main` with no integrity checks:

- **x86-64**: Multiboot2 header, loaded by GRUB/QEMU `-kernel`, assumes long mode
- **AArch64**: DTB pointer from firmware/U-Boot, EL2→EL1 transition, parks secondary cores
- **RISC-V**: OpenSBI S-mode handoff with hart ID + DTB

The `security` crate already has production-quality crypto: SHA-3-256 (`sha3_256`), Ed25519 (`ed25519_sign`/`ed25519_verify`), ML-DSA-65, and hybrid signatures. The `security::crypto::verify` module already handles ONNX model signature verification with `hash_model()` and `verify_model_hybrid()`. However, none of this is wired into the boot path.

The security model spec (06) lists "compromised firmware/UEFI" as an accepted risk, but does not address kernel self-integrity or boot measurement — things SmallAIOS can verify in software without depending on platform firmware.

## Goals / Non-Goals

**Goals:**
- Implement software-level boot integrity verification that works identically across all 3 architectures
- Add a boot measurement log that records SHA-3-256 hashes of kernel sections, DTB/config, and loaded ONNX models
- Wire existing `security::crypto::verify` into the ONNX model load path so models are verified before execution
- Add a kernel self-integrity check (embedded hash verified at early boot)
- Document platform-specific hardware boot security capabilities (UEFI Secure Boot, ARM TF-A, OpenSBI PMP) in a comparison matrix
- Gate all new code behind `verified-boot` feature flag — zero impact on default boot

**Non-Goals:**
- Implementing UEFI Secure Boot, ARM TrustZone, or RISC-V PMP drivers (hardware-dependent)
- TPM or hardware security module integration (no hardware access in WSL)
- Remote attestation protocols (future work)
- Changing the default boot flow (feature is opt-in)

## Decisions

### 1. Boot measurement log as kernel ring buffer

**Decision**: Store boot measurements in a fixed-size array in `kernel::state`, similar to the existing log ring buffer.

**Rationale**: The kernel already uses ring buffers for logging. A `BootMeasurementLog` with capacity for 32 entries (kernel, DTB, config, ONNX models) is sufficient and requires no heap allocation. Each entry records: component name, SHA-3-256 hash, timestamp, and verification status.

**Alternative considered**: Append to the existing audit log — rejected because boot measurements have different retention requirements (must be available for the lifetime of the boot, not subject to ring buffer rotation).

### 2. Ed25519 for kernel self-hash signature (not ML-DSA-65)

**Decision**: Use Ed25519 for the embedded kernel hash signature rather than ML-DSA-65 or hybrid.

**Rationale**: The kernel hash is verified exactly once at early boot, before the PQC stack is needed. Ed25519 signatures are 64 bytes vs 3,309 bytes for ML-DSA-65 — critical when embedding in the binary. The threat of quantum attacks on a kernel hash that's replaced every build is negligible. ML-DSA-65 can be used for long-lived model signatures (which the verify module already supports).

**Alternative considered**: Hybrid Ed25519+ML-DSA-65 — rejected for binary size impact (~3.4 KB per embedded signature).

### 3. Build-time hash embedding via `include_bytes!` + build script

**Decision**: Use a two-pass build approach:
1. First pass builds the kernel and computes SHA-3-256 of the `.text` + `.rodata` sections
2. Hash is signed with a build-time Ed25519 key and embedded as a `#[link_section]` constant
3. At boot, the kernel hashes its own `.text` + `.rodata` and verifies against the embedded signed hash

**Rationale**: This avoids the chicken-and-egg problem (can't hash a binary that contains its own hash) by hashing only the code/rodata sections and embedding the signed hash in a separate section (`.boot_sig`). The linker script already defines section boundaries via symbols (`__text_start`, `__text_end`, etc.).

**Alternative considered**: External signature file loaded alongside kernel — rejected because it requires a filesystem or bootloader protocol change.

### 4. Model verification at load time via existing verify module

**Decision**: Call `security::crypto::verify::verify_model_hybrid()` (or `verify_model_ml_dsa()`) in the ONNX runtime's model load path. Add a `VerificationPolicy` enum: `Enforce` (reject unsigned models), `WarnOnly` (log but allow), `Disabled`.

**Rationale**: The verification code already exists and is tested. Only the call site and policy enforcement are missing. A policy enum lets deployments choose their security posture.

### 5. Comparison matrix as a spec document (not code)

**Decision**: The cross-platform comparison is a spec document (`boot-security-matrix/spec.md`) with structured requirements, not runtime code.

**Rationale**: The comparison documents what each platform offers and what SmallAIOS should leverage per-architecture. It informs future hardware-dependent work without requiring implementation now.

## Risks / Trade-offs

- **[Build complexity]** Two-pass build for kernel hash embedding adds a build step → Mitigation: Only runs when `verified-boot` feature is enabled; default build unchanged
- **[Boot time]** SHA-3-256 of kernel text section adds latency → Mitigation: <1ms for an 8MB kernel on modern hardware; negligible vs boot target of <50ms
- **[False sense of security]** Software-only verification can't protect against firmware-level attacks → Mitigation: Document clearly in the spec that this protects against post-boot tampering and supply chain issues, not firmware compromise. Hardware boot security is documented as future work in the comparison matrix
- **[Key management]** Build-time signing key must be protected → Mitigation: Key is only used in CI/release builds; dev builds skip verification by default (feature flag OFF)
