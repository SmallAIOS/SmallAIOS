## 1. Feature Flag and Configuration

- [x] 1.1 Add `verified-boot` feature flag to `kernel/Cargo.toml` and `security/Cargo.toml`
- [x] 1.2 Add `verified-boot` feature flag to `onnx-rt/Cargo.toml` with dependency on `security/verified-boot`
- [x] 1.3 Implement `VerificationPolicy` enum (Enforce, WarnOnly, Disabled) in `security::crypto::verify`
- [x] 1.4 Add boot-time policy configuration in `kernel::state` (default WarnOnly when feature enabled)

## 2. Boot Measurement Log

- [x] 2.1 Define `BootMeasurement` struct (component_id: [u8;64], hash: Sha3_256Digest, timestamp: u64, status: VerifyStatus)
- [x] 2.2 Define `VerifyStatus` enum (Verified, Unverified, Failed)
- [x] 2.3 Implement `BootMeasurementLog` with fixed capacity of 32 entries and `add_measurement()` method
- [x] 2.4 Add `seal()` method to make the log immutable after boot and `is_sealed()` query
- [x] 2.5 Add `entries()` method to return all recorded measurements for IPC queries
- [x] 2.6 Integrate `BootMeasurementLog` into `kernel::state` behind `verified-boot` feature gate
- [x] 2.7 Write unit tests for measurement log: add, seal, immutability after seal, capacity limit

## 3. Kernel Self-Integrity Verification

- [x] 3.1 Add linker script symbols for `.text` and `.rodata` section boundaries in x86_64, aarch64, riscv64 linker scripts
- [x] 3.2 Implement `compute_kernel_hash()` in `kernel::boot_integrity` that SHA-3-256 hashes `.text` + `.rodata` using section boundary symbols
- [x] 3.3 Define `.boot_sig` section structure: Ed25519 signature (64 bytes) + expected hash (32 bytes) + magic bytes (8 bytes)
- [x] 3.4 Implement `verify_kernel_integrity()` that computes hash, reads `.boot_sig`, and verifies Ed25519 signature
- [x] 3.5 Wire `verify_kernel_integrity()` into early boot path (after BSS clear, before kernel_main body) behind feature gate
- [x] 3.6 Implement policy-based response: halt on failure in Enforce mode, warn in WarnOnly mode
- [x] 3.7 Record kernel verification result in boot measurement log
- [x] 3.8 Handle missing `.boot_sig` section gracefully (status Unverified, follow policy)
- [x] 3.9 Write unit tests for kernel hash computation and signature verification logic

## 4. ONNX Model Signature Verification Integration

- [x] 4.1 Add model verification call in ONNX runtime model load path using existing `verify_model_hybrid()` / `verify_model_ml_dsa()`
- [x] 4.2 Implement policy enforcement at model load: reject in Enforce, warn in WarnOnly, skip in Disabled
- [x] 4.3 Record model verification result in boot measurement log (model name, hash, status)
- [x] 4.4 Handle unsigned models (no signature block): check policy, record as Unverified or reject
- [x] 4.5 Handle tampered models (hash mismatch): always reject regardless of policy, record as Failed
- [x] 4.6 Write unit tests for model verification integration with each policy mode

## 5. Per-Architecture Boot Measurement Hooks

- [x] 5.1 Add DTB measurement hook in `arch/aarch64` boot path: hash DTB blob, record in measurement log
- [x] 5.2 Add DTB measurement hook in `arch/riscv64` boot path: hash DTB blob, record in measurement log
- [x] 5.3 Add Multiboot2 info measurement hook in `arch/x86_64` boot path: hash info structure, record in measurement log
- [x] 5.4 Write unit tests for per-architecture measurement hooks (mock section addresses)

## 6. Boot Security Comparison Matrix

- [x] 6.1 Write x86-64 boot security analysis: UEFI Secure Boot, Intel Boot Guard, TPM measured boot, GRUB shim chain
- [x] 6.2 Write AArch64 boot security analysis: ARM Trusted Firmware (ATF/TF-A), TrustZone, U-Boot verified boot, OP-TEE
- [x] 6.3 Write RISC-V boot security analysis: OpenSBI, PMP (Physical Memory Protection), vendor secure boot (SiFive, T-Head)
- [x] 6.4 Create comparison matrix table: firmware trust chain, hardware root of trust, measured boot, secure boot, runtime integrity per arch
- [x] 6.5 Document SmallAIOS integration points per architecture (Yes/No/Partial with rationale)
- [x] 6.6 Document trust boundaries per boot stage per architecture (firmware-verified, SmallAIOS-verified, unverified)
- [x] 6.7 Write platform-specific deployment recommendations for x86-64 (UEFI key enrollment, kernel signing)
- [x] 6.8 Write platform-specific deployment recommendations for AArch64 (ATF config, U-Boot FIT signing)
- [x] 6.9 Write platform-specific deployment recommendations for RISC-V (OpenSBI PMP, vendor boot options)

## 7. Integration and Testing

- [x] 7.1 Add `verified-boot` feature to CI matrix (clippy + test with feature enabled)
- [x] 7.2 Integration test: full boot with verified-boot enabled, check measurement log contains kernel entry
- [x] 7.3 Integration test: load signed ONNX model in Enforce mode, verify success
- [x] 7.4 Integration test: load unsigned ONNX model in Enforce mode, verify rejection
- [x] 7.5 Integration test: load tampered ONNX model, verify rejection regardless of policy
- [x] 7.6 Integration test: verify measurement log is sealed after boot and immutable
- [x] 7.7 Update `kernel` crate feature flags documentation in CLAUDE.md
