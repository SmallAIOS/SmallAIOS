# op-tee-bridge-v1

## Summary

SmallAIOS on AArch64 runs in **Normal World at EL1** (or EL2 in the unikernel path that came out of `unikernel-orin-bringup-v1`). The ARM TrustZone architecture defines a parallel **Secure World** at S-EL1, accessed through the firmware-managed Secure Monitor at EL3 via the `SMC` (Secure Monitor Call) instruction. On Jetson Orin specifically, JetPack 6 ships Arm Trusted Firmware (TF-A) at EL3 as BL31 and **optionally** ships OP-TEE OS as the S-EL1 payload (BL32). Today SmallAIOS has zero ability to call into Secure World: no SMC driver, no GP TEE Client API, no awareness of OP-TEE's existence. `docs/boot-security-matrix.md` AArch64 row marks TrustZone and OP-TEE as **No** under "SmallAIOS Integration".

This change adds an OP-TEE client bridge to SmallAIOS — a minimal Normal World driver implementing the Global Platform TEE Client API subset that SmallAIOS actually needs (session open/close, command invoke, shared-memory register/unregister), backed by an SMC dispatcher that issues GP-standard SMC function IDs against the OP-TEE OS running in Secure World. The driver speaks the OP-TEE-specific SMC ABI (`OPTEE_SMC_CALL_WITH_ARG` = `0x32000004`, the standard Linux OP-TEE driver entry point) so it interoperates with an unmodified upstream OP-TEE OS build. SmallAIOS does not ship OP-TEE OS itself — OP-TEE is built and signed by the platform vendor (NVIDIA for Tegra Orin) and loaded by TF-A. SmallAIOS just talks to it.

Use cases that the bridge unlocks:

1. **Sealed key storage.** SmallAIOS's release-signing private key, ONNX model encryption keys, and PQC long-term identity keys move into OP-TEE Secure Storage. The Normal World never holds the raw bytes.
2. **Model-signature verification in Secure World.** A pinning Trusted Application (TA) holds the model-signing root public key. SmallAIOS sends the model's hash + signature to the TA over a session; the TA verifies and returns `OK`/`FAIL` without exposing the root key.
3. **Boot-integrity sealing.** PCRs from `boot-root-of-trust-v1` Phase 1 (x86-64 TPM 2.0) have no AArch64 counterpart on most Tegra hardware. With OP-TEE present, SmallAIOS seals its boot measurement log into Secure Storage at boot, then attests later (`remote-attestation-v1`) by asking the TA to unseal-and-sign with a Secure-World-resident key.
4. **PQC key sealing in the TEE.** ML-DSA-65 private keys are large (~4 KB). Storing them in OP-TEE Secure Storage and using the TA as an opaque signing oracle keeps the Normal World free of long-term private-key material and aligns with the project's PQC-default stance.

Capability: `security-tee` (new). The bridge lives in `security/src/tee/` (new module) with the platform-specific SMC layer in `arch/aarch64/src/smc.rs` (new module). The Cargo feature is `op-tee` on `smallaios-security`, default OFF (builds without OP-TEE see no overhead; production AArch64 deployments enable it).

## Why

- **TrustZone is the AArch64 hardware root of trust SmallAIOS isn't using.** The boot-security-matrix doc explicitly lists TrustZone under "AArch64 Hardware Root of Trust" alongside "SoC-fused BootROM keys", and lists "SmallAIOS Integration: No". TF-A's BL31 secure monitor is the *required* runtime layer SmallAIOS sits above on every modern ARM SoC; OP-TEE at S-EL1 is the *standard* way to access that secure monitor for application services. Bridging into it costs roughly the same effort as the AArch64 measured-boot work in `boot-root-of-trust-v1` Phase 2 (~3-4 weeks) and unlocks proportionally more capability — sealed storage, attestation backing, signature-verification oracle.
- **Tegra Orin already ships an OP-TEE-ready firmware stack.** Per NVIDIA's L4T 36.4 release notes (`docs/orin-secure-boot-flow.md`-style document on NVIDIA Docs), the JetPack 6 firmware chain is `BootROM → MB1 → MB2 → TF-A BL31 → optional OP-TEE BL32 → BL33 (U-Boot) → kernel`. OP-TEE is built into the BSP and can be enabled by flag — no firmware re-flash needed in many deployments. SmallAIOS's job is the Normal World half; the Secure World half exists.
- **GP TEE Client API is the standard, well-specified surface.** GlobalPlatform's TEE Client API (TEE_Client_API_Specification, v1.0 / v1.0.2) defines a stable C-level shape (`TEEC_Context`, `TEEC_Session`, `TEEC_InvokeCommand`, `TEEC_RegisterSharedMemory`, ...). Linux's `tee-supplicant` + `optee_armtz` driver implement it; SmallAIOS implements the same shape in Rust. Trusted Applications targeting this surface (the OP-TEE samples, NXP's Crypto TA, vendor-signed key-manager TAs) are immediately usable without TA-side changes.
- **Sealed key storage is the project's biggest missing piece.** The `verified-boot` story signs ONNX models with a SmallAIOS-managed key. If that key lives in DRAM next to the kernel that owns the verification logic, an attacker who roots the kernel owns the key. The same is true of the release-signing key (proposal: `boot-root-of-trust-v1` Phase 4). OP-TEE Secure Storage backed by a SoC-fused hardware-unique key gives us a place to put long-term private material where Normal-World compromise doesn't equal key compromise.
- **PQC-default fits the TEE story naturally.** ML-DSA-65 + ML-KEM-768 private keys are kilobytes-each, so even tiny TAs (a few-KB binary) become a meaningful key vault. The bridge lets SmallAIOS treat the TA as an opaque PQC signing oracle: "hash these bytes with SHA-3, ML-DSA-65 sign them with key handle K, return the signature". The Normal World never sees the private key, and the TA can enforce policy ("key K is only usable to sign objects whose first byte is 0x42") that the Normal World cannot.
- **DO-178C DAL A alignment.** Storing safety-critical signing keys outside the certified runtime *adds* a trust boundary that simplifies the runtime's certification scope. A TA with a 50-line key-vault interface is easier to certify (or third-party-vouch) than a full kernel; pushing key handling out of the kernel reduces the certification surface area of the kernel itself.

## Bridge architecture

```
SmallAIOS Normal World (EL1/EL2)
┌────────────────────────────────────────────────────┐
│  ONNX runtime, kernel, applications                │
│           │                                        │
│           ▼                                        │
│  security/src/tee/  (GP TEE Client API in Rust)    │
│  ├── TeeContext::initialize()                      │
│  ├── TeeSession::open(uuid, &[Param])              │
│  ├── TeeSession::invoke(cmd_id, &[Param])          │
│  ├── SharedMemory::register(&[u8], dir)            │
│           │                                        │
│           ▼                                        │
│  arch/aarch64/src/smc.rs  (raw SMC dispatch)       │
│  ├── smc_call(fid, a1, a2, a3, a4, a5, a6) →       │
│  │     SmcResult                                   │
│  │   (issues `smc #0` instruction, x0=fid, ...)    │
└─────────────────────────────│──────────────────────┘
                              │ smc #0
                              ▼
              ┌─────────────────────────────────────┐
              │  TF-A BL31 Secure Monitor (EL3)     │
              │  Routes OP-TEE FIDs → OP-TEE OS     │
              └─────────────────────────────────────┘
                              │
                              ▼
              ┌─────────────────────────────────────┐
              │  OP-TEE OS (S-EL1, BL32)            │
              │  Manages Trusted Applications       │
              │  ├── TA: Key Vault                  │
              │  ├── TA: Model Signature Verifier   │
              │  └── TA: Attestation Signer         │
              └─────────────────────────────────────┘
```

The bridge is a single `cfg(feature = "op-tee")` module path. When OP-TEE is absent (the SMC returns `OPTEE_SMC_RETURN_UNKNOWN_FUNCTION`), the bridge fails initialization gracefully and SmallAIOS falls back to today's software-only key storage. Callers see a typed `Result<TeeContext, TeeError::NotPresent>` and choose behavior accordingly.

## What ships

1. **`arch/aarch64/src/smc.rs`** — raw SMC dispatch. Implements ARM SMC Calling Convention (DEN 0028C) `smc #0` instruction wrapper, returning the 4 return registers (`x0`-`x3`) per the convention. ~50 LOC plus tests.
2. **`security/src/tee/mod.rs`** — GP TEE Client API surface in `#![no_std]` Rust. ~500 LOC. Exposes `TeeContext`, `TeeSession`, `SharedMemory`, `Operation`, `Param` (zero/value/ref/output flavors), `Error`.
3. **`security/src/tee/optee_msg.rs`** — OP-TEE-specific SMC message format (`OPTEE_MSG_ARG`, `OPTEE_MSG_PARAM`, ...). The on-the-wire format the OP-TEE OS expects in shared memory.
4. **`security/src/tee/smc_ids.rs`** — OP-TEE standard SMC function IDs (`OPTEE_SMC_CALL_GET_OS_REVISION = 0x32000000`, `OPTEE_SMC_CALL_WITH_ARG = 0x32000004`, `OPTEE_SMC_RPC_FUNC_FREE = 0x32000005`, etc.) as `const u32`.
5. **`security/src/tee/shm_pool.rs`** — shared-memory pool. SmallAIOS allocates a contiguous physical region (described to OP-TEE via `OPTEE_SMC_GET_SHM_CONFIG`), maps it into Normal World as a heap, and uses it as the parameter-passing zone for invokes. ~200 LOC.
6. **`security/src/tee/rpc.rs`** — Reverse-call (RPC) handling. OP-TEE TAs sometimes need to ask the Normal World for clock-reads, console output, or wait-for-interrupt; SmallAIOS implements the small subset of RPCs needed by the TAs we run. ~150 LOC.
7. **`docs/op-tee-bridge.md`** (new) — architecture, build/run instructions for Tegra Orin (which OP-TEE BL32 builds NVIDIA ships, how to confirm OP-TEE is loaded), key-vault TA reference, troubleshooting.
8. **CI: `op-tee-qemu-smoke`** — boots a kernel-build under QEMU `virt` with an unmodified upstream OP-TEE OS as BL32, drives a hello-world TA, asserts the round-trip works. Advisory initially.

## Use cases unlocked (delivered as follow-up changes)

This change ships **the bridge only**. The use cases that motivate it land as separate small changes:

- **`tee-key-vault-v1`** — moves the SmallAIOS release-signing private key and the ONNX-model-encryption keys into OP-TEE Secure Storage. Uses the bridge from this change. Estimated +1-2 weeks.
- **`tee-model-signature-verify-v1`** — replaces the in-kernel model-signature check in `onnx-rt` with a TA-side check. Estimated +1-2 weeks.
- **Used by `remote-attestation-v1` AArch64 backend** — see that change for the consumer side.
- **Used by `boot-root-of-trust-v1` Phase 2** for chain sealing — optional, lands as an additive enhancement to that change's spec.

Splitting the bridge from its consumers keeps this change a tight ~3-4 weeks of plumbing work with a clean acceptance criterion (hello-world TA round-trip), and lets the consumers land independently.

## Out of scope

- **Shipping our own TA binaries.** The TA build toolchain is OP-TEE-side (cross-compile with the OP-TEE TA devkit, sign with the platform-vendor TA-signing key). Per-use-case TAs land in their respective follow-up changes; this change provides the Normal-World half only. We document how to build / load a TA but ship none.
- **Multi-session / multi-context concurrency.** Phase 1 supports one `TeeContext` and one `TeeSession` open at a time. Concurrency lands when a consumer needs it (likely `tee-key-vault-v1` if multiple subsystems need parallel signing).
- **x86-64 SGX / SEV-SNP**. SmallAIOS's x86-64 confidential-compute story is `confidential-compute-v1` (separate change). OP-TEE is ARM-only by definition.
- **OP-TEE OS or TF-A modifications.** The bridge speaks the *standard* GP / OP-TEE SMC ABI against unmodified upstream firmware. Any SmallAIOS-specific Secure-World code (a custom TA, a vendor-specific PSA service) is its own change.
- **TLS / network transport for sessions.** All TEE traffic is local SMC; no network surface exists in this change.

## Effort estimate

| Sub-task | Scope | Estimate |
|----------|-------|----------|
| 1 | `smc.rs` raw dispatch + tests | ~2 days |
| 2 | GP TEE Client API surface in Rust | ~1 week |
| 3 | OP-TEE message format + SMC IDs | ~3 days |
| 4 | Shared-memory pool | ~3 days |
| 5 | RPC handling (subset) | ~3 days |
| 6 | QEMU + OP-TEE OS CI smoke | ~3 days |
| 7 | Docs + Jetson Orin runbook | ~2 days |
| **Total** | | **~3-4 weeks** |

## DO-178C alignment

Moving long-term private-key material out of the certified kernel runtime into a separately-managed Secure World component is a textbook trust-boundary reduction. The DAL A claim "the certified kernel never holds the release-signing private key in cleartext memory" becomes provable by inspection of the kernel's key-storage code (which only holds TEE *handles*, never raw bytes). The TA that holds the keys can be certified separately to a lower DAL or vouched-for via vendor attestation — the boundary is the SMC interface, which is a small, well-specified surface (the GP TEE Client API).

## PQC stance

The bridge is crypto-agnostic at the SMC level — TAs decide what crypto they implement. The follow-up `tee-key-vault-v1` TA SHALL hold ML-DSA-65 + Ed25519 hybrid keys and expose a hybrid-signing operation. The `tee-model-signature-verify-v1` TA SHALL accept ML-DSA-65 + Ed25519 hybrid signatures. The bridge itself imposes no crypto choice; the follow-ups carry the PQC-default forward.
