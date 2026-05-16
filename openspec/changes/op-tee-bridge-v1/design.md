# Design — op-tee-bridge-v1

## Goal

A Normal-World OP-TEE client driver in `#![no_std]` Rust that round-trips a `TEEC_InvokeCommand` against an unmodified upstream OP-TEE OS BL32 image, observable as:

- A "hello world" TA returning a fixed response over an `OPTEE_SMC_CALL_WITH_ARG` session.
- A captured serial log showing `TF-A → OP-TEE → SmallAIOS` SMC traversal latency in the < 100 µs range (well within ARM SMC call-and-return budgets per Arm DEN 0028C).
- The bridge gracefully reporting `TeeError::NotPresent` when SmallAIOS boots on hardware without OP-TEE installed.

## Alternatives considered

### 1. Use Linux's `optee_armtz` driver via shim

Rejected. The Linux driver is GPL-2.0 and assumes a Linux device-tree-driven probe path, request-queue infrastructure, and userspace `tee-supplicant` daemon for RPC handling. None of those map cleanly onto SmallAIOS's `#![no_std]` runtime. A wholesale port would also import GPL code into an Apache-2.0 / MIT-licensed workspace (license review surface SmallAIOS deliberately avoids).

### 2. Implement the full GP TEE Client API including TEEC_RequestCancellation, TEEC_NotifyEvent, multi-session contexts

Rejected for Phase 1. The full GP spec is ~150 pages and includes asynchronous cancellation, event notifications, and multi-context bookkeeping that no SmallAIOS use case needs at the start. Phase 1 ships the subset that round-trips a synchronous `TEEC_InvokeCommand` — about 30% of the spec, the part every real-world client actually uses.

### 3. Use ARM PSA Crypto API instead of OP-TEE

Considered. PSA Crypto is a higher-level abstraction over a TEE-resident crypto service. OP-TEE OS ships a PSA Crypto TA, so we could target PSA directly and skip the lower-level GP API.

Rejected because (a) PSA Crypto narrows the bridge to "crypto operations only", but we want sealing, attestation, and possibly custom TA workloads later — the lower-level GP API supports all of those; (b) the PSA abstraction is just a TA built on top of GP, so building the GP bridge is the same engineering work plus optionality; (c) `remote-attestation-v1` will benefit from PSA Initial Attestation API (PSA-IA) — that's an *additional* TA that runs on top of the GP bridge we're building here, not an alternative to it.

The PSA Crypto / PSA-IA TAs will be consumed by the follow-up changes. This change provides the GP-level bridge they sit on.

### 4. SMC dispatch via firmware/PSCI-like interface instead of OP-TEE-specific FIDs

Rejected. OP-TEE's SMC ABI is a documented stable interface (`OPTEE_SMC_CALL_WITH_ARG` and friends have been stable across OP-TEE OS 3.x → 4.x). The GP TEE Client API maps cleanly to it. Building a generic "TEE dispatch" layer that could swap to a non-OP-TEE secure OS is over-engineering — the only Secure-World OS that runs at S-EL1 on the target Tegra Orin platform is OP-TEE.

### 5. Skip OP-TEE entirely, rely on Tegra-specific NVIDIA SE / Secure Engine

Considered briefly. NVIDIA's Secure Engine (SE2) on Tegra234 provides AES, SHA, RSA, and TRNG accelerators accessible from Secure World. There is no documented Normal-World driver path that doesn't go through OP-TEE; NVIDIA's stack uses OP-TEE TAs to expose SE2 services. Going direct would require reverse-engineering closed NVIDIA Secure-World code. Rejected — OP-TEE is the right abstraction layer for both portability and access.

## SMC dispatch (Phase 1's smallest module)

```rust
// arch/aarch64/src/smc.rs
#[derive(Clone, Copy, Debug)]
pub struct SmcResult { pub x0: u64, pub x1: u64, pub x2: u64, pub x3: u64 }

/// SMC32 / SMC64 dispatch per Arm DEN 0028C.
/// Issues `smc #0` with x0..x6 set, returns x0..x3.
/// Safe: SMC is a privileged instruction but inherently synchronous; from
/// EL1/EL2 it returns control to the caller. No memory model concerns
/// beyond the input/output registers.
pub unsafe fn smc_call(
    fid: u32, a1: u64, a2: u64, a3: u64, a4: u64, a5: u64, a6: u64,
) -> SmcResult {
    let mut x0: u64 = fid as u64;
    let (mut x1, mut x2, mut x3): (u64, u64, u64) = (a1, a2, a3);
    core::arch::asm!(
        "smc #0",
        inout("x0") x0,
        inout("x1") x1,
        inout("x2") x2,
        inout("x3") x3,
        in("x4") a4, in("x5") a5, in("x6") a6,
        options(nomem, nostack, preserves_flags),
    );
    SmcResult { x0, x1, x2, x3 }
}
```

That's the entire low-level surface. Everything else is bytes-in / bytes-out built on top.

## Trust model

| Component | Trusted by SmallAIOS for what |
|-----------|------------------------------|
| TF-A BL31 | EL3 secure monitor routing SMCs faithfully; signed by platform vendor. |
| OP-TEE OS BL32 | Holds TA private state, enforces TA isolation; signed by platform vendor. |
| Trusted Application (per-use-case) | Holds key material, enforces use-case policy; signed by TA-signing key (vendor or SmallAIOS). |
| Normal World (SmallAIOS) | Caller of TAs; SmallAIOS holds *TEE handles* and *shared-memory pointers*, never long-term private material. |

The threat model assumes Normal World can be compromised independently of Secure World. A compromised Normal World can call arbitrary TA commands but cannot exfiltrate sealed-storage contents, raw private keys, or TA internal state — that's the TEE's whole job.

The threat model **does** assume TF-A and OP-TEE OS are uncompromised. If the platform vendor's secure-firmware signature is forged, the bridge offers no defense — that's the responsibility of the platform's BootROM-rooted secure-boot chain (out of scope for this change; partially covered by `boot-root-of-trust-v1` Phase 4 and 100% covered by vendor-fused boot guards documented but not automated there).

## Shared-memory pool design

OP-TEE OS needs Normal World shared memory for parameter passing (the GP `TEEC_RegisterSharedMemory` surface). The OP-TEE OS publishes the allowed shared-memory range at boot via `OPTEE_SMC_GET_SHM_CONFIG` (function ID `0x32000007`); SmallAIOS reads that range and uses it as a heap.

Two implementation choices:

- **Dedicated DRAM region described in DTB** (`/reserved-memory/optee-shm { reg = <...>; };`). NVIDIA's Orin BSP uses this; it's pre-mapped, no kernel-side allocator interaction needed. Used when available.
- **Dynamically allocated from the kernel heap, registered with OP-TEE OS via `OPTEE_SMC_RPC_FUNC_ALLOC`/`FREE`**. Required when no DTB-reserved region exists. Adds more SMC traffic per session but works everywhere.

Phase 1 implements both, prefers DTB-reserved when available, falls back to dynamic. The choice is logged in `BootMeasurementLog` for audit clarity.

A shared-memory parameter is a 4-tuple `(phys_addr, size, direction, attr)`. The bridge holds a `SharedMemory<'_>` lifetime-anchored to a SmallAIOS-side buffer; on drop, it unregisters from OP-TEE. The lifetime tracking is pure Rust — no global registry, no leakable handles.

## GP TEE Client API mapping

| GP API | SmallAIOS Rust API | SMC FID | Notes |
|--------|--------------------|---------|-------|
| `TEEC_InitializeContext` | `TeeContext::new()` | `OPTEE_SMC_CALL_GET_OS_REVISION` (probe) | Returns `Err(NotPresent)` on unknown FID. |
| `TEEC_OpenSession` | `TeeContext::open_session(uuid)` | `OPTEE_SMC_CALL_WITH_ARG` (cmd = `OPTEE_MSG_CMD_OPEN_SESSION`) | UUID is the TA's GP UUID. |
| `TEEC_InvokeCommand` | `TeeSession::invoke(cmd_id, &[Param])` | `OPTEE_SMC_CALL_WITH_ARG` (cmd = `OPTEE_MSG_CMD_INVOKE_COMMAND`) | The bread-and-butter call. |
| `TEEC_CloseSession` | `Drop for TeeSession` | `OPTEE_SMC_CALL_WITH_ARG` (cmd = `OPTEE_MSG_CMD_CLOSE_SESSION`) | RAII-driven. |
| `TEEC_RegisterSharedMemory` | `SharedMemory::register(&buf, dir)` | `OPTEE_MSG_CMD_REGISTER_SHM` | Returns a lifetime-bound handle. |
| `TEEC_AllocateSharedMemory` | covered by `SharedMemory::register` | n/a | We always provide the buffer; no need for OP-TEE-allocated path. |
| `TEEC_FinalizeContext` | `Drop for TeeContext` | n/a (no FID — local cleanup only) | RAII. |
| `TEEC_RequestCancellation` | **deferred** | `OPTEE_MSG_CMD_CANCEL` | Add when a consumer needs it. |

`Param` is a tagged enum mirroring GP's `TEEC_Parameter`:

```rust
pub enum Param<'a> {
    None,
    Value { a: u64, b: u64 },
    ValueOutput { a: &'a mut u64, b: &'a mut u64 },
    MemRef(&'a SharedMemory<'a>, usize, usize),  // offset, size
    MemRefOutput(&'a mut SharedMemory<'a>, usize, usize),
}
```

This is the entire client surface. ~40 public API items, all `#![no_std]`-friendly.

## RPC handling

OP-TEE TAs occasionally need Normal World assistance — wall-clock time, console output, wait-for-interrupt (used by some implementations of secure timers). The SMC return convention encodes "I need RPC, please call me back" via `OPTEE_SMC_RPC_FUNC_*` return codes. Phase 1 handles the minimal RPC subset:

| RPC | What SmallAIOS does |
|-----|--------------------|
| `OPTEE_SMC_RPC_FUNC_ALLOC` | Allocate from the dynamic shared-memory path, return phys addr. |
| `OPTEE_SMC_RPC_FUNC_FREE` | Free a previously-allocated shared-memory region. |
| `OPTEE_SMC_RPC_FUNC_FOREIGN_INTR` | Re-issue the SMC after handling the interrupt (cooperative). |
| `OPTEE_SMC_RPC_FUNC_CMD` (subcmd: wait-for-keypress, get-time) | Phase 1 returns `OPTEE_SMC_RETURN_ENOTSUP` for any subcmd not on a documented allowlist. |

Anything beyond that allowlist is rejected — TAs that depend on it get a clean failure rather than a partial implementation. The allowlist grows as use-case TAs need more.

## CI smoke

The OP-TEE project ships `qemu_v8.mk` build configuration producing a complete TF-A + OP-TEE OS + Buildroot image bootable under `qemu-system-aarch64 -M virt -cpu cortex-a57 -smp 2 -bios bl1.bin`. The smoke job builds that fixture (one-time CI cache), boots SmallAIOS as BL33 (replacing the Buildroot Linux that the OP-TEE upstream provides), and runs a SmallAIOS test program that opens a session to OP-TEE's reference "hello_world" TA and asserts the round-trip works.

The job is advisory at land time and promotes to gate after a stable week. The OP-TEE fixture is large (~50 MB), so the CI image will pre-bake it to avoid per-PR rebuild time.

## What this change explicitly does NOT do

- **Does not ship Secure-World code.** No TA source in this repo, no TF-A patches, no OP-TEE OS modifications. The bridge talks to *unmodified upstream* OP-TEE OS.
- **Does not enable OP-TEE on Tegra Orin where NVIDIA's BSP doesn't ship it.** Users of Tegra Orin variants where OP-TEE BL32 is not built into the firmware see `TeeError::NotPresent` and the bridge no-ops cleanly. Enabling OP-TEE in the BSP is an NVIDIA-side decision.
- **Does not modify the boot path.** OP-TEE is loaded by TF-A before SmallAIOS gets control. SmallAIOS just discovers it via SMC probe at runtime.
- **Does not introduce a new dependency layer.** `security/src/tee/` is Layer 0 (foundation); `arch/aarch64/src/smc.rs` is Layer 2 (HAL). The dependency direction respects the workspace's existing 4-layer model.
