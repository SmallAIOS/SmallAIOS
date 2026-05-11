# Tasks — op-tee-bridge-v1

## 0. Prerequisites

- [ ] 0.1 Confirm the upstream OP-TEE OS `qemu_v8.mk` build is reproducible on the CI runner image. Document exact commit SHA pinning in `docs/op-tee-bridge.md`.
- [ ] 0.2 Capture a representative Tegra Orin TF-A + OP-TEE firmware version from a JetPack 6.2.1 install (`tegra_uefi_versions`, `cat /proc/device-tree/firmware/optee/compatible` if available). Paste in PR description.
- [ ] 0.3 Confirm the Tegra Orin BSP exposes `/reserved-memory/optee@…` in its DTB. If not, document the dynamic-shared-memory fallback path in design.md as the only path; update the proposal "Out of scope" if needed.
- [ ] 0.4 Pin the `unikernel-orin-bringup-v1` DTB-parser improvements (or absorb them into this change's prerequisites) — `parse_dtb` must support `/reserved-memory/` subnodes for the shared-memory pool detection.

## 1. SMC dispatch (Layer 2 HAL)

- [ ] 1.1 Create `arch/aarch64/src/smc.rs` with `SmcResult` struct and `smc_call(fid, a1..a6) -> SmcResult` using inline asm per Arm DEN 0028C.
- [ ] 1.2 Unit-test `smc_call` against a mock dispatcher (cfg(test) version that intercepts `smc #0` via a function pointer set by the test). Verifies register layout matches Arm SMC Calling Convention.
- [ ] 1.3 Document the SMC ABI assumptions in a doc comment: `fid` in x0 (32-bit SMC32 or 64-bit SMC64 indicated by bit 30), arguments in x1-x6, returns in x0-x3.
- [ ] 1.4 Ensure `smc.rs` is gated `#[cfg(target_arch = "aarch64")]` and feature `op-tee` (no SMC instruction on x86-64 / RISC-V).

## 2. OP-TEE SMC IDs + message format

- [ ] 2.1 Create `security/src/tee/smc_ids.rs` defining the OP-TEE standard SMC FIDs as `const u32`: `OPTEE_SMC_CALL_GET_OS_REVISION = 0x32000000`, `OPTEE_SMC_CALL_WITH_ARG = 0x32000004`, `OPTEE_SMC_RPC_FUNC_ALLOC = 0xFFFFFFFE`, etc. Source: OP-TEE OS `core/include/optee_smc.h` (canonical reference, pin commit SHA).
- [ ] 2.2 Create `security/src/tee/optee_msg.rs` defining `OPTEE_MSG_ARG` and `OPTEE_MSG_PARAM` structs matching OP-TEE OS's `optee_msg.h` exactly. Use `#[repr(C)]` and assert struct sizes match the C definitions at compile time.
- [ ] 2.3 Define the GP `TeeError` enum with variants for OP-TEE return codes (`TEEC_ERROR_BAD_FORMAT`, `TEEC_ERROR_NOT_SUPPORTED`, ...) plus SmallAIOS-specific (`NotPresent`, `SharedMemoryExhausted`).
- [ ] 2.4 Implement `From<u32> for TeeError` mapping OP-TEE numeric codes to enum variants.

## 3. Shared-memory pool

- [ ] 3.1 Create `security/src/tee/shm_pool.rs` exposing `ShmPool::initialize() -> Result<ShmPool>` and `ShmPool::alloc(size, align) -> Result<ShmBlock>`.
- [ ] 3.2 Implement the DTB-reserved-region path: query `parse_dtb` for `/reserved-memory/optee-shm` (or vendor-specific equivalent), map it as a heap.
- [ ] 3.3 Implement the dynamic path: on first allocation, ask OP-TEE via `OPTEE_SMC_RPC_FUNC_ALLOC` for a fresh region.
- [ ] 3.4 Implement `Drop for ShmBlock` that returns the block to the pool / OP-TEE.
- [ ] 3.5 Unit-test pool exhaustion behavior (should return `TeeError::SharedMemoryExhausted`, not panic).

## 4. GP TEE Client API surface

- [ ] 4.1 Create `security/src/tee/mod.rs` exposing public types: `TeeContext`, `TeeSession`, `SharedMemory`, `Param`, `Operation`.
- [ ] 4.2 Implement `TeeContext::new() -> Result<TeeContext, TeeError>`: probe via `OPTEE_SMC_CALL_GET_OS_REVISION`. Returns `Err(NotPresent)` on `OPTEE_SMC_RETURN_UNKNOWN_FUNCTION`.
- [ ] 4.3 Implement `TeeContext::open_session(uuid: &Uuid) -> Result<TeeSession>` using `OPTEE_MSG_CMD_OPEN_SESSION` over `OPTEE_SMC_CALL_WITH_ARG`.
- [ ] 4.4 Implement `TeeSession::invoke(cmd_id: u32, params: &mut [Param]) -> Result<(), TeeError>` using `OPTEE_MSG_CMD_INVOKE_COMMAND`. Encode the four params into `OPTEE_MSG_PARAM` slots per GP conventions.
- [ ] 4.5 Implement `Drop for TeeSession` calling `OPTEE_MSG_CMD_CLOSE_SESSION`.
- [ ] 4.6 Implement `Drop for TeeContext` (local cleanup only, no SMC).
- [ ] 4.7 Implement `SharedMemory::register(buf: &[u8], dir: ShmDir)` returning a lifetime-anchored handle. Drop unregisters.

## 5. RPC handling

- [ ] 5.1 Create `security/src/tee/rpc.rs` implementing the SMC return-code → RPC dispatch loop. After `smc_call`, if the return is an RPC request, handle it locally and re-issue.
- [ ] 5.2 Implement `OPTEE_SMC_RPC_FUNC_ALLOC` handler (delegates to `ShmPool`).
- [ ] 5.3 Implement `OPTEE_SMC_RPC_FUNC_FREE` handler.
- [ ] 5.4 Implement `OPTEE_SMC_RPC_FUNC_FOREIGN_INTR` handler (re-issue after a hint to the scheduler).
- [ ] 5.5 Implement allowlist enforcement for `OPTEE_SMC_RPC_FUNC_CMD` subcommands. Reject unknown subcommands with `OPTEE_SMC_RETURN_ENOTSUP`.

## 6. Cargo wiring

- [ ] 6.1 Add `op-tee = []` feature to `smallaios-security` `Cargo.toml`. Default OFF.
- [ ] 6.2 Gate the `tee` module on `cfg(feature = "op-tee")`.
- [ ] 6.3 Ensure a default-features build of every crate that depends on `smallaios-security` still works (the bridge is purely additive).
- [ ] 6.4 Add doc-comment to the feature explaining the AArch64-only scope and the GP TEE Client API mapping.

## 7. CI smoke

- [ ] 7.1 Pre-bake the OP-TEE upstream `qemu_v8.mk` artifacts into the CI runner image (TF-A `bl1.bin`, OP-TEE OS `tee.bin`, U-Boot or direct BL33 stub). Pin to a specific OP-TEE OS commit SHA.
- [ ] 7.2 Add `op-tee-qemu-smoke` CI job: builds the SmallAIOS kernel with `--features op-tee`, packages it as BL33, boots under `qemu-system-aarch64 -M virt -cpu cortex-a57 -smp 2 -bios bl1.bin -semihosting-config enable=on,target=native`.
- [ ] 7.3 Smoke-test program in `tests/op_tee_smoke.rs`: opens a session to OP-TEE's built-in `pta_invoke_tests` PTA (UUID `d96a5b40-c3e5-21e3-8794-1002a5d5c61b`), invokes the no-op test, asserts success.
- [ ] 7.4 Mark advisory (`continue-on-error: true`) at land. Promote to gate after one stable week.

## 8. Docs

- [ ] 8.1 Create `docs/op-tee-bridge.md` covering: architecture diagram, SMC ABI summary, GP API mapping table, how to verify OP-TEE is present on Tegra Orin (`/sys/firmware/devicetree/base/firmware/optee/method` exists on L4T; on SmallAIOS, the bridge's probe API is the only check), Trusted Application development pointers (link to OP-TEE upstream docs — we don't reproduce them).
- [ ] 8.2 Update `docs/boot-security-matrix.md` AArch64 row: TrustZone column flips from **No** to **Yes (bridge)**, OP-TEE column flips from **No** to **Yes (client-side)**.
- [ ] 8.3 Update `CLAUDE.md` "Current state" to note the OP-TEE bridge capability.
- [ ] 8.4 Add a `docs/op-tee-bridge.md` troubleshooting section covering: `TeeError::NotPresent` on hardware where OP-TEE was expected (probably means the BL32 image isn't built — check the platform vendor's firmware build), shared-memory exhaustion (raise the DTB-reserved size), unknown TA UUID (TA is not loaded by OP-TEE OS — check OP-TEE's TA loading mechanism for the platform).

## 9. Close-out

- [ ] 9.1 Capture a successful round-trip log against the OP-TEE QEMU fixture in the PR description. Format: SMC probe response → session open → invoke result → session close.
- [ ] 9.2 Run `cargo geiger` on `security/src/tee/`; document any `unsafe` blocks (the SMC inline-asm is the only one expected). Justify each in code comments.
- [ ] 9.3 Run `cargo clippy --features op-tee -- -D warnings`. Fix or document any warnings.
- [ ] 9.4 `openspec validate op-tee-bridge-v1` returns valid.
- [ ] 9.5 PR title: `feat(security/tee): op-tee-bridge-v1 — Normal-World OP-TEE client driver`. Target `develop`.
