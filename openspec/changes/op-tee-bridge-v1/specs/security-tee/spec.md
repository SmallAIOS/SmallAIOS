## ADDED Requirements

### Requirement: GP TEE Client API surface in #![no_std] Rust

The `smallaios-security` crate SHALL provide a Normal-World OP-TEE client implementing the Global Platform TEE Client API subset required for synchronous `TEEC_InvokeCommand` round-trips, gated by a new `op-tee` Cargo feature, available only when compiled for AArch64.

#### Scenario: Context initialization probes for OP-TEE presence

- **GIVEN** a SmallAIOS build with `--features op-tee` on AArch64
- **WHEN** the kernel calls `TeeContext::new()`
- **THEN** the bridge SHALL issue `OPTEE_SMC_CALL_GET_OS_REVISION` (FID `0x32000000`) via `smc_call`
- **AND** if the SMC returns a valid OS revision, the call SHALL return `Ok(TeeContext { … })` with the OP-TEE OS version recorded
- **AND** if the SMC returns `OPTEE_SMC_RETURN_UNKNOWN_FUNCTION` (`0xFFFFFFFF`), the call SHALL return `Err(TeeError::NotPresent)` and the caller SHALL be free to fall back to software-only behavior

#### Scenario: Build without op-tee feature is byte-identical to today

- **GIVEN** a SmallAIOS build without `--features op-tee`
- **THEN** the resulting kernel binary SHALL NOT link any `security::tee` symbols
- **AND** the resulting kernel binary SHALL NOT contain the `smc #0` instruction emitted by `arch/aarch64/src/smc.rs`
- **AND** all software-only key paths SHALL behave exactly as before this change

#### Scenario: Session open + invoke + close round-trip

- **GIVEN** an initialized `TeeContext`, a TA UUID, and a TA that exposes a known command
- **WHEN** the caller does `let session = ctx.open_session(uuid)?; session.invoke(cmd_id, &mut [Param::Value { a: 1, b: 2 }])?;` and then drops the session
- **THEN** three SMC calls SHALL be issued in order: `OPTEE_MSG_CMD_OPEN_SESSION`, `OPTEE_MSG_CMD_INVOKE_COMMAND`, `OPTEE_MSG_CMD_CLOSE_SESSION`, all wrapped in `OPTEE_SMC_CALL_WITH_ARG` (FID `0x32000004`)
- **AND** the per-call `OPTEE_MSG_ARG` and `OPTEE_MSG_PARAM` structs in shared memory SHALL exactly match the OP-TEE upstream layout (`optee_msg.h` from the pinned OP-TEE OS commit)
- **AND** the session's `Drop` impl SHALL be infallible — close-failure SHALL be logged but SHALL NOT panic

#### Scenario: Parameter encoding covers Value, ValueOutput, MemRef, MemRefOutput

- **GIVEN** an `invoke` call with the four parameter slots populated with `Param::Value { a, b }`, `Param::ValueOutput { a, b }`, `Param::MemRef(&shm, off, len)`, `Param::MemRefOutput(&mut shm, off, len)` respectively
- **WHEN** the bridge encodes the OPTEE_MSG_PARAM slots
- **THEN** slot 0 SHALL carry `OPTEE_MSG_ATTR_TYPE_VALUE_INPUT` with the (a, b) packed values
- **AND** slot 1 SHALL carry `OPTEE_MSG_ATTR_TYPE_VALUE_OUTPUT` and on return the bridge SHALL write back the output (a, b) into the caller's `&mut u64` references
- **AND** slot 2 SHALL carry `OPTEE_MSG_ATTR_TYPE_RMEM_INPUT` (registered memory) with the phys address + offset + length
- **AND** slot 3 SHALL carry `OPTEE_MSG_ATTR_TYPE_RMEM_OUTPUT` and the output buffer SHALL be readable by the caller after the invoke returns

### Requirement: Shared-memory pool with DTB-reserved and dynamic backends

The bridge SHALL maintain a shared-memory pool for parameter passing to OP-TEE, preferring a DTB-reserved-region backend when available and falling back to dynamic allocation via `OPTEE_SMC_RPC_FUNC_ALLOC` otherwise.

#### Scenario: DTB-reserved region is used when present

- **GIVEN** a DTB exposing `/reserved-memory/optee-shm { reg = <…>; };` (or a vendor-specific equivalent recognized by the bridge)
- **WHEN** `ShmPool::initialize()` runs
- **THEN** the pool SHALL be backed by the DTB-described region
- **AND** the boot measurement log SHALL record `OpTeeShmBackend::DtbReserved { base, size }`

#### Scenario: Dynamic allocation fallback

- **GIVEN** a DTB with no `/reserved-memory/optee-shm` (or equivalent) node
- **WHEN** `ShmPool::initialize()` runs
- **THEN** the pool SHALL initialize in dynamic mode
- **AND** the first `ShmBlock` allocation SHALL issue `OPTEE_SMC_RPC_FUNC_ALLOC` to OP-TEE
- **AND** the boot measurement log SHALL record `OpTeeShmBackend::Dynamic`

#### Scenario: Pool exhaustion fails cleanly

- **GIVEN** a pool with `N` bytes total capacity already fully allocated
- **WHEN** the caller requests one more byte
- **THEN** `ShmPool::alloc` SHALL return `Err(TeeError::SharedMemoryExhausted)`
- **AND** the bridge SHALL NOT panic, leak, or corrupt previously-allocated blocks

### Requirement: RPC handling with documented allowlist

The bridge SHALL handle the OP-TEE RPC subset required for the `pta_invoke_tests` no-op smoke (allocation/free/foreign interrupt) and SHALL reject all other RPC subcommands with `OPTEE_SMC_RETURN_ENOTSUP`.

#### Scenario: Allowlisted RPCs are serviced

- **GIVEN** an OP-TEE TA invocation that triggers `OPTEE_SMC_RPC_FUNC_ALLOC` for an additional shared-memory block
- **WHEN** the SMC returns the RPC request
- **THEN** the bridge SHALL allocate from `ShmPool`, populate the response registers per the OP-TEE RPC convention, and re-issue the SMC to resume the invocation
- **AND** the same SHALL hold for `OPTEE_SMC_RPC_FUNC_FREE` and `OPTEE_SMC_RPC_FUNC_FOREIGN_INTR`

#### Scenario: Non-allowlisted RPCs are rejected

- **GIVEN** a TA invocation that issues an `OPTEE_SMC_RPC_FUNC_CMD` subcommand not on the bridge's allowlist (e.g. wait-for-keypress)
- **WHEN** the bridge sees the RPC return
- **THEN** the bridge SHALL respond with `OPTEE_SMC_RETURN_ENOTSUP` (`0xFFFFFFFD`)
- **AND** the TA invocation SHALL fail with a clean `TeeError::NotSupported` rather than hanging or panicking

### Requirement: Raw SMC dispatch with Arm SMC Calling Convention compliance

The `smallaios-arch-aarch64` crate SHALL expose an `unsafe fn smc_call(fid, a1..a6) -> SmcResult` implementing Arm DEN 0028C SMC dispatch, used by the OP-TEE bridge and available for future Secure-World callers.

#### Scenario: Register layout matches Arm DEN 0028C

- **WHEN** `smc_call(fid, a1, a2, a3, a4, a5, a6)` is invoked from EL1 or EL2
- **THEN** the inline asm SHALL place `fid` (as u64) in x0, `a1` in x1, ..., `a6` in x6
- **AND** the asm SHALL emit a single `smc #0` instruction
- **AND** on return the asm SHALL capture x0-x3 into the returned `SmcResult`
- **AND** the function SHALL be marked `unsafe` because SMC is a privileged instruction whose effects are platform-defined

#### Scenario: smc.rs only compiles on AArch64 with op-tee feature

- **GIVEN** a build invocation for x86-64 or RISC-V, or AArch64 without `--features op-tee`
- **THEN** `arch/aarch64/src/smc.rs` SHALL not be compiled (cfg-gated out)
- **AND** no `smc #0` instruction SHALL appear in the resulting binary

### Requirement: Documentation surface

The repository SHALL document the OP-TEE bridge architecture, its scope limits, and the Tegra Orin operational notes in `docs/op-tee-bridge.md`.

#### Scenario: Architecture and ABI documentation

- **THEN** `docs/op-tee-bridge.md` SHALL exist and SHALL contain the bridge architecture diagram, the GP TEE Client API → SMC FID mapping table, the SMC calling convention summary, and the OP-TEE OS commit-SHA pin
- **AND** `docs/boot-security-matrix.md` AArch64 row SHALL be updated: "TrustZone" column annotated "Bridge (op-tee-bridge-v1)", "OP-TEE" column annotated "Client-side (op-tee-bridge-v1)"

#### Scenario: Troubleshooting and platform notes

- **THEN** `docs/op-tee-bridge.md` SHALL include a "Troubleshooting" section covering: NotPresent on platforms where OP-TEE was expected (causes + fixes), shared-memory exhaustion (how to raise the DTB-reserved size), and unknown TA UUID (the TA isn't loaded by OP-TEE OS)
- **AND** the doc SHALL include a "Tegra Orin operational notes" section pinning the JetPack version where OP-TEE BL32 ships built-in vs requires a custom firmware build
