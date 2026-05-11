## ADDED Requirements

### Requirement: Pointer Authentication (PAC) enabled on Tegra234 builds

The kernel SHALL enable ARMv8.5-A Pointer Authentication on every Cortex-A78AE-class platform (default-on for `--features tegra234`), installing five per-boot keys derived from a hardware RNG, and SHALL refuse to boot if the silicon does not advertise PAC support.

#### Scenario: PAC keys installed at boot

- **GIVEN** a SmallAIOS kernel built with `--features tegra234,mte-pac` running on an Orin NX device
- **WHEN** the boot sequence reaches `security::init()`
- **THEN** the kernel SHALL read `ID_AA64ISAR1_EL1.{APA, GPA}` to confirm PAC support — if either field is zero the kernel SHALL panic with `[pac] silicon does not support PAC`
- **AND** the kernel SHALL derive five 128-bit keys (`APIA`, `APIB`, `APDA`, `APDB`, `APGA`) from the hardware TRNG (`RNGSR_EL0`) on platforms that advertise it
- **AND** the kernel SHALL fall back to a `CNTPCT_EL0`-mixed PRNG with a boot-time warning if TRNG is unavailable
- **AND** the kernel SHALL write all five keys to their `*KeyHi_EL1` / `*KeyLo_EL1` system registers and SHALL set `SCTLR_EL1.{EnIA, EnIB, EnDA, EnDB}` to enable PAC instructions in EL1
- **AND** the kernel SHALL print a structured boot line of the form `[pac] keys installed, branch-protection=pac-ret active`

#### Scenario: Return-address corruption is trapped

- **GIVEN** a `--features mte-pac` build with branch-protection compiled in
- **WHEN** a stack-frame corruption causes a function's signed return address to fail authentication
- **THEN** the CPU SHALL raise a PAC fault exception (`ESR_EL1.EC = 0x1C`)
- **AND** the kernel SHALL log a structured line `[pac-fault] pc=<addr> elr=<addr> reason=auth-failure` and SHALL panic — the corrupted control flow SHALL NOT execute

#### Scenario: Capability handle integrity via APDA signing

- **GIVEN** the capability system in `kernel/src/cap.rs` building with `--features mte-pac`
- **WHEN** a capability handle is constructed
- **THEN** the kernel SHALL sign the handle with `pacda` using the `ResourceType` discriminant as the PAC modifier
- **AND** when the handle is dereferenced the kernel SHALL authenticate the signature with `autda`
- **AND** any modification to the handle (including a `ResourceType` flip via a buggy `transmute`) SHALL cause `autda` to fail and SHALL trap before the handle can be used for a syscall authorization decision

### Requirement: Memory Tagging Extension (MTE) enabled in synchronous mode on Tegra234

The kernel SHALL enable ARMv8.5-A Memory Tagging Extension in synchronous mode on every Cortex-A78AE-class platform (default-on for `--features tegra234`), with per-allocation random tags assigned by the global allocator and a fault handler that converts mismatches into structured panics.

#### Scenario: Allocator tags every allocation

- **GIVEN** a kernel built with `--features mte-pac` (sync MTE, the default mode)
- **WHEN** any code path calls into the global allocator (`GlobalAlloc::alloc`) and the allocation succeeds
- **THEN** the allocator SHALL pick a random 4-bit tag from the range 1-15 (zero reserved)
- **AND** the allocator SHALL write that tag to every 16-byte granule of the allocation via the `stg` instruction
- **AND** the allocator SHALL embed the tag in bits 56-59 of the returned pointer so subsequent loads/stores via that pointer carry the matching tag
- **AND** symmetric `dealloc` SHALL clear the granule tags (write tag-zero) so a stale pointer from a use-after-free cannot match

#### Scenario: Tag mismatch on load/store is trapped

- **GIVEN** a sync-MTE build where the allocator has tagged a buffer with tag `N`
- **WHEN** code accesses that buffer via a pointer whose embedded tag is `M ≠ N` (a use-after-free or off-by-one OOB)
- **THEN** the CPU SHALL raise a Data Abort with `ESR_EL1.EC = 0x25` and `FSC = 0x11` (tag-check fault) at the offending instruction
- **AND** the kernel fault handler SHALL read `FAR_EL1`, `ELR_EL1`, the address tag from bits 56-59 of `FAR_EL1`, and the granule tag via `ldg`
- **AND** the handler SHALL log a structured line `[mte-fault] pc=<addr> addr=<addr> tag_pointer=<N> tag_memory=<M>`
- **AND** the handler SHALL panic — execution SHALL NOT continue past the offending access

#### Scenario: Safety-critical builds route MTE faults to the watchdog

- **GIVEN** a kernel built with `--features mte-pac,mte-watchdog`
- **WHEN** an MTE fault fires
- **THEN** instead of panicking immediately the kernel SHALL signal the hardware watchdog
- **AND** the kernel SHALL emit a coredump-shaped serial dump containing the structured fault info plus a stack trace
- **AND** the kernel SHALL then halt — the watchdog reset SHALL be the recovery path

#### Scenario: Async-mode opt-out for non-safety-critical builds

- **GIVEN** a kernel built with `--features mte-pac,mte-async`
- **WHEN** `mte::enable_sync()` runs
- **THEN** the kernel SHALL instead write `SCTLR_EL1.TCF = 0b10` to enable async tag-check
- **AND** the fault may be deferred to the next exception entry rather than raised at the offending instruction (with a corresponding loss of fault-PC precision)
- **AND** documentation in `docs/aarch64-security.md` SHALL warn that async mode is intended for non-safety-critical performance-sensitive builds

### Requirement: PAC/MTE boot wiring ordering and idempotence

The `security::init` sequence SHALL run in a well-defined order with respect to interrupt vector installation and global allocator setup, and SHALL be idempotent in the sense that re-entering boot from a watchdog-induced reset converges to the same enabled state.

#### Scenario: Ordering — interrupts first, then PAC, then allocator, then MTE

- **GIVEN** the boot sequence in `arch/aarch64/src/main.rs` (or `main_uefi.rs`)
- **WHEN** the kernel enters `kernel_main`
- **THEN** the sequence SHALL be: (1) install exception vectors, (2) call `security::pac::install_keys()` + `security::pac::enable_in_sctlr()`, (3) install the global allocator, (4) call `security::mte::enable_sync()` (or `enable_async()` with the `mte-async` feature), (5) proceed to platform driver init
- **AND** no function call past the boot stub SHALL execute without PAC return-address signing active
- **AND** no allocator call SHALL execute without MTE tagging active

#### Scenario: UEFI-residual SCTLR state is overwritten, not inherited

- **GIVEN** an Orin NX boot via UEFI firmware that may have programmed its own PAC/MTE bits in `SCTLR_EL1` before `ExitBootServices`
- **WHEN** `security::init` runs
- **THEN** the kernel SHALL first clear the relevant `SCTLR_EL1` bits (`TCF`, `ATA`, `EnIA`, `EnIB`, `EnDA`, `EnDB`) to a known-zero baseline
- **AND** the kernel SHALL then install its own keys and SHALL re-enable the bits per its own policy — UEFI's prior choices SHALL NOT carry over
