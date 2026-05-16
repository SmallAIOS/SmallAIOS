# kernel-security Specification

## Purpose
TBD - created by archiving change spec-exec-mitigations-v1. Update Purpose after archive.
## Requirements
### Requirement: Speculative-execution mitigations at syscall entry

The kernel SHALL insert explicit speculation barriers at every syscall entry point, after the capability check succeeds and before any decode of attacker-controlled syscall arguments, for every architecture target.

#### Scenario: x86_64 emits LFENCE after capability check

- **GIVEN** a SmallAIOS kernel built for `x86_64-unknown-none` with `--features spec-exec-x86` (default-on for the release profile)
- **WHEN** a syscall enters the trampoline in `arch/x86_64/src/syscall.rs`
- **THEN** the trampoline SHALL invoke `require_capability` first
- **AND** SHALL emit an `lfence` instruction immediately after a successful capability check and before any load that uses caller-supplied addresses
- **AND** a disassembly-level CI audit (`spec-exec-disasm-audit`) SHALL assert the `lfence` opcode appears at the expected offset

#### Scenario: aarch64 emits CSDB after capability check

- **GIVEN** a SmallAIOS kernel built for `aarch64-unknown-uefi` with `--features tegra234,spec-exec-aarch64`
- **WHEN** a syscall enters the trampoline in `arch/aarch64/src/syscall.rs`
- **THEN** the trampoline SHALL emit a `csdb` (Consumption of Speculative Data Barrier) instruction immediately after the capability check
- **AND** for capability-gated DMA-setup syscall paths the trampoline SHALL ALSO emit `dsb sy` before transferring control into the DMA driver

#### Scenario: RISC-V emits fence.i scaffolding ahead of ratified CFI

- **GIVEN** a SmallAIOS kernel built for `riscv64gc-unknown-none-elf` (scaffolding only — syscall workloads not yet exercised)
- **WHEN** the syscall trampoline placeholder runs
- **THEN** the trampoline SHALL emit `fence.i` at the privileged-transition boundary so the entry shape is correct for future Zicfilp / Zicfiss adoption
- **AND** a boot log line SHALL note `[spec-exec-riscv] scaffolding — extensions not yet ratified, software mitigations partial`

### Requirement: Speculation barriers at ONNX op-dispatch boundaries

The ONNX runtime's operator dispatch table SHALL be hardened against speculative branch-target injection on every architecture, using compiler-emitted mitigations and runtime placement in read-only memory.

#### Scenario: x86_64 op dispatch uses Retpoline thunks

- **GIVEN** a `--features spec-exec-x86` release build
- **WHEN** the ONNX runtime dispatches to an operator via its function-pointer table
- **THEN** the emitted indirect call SHALL go through a Retpoline thunk rather than a naked `jmp *%reg` / `call *%reg`
- **AND** disassembly of the dispatch site SHALL show the expected `call 0(%rsp); int3` Retpoline shape
- **AND** the op-dispatch table SHALL reside in `.rodata` after init — writes to the table after init SHALL fault on the page-table permission bits

#### Scenario: aarch64 op dispatch lands on BTI guards

- **GIVEN** a `--features spec-exec-aarch64` release build (BTI enabled by the `aarch64-mte-pac-hardening-v1` change's codegen flags)
- **WHEN** the ONNX runtime dispatches to an operator via its function-pointer table
- **THEN** every dispatch target function SHALL begin with a `bti c` (or `bti jc`) landing-pad instruction
- **AND** a branch to an address without a BTI landing pad SHALL raise a branch-target exception

#### Scenario: Op-dispatch table is read-only after init

- **GIVEN** the ONNX runtime initialization sequence
- **WHEN** the runtime finishes installing operator function pointers into the dispatch table
- **THEN** the kernel SHALL ensure the table resides in a `.rodata` section (or equivalent read-only memory region)
- **AND** any post-init write to the table from any code path SHALL fault

### Requirement: x86_64 IBRS / IBPB / STIBP MSR programming

The x86_64 kernel SHALL detect the Spectre-class hardware mitigations at boot via `CPUID`, configure them appropriately, and emit `IBPB` at the syscall-entry boundary.

#### Scenario: Enhanced IBRS detected and set once

- **GIVEN** an x86_64 platform that reports Enhanced IBRS support via `CPUID` leaf 7 `EDX[29]`
- **WHEN** `spec_exec::init()` runs during boot
- **THEN** the kernel SHALL set `IA32_SPEC_CTRL.IBRS = 1` once and rely on its sticky semantics — no per-entry IBRS toggling
- **AND** the boot log SHALL include `[spec-exec-x86] IBRS=enhanced`

#### Scenario: Legacy IBRS toggled per syscall entry

- **GIVEN** an x86_64 platform that reports only legacy IBRS (no Enhanced IBRS in `CPUID`)
- **WHEN** the syscall trampoline runs
- **THEN** the trampoline SHALL set `IA32_SPEC_CTRL.IBRS = 1` at entry and clear it at exit
- **AND** the boot log SHALL include `[spec-exec-x86] IBRS=legacy-per-entry` with a warning that latency is higher

#### Scenario: IBPB emitted on every syscall entry

- **GIVEN** any x86_64 `--features spec-exec-x86` build
- **WHEN** a syscall enters the trampoline
- **THEN** the trampoline SHALL write to `IA32_PRED_CMD` to emit an `IBPB` after capability check
- **AND** an opt-out Cargo feature `spec-exec-ibpb-off` SHALL skip the `IBPB` for performance-mode-only deployments that explicitly accept the residual Spectre v2 risk — its documentation SHALL state the trade-off

### Requirement: aarch64 silicon-level mitigation detection

The aarch64 kernel SHALL detect the Cortex-A78AE-class silicon mitigations (CSV2 / CSV3) via system register reads at boot and SHALL select the appropriate mitigation profile.

#### Scenario: Silicon advertises Spectre-v2 hardware mitigation

- **GIVEN** an aarch64 platform reporting `ID_AA64PFR0_EL1.CSV2 ≥ 1` (Cortex-A78AE and most ARMv8.5-A+ silicon)
- **WHEN** `spec_exec::init()` runs during boot
- **THEN** the kernel SHALL select the hardware-mitigation profile — no software Retpoline-shaped insertion in indirect-call sites
- **AND** the boot log SHALL include `[spec-exec-aarch64] CSV2=<n> CSV3=<n> profile=hardware-mitigated`

#### Scenario: Silicon downgrades CSV2 (warn + fallback)

- **GIVEN** a hypothetical future aarch64 platform reporting `ID_AA64PFR0_EL1.CSV2 = 0` (no hardware mitigation)
- **WHEN** `spec_exec::init()` runs
- **THEN** the kernel SHALL log a warning and SHALL select the software-mitigation profile (extra `csdb` insertions at indirect-call sites)
- **AND** the boot SHALL continue — the kernel SHALL NOT panic, since the software path is functional, just slower

### Requirement: Meltdown structural immunity documented as a safety-case property

The unikernel single-address-space architecture SHALL be documented as structurally immune to Meltdown (CVE-2017-5754), and the DO-178C safety case SHALL cite this structural property rather than an applied software mitigation.

#### Scenario: Safety case lists Meltdown as structurally absent

- **GIVEN** the `kernel-security` spec and `docs/spec-exec-audit.md`
- **WHEN** a reviewer inspects the mitigation matrix for Meltdown
- **THEN** the entry SHALL state "structurally absent — no user/kernel page-table split exists in the unikernel address-space model"
- **AND** the safety case SHALL cite the unikernel architectural model (`docs/architecture.md`) as the evidence
- **AND** no software mitigation (KPTI-equivalent) SHALL be applied because none is required

#### Scenario: A future hypervisor / multi-AS change re-opens the question

- **GIVEN** a future SmallAIOS change introducing a user/kernel address-space split (e.g., a hypervisor mode)
- **THEN** the change SHALL be required to re-evaluate Meltdown applicability and SHALL add the relevant mitigation if applicable
- **AND** the `kernel-security` spec SHALL be updated alongside that change

### Requirement: Trust-boundary mitigation audit matrix

The `kernel-security` spec SHALL include a complete trust-boundary × architecture × attack-class matrix populated for every supported architecture, maintained in `docs/spec-exec-audit.md`, and refreshed on every new CVE in the speculative-execution class.

#### Scenario: Matrix covers all five trust boundaries

- **GIVEN** the `docs/spec-exec-audit.md` audit document
- **WHEN** a reviewer inspects it
- **THEN** the matrix SHALL include rows for: (a) syscall entry, (b) capability check, (c) ONNX op-dispatch indirect call, (d) GPU command submission, (e) bus-backed dataflow runner message receive
- **AND** for each row, the matrix SHALL have a column per supported architecture (x86_64, aarch64, riscv64) and a sub-column per attack class (Spectre v1, v2, v4, Meltdown, Retbleed, Spectre-BHB)
- **AND** each cell SHALL state the applied mitigation (compiler flag, kernel-emitted instruction, structural absence, or "partial — see notes")

#### Scenario: New CVE in the class triggers a re-audit

- **GIVEN** a new CVE published in the speculative-execution attack class
- **WHEN** a maintainer becomes aware of it
- **THEN** they SHALL open a new OpenSpec change to re-audit the affected boundaries
- **AND** the `docs/spec-exec-audit.md` matrix SHALL be updated as part of that change
- **AND** any newly-needed mitigation SHALL be added as a delta to the `kernel-security` spec

