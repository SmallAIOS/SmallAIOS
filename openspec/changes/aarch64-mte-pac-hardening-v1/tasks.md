# Tasks — aarch64-mte-pac-hardening-v1

## 0. Hardware + toolchain verification (prereq)

- [ ] 0.1 Confirm Orin NX silicon advertises MTE + PAC: on a JetPack 6 host, run `cat /proc/cpuinfo | grep -E "(features|isar)" | head -2` and `lscpu | grep Flags`. Expected: `mte`, `paca`, `pacg` in the features list. Paste in the PR description.
- [ ] 0.2 Confirm `ID_AA64ISAR1_EL1.{APA, GPA}` and `ID_AA64PFR1_EL1.MTE` indicate the expected ARMv8.5-A levels by reading them from EL1 via a small Rust probe binary (run under KVM on Orin via the existing `run-jetson-kvm` recipe).
- [ ] 0.3 Confirm the pinned Rust toolchain `nightly-2026-02-01` accepts `-C target-feature=+mte,+pauth` and `-C codegen-options=branch-protection=pac-ret` for `aarch64-unknown-uefi` and `aarch64-unknown-none`. Document in `docs/aarch64-security.md`.
- [ ] 0.4 Confirm QEMU `qemu-system-aarch64` ≥ 7.0 supports `-cpu cortex-a78,mte=on,pauth=on` (it does; document the exact version requirement).

## 1. Phase 1 — PAC (Pointer Authentication)

### 1a. Module scaffolding

- [ ] 1.1 Create `arch/aarch64/src/security/mod.rs` with public API: `pub fn init()`, `pub fn pac_enabled() -> bool`, `pub fn mte_enabled() -> bool`. Doc-comment cites the ARMv8.5-A spec sections.
- [ ] 1.2 Create `arch/aarch64/src/security/pac.rs` with `pub fn install_keys()` and `pub fn enable_in_sctlr()`. Inline asm via `core::arch::asm!` for the `MSR APIAKeyHi_EL1, x0` family.
- [ ] 1.3 Add `mte-pac` Cargo feature to `arch/aarch64/Cargo.toml`. Doc-comment distinguishes it from MTE-only / PAC-only sub-features.
- [ ] 1.4 Default-on for `tegra234` builds via `default-features = ["mte-pac"]` in the relevant Cargo.toml stanza.

### 1b. Key derivation + install

- [ ] 1.5 Implement `pac::derive_keys_from_trng()` — reads `RNGSR_EL0` (or falls back to a CNTPCT-mixed Xoshiro PRNG with a boot-time warning if TRNG is unavailable) and produces five 128-bit keys.
- [ ] 1.6 Implement `pac::install_keys(keys: &PacKeys)` — writes all five key pairs (`APIAKeyHi_EL1`/`Lo`, `APIBKeyHi_EL1`/`Lo`, `APDAKeyHi_EL1`/`Lo`, `APDBKeyHi_EL1`/`Lo`, `APGAKeyHi_EL1`/`Lo`) via inline asm.
- [ ] 1.7 Implement `pac::enable_in_sctlr()` — read `SCTLR_EL1`, set `EnIA | EnIB | EnDA | EnDB`, write back, `isb`. Clear them first if UEFI left stale bits set.
- [ ] 1.8 Add `mte-pac-deterministic` Cargo feature that uses fixed keys (printed at boot) for development debugger use. Off by default.

### 1c. Compiler-level branch protection

- [ ] 1.9 Add `[target.'cfg(target_arch = "aarch64")']` rustflags `-C codegen-options=branch-protection=pac-ret,bti` to `.cargo/config.toml` (or set via `RUSTFLAGS` in the `mte-pac`-feature build job). Verify with `objdump -d` that emitted prologues contain `pacibsp` / `autibsp`.
- [ ] 1.10 Verify BTI is also on (compiler should emit `bti c` at function entries) — this is free baseline hardening.

### 1d. Capability handle signing via APDA

- [ ] 1.11 Modify `kernel/src/cap.rs` so that capability handles are signed with `pacda` on construction using `(resource_type as u64)` as the modifier; signature is verified with `autda` on dereference.
- [ ] 1.12 Audit all sites that construct/dereference capabilities to use the new signed accessors. Run the existing capability tests to confirm no regression.

### 1e. ONNX op dispatch signing via APGA

- [ ] 1.13 Modify `onnx-rt`'s op-dispatch table so each function pointer is signed with `pacga` (modifier = op-name hash) at table init; the dispatch call site authenticates before calling.
- [ ] 1.14 Run the existing ONNX runtime tests to confirm no regression.

### 1f. PAC boot wiring

- [ ] 1.15 Modify `arch/aarch64/src/main.rs` (and `main_uefi.rs`) to call `security::pac::install_keys()` + `security::pac::enable_in_sctlr()` after exception vectors are installed and before any function call past the boot stub. Print `[pac] keys installed, branch-protection active` on success.
- [ ] 1.16 PAC fault path: `autia` failure raises a PAC-fault exception (`ESR_EL1.EC = 0x1C`). Add a handler in `interrupts.rs` that logs `[pac-fault] pc=…` and panics.
- [ ] 1.17 Test on QEMU `cortex-a78,pauth=on`: assert the kernel boots, prints the PAC banner, runs the existing test suite to completion.

## 2. Phase 2 — MTE (Memory Tagging Extension)

### 2a. MTE module + allocator hook

- [ ] 2.1 Create `arch/aarch64/src/security/mte.rs` with `pub fn enable_sync()`, `pub fn handle_fault(esr, far) -> !`, and the `tag_alloc`/`tag_dealloc` allocator helpers.
- [ ] 2.2 Modify `kernel/src/mem/heap.rs` (or wherever `GlobalAlloc` is implemented) to call `mte::tag_alloc(ptr, size)` after every successful allocation and `mte::tag_dealloc(ptr, size)` before every deallocation. Gated behind `#[cfg(feature = "mte-pac")]`.
- [ ] 2.3 Implement `mte::tag_alloc(ptr, size)` — pick a random tag from 1-15 via the PAC-share TRNG, embed it in `ptr` bits 56-59, write the tag to all granules via the `stg` instruction.
- [ ] 2.4 Benchmark on Orin: run the existing `bench/` ONNX inference benchmarks with and without MTE. Gate merge on <5% steady-state throughput regression on the representative model.

### 2b. SCTLR enable + fault handler

- [ ] 2.5 Implement `mte::enable_sync()` — write `SCTLR_EL1.TCF = 0b01` (sync MTE), `SCTLR_EL1.ATA = 1` (enable MTE in EL1 normal). `isb` to commit.
- [ ] 2.6 Modify `arch/aarch64/src/interrupts.rs` Data Abort handler to dispatch `ESR_EL1.EC == 0x25 && FSC == 0x11` (sync tag-check fault) to `security::mte::handle_fault`.
- [ ] 2.7 Implement `mte::handle_fault` — read `FAR_EL1`, `ELR_EL1`, `ESR_EL1`; decode the address tag (bits 56-59) and the granule tag (via `ldg` on `FAR_EL1`); log `[mte-fault] pc=… addr=… tag_pointer=N tag_memory=M`; panic.
- [ ] 2.8 Add an `mte-watchdog` Cargo feature that, in lieu of panic, signals the hardware watchdog and emits a coredump-shaped serial dump before halt.

### 2c. MTE async opt-out

- [ ] 2.9 Add an `mte-async` Cargo feature that programs `SCTLR_EL1.TCF = 0b10` (async mode) instead. Document the precision trade-off in `docs/aarch64-security.md`.

### 2d. MTE boot wiring

- [ ] 2.10 Modify `arch/aarch64/src/main.rs` / `main_uefi.rs` to call `security::mte::enable_sync()` after `interrupts::init()` and *after* the global allocator is installed but before any other allocation happens.
- [ ] 2.11 Print `[mte] sync tag-check enabled` on success.

### 2e. Tests + smoke

- [ ] 2.12 Add a `just mte-fault-test` recipe that runs a tiny test binary under QEMU `-cpu cortex-a78,mte=on` which deliberately reads an allocation through a stale (wrong-tagged) pointer, asserts the fault handler fires with the expected structured log.
- [ ] 2.13 Add an `aarch64-mte-pac-smoke` CI job (advisory initially) that runs the smoke under QEMU.
- [ ] 2.14 On Orin NX hardware: capture the boot output showing `[pac] keys installed` and `[mte] sync tag-check enabled`, the existing ONNX test suite completion, and a deliberate-tag-mismatch fault triggered by an injected test syscall. Paste in the PR description.

## 3. Docs

- [ ] 3.1 Create `docs/aarch64-security.md` covering: MTE/PAC overview, what each key/feature catches, the exact compiler flags + `cargo build` invocation, the per-boot key derivation, the fault decode procedure (interpreting `ESR_EL1.EC`, `FAR_EL1`, granule tag), the Cargo sub-features (`mte-async`, `mte-watchdog`, `mte-pac-deterministic`), and DO-178C alignment notes.
- [ ] 3.2 Update `docs/architecture.md` Layer 2 (HAL) section to call out hardware memory-safety on aarch64.
- [ ] 3.3 Update `CLAUDE.md` "Current state" to note MTE+PAC active on `tegra234` builds.

## 4. Verify + archive

- [ ] 4.1 Run `openspec validate aarch64-mte-pac-hardening-v1 --strict`.
- [ ] 4.2 Squash-merge both phases.
- [ ] 4.3 Open archive PR moving the change to `openspec/changes/archive/YYYY-MM-DD-aarch64-mte-pac-hardening-v1`.
