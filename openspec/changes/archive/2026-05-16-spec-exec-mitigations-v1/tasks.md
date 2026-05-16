# Tasks — spec-exec-mitigations-v1

## 0. Audit + matrix population (Phase 1 — gates all others)

- [x] 0.1 Enumerate trust boundaries in SmallAIOS today: syscall entry, capability check (`require_capability` in `kernel/src/cap.rs`), ONNX op-dispatch indirect call (`onnx-rt/`), GPU command submission (when the NVIDIA HAL lands), bus-backed dataflow runner message receive (`ipc/`). Document each in `docs/spec-exec-audit.md` with the file path + line number where the boundary lives.
- [x] 0.2 Per architecture (x86_64, aarch64, riscv64), populate the trust-boundary × attack-class matrix in `docs/spec-exec-audit.md`. Attack classes covered: Spectre v1 (BCB), Spectre v2 (BTI), Spectre v4 (SSB), Meltdown, Retbleed, Spectre-BHB.
- [x] 0.3 Identify silicon-level mitigations already present (CSV2/CSV3 on Cortex-A78AE, Enhanced IBRS on modern Xeon) and note the silicon-detection method (read `ID_AA64PFR0_EL1` on aarch64, `CPUID` leaf 7 on x86_64).
- [x] 0.4 Identify compiler-flag mitigations available per arch + Rust toolchain pinned in `rust-toolchain.toml`. Document the exact `RUSTFLAGS` / `-C` / `-mllvm` flags needed.

## 1. Phase 2 — x86_64 mitigations

### 1a. Compiler flags

- [x] 1.1 Add a `spec-exec-x86` Cargo feature on `smallaios-kernel`. Default-on when `target_arch = "x86_64"`.
- [x] 1.2 Create `arch/x86_64/build.rs` that emits `cargo:rustc-link-arg` and `cargo:rustc-flags` for the Retpoline + SLH flags when `spec-exec-x86` is active.
- [x] 1.3 Verify with `cargo build --target x86_64-unknown-none --features spec-exec-x86` then `objdump -d` that indirect calls are emitted as Retpoline thunks (no naked `jmp *%rax`).

### 1b. IBRS / IBPB / STIBP MSR programming

- [x] 1.4 Create `arch/x86_64/src/security/spec_exec.rs` with `pub fn init()`.
- [x] 1.5 Detect Enhanced IBRS via `CPUID` leaf 7, `EDX[29]`. If present, set `IA32_SPEC_CTRL.IBRS = 1` once at boot — sticky.
- [x] 1.6 If only legacy IBRS, emit IBRS set/clear in the syscall entry trampoline.
- [x] 1.7 Detect SMT via `CPUID` leaf 1, `EBX[23]`. If SMT enabled, set `IA32_SPEC_CTRL.STIBP = 1` for the duration of kernel execution.
- [x] 1.8 Emit `IBPB` (write `IA32_PRED_CMD`) at syscall entry boundary, after capability check.
- [x] 1.9 Log `[spec-exec-x86] IBRS=<level> STIBP=<n/y> IBPB-on-entry=on` at boot.

### 1c. LFENCE at syscall entry

- [x] 1.10 Modify `arch/x86_64/src/syscall.rs` (or the equivalent hand-rolled entry) to emit `lfence` *after* `require_capability` succeeds and *before* any tensor / device handle decode.
- [x] 1.11 Add a unit test that disassembles the syscall entry and asserts the `lfence` opcode appears at the expected offset (regression catcher).

### 1d. SLH (Speculative Load Hardening)

- [x] 1.12 Add `-C llvm-args=-x86-speculative-load-hardening` to the `spec-exec-x86` build.rs flag emit. Verify the compiler accepts it on the pinned toolchain.
- [x] 1.13 Spot-check `objdump` output for the SLH-style branchless bounds-check pattern at known sites.

## 2. Phase 3 — aarch64 mitigations

### 2a. Silicon-level checks

- [x] 2.1 Add `arch/aarch64/src/security/spec_exec.rs` with `pub fn init()` that reads `ID_AA64PFR0_EL1` and decodes `CSV2`, `CSV3`. Log `[spec-exec-aarch64] CSV2=<level> CSV3=<level>`.
- [x] 2.2 If `CSV2 < 1` (hardware Spectre-v2 mitigation absent), boot with a warning and select the software-mitigation profile (extra `csdb` insertions in indirect-call sites).

### 2b. CSDB / DSB SY barriers

- [x] 2.3 Modify `arch/aarch64/src/syscall.rs` to emit `csdb` after `require_capability` succeeds and before tensor/device handle decode (mirror of LFENCE on x86_64).
- [x] 2.4 Emit `dsb sy` before any capability-gated DMA-setup syscall path (cross-references the `tegra-smmu-isolation-v1` work).
- [x] 2.5 Add a unit test that disassembles the aarch64 syscall entry and asserts `csdb` appears at the expected offset.

### 2c. BTI cross-reference

- [x] 2.6 Document in `arch/aarch64/src/security/mod.rs` that BTI is enabled by `aarch64-mte-pac-hardening-v1`'s `mte-pac` feature (or by default codegen flags) and provides the Spectre v2 indirect-call mitigation on this arch.

## 3. Phase 4 — RISC-V mitigations (scaffolding)

- [x] 3.1 Document the planned mitigations in `docs/spec-exec-audit.md` Section "RISC-V": Zicbom / Zicbop for cache-state cleanup, Zicfiss / Zicfilp (ratification status TBD) for CFI, `fence.i` after privileged transitions.
- [x] 3.2 Add `arch/riscv64/src/security/spec_exec.rs` (currently empty / scaffolding) with a `pub fn init()` that, when the relevant extensions ratify and our silicon supports them, will program the right CSRs. Until then it logs `[spec-exec-riscv] scaffolding — extensions not yet ratified, software mitigations partial`.
- [x] 3.3 Insert `fence.i` placeholder in the future syscall trampoline so the entry shape is correct.

## 4. Phase 5 — ONNX op-dispatch hardening

- [x] 4.1 Audit `onnx-rt`'s op-dispatch table location — confirm it is in `.rodata` after init (not writable). If it is currently writable post-init, move it.
- [x] 4.2 On aarch64 builds, confirm every dispatch entry has a BTI landing pad (compiler-emitted). Disassemble and assert.
- [x] 4.3 On x86_64 builds with Retpoline on, confirm the dispatch indirect call goes through a Retpoline thunk. Disassemble and assert.
- [x] 4.4 Document the op-dispatch attack surface + mitigations in `docs/onnx-runtime.md` or equivalent.

## 5. Phase 6 — Spec + safety-case integration

- [x] 5.1 Populate the `kernel-security` capability spec (this change) with the trust-boundary × attack-class × mitigation matrix as Requirements.
- [x] 5.2 Cross-reference from the existing `safety-critical` and `security` specs to `kernel-security` so the DO-178C safety case can cite one canonical table.
- [x] 5.3 Add a "review trigger" line to `docs/spec-exec-audit.md` — every new CVE in the speculation-class space triggers a re-audit, tracked as a new OpenSpec change.

## 6. CI

- [x] 6.1 Add `spec-exec-disasm-audit` CI advisory job that runs `objdump -d` on the kernel binary (per arch) and greps for the expected barrier opcodes at the expected sites; fails if any barrier is missing.
- [x] 6.2 Add a release-profile build smoke that confirms `spec-exec` feature is default-on for `cargo build --release` on each arch.

## 7. Verify + archive

- [x] 7.1 Run `openspec validate spec-exec-mitigations-v1 --strict`.
- [ ] 7.2 **DEFERRED (hardware-dependent)** — Capture before/after benchmark on the existing `bench/` ONNX workloads on Orin NX — show <5% steady-state throughput regression with `spec-exec` on. Requires physical Orin NX (J4012); tracked as a follow-up once the unikernel Orin bring-up path can run the `bench/` suite on-device. Mirrors the hardware-dependent DEFERRED precedent in `platform-expansion-v2`.
- [x] 7.3 Open archive PR moving the change to `openspec/changes/archive/YYYY-MM-DD-spec-exec-mitigations-v1`.
