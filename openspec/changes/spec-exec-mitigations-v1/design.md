# Design — spec-exec-mitigations-v1

## Goal

Five trust boundaries × three architectures × the relevant speculation classes (v1, v2, v4, Meltdown, Retbleed, Spectre-BHB) yields a 5 × 3 × 6 = 90-cell mitigation matrix. The deliverable is a populated matrix plus the code/config changes implied by it. The matrix lives in the new `kernel-security` capability spec; the code changes are scattered across `arch/*/src/syscall.rs`, `kernel/src/cap.rs`, `.cargo/config.toml`, and `onnx-rt/`.

## Design decisions

### Decision 1: Single `kernel-security` capability spec, not per-arch specs

Trust boundaries are *cross-arch* concepts ("the syscall entry point", "the capability check") even though the mitigation instructions differ per ISA. A reviewer asking "is this kernel hardened against Spectre v2 at the op-dispatch boundary?" wants a single answer in a single document, with per-arch citations under each Requirement. The alternative — three specs (`x86-security`, `aarch64-security`, `riscv-security`) — fragments the audit trail and makes the safety case harder to assemble.

`aarch64-mte-pac-hardening-v1`'s `arch-aarch64-security` spec stays separate because *those* mitigations are CPU-feature-specific memory protections, not trust-boundary-specific speculation barriers. The two specs cross-reference each other.

### Decision 2: Compiler flags via Cargo feature, not blanket `RUSTFLAGS`

The chosen feature is `spec-exec` on `smallaios-kernel`, default-on for `--release` profiles. It pulls in arch-specific sub-features (`spec-exec-x86`, `spec-exec-aarch64`, `spec-exec-riscv`) selected by `cfg(target_arch)`. Each sub-feature gates an arch-specific `build.rs` that emits `cargo:rustc-link-arg` and `cargo:rustc-flags` entries.

Why not just set `RUSTFLAGS` globally in `.cargo/config.toml`: those flags apply to *every* dep including build-script-host code, which is undesirable (Retpoline thunks have no meaning when compiling a build script on a Mac laptop). The feature-gated `build.rs` path applies the flags only to the kernel and its arch HALs.

### Decision 3: IBRS on always-set, IBPB on syscall-entry only

Intel's recommendation for kernel mitigation:

- **IBRS (Indirect Branch Restricted Speculation)** — set in `IA32_SPEC_CTRL` on every kernel entry, cleared on kernel exit. Modern CPUs (Ice Lake+) support "Enhanced IBRS" which is sticky — set once, no per-entry overhead. SmallAIOS targets the modern x86 silicon assumption (Sapphire Rapids+ for inference servers), so we use Enhanced IBRS where available, fall back to per-entry IBRS where not (with a boot log line documenting the choice).
- **IBPB (Indirect Branch Predictor Barrier)** — flushes branch prediction state. Expensive (~hundreds of cycles). We emit it on syscall entry — every transition from "untrusted task graph" to "kernel" — but not on intra-kernel function calls. This is the same policy Linux uses for syscall entry.
- **STIBP (Single Thread Indirect Branch Predictors)** — relevant only with SMT. Cortex-A78AE has no SMT; modern Xeon variants do. We set STIBP if SMT is detected at boot.

### Decision 4: LFENCE placement — *after* capability check, *before* attacker-controlled load

The Spectre v1 pattern that matters in SmallAIOS:

```rust
// pseudocode, hand-rolled in arch/x86_64/src/syscall.rs
fn syscall_entry(args: &SyscallArgs) -> i64 {
    // 1. Decode syscall number + capability check
    let cap = check_capability(args.cap_handle)?;
    // 2. Speculative window: CPU may speculate through here even if check_capability returned an error
    lfence();  // <-- explicit fence, after the check
    // 3. Now-safe load from args
    let tensor = decode_tensor_handle(args.tensor)?;
    ...
}
```

Putting `lfence` *before* the capability check is wrong — it serializes the harmless decode but does nothing about the speculation past the check. The right placement is documented in this design doc and enforced by code review (no lint can catch placement automatically). We add a unit test that disassembles the syscall entry and asserts the `lfence` opcode appears between the check and the first attacker-controlled load.

### Decision 5: CSV2/CSV3 silicon check on aarch64 boot

Cortex-A78AE advertises hardware mitigation of branch-target-injection (Spectre v2) via `ID_AA64PFR0_EL1.{CSV2, CSV3}`. If the silicon reports `CSV2 ≥ 1` (hardware mitigated) and `CSV3 ≥ 1` (similar for cache-allocate side channels), the kernel can skip software-driven Retpoline-shaped mitigations. We read the register at boot; the result is logged and the build profile (`spec-exec-aarch64-hw` vs `spec-exec-aarch64-sw`) is selected accordingly.

If the silicon reports the lower levels (hypothetical future Orin-class part), the kernel boots with a warning and the software-mitigation path is taken (which adds some `csdb` insertions to the op-dispatch).

### Decision 6: SmallAIOS structural Meltdown immunity

Meltdown (CVE-2017-5754) exploits speculative loads through the user-space view of kernel pages. Linux's KPTI mitigation maps kernel pages out of the user-space page table.

SmallAIOS is a unikernel: there is no user-space page table. The "task graph" runs in the same address space as the kernel. There is nothing to leak across, because there is no privilege boundary at the page-table level. Spectre v1 and v2 still apply (those exploit speculation across *function-call* boundaries, not page-table boundaries), but Meltdown specifically does not.

We document this as a structural property — "Meltdown does not apply to the SmallAIOS unikernel address-space model" — rather than as a software mitigation. The DO-178C safety case explicitly cites the structural absence rather than the absence of an applied mitigation, because the two are different evidence kinds.

## Alternatives considered

### Alt A: Rely on compiler defaults, no kernel-side instrumentation

**Rejected.** Compiler defaults for `*-unknown-none` targets do not include Retpoline, SLH, or LFENCE-after-bounds-check. They never will (those flags add overhead, and the compiler's stance is "no_std users opt in"). We are the no_std user; we must opt in.

### Alt B: Disable indirect calls in the ONNX dispatcher (compile-time op switch)

**Considered, deferred.** Generating a giant `match op_id { 0 => Conv::dispatch, 1 => Gemm::dispatch, ... }` instead of a function-pointer table eliminates the indirect-branch attack surface entirely. The catch: the `match` compiles to a different shape per LLVM optimization (it may itself become an indirect jump via a jump table in `.rodata`, defeating the point). And the code-gen bloat is large for the ~150 ONNX op variants. We pick the function-pointer table + per-arch indirect-call hardening (BTI / Retpoline) as the better engineering trade. Revisit if a compiler analysis ever proves the `match` shape avoids the jump table.

### Alt C: Compile with `-mllvm -x86-speculative-execution-side-effect-suppression` instead of explicit fences

**Considered, on for builds that support it.** Modern LLVM has an SLH-class pass that inserts speculation barriers automatically based on a CPU-fence cost model. Output is broadly Retpoline-shaped but the placement is compiler-driven, not author-driven. Mitigation: enable on x86_64 builds where LLVM supports it cleanly; keep the *explicit* `lfence` in syscall entry as a backstop in case the LLVM pass misses it (defense in depth).

### Alt D: Defer until DO-178C tooling demands it

**Rejected** for the same reason as the MTE/PAC change. Speculative side channels are now a baseline mitigation expectation for any safety-critical OS, with or without explicit DO-178C tooling support.

## Risks

### Risk 1: IBPB overhead in inference-heavy workloads

IBPB costs ~100-500 cycles depending on the microarchitecture. ONNX-runtime workloads that do many small syscalls (e.g., tensor pool churn) could see measurable throughput drops. Mitigation: (a) benchmark before/after; (b) batch syscalls where possible (this is a separate optimization angle, but spec-exec mitigations highlight where it matters); (c) `spec-exec-ibpb-off` Cargo opt-out for performance-mode-only deployments that explicitly accept the residual Spectre v2 risk.

### Risk 2: LLVM flag drift between Rust toolchain versions

The `-mllvm -x86-speculative-load-hardening` flag has shifted shape across LLVM versions. The 2026-02-01 pinned toolchain bundles LLVM 19; we test on that version and pin the flag form in `.cargo/config.toml`. Toolchain upgrades go through a separate compatibility check.

### Risk 3: Per-arch coverage gaps

We have full coverage on aarch64 (silicon checked, barriers inserted, op-dispatch hardened) and x86_64 (compiler flags + explicit fences + MSR programming). RISC-V coverage is partial — the relevant CFI extensions are not yet ratified. Mitigation: document the gap in `docs/spec-exec-audit.md` and treat RISC-V Phase 4 as scaffolding to be filled in when Zicfilp lands. Production deployments on RISC-V should be flagged in the safety case as "speculation mitigations partial".

### Risk 4: False sense of security

Speculation mitigations are *not* a complete defense — side channels keep being discovered (Meltdown 2018, Spectre 2018, Foreshadow 2018, MDS 2019, CrossTalk 2020, Retbleed 2022, Downfall / Inception 2023, GhostRace 2024). This change covers the *known* classes; new classes will emerge. Mitigation: the `kernel-security` spec includes a "review trigger" — every new CVE in this class triggers a review of the mitigation matrix, tracked as an OpenSpec change.

### Risk 5: Compiler-emitted barriers missing in `panic_handler` or other compiler-magic functions

The Rust compiler may emit indirect calls in `panic_handler`, `core::ops::Drop`, async state machines, etc. that are not under our direct control. Mitigation: the unit tests disassemble the linked kernel binary and assert (a) no unprotected indirect branches in `arch/*/src/syscall.rs`, (b) every Retpoline thunk is the expected shape on x86_64, (c) every aarch64 indirect branch has a BTI landing pad.

## Build/CI surface

- New Cargo feature `spec-exec` on `smallaios-kernel` (default-on for release), with arch-specific sub-features.
- New module `arch/x86_64/src/security/spec_exec.rs` — `init()` programs IBRS, IBPB-on-entry trampolines, LFENCE placement.
- New module `arch/aarch64/src/security/spec_exec.rs` — CSV2/CSV3 silicon check, CSDB / DSB SY insertion points.
- Modify `arch/*/src/syscall.rs` to insert explicit barriers at trust boundaries.
- New `docs/spec-exec-audit.md` with the full trust-boundary × arch × attack-class matrix.
- New CI advisory job `spec-exec-disasm-audit` — runs `objdump -d` on the built kernel and asserts the expected barrier opcodes appear at the expected addresses. Catches regressions where someone refactors syscall entry and accidentally drops the `lfence` / `csdb`.

## What this change explicitly does NOT do

- Does not modify cryptographic code paths — they have their own (already-applied) constant-time discipline.
- Does not change syscall ABI — barriers are CPU-internal, invisible to callers.
- Does not enable SMT-specific mitigations on Cortex-A78AE (no SMT in silicon).
- Does not patch any specific CVE individually — it covers classes of attacks via class-level mitigations.
- Does not introduce a JIT — Spectre-RSB / Spectre-BTI variants that need a JIT remain structurally absent.
