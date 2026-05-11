# spec-exec-mitigations-v1

## Summary

Speculative-execution side channels — Spectre v1 (bounds-check bypass), Spectre v2 (branch-target injection), Spectre v4 (speculative store bypass), Meltdown (rogue data-cache load), Retbleed (return-address mispredict on AMD/Intel) — have reshaped what a "correct" trust boundary means on modern out-of-order CPUs. Compiler-default mitigations exist on every architecture SmallAIOS targets (x86_64, aarch64, riscv64), but they are *not all on* by default in `#![no_std]` Rust profiles, and the kernel-side discipline of inserting explicit serialization at trust boundaries (syscall entry, capability checks, indirect calls into ONNX op dispatch) is a code-author responsibility that is currently uncovered in our spec set.

This change audits SmallAIOS's trust boundaries, adds per-arch speculation barriers at the documented points (syscall entry, capability check, indirect-call sites in the ONNX op dispatcher, GPU command submission), wires the compiler-level mitigations (LFENCE / SLH / Retpoline on x86, CSDB on aarch64, future fence.t on RISC-V when ratified), and documents the coverage in a new `kernel-security` capability spec. The work is deliberately conservative — these are well-trodden mitigation patterns, not a research contribution. The novelty is making them explicit in SmallAIOS's safety case rather than relying on implicit compiler defaults.

## Why

- **Trust boundaries that look like ordinary function calls in Rust are not protected against speculation by default.** When userspace (a future construct in SmallAIOS — today everything is in-kernel, but the syscall surface remains the trust boundary between "untrusted task graph" and "kernel-managed capability state") invokes a syscall, the CPU may speculatively load through attacker-controlled addresses before the capability check resolves. Spectre v1 is the classic instance: a bounds check that always succeeds in observed execution can mispredict, and the speculative path loads attacker-chosen memory into cache, becoming observable via a cache-timing side channel. Rust's type system catches none of this — it is a CPU-level concern that needs explicit `lfence` (x86) or `csdb` (aarch64) at the right point in the syscall entry assembly.
- **Compiler-level mitigations have known gaps that documentation must address.** Retpoline (the GCC/clang mitigation for Spectre v2) is on by default in upstream Rust for `x86_64-unknown-linux-musl` but *off* by default for `x86_64-unknown-none` (our bare-metal target). IBRS / IBPB / STIBP — Intel's hardware-level branch prediction barriers — require explicit MSR writes from the kernel; they are not compiler-driven. Speculative Load Hardening (SLH) on x86 is a `-mllvm` flag we don't currently pass. AArch64's `csdb` / `dsb sy` barriers are inserted by the compiler at `core::hint::black_box` boundaries but never inside hand-rolled syscall entry code (`arch/aarch64/src/syscall.rs`). We need explicit instrumentation.
- **SmallAIOS's ONNX op-dispatch indirect-call table is a textbook Spectre v2 surface.** The runtime in `onnx-rt` dispatches each ONNX operator through a function-pointer table — exactly the indirect-branch construct that branch-target injection exploits. Compiler-driven Retpoline addresses this on x86_64; the aarch64 equivalent is BTI (Branch Target Identification, already enabled via `aarch64-mte-pac-hardening-v1`'s codegen flags). RISC-V has no equivalent yet — we document the gap and plan for the (still-being-ratified) Zicfilp control-flow-integrity extension.
- **DO-178C DAL A asks "what is the worst observable side effect of an out-of-order CPU mispredict" — and demands a defensible answer.** Avionics certification is starting to ask explicit questions about speculative side-channels following CVE-2022-23960 (Retbleed) and the Branch History Injection class of attacks (Spectre-BHB). Our cert-track positioning needs to enumerate covered attacks, applied mitigations, and residual risk explicitly. This change produces that documentation as a spec, not just a code patch.

## Scope — phases

The work fans out per-architecture but shares one tabular structure: at each documented trust boundary, what speculation barrier (compiler-emitted, kernel-emitted, or both) does each arch need?

### Phase 1 — Trust-boundary audit (~3-4 days)

Produce a per-arch table covering five trust boundaries: (a) syscall entry, (b) capability check, (c) ONNX op-dispatch indirect call, (d) GPU command submission, (e) bus-backed dataflow runner message receive. For each, document the speculation-exploitable code path, the currently-applied mitigation (if any), and the gap.

Audit deliverable: `docs/spec-exec-audit.md` table. No code changes in Phase 1.

### Phase 2 — x86_64 mitigations (~1 week)

- **Compiler flags:** enable Retpoline for `x86_64-unknown-none` via `RUSTFLAGS = -C target-feature=+retpoline-external-thunk,+retpoline-indirect-branches,+retpoline-indirect-calls` set in `.cargo/config.toml` under a `spec-exec-x86` Cargo feature.
- **SLH (Speculative Load Hardening):** add `-C llvm-args=-x86-speculative-load-hardening` for kernel build targets. This is the LLVM-emitted SLH that converts speculative-bounds-bypass into a hard branch.
- **IBRS / IBPB / STIBP MSR programming:** on boot, after exception vectors install but before the first syscall is dispatched, write `IA32_SPEC_CTRL` to enable IBRS (Indirect Branch Restricted Speculation). On every syscall entry, emit an `IBPB` (Indirect Branch Predictor Barrier) — costly but eliminates the cross-privilege branch prediction state. Document the latency overhead.
- **LFENCE at syscall entry:** explicit `lfence` instruction in the syscall trampoline (`arch/x86_64/src/syscall.rs` if it exists; otherwise the equivalent hand-rolled entry) immediately after capability check, before any attacker-controlled-address load. Spectre v1 mitigation.
- **Meltdown:** Tegra234 is aarch64-only, so Meltdown specifically doesn't apply to our reference platform. For x86_64 we document KPTI (Kernel Page Table Isolation) as the standard Linux-world answer; in our unikernel single-address-space model, Meltdown is structurally absent (there is no user-space page-table view to leak from), and we document that as the architectural mitigation.

### Phase 3 — aarch64 mitigations (~3-4 days)

- **CSDB barriers at syscall entry:** insert `csdb` (Consumption of Speculative Data Barrier) into the syscall entry in `arch/aarch64/src/syscall.rs` after the capability check, before any tensor / device handle dereference. The `csdb` barrier is a no-op on CPUs without Spectre v1 hardware speculation, costless on Cortex-A78AE where it matters.
- **DSB SY at capability boundaries:** `dsb sy` (Data Synchronization Barrier, system-wide) before transferring control across a capability check that gates DMA setup. Same shape as `csdb` but stronger — pairs with the SMMU work in `tegra-smmu-isolation-v1`.
- **BTI (Branch Target Identification):** already enabled by `aarch64-mte-pac-hardening-v1`; we document the cross-reference.
- **ARM-specific Spectre v2 mitigation:** Cortex-A78AE implements the "hardware mitigation" of Spectre v2 (per Arm's TRM) via `CSV2` and `CSV3` features; we read `ID_AA64PFR0_EL1` to confirm and refuse to boot with a warning if the silicon downgrades them. The kernel does not need to install software Retpoline analogs on this silicon, but we document the assumption.

### Phase 4 — RISC-V mitigations (~2-3 days)

RISC-V's speculative-execution story is the youngest. The relevant extensions:

- **Zicbom / Zicbop / Zicboz** (cache-block management) — used for explicit cache-state cleanup, partial mitigation for cache-timing channels.
- **Zicfiss / Zicfilp** (Shadow stack + landing-pad CFI) — Spectre v2-class mitigation analogous to PAC + BTI on aarch64. Still being ratified at proposal time.
- **`fence.i` after privileged transitions** — instruction-fetch barrier between privilege levels.

The bring-up state of RISC-V in SmallAIOS is `riscv64gc-unknown-none-elf` boot only; full syscall paths aren't exercised yet. Phase 4 documents the planned mitigations and inserts `fence.i` at the future syscall trampoline boundary so the entry shape is correct even without active workloads. Production-grade RISC-V mitigations are a follow-up change once Zicfilp ratifies.

### Phase 5 — ONNX op-dispatch hardening (~2-3 days)

The ONNX runtime's op-dispatch table (one function pointer per ONNX operator) is a Spectre v2 attack surface. Mitigations:

- **On aarch64:** BTI on every dispatch entry is sufficient (already enabled via `aarch64-mte-pac-hardening-v1`).
- **On x86_64:** Retpoline-driven dispatch is sufficient when `+retpoline-indirect-calls` is on.
- **On both:** add a kernel-side audit that the dispatch table is read-only after init (move to `.rodata`) so an attacker who has gained code-execution cannot tamper with the table itself.

### Phase 6 — Documentation + safety-case integration (~2-3 days)

Produce the formal mitigation matrix as part of the `kernel-security` capability spec. Cross-reference to the existing `safety-critical` and `security` specs so the DO-178C safety case can cite this single table.

## Out of scope

- **Side-channel-resistant cryptography.** The PQC stack already uses constant-time implementations; that's tracked in `pqc-crypto` not here.
- **Hyperthreading-specific mitigations.** Cortex-A78AE has no SMT; future SmallAIOS targets with SMT (rare in the embedded inference space) would need separate review. Documented but not patched.
- **Pre-boot speculative leakage.** UEFI firmware's own speculative behavior is out of our control. We document it as a residual risk.
- **`flush_cache` instruction-level audit.** Cache-flush is sometimes used as a side-channel mitigation but mostly as a coherence mechanism in our codebase. A future audit could classify each `flush` for its security intent.
- **JIT / dynamic codegen.** SmallAIOS has no JIT path; the ONNX runtime is fully AOT. The Spectre v1 patterns that depend on JIT-emitted gadgets (Spectre-RSB chains) are structurally absent.
- **Microcode update plumbing.** Intel / AMD microcode updates are the manufacturer's responsibility; we don't ship them. We document the assumption that the deployment environment keeps microcode current.

## Sequencing

Phase 1 (audit) gates everything else and lands first. Phases 2 (x86_64), 3 (aarch64), 4 (RISC-V) can land in parallel — different files, different conditional compilation paths. Phase 5 (op-dispatch) is small but depends on Phases 2-3 to define the per-arch context. Phase 6 (docs) closes the change.

This change runs in parallel with `tegra-smmu-isolation-v1` (DMA-side, no CPU speculation touch) and depends on `aarch64-mte-pac-hardening-v1` for BTI being on (a hard dep for Phase 3 / 5 aarch64 coverage). On the x86_64 side it is fully independent.

## Effort estimate

| Phase | Scope | Estimate |
|-------|-------|----------|
| 1 | Audit + trust-boundary table | ~3-4 days |
| 2 | x86_64 (Retpoline + SLH + IBRS / IBPB + LFENCE) | ~1 week |
| 3 | aarch64 (CSDB + DSB SY + CSV2/3 silicon check) | ~3-4 days |
| 4 | RISC-V (fence.i scaffolding, Zicfilp documentation) | ~2-3 days |
| 5 | ONNX op-dispatch hardening | ~2-3 days |
| 6 | Spec + safety-case documentation | ~2-3 days |
| **Total** | | **~2 weeks** |
