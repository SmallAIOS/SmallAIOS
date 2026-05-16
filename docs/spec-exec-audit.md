# Speculative-Execution Trust-Boundary Audit

> **Change:** `spec-exec-mitigations-v1` — Phase 1 deliverable (tasks 0.1–0.4).
> **Toolchain audited:** `nightly-2026-02-01` (LLVM 19), per `rust-toolchain.toml`.
> **Status:** Living document. Every new CVE in the speculative-execution
> attack class triggers a re-audit (see *Review Trigger* at the end).

## 1. Scope

This audit enumerates SmallAIOS's trust boundaries, then populates a
boundary × architecture × attack-class matrix. A "trust boundary" is a
control-flow transition where speculation past an unresolved check could
load or branch through attacker-influenced state. SmallAIOS today runs
everything in one address space (unikernel, no user/kernel page-table
split), so the relevant boundaries are *function-call / dispatch-shaped*,
not *page-table-shaped*. This is load-bearing for the Meltdown row below.

## 2. Trust boundaries (task 0.1)

| # | Boundary | Code location (file:line) | Speculation-exploitable path |
|---|----------|---------------------------|------------------------------|
| (a) | Syscall entry — x86_64 | `arch/x86_64/src/syscall.rs:123` `syscall_entry` (naked) → `smallaios_kernel::syscall::dispatch` (`kernel/src/syscall/mod.rs:509`) | CPU may speculate through `dispatch` and into an argument-indexed load before the syscall number / capability check resolves. |
| (a) | Syscall entry — aarch64 | `arch/aarch64/src/syscall.rs:87` `sync_exception_entry` (naked vector) → `svc_handler:46` → `dispatch` | Same shape; SVC vector decodes `regs[]` then calls `dispatch`. Speculative load through `regs[..]` indices. |
| (a) | Syscall entry — riscv64 | `arch/riscv64/src/syscall.rs:25` `handle_ecall` → `dispatch` | Scaffolding only — U-mode `ecall` path; syscall workloads not yet exercised. Entry shape must still be correct for future hardening. |
| (b) | Capability check | `kernel/src/state.rs:249` `check_capability` → `registry.check(...)`; wrapper `kernel/src/syscall/memory.rs:112` `require_capability` | A bounds/permission check that *always succeeds in observed runs* can mispredict; the speculative path past `check_capability(...)?` can load attacker-chosen memory (classic Spectre v1 / bounds-check-bypass). This is the **primary** boundary the explicit fences target. |
| (c) | ONNX op-dispatch | `onnx-rt/src/executor.rs:441` `match op_type { ... }`, `:867` `match node.op_type.as_str()` | **Finding:** dispatch is a Rust `match` on a string op-type, **not** a function-pointer table. See §4 — this materially reduces the Spectre v2 surface vs. the proposal's worst-case assumption. The `match` may still lower to a jump table in `.rodata`; the table is read-only by construction (string-keyed match arms, no runtime-mutable fn pointers). |
| (d) | GPU command submission | NVIDIA HAL — not yet landed on this branch (CUDA path is container-only; bare-metal NVIDIA HAL is an architectural stub per `CLAUDE.md`) | **Deferred / future.** No kernel-side GPU command ring exists on the bare-metal path yet. Documented as a placeholder row; revisited when the NVIDIA bare-metal HAL lands. |
| (e) | Bus dataflow runner receive | `ipc/src/pubsub.rs:179` `Subscriber::receive`, `ipc/src/reqrep.rs:92` `Queryable::receive_query` | Message-receive decodes a length-prefixed buffer. Speculative read past a length check could touch out-of-bounds ring memory. Lower severity (no privilege transition — same address space), documented for completeness. |

## 3. Boundary × architecture × attack-class matrix (task 0.2)

Attack classes: **v1** Spectre-v1 bounds-check bypass · **v2** Spectre-v2
branch-target injection · **v4** Spectre-v4 speculative-store-bypass ·
**MD** Meltdown · **RB** Retbleed · **BHB** Spectre-BHB (branch-history
injection).

Cell legend: `fence` = kernel-emitted barrier · `cc` = compiler-emitted
(Retpoline/SLH/BTI/CSDB) · `hw` = silicon-level (eIBRS / CSV2 / CSV3) ·
`struct` = structurally absent · `n/a` = not applicable on this arch ·
`scaffold` = entry-shape correct, full mitigation pending ratification.

### (a) Syscall entry

| Arch | v1 | v2 | v4 | MD | RB | BHB |
|------|----|----|----|----|----|-----|
| x86_64 | `fence` LFENCE after cap-check (1.10) | `cc` Retpoline + `hw` eIBRS/IBPB (1.1–1.9) | `hw` SSBD via IA32_SPEC_CTRL | `struct` no user/kernel AS split | `cc` Retpoline + `hw` IBPB-on-entry | `hw` eIBRS + `cc` Retpoline |
| aarch64 | `fence` CSDB after cap-check (2.3) | `hw` CSV2≥1 on A78AE + `cc` BTI | `hw` SSBS (A78AE) | `struct` no user/kernel AS split | `hw` CSV2 (no RSB mispredict path on A78AE) | `hw` CSV2 + `cc` BTI |
| riscv64 | `scaffold` `fence.i` placeholder (3.3) | `scaffold` Zicfilp not ratified | `scaffold` | `struct` no user/kernel AS split | `scaffold` | `scaffold` |

### (b) Capability check

| Arch | v1 | v2 | v4 | MD | RB | BHB |
|------|----|----|----|----|----|-----|
| x86_64 | `fence` LFENCE between `check_capability(...)?` and first attacker-addressed load | `cc` Retpoline (the `?` early-return is an indirect-free branch; no fn-ptr) | `hw` SSBD | `struct` | `cc`+`hw` | `hw`+`cc` |
| aarch64 | `fence` CSDB at same point; `dsb sy` before cap-gated DMA setup (2.4) | `hw` CSV2 + `cc` BTI | `hw` SSBS | `struct` | `hw` | `hw`+`cc` |
| riscv64 | `scaffold` | `scaffold` | `scaffold` | `struct` | `scaffold` | `scaffold` |

### (c) ONNX op-dispatch

| Arch | v1 | v2 | v4 | MD | RB | BHB |
|------|----|----|----|----|----|-----|
| x86_64 | `cc` SLH on bounds in op kernels | `cc` Retpoline + table in `.rodata` (4.1, 4.3) | `hw` SSBD | `struct` | `cc` | `hw`+`cc` |
| aarch64 | `cc` SLH-equivalent / value-predication | `hw` CSV2 + `cc` BTI landing pads (4.2) | `hw` SSBS | `struct` | `hw` | `hw`+`cc` |
| riscv64 | `scaffold` | `scaffold` | `scaffold` | `struct` | `scaffold` | `scaffold` |

### (d) GPU command submission — *deferred*

All cells `n/a (deferred)` — no bare-metal NVIDIA command ring exists on
this branch. Row retained so the matrix is structurally complete; the
NVIDIA-HAL change that lands the command ring inherits the obligation to
populate this row.

### (e) Bus dataflow runner receive

| Arch | v1 | v2 | v4 | MD | RB | BHB |
|------|----|----|----|----|----|-----|
| all | `cc` bounds-check on ring length (Rust slice bounds + SLH where on) | `cc` Retpoline/BTI per arch | `hw` SSBD/SSBS | `struct` | per arch | per arch |

Severity note: boundary (e) carries no privilege transition (same address
space, no capability gate crossed), so a speculative OOB read there leaks
only data already reachable by the same task graph. Documented; no extra
kernel-emitted fence is warranted beyond the compiler-default slice bounds.

## 4. Key finding — ONNX dispatch is a `match`, not a fn-ptr table (task 0.2)

The proposal's worst-case assumption (Spectre v2 surface via a
function-pointer dispatch table) does **not** hold: `onnx-rt`'s operator
dispatch is a Rust `match node.op_type.as_str()` (`executor.rs:441,867`).
Match arms resolve to direct calls; LLVM may lower a large string-`match`
to a jump table, but the table is compiler-generated, immutable, and lives
in `.rodata` by construction — there is no runtime-writable array of
function pointers an attacker could poison. The residual v2 surface is the
ordinary indirect-branch shape any large `match` produces, fully covered
by per-arch `cc` mitigations (Retpoline on x86_64, BTI+CSV2 on aarch64).
Task 4.1's "ensure the table is read-only after init" therefore becomes
"assert the lowered jump table is in `.rodata`" rather than "relocate a
mutable table" — a verification task, not a code-restructuring task.

## 4b. ONNX op-dispatch hardening (Phase 5 — tasks 4.1–4.4)

**Attack surface.** The ONNX runtime selects an operator implementation
per graph node. The dispatch is a Rust `match node.op_type.as_str()`
(`onnx-rt/src/executor.rs:441`, `:867`; the CUDA fast-path mirror is
`try_cuda_dispatch:441`). There is **no runtime-mutable function-pointer
table**. The Spectre-v2 surface is therefore the ordinary indirect-branch
that LLVM may emit when lowering a large string-`match` to a jump table —
not a poisonable dispatch array.

**Mitigations and how they are verified:**

- **4.1 — table is read-only after init.** The jump table LLVM emits for
  the `match` is placed in `.rodata` by construction (it is compiler-owned,
  not a `static mut`/`&[fn()]` the runtime fills in). No relocation work is
  required; the obligation is *verification*: the `spec-exec-disasm-audit`
  CI job (`scripts/spec-exec-disasm-audit.sh`) asserts a `.rodata` section
  exists and that there is no writable dispatch array. There is no
  `static mut` operator table in `onnx-rt` to move.
- **4.2 — aarch64 BTI landing pads.** BTI is enabled by the
  `aarch64-mte-pac-hardening-v1` change's `-Z branch-protection=bti`
  codegen. Every indirect-branch target then begins with a `bti` landing
  pad; a branch to a non-`bti` address raises a Branch Target exception.
  Verified by the disasm-audit job (greps for `bti` opcodes on the
  aarch64 build); cross-referenced — this change does not re-implement BTI.
- **4.3 — x86_64 Retpoline dispatch.** With `--features spec-exec-x86`
  the Retpoline codegen flags (Phase 2) thunk every indirect branch,
  including the `match` jump table's indirect jump. The disasm-audit job
  asserts there is no naked `call *%reg` / `jmp *%reg` in the linked
  kernel.
- **4.4 — documentation.** This section *is* the canonical op-dispatch
  attack-surface + mitigation record. `docs/onnx-runtime.md` does not
  exist; the `onnx-runtime` capability spec and this audit doc are the
  equivalent. The net: op-dispatch is a *smaller* Spectre-v2 surface than
  the proposal's worst case, fully covered by the per-arch compiler
  mitigations already scheduled in Phases 2–3, with a CI assertion as the
  regression catcher.

## 5. Silicon-level mitigations already present (task 0.3)

| Mitigation | Arch | Detection method | SmallAIOS reference platform |
|------------|------|------------------|------------------------------|
| Enhanced IBRS | x86_64 | `CPUID` leaf 7, `EDX[29]` | Datacenter Xeon (Sapphire Rapids+) assumption |
| STIBP (SMT) | x86_64 | `CPUID` leaf 1, `EBX[23]` for SMT presence | Set only if SMT detected |
| SSBD (Spectre v4) | x86_64 | `CPUID` 7 `EDX[31]`, programmed via `IA32_SPEC_CTRL` | — |
| CSV2 | aarch64 | `ID_AA64PFR0_EL1.CSV2` ≥ 1 | **Cortex-A78AE (Jetson Orin) reports CSV2≥1** — HW-mitigated Spectre v2 |
| CSV3 | aarch64 | `ID_AA64PFR0_EL1.CSV3` ≥ 1 | A78AE reports CSV3≥1 — HW-mitigated cache-allocate side channel |
| SSBS | aarch64 | `ID_AA64PFR1_EL1.SSBS` | A78AE supports speculative-store-bypass-safe |
| Zicfilp/Zicfiss | riscv64 | CSR probe (post-ratification) | **Not yet ratified** — scaffolding only |

## 6. Compiler-flag mitigations available (task 0.4)

Pinned toolchain `nightly-2026-02-01` bundles **LLVM 19**. Verified flag
forms for that LLVM:

| Flag | Arch | Effect | Where set |
|------|------|--------|-----------|
| `-C target-feature=+retpoline-external-thunk,+retpoline-indirect-branches,+retpoline-indirect-calls` | x86_64 | Spectre v2 indirect-branch thunking | `spec-exec-x86` feature → `arch/x86_64/build.rs` |
| `-C llvm-args=-x86-speculative-load-hardening` | x86_64 | SLH — branchless bounds hardening (Spectre v1) | same build.rs (defense-in-depth alongside explicit LFENCE) |
| `-Z branch-protection=bti` (or codegen default on `aarch64-unknown-uefi`) | aarch64 | BTI landing pads — Spectre v2 indirect-call CFI | enabled by `aarch64-mte-pac-hardening-v1`; cross-referenced here |
| `csdb` / `dsb sy` (hand-emitted) | aarch64 | Spectre v1 / DMA-ordering barriers | `arch/aarch64/src/syscall.rs` (this change) |
| `lfence` (hand-emitted) | x86_64 | Spectre v1 barrier after cap-check | `arch/x86_64/src/syscall.rs` (this change) |
| `fence.i` (hand-emitted, scaffold) | riscv64 | Instruction-fetch barrier at privileged transition | `arch/riscv64/src/syscall.rs` placeholder |

Why feature-gated `build.rs` and not global `RUSTFLAGS`: global flags
apply to build-script host code too (Retpoline is meaningless compiling a
build script on a dev laptop). The `spec-exec` Cargo feature pulls
arch-specific sub-features (`spec-exec-x86`/`-aarch64`/`-riscv`) selected
by `cfg(target_arch)`, each gating an arch `build.rs` flag emit.

## 7. Meltdown — structural absence (design Decision 6)

Meltdown (CVE-2017-5754) exploits speculative loads through a *user-space
view of kernel pages*. SmallAIOS is a unikernel: the task graph runs in
the **same address space** as the kernel; there is no user/kernel
page-table split, hence no cross-privilege page-table view to leak from.
Spectre v1/v2 still apply (they exploit speculation across *call*
boundaries, not page-table boundaries) — Meltdown specifically does not.
The DO-178C safety case cites the architectural model (`docs/architecture.md`)
as the evidence; **no KPTI-equivalent is applied because none is required**.
A future change introducing a user/kernel AS split (e.g. a hypervisor
mode) must re-open this row.

## 8. Residual risks (carried from design.md)

- **New attack classes will emerge** (Meltdown'18 → Retbleed'22 →
  Inception'23 → GhostRace'24). This audit covers *known* classes;
  the Review Trigger below institutionalizes re-audit.
- **RISC-V coverage is partial** — Zicfilp/Zicfiss not ratified; Phase 4
  is entry-shape scaffolding only. Production RISC-V deployments are
  flagged "speculation mitigations partial" in the safety case.
- **Pre-boot firmware speculation** (UEFI) is out of our control —
  documented residual.
- **IBPB latency** in syscall-heavy ONNX workloads — bench-gated
  (task 7.2, <5% steady-state target); `spec-exec-ibpb-off` opt-out
  exists for performance-mode deployments that accept the residual v2 risk.

## 9. Review Trigger (task 5.3)

**Any new CVE in the speculative-execution attack class triggers a
re-audit of the affected boundaries.** The maintainer aware of the CVE
SHALL open a new OpenSpec change, refresh §3 of this document, and add any
newly-needed mitigation as a delta to the `kernel-security` spec. This
document is the canonical table the DO-178C safety case cites.
