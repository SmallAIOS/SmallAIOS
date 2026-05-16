# Speculative-Execution Mitigation Audit — RISC-V (Phase 4, scaffolding)

> **Status: SCAFFOLDING / PARTIAL.** This document covers only the RISC-V
> (`riscv64gc-unknown-none-elf`) slice of OpenSpec change
> `spec-exec-mitigations-v1`, Phase 4. It is intentionally a **separate**
> file from the cross-arch spine document `docs/spec-exec-audit.md`; the
> RISC-V content here will be folded into / linked from that canonical audit
> centrally once the spine PR lands. Do not duplicate the cross-arch matrix
> here.
>
> **Production RISC-V deployments MUST be flagged "speculation mitigations
> partial" in the DO-178C safety case.** See [Safety-case
> impact](#safety-case-impact).

## Scope and current reality

RISC-V is a **boot-only** target in SmallAIOS today. The kernel boots to the
halt loop; no U-mode task graph and no syscall workload is exercised yet (the
syscall path `arch/riscv64/src/syscall.rs::handle_ecall` →
`smallaios_kernel::syscall::dispatch` exists and is unit-tested on the host,
but is not driven by a real workload on RISC-V silicon).

Critically, the hardware extensions that would let RISC-V mount a *real*
defense against the Spectre-class attacks at the syscall trust boundary are
**mostly unratified as of 2026-02**. We therefore cannot ship the
silicon-programming half of the mitigation; what we ship is:

1. The architecturally-defined `fence.i` instruction-fetch barrier at the
   privileged-transition boundary (correct-by-construction now).
2. A scaffolding module (`arch/riscv64/src/security/spec_exec.rs`) whose
   *shape* is fixed so the ratification follow-up is a body fill-in, not a
   redesign.
3. This audit document, recording the gap explicitly so the safety case can
   cite it.

## Planned mitigations

### Cache-state cleanup — Zicbom / Zicbop / Zicboz

| Extension | Purpose | Ratification | Use in SmallAIOS |
|-----------|---------|--------------|------------------|
| **Zicbom** | Cache-Block Management: `cbo.clean` / `cbo.flush` / `cbo.inval` | Ratified | At the privileged transition, clean/flush the capability-check working set so the cache-state residue an attacker can probe via timing is bounded. **Partial** Spectre-v1 cache-timing mitigation only — does not close the speculation window itself. |
| **Zicbop** | Cache-Block Prefetch: `prefetch.{i,r,w}` | Ratified | Controlled prefetch to normalize cache state across the boundary (defense-in-depth alongside Zicbom). Hint-only; no architectural guarantee, so it is *adjunct*, never the primary mitigation. |
| **Zicboz** | Cache-Block Zero: `cbo.zero` | Ratified | Zeroing scratch lines that held attacker-influenced data before returning to U-mode. |

Although Zicbo* is ratified, **no silicon in the SmallAIOS validation matrix
advertises it yet**, and the cache-block size (`CBOM_BLOCK_SIZE`) must be
discovered from the DTB before any `cbo.*` may be issued safely. The cleanup
path is therefore gated off in `security::spec_exec::init()` until both
(a) the DTB exposes the block size and (b) a validated target advertises the
extension. Even fully wired, this is a **partial** mitigation: cache-state
cleanup narrows a side channel; it does not provide branch-target or
shadow-stack integrity.

### Control-flow integrity — Zicfilp / Zicfiss (NOT ratified)

These are the load-bearing Spectre-v2-class mitigations, and they are the
ones we **cannot** ship:

| Extension | CFI edge | Attack class addressed | Ratification (2026-02) |
|-----------|----------|------------------------|------------------------|
| **Zicfilp** | Forward-edge (landing pads on indirect-branch targets) | Spectre v2 / branch-target-injection (BTI) | **NOT ratified.** No stable encoding for the CSRs (`menvcfg.LPE` / `senvcfg.LPE`, `mseccfg` lockdown bits) we would program. |
| **Zicfiss** | Backward-edge (hardware shadow stack) | Return-address corruption / Retbleed-class | **NOT ratified.** No stable `ssp` shadow-stack-pointer CSR / `menvcfg.SSE` / `senvcfg.SSE` to rely on. |

Because both are unratified, SmallAIOS:

- Does **not** depend on any Zicfilp/Zicfiss encoding (no speculative use of
  a CSR that may be renumbered before ratification).
- Fixes the *call shape* now (`security::spec_exec::init()` is on the boot
  path with a documented TODO enumerating exactly which CSRs it will program
  once the extensions ratify and silicon is detected) so the follow-up
  change is a contained fill-in.
- Leaves the RISC-V forward/backward-edge CFI posture as **absent** in the
  cross-arch matrix — contrast aarch64 (BTI via
  `aarch64-mte-pac-hardening-v1`) and x86_64 (Retpoline + IBRS), which do
  have shipped mitigations on these edges.

### Instruction-fetch barrier — `fence.i` (shipped now)

`fence.i` (Zifencei) **is** part of the RV64GC baseline and is shipped now.

- **Where:** `arch/riscv64/src/syscall.rs::handle_ecall`, via the
  `fence_i_barrier()` helper.
- **Placement:** emitted **after** `smallaios_kernel::syscall::dispatch`
  returns and **before** `handle_ecall` returns toward the `sret` (in
  `arch/riscv64/src/trap.rs`) that resumes U-mode.
- **Why this side:** the *after-dispatch / pre-return* position is the
  load-bearing one. RISC-V does not make a hart's stores to the instruction
  stream visible to that hart's *subsequent instruction fetch* without an
  explicit `fence.i`. A syscall may legitimately mutate code the resuming
  context will fetch (future code-load / page-remap / module-style calls);
  the barrier guarantees the resuming hart observes that mutation, and it
  also bounds any stale-instruction window opened while the kernel serviced
  the call. A `fence.i` *before* dispatch would order fetches against
  pre-syscall state, which is not what the privileged-transition boundary
  needs.
- **What it is NOT:** `fence.i` is an instruction-fetch / fetch-coherence
  barrier, **not** a speculation barrier. It does **not** close the
  Spectre-v1 (bounds-check bypass) or Spectre-v2 (branch-target-injection)
  windows. It is shipped now purely so the privileged-transition entry
  *shape* is correct-by-construction before syscall workloads come online;
  it is retained even after Zicfilp/Zicfiss land (defense in depth +
  instruction-fetch coherence).

## Trust-boundary posture (RISC-V column)

The canonical trust-boundary × attack-class matrix lives in the spine
document `docs/spec-exec-audit.md`. The RISC-V column summarizes as:

| Trust boundary | Spectre v1 (BCB) | Spectre v2 (BTI) | Spectre v4 (SSB) | Meltdown | Retbleed | Spectre-BHB |
|----------------|------------------|------------------|------------------|----------|----------|-------------|
| Syscall entry (`syscall.rs::handle_ecall`) | Partial — `fence.i` shipped (coherence, not a spec barrier); Zicbo* cache cleanup gated off | **Absent** — needs Zicfilp (unratified) | Absent — no ratified SSB control on RISC-V | **N/A (structural)** — unikernel, single address space, no user/kernel page-table split (same rationale as the spine doc's Meltdown row) | **Absent** — needs Zicfiss shadow stack (unratified) | Absent — no ratified BHB control |
| Capability check / ONNX op-dispatch / bus receive | Inherits the above; not separately mitigated on RISC-V at this phase | Absent | Absent | N/A (structural) | Absent | Absent |

"Partial" / "Absent" here is **stronger language than the other arches** by
design — RISC-V genuinely lacks the ratified primitives the spine doc's
x86_64 / aarch64 columns rely on.

## Safety-case impact

- RISC-V production deployments **MUST** carry the explicit safety-case
  annotation **"speculation mitigations partial"**. This is not a
  formality: forward-edge (Spectre v2) and backward-edge (Retbleed-class)
  CFI are *absent*, not merely degraded.
- The boot record emits a single grep-able marker line so this is auditable
  from logs:
  `[spec-exec-riscv] scaffolding — extensions not yet ratified, software mitigations partial`
  (emitted by `security::spec_exec::init()` on the boot path).
- Meltdown is recorded as **N/A (structural)** — the SmallAIOS unikernel has
  no user/kernel page-table privilege split, so Meltdown structurally does
  not apply (same evidence kind as the spine document's cross-arch Meltdown
  rationale). This is an absence-of-applicability claim, not an
  absence-of-mitigation claim.

## Review trigger

Promote RISC-V from *partial* toward *full* and re-audit when **any** of:

- Zicfilp and/or Zicfiss reach ratified status with stable CSR encodings.
- A validated SmallAIOS RISC-V target advertises Zicbom (with DTB-discoverable
  `CBOM_BLOCK_SIZE`) so the cache-cleanup path can be enabled.
- A syscall workload is brought online on RISC-V silicon (the barrier shape
  becomes load-bearing in practice, not just by construction).
- A new CVE lands in the speculation-class space affecting RISC-V silicon in
  the validation matrix (tracked, per the spine doc's review-trigger policy,
  as a new OpenSpec change).

## Cross-references

- Spine / cross-arch audit: `docs/spec-exec-audit.md` (owned by the spine PR;
  this RISC-V file is merged/linked into it centrally).
- OpenSpec change: `openspec/changes/spec-exec-mitigations-v1/` (Phase 4 =
  RISC-V scaffolding; tasks tracked on the spine branch).
- Scaffolding module + ratification TODO:
  `arch/riscv64/src/security/spec_exec.rs`.
- Barrier placement + rationale:
  `arch/riscv64/src/syscall.rs` (`fence_i_barrier`).
