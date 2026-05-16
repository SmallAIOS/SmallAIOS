# SmallAIOS Advanced Features Roadmap

> **Status:** Living roadmap. Last revised 2026-05-10. Authoritative for the
> 15 advanced-feature proposals drafted in batches 1–4 alongside the 22
> currently active OpenSpec changes. Re-revise after each quarterly review.
>
> **Audience:** This document serves two readers. Engineering teams should
> read it as a build sequence with concrete acceptance criteria.
> Funding / leadership should read sections **Executive Summary**, **Vision**,
> **Hardware Procurement Plan**, **DO-178C Certification Claim Map**, and the
> **Risk Register** — those frame the business case.

## Executive Summary

SmallAIOS has reached the end of its "prove the unikernel boots and serves
ONNX models" phase. The next 18–24 months turn the prototype into a
DO-178C DAL A-certifiable inference platform on the **Jetson Orin
industrial** SKU at the edge, and into a **confidential-compute AI
inference appliance** in the datacenter — both with the same crate
workspace, the same PQC stack, and the same ~46-syscall surface. Fifteen
new proposals — split across four batches now in parallel drafting — plus
ten of the 22 currently active OpenSpec changes constitute that work.
This roadmap sequences them.

The critical insight is that **the boot-trust path is the bottleneck.**
Memory-safety hardening (MTE, PAC, SMMU isolation, CHERI), confidential
compute (CCA, SEV-SNP, TDX), and remote attestation all require a verified
chain from immutable hardware root through SmallAIOS to the model artifact.
That chain does not exist today on any SmallAIOS target. Building it —
TPM2 on x86, OP-TEE bridge on ARM, hardware RoT integration on Tegra234 —
is the spine of the next four quarters. Everything else either feeds that
spine or rides on top of it.

## Current State (2026-05-10)

What we have right now, on `origin/develop`:

- **Tests passing:** 4,773 (CLAUDE.md reports 4,143; the recent reorg PRs
  pushed it higher — confirm with `just test` after the in-flight `develop`
  reorg lands).
- **GPU-validated container path:** `Dockerfile.jetson` on Jetson Orin NX
  16 GB (P3767-0000 + P3768-0000, JetPack 6.2.1 / L4T R36.4.7). ResNet-50
  hybrid + CUDA graph capture + multi-stream overlap, all merged.
- **Unikernel KVM-on-L4T:** `just run-jetson-kvm` boots the SmallAIOS
  AArch64 kernel on Orin's A78AE cores under KVM. Phase 1 of
  `unikernel-orin-bringup-v1` is **done** (1.1–1.8 checked).
- **Unikernel UEFI USB boot:** Phase 2 of `unikernel-orin-bringup-v1` is
  ~55% done. UEFI entry, DTB discovery via `EFI_DTB_TABLE_GUID`, and
  banner-via-`con_out` are landed. **Open blocker:** Tegra Combined UART
  (TCU) post-`ExitBootServices` access faults (SError EC=0x2F / sync
  translation fault) — we don't yet have the NVIDIA-side privileged
  path. This blocker is on the critical path of every Q4 2026
  Orin-unikernel item below.
- **Formal verification scaffolding:** 25 TLA+ models (22 in CI),
  6 SPIN/Promela models, 6 Lean 4 proofs (not yet CI-gated — that's
  Phase 1 of `formal-proving-and-redteam-v1`).
- **PQC stack:** ML-KEM-768 + ML-DSA-65 hybrid mode is default. No NIST
  KAT gate yet, no differential fuzz vs `pqcrypto-{kyber,dilithium}` —
  also tracked under `formal-proving-and-redteam-v1` Phase 2.
- **22 active OpenSpec changes:** see Proposal Inventory below.
  Notable: `unikernel-orin-bringup-v1` (in flight), `embedded-overlay-v1`
  Phase 5 landed PR #185 (ML-DSA-65 signature integrity on model loads),
  `formal-proving-and-redteam-v1` umbrella drafted but un-started.

## Vision

By the end of **Q3 2027** SmallAIOS shall:

1. Run as a DO-178C DAL A-certifiable unikernel on Jetson Orin Industrial
   (AGX Orin Industrial, the SKU with extended-temperature, lockstep
   Cortex-R52F safety island, and 10-year supply commitment) — booting
   bare-metal from a hardware root of trust, with kernel binary and model
   artifacts ML-DSA-65 signature-verified before execution, MTE + PAC
   enabled, GPU bounded by Tegra SMMU contexts, and a watchdog/lockstep
   fault-detection envelope around the inference loop.

2. Run as a confidential-compute inference appliance on **ARM Neoverse
   N3/V4 with CCA Realms** (datacenter) and on **AMD EPYC SEV-SNP** /
   **Intel Xeon TDX** (x86 datacenter) — with remote attestation of both
   kernel measurement and model hash to the customer's KMS before any
   inference dispatch.

3. Continue to serve as a permissive-license, container-deployable AI
   inference runtime on commodity Jetson Orin / x86 / AArch64 server
   hardware. The container path is not deprecated by the unikernel path;
   they share crates and ship together.

The two deployment shapes — DAL A edge and confidential datacenter —
share the same boot-trust chain primitives (RoT measurement, signature
verification, attestation), the same memory-safety hardening (MTE/PAC),
and the same crate workspace. They diverge only in their hardware-specific
trusted-execution mode (TrustZone/CCA on ARM, SEV-SNP/TDX on x86) and
their certification audience (FAA/EASA aviation vs FedRAMP datacenter).

## Proposal Inventory

The 15 NEW proposals (batches 1–4, currently being drafted on sibling
branches not yet pushed to `origin`) plus the 22 ACTIVE OpenSpec changes
relevant to the roadmap. Existing changes that don't intersect with the
new work (e.g. `llm-api-translation-v1`, `dynamic-batching-v1`,
`fp8-vision-inference-v1`) are listed for completeness but not sequenced
here — they ride on their own cadence.

### Legend

- **Tier 0** — Boot trust spine. Blocks almost everything else.
- **Tier 1** — Memory-safety hardening. Rides on Tier 0.
- **Tier 2** — Scheduling/safety guarantees. Independent of Tier 0/1.
- **Tier 3** — Datacenter scale-out. Independent of Tier 0/1, but the
  confidential-compute story rides on Tier 0.
- **Tier 4** — Supporting (formal proving, telemetry, automotive).

### Batch 1 — Memory Safety (5 proposals, NEW)

| Name | Tier | Status | Est. effort | Hard deps |
|---|---|---|---|---|
| `tegra-smmu-isolation-v1` | 1 | Draft | 6–8 weeks | `unikernel-orin-bringup-v1` Phase 2 complete, GPU access from kernel mode |
| `aarch64-mte-pac-hardening-v1` | 1 | Draft | 4–6 weeks | `unikernel-orin-bringup-v1` Phase 2 complete; A78AE MTE confirmed in EL1 |
| `spec-exec-mitigations-v1` | 1 | Draft | 3–4 weeks | None (cross-arch) |
| `ecc-scrubbing-v1` | 1 | Draft | 4–5 weeks | Hardware: Orin Industrial (ECC LPDDR5) OR datacenter ECC DIMMs |
| `cheri-capability-v1` | 1 | Draft | 8–12 weeks (research-grade) | Morello board — see Open Questions |

### Batch 2 — Boot / Attestation (4 proposals, NEW)

| Name | Tier | Status | Est. effort | Hard deps |
|---|---|---|---|---|
| `boot-root-of-trust-v1` | 0 | Draft | 6–8 weeks | x86: TPM2.0 driver (NEW). ARM: deferred to `op-tee-bridge-v1` |
| `op-tee-bridge-v1` | 0 | Draft | 5–7 weeks | `unikernel-orin-bringup-v1` Phase 2 complete (SMC dispatch needs working EL1 kernel) |
| `remote-attestation-v1` | 0 | Draft | 4–6 weeks | x86: `boot-root-of-trust-v1` Phase 1. ARM: `op-tee-bridge-v1`. PQC: existing ML-DSA-65 |
| `confidential-compute-v1` | 0/3 | Draft | 10–14 weeks | x86: `boot-root-of-trust-v1` + SEV-SNP or TDX hardware. ARM: CCA silicon (TBD H2 2026) |

### Batch 3 — Scheduling Safety (2 proposals, NEW)

| Name | Tier | Status | Est. effort | Hard deps |
|---|---|---|---|---|
| `deterministic-scheduling-v1` | 2 | Draft | 5–7 weeks | Existing scheduling model (no new hw) |
| `watchdog-lockstep-v1` | 2 | Draft | 8–10 weeks | Orin Industrial OR Zynq UltraScale+ (Cortex-R5F lockstep pair) — see Hardware Procurement |

### Batch 4 — Scale-out (4 proposals, NEW)

| Name | Tier | Status | Est. effort | Hard deps |
|---|---|---|---|---|
| `numa-aware-tensor-alloc-v1` | 3 | Draft | 4–5 weeks | Multi-socket x86 or AArch64 server hardware |
| `persistent-memory-v1` | 3 | Draft | 6–8 weeks | Intel Optane DCPMM (EOL) OR CXL.mem hardware (2026–2027 availability) |
| `gpu-mig-partitioning-v1` | 3 | Draft | 4–6 weeks | NVIDIA A100/H100/B200 (datacenter) — Jetson Orin does NOT support MIG |
| `tsn-integration-v1` | 3 | Draft | 6–8 weeks | TSN-capable NIC (Intel i225-IT or NXP LS1028A) supporting 802.1Qbv |

### Active OpenSpec changes relevant to this roadmap

| Name | Tier | Progress | Roadmap role |
|---|---|---|---|
| `unikernel-orin-bringup-v1` | 0 | 21/38 | **Critical-path.** Phase 2 TCU-post-EBS blocker gates almost everything ARM |
| `embedded-overlay-v1` | 0 | Phase 5 landed (PR #185) | Provides ML-DSA-65 model integrity policy that `boot-root-of-trust-v1` extends to the kernel binary |
| `embedded-filesystem-v1` | 0 | 0/131 | A/B boot infrastructure that `remote-update-v1` and `boot-root-of-trust-v1` chain through |
| `embedded-flash-fs-v1` | 0 | 0/71 | Raw-flash littlefs for MCU/FPGA targets; `cheri-capability-v1` and `fpga-*` reach further into that surface |
| `formal-proving-and-redteam-v1` | 4 | 0/119 | **Foundation for the DAL A claim map.** Phase 1 Lean-in-CI must land before any of the new tier-0/1 proposals can claim formal evidence |
| `management-login-v1` | 0 | 0/121 | Provides the auth surface that `remote-attestation-v1` quote-verification rides through |
| `remote-update-v1` | 0 | drafted | A/B rollback + ML-DSA-65 in-field updates; `boot-root-of-trust-v1` extends the measurement chain to updates |
| `automotive-bus-management-v1` | 4 | drafted | UDS-over-ISO-TP; `tsn-integration-v1` is its Ethernet sibling — sequence together |
| `fpga-accelerator-hal-v1` | — | 0/53 | Foundation for `fpga-dpu-backend-v1`, `fpga-custom-npu-v1`, `fpga-manager-v1` — independent of the 15-new track |
| `system-power-control-v1` | 2 | drafted | Provides PSCI reset path that `watchdog-lockstep-v1` ties into for fault-driven reset |
| `telemetry-otel-export-v1` | 4 | drafted | DAL A audit-evidence pipeline; `remote-attestation-v1` quote logs flow through it |
| `console-monitor-v1` | 4 | drafted | Independent; rides on `management-login-v1` |
| `network-management-v1` | 3 | drafted | Pre-req for `tsn-integration-v1` (multi-NIC routing & bonding) |

Not roadmapped here (independent cadence): `llm-api-translation-v1`,
`dynamic-batching-v1`, `fp8-vision-inference-v1`,
`onnx-full-coverage-roadmap-v1`, `codeql-quality-cleanup-v1`,
`project-usage-telemetry-v1`.

## Dependency Graph

ASCII below. Read top-to-bottom as time; arrows mean "blocks". Items
side-by-side parallelize. Existing active changes are in `[brackets]`;
new proposals are bare.

```
                                    Q3 2026
                                       │
        ┌──────────────────────────────┼──────────────────────────────────┐
        │                              │                                  │
[unikernel-orin-bringup-v1]    [formal-proving-and-redteam-v1]    spec-exec-mitigations-v1
   Phase 2 TCU post-EBS              Phase 1 (Lean-CI, SMT,             (cross-arch,
   (current blocker)                  TLA+ delegation)                   independent)
        │                              │
        │ unblocks ARM kernel-mode     │ unblocks DAL A claim format
        ▼                              ▼
                                    Q4 2026
        │                              │
        ├──────────────────┬───────────┴─────────────┬───────────────────┐
        ▼                  ▼                         ▼                   ▼
boot-root-of-trust-v1   op-tee-bridge-v1     aarch64-mte-pac-      ecc-scrubbing-v1
(x86 Phase 1: TPM2)     (ARM SMC dispatch +  hardening-v1          (independent, but
                         OP-TEE pseudo-TA)   (needs Phase 2 EL1)    ECC DIMM/Orin Ind.)
        │                  │                         │
        │ both feed remote-attestation              │
        └─────────┬────────┘                         │
                  ▼                                  │
                                    Q1 2027
        ┌─────────────────────┐                      │
        ▼                     ▼                      ▼
remote-attestation-v1   tegra-smmu-isolation-v1   deterministic-scheduling-v1
(consumes RoT quotes)   (needs Orin EL1 kernel +  (independent — TLA+ +
                         SMMU programming via       OperatorBudget rigor)
                         GIC/SMC path)
        │                     │                      │
        │ both feed CC story  │ unblocks GPU isolation
        ▼                     ▼                      │
                                    Q2 2027
        ┌──────────────────────────────────────┐    │
        ▼                                      ▼    │
confidential-compute-v1                  watchdog-lockstep-v1
  ├─ x86 SEV-SNP / TDX (requires hw)    (Orin Industrial Cortex-R52F
  └─ ARM CCA (requires Neoverse N3/V4    safety island OR Zynq R5F
     silicon, TBD shipment date)         lockstep pair)
        │
        ├─────────────┬─────────────┬──────────────┐
        ▼             ▼             ▼              ▼
                                    Q3 2027+
numa-aware-     persistent-     gpu-mig-       tsn-integration-v1
tensor-alloc-v1 memory-v1       partitioning-  (rides on
                                v1             network-management-v1)
                                (A100/H100/
                                 B200 only)

   Far future / research:
   cheri-capability-v1 — Morello-only until production CHERI ARM ships
```

The two parallel critical paths through the diagram:

- **ARM edge path:** `unikernel-orin-bringup-v1` Phase 2 → `op-tee-bridge-v1` →
  `aarch64-mte-pac-hardening-v1` + `tegra-smmu-isolation-v1` →
  `remote-attestation-v1` → `confidential-compute-v1` (ARM CCA when
  silicon ships).
- **x86 datacenter path:** `boot-root-of-trust-v1` Phase 1 (TPM2) →
  `remote-attestation-v1` → `confidential-compute-v1` (SEV-SNP or TDX).

Both paths share `remote-attestation-v1` as the convergence point. That
sequencing decision is what keeps the team from forking the codebase.

## Critical Path Analysis

The single highest-leverage unblock is **`unikernel-orin-bringup-v1`
Phase 2 task 2.10/2.11** — the TCU post-`ExitBootServices` UART access
that currently faults with SError EC=0x2F / sync translation fault.
Until that is resolved, every kernel-mode ARM feature
(`aarch64-mte-pac-hardening-v1`, `tegra-smmu-isolation-v1`,
`op-tee-bridge-v1`, `watchdog-lockstep-v1` on the Orin Cortex-R52F)
is gated by interim `con_out`-routed output that doesn't survive EBS.
The likely resolution paths, in order of cost:

1. **SMC into TF-A vendor namespace** — cheapest if NVIDIA exposes a
   TCU-print SMC in their reference TF-A. 1–2 weeks to discover and
   wire up.
2. **DTB parse + our own page tables + TCU MMIO map** — full solution.
   3–4 weeks. Path generalizes to every other Tegra234 MMIO peripheral.
3. **Status-quo: keep `con_out` until EBS deferred indefinitely** —
   not viable; UEFI services aren't allowed to persist into a DAL A
   kernel. Rule it out.

Phase 2 sub-PR 2e.1 (or whatever the follow-up labels it) is the gate
on which Q4 2026's ARM critical path hinges. **Recommend dedicating one
engineer to it for Q3 2026** without parallelizing onto adjacent
proposals.

Items that **can** parallelize Q3:

- `spec-exec-mitigations-v1` is cross-arch and touches only
  `compiler-fence` placement + KPTI-style page-table split — no
  hardware blocker. Land in Q3.
- `formal-proving-and-redteam-v1` Phase 1 (Lean-in-CI, SMT scaffold,
  delegation/scheduler TLA+ invariants) is a 3-week solo effort, no
  hardware blocker. Land in Q3.
- `boot-root-of-trust-v1` **x86 Phase 1** (TPM2.0 driver + Multiboot2
  measurement extend) is independent of the ARM TCU unblock. Land in
  Q3 on an x86 reference machine.

## Hardware Procurement Plan

| Hardware | Cost (USD, approx.) | Lead time | Proposals it unblocks | Acquire by |
|---|---|---|---|---|
| **Jetson AGX Orin Industrial** (64 GB, ECC, -40 to +85 °C, 10-yr supply, Cortex-R52F safety island) | $2,500–$3,500 | 8–12 weeks (NVIDIA Industrial channel, not Arrow) | `ecc-scrubbing-v1` (LPDDR5 ECC), `watchdog-lockstep-v1` (R52F), `aarch64-mte-pac-hardening-v1` (A78AE FEAT_MTE), `confidential-compute-v1` (Realm Management Extension on A78AE — disabled but present) | **Q3 2026** — order now |
| **x86-64 host w/ TPM 2.0 + SEV-SNP** (AMD EPYC 9004 series Genoa/Bergamo or 9005 Turin) | $4,000–$8,000 server + $300 TPM module if not soldered | 2–4 weeks (Dell/Supermicro standard SKU) | `boot-root-of-trust-v1` x86, `confidential-compute-v1` x86 (SEV-SNP), `remote-attestation-v1` x86 path | **Q3 2026** — order now |
| **Intel Xeon TDX-capable host** (Sapphire Rapids 4th gen Xeon Scalable w/ TDX BIOS enabled) | $4,000–$10,000 | 2–4 weeks; TDX BIOS enablement varies by OEM — confirm before purchase | `confidential-compute-v1` x86 alternate path (TDX) | **Q4 2026** — order after SEV-SNP path lands |
| **ARM Neoverse N3 or V4 (CCA)** — Ampere AmpereOne MX (announced) or AWS Graviton4 (TBD CCA enablement) | $unknown — possibly cloud-only until 2027 | **TBD — H2 2026 / H1 2027** | `confidential-compute-v1` ARM path | **Q1 2027 reassess** — see Open Questions |
| **Zynq UltraScale+ MPSoC** — KR260 Robotics Starter Kit (preferred; quad A53 + Cortex-R5F lockstep pair + ~256K LUTs) | $349 (Mouser/Digi-Key) | 2 weeks | `fpga-accelerator-hal-v1`, `fpga-dpu-backend-v1`, `fpga-manager-v1`, `fpga-custom-npu-v1`, plus a **secondary lockstep platform for `watchdog-lockstep-v1`** | **Q3 2026** — already on order per FPGA proposal track |
| **TSN-capable NIC** — Intel i225-IT (PCIe card) or NXP LS1028A (eval board) | $50 (i225-IT) – $500 (LS1028A devkit) | 1–2 weeks | `tsn-integration-v1`, ties into `network-management-v1` interface bonding | **Q1 2027** |
| **CXL.mem expander OR Optane DCPMM** | CXL: $unknown, vendor preview only as of 2026-05; Optane: end-of-life, secondary-market $300–$800/module | TBD | `persistent-memory-v1` | **Q2 2027 reassess** — likely defer to CXL availability |
| **NVIDIA H100 80GB or B200 (MIG-capable datacenter GPU)** | $25,000–$45,000 (H100 80GB SXM), B200 not on open market 2026-05 | 12+ weeks (allocation-constrained) | `gpu-mig-partitioning-v1` | **Q2 2027** — likely cloud-rent rather than buy; A100 PCIe at $10k is the fallback |
| **Morello board** (ARM research-grade CHERI prototype) | ~$1,000 (Arm Research loan, not retail) | 4–8 weeks (research-program application) | `cheri-capability-v1` — *only viable platform until production CHERI ARM* | **Q4 2026 apply** — multi-quarter horizon |

**Procurement priority for Q3 2026:**

1. Jetson AGX Orin Industrial (long lead time, gates the most work).
2. AMD EPYC SEV-SNP host (short lead, gates x86 confidential-compute).
3. KR260 (cheap, gates FPGA + lockstep secondary).

Total Q3 2026 procurement: ~$10,000–$15,000 capital.

## DO-178C Certification Claim Map

The DAL A claim is a composition of evidence statements. Each row below
maps a **claim** the certification audit will ask us to substantiate to
the **proposals** that produce the evidence. This is how the roadmap
funds itself: each proposal is not a feature in isolation, it is a row
in this table.

| Certification claim | Evidence proposal(s) | Status today | DO-178C objective |
|---|---|---|---|
| **Kernel binary integrity at boot** | `boot-root-of-trust-v1` (TPM2 PCR extend + ML-DSA-65 self-hash). Existing `verified-boot` Cargo feature covers post-load self-check; this extends the chain to firmware-side measurement. | Software-side hash only | A-3.5 (executable object code conforms to source) |
| **Model artifact integrity at load** | `embedded-overlay-v1` Phase 5 (landed, PR #185 — ML-DSA-65 over model upload) | **Done** | A-3.5 |
| **Memory-safety bounds checking** | `aarch64-mte-pac-hardening-v1` (synchronous MTE on heap allocs, PAC on return addresses), `cheri-capability-v1` (research-grade pointer-capability bounds — Morello only) | Rust type system + miri only | A-3.6 (verification of executable code) |
| **Transient-execution side-channel mitigation** | `spec-exec-mitigations-v1` (Spectre v1 LFENCE/CSDB, Spectre v2 retpolines, KPTI on x86, BHB clear on ARM) | None | A-3.7 (test coverage of high-level requirements) |
| **Memory ECC fault detection** | `ecc-scrubbing-v1` (LPDDR5 ECC scrubber driver + uncorrectable-error → safe-state transition) | None | A-3.7 + A-3.10 (executable code complete and correct) |
| **Input-output traceability (deterministic dispatch)** | `deterministic-scheduling-v1` (operator-boundary WCET budget enforcement, already partially landed via `timer-hal-wcet-v1`; this extends to MISRA-Rust + worst-case path analysis), `formal-proving-and-redteam-v1` Phase 1 (SchedStateMachine TLA+ invariants) | Partial — `OperatorBudget` live | A-5.1 (control flow analysis) |
| **Fault-detection coverage (single fault tolerance)** | `watchdog-lockstep-v1` (R52F lockstep checker on Orin Industrial, or A53-pair lockstep on Zynq R5F) | None | A-3.7 + DO-178C §6.4.4.2 |
| **GPU/peripheral isolation from kernel** | `tegra-smmu-isolation-v1` (per-engine SMMU context for GA10B host1x channels) | None — GPU has full DRAM view | A-3.6 |
| **Boot measurement attestation to operator** | `remote-attestation-v1` (TPM2 quote on x86, OP-TEE attestation key on ARM, PQC-signed) | None | A-6.3 (correctness of test cases — ground truth for attestation) |
| **Confidential model execution (data-at-rest + data-in-use)** | `confidential-compute-v1` (SEV-SNP/TDX/CCA Realm), `op-tee-bridge-v1` (key storage) | None | DAL A doesn't require this directly; FedRAMP customers do |
| **Update integrity (in-field)** | `remote-update-v1` (A/B + ML-DSA-65 chain extends through update), `embedded-filesystem-v1` (A/B boot infrastructure) | Drafted, not implemented | A-3.5 |
| **Formal verification evidence trail** | `formal-proving-and-redteam-v1` Phase 1 (Lean-in-CI, SMT for `BumpAllocator`, TLA+ delegation/scheduler invariants) | 22 TLA+ models in CI; Lean not gated | A-7.1–A-7.5 (verification methods) |
| **Adversarial / red-team test evidence** | `formal-proving-and-redteam-v1` Phase 2+3 (syscall ABI fuzzer, PQC differential KAT, capability red-team property tests) | None | A-7.3 (test coverage) |
| **Determinism on multicore (AMP partitioning)** | `deterministic-scheduling-v1` extends existing AMP-over-SMP model with `OperatorBudget` enforced quotas. Maps to ARINC 653 time-partitioning argument. | Partial | A-5.1 |
| **Audit log immutability** | Existing `mgmt-audit-log` capability (`management-login-v1`); `remote-attestation-v1` extends with PQC-signed quote receipts | Drafted | A-6.3 (records) |
| **Power/reset safety** | `system-power-control-v1` (PSCI / ACPI graceful path); `watchdog-lockstep-v1` (fault-driven reset) | Drafted | DAL A safety state |

The table makes funding easy to defend: every row is a claim the
certification body will ask for. The roadmap discharges them in an order
that doesn't double-back.

## Phase Plan (Quarterly)

### Q3 2026 (Jul–Sep 2026) — Unblock the ARM kernel + start the x86 trust spine

**Targets:**

- **Unblock TCU post-ExitBootServices on Tegra234.** Complete
  `unikernel-orin-bringup-v1` Phase 2 sub-PR 2e.1 + 2f (GICv3 + timer
  + minimal interrupt dispatch on real Orin hardware). **One engineer,
  full-time.** Acceptance: kernel banner from a kernel-side
  `tegra234_uart::putc()` over the J-class carrier's TTL header, with
  UEFI services exited. This is the single most important Q3
  deliverable.
- **Land `formal-proving-and-redteam-v1` Phase 1.** Lean-in-CI hard gate
  (`lean-verify` job), Z3 SMT scaffolding behind `formal-smt` feature,
  `CapabilitySecurity.tla` delegation-monotonicity + `SchedStateMachine.tla`
  fairness invariants in TLC. Acceptance: all 6 Lean proofs type-check
  in CI on every PR, 1 SMT proof of `BumpAllocator` monotonicity
  passing.
- **Land `spec-exec-mitigations-v1`.** Cross-arch: Spectre v1 fences
  on capability-handle lookup paths, Spectre v2 retpolines (x86) /
  BHB clearing (ARM), KPTI-equivalent for x86 kernel/user split (note:
  unikernel single-address-space makes KPTI partly NA — the proposal
  scopes this honestly).
- **Begin `boot-root-of-trust-v1` Phase 1 (x86 only).** TPM2 CRB
  interface driver, Multiboot2 measurement extend into PCR 8/9
  (kernel) and PCR 10 (model artifacts). Acceptance: `tpm2_pcrread`
  on the host machine before and after boot shows measurements
  matching `docs/security/boot-measurements-v1.md` golden values.
- **Procure hardware.** Place Orin Industrial order, EPYC SEV-SNP host
  order, KR260 order (if not already in-flight via FPGA track).
- **Drive existing active changes opportunistically.**
  `embedded-filesystem-v1` is at 0/131 and blocks `remote-update-v1`
  signature-chain integration; it shouldn't slip later than Q4 2026.

**End-of-Q3 acceptance criteria:**

- Kernel-side TCU output post-EBS works on Orin.
- Lean proofs type-check in CI.
- 1 SMT proof passing on `BumpAllocator`.
- Spectre v1/v2 + KPTI-equivalent landed on x86 + ARM unikernel builds.
- x86 boot-root-of-trust measures kernel into TPM2 PCRs.
- Hardware ordered and ship dates confirmed.

### Q4 2026 (Oct–Dec 2026) — ARM trust spine + memory safety

**Targets:**

- **Land `op-tee-bridge-v1` (ARM trust spine).** SMC dispatch to OP-TEE
  Pseudo-TA for key storage + attestation key generation. Builds on the
  Q3 TCU unblock — needs working EL1 kernel with SMC capability.
  Acceptance: SmallAIOS can request a fresh ML-DSA-65 keypair generated
  inside OP-TEE and use its public key for outbound TLS handshakes.
- **Land `aarch64-mte-pac-hardening-v1`.** Synchronous MTE on every
  heap allocation (validates Orin A78AE EL1 MTE — confirm in
  proposal's task 0); PAC on return addresses via Rust `-C
  target-feature=+pauth`. Acceptance: deliberate `*core::ptr::null_mut()`
  write traps cleanly with MTE tag mismatch; ROP test case (in
  `red-team-tests/`) fails to chain on PAC-enabled build.
- **Land `boot-root-of-trust-v1` Phase 2 (ARM).** TF-A measured-boot
  event log integration. Reuses Phase 1 PCR extend semantics but routes
  through OP-TEE attestation key for signing. Depends on
  `op-tee-bridge-v1`.
- **Land `remote-attestation-v1`.** Once x86 TPM2 measurement +
  ARM OP-TEE attestation key exist, both feed the same attestation
  quote format (PQC-signed, ML-DSA-65). Acceptance: a stub remote
  verifier (test harness) accepts a quote from each of x86 and ARM,
  validates the chain, and ties to a documented PCR/measurement
  baseline.
- **Land `ecc-scrubbing-v1`.** Orin Industrial LPDDR5 ECC driver +
  Linux-style scrubber + uncorrectable-error → safe-state transition
  (escalates to `system-power-control-v1` graceful shutdown or
  `watchdog-lockstep-v1` reset path when those land).
- **Land `embedded-filesystem-v1` Phase 1.** A/B boot partition
  infrastructure. Unblocks `remote-update-v1` and the boot-trust chain
  extending through field updates.

**End-of-Q4 acceptance criteria:**

- Both x86 and ARM produce PQC-signed attestation quotes.
- MTE + PAC enabled on Orin Industrial unikernel builds.
- ECC scrubber running, uncorrectable-error path tested with
  fault-injection.
- A/B boot partitions selectable on x86 + Orin.

### Q1 2027 (Jan–Mar 2027) — GPU isolation + scheduling rigor

**Targets:**

- **Land `tegra-smmu-isolation-v1`.** Per-engine SMMU context for the
  GA10B host1x channels. Requires the kernel to have working SMMU
  programming (currently the GPU sees full DRAM). Acceptance: a
  deliberately-malicious operator-supplied tensor pointer outside the
  inference workload's SMMU context faults inside the GPU rather than
  reading kernel memory. **Note:** this is the proposal most likely to
  hit surprise hardware quirks — see Risk Register.
- **Land `deterministic-scheduling-v1`.** Extends the existing
  `OperatorBudget` enforcement (timer-hal-wcet-v1) with WCET
  ground-truth measurement + a stricter `SchedStateMachine` TLA+ proof
  of bounded jitter under worst-case interference. Maps to ARINC 653
  time-partitioning evidence for the DAL A claim.
- **Land `remote-update-v1`.** A/B in-field updates with ML-DSA-65
  signature verification. Builds on Q4's `embedded-filesystem-v1`
  Phase 1 + `boot-root-of-trust-v1`. Acceptance: a deliberately-bad
  signed image rolls back to slot A within the watchdog timeout; a
  good signed image survives reboot.
- **Land `formal-proving-and-redteam-v1` Phase 2+3.** Syscall ABI fuzzer
  (`fuzz_syscall_abi`), PQC NIST KAT gate, PQC differential fuzz vs
  `pqcrypto-{kyber,dilithium}`, red-team property-test crate. Adversarial
  evidence accrues here for the DAL A audit trail.
- **Start `tsn-integration-v1`** (hardware-permitting). Requires
  Intel i225-IT or NXP LS1028A in hand. Rides on `network-management-v1`'s
  multi-interface routing.

**End-of-Q1 acceptance criteria:**

- GPU isolation: SMMU context fault tested.
- WCET bounded jitter measured + documented.
- A/B rollback works in field-update scenario.
- All formal-proving-and-redteam phases landed; red-team scenarios
  cover the threat model rows.

### Q2 2027 (Apr–Jun 2027) — Fault tolerance + first confidential-compute

**Targets:**

- **Land `watchdog-lockstep-v1`.** On Orin Industrial: Cortex-R52F
  safety island acts as lockstep checker for the inference path.
  Acceptance: deliberate fault injection (CPU halt, MMU misconfig)
  triggers R52F-initiated reset within the watchdog budget.
- **Land `confidential-compute-v1` x86 Phase 1 (SEV-SNP).** SmallAIOS
  boots inside an AMD SEV-SNP guest, attestation quote includes
  AMD's PSP attestation report bound to the SmallAIOS measurement.
  Customer's remote verifier validates both. **Open:** Intel TDX
  alternate path lands here or slips to Q3 depending on TDX hardware
  availability.
- **Begin `confidential-compute-v1` ARM Phase 1 (CCA).** Only
  if Neoverse N3/V4 silicon has shipped. Otherwise: design-only this
  quarter, implementation slips to Q3 2027 or later.
- **Land `numa-aware-tensor-alloc-v1`.** First multi-socket server
  proposal. Acceptance: ResNet-50 throughput on a dual-socket EPYC
  9004 box is ≥1.7× single-socket (vs 2.0× ideal — NUMA-aware
  allocation closes the gap).
- **Polish DAL A audit dossier draft.** All evidence rows in the
  certification-claim map should have at least an "acceptance criterion
  met" date. Begin engaging an SAR (Software Approval Representative)
  for early review of evidence-claim composition.

**End-of-Q2 acceptance criteria:**

- Watchdog/lockstep fault injection test green.
- One confidential-compute path (SEV-SNP) end-to-end with attestation.
- NUMA-aware tensor allocator measurably reduces remote-NUMA traffic.

### Q3 2027 and beyond — Scale-out + research

- **`confidential-compute-v1` ARM CCA path** (silicon-availability-gated).
- **`gpu-mig-partitioning-v1`** (H100/B200 datacenter GPU, cloud rent
  or owned).
- **`persistent-memory-v1`** (CXL.mem availability-gated — assume
  H2 2027).
- **`tsn-integration-v1`** if not landed in Q1/Q2.
- **`automotive-bus-management-v1`** UDS-over-ISO-TP rides alongside
  TSN.
- **`cheri-capability-v1`** (Morello research artifact; no production
  schedule until production-grade CHERI ARM silicon — TBD multi-year).
- **DO-178C DAL A formal certification submission.** Target: end of
  Q4 2027 — dossier complete by Q3 2027, 1 quarter for SAR review +
  iteration.

## Risk Register

The five proposals with the highest schedule/scope risk.

### Risk 1 — `confidential-compute-v1` (HIGH)

**Risk:** ARM CCA production silicon (Neoverse N3 / V4 with Realm
Management Extension fused on) is "announced for late 2026" as of this
roadmap's write date. NVIDIA's Grace CPU is Neoverse V2 — **no CCA**.
Ampere AmpereOne MX has been announced but ship dates and CCA enablement
are unclear. AWS Graviton4 (Neoverse V2) may add CCA in a refresh — TBD.

**Impact:** ARM confidential-compute slips to Q3 2027 at earliest, possibly
indefinitely. Customer story for confidential AI inference on ARM is
deferred.

**Mitigation:**
- x86 SEV-SNP path lands first (Q2 2027), provides confidential-compute
  story for AMD EPYC datacenter customers regardless of ARM silicon.
- ARM CCA proposal stays in draft with design-only work — when silicon
  ships, implementation is a quarter not a year.
- Consider Intel TDX as third path (Sapphire Rapids 4th gen, available
  now) — extra implementation cost but reduces concentration risk on
  AMD.

### Risk 2 — `watchdog-lockstep-v1` (HIGH)

**Risk:** Cortex-R52F safety island programming on Tegra234 is poorly
documented. NVIDIA's reference is buried inside Drive AGX Orin's
DRIVE OS — accessible only under NDA to automotive Tier 1 customers.
Without that documentation, we are reverse-engineering the IPC channel
between A78AE and R52F, the R52F's own boot path, and the lockstep
checker programming model.

**Impact:** Could slip 1–2 quarters. Worst case, drop to the **Zynq R5F
lockstep path** which is fully open-source but a different SKU.

**Mitigation:**
- Apply for NVIDIA Drive AGX developer program access in Q3 2026 — gives
  access to the R52F manual under NDA, which is fine for engineering
  but the resulting Rust driver code is unencumbered.
- Maintain the Zynq UltraScale+ R5F lockstep path as a parallel demo
  (uses the KR260 already on order for FPGA work). DAL A claim works
  on either platform — the claim is "fault detection coverage", not
  "fault detection coverage on Orin specifically".
- Document scope clearly: v1 of this proposal lands on whichever
  platform unblocks first.

### Risk 3 — `persistent-memory-v1` (MEDIUM-HIGH)

**Risk:** Intel Optane DCPMM is **end-of-life** (Intel canceled the
product line in 2022). CXL.mem is the successor but production silicon
in volume is 2026/2027. Module availability + CPU support is fragmented
across Sapphire Rapids 4th gen (CXL 1.1), Emerald Rapids 5th gen
(CXL 1.1), Granite Rapids (CXL 2.0). AArch64 CXL.mem support is even
less mature.

**Impact:** Without persistent memory hardware, the proposal becomes a
QEMU-only demo. Real customer value is gated by CXL.mem ecosystem
maturity.

**Mitigation:**
- Reframe v1 of the proposal as **"persistent memory abstraction trait"**
  rather than a concrete platform. Trait lives in `compute` or a new
  `pmem` crate; backends are pluggable. QEMU-based `pmem_emul` driver
  for development, real CXL.mem driver lands when hardware does.
- Defer until Q3 2027 minimum. Recheck CXL.mem availability quarterly.

### Risk 4 — `op-tee-bridge-v1` (MEDIUM)

**Risk:** OP-TEE on Tegra234 specifically requires NVIDIA-blessed
secure-world firmware. JetPack 6 ships OP-TEE OS but the TA-signing
keys live with NVIDIA — we cannot ship our own pseudo-TAs without
NVIDIA cooperation. The fallback (run our own OP-TEE OS) requires
re-fusing the Orin's secure-boot keys, which is irreversible and
voids the unit for L4T container path return-trips.

**Impact:** ARM key-storage + attestation key generation path is
gated on either NVIDIA's TA signing cooperation or a fused unit we
can't reuse. Could slip 1 quarter while we work the NVIDIA
relationship.

**Mitigation:**
- Open the conversation with NVIDIA Drive / Jetson developer relations
  in Q3 2026 — same channel as the Risk 2 NDA conversation.
- Document a **second-best fallback**: TF-A's Standard Services SMC
  interface (open-source, no TA signing required) for a strict
  subset of operations (random number generation, monotonic counter,
  hash extend). This won't provide attestation-key generation but
  unblocks `boot-root-of-trust-v1` Phase 2 on ARM as a Tier-A
  fallback.
- For non-Tegra ARM (Zynq, Ampere), OP-TEE is unencumbered.

### Risk 5 — `aarch64-mte-pac-hardening-v1` (MEDIUM)

**Risk:** Cortex-A78AE is **announced** to support FEAT_MTE in EL0/EL1.
We have not yet confirmed it's enabled in JetPack 6's Tegra234 fuse
configuration. NVIDIA may have disabled FEAT_MTE for production yield
reasons — happened on early Cortex-A78 silicon.

**Impact:** If MTE is fused-off, this proposal becomes "PAC + software
heap canaries only" on Tegra234. PAC alone is a worthwhile DAL A
evidence row, but MTE is the headline. Could also slip to a non-Tegra
ARM platform.

**Mitigation:**
- Confirm FEAT_MTE availability in Q3 2026 by reading the Orin's
  `ID_AA64PFR1_EL1` register from EL1 (once the post-EBS kernel
  works). Cost: one afternoon. **Schedule this as part of
  `unikernel-orin-bringup-v1` Phase 2 acceptance, not as a separate
  task.**
- Have a fallback PAC-only mode in the proposal's scope.

## Open Questions

These need user or leadership input before committing to the sequence
above.

1. **`cheri-capability-v1` — Morello-only is fine, or kill until
   production silicon?** Morello is a research-grade ARM-Research board,
   not a production SKU. CHERI ARM in production silicon is multi-year
   horizon. Two options: (a) keep `cheri-capability-v1` as an active
   research track on Morello and let it inform the eventual production
   port; (b) archive it until production CHERI ARM exists. **Recommend
   (a)** — applying for Morello access is cheap, the research informs
   the rest of the memory-safety track.

2. **TDX vs SEV-SNP first?** Both are mature in 2026. SEV-SNP has wider
   datacenter deployment (AMD EPYC market share). TDX has cleaner
   integration with Intel's DCAP attestation flow that many
   confidential-compute customers already use. **Recommend SEV-SNP
   first** because the AMD EPYC procurement is already on the Q3 2026
   list; TDX as Q3 2027 follow-up.

3. **DAL A scope — Orin Industrial only, or AGX Xavier Industrial as
   well?** Xavier Industrial has 10-year supply guarantees and is already
   shipping in DAL A-targeting customer programs (Sierra Nevada, Boeing).
   But it's a different SoC (Carmel cores vs A78AE, GICv2 vs GICv3,
   different SMMU). Supporting both doubles the BSP burden.
   **Recommend Orin Industrial only for v1 of the DAL A claim**; add
   Xavier Industrial as a v2 if customer demand materializes.

4. **`tsn-integration-v1` vs `automotive-bus-management-v1` ordering?**
   Both are Tier 4 in this roadmap. TSN is Ethernet, CAN/UDS is the
   diagnostic bus. They share no code but share an operator audience.
   **Recommend TSN first (Q1 2027)** because the TSN-capable NIC is
   $50 (Intel i225-IT) and the CAN bus work needs a real automotive
   testbed.

5. **Drafting-branch landing order.** The 4 sibling agents drafting
   batches 1–4 may land their PRs in any order, but the dependency
   graph in this doc presumes batches 1 (memory-safety) + 2
   (boot/attestation) land first. **Recommend reviewing batch 2 PRs
   first** when they push, then batch 1, since batch 1 has more hard
   dependencies on batch 2's boot-trust spine.

6. **Sequencing conflict spotted:** `confidential-compute-v1` lists
   x86 SEV-SNP / TDX *and* ARM CCA. The proposal might be too broad —
   consider splitting into `confidential-compute-x86-v1` and
   `confidential-compute-arm-v1` when batch 2 lands, since their
   schedules are years apart. **Document this as a v1.1 reorganization,
   not a v1 blocker.**

7. **What's in the in-flight `develop` reorg?** The local working tree
   shows a large staged deletion of `fs/`, `mgmt/`, `auth/`,
   `formal/coq/`, `formal/spin/` paths plus renames out of
   `openspec/changes/archive/`. This roadmap assumes the reorg
   preserves the crate set; if it doesn't, the architecture section
   above will need a revision pass.
