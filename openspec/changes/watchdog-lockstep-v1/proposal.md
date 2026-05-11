# watchdog-lockstep-v1

## Summary

Two related safety primitives that together provide the "fault detection" surface DO-178C DAL A and ISO 26262 ASIL-D require for an AI inference workload running on a single SoC:

- **Phase 1 — Software watchdog (Tier 3a, ~2-3 weeks).** A kernel timer-driven watchdog task running in `SchedulingClass::System` that fires if the inference task doesn't check in within a configurable deadline. Two recovery actions are supported: (a) panic + reboot via the existing `kernel::state` shutdown path, or (b) fall through to an alternate model (a smaller, faster, certified-baseline model) and continue serving. The watchdog plumbing already has stub syscall entry points (`sys_watchdog_pet` and `sys_watchdog_remaining` in `kernel/src/syscall/system.rs`) — this change wires them to real per-task deadlines, integrates with the cooperative scheduler's tick infrastructure (`kernel/src/sched/timer.rs`), and adds the alternate-model fallback escape hatch on the ONNX runtime side. Independent of any specific Tegra hardware — runs on every supported platform.
- **Phase 2 — Dual-core lockstep (Tier 3b, ~4-5 weeks).** For Tegra Orin Industrial / Automotive SKUs that ship A78AE cores with hardware split/lock support, the kernel SHALL configure two cores into hardware lockstep and execute the same inference path on both replicas, voting outputs at operator boundaries. The Cortex-A78AE's *AE* designator specifically denotes Arm's Automotive Enhanced core with native split/lock mode — when the two cores are in lock mode, they execute the same instruction stream and Arm's lockstep comparator hardware automatically detects mismatches (soft errors caused by radiation, transient faults, single-event upsets). The unikernel observes this as either "everything is fine and the inference completes" or "the lockstep comparator raised a fault" — handled by a new abort path. Voting at op boundaries means a soft fault is contained to a single operator: replay the operator, re-vote, escalate if it diverges a second time. **Crucial hardware prerequisite:** the J4012 Orin NX dev kit's silicon does NOT ship in the Industrial/Automotive AE-locked SKU — this work is gated on access to an Orin Industrial reference platform (P3737-derived module on a carrier that exposes the lockstep-mode strap pins). The dev-kit-only deployment path falls back to the software-watchdog-only mode in Phase 1.

Both phases compose with `deterministic-scheduling-v1` (the sibling change in this batch). Watchdog runs in all modes; lockstep is meaningless without determinism (a lockstep vote between two non-deterministic replicas can legitimately disagree from timing alone, so the vote is unsound). The hard dependency is documented as a precondition in this proposal's "Sequencing" section.

## Why

- **DO-178C DAL A "fault detection coverage" credit needs both layers.** DAL A requires the system to detect and respond to faults that affect safety-critical outputs. A software watchdog catches *liveness* faults (the inference is stuck or has crashed silently) and unlocks credit for the "the system cannot remain in a faulted state for more than `T_deadline`" certification claim. Hardware lockstep voting catches *correctness* faults (the inference completed but produced the wrong answer due to a soft error) and unlocks credit for the "single-bit transient faults are detected and recovered" claim. Without both, the DAL A claim is incomplete.
- **ISO 26262 Diagnostic Coverage for ASIL-D inference paths.** Automotive deployments on Orin Industrial demand ASIL-D coverage. The single-channel inference path on a single A78AE core gives at most ASIL-B coverage; dual-channel lockstep with voting is the standard route to ASIL-D. The lockstep comparator hardware in the A78AE is *specifically* there for this purpose — Arm sells the AE variant into automotive precisely for the safety credit.
- **The watchdog plumbing is half-built and stubs are misleading.** `kernel/src/syscall/system.rs` already exposes `sys_watchdog_pet` (syscall 0x55) and `sys_watchdog_remaining` (syscall 0x56), but both currently `TODO: Write to hardware watchdog service register` and the latter unconditionally returns `30000`. The scheduler's `TaskType::Watchdog` exists with `SchedulingClass::System` priority and the `kernel/src/sched/timer.rs` module documents `Watchdog tick integration` as a goal — but no actual watchdog task is spawned at boot, and no real deadline is checked. This change finishes that wiring with real semantics, then adds the lockstep layer on top.
- **The alternate-model fallback is what makes the watchdog useful in production.** A bare "panic + reboot" watchdog is correct but unfriendly: the inference service goes dark for the boot duration. The alternate-model fallback path lets the system continue serving with a degraded (smaller, faster, pre-certified) model while the primary path is re-initialized or escalated to a higher-level recovery. This is the standard ARINC-653-style "alternative-mode" pattern, applied to inference workloads.
- **A78AE split/lock is documented but unused.** The Tegra234 BSP work in `unikernel-orin-bringup-v1` (Phase 2) lands the GICv3 + UART foundation but does not touch the A78AE-specific safety features. This change adds a `arch-aarch64-lockstep` capability covering the boot-time configuration of the cluster into lock mode (via the `CPUACTLR_EL1` / `CLUSTERACTLR_EL1` / `RVBAR` / `EDPRCR` setup the A78AE TRM documents) and the runtime handling of lockstep-comparator faults (via the existing AArch64 SError / synchronous-exception path with new diagnostic decoding).

## Phase 1 — Software watchdog (independent, ships first)

### What changes

- A `Watchdog` task is spawned at `kernel::init` time with `SchedulingClass::System` and `TaskType::Watchdog`. The task runs once every `T_check = 100ms` (configurable via boot arg) and inspects all registered inference tasks for their `last_pet_tick` field.
- `sys_watchdog_pet` writes the current scheduler tick (from `kernel/src/sched/timer.rs::TICK_COUNT`) into the calling task's `last_pet_tick`. The existing syscall surface stays unchanged; the implementation moves from `TODO` to real plumbing.
- `sys_watchdog_remaining` returns `(deadline_ticks - (current_tick - last_pet_tick))` clamped to non-negative.
- When the watchdog task observes `(current_tick - last_pet_tick) > T_deadline` for any inference task, it triggers the configured fault response: `WatchdogPolicy::PanicReboot` (default) or `WatchdogPolicy::FallbackModel`.
- `WatchdogPolicy::FallbackModel` requires the application to have pre-registered a fallback `Session` via a new `Session::register_fallback(primary)` API. On watchdog fire, the runtime swaps the primary session out, brings up the fallback session, and continues serving with a `FallbackEngaged` metric incremented in `onnx-rt/src/profile.rs`.
- The cooperative scheduler's existing `should_continue_inference` check (per `docs/scheduling-model.md` design rule 4) is extended to also consult the watchdog state — if the watchdog has fired, the next op-boundary yield aborts the inference rather than returning to it.
- Default deadline values are documented in `docs/watchdog.md` (new): 100ms for inference tasks, 1s wall-clock cap for the whole inference, configurable per-task via a new `SessionConfig::watchdog_deadline_ms` field.

### What does not change

- The two existing watchdog syscall numbers (0x55, 0x56) and their POSIX-ish ABI stay the same. The implementations are filled in; the contract is unchanged.
- The hardware watchdog register (Tegra234 WDT, Cortex-A78AE secure-watchdog) is *not* touched by Phase 1 — that's still a TODO in `sys_watchdog_pet`. The software watchdog is sufficient for inference-task liveness; hardware-watchdog integration is a separate concern tracked for a follow-up.

## Phase 2 — Dual-core lockstep (gated on Industrial hardware)

### What changes

- A new `arch-aarch64-lockstep` capability is added to `arch/aarch64/src/lib.rs` (gated on a new `lockstep` Cargo feature on `smallaios-arch-aarch64`, which is itself gated on `tegra234-industrial` — see "Hardware prerequisites" below).
- At boot (in `boot.rs` for `aarch64-unknown-none` or `boot_uefi.rs` for the UEFI path), if `lockstep` is enabled, the kernel writes the A78AE `CLUSTERECTLR_EL1` / `CLUSTERACTLR_EL1` "split/lock mode" bits to put cores 0 and 1 into hardware lock mode *before* the GICv3 initialization. The Arm Cortex-A78AE TRM (DDI 0626) section 4.5.1 "Split/lock mode" is the reference; the BSP-specific Tegra234 documentation for which strap pins to assert is the second reference (NVIDIA-confidential; we cite the documented procedure without reproducing it).
- The lockstep-comparator fault, raised by the A78AE's compare unit when the two cores' outputs diverge, surfaces as an `SError` or synchronous abort to the leader core. A new `arch::aarch64::lockstep` module decodes the EC + ISS bits to identify a lockstep fault distinctly from a normal page fault, alignment fault, or other AArch64 exception cause. On lockstep fault: the kernel logs the diagnostic, marks the current operator as "needs replay", and returns control to the executor's voting path.
- A new `LockstepVoter` type in `onnx-rt/src/lockstep.rs` runs alongside the executor. At every operator boundary (the same boundary where the cooperative scheduler yields per `docs/scheduling-model.md`), the voter compares the operator's output tensor with the replica's output tensor. Two strategies:
  - **Hardware-comparator mode** (preferred, A78AE lock mode): the comparator hardware does the bit-compare automatically and raises a fault on mismatch. The voter consumes the fault event.
  - **Software-comparator mode** (fallback, for testing on non-AE hardware): the voter explicitly bit-compares the two output tensors. Used for CI verification on QEMU (which does not model the AE comparator) and on dev-kit Orin (where the silicon does not support lock mode).
- A divergence triggers operator replay (re-run the operator with the same inputs, re-vote). A second divergence escalates to the watchdog's fault policy (typically `PanicReboot`).
- The voter's compare strategy is selected at runtime based on the detected platform: hardware-comparator on Orin Industrial, software-comparator on dev kits and CI runners. Both are spec'd; tests cover both.

### Hardware prerequisites — important

- **The Seeed reComputer J4012 (Jetson Orin NX 16 GB dev kit) does NOT support A78AE lockstep mode.** Arm and NVIDIA segment the A78AE silicon: the AE-AS (Automotive Specific) variants ship the lock-mode strap pins; the consumer-grade Orin Nano, Orin NX, AGX Orin dev kit variants do not. This is a hardware-level distinction baked into the Tegra234 silicon, not a Cargo-feature or firmware-switchable knob. The Orin Industrial (Drive Orin AGX Industrial / Hyperion Industrial reference platforms) and the automotive Drive Orin variants are the SKUs that ship lockstep-capable silicon.
- **Implication for verification:** Phase 2 development can proceed on dev-kit hardware using `software-comparator` mode (which exercises the voting logic, the replay-on-divergence path, and the escalation to the watchdog) but the *hardware-comparator* mode can only be verified end-to-end on an Industrial reference platform. This is called out in `tasks.md` as a "platform verification gate" — Phase 2 ships software-comparator-tested only; hardware-comparator is gated on hardware access and lands as a follow-up sub-PR once we have access.
- **The `software-comparator` mode is itself a useful safety primitive** even on dev kits: two sessions running the same model on the same input should produce bit-identical outputs in deterministic mode (per `deterministic-scheduling-v1`), so a divergence in software-comparator mode catches Heisenbugs in the runtime regardless of whether the underlying hardware is AE-locked. We ship it as a first-class mode, not just a test stub.

### What does not change

- The default boot path. Lockstep is opt-in via the `lockstep` Cargo feature *and* the `--lockstep` runtime flag *and* the presence of detected Industrial silicon. Three layers of opt-in so that nobody accidentally turns lockstep on and pays the 2× core cost for nothing.
- The cooperative scheduler's yield-at-op-boundary contract. The voter hooks into the same boundary the scheduler uses; it does not introduce a new preemption point.
- The single-stream CUDA path enforced by deterministic mode. Lockstep voting compares CPU-side output tensors after each operator; GPU-side execution remains single-threaded per replica.

## Relation to prior work

- **Depends on `deterministic-scheduling-v1`.** Lockstep voting between two replicas requires that both replicas produce identical outputs in the absence of a fault. Non-deterministic mode permits legitimate divergence (different RNG draws, different multi-stream order), which makes voting unsound. The watchdog half of this proposal does not depend on determinism and can ship first if needed.
- **Depends on `unikernel-orin-bringup-v1` Phase 2 (Tegra234 BSP).** The lockstep boot-time configuration writes to A78AE cluster registers that are only accessible once the Tegra234 BSP has reached the early-boot phase where the cluster topology is known. Phase 1 of *this* change (software watchdog) is platform-agnostic and can ship before the BSP.
- **Composes with `timer-hal-wcet-v1` (archived).** The watchdog uses the same `Timestamp::now()` primitive that `timer-hal-wcet-v1` wired up. No new timer infrastructure is needed.
- **Extends the existing `safety-critical` capability spec** (`openspec/specs/safety-critical/spec.md`) which today covers DO-178C process compliance. We add the *implementation* surface — the watchdog task, the lockstep voter — that the process spec already implies should exist.

## Out of scope

- **Hardware watchdog register integration.** The Tegra234 WDT (`/soc/watchdog@...`) and the secure-watchdog accessed via TF-A are reachable but require firmware-side configuration (the WDT can only be poked in EL3 by default). A future change can wire `sys_watchdog_pet` to the hardware WDT in addition to the software deadline check; v1 of this change covers software-watchdog only.
- **Triple Modular Redundancy.** True TMR needs three cores plus a majority voter and recovers from a single faulty replica. The A78AE only supports dual lockstep; TMR would require a separate voter implementation outside the AE silicon or a different platform entirely. Tracked as a "future safety" stretch; explicitly out of scope here.
- **Asymmetric replicas (different runtimes voting).** Voting between a CPU-only and a GPU-accelerated replica is interesting (catches numerical bugs in either path) but compounds with quantization, kernel-selection differences, and floating-point reduction-order differences. Out of scope; v1 votes between two byte-identical sessions only.
- **Network-attached redundancy.** Voting across a network link (one replica on the inference node, one on a paired safety node) is a higher-tier safety concept and needs network determinism that we don't have. Out of scope.
- **Failover between primary and fallback models triggered by anything other than watchdog timeout.** E.g. failover on accuracy drop, on temperature throttling, on power-cap. Each of these is its own change.

## Sequencing

| Order | What | Why this order |
|-------|------|----------------|
| 1 | `deterministic-scheduling-v1` lands first | Required so lockstep replicas can produce bit-identical outputs |
| 2 | `unikernel-orin-bringup-v1` Phase 2 lands the Tegra234 BSP | Required so lockstep boot-time config can write to A78AE cluster registers |
| 3 | This change Phase 1 (software watchdog) lands | Independent of (2); can land in parallel with `deterministic-scheduling-v1` |
| 4 | This change Phase 2 software-comparator mode lands | Requires (1) and (3); does not require (2) for development, only for end-to-end |
| 5 | This change Phase 2 hardware-comparator mode lands | Requires (2) and access to Orin Industrial reference platform |

Phases 1 and 2 of this change can be split into separate PRs to keep review tractable; the proposal narrates both because they share a capability boundary (`kernel-safety`) and share a verification surface (the same fault-handling test suite covers both, with the voter as the new piece).

## Effort estimate

| Sub-area | Scope | Estimate |
|----------|-------|----------|
| Phase 1 — software watchdog + alternate-model fallback | `kernel/src/sched/`, `onnx-rt/src/session.rs` | ~2-3 weeks |
| Phase 2 — `LockstepVoter` software-comparator mode | `onnx-rt/src/lockstep.rs`, executor hooks | ~1-2 weeks |
| Phase 2 — A78AE lock-mode boot configuration | `arch/aarch64/src/lockstep.rs`, `boot.rs` plumbing | ~2-3 weeks (gated on Industrial hardware access) |
| Phase 2 — fault decoder + replay path | `arch/aarch64/src/interrupts.rs` extension, executor replay | ~1 week |
| Docs + CI + tests | `docs/watchdog.md`, `docs/lockstep.md`, CI advisory jobs | ~1 week |
| **Total** | | **~6-8 weeks** |
