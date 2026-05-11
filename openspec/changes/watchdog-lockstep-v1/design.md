# Design — watchdog-lockstep-v1

## Goal

Add the two missing safety-critical primitives for AI inference on a single SoC: a software watchdog (catches liveness faults; the inference is stuck or has crashed) and dual-core hardware lockstep voting (catches correctness faults; the inference completed but produced a wrong answer due to a soft error). Both are required to unlock DO-178C DAL A "fault detection coverage" credit and ISO 26262 ASIL-D Diagnostic Coverage credit on the certifiable inference path.

The deliverables are spec'd separately because they have different hardware prerequisites (watchdog: none; lockstep: Orin Industrial A78AE-AS silicon) but they live behind a single `kernel-safety` capability boundary because they share a fault-handling surface — the watchdog is what fires when the lockstep replay-and-revote sequence escalates.

## Phase 1 — software watchdog

### Why a software watchdog when the syscall surface implies hardware

`sys_watchdog_pet` (0x55) and `sys_watchdog_remaining` (0x56) are documented in `kernel/src/syscall/system.rs` as "service the hardware watchdog timer" and "query remaining watchdog time (in milliseconds)". The current implementations are stubs (`TODO: Write to hardware watchdog service register`) and the second unconditionally returns 30000ms.

A real hardware watchdog on Tegra234 lives in two places: (a) the Tegra WDT (`/soc/watchdog@2380000` in the JetPack DTB), accessible from EL2 with the appropriate firmware-side configuration, and (b) the secure WDT accessed via TF-A in EL3. Either one requires Tegra-specific bring-up that's gated on the `unikernel-orin-bringup-v1` Phase 2 BSP. Today we don't have it.

What we *do* have is the cooperative scheduler's `kernel/src/sched/timer.rs` tick infrastructure (`TICK_COUNT` atomic, monotonic). We can build a *software* watchdog on top of the tick counter that catches every inference-task liveness fault — task stuck in an infinite loop, task crashed silently, task waiting on a never-completing future — and the only thing we miss is "the entire kernel deadlocked or hung". For the AI inference workload that's a meaningful safety improvement; the remaining gap (kernel-level hang) is what the eventual hardware WDT is for.

The software watchdog is also testable on every supported platform (x86-64 host, AArch64 QEMU, Jetson dev kit, Orin Industrial) — it does not need Tegra-specific firmware. We get DO-178C "process-level liveness detection" credit on every deployment.

### Watchdog mechanics

A `Watchdog` task is spawned at `kernel::init` with `TaskType::Watchdog` and `SchedulingClass::System`. It runs once every `T_check = 100ms` (configurable via boot arg `--watchdog-check-ms`). On each invocation:

1. Walks the list of registered inference tasks (a new structure in `kernel/src/sched/executor.rs` — a `RegisteredInferenceTasks` slab).
2. For each task, computes `(current_tick - task.last_pet_tick)` against `task.watchdog_deadline_ms`.
3. If any task is over its deadline, triggers the configured `WatchdogPolicy`.

`sys_watchdog_pet`:
- Stamps the calling task's `last_pet_tick` with `TICK_COUNT.load(Ordering::Acquire)`.
- Returns `0` on success.
- Returns `-EINVAL` if the calling task is not a registered inference task.

`sys_watchdog_remaining`:
- Reads the calling task's `last_pet_tick` and `watchdog_deadline_ms`.
- Returns `max(0, deadline_ticks - (current_tick - last_pet_tick))` converted to milliseconds.

The inference task pets the watchdog at every operator-boundary yield (the same yield that the cooperative scheduler uses for preemption — see `docs/scheduling-model.md`). Adding the pet to the yield path is one new line in the `yield_fn` callback.

### `WatchdogPolicy` enum

```text
enum WatchdogPolicy {
    PanicReboot,    // Existing kernel::state::shutdown() with reason = WatchdogFired
    FallbackModel,  // Activate the pre-registered fallback session, continue serving
}
```

Default is `PanicReboot` (the conservative DO-178C-aligned default). `FallbackModel` requires the application to have called `Session::register_fallback(fallback_session)` on the primary session at startup; without a registered fallback, the policy degrades to `PanicReboot` and a warning is logged.

The `FallbackModel` path:

1. On watchdog fire, the primary session is marked `Aborted`.
2. The pre-registered fallback session is activated; subsequent inference requests are routed to it.
3. A `FallbackEngaged` metric counter is incremented in `onnx-rt/src/profile.rs`.
4. Optionally (configurable), the primary session is rebuilt asynchronously in the background after a cool-down. If the rebuild succeeds, the fallback is deactivated and the primary resumes service. If the rebuild fails twice, the policy escalates to `PanicReboot`.

Recovery semantics are deliberately conservative — DO-178C DAL A prefers "fail-fast" to "fail-soft" because fail-soft can mask repeated faults. Operators who want fail-soft must opt in explicitly via `FallbackModel`, and the system must log the engagement loudly.

### Alternatives considered for the watchdog

**Per-operator deadline (not per-inference).** Rejected — operators vary by 5+ orders of magnitude in cost (a Relu is microseconds; a self-attention layer over 4096 tokens is hundreds of milliseconds). Per-operator deadlines would either fire on legitimate large ops or be loose enough not to catch real hangs. The per-inference deadline at 1s default plus the existing `OperatorBudget` hard-limit at 10× soft budget (already wired in `timer-hal-wcet-v1`) together cover both timescales.

**One watchdog task vs. per-core watchdogs.** Rejected per-core. A single watchdog on Core 0 (the AMP System/IPC core per `docs/scheduling-model.md`) sees all registered inference tasks via the shared registry. Per-core watchdogs would add coordination overhead with no detection-quality gain.

**Pet on syscall entry instead of yield.** Rejected. Pet on yield is more frequent (every op boundary) than pet on syscall entry (only when the user calls a syscall, which an inference loop might not do for hundreds of operators). Yield-pet catches "operators are running but the inference is in an infinite operator-replay loop" — which a syscall-pet would miss.

## Phase 2 — dual-core lockstep

### Why dual lockstep, not TMR

TMR (Triple Modular Redundancy) is the canonical fault-tolerance pattern: three replicas vote, majority wins, single faulty replica is masked. It requires three execution units plus a separate voter. For a SmallAIOS workload on Orin, the available pattern is *dual* lockstep — Arm A78AE supports it in hardware; A78 (non-AE) does not. Building TMR on a SoC that only ships dual lockstep means we'd be doing it in software across three logical replicas, which loses the hardware-comparator credit. Worse, the third core would have to come from somewhere — on Orin Industrial the A78AE complex is 8 cores in 4 paired clusters, so dedicating 3 to TMR leaves only 5 for everything else and breaks the AMP topology.

Dual lockstep with replay-on-divergence is the right pattern here: a single fault is *detected* (not masked), and the system either re-runs the operator and re-votes (catches transient single faults — recoverable) or escalates to the watchdog (catches persistent faults — fail-fast). This matches the standard ISO 26262 "1oo2 with diagnostics" architecture for ASIL-D.

### Hardware vs software comparator

Two modes, selected at runtime based on platform detection:

| Mode | When used | Detection of soft fault | Detection of design bug |
|------|-----------|-------------------------|-------------------------|
| Hardware comparator | A78AE silicon in lock mode, lockstep-strap pins asserted | Yes (cycle-level) | No |
| Software comparator | Dev kit, CI runner, QEMU, A78-non-AE silicon | No (no AE compare unit) | Yes (catches runtime bugs that cause replica divergence) |

The software comparator runs *two* sessions in the same address space, on different cores (different AMP-assigned Inference cores), with `deterministic = true`. At every operator boundary, the voter bit-compares the two sessions' output tensors. The cost is approximately 2× core occupancy and 2× memory for activations. The benefit on dev kits is catching runtime bugs where the deterministic-mode contract is violated (e.g. an operator that accidentally reads host time and produces different output across the replicas). On Orin Industrial in hardware-comparator mode, the software comparator can be left enabled as a redundant check.

For CI verification, the software comparator is the primary surface. QEMU does not model the A78AE compare unit, so the hardware-comparator path can only be exercised on physical Orin Industrial hardware. We accept that gap and gate the hardware-comparator's end-to-end verification on hardware-access milestones.

### A78AE lock-mode boot configuration

The A78AE TRM (Arm DDI 0626) section 4.5.1 documents the "Split/lock mode" configuration. The relevant register is `CLUSTERECTLR_EL1` (cluster-level), and the configuration must be applied *before* the secondary cores are released from reset.

On Tegra234, the strap pins that enable lock-mode on the AE-AS variants are set at chip manufacturing time — they are not firmware-switchable. The kernel's job is to (a) check via a feature register that lock-mode is enabled, (b) configure the cluster registers consistent with lock-mode operation, and (c) configure the GICv3 distributor so that interrupts intended for the lockstep pair are delivered to the leader core only.

The boot sequence in `arch/aarch64/src/boot.rs` (for the `aarch64-unknown-none` path) or `arch/aarch64/src/boot_uefi.rs` (for the UEFI path) gains a new step *before* GICv3 init:

1. Read `CLUSTERIDR_EL1` and `CLUSTERREVIDR_EL1` to confirm A78AE silicon.
2. Read the implementation-defined lockstep status bit (TRM section 4.5.1 documents which bit; we wire it).
3. If lockstep is hardware-enabled, write the cluster-control registers to gate the comparator behavior we want.
4. Continue with GICv3 init, ensuring the redistributor for the follower core is configured as a passive observer.

If lockstep is *not* hardware-enabled but the `lockstep` Cargo feature is on, the kernel logs a clear diagnostic and falls back to software-comparator mode on two non-locked cores. This is the dev-kit deployment path — Cargo feature is enabled for CI, but the silicon does not back it.

### Fault detection and replay

When the A78AE compare unit detects a divergence between the lockstep pair, it raises an asynchronous error (`SError`) on the leader core. The existing `arch/aarch64/src/interrupts.rs` SError path gains a new fault decoder that inspects the syndrome register (`ESR_EL1`) for the implementation-defined lockstep-fault EC + ISS pattern (TRM section 11 documents the exact bit layout).

On detected lockstep fault:

1. The `arch::aarch64::lockstep` module captures the fault context (which operator was running, the input pointers, the output buffer state).
2. Control returns to the executor's voting hook.
3. The executor consults the `LockstepVoter`'s state: "this is the first divergence for this operator → replay".
4. The operator is re-run from the saved inputs. The comparator hardware re-engages. If the replay succeeds (no divergence), the inference continues normally and a `LockstepTransient` metric is incremented.
5. If the replay also diverges, the `LockstepVoter` returns `Permanent`, the inference is aborted, and the watchdog policy is invoked. For DAL A this is typically `PanicReboot`; for less critical workloads it can fall through to `FallbackModel`.

The software-comparator path does the same dance without the hardware fault; the voter explicitly bit-compares the two replicas' outputs at each operator boundary and triggers the same replay-or-escalate sequence on divergence.

### Composition with deterministic mode

Lockstep voting is meaningless without deterministic mode (a vote between two non-deterministic replicas can legitimately disagree from RNG draws and multi-stream timing). The `lockstep` runtime flag forces `deterministic = true` on the underlying `Session`s, with a warning if the user explicitly disabled determinism elsewhere. The hard precondition is checked at session construction; a non-deterministic lockstep session is rejected with a typed error.

### Alternatives considered for lockstep

**Software-only redundancy without AE silicon.** We considered building a pure-software dual-channel mode that runs on any AArch64 SoC. Rejected as the primary path: without the AE compare unit, software comparison adds 2× core cost for *less* detection coverage (we can detect software bugs that violate determinism but cannot detect a hardware soft error that happened to corrupt both replicas identically). Software comparator stays as a *fallback* and *CI* path, not as the primary safety claim.

**Run lockstep on three cores (1oo3).** Rejected per the dual-lockstep-vs-TMR analysis above.

**GPU-side lockstep.** The GA10B GPU on Orin does not have hardware lockstep at the SM level. Software comparison of two GPU sessions on different SMs would catch SM-level transient faults but not detect them at the cycle level. Out of scope for v1; tracked for future work as a separate change.

**Lockstep on Core 0.** Rejected — Core 0 is the System/IPC core per AMP topology. Lockstep should run on the Inference cores. We assign cores 1 and 2 (the first paired AE cluster after Core 0) to the lockstep pair on Industrial silicon; the AMP topology shifts to "Core 0 = System/IPC, Cores 1-2 = lockstep Inference, Cores 3-7 = additional Inference partitions".

## Watchdog + lockstep — shared fault surface

The two phases share the fault-handling code path. Lockstep escalation feeds the watchdog policy (the same `WatchdogPolicy` enum determines whether a permanent lockstep divergence triggers reboot or fallback). The watchdog can fire from a slow operator independently of lockstep; the metrics distinguish the two causes.

```text
                +---------------------+
                |  Inference task     |
                |  (op-boundary yield)|
                +------+--------------+
                       |
              ---------+---------
              |                 |
              v                 v
    +---------------+   +-------------------+
    | Pet watchdog  |   | LockstepVoter     |
    | (if deadline  |   | compare outputs   |
    | OK)           |   +---+----+----------+
    +---+-----------+       |    |
        | over deadline     |    | divergence
        |                   |    v
        |          +--------+----+-----+
        |          | Replay operator   |
        |          | (1 retry)         |
        |          +--------+----------+
        |                   |
        |                   v
        |          +--------+----------+
        |          | Re-vote outputs   |
        |          +--------+----+-----+
        |                   |    | persistent
        v                   v    v
   +--------------+    +----+----+---------+
   | WatchdogFired|    | LockstepPermanent |
   +-------+------+    +---+----+----------+
           |               |
           +---+----+------+
               v    v
       +-------+----+------+
       | WatchdogPolicy    |
       | PanicReboot OR    |
       | FallbackModel     |
       +-------------------+
```

## Cost model

| Mode | Core occupancy | Memory overhead | Detection coverage |
|------|----------------|------------------|---------------------|
| No watchdog, no lockstep (status quo) | 1× | 1× | None |
| Watchdog only | 1× + epsilon (1 System-class task at 100ms tick) | 1× + small registry | Liveness only |
| Watchdog + software comparator | 2× | 2× activations | Liveness + determinism-violation detection |
| Watchdog + hardware comparator (Orin Industrial) | 2× (cores in lock mode) | 1× activations (single execution path, hardware doubles it) | Liveness + hardware soft-error detection |
| Watchdog + hardware comparator + software comparator (defense-in-depth) | 2× | 2× activations | All of the above |

For DO-178C DAL A the recommended mode is "Watchdog + hardware comparator on Industrial silicon". For dev-kit deployments the practical mode is "Watchdog + software comparator", giving most of the credit (liveness coverage + determinism enforcement) without the hardware prerequisite.

## What this change explicitly does NOT do

- Does not touch hardware-watchdog register access (Tegra WDT, secure WDT). Tracked as a follow-up gated on the `unikernel-orin-bringup-v1` BSP and on EL3 firmware access from TF-A.
- Does not change the default boot path on any platform. Lockstep is triple-opt-in (Cargo feature + runtime flag + detected silicon).
- Does not change the cooperative scheduler's yield contract. The voter hooks into the existing op-boundary yield; it does not introduce new preemption points.
- Does not implement TMR. Dual lockstep + replay is the v1 ceiling.
- Does not implement asymmetric replicas (CPU vs GPU; different runtimes). Replicas are byte-identical sessions in v1.
- Does not implement network-attached redundancy. Both replicas are local to the same SoC.
- Does not modify the `safety-critical` capability spec's DO-178C process requirements. This change adds *implementation* of fault-detection mechanics; the process spec is unchanged.
