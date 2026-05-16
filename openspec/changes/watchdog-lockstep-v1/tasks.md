# Tasks — watchdog-lockstep-v1

## 0. Preconditions and platform survey

- [ ] 0.1 Confirm `deterministic-scheduling-v1` has landed on `develop` before starting Phase 2 work — lockstep voting is meaningless without deterministic mode. Phase 1 (software watchdog) has no dependency on determinism and can start any time.
- [ ] 0.2 Capture the Tegra234 platform survey for lockstep-capable silicon. On a Jetson Orin NX dev kit (P3767-0000 / P3768-0000 / J4012), confirm via the Arm Cortex-A78AE TRM-documented feature register that **lockstep is NOT hardware-enabled** — record this in `notes/0.2-orin-dev-kit-lockstep-survey.md` so future contributors don't assume the dev kit supports it. On Orin Industrial reference platforms (if hardware access is available), confirm lockstep IS enabled and record the feature-register values.
- [ ] 0.3 Inventory every existing `sys_watchdog_*` reference: search `kernel/src/syscall/system.rs`, `posix/`, `container/`, `bench/`. Confirm no production code currently relies on the stub behavior (returning 30000ms from `sys_watchdog_remaining`). Document findings in `notes/0.3-existing-watchdog-callers.md`.

## 1. Phase 1 — Software watchdog

### 1a. Watchdog task scaffolding

- [ ] 1.1 Add `last_pet_tick: AtomicU64` and `watchdog_deadline_ms: u32` fields to `Task` in `kernel/src/sched/task.rs`. Initialize `last_pet_tick = 0`, `watchdog_deadline_ms = 1000` (1 second default).
- [ ] 1.2 Create `kernel/src/sched/watchdog.rs` housing the `Watchdog` task implementation, the `WatchdogPolicy` enum (`PanicReboot`, `FallbackModel`), and the `RegisteredInferenceTasks` slab.
- [ ] 1.3 Spawn the `Watchdog` task at `kernel::init` with `TaskType::Watchdog` and `SchedulingClass::System`. Configurable check interval via `--watchdog-check-ms=N` boot arg, default 100ms.
- [ ] 1.4 The watchdog task body walks the registry, computes elapsed ticks since `last_pet_tick`, compares against `watchdog_deadline_ms`, and on violation invokes the policy. Each iteration must complete in < 1 ms per the System-class constraint.

### 1b. Syscall implementation

- [ ] 1.5 Replace the `TODO` stub in `sys_watchdog_pet` (`kernel/src/syscall/system.rs`) with: stamp the calling task's `last_pet_tick` from `TICK_COUNT.load(Acquire)`. Return `0` on success, `-EINVAL` if the calling task is not in the registered-inference-tasks set.
- [ ] 1.6 Replace the `TODO` stub in `sys_watchdog_remaining` with: read `last_pet_tick` and `watchdog_deadline_ms`, return `max(0, deadline_ticks - elapsed_ticks)` converted to milliseconds. Remove the hardcoded `return 30000`.
- [ ] 1.7 Update the unit tests in `kernel/src/syscall/system.rs` (`test_sys_watchdog_pet_returns_success`, `test_sys_watchdog_remaining_returns_value`) to assert the new behavior — pet stamps the tick, remaining returns the actual deadline minus elapsed.
- [ ] 1.8 Add a new unit test verifying that `sys_watchdog_remaining` returns 0 when the deadline has elapsed.

### 1c. Cooperative-scheduler integration

- [ ] 1.9 In the existing `yield_fn` callback path (`onnx-rt/src/executor.rs`), add a call to `sys_watchdog_pet` at every operator-boundary yield. This is one line; document the change in `docs/scheduling-model.md`.
- [ ] 1.10 Update `docs/scheduling-model.md` "Design Guidelines for Contributors" with rule 9: "Inference tasks must pet the software watchdog at every operator-boundary yield. The `yield_fn` callback does this automatically; custom executors must do it manually."

### 1d. Alternate-model fallback

- [ ] 1.11 Add `Session::register_fallback(fallback: Session)` to `onnx-rt/src/session.rs` that links a fallback session to a primary. The primary holds a pointer to the fallback; the fallback is constructed lazily so its model load cost is only paid if needed.
- [ ] 1.12 Add `SessionConfig::watchdog_deadline_ms: Option<u32>` (default `None` = use kernel default). Plumb it through to the task's `watchdog_deadline_ms` field.
- [ ] 1.13 On watchdog fire with `WatchdogPolicy::FallbackModel`, mark the primary session `Aborted`, activate the fallback, route subsequent requests to the fallback, increment a `FallbackEngaged` metric in `onnx-rt/src/profile.rs`.
- [ ] 1.14 Add background-rebuild logic: after a configurable cool-down (default 30s), attempt to rebuild the primary session. If successful twice in a row, deactivate the fallback. If failed twice, escalate the policy to `PanicReboot`.

### 1e. Watchdog tests

- [ ] 1.15 Unit test: spawn a watchdog task with `T_check = 10ms` and a fake inference task with `deadline = 50ms`. Verify the watchdog fires within 60ms of the last pet.
- [ ] 1.16 Unit test: verify `WatchdogPolicy::PanicReboot` invokes `kernel::state::shutdown(WatchdogFired)`.
- [ ] 1.17 Unit test: verify `WatchdogPolicy::FallbackModel` activates the registered fallback session and increments `FallbackEngaged`.
- [ ] 1.18 Integration test (using the existing host-mode test runner): run a deliberately-slow ONNX inference under deterministic mode + watchdog enabled, verify the watchdog catches the hang within the deadline.
- [ ] 1.19 Negative test: `FallbackModel` policy with no registered fallback degrades to `PanicReboot` with a logged warning.

### 1f. Watchdog docs + close-out

- [ ] 1.20 Create `docs/watchdog.md` covering: the software-watchdog contract, the syscall ABI (0x55 pet, 0x56 remaining), the configurable parameters (`--watchdog-check-ms`, `SessionConfig::watchdog_deadline_ms`), the `WatchdogPolicy` enum semantics, the alternate-model fallback workflow, the DO-178C DAL A liveness-detection claim it unlocks, and the future hardware-WDT integration roadmap.
- [ ] 1.21 Update `CLAUDE.md` "Current state" to note that the software watchdog is wired end-to-end.
- [ ] 1.22 PR title: `feat(kernel,onnx-rt): watchdog-lockstep-v1 phase 1 — software watchdog + fallback`. Target `develop`.
- [ ] 1.23 PR green + reviewer sign-off + squash-merge.

## 2. Phase 2 — Lockstep voting (software comparator)

### 2a. Voter scaffolding

- [ ] 2.1 Create `onnx-rt/src/lockstep.rs` housing `LockstepVoter` (tracks two replica sessions and their per-op output tensors), `LockstepMode` enum (`HardwareComparator`, `SoftwareComparator`), and `LockstepResult` enum (`Match`, `Transient`, `Permanent`).
- [ ] 2.2 Add `SessionConfig::lockstep: Option<LockstepMode>` field (default `None`). When `Some`, force `deterministic = true` and reject the session if the user explicitly set `deterministic = false`.
- [ ] 2.3 Lockstep mode requires two underlying `Session`s with identical configs. The `LockstepVoter` constructs both and owns them.

### 2b. Voter integration with the executor

- [ ] 2.4 At each operator-boundary yield in `onnx-rt/src/executor.rs`, if lockstep is active, the voter bit-compares the two replicas' output tensors for the just-completed operator. The compare uses a fast SIMD-friendly memcmp on the contiguous tensor bytes.
- [ ] 2.5 On `Match`: continue normally.
- [ ] 2.6 On first divergence (`Transient`): roll both replicas back to the operator's saved inputs, re-run the operator, re-vote. Increment a `LockstepTransient` metric.
- [ ] 2.7 On second divergence (`Permanent`): increment `LockstepPermanent` metric, invoke the configured `WatchdogPolicy` (the lockstep voter consumes the same policy enum as the watchdog).

### 2c. AMP topology adjustment

- [ ] 2.8 When lockstep is enabled, the AMP topology shifts: Core 0 stays System/IPC, Cores 1 and 2 become the lockstep replica pair (Inference partition A), Cores 3-N remain Inference (additional partitions). Document this in `docs/scheduling-model.md`.
- [ ] 2.9 Add a `CpuAffinity::LockstepPair { leader: u8, follower: u8 }` variant. The leader publishes outputs to the voter; the follower runs the replica computation and publishes its outputs for comparison.

### 2d. Software-comparator tests

- [ ] 2.10 Unit test: two replicas running the same model on the same input in `SoftwareComparator` mode produce `Match` at every op boundary.
- [ ] 2.11 Unit test: injecting a deliberate divergence in one replica (e.g. corrupting the output tensor between operators) triggers `Transient` → replay → `Match`.
- [ ] 2.12 Unit test: injecting a persistent divergence (corrupt every operator output) triggers `Permanent` and invokes the configured policy.
- [ ] 2.13 Negative test: constructing a `LockstepVoter` with `deterministic = false` returns a typed error.

### 2e. Software-comparator close-out

- [ ] 2.14 Run `just test-determinism` with `--lockstep` enabled against the existing reproducibility fixtures; confirm `LockstepTransient` and `LockstepPermanent` counters stay at 0 across 100 runs.
- [ ] 2.15 Document the software-comparator mode in a new `docs/lockstep.md` — separate from `docs/watchdog.md` because the audience is different (lockstep is a higher tier of safety credit and only relevant to ASIL-D / DAL A deployments).
- [ ] 2.16 PR title: `feat(onnx-rt,kernel): watchdog-lockstep-v1 phase 2a — software-comparator lockstep`. Target `develop`.
- [ ] 2.17 PR green + reviewer sign-off + squash-merge.

## 3. Phase 2 — Lockstep (hardware comparator, A78AE)

### 3a. A78AE detection

- [ ] 3.1 Add `arch/aarch64/src/lockstep.rs` housing the A78AE detection + configuration code. Gated by a new `lockstep` Cargo feature on `smallaios-arch-aarch64`, itself gated on `tegra234`.
- [ ] 3.2 At boot (in `arch/aarch64/src/boot.rs` or `boot_uefi.rs` depending on target), read `CLUSTERIDR_EL1` and `CLUSTERREVIDR_EL1` to confirm A78AE silicon. Read the implementation-defined lockstep status bit. If lockstep is hardware-enabled, log "A78AE lockstep silicon detected, configuring for lock mode" via the early UART.
- [ ] 3.3 If `lockstep` Cargo feature is on but silicon detection returns "no lockstep" (the dev-kit case), log "lockstep feature enabled but silicon does not support hardware lockstep — falling back to software-comparator mode" and continue.

### 3b. Cluster register configuration

- [ ] 3.4 Write the A78AE cluster-control registers per the Arm Cortex-A78AE TRM (Arm DDI 0626) section 4.5.1 to gate the cluster into lock mode. Reference the implementation-defined `CLUSTERECTLR_EL1` writes; this work requires close reading of the TRM and is gated on hardware access for verification.
- [ ] 3.5 Configure the GICv3 redistributor for the follower core as a passive observer — interrupts targeted at the lockstep pair are delivered to the leader's redistributor only.
- [ ] 3.6 Confirm via a post-boot diagnostic that the cluster is operating in lock mode (read the same status register from EL2 after the configuration and confirm the expected value).

### 3c. Hardware-fault decoder

- [ ] 3.7 Extend `arch/aarch64/src/interrupts.rs` SError handling: decode `ESR_EL1` for the implementation-defined lockstep-fault EC + ISS pattern per the A78AE TRM section 11. On match, route to the new `arch::aarch64::lockstep::handle_fault` function.
- [ ] 3.8 The fault handler captures the fault context (saved-input pointers from the executor's current-operator state) and returns control to the voter's replay path.
- [ ] 3.9 Distinguish lockstep faults from other AArch64 exception causes (page fault, alignment, etc.) — only lockstep faults should engage the replay path.

### 3d. Hardware-comparator tests

- [ ] 3.10 The hardware-comparator path can only be end-to-end verified on Orin Industrial silicon. Document the hardware-access requirement in the PR description and tag the change blocked on hardware availability.
- [ ] 3.11 In the meantime, add a unit test that mocks the SError + ISS bits and confirms the fault decoder routes correctly. The decoder logic is platform-independent and testable on the host.
- [ ] 3.12 When hardware access is available: run a deliberate-fault-injection test (e.g. a kernel patch that corrupts one replica's register state mid-operator) and confirm the comparator catches it. Record the test evidence in the PR description.

### 3e. Hardware-comparator close-out

- [ ] 3.13 Update `docs/lockstep.md` with a "Hardware comparator mode" section covering: silicon detection, cluster configuration, fault decoding, AMP topology in lockstep mode, the DO-178C / ISO 26262 credit claim, and the hardware-access prerequisite.
- [ ] 3.14 PR title: `feat(arch/aarch64,onnx-rt): watchdog-lockstep-v1 phase 2b — A78AE hardware lockstep`. Target `develop`. Mark as draft until Orin Industrial hardware verification is captured.
- [ ] 3.15 PR green + on-Industrial-hardware fault-injection evidence + reviewer sign-off + squash-merge.

## 4. CI advisory jobs

- [ ] 4.1 Add a `watchdog-fires-test` advisory CI job that runs the watchdog hang-detection integration test in QEMU; gates on Phase 1 landing.
- [ ] 4.2 Add a `lockstep-software-comparator` advisory CI job that runs the software-comparator unit tests + a smoke `just test-determinism --lockstep` invocation; gates on Phase 2a landing.
- [ ] 4.3 Document the promote-to-gate criterion in the workflow YAML for each job.

## 5. Cross-phase verification + archive

- [ ] 5.1 Run `openspec validate watchdog-lockstep-v1 --strict` after all sub-PRs land.
- [ ] 5.2 Update the `safety-critical` capability spec's traceability section to reference the new `kernel-safety` and `arch-aarch64-lockstep` capabilities as the implementation surfaces backing the DO-178C "fault detection coverage" objective.
- [ ] 5.3 Archive: move to `openspec/changes/archive/YYYY-MM-DD-watchdog-lockstep-v1` and sync the spec deltas to `openspec/specs/kernel-safety/` and `openspec/specs/arch-aarch64-lockstep/`.

## 6. Phase split escape hatch

- [ ] If Phase 2 (lockstep) work blocks on Orin Industrial hardware access for more than ~4 weeks after Phase 1 lands, split Phase 2 out as `lockstep-voting-v1` (a new change) and archive this change with only Phase 1 (software watchdog + fallback) implemented. Update `proposal.md` "Out of scope" to reference the new change name and the hardware-access prerequisite.
