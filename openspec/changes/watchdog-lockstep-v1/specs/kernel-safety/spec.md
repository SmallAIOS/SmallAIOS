# Capability: kernel-safety

## ADDED Requirements

### Requirement: Software watchdog for inference tasks

The kernel SHALL implement a software watchdog task running in `SchedulingClass::System` that detects when registered inference tasks fail to check in within a configurable deadline and triggers a configured fault-response policy on violation.

#### Scenario: Watchdog task is spawned at boot

- **GIVEN** a SmallAIOS kernel boot
- **WHEN** `kernel::init` completes
- **THEN** a `Watchdog` task SHALL be spawned with `TaskType::Watchdog` and `SchedulingClass::System`
- **AND** the task SHALL run at a configurable check interval (`--watchdog-check-ms` boot argument, default 100ms)
- **AND** each invocation of the watchdog task SHALL complete in less than 1 ms (the System-class hard-RT constraint per `docs/scheduling-model.md`)

#### Scenario: Inference task pets the watchdog at every operator-boundary yield

- **GIVEN** an inference task running in cooperative-yield mode
- **WHEN** the task reaches an operator boundary and the `yield_fn` callback fires
- **THEN** the runtime SHALL invoke `sys_watchdog_pet` (syscall 0x55) to stamp the task's `last_pet_tick` with the current scheduler tick
- **AND** the syscall SHALL return `0` on success
- **AND** the syscall SHALL return `-EINVAL` if the calling task is not in the registered inference tasks set

#### Scenario: Watchdog detects a stuck inference task

- **GIVEN** an inference task that has not petted the watchdog within its configured `watchdog_deadline_ms`
- **WHEN** the watchdog task runs its next check interval
- **THEN** the watchdog SHALL detect the violation
- **AND** the watchdog SHALL invoke the configured `WatchdogPolicy`
- **AND** the watchdog SHALL log a diagnostic identifying the violating task

#### Scenario: Remaining-time query is accurate

- **GIVEN** an inference task that petted the watchdog `T_elapsed` ticks ago with a deadline of `D` ticks
- **WHEN** the task invokes `sys_watchdog_remaining` (syscall 0x56)
- **THEN** the syscall SHALL return `max(0, D - T_elapsed)` converted to milliseconds
- **AND** the syscall SHALL NOT return a hardcoded constant
- **AND** the syscall SHALL return 0 when the deadline has elapsed

### Requirement: Watchdog fault-response policy

The kernel SHALL support at least two fault-response policies for a watchdog violation: `PanicReboot` (default) and `FallbackModel` (alternate-model swap-and-continue).

#### Scenario: PanicReboot policy invokes orderly shutdown

- **GIVEN** a watchdog configured with `WatchdogPolicy::PanicReboot`
- **AND** an inference task that has exceeded its deadline
- **WHEN** the watchdog detects the violation
- **THEN** the watchdog SHALL invoke `kernel::state::shutdown` with the reason `WatchdogFired`
- **AND** the kernel SHALL log the shutdown reason and the violating task identity to syslog before halting

#### Scenario: FallbackModel policy activates the registered fallback session

- **GIVEN** a primary session that has called `Session::register_fallback(fallback_session)`
- **AND** the primary's watchdog is configured with `WatchdogPolicy::FallbackModel`
- **AND** the primary has exceeded its deadline
- **WHEN** the watchdog detects the violation
- **THEN** the primary session SHALL be marked `Aborted`
- **AND** subsequent inference requests SHALL be routed to the fallback session
- **AND** a `FallbackEngaged` metric counter SHALL be incremented in `onnx-rt/src/profile.rs`
- **AND** the system SHALL log the fallback engagement loudly so operators can observe the degraded mode

#### Scenario: FallbackModel without a registered fallback degrades to PanicReboot

- **GIVEN** a primary session with `WatchdogPolicy::FallbackModel` but no fallback registered
- **WHEN** the watchdog fires
- **THEN** the policy SHALL degrade to `PanicReboot`
- **AND** the system SHALL log a warning naming the absent fallback and the policy-degradation behavior

#### Scenario: Background primary-rebuild after fallback engagement

- **GIVEN** a fallback engaged via `WatchdogPolicy::FallbackModel`
- **WHEN** the configurable cool-down expires (default 30s)
- **THEN** the runtime SHALL attempt to rebuild the primary session asynchronously
- **AND** two consecutive successful rebuilds SHALL deactivate the fallback and resume primary service
- **AND** two consecutive failed rebuilds SHALL escalate the policy to `PanicReboot`

### Requirement: Watchdog configurable per session

`SessionConfig` SHALL expose a `watchdog_deadline_ms: Option<u32>` field that overrides the kernel-default watchdog deadline for the session's inference task.

#### Scenario: Per-session deadline overrides the kernel default

- **GIVEN** a kernel booted with the default 1000ms watchdog deadline
- **AND** a session constructed with `SessionConfig { watchdog_deadline_ms: Some(250), .. }`
- **WHEN** the session's inference task is registered with the watchdog
- **THEN** the task's `watchdog_deadline_ms` SHALL be 250 (not 1000)
- **AND** the watchdog SHALL fire on the per-session deadline, not the kernel default

#### Scenario: Default deadline is used when not overridden

- **GIVEN** a session constructed with `SessionConfig { watchdog_deadline_ms: None, .. }`
- **WHEN** the session's inference task is registered with the watchdog
- **THEN** the task's `watchdog_deadline_ms` SHALL be the kernel-default value
- **AND** the kernel default SHALL be 1000ms unless overridden by the `--watchdog-deadline-ms` boot argument

### Requirement: Lockstep voting integration with watchdog

The kernel SHALL surface a `LockstepVoter` capability that detects divergences between two lockstep replica sessions at operator boundaries and uses the watchdog's `WatchdogPolicy` enum to determine the fault-response action on persistent divergence.

#### Scenario: Voter detects a transient divergence and replays

- **GIVEN** two lockstep replicas executing the same model on the same input in deterministic mode
- **AND** a first-time divergence detected at an operator boundary
- **WHEN** the voter consults its replay state
- **THEN** the voter SHALL roll both replicas back to the operator's saved inputs
- **AND** the voter SHALL re-run the operator and re-compare outputs
- **AND** the voter SHALL increment a `LockstepTransient` metric counter
- **AND** the inference SHALL continue normally if the replay matches

#### Scenario: Voter escalates a persistent divergence

- **GIVEN** a lockstep replay that itself diverges (second consecutive divergence at the same operator)
- **WHEN** the voter consults its replay state
- **THEN** the voter SHALL increment a `LockstepPermanent` metric counter
- **AND** the voter SHALL invoke the configured `WatchdogPolicy` (`PanicReboot` or `FallbackModel`)
- **AND** the persistent-divergence event SHALL be logged with the operator name, the first-divergence and second-divergence byte offsets, and the replica identifiers

#### Scenario: Lockstep without deterministic mode is rejected

- **GIVEN** a `SessionConfig` with `lockstep = Some(_)` and `deterministic = false`
- **WHEN** the session is constructed
- **THEN** session construction SHALL return a typed error explaining that lockstep voting requires deterministic mode
- **AND** the error message SHALL reference `docs/lockstep.md` for further reading

### Requirement: Watchdog documentation and observability

The repository SHALL provide a `docs/watchdog.md` document covering the software-watchdog contract, the syscall ABI, the configurable parameters, the `WatchdogPolicy` semantics, and the alternate-model fallback workflow.

#### Scenario: Documentation is discoverable and complete

- **GIVEN** a new contributor or operator
- **WHEN** they read `docs/watchdog.md`
- **THEN** they SHALL find: (a) the syscall ABI (0x55 pet, 0x56 remaining) including return-code semantics, (b) the configurable parameters (`--watchdog-check-ms`, `--watchdog-deadline-ms`, `SessionConfig::watchdog_deadline_ms`), (c) the `WatchdogPolicy` enum and its degradation rules, (d) the alternate-model fallback workflow including the cool-down + rebuild semantics, (e) the DO-178C DAL A liveness-detection claim the watchdog unlocks, and (f) the future hardware-WDT integration roadmap

#### Scenario: Lockstep documentation is separated

- **GIVEN** a `docs/lockstep.md` document
- **THEN** it SHALL be distinct from `docs/watchdog.md` because lockstep is a higher tier of safety credit and a different operator audience
- **AND** it SHALL cross-link to `docs/watchdog.md` for the shared fault-response policy semantics
- **AND** it SHALL cover both software-comparator and hardware-comparator modes
