# Capability: arch-aarch64-lockstep

## ADDED Requirements

### Requirement: A78AE silicon detection at boot

The AArch64 boot path SHALL detect whether the running silicon is a Cortex-A78AE variant with hardware lockstep support and SHALL configure the kernel's lockstep mode (`HardwareComparator` vs `SoftwareComparator`) based on the detection result.

#### Scenario: Detect A78AE-AS silicon on Orin Industrial

- **GIVEN** a SmallAIOS kernel booting on Tegra Orin Industrial / Drive Orin AGX Industrial hardware (the SKUs that ship Arm Cortex-A78AE Automotive Specific silicon)
- **AND** the `lockstep` Cargo feature is enabled at build time
- **WHEN** the kernel reaches the early-boot lockstep configuration step (before GICv3 init)
- **THEN** the kernel SHALL read `CLUSTERIDR_EL1` and `CLUSTERREVIDR_EL1` to confirm A78AE silicon
- **AND** the kernel SHALL read the Arm-implementation-defined lockstep status bit per the Cortex-A78AE TRM (Arm DDI 0626) section 4.5.1
- **AND** if lockstep is hardware-enabled, the kernel SHALL log "A78AE lockstep silicon detected" via the early UART
- **AND** the kernel SHALL select `LockstepMode::HardwareComparator` as the runtime mode

#### Scenario: Fall back to software comparator on dev-kit silicon

- **GIVEN** a SmallAIOS kernel booting on Jetson Orin NX dev kit silicon (P3767-0000 module, the consumer-grade variant that does NOT ship A78AE-AS lockstep silicon)
- **AND** the `lockstep` Cargo feature is enabled at build time
- **WHEN** the kernel reaches the early-boot lockstep configuration step
- **THEN** the kernel SHALL detect that lockstep is NOT hardware-enabled
- **AND** the kernel SHALL log "lockstep feature enabled but silicon does not support hardware lockstep — falling back to software-comparator mode"
- **AND** the kernel SHALL select `LockstepMode::SoftwareComparator` as the runtime mode
- **AND** the boot SHALL continue normally; software comparator is a first-class supported mode, not an error

#### Scenario: Lockstep feature off, no detection

- **GIVEN** a SmallAIOS kernel built without the `lockstep` Cargo feature
- **WHEN** the kernel boots
- **THEN** the lockstep detection SHALL be skipped entirely
- **AND** the kernel SHALL NOT write to A78AE cluster control registers
- **AND** the boot path SHALL be bit-for-bit identical to the pre-`watchdog-lockstep-v1` AArch64 boot

### Requirement: A78AE cluster configuration for hardware lockstep

When hardware lockstep is detected and selected, the AArch64 boot path SHALL configure the A78AE cluster registers per the Cortex-A78AE TRM to gate the cluster into lock mode before any secondary cores are released from reset.

#### Scenario: Cluster registers are configured before secondary core release

- **GIVEN** a boot path that has selected `LockstepMode::HardwareComparator`
- **WHEN** the kernel configures the A78AE cluster registers per Cortex-A78AE TRM (Arm DDI 0626) section 4.5.1
- **THEN** the configuration SHALL complete before any secondary core is released from reset
- **AND** the kernel SHALL read back the relevant cluster status register after writing to confirm the cluster is operating in lock mode
- **AND** a failed readback SHALL be a fatal boot error (the system cannot continue in a partial-lockstep state)

#### Scenario: GICv3 redistributor configured for follower-as-passive

- **GIVEN** a boot path that has configured the A78AE cluster for lock mode
- **WHEN** the kernel initializes the GICv3 redistributor for each lockstep-paired core
- **THEN** the redistributor for the follower core SHALL be configured as a passive observer
- **AND** interrupts targeted at the lockstep pair SHALL be delivered to the leader's redistributor only
- **AND** the follower SHALL NOT receive direct interrupt routing (the AE compare unit handles synchronization)

### Requirement: Lockstep fault decoding and replay routing

The AArch64 exception handlers SHALL distinguish lockstep-comparator faults from other AArch64 exception causes and SHALL route them to the `LockstepVoter`'s replay path.

#### Scenario: Lockstep fault is identifiable in ESR_EL1

- **GIVEN** a hardware-comparator lockstep configuration
- **AND** the A78AE compare unit raises an `SError` on a detected divergence between the lockstep pair
- **WHEN** the AArch64 SError handler reads `ESR_EL1`
- **THEN** the handler SHALL decode the implementation-defined `EC` + `ISS` bits per the Cortex-A78AE TRM (Arm DDI 0626) section 11
- **AND** the handler SHALL distinguish the lockstep-fault pattern from page-fault, alignment-fault, and other exception causes
- **AND** a lockstep fault SHALL be routed to `arch::aarch64::lockstep::handle_fault`; other faults SHALL continue to follow the existing exception path

#### Scenario: Fault context capture preserves replay inputs

- **GIVEN** a lockstep fault raised mid-operator
- **WHEN** `arch::aarch64::lockstep::handle_fault` runs
- **THEN** the handler SHALL capture the executor's current-operator context: the saved input pointers, the output buffer state, the operator identity and graph index
- **AND** the handler SHALL return control to the executor's voting hook, which consults the `LockstepVoter` to decide replay-or-escalate
- **AND** the captured context SHALL be sufficient to re-run the operator from its inputs without re-running any preceding operator

#### Scenario: Non-lockstep exceptions are unaffected

- **GIVEN** a lockstep-configured system
- **WHEN** a non-lockstep exception occurs (e.g. a page fault from a non-replicated execution path)
- **THEN** the existing exception handler SHALL handle it via the pre-`watchdog-lockstep-v1` path
- **AND** the lockstep-fault decoder SHALL not interfere with normal exception handling

### Requirement: AMP topology adjustment for lockstep mode

When lockstep is active, the AMP core assignment SHALL allocate the lockstep replica pair to the first available Inference cores (Cores 1 and 2 by default) and SHALL document the resulting topology in `docs/scheduling-model.md`.

#### Scenario: Default AMP topology with lockstep

- **GIVEN** a SmallAIOS kernel booted with `lockstep` enabled on an 8-core Orin Industrial system
- **WHEN** the AMP topology is established
- **THEN** Core 0 SHALL be the System/IPC core (unchanged)
- **AND** Cores 1 and 2 SHALL form the lockstep replica pair (Inference partition A)
- **AND** Cores 3-7 SHALL be additional Inference partitions (B, C, ...) running non-replicated workloads if any
- **AND** the topology SHALL be documented in `docs/scheduling-model.md`

#### Scenario: Lockstep is incompatible with work-stealing

- **GIVEN** a lockstep-active system
- **AND** the deterministic-scheduling-v1 contract (work-stealing disabled in deterministic mode)
- **WHEN** the scheduler considers whether to steal tasks
- **THEN** work-stealing SHALL remain disabled
- **AND** the lockstep replica pair SHALL not have tasks stolen from or to them — they execute only their replicated workload
- **AND** if the scheduler observes a configuration that violates this invariant (lockstep on with work-stealing on), it SHALL refuse to boot and log a clear diagnostic

### Requirement: Hardware-comparator mode is opt-in via three layers

Hardware-comparator lockstep mode SHALL require all three of: (a) the `lockstep` Cargo feature enabled at build time, (b) the `--lockstep` runtime flag set at boot, and (c) detected A78AE-AS silicon with hardware lockstep enabled.

#### Scenario: Triple opt-in prevents accidental enablement

- **GIVEN** any one of: (a) `lockstep` Cargo feature off, (b) `--lockstep` runtime flag absent, (c) detected silicon without hardware lockstep
- **WHEN** the kernel boots
- **THEN** the hardware-comparator mode SHALL NOT be engaged
- **AND** the system SHALL boot in the highest applicable fallback mode (software-comparator if (c) only, no-lockstep otherwise)

#### Scenario: All three layers present enables hardware comparator

- **GIVEN** all three of: (a) `lockstep` Cargo feature on, (b) `--lockstep` runtime flag set, (c) detected A78AE-AS silicon with lockstep enabled
- **WHEN** the kernel boots
- **THEN** `LockstepMode::HardwareComparator` SHALL be engaged
- **AND** the cluster configuration SHALL be performed per the requirements above
- **AND** the AE compare unit SHALL be active for the lockstep replica pair

### Requirement: Lockstep verification surface

The repository SHALL provide unit tests that exercise the lockstep-fault decoder logic against synthetic ESR_EL1 values and SHALL document the hardware-access requirement for end-to-end hardware-comparator verification.

#### Scenario: Fault decoder is testable on the host

- **GIVEN** a host-mode test build of `arch/aarch64/src/lockstep.rs`
- **WHEN** the unit tests run
- **THEN** the tests SHALL feed synthetic `ESR_EL1` values matching documented lockstep-fault and non-lockstep-fault patterns
- **AND** the decoder SHALL correctly classify each value
- **AND** the tests SHALL not require physical A78AE hardware

#### Scenario: End-to-end hardware verification is gated on hardware access

- **GIVEN** a draft PR landing the hardware-comparator code path
- **WHEN** the PR is reviewed
- **THEN** the PR description SHALL document that end-to-end verification requires access to an Orin Industrial / Drive Orin AGX Industrial reference platform
- **AND** the PR SHALL remain draft until on-Industrial-hardware fault-injection evidence (e.g. deliberate register-state corruption captured by the comparator) is captured and pasted into the PR description
- **AND** the software-comparator unit tests SHALL be the gate for PR-CI passes; hardware-comparator end-to-end is a release-gate, not a PR-gate
