## ADDED Requirements

### Requirement: Background ECC memory scrubbing service

The kernel SHALL provide a background memory-scrubbing service that periodically walks every configured DRAM region, surfaces correctable and uncorrectable ECC error counts per region, and runs as a cooperative async task without blocking inference workloads, on builds with the `ecc-scrub` Cargo feature enabled.

#### Scenario: Scrub task registered at boot

- **GIVEN** a SmallAIOS kernel built with `--features ecc-scrub` (and a platform that advertises ECC DRAM)
- **WHEN** `kernel_main` reaches the scrub-init phase after `mem::init` completes
- **THEN** the kernel SHALL register default scrub regions for the kernel heap, the ONNX-resident weight region (when known), and the kernel `.bss` / `.data` sections
- **AND** the kernel SHALL spawn one async scrub task into the cooperative scheduler
- **AND** the task SHALL begin advancing the cursor for the first registered region

#### Scenario: Configurable per-region interval

- **GIVEN** a `ScrubConfig` with regions `A` (interval 6 h) and `B` (interval 24 h)
- **WHEN** the scrub task runs
- **THEN** region `A` SHALL complete approximately four cycles for every cycle of region `B`
- **AND** the `stats(name).cycles_completed` counter SHALL reflect the per-region progress accurately
- **AND** the cooperative-yield behavior SHALL ensure no scrub chunk holds the scheduler for longer than the configured chunk-size budget

#### Scenario: Stats surfaced via telemetry

- **GIVEN** the scrub service active for at least one full cycle on a registered region
- **WHEN** an operator queries the kernel telemetry surface
- **THEN** the response SHALL include per-region `ScrubStats { cycles_completed, last_cycle_duration, correctable_errors, uncorrectable_errors, cursor_position, advanced_at }`
- **AND** all counters SHALL be monotonic across the boot — they SHALL reset only on kernel reboot

### Requirement: Tegra234 EMC hardware-accelerated scrub backend

When running on a Tegra234 platform with hardware ECC support, the scrub service SHALL drive the EMC's hardware patrol-scrub registers in preference to the software backend.

#### Scenario: EMC probe succeeds and is selected

- **GIVEN** a SmallAIOS kernel built with `--features tegra234,ecc-scrub` running on an Orin NX
- **WHEN** the scrub service initializes
- **THEN** it SHALL walk the DTB for a node matching `compatible = "nvidia,tegra234-emc"` and SHALL read the EMC MMIO base from the node's `reg` property
- **AND** it SHALL probe `EMC_ECC_STATUS` to confirm the controller responds — if the probe fails the service SHALL fall back to the software backend with a warning log line
- **AND** on success the service SHALL log `[ecc-scrub] backend=tegra-emc, ECC=enabled`

#### Scenario: Boot-time demand-mode wipe + baseline

- **GIVEN** a successful EMC probe
- **WHEN** the scrub service finishes init
- **THEN** it SHALL run one demand-mode scrub of the kernel image + heap regions, blocking until complete
- **AND** it SHALL read `EMC_ECC_STATUS` and log the post-baseline correctable / uncorrectable counts
- **AND** any non-zero uncorrectable count at this point SHALL be treated as a fatal pre-existing fault and SHALL trigger a kernel panic with a structured `[ecc-scrub] uncorrectable error at <addr>` message

#### Scenario: Patrol-mode steady state

- **GIVEN** the boot-time wipe completed cleanly
- **WHEN** the scrub task enters steady state
- **THEN** it SHALL drive the EMC in patrol mode, advancing the cursor one chunk at a time
- **AND** it SHALL poll `EMC_ECC_SCRUB_STATUS.DONE` between chunks with cooperative yields so inference workloads are not blocked
- **AND** when one full cycle of a region completes, the `cycles_completed` counter SHALL increment and the next due region SHALL be selected per its configured interval

### Requirement: Software-fallback scrub backend

When no hardware scrub backend is available, the scrub service SHALL fall back to a portable software backend that reads and re-writes every `usize` in each region, triggering DRAM-controller ECC corrections as a read side-effect.

#### Scenario: Software backend on non-Tegra platform

- **GIVEN** a SmallAIOS kernel built with `--features ecc-scrub` on a platform without a hardware scrub backend (e.g., development x86 build, RISC-V dev board)
- **WHEN** the scrub service initializes and no hardware backend probes successfully
- **THEN** the service SHALL select the software backend
- **AND** SHALL log `[ecc-scrub] backend=software, hardware scrub unavailable`
- **AND** the scrub task SHALL still advance the cursor per the configured interval

#### Scenario: Software backend preserves content

- **GIVEN** an active software-backend scrub on a region containing application data
- **WHEN** the scrub completes a full cycle
- **THEN** the region content SHALL be byte-identical to its pre-scrub content (the scrub is a read-modify-write of the existing value, not a write of a new value)
- **AND** unit tests SHALL verify this property against a synthetic test region

### Requirement: Watchdog correlation with scrub progress

The scrub task's cursor-advance SHALL be a watchdog-feed event, and a stalled scrub task SHALL trigger a watchdog reset, with the aggressiveness selectable by Cargo feature for development vs. safety-critical deployments.

#### Scenario: Aggressive mode (safety-critical default)

- **GIVEN** a kernel built with `--features ecc-scrub,scrub-watchdog-aggressive` (the default on `ecc-scrub`)
- **WHEN** the scrub cursor fails to advance for `watchdog_threshold` seconds (default 60 s)
- **THEN** the watchdog SHALL fire and reset the system
- **AND** on boot after the reset the kernel SHALL detect the `WdReason::ScrubStall` reset code and SHALL log it prominently

#### Scenario: Permissive mode (development)

- **GIVEN** a kernel built with `--features ecc-scrub,scrub-watchdog-permissive`
- **WHEN** the scrub task heartbeats (regardless of cursor advance)
- **THEN** the watchdog SHALL be fed
- **AND** a stalled cursor SHALL NOT cause a reset — but the scrub stats SHALL surface the stall via the `advanced_at` timestamp being stale

#### Scenario: Mutually exclusive features

- **GIVEN** an attempted build with both `scrub-watchdog-aggressive` and `scrub-watchdog-permissive` enabled
- **THEN** the Cargo build SHALL fail with a `compile_error!` message
- **AND** the doc-comments on each feature SHALL state the mutual exclusion

### Requirement: Scrub service is opt-out on platforms without ECC DRAM

The scrub service SHALL be safe to enable on platforms that do not advertise ECC DRAM — in that case the service SHALL log its absence and SHALL NOT enable the scrub task, so the binary remains correct.

#### Scenario: Non-ECC platform gracefully disables scrub

- **GIVEN** a kernel built with `--features ecc-scrub` on a platform whose DRAM controller does not advertise ECC support (e.g., a typical desktop x86 box without ECC RAM, a RISC-V dev board with plain LPDDR4)
- **WHEN** the scrub service initializes
- **THEN** it SHALL detect the lack of ECC support via the platform-specific probe
- **AND** SHALL log `[ecc-scrub] DRAM does not advertise ECC; service disabled`
- **AND** SHALL NOT spawn the scrub task
- **AND** the kernel SHALL continue to boot normally — the missing scrub service SHALL NOT be fatal
