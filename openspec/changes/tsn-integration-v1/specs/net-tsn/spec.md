## ADDED Requirements

### Requirement: IEEE 802.1AS gPTP clock synchronization

When the `net` crate is built with the `tsn` Cargo feature and the host has a TSN-capable NIC with hardware timestamping, the kernel SHALL implement an IEEE 802.1AS (gPTP) profile of PTPv2 and SHALL synchronize the local hardware clock to a TSN-domain grandmaster with sub-microsecond mean offset.

#### Scenario: Hardware-timestamped gPTP slave to an upstream grandmaster

- **GIVEN** the unikernel is built with `--features tsn`
- **GIVEN** the host has an Intel i210 (or equivalent supported NIC) with hardware timestamping enabled
- **GIVEN** the host is connected to a TSN-capable switch acting as a gPTP grandmaster (or forwarding from one)
- **WHEN** the unikernel boots and the gPTP daemon initializes
- **THEN** the daemon SHALL exchange `PDelay_Req` / `PDelay_Resp` / `PDelay_Resp_Follow_Up` messages with the upstream peer to measure mean path delay
- **AND** the daemon SHALL consume `Sync` + `Follow_Up` from the grandmaster and adjust the local Precision Hardware Clock (PHC) accordingly
- **AND** within 60 seconds of boot, the running mean of `tsn.gptp.offset_ns` over a 10-minute window SHALL be < 500 ns
- **AND** `tsn.gptp.path_delay_ns` SHALL be reported and SHALL be stable to within ±100 ns on a quiet network

#### Scenario: Software-timestamp fallback warns loudly

- **GIVEN** the unikernel is built with `--features tsn`
- **GIVEN** the host's NIC does not support hardware timestamping (e.g., a development virtio-net adapter)
- **WHEN** the gPTP daemon initializes
- **THEN** the daemon SHALL fall back to software timestamping
- **AND** the daemon SHALL emit a `warn!` log at boot stating that software-only mode cannot achieve sub-microsecond accuracy and is suitable for development / testing only
- **AND** production deployments SHALL be expected to use a NIC from the supported set (Intel i210 / i225 / i226 / E810, or equivalents documented in `docs/tsn-integration.md`)

### Requirement: IEEE 802.1Qbv scheduled-traffic gate enforcement

When the `tsn` feature is enabled and the host's NIC supports 802.1Qbv hardware gates, the kernel SHALL program a Gate Control List on the NIC from the configured schedule and SHALL rely on the NIC to enforce gate-open / gate-close boundaries at gPTP-synchronized times.

#### Scenario: GCL programmed at boot from TOML configuration

- **GIVEN** the operator provides a TOML schedule via `SMALLAIOS_TSN_SCHEDULE` (container path) or boot argument (unikernel path) describing a `cycle_time_ns`, a sequence of `gate_control_list` entries, and an interface binding
- **GIVEN** the bound NIC supports 802.1Qbv (e.g., Intel i210)
- **WHEN** the unikernel boots and the TSN subsystem initializes
- **THEN** the kernel SHALL parse and validate the TOML schedule (total entry duration matches `cycle_time_ns`; all gate-state bitmasks well-formed)
- **AND** the kernel SHALL translate the GCL into the NIC's vendor-specific register format
- **AND** the kernel SHALL program the NIC via the `TsnNicDriver::set_gcl` trait method
- **AND** the first cycle SHALL start at a gPTP-aligned base time after the gPTP daemon has reached steady-state synchronization

#### Scenario: Wire-line enforcement of gate windows

- **GIVEN** a configured 5 ms cycle with a 1 ms `inference-result` window (gates TC7 / TC6 open) followed by a 4 ms `best-effort` window (gates TC0-TC3 open)
- **GIVEN** the unikernel is generating traffic on both TC7 and TC0 traffic classes
- **WHEN** an external TAP captures wire-line traffic
- **THEN** TC7 frames SHALL appear on the wire exclusively within the 1 ms scheduled window of each cycle
- **AND** TC0 frames SHALL appear exclusively within the 4 ms best-effort window
- **AND** the `tsn.qbv.gate_open_late_count` counter SHALL be 0 under nominal operation; non-zero indicates a clock-sync issue or schedule misconfiguration

### Requirement: Cooperative scheduler honors TSN-derived deadlines

The cooperative inference scheduler SHALL accept per-op deadlines tied to TSN scheduled-traffic windows and SHALL evaluate the deadline at each op-boundary yield point.

#### Scenario: Op completes before deadline

- **GIVEN** an inference op chain associated with the `inference-result` TSN window (deadline = end of the 1 ms window)
- **GIVEN** the chain's actual execution cost is well within 1 ms
- **WHEN** the scheduler dispatches the chain
- **THEN** each op-boundary yield SHALL re-evaluate `remaining_time = deadline - gptp_now`
- **AND** the chain SHALL complete before the deadline
- **AND** the `tsn.deadline.met_count{window_id="inference-result"}` counter SHALL increment by one

#### Scenario: Deadline miss with Warn action

- **GIVEN** an op chain with `on_miss = "warn"` configured for its window
- **GIVEN** the chain's actual execution exceeds the deadline
- **WHEN** the scheduler detects, at an op-boundary yield, that `gptp_now + estimated_next_op_cost > deadline`
- **THEN** the scheduler SHALL log a structured warning naming the window, the deadline, and the projected overshoot
- **AND** the scheduler SHALL continue executing the chain (Warn = best effort)
- **AND** the `tsn.deadline.miss_count{window_id=...}` counter SHALL increment by one

#### Scenario: Deadline miss with Abort action

- **GIVEN** an op chain with `on_miss = "abort"` configured for its window (hard-real-time profile)
- **GIVEN** the chain's actual execution would exceed the deadline
- **WHEN** the scheduler detects the projected overshoot
- **THEN** the scheduler SHALL abandon the remaining ops in the chain
- **AND** the scheduler SHALL emit a structured warning naming the window and the abandoned op
- **AND** the scheduler SHALL await the next cycle's window for the next chain
- **AND** the `tsn.deadline.miss_count{window_id=...}` counter SHALL increment by one

### Requirement: NIC driver shim is abstracted for future NIC support

The `TsnNicDriver` trait SHALL abstract NIC-specific TSN feature programming so that new NICs (Intel i225/i226, Intel E810, Marvell Prestera, NXP S32G built-in switch) can be added without changes to the gPTP daemon, scheduler, or configuration parser.

#### Scenario: Adding a new NIC requires only implementing the trait

- **GIVEN** a future change adding Intel i225 support
- **WHEN** the change implements the `TsnNicDriver` trait for the i225 register layout
- **THEN** no changes SHALL be required to `net/src/tsn/gptp.rs`, `net/src/tsn/qbv.rs`, `kernel/src/sched/tsn.rs`, or the TOML parser
- **AND** the existing TOML schedule format SHALL transparently support the new NIC by setting `nic = "i225"` in the configuration

### Requirement: Jetson Orin is explicitly out of scope for Qbv enforcement

The kernel SHALL NOT advertise full TSN endpoint capability on Jetson Orin and SHALL document the limitation prominently.

#### Scenario: Jetson Orin can run gPTP-slave only

- **GIVEN** a Jetson Orin NX or AGX host with the Tegra234 EQOS Ethernet controller
- **WHEN** the unikernel is built with `--features tsn` (research / experimental case)
- **THEN** the gPTP daemon SHALL function (Tegra234 EQOS supports gPTP hardware timestamping)
- **AND** the Qbv subsystem SHALL detect at boot that the EQOS controller does NOT support 802.1Qbv gate enforcement and SHALL emit a clear error
- **AND** the error message SHALL state that Jetson can be a gPTP slave but cannot be a scheduled-traffic endpoint, and point at `docs/tsn-integration.md` for the supported NIC matrix

#### Scenario: Production Jetson AI workflows are unaffected

- **GIVEN** the existing `just build-container-arm` / `just docker-build-jetson` workflows without `--features tsn`
- **WHEN** those workflows are run
- **THEN** the produced artifacts SHALL be identical to develop (no TSN code paths compiled in, no behavior change)

### Requirement: Single-NIC initial support, extensible

The v1 implementation SHALL ship with Intel i210 support as the canonical reference NIC; additional NICs SHALL be added in follow-up changes.

#### Scenario: Intel i210 is the v1 reference

- **GIVEN** the v1 implementation
- **WHEN** a deployment uses an Intel i210
- **THEN** gPTP synchronization SHALL meet the < 500 ns mean offset target
- **AND** Qbv GCL programming SHALL succeed with up to 16 GCL entries
- **AND** the i210-specific gate-granularity (1 µs rounding) SHALL be documented in `docs/tsn-integration.md`

#### Scenario: Other NICs report unsupported in v1

- **GIVEN** a host with a non-i210 TSN-capable NIC (e.g., i225, E810) in v1
- **WHEN** the unikernel boots with `--features tsn` and the TOML schedule references that NIC
- **THEN** the kernel SHALL emit a clear error stating the NIC is not supported in v1 and pointing at the supported-NIC matrix
- **AND** the error SHALL include a link to the GitHub issue tracker for requesting / contributing additional NIC support
