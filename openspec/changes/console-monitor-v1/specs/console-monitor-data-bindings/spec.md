## ADDED Requirements

### Requirement: Consumption-Only Telemetry Access

The console monitor SHALL read all displayed data via the same `mgmt::Config` / telemetry channel a Zenoh subscriber would use. It SHALL NOT introduce a new kernel-internal interface, SHALL NOT escalate privilege, SHALL NOT add collectors or data sources, and SHALL NOT add fields to `Config`.

#### Scenario: No new Config fields

- **WHEN** a reviewer diffs the `mgmt::Config` surface before and after this change
- **THEN** no new fields SHALL be present

#### Scenario: Same channel as an external subscriber

- **WHEN** the monitor and an external Zenoh subscriber observe the same metric key at the same time
- **THEN** both SHALL observe the same published values via the same telemetry channel

### Requirement: Field-to-Source Binding Table

Every on-screen field SHALL have a documented binding to its publishing source in the existing telemetry pipeline: CPU (per-core utilization and load average from the scheduler's run-queue stats, topology from `kernel`); Memory (page-allocator counts from `kernel`); GPU (utilization from `arch/nvidia` CUDA driver profiling counters under `gpu-profile`, plus graph-cache hit rate and capture count from the `gpu-resident-vision-hybrid-v1` cache); Network (per-interface RX/TX byte/packet/error counters from `net`); Filesystem (per-mount IOPS/bytes/latency from `peripheral` block-device drivers); Peripheral I/O (per-bus byte/event counters from `peripheral::{i2c, spi, gpio, uart, camera_csi, audio_i2s}`); Models (per-model QPS/p50/p99/batch fill rate/error count from the `mgmt-zenoh-telemetry` publishers in `container/`); Sessions (read-only view of the live session table from `auth`).

#### Scenario: Every rendered field traces to a source

- **WHEN** a reviewer inspects the data-source bindings for each rendered field
- **THEN** every field SHALL map to exactly one documented publishing source from the list above
- **AND** no field SHALL be computed from an undocumented source

#### Scenario: GPU trait bindings are backend-agnostic

- **WHEN** a GPU backend other than NVIDIA implements the same telemetry trait in a future change
- **THEN** the monitor SHALL display its values through the same binding without code changes to the binding table
- **AND** backends that do not implement the trait SHALL surface as missing (rendered `n/a` per `console-monitor-tui`)

### Requirement: Binding Regression CI Test

CI SHALL include a test asserting that every source bound by the monitor is still published under its bound key. Removing or renaming a published source consumed by the monitor SHALL fail this test loudly rather than letting the monitor silently render zeros.

#### Scenario: Renamed telemetry key fails CI

- **WHEN** a change renames a published metric key that the monitor binds to
- **THEN** the binding regression test SHALL fail
- **AND** the failure message SHALL name the missing source

#### Scenario: Intact bindings pass

- **WHEN** all bound sources are published under their expected keys
- **THEN** the binding regression test SHALL pass
