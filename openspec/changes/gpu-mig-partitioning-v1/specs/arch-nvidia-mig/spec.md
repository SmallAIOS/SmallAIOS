## ADDED Requirements

### Requirement: MIG slice detection at CUDA context init

When the `smallaios-arch-nvidia` crate is built with the `mig` Cargo feature, the kernel SHALL detect whether the CUDA device assigned to the unikernel is a MIG slice or a full GPU, using the CUDA driver API.

#### Scenario: Detect a MIG slice via UUID prefix and MIG_MODE attribute

- **GIVEN** the unikernel is built with `--features mig`
- **GIVEN** the assigned CUDA device is a MIG slice on an A100, H100, H200, or B100-class GPU
- **WHEN** the kernel initializes the CUDA context
- **THEN** the kernel SHALL call `cuDeviceGetUuid` and observe the UUID begins with the prefix `MIG-`
- **AND** the kernel SHALL call `cuDeviceGetAttribute(CU_DEVICE_ATTRIBUTE_MIG_MODE)` and observe the result is `1`
- **AND** the kernel SHALL classify the device as a MIG slice in the `NvidiaDevice` descriptor (`is_mig_slice = true`)
- **AND** boot logs SHALL include `arch-nvidia: detected MIG slice <uuid> profile=<profile>` at info level

#### Scenario: Detect a full GPU on a MIG-capable host where MIG is not configured

- **GIVEN** the unikernel is built with `--features mig`
- **GIVEN** the assigned CUDA device is a full A100 / H100 / H200 GPU (MIG either disabled or not configured)
- **WHEN** the kernel initializes the CUDA context
- **THEN** the kernel SHALL observe the UUID begins with `GPU-` (not `MIG-`)
- **AND** the kernel SHALL classify the device as a full GPU (`is_mig_slice = false`)
- **AND** the kernel SHALL emit a runtime error and refuse to start, with a message pointing the operator to either rebuild without `--features mig` or assign a MIG slice
- **AND** the error message SHALL reference `docs/gpu-mig.md` for the support matrix

### Requirement: Slice-aware resource budgeting honors hardware partition limits

When running on a MIG slice, the kernel SHALL respect the slice's hardware partition limits (DRAM, SM count, L2 cache) when sizing GPU resources.

#### Scenario: GPU tensor pool honors the slice's DRAM partition

- **GIVEN** a unikernel running on a 1g.10gb MIG slice (10 GB DRAM partition) of an A100 80GB
- **WHEN** the GPU tensor pool is initialized
- **THEN** the pool's DRAM ceiling SHALL be the slice's partition size (10 GB), not the full GPU's memory (80 GB)
- **WHEN** an allocation request exceeds 10 GB
- **THEN** the call SHALL return `Err(MemError::OutOfDeviceMemory)` BEFORE invoking `cudaMalloc`
- **AND** the error message SHALL include the slice's profile and partition size

#### Scenario: Scheduler stream count bounded by the slice's SM count

- **GIVEN** a unikernel running on a 1g.10gb MIG slice (14 SMs on A100) or a 1g.10gb-equivalent slice
- **WHEN** the inference scheduler initializes its CUDA stream pool
- **THEN** the stream count SHALL be bounded by the slice's compute pipeline count (1 for A100 slices, more for H100)
- **AND** the scheduler SHALL NOT launch more concurrent streams than the slice's hardware can isolate

### Requirement: MIG-aware telemetry

When running on a MIG slice, the kernel SHALL extend the `gpu-profile` telemetry path with MIG-specific fields.

#### Scenario: Telemetry includes slice identity and profile

- **GIVEN** a unikernel built with `--features mig` and `--features gpu-profile`
- **GIVEN** the unikernel is running on a MIG slice
- **WHEN** the unikernel exits and the `CudaRuntime::drop` dump is captured
- **THEN** the dump SHALL include `device.uuid`, `device.is_mig = true`, `device.mig.profile`, `device.mig.gpu_instance_id`, `device.mig.compute_instance_id`, `device.mig.dram_partition_bytes`
- **AND** the dump SHALL include `device.mig.peak_memory_used_bytes` sampled at exit
- **AND** the dump SHALL include `device.mig.peak_sm_utilization_pct` (sampled from NVML where available; 0 otherwise)

#### Scenario: Non-MIG telemetry is unchanged

- **GIVEN** a unikernel built with `--features mig` and `--features gpu-profile`
- **GIVEN** the unikernel is running on a full GPU (not in this requirement's MIG path — but the build flag is on)
- **WHEN** the existing fail-fast check ensures the run does not start, the telemetry dump SHALL NOT be produced (by construction); separately, when the unikernel is built WITHOUT `--features mig` and runs on any device, telemetry SHALL be identical to develop (no `mig.*` fields).

### Requirement: Fail-fast on mismatched feature flag and device

The kernel SHALL detect and surface misconfigurations where the `mig` feature flag does not match the assigned device's MIG status.

#### Scenario: Build with --features mig, assigned a non-MIG device

- **GIVEN** the unikernel is built with `--features mig`
- **GIVEN** the assigned device is NOT a MIG slice (full GPU, non-MIG GPU like L4 or L40, Jetson, or no GPU)
- **WHEN** the kernel initializes the CUDA context
- **THEN** the kernel SHALL emit a structured error to the boot log and SHALL refuse to enter the main inference loop
- **AND** the error SHALL identify the assigned device (UUID + product name)
- **AND** the error SHALL provide a hint for resolution (rebuild without `--features mig`, or assign a MIG slice via `NVIDIA_VISIBLE_DEVICES=MIG-GPU-...`)
- **AND** the error SHALL reference `docs/gpu-mig.md` for the support matrix

#### Scenario: Build without --features mig, assigned a MIG slice

- **GIVEN** the unikernel is built WITHOUT `--features mig`
- **GIVEN** the assigned device IS a MIG slice
- **WHEN** the kernel initializes the CUDA context
- **THEN** the kernel SHALL detect the MIG-prefixed UUID (best-effort, no NVML required) and SHALL log a warning at boot indicating MIG-specific telemetry will be unavailable
- **AND** the unikernel SHALL continue to run normally on the slice (the CUDA runtime transparently honors the slice's resource limits)

#### Scenario: Build-time exclusion of mig + tegra-orin

- **GIVEN** a build invocation that enables both `--features mig` AND `--features tegra-orin`
- **WHEN** `cargo build` runs
- **THEN** the build SHALL fail with a `compile_error!` message stating that `mig` is for datacenter GPUs (A100/H100/H200/B100), `tegra-orin` is for Jetson Orin (Ampere GA10B) which does not support MIG, and that the operator must enable one or the other but not both
- **AND** the error message SHALL reference `docs/gpu-mig.md` for the support matrix

### Requirement: Jetson Orin is explicitly out of scope for MIG

The kernel SHALL NOT advertise MIG support on Jetson Orin (any SKU) and SHALL document the absence prominently.

#### Scenario: Jetson Orin documentation callout

- **GIVEN** a user reading `docs/gpu-mig.md` or the README hardware matrix
- **WHEN** the user looks for Jetson MIG support
- **THEN** the documentation SHALL include a prominent callout stating that Jetson Orin (Nano, NX, AGX) silicon does NOT support MIG and SHALL refer the user to the existing Jetson container path (`Dockerfile.jetson`) for Jetson workflows

#### Scenario: Jetson build path is unaffected by the MIG feature

- **GIVEN** the existing `just build-container-arm` / `just docker-build-jetson` workflows
- **WHEN** those workflows are run without `--features mig`
- **THEN** the produced artifacts SHALL be identical to develop (no MIG code paths compiled in, no behavior change)
