## ADDED Requirements

### Requirement: Unified ComputeProvider Trait
The system SHALL define a `ComputeProvider` trait that abstracts GPU compute across all vendor backends (NVIDIA, AMD, Intel, Apple).

#### Scenario: Trait exposes device lifecycle
- **WHEN** a GPU backend implements `ComputeProvider`
- **THEN** it MUST provide `device_info()` returning device name, memory size, and compute capability
- **AND** MUST provide `init()` to initialize the device for compute workloads

#### Scenario: Trait exposes memory management
- **WHEN** a GPU backend implements `ComputeProvider`
- **THEN** it MUST provide `alloc(size)` to allocate device memory
- **AND** MUST provide `free(buffer)` to release device memory
- **AND** MUST provide `copy_host_to_device` and `copy_device_to_host` for data transfer

#### Scenario: Trait exposes kernel dispatch
- **WHEN** a GPU backend implements `ComputeProvider`
- **THEN** it MUST provide `load_kernel(name, source)` to compile/load a compute kernel
- **AND** MUST provide `launch(kernel, grid, block, args)` to execute the kernel
- **AND** MUST provide `synchronize()` to wait for all pending kernel completions

#### Scenario: Trait supports operator capability query
- **WHEN** the ONNX executor queries a backend with `supports_op(op_name)`
- **THEN** the backend MUST return `true` only for operators with real GPU kernel implementations
- **AND** MUST return `false` for operators that should fall back to CPU

### Requirement: Enum-Based Backend Selection
The system SHALL use compile-time feature flags and an enum dispatch pattern to select the active GPU backend.

#### Scenario: Select backend via feature flag
- **WHEN** the system is compiled with `feature = "metal"`
- **THEN** the `GpuBackend::Metal` variant MUST be available
- **AND** variants for disabled features MUST be compiled away

#### Scenario: CPU fallback when no GPU available
- **WHEN** no GPU feature flag is enabled or GPU initialization fails
- **THEN** the system MUST fall back to `GpuBackend::Cpu`
- **AND** all operators MUST execute on the CPU execution path

### Requirement: Existing GPU Crate Trait Compliance
All existing GPU crates (NVIDIA, AMD, Intel) SHALL implement the `ComputeProvider` trait by mapping their existing interfaces.

#### Scenario: NVIDIA CudaProvider implements trait
- **WHEN** the NVIDIA crate is compiled with the `cuda` feature
- **THEN** `CudaProvider` MUST implement `ComputeProvider`
- **AND** trait methods MUST delegate to existing `ComputeEngine`, `VramAllocator`, and `DmaEngine`

#### Scenario: AMD and Intel providers implement trait
- **WHEN** the AMD or Intel GPU crate is compiled
- **THEN** `RocmProvider` and `LevelZeroProvider` MUST each implement `ComputeProvider`
- **AND** implementations MAY return stub/not-implemented errors for hardware-dependent operations
