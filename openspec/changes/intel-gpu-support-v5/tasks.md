# Tasks: Intel GPU Support (intel-gpu-support-v5)

## Phase 1: Crate Scaffolding
- [x] T1: Create `arch/intel_gpu/Cargo.toml` with workspace integration and feature flags
- [x] T2: Create `arch/intel_gpu/src/lib.rs` with module declarations and GpuError enum
- [x] T3: Add `arch/intel_gpu` to workspace members in root `Cargo.toml`

## Phase 2: PCIe Enumeration (pcie.rs)
- [x] T4: Implement PciAddress, BarType, BaseAddressRegister structs
- [x] T5: Implement PciDevice with is_intel(), is_display_controller(), is_3d_controller()
- [x] T6: Implement PciScanner with scan(), intel_devices(), device_count()
- [x] T7: Mock scan with Arc A770, A750, DC Max 1550, non-Intel device
- [x] T8: Write 20 unit tests for pcie.rs

## Phase 3: GPU Identification (gpu_id.rs)
- [x] T9: Define GpuArchitecture enum (XeLP, XeHPG, XeHPC, XeLPG, Xe2, Unknown)
- [x] T10: Define ComputeCapability with EU count comparison
- [x] T11: Define GpuInfo struct with Intel-specific fields
- [x] T12: Implement identify_gpu() for all supported device IDs
- [x] T13: Write 20 unit tests for gpu_id.rs

## Phase 4: Memory Management (memory.rs)
- [x] T14: Implement MemoryRegion, GpuAllocation structs
- [x] T15: Implement VramAllocator with 70/30 static/dynamic split
- [x] T16: Implement alloc, free, used_bytes, free_bytes, total_used, total_free
- [x] T17: Write 17 unit tests for memory.rs

## Phase 5: DMA Engine (dma.rs)
- [x] T18: Implement DmaDirection, DmaStatus, TransferId, DmaTransfer
- [x] T19: Implement DmaEngine with submit, start, complete, fail, cancel
- [x] T20: Write 18 unit tests for dma.rs

## Phase 6: Compute Engine (compute.rs)
- [x] T21: Define SimdWidth, Dim3, LaunchConfig for Intel EU model
- [x] T22: Implement KernelId, KernelStatus, Kernel structs
- [x] T23: Implement ComputeEngine with launch, dispatch, complete, fail, synchronize
- [x] T24: Write 20 unit tests for compute.rs

## Phase 7: GPU Initialization (gpu_init.rs)
- [x] T25: Define GpuState, PowerState, GpuRegisters, InitConfig
- [x] T26: Implement GpuContext lifecycle: map_bars, initialize_engines, start, suspend, resume, reset
- [x] T27: Write 20 unit tests for gpu_init.rs

## Phase 8: SPIR-V Kernel Registry (spirv_kernels.rs)
- [x] T28: Define SpirvKernelType, DataPrecision, SpirvKernel
- [x] T29: Implement SpirvRegistry with register_defaults (11 standard kernels)
- [x] T30: Write 25 unit tests for spirv_kernels.rs

## Phase 9: Level Zero Provider (level_zero_provider.rs)
- [x] T31: Define OperatorMapping, ExecutionStep, ExecutionPlan, ProviderStatus
- [x] T32: Implement LevelZeroProvider wiring all subsystems
- [x] T33: Implement map_operator, launch_kernel, load_weights, allocate_workspace, synchronize
- [x] T34: Write 22 unit tests for level_zero_provider.rs

## Phase 10: Verification
- [x] T35: cargo check passes
- [x] T36: cargo test passes (150+ tests)
- [x] T37: cargo clippy -- -D warnings passes with zero warnings
- [x] T38: cargo fmt -- --check passes
