## ADDED Requirements

### Requirement: SMMU500 driver initialization on Tegra234

The kernel SHALL initialize the Tegra234 System MMU (ARM SMMU500, SMMUv2-shaped) during boot, before any DMA-capable peripheral driver is allowed to issue DMA transactions, on builds enabling the `smmu` Cargo feature on `smallaios-arch-aarch64`.

#### Scenario: SMMU registers are probed from the device tree

- **GIVEN** a SmallAIOS kernel built with `--features tegra234,smmu` running on an Orin NX device with a JetPack 6 / L4T R36.4 device tree
- **WHEN** `smmu::init(dtb)` runs during boot
- **THEN** the driver SHALL walk the DTB for nodes matching `compatible = "arm,mmu-500"` or `compatible = "nvidia,tegra234-smmu"` and SHALL read the SMMU MMIO base address from the node's `reg` property
- **AND** the driver SHALL read `GR0::IDR0`, `GR0::IDR1`, and `GR0::IDR2` to determine the number of context banks, the number of stream-match registers, and the supported translation regimes
- **AND** the driver SHALL log a structured boot line of the form `[smmu] SMMU500 attached, <N> context banks, <M> stream-match entries` over the configured serial console

#### Scenario: Boot-time policy transitions from identity-map to enforce

- **GIVEN** the SMMU driver has completed register probing
- **WHEN** the kernel proceeds past `smmu::init` and before any peripheral driver init
- **THEN** the SMMU SHALL be programmed with a deny-all default for every stream ID (S2CR type = fault) — peripherals that have not been explicitly attached via `attach_stream` SHALL be unable to issue DMA
- **AND** a 100-millisecond grace period SHALL log any UEFI-residual DMA faults as informational warnings rather than fatal errors
- **AND** after the grace period any subsequent fault SHALL be handled by the structured fault handler (see "SMMU fault handling" requirement)

#### Scenario: No DMA-capable driver init runs before SMMU init

- **GIVEN** the kernel boot sequence
- **WHEN** the kernel boots
- **THEN** `smmu::init` SHALL be called after `mem::init` and before any DMA-capable driver init (host1x, GPU HAL, PCIe controller, USB XHCI, SDMMC)
- **AND** a debug assertion SHALL trip if a driver init runs while `smmu::is_initialized()` returns false

### Requirement: Stream-ID-based DMA isolation for GPU tensor mappings

The `sys_tensor_map_gpu` syscall SHALL, on builds with the `smmu` feature active, route GPU-visible tensor pointers through a per-stream IOMMU translation rather than handing the GPU a raw physical address.

#### Scenario: sys_tensor_map_gpu returns an IOVA, not a physical address

- **GIVEN** a tensor allocated via `sys_mem_alloc` and the caller has both `TensorBuffer:WRITE` on the handle and `GpuDevice:EXECUTE` on the target device
- **WHEN** the caller invokes `sys_tensor_map_gpu(handle, device_id)`
- **THEN** the syscall SHALL allocate an IOVA range from the GPU stream's IOVA allocator covering the tensor's physical pages
- **AND** the syscall SHALL install an SMMU stage-1 mapping for the GPU's Tegra234 stream ID (`0x07`) pointing the IOVA range at the tensor's physical pages with the requested permissions
- **AND** the syscall SHALL flush the SMMU TLB for the GPU stream before returning
- **AND** the syscall SHALL return the IOVA as the `GpuPtr` value — the value SHALL NOT equal the physical address of the tensor pages in the general case
- **AND** the cooperative-async runtime SHALL treat the TLB-flush poll loop as a yield point per the SmallAIOS scheduling model

#### Scenario: Capability denial blocks the SMMU programming

- **GIVEN** a caller that does NOT hold `GpuDevice:EXECUTE` on the requested device
- **WHEN** the caller invokes `sys_tensor_map_gpu(handle, device_id)`
- **THEN** the syscall SHALL return the existing capability-denial error code unchanged
- **AND** the SMMU SHALL NOT be modified — no IOVA allocation, no page-table edit, no TLB flush

#### Scenario: Symmetric unmap releases SMMU resources

- **GIVEN** a tensor that has been mapped to the GPU via `sys_tensor_map_gpu` (returning IOVA `X`)
- **WHEN** the caller invokes `sys_tensor_unmap_gpu(handle, device_id)`
- **THEN** the syscall SHALL remove the SMMU stage-1 mapping covering IOVA `X`
- **AND** the syscall SHALL flush the SMMU TLB for the GPU stream
- **AND** the syscall SHALL return the IOVA range to the GPU stream's IOVA allocator so that a subsequent `sys_tensor_map_gpu` for a different tensor MAY reuse those IOVAs

### Requirement: Per-peripheral stream attachment for non-GPU DMA paths

Every DMA-capable peripheral driver in SmallAIOS SHALL, on `smmu`-feature builds, attach its hardware engine's stream ID to the SMMU before enabling DMA in that engine.

#### Scenario: USB XHCI attaches before enabling DMA

- **GIVEN** the `smallaios-usb` XHCI controller driver running on a `--features tegra234,smmu` build
- **WHEN** the driver's `init` function executes
- **THEN** the driver SHALL call `smmu::attach_stream(StreamId::USB, <driver's root PA>, Perms::RW)` before writing any USB controller register that enables DMA
- **AND** a deny-all DMA attempt from the USB controller SHALL produce a kernel fault interrupt (see "SMMU fault handling") rather than scribbling DRAM

#### Scenario: host1x, PCIe, SDMMC attach symmetrically

- **GIVEN** the host1x v6 driver, PCIe root-complex driver, and SDMMC driver
- **WHEN** each driver's init executes on a `--features tegra234,smmu` build
- **THEN** each driver SHALL call `smmu::attach_stream` with its corresponding Tegra234 stream ID(s) before enabling DMA in its engine
- **AND** the stream-ID constants used SHALL come from `arch/aarch64/src/smmu/stream_id.rs` (the canonical Tegra234 TRM-derived constants) — drivers SHALL NOT hand-write stream-ID literals

### Requirement: SMMU fault handling and telemetry

The SMMU global fault interrupt SHALL be routed to a structured fault handler that logs the offending stream, address, and syndrome, and SHALL increment per-stream fault counters surfaced via the kernel telemetry interface.

#### Scenario: Fault on an unmapped IOVA produces structured output

- **GIVEN** an SMMU-enabled build with at least one stream attached
- **WHEN** any peripheral attempts a DMA access that the SMMU page-table walk fails (unmapped IOVA, permission violation, or external abort)
- **THEN** the SMMU SHALL raise a global fault interrupt
- **AND** the kernel handler SHALL read `FSR`, `FSYNR0`, `FSYNR1`, `FAR` from the relevant context bank and produce a structured `SmmuFault { stream_id, iova, fault_kind, syndrome }` value
- **AND** the handler SHALL log the structured fault to the boot console with severity `error`
- **AND** the per-stream fault counter SHALL increment

#### Scenario: Fatal-fault Cargo feature converts faults to panics

- **GIVEN** a kernel built with `--features tegra234,smmu,smmu-fatal-fault`
- **WHEN** any SMMU fault occurs after the boot-time grace period
- **THEN** the kernel SHALL panic with a structured message naming the stream ID, IOVA, and fault kind — the run SHALL NOT continue
- **AND** this behavior SHALL be the recommended configuration for safety-critical (DO-178C DAL A) builds where any DMA fault indicates a definite software defect

#### Scenario: Telemetry exposes per-stream counters

- **GIVEN** a kernel built with `--features tegra234,smmu` and the existing kernel telemetry surface enabled
- **WHEN** an operator queries the telemetry endpoint
- **THEN** the response SHALL include per-stream-ID fault-counter values for every stream that has ever been attached during the current boot
- **AND** the counters SHALL be monotonic — they SHALL reset only on kernel boot, never on attach/detach
