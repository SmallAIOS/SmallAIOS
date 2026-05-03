## ADDED Requirements

### Requirement: AXI Master Driver Framework

The change SHALL provide a reusable AXI master driver framework usable from the `arch/aarch64-zynqmp` board crate (and future Zynq/Versal boards). The framework SHALL expose typed register access primitives, MMIO read/write barriers, and an AXI peripheral abstraction that other drivers (DMA, accelerator backends) can compose with.

#### Scenario: Typed register access compiles

- **WHEN** a driver author writes `let v: u32 = my_axi_peripheral.reg::<CtrlReg>().read();`
- **THEN** the code SHALL compile and SHALL emit a single 32-bit MMIO read with a `DMB`/`DSB` barrier appropriate for AArch64

#### Scenario: Mismatched register width fails to compile

- **WHEN** a driver author attempts to read a `u64` from a register typed as `u32`
- **THEN** the code SHALL fail to compile

### Requirement: AXI-DMA Engine Driver

The framework SHALL provide a driver for AXI-DMA scatter-gather descriptor-based transfers, supporting MM2S (memory-to-stream) and S2MM (stream-to-memory) channels. The driver SHALL expose an async-style interface that completes when the DMA engine raises its IRQ.

#### Scenario: MM2S transfer completes

- **WHEN** a caller submits an MM2S descriptor pointing to a 64 KiB buffer with a configured stream destination
- **THEN** the driver SHALL program the DMA engine, return immediately, and complete its future when the DMA's completion IRQ fires
- **AND** the buffer's bytes SHALL have been streamed to the destination

#### Scenario: S2MM transfer completes

- **WHEN** a caller submits an S2MM descriptor pointing to a 64 KiB receive buffer
- **THEN** the driver SHALL program the DMA engine and complete its future when the IRQ fires
- **AND** the receive buffer SHALL contain the streamed bytes

#### Scenario: Cancelled future cleans up

- **WHEN** a caller drops an in-flight DMA future before the IRQ fires
- **THEN** the driver SHALL halt the channel, drain or discard any pending bytes, and free the descriptor
- **AND** subsequent transfers SHALL succeed without leaked descriptors

### Requirement: Typed Cache-Coherency for PS-PL Buffers

DMA-capable buffers crossing the PS-PL boundary SHALL be expressed using a typed wrapper that encodes the AXI port semantics. Coherent ports (HPC0/HPC1, ACP) SHALL NOT expose explicit cache-maintenance methods. Non-coherent ports (HP0–HP3) SHALL expose `clean_for_device()` and `invalidate_for_cpu()` and SHALL require callers to invoke them at the appropriate boundaries.

#### Scenario: Coherent buffer skips manual cache maintenance

- **WHEN** a driver allocates `DmaBuffer<HpcPort>` and submits it to a DMA transfer
- **THEN** no `clean_for_device` or `invalidate_for_cpu` calls SHALL be required for correctness
- **AND** any attempt to call those methods SHALL fail to compile

#### Scenario: Non-coherent buffer requires explicit maintenance

- **WHEN** a driver allocates `DmaBuffer<HpPort>` and writes data to it from the CPU
- **THEN** the driver SHALL call `clean_for_device()` before submitting the buffer to a DMA transfer toward PL
- **AND** SHALL call `invalidate_for_cpu()` before reading PL-written bytes back from the CPU

#### Scenario: Forgetting non-coherent maintenance is detectable in tests

- **WHEN** a unit test wraps a `DmaBuffer<HpPort>` with a debug-mode tracker that records cache maintenance calls
- **AND** the test simulates a write-then-DMA-submit sequence without calling `clean_for_device`
- **THEN** the tracker SHALL record the missing call
- **AND** the test SHALL fail with a diagnostic identifying the missing maintenance

### Requirement: IRQ Routing Integration

The AXI-DMA driver SHALL register its completion interrupts via the GIC-400 driver (from the `zynqmp-board` capability). Multiple DMA channels SHALL be able to share IRQ handling without lost wakeups under concurrent load.

#### Scenario: Two channels complete concurrently

- **WHEN** two DMA channels are active and their completion IRQs fire within a short window
- **THEN** both channels' futures SHALL be woken
- **AND** no completion notification SHALL be lost
- **AND** the order of completion observation SHALL match the order of IRQ assertion

### Requirement: No Layer Violations or Runtime Dependencies

The framework SHALL live in dependency Layer 2. It SHALL NOT depend on any Layer 3 crate. It SHALL NOT introduce new runtime crate dependencies (must remain `#![no_std]`, no new third-party crates).

#### Scenario: Cargo.toml introduces no new third-party deps

- **WHEN** the `Cargo.toml` of the framework crate (or module) is reviewed
- **THEN** all dependencies SHALL be path dependencies on existing workspace crates
- **AND** no `crates.io` registry dependency SHALL be added

#### Scenario: arch-check passes

- **WHEN** `just arch-check` is run after the framework lands
- **THEN** no new layering violations SHALL be reported
