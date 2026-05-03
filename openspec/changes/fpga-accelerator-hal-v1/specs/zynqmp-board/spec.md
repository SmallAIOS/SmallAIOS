## ADDED Requirements

### Requirement: arch/aarch64-zynqmp Crate Existence

A new crate `arch/aarch64-zynqmp` SHALL be added to the workspace, providing board support for AMD/Xilinx Zynq UltraScale+ MPSoC (specifically the K26 SOM as used in KV260 / KR260). The crate SHALL be `#![no_std]`, edition 2021, and SHALL live in dependency Layer 2 (HAL/Drivers).

#### Scenario: Crate builds for aarch64-unknown-none

- **WHEN** `cargo build -p arch-aarch64-zynqmp --target aarch64-unknown-none` is run
- **THEN** the build SHALL succeed
- **AND** the resulting artifact SHALL be `#![no_std]`

#### Scenario: No layer violations introduced

- **WHEN** `just arch-check` is run after this crate is added
- **THEN** no new layering violations SHALL be reported
- **AND** the crate SHALL depend only on Layer 0 and Layer 1 crates

### Requirement: EL1 Boot via FSBL+ATF Chain

The board crate SHALL be designed to run as the EL1 OS payload loaded by AMD's First Stage Bootloader (FSBL) and ARM Trusted Firmware (ATF) in a `BOOT.BIN` image. The crate SHALL NOT contain its own FSBL replacement, PMU firmware, or EL3 secure monitor.

#### Scenario: Entry point matches ATF handoff convention

- **WHEN** the crate's entry point is examined
- **THEN** it SHALL accept the AArch64 boot calling convention used by ATF (DTB pointer in `x0`, zeros in `x1`–`x3`)
- **AND** it SHALL begin execution at EL1
- **AND** it SHALL NOT attempt to re-initialize EL3 or EL2 components

#### Scenario: BOOT.BIN packaging documented

- **WHEN** an engineer follows `docs/zynqmp-boot.md`
- **THEN** they SHALL be able to package the SmallAIOS ELF, AMD FSBL, and ATF into a `BOOT.BIN` using `bootgen`
- **AND** the documentation SHALL pin a specific Vitis version known to produce a working image

### Requirement: Cadence UART Console Driver

The board crate SHALL provide a UART driver for the Cadence UART IP used on Zynq UltraScale+ (NOT the ARM PL011). The driver SHALL be usable as the kernel console for early boot and for runtime logging.

#### Scenario: Boot banner prints to UART0

- **WHEN** SmallAIOS boots on QEMU configured for `xlnx-zcu102` (or equivalent ZynqMP machine model)
- **THEN** a boot banner SHALL appear on the emulated UART0
- **AND** the output SHALL match the existing kernel boot banner contents (subject to platform tag)

#### Scenario: Driver does not depend on PL011

- **WHEN** the Cadence UART driver source is reviewed
- **THEN** it SHALL NOT import from any PL011 driver module
- **AND** it SHALL implement its own register definitions for the Cadence UART IP

### Requirement: GIC-400 Interrupt Controller Support

The board crate SHALL provide an ARM GIC-400 (GICv2) driver covering Distributor, CPU Interface, init for SPI/PPI/SGI interrupt classes, priority masking, and EOI signaling. The driver SHALL be sufficient to handle interrupts from PS peripherals (UART, generic timer) and from PL-side devices routed through the GIC.

#### Scenario: Generic timer interrupt is handled

- **WHEN** the generic timer is programmed to fire after 10 ms and the kernel waits for an interrupt
- **THEN** the GIC-400 driver SHALL route the timer PPI to the kernel's IRQ handler
- **AND** the handler SHALL receive the interrupt and signal EOI

#### Scenario: PL-routed SPI is handled

- **WHEN** a PL-resident device asserts an interrupt routed to a Shared Peripheral Interrupt (SPI) line
- **THEN** the GIC-400 driver SHALL deliver the interrupt to the registered handler
- **AND** the handler SHALL be able to identify the source SPI number

### Requirement: ARMv8 Generic Timer Support

The board crate SHALL provide a driver for the ARMv8 generic timer suitable for kernel tick generation and one-shot deadlines. Counter frequency SHALL be read from `CNTFRQ_EL0`.

#### Scenario: Frequency is read from CNTFRQ_EL0

- **WHEN** the timer driver initializes
- **THEN** it SHALL read `CNTFRQ_EL0` and use that value for time conversions
- **AND** it SHALL NOT hard-code a frequency

#### Scenario: One-shot deadline fires (functional)

- **WHEN** the timer is programmed to fire at a specific `CNTPCT_EL0` value
- **THEN** an interrupt SHALL be raised at or after that value
- **AND** the registered IRQ handler SHALL run to completion in the QEMU `xlnx-zcu102` machine
- **AND** the handler SHALL observe `CNTPCT_EL0 >= deadline` on entry

#### Scenario: One-shot deadline latency (real hardware)

- **WHEN** the timer is programmed to fire at a specific `CNTPCT_EL0` value on a real KV260 / KR260
- **AND** the latency between the deadline and the IRQ handler entry is measured over at least 1000 trials
- **THEN** the 99th-percentile latency SHALL be under 5 µs
- **AND** the measurement methodology SHALL be documented in `docs/zynqmp-boot.md`
- **NOTE:** This scenario is gated to real-hardware test runs only; QEMU scheduling jitter makes the bound unmeasurable in emulation.

### Requirement: DDR Memory Map Definition

The board crate SHALL define the DDR memory map used by SmallAIOS on Zynq UltraScale+ (PS DDR base, size, OCM range, reserved regions for ATF/PMU). The memory map SHALL be expressed as Rust constants the kernel allocator and page tables can consume.

#### Scenario: Memory map is consistent with FSBL handoff

- **WHEN** SmallAIOS boots via the FSBL+ATF chain on QEMU `xlnx-zcu102`
- **THEN** the kernel's DDR base and usable size SHALL match what FSBL reports in the device tree
- **AND** the kernel SHALL NOT use any region marked reserved by ATF (BL31 reserved region, PMU reserved region)

#### Scenario: OCM region is excluded from general allocator

- **WHEN** the kernel allocator is initialized
- **THEN** the on-chip memory (OCM) region SHALL be excluded from the general-purpose heap
- **AND** OCM SHALL be made available as a separately-addressable region for future use (e.g., DMA descriptors)

### Requirement: A53-Only Execution

For this change, the board crate SHALL target only the four Cortex-A53 application cores. Cortex-R5F real-time cores and the Mali-400 GPU SHALL NOT be initialized or used.

#### Scenario: R5F cores are not started

- **WHEN** SmallAIOS boots on this board
- **THEN** no code SHALL release the R5F cores from reset
- **AND** the R5F state SHALL remain whatever ATF/FSBL leaves it as

#### Scenario: Mali-400 is not initialized

- **WHEN** SmallAIOS boots on this board
- **THEN** no code SHALL touch the Mali-400 register space
- **AND** any future GPU-related crate (`arch/aarch64-mali400`) SHALL be a separate change
