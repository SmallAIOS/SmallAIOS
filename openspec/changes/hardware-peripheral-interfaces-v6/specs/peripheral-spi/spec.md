# Peripheral SPI Specification

## Overview

SPI (Serial Peripheral Interface) master controller for SmallAIOS. Provides a platform-agnostic HAL trait for SPI bus access with full-duplex transfers, chip-select management, and DMA support. Concrete drivers for ARM MMIO, RISC-V MMIO, and Xilinx AXI Quad SPI FPGA soft-IP. Gated by the `spi` feature flag (default OFF). All SPI operations require the `CAP_SPI` capability.

Primary use cases: radar ADC data streams (TI AWR/IWR SPI interface), flash storage, ADC/DAC peripherals, high-speed sensor data.

Reference: Motorola SPI specification, Xilinx AXI Quad SPI Product Guide (PG153).

## Feature Gate

- **Cargo feature:** `spi` (default OFF)
- WHEN the `spi` feature is disabled, THEN zero SPI code SHALL be compiled into the binary
- WHEN the `spi` feature is enabled, THEN the `SpiController` trait, configuration types, and at least one platform driver SHALL be available

## HAL Trait: `SpiController`

### REQ-SPI-001: SPI Clock Modes

The system SHALL support all four SPI clock modes:

| Mode | CPOL | CPHA | Description |
|------|------|------|-------------|
| 0 | 0 | 0 | Clock idle low, sample on leading edge |
| 1 | 0 | 1 | Clock idle low, sample on trailing edge |
| 2 | 1 | 0 | Clock idle high, sample on leading edge |
| 3 | 1 | 1 | Clock idle high, sample on trailing edge |

### REQ-SPI-002: Clock Speed Configuration

The system SHALL support configurable SPI clock frequencies:
- Minimum: 100 kHz
- Maximum: 50 MHz (hardware dependent)
- The driver SHALL select the nearest achievable frequency not exceeding the requested value
- WHEN the requested frequency cannot be achieved, THEN the driver SHALL return `HalError::InvalidConfig`

### REQ-SPI-003: SpiConfig Structure

```rust
pub enum SpiMode {
    Mode0,  // CPOL=0, CPHA=0
    Mode1,  // CPOL=0, CPHA=1
    Mode2,  // CPOL=1, CPHA=0
    Mode3,  // CPOL=1, CPHA=1
}

pub enum SpiBitOrder {
    MsbFirst,
    LsbFirst,
}

pub struct SpiConfig {
    pub mode: SpiMode,
    pub clock_hz: u32,          // Requested clock frequency in Hz
    pub bit_order: SpiBitOrder,
    pub word_size: u8,          // Bits per word: 8, 16, or 32
    pub cs_active_low: bool,    // Chip select polarity (true = active low, default)
    pub timeout_us: u32,        // Transaction timeout in microseconds
}
```

### REQ-SPI-004: Core SpiController Trait

```rust
pub trait SpiController {
    /// Initialize the SPI controller with the given configuration.
    fn init(&mut self, config: SpiConfig) -> Result<(), HalError>;

    /// Full-duplex transfer: simultaneously write tx_data and read into rx_buf.
    /// tx_data and rx_buf MUST be the same length.
    /// Chip select is asserted before transfer and deasserted after.
    fn transfer(&mut self, cs: u8, tx_data: &[u8], rx_buf: &mut [u8]) -> Result<(), HalError>;

    /// Write-only transfer (MOSI only, MISO data discarded).
    /// Chip select is asserted before and deasserted after.
    fn write(&mut self, cs: u8, data: &[u8]) -> Result<(), HalError>;

    /// Read-only transfer (sends zeros on MOSI, captures MISO).
    /// Chip select is asserted before and deasserted after.
    fn read(&mut self, cs: u8, buf: &mut [u8]) -> Result<(), HalError>;

    /// Start a DMA-based transfer for bulk data (e.g., radar ADC streams).
    /// Returns a DmaToken for polling completion via the FpgaFabric or DMA HAL.
    /// Only available on platforms with DMA support; returns HalError::NotSupported otherwise.
    fn transfer_dma(&mut self, cs: u8, tx_addr: u64, rx_addr: u64, len: u32) -> Result<DmaToken, HalError>;

    /// Manually assert chip select (for multi-transfer sequences).
    fn cs_assert(&mut self, cs: u8) -> Result<(), HalError>;

    /// Manually deassert chip select.
    fn cs_deassert(&mut self, cs: u8) -> Result<(), HalError>;

    /// Reset the SPI controller.
    fn reset(&mut self) -> Result<(), HalError>;
}
```

### REQ-SPI-005: Chip Select Management

- The controller SHALL support up to 4 hardware chip-select lines (CS0–CS3)
- WHEN `cs` index exceeds the number of available CS lines, THEN the driver SHALL return `HalError::OutOfRange`
- WHEN `transfer()`, `write()`, or `read()` is called, THEN CS SHALL be automatically asserted before the first clock edge and deasserted after the last
- WHEN `cs_assert()` / `cs_deassert()` are used for manual control, THEN automatic CS management SHALL be bypassed for that sequence
- CS lines SHALL be active-low by default (configurable via `cs_active_low`)

### REQ-SPI-006: DMA Support

- DMA transfers SHALL use the existing `DmaToken` / `DmaDescriptor` types from `kernel/src/hal.rs`
- WHEN DMA is available, `transfer_dma()` SHALL program the DMA controller and return immediately with a `DmaToken`
- The caller polls for completion via `FpgaFabric::dma_poll()` or a platform-specific DMA poll
- WHEN DMA is not available on the platform, `transfer_dma()` SHALL return `HalError::NotSupported`

## Platform Drivers

### REQ-SPI-010: ARM MMIO Driver

WHEN running on ARM64 platforms with memory-mapped SPI controllers, THEN the driver SHALL:
- Support Designware SSI / ARM PL022-compatible register layouts
- Use volatile MMIO reads/writes with appropriate memory barriers
- Support DTB-based discovery of SPI controller base addresses
- Integrate with GICv3 for interrupt-driven transfers

Supported platforms: NVIDIA Jetson (Tegra SPI), Raspberry Pi (Broadcom SPI), generic PL022.

### REQ-SPI-011: RISC-V MMIO Driver

WHEN running on RISC-V platforms, THEN the driver SHALL:
- Support SiFive SPI controller register layout
- Use volatile MMIO with fence instructions
- Support DTB-based discovery via `compatible = "sifive,spi0"` string
- Integrate with PLIC for interrupt delivery

### REQ-SPI-012: Xilinx AXI Quad SPI FPGA Driver

WHEN running on FPGA platforms with Xilinx AXI Quad SPI soft-IP, THEN the driver SHALL:
- Access the AXI Quad SPI register set via the existing `FpgaFabric` trait
- Support the register map: SPICR, SPISR, SPI DTR, SPI DRR, SPISSR, TX_FIFO_OCY, RX_FIFO_OCY, DGIER, IPISR, IPIER
- Support standard SPI, dual SPI, and quad SPI modes
- Support DTB-based discovery via `compatible = "xlnx,xps-spi-2.00.a"` string
- Support DMA via `FpgaFabric::dma_start` for bulk radar/ADC data

Supported platforms: Zynq UltraScale+, PolarFire SoC (FPGA fabric).

## Capability Integration

### REQ-SPI-020: Capability Gating

- A new capability type `CAP_SPI` SHALL be defined in the `security` crate
- WHEN a process calls `sys_dev_open()` for an SPI device, THEN the kernel SHALL verify `CAP_SPI`
- `CAP_SPI` SHALL be granular to individual SPI bus instances (e.g., `CAP_SPI(bus=0)`)

### REQ-SPI-021: Syscall Interface

SPI devices SHALL be accessible via the existing device syscall interface:
- `sys_dev_enumerate()` — lists discovered SPI controllers
- `sys_dev_open(id)` — opens an SPI controller (checks `CAP_SPI`)
- `sys_dev_ioctl(handle, SPI_SET_CONFIG, &config)` — configure mode/clock/CS
- `sys_dev_ioctl(handle, SPI_TRANSFER, &transfer_req)` — full-duplex transfer
- `sys_dev_ioctl(handle, SPI_WRITE, &write_req)` — write-only
- `sys_dev_ioctl(handle, SPI_READ, &read_req)` — read-only
- `sys_dev_ioctl(handle, SPI_TRANSFER_DMA, &dma_req)` — DMA transfer
- `sys_dev_close(handle)` — close device

## Safety and Verification

### REQ-SPI-030: Formal Verification

A TLA+ model SHALL verify:
- SPI clock/data phase relationship correctness for all 4 modes
- Chip select assertion/deassertion ordering (no glitches)
- DMA completion before buffer reuse

### REQ-SPI-031: Test Coverage

- Unit tests for all trait methods with mock hardware
- Integration tests for each platform driver
- Fuzz testing of clock divider calculation
- MC/DC coverage on CS management and mode selection paths

### REQ-SPI-032: WCET Bounds

Each SPI operation SHALL have documented worst-case execution time:
- `transfer(cs, &[u8; N], &mut [u8; N])`: WCET = overhead + N × 8 / clock_hz + timeout_us
- `write(cs, &[u8; N])`: WCET = overhead + N × 8 / clock_hz + timeout_us
- `transfer_dma()`: WCET = setup_overhead (DMA completion is async)
