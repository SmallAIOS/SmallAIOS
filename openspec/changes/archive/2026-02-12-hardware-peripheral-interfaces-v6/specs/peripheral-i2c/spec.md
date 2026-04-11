# Peripheral I2C Specification

## Overview

I2C (Inter-Integrated Circuit) master controller interface for SmallAIOS. Provides a platform-agnostic HAL trait for I2C bus access, with concrete drivers for ARM MMIO, RISC-V MMIO, Xilinx AXI IIC FPGA soft-IP, and a bitbang GPIO fallback. Gated by the `i2c` feature flag (default OFF). All I2C operations require the `CAP_I2C` capability.

Reference: NXP I2C-bus specification (UM10204 Rev. 7.0), Xilinx AXI IIC Product Guide (PG090).

## Feature Gate

- **Cargo feature:** `i2c` (default OFF)
- WHEN the `i2c` feature is disabled, THEN zero I2C code SHALL be compiled into the binary
- WHEN the `i2c` feature is enabled, THEN the `I2cController` trait, configuration types, and at least one platform driver SHALL be available

## HAL Trait: `I2cController`

### REQ-I2C-001: I2C Speed Modes

The system SHALL support the following I2C clock speeds:

- **Standard mode:** 100 kHz
- **Fast mode:** 400 kHz
- **Fast-mode Plus:** 1 MHz

WHEN an unsupported speed is requested, THEN the driver SHALL return `HalError::InvalidConfig`.

### REQ-I2C-002: Addressing Modes

The system SHALL support:

- **7-bit addressing:** Addresses 0x08–0x77 (excluding reserved ranges per UM10204 §3.1.12)
- **10-bit addressing:** Addresses 0x000–0x3FF using the two-byte address prefix (0b11110xx)

WHEN address 0x00 (general call) is used, THEN the driver SHALL return `HalError::InvalidConfig` unless explicitly enabled in the configuration.

### REQ-I2C-003: I2cConfig Structure

```rust
pub enum I2cSpeed {
    Standard,   // 100 kHz
    Fast,       // 400 kHz
    FastPlus,   // 1 MHz
}

pub enum I2cAddressMode {
    SevenBit,
    TenBit,
}

pub struct I2cConfig {
    pub speed: I2cSpeed,
    pub address_mode: I2cAddressMode,
    pub timeout_us: u32,        // Transaction timeout in microseconds
    pub enable_clock_stretch: bool,
}
```

### REQ-I2C-004: Core I2cController Trait

```rust
pub trait I2cController {
    /// Initialize the I2C controller with the given configuration.
    /// Returns HalError::InitFailed if the controller does not respond.
    fn init(&mut self, config: I2cConfig) -> Result<(), HalError>;

    /// Write bytes to a device at the given address.
    /// Generates START, address+W, data bytes, STOP.
    /// Returns HalError::NackReceived if the device does not acknowledge.
    /// Returns HalError::ArbitrationLost if another master wins arbitration.
    /// Returns HalError::Timeout if the transaction exceeds timeout_us.
    fn write(&mut self, addr: u16, data: &[u8]) -> Result<(), HalError>;

    /// Read bytes from a device at the given address.
    /// Generates START, address+R, reads len bytes with ACK (NACK on last), STOP.
    fn read(&mut self, addr: u16, buf: &mut [u8]) -> Result<usize, HalError>;

    /// Write then read in a single transaction using repeated START.
    /// Generates START, address+W, write_data, REPEATED START, address+R, reads into buf, STOP.
    /// This is the standard pattern for register reads on I2C sensors/codecs.
    fn write_read(&mut self, addr: u16, write_data: &[u8], buf: &mut [u8]) -> Result<usize, HalError>;

    /// Reset the I2C controller and bus (generate 9 clock pulses to unstick SDA).
    fn reset(&mut self) -> Result<(), HalError>;

    /// Return whether the bus is currently busy (SDA or SCL held low).
    fn is_busy(&self) -> bool;
}
```

### REQ-I2C-005: Error Handling

New `HalError` variants SHALL be added:

- `NackReceived` — target device did not acknowledge address or data byte
- `ArbitrationLost` — another master won bus arbitration during multi-master operation

WHEN a NACK is received during address phase, THEN the driver SHALL generate a STOP condition and return `HalError::NackReceived`.
WHEN a NACK is received during data phase, THEN the driver SHALL generate a STOP condition, return `HalError::NackReceived`, and report the number of bytes successfully transferred (via a future extension or log).
WHEN arbitration is lost, THEN the driver SHALL release the bus and return `HalError::ArbitrationLost`.

### REQ-I2C-006: Clock Stretching

WHEN `enable_clock_stretch` is true, THEN the master SHALL wait for SCL to be released by the slave before proceeding.
WHEN the clock stretch exceeds `timeout_us`, THEN the driver SHALL return `HalError::Timeout` and attempt bus recovery (9 clock pulses).

### REQ-I2C-007: Bus Recovery

The `reset()` method SHALL:
1. Generate 9 clock pulses on SCL with SDA released
2. Generate a STOP condition (SDA low→high while SCL high)
3. Verify SDA is released (high)
4. Return `HalError::HardwareError` if SDA remains stuck low after recovery

## Platform Drivers

### REQ-I2C-010: ARM MMIO Driver

WHEN running on ARM64 (AArch64) platforms with memory-mapped I2C controllers, THEN the driver SHALL:
- Use volatile MMIO reads/writes with appropriate memory barriers
- Support DTB-based discovery of I2C controller base addresses and IRQ numbers
- Support common ARM I2C controller register layouts (Designware I2C compatible)
- Integrate with GICv3 interrupt controller for I2C interrupt delivery

Supported platforms: NVIDIA Jetson (Tegra I2C), Raspberry Pi (Broadcom BSC), generic Designware I2C.

### REQ-I2C-011: RISC-V MMIO Driver

WHEN running on RISC-V platforms, THEN the driver SHALL:
- Use volatile MMIO reads/writes with fence instructions for ordering
- Support DTB-based discovery via `compatible = "sifive,i2c0"` or `"opencores,i2c"` strings
- Integrate with PLIC for interrupt delivery

Supported platforms: SiFive HiFive, PolarFire SoC (hard I2C on RISC-V subsystem).

### REQ-I2C-012: Xilinx AXI IIC FPGA Driver

WHEN running on FPGA platforms with Xilinx AXI IIC soft-IP, THEN the driver SHALL:
- Access the AXI IIC register set via the existing `FpgaFabric` trait (`read_reg`/`write_reg`)
- Support the AXI IIC register map: CR, SR, TX_FIFO, RX_FIFO, ADR, TX_FIFO_OCY, RX_FIFO_OCY, ISR, IER
- Support DTB-based discovery via `compatible = "xlnx,xps-iic-2.00.a"` string
- Support DMA mode for bulk transfers via the `FpgaFabric::dma_start` interface

Supported platforms: Zynq UltraScale+, PolarFire SoC (FPGA fabric side).

### REQ-I2C-013: Bitbang GPIO Fallback

WHEN no dedicated I2C controller is available and the `gpio` feature is also enabled, THEN a software bitbang driver SHALL be available that:
- Uses two GPIO pins (SDA, SCL) configured as open-drain outputs
- Implements the I2C protocol in software with timing delays
- Supports Standard mode (100 kHz) only (timing cannot be guaranteed for faster modes)
- Returns `HalError::NotSupported` for Fast and Fast-Plus modes

### REQ-I2C-014: DTB-Based Discovery

WHEN the platform provides a Device Tree Blob (DTB), THEN the I2C subsystem SHALL:
- Parse DTB nodes with `compatible` strings matching supported controllers
- Extract `reg` property for MMIO base address and size
- Extract `interrupts` property for IRQ assignment
- Extract `clock-frequency` property for bus speed (default: 100000)
- Register discovered controllers with the device manager

## Capability Integration

### REQ-I2C-020: Capability Gating

- A new capability type `CAP_I2C` SHALL be defined in the `security` crate
- WHEN a process calls `sys_dev_open()` for an I2C device, THEN the kernel SHALL verify the caller holds `CAP_I2C`
- WHEN a process does not hold `CAP_I2C`, THEN `sys_dev_open()` SHALL return `EPERM`
- `CAP_I2C` SHALL be granular to individual I2C bus instances (e.g., `CAP_I2C(bus=0)` vs `CAP_I2C(bus=1)`)

### REQ-I2C-021: Syscall Interface

I2C devices SHALL be accessible via the existing device syscall interface:
- `sys_dev_enumerate()` — lists discovered I2C controllers
- `sys_dev_open(id)` — opens an I2C controller (checks `CAP_I2C`)
- `sys_dev_ioctl(handle, I2C_SET_CONFIG, &config)` — configure speed/addressing
- `sys_dev_ioctl(handle, I2C_WRITE, &write_req)` — write transaction
- `sys_dev_ioctl(handle, I2C_READ, &read_req)` — read transaction
- `sys_dev_ioctl(handle, I2C_WRITE_READ, &write_read_req)` — combined write+read
- `sys_dev_close(handle)` — close device

## Safety and Verification

### REQ-I2C-030: Formal Verification

A TLA+ model SHALL verify:
- I2C bus arbitration correctness (multi-master scenarios)
- No deadlock when clock stretching with timeout
- STOP condition is always generated after NACK

### REQ-I2C-031: Test Coverage

- Unit tests for all trait methods with mock hardware
- Integration tests for each platform driver against register-level simulations
- MC/DC coverage on all safety-critical decision points (arbitration, NACK handling, bus recovery)
- Fuzzing of address/data parameters to verify bounds checking

### REQ-I2C-032: WCET Bounds

Each I2C operation SHALL have a documented worst-case execution time:
- `write(addr, &[u8; N])`: WCET = overhead + (N + 1) × 9 × bit_period + timeout_us
- `read(addr, &mut [u8; N])`: WCET = overhead + (N + 1) × 9 × bit_period + timeout_us
- `write_read(addr, &[u8; M], &mut [u8; N])`: WCET = overhead + (M + N + 2) × 9 × bit_period + timeout_us
- `reset()`: WCET = 9 × 9 × bit_period_100khz (always at standard mode)
