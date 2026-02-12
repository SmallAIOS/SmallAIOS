# Peripheral GPIO Specification

## Overview

GPIO (General-Purpose Input/Output) controller for SmallAIOS. Provides a platform-agnostic HAL trait for digital pin control with interrupt support, pull configuration, and atomic set/clear operations. Concrete drivers for ARM PL061/generic MMIO, RISC-V MMIO, and Xilinx AXI GPIO FPGA soft-IP. Gated by the `gpio` feature flag (default OFF). All GPIO operations require the `CAP_GPIO` capability.

Primary use cases: interrupt lines from sensors/radar ("data ready"), inference-triggered actuation (relays, LEDs, alarms), chip-select bitbang for SPI, and SDA/SCL bitbang for I2C fallback.

Reference: ARM PrimeCell GPIO PL061 Technical Reference Manual, Xilinx AXI GPIO Product Guide (PG144).

## Feature Gate

- **Cargo feature:** `gpio` (default OFF)
- WHEN the `gpio` feature is disabled, THEN zero GPIO code SHALL be compiled into the binary
- WHEN the `gpio` feature is enabled, THEN the `GpioController` trait, configuration types, and at least one platform driver SHALL be available

## HAL Trait: `GpioController`

### REQ-GPIO-001: Pin Modes

The system SHALL support the following pin modes:

```rust
pub enum GpioPinMode {
    /// High-impedance input.
    Input,
    /// Push-pull output.
    Output,
    /// Open-drain output (for I2C bitbang, wired-OR).
    OpenDrain,
    /// Alternate function (peripheral-controlled, e.g., I2C, SPI, UART).
    AlternateFunction(u8),
    /// Analog (disable digital input buffer, for ADC pins).
    Analog,
}
```

### REQ-GPIO-002: Pull Configuration

```rust
pub enum GpioPull {
    None,       // Floating (no pull)
    PullUp,     // Internal pull-up resistor
    PullDown,   // Internal pull-down resistor
}
```

WHEN a pull configuration is requested but not supported by hardware, THEN the driver SHALL return `HalError::NotSupported`.

### REQ-GPIO-003: Interrupt Edge Configuration

```rust
pub enum GpioInterruptEdge {
    Rising,
    Falling,
    Both,
    LevelHigh,
    LevelLow,
}
```

### REQ-GPIO-004: GpioConfig Structure

```rust
pub struct GpioConfig {
    pub mode: GpioPinMode,
    pub pull: GpioPull,
    pub interrupt: Option<GpioInterruptEdge>,
    pub debounce_us: u32,   // 0 = no debounce
}
```

### REQ-GPIO-005: Core GpioController Trait

```rust
pub trait GpioController {
    /// Configure a single GPIO pin.
    /// Returns HalError::OutOfRange if pin number exceeds the controller's pin count.
    fn configure(&mut self, pin: u8, config: GpioConfig) -> Result<(), HalError>;

    /// Read the current level of an input pin (true = high, false = low).
    fn read(&self, pin: u8) -> Result<bool, HalError>;

    /// Set an output pin high.
    fn set_high(&mut self, pin: u8) -> Result<(), HalError>;

    /// Set an output pin low.
    fn set_low(&mut self, pin: u8) -> Result<(), HalError>;

    /// Toggle an output pin.
    fn toggle(&mut self, pin: u8) -> Result<(), HalError>;

    /// Atomically set multiple pins using a bitmask.
    /// set_mask: bits to set high. clear_mask: bits to set low.
    /// Uses a single MMIO write to avoid glitches.
    fn set_mask(&mut self, set_mask: u32, clear_mask: u32) -> Result<(), HalError>;

    /// Read all pins as a bitmask (up to 32 pins per controller).
    fn read_all(&self) -> Result<u32, HalError>;

    /// Enable interrupt for a pin (must be configured with interrupt edge first).
    fn enable_interrupt(&mut self, pin: u8) -> Result<(), HalError>;

    /// Disable interrupt for a pin.
    fn disable_interrupt(&mut self, pin: u8) -> Result<(), HalError>;

    /// Acknowledge and clear a pending interrupt. Returns the pin number that triggered.
    fn irq_handler(&mut self) -> Result<u8, HalError>;

    /// Return the number of GPIO pins on this controller.
    fn pin_count(&self) -> u8;

    /// Reset all pins to input/floating (safe default).
    fn reset(&mut self) -> Result<(), HalError>;
}
```

### REQ-GPIO-006: Atomic Pin Operations

- `set_high()`, `set_low()`, and `set_mask()` SHALL use atomic MMIO writes where hardware supports it (ARM: BSRR register, PL061: DATA with address masking)
- WHEN atomic operations are not supported, THEN the driver SHALL use a read-modify-write with interrupts disabled to prevent race conditions
- `toggle()` SHALL use read-modify-write and MUST be called with interrupts disabled if concurrent access is possible

### REQ-GPIO-007: Debounce

- WHEN `debounce_us > 0`, THEN the controller SHALL filter input transitions shorter than the specified duration
- Hardware debounce SHALL be used when available (PL061 has no hardware debounce; software timer-based fallback)
- Debounce SHALL apply to both interrupt generation and `read()` calls

### REQ-GPIO-008: Interrupt Delivery

- GPIO interrupts SHALL integrate with the platform interrupt controller (GICv3 on ARM, PLIC on RISC-V)
- WHEN an edge is detected matching the configured `GpioInterruptEdge`, THEN an interrupt SHALL be raised
- The `irq_handler()` method SHALL clear the pending interrupt and return the triggering pin number
- Multiple pins may share a single interrupt line; the handler SHALL read the interrupt status register to identify which pin(s) triggered

## Platform Drivers

### REQ-GPIO-010: ARM PL061 / Generic MMIO Driver

WHEN running on ARM64 platforms, THEN the driver SHALL:
- Support ARM PL061 register layout (GPIODATA, GPIODIR, GPIOIS, GPIOIBE, GPIOIEV, GPIOIE, GPIORIS, GPIOMIS, GPIOIC, GPIOAFSEL)
- Support the PL061 data register address masking for atomic bit access
- Support 8 pins per PL061 instance (multiple instances for more pins)
- Support DTB-based discovery via `compatible = "arm,pl061"` string
- Also support generic GPIO MMIO for Broadcom (RPi) and Tegra (Jetson) layouts

### REQ-GPIO-011: RISC-V MMIO Driver

WHEN running on RISC-V platforms, THEN the driver SHALL:
- Support SiFive GPIO register layout
- Support DTB-based discovery via `compatible = "sifive,gpio0"` string
- Integrate with PLIC for interrupt delivery
- Support up to 32 pins per controller instance

### REQ-GPIO-012: Xilinx AXI GPIO FPGA Driver

WHEN running on FPGA platforms with Xilinx AXI GPIO soft-IP, THEN the driver SHALL:
- Access registers via the existing `FpgaFabric` trait
- Support dual-channel AXI GPIO (Channel 1 and Channel 2, up to 32 pins each)
- Support register map: GPIO_DATA, GPIO_TRI, GPIO2_DATA, GPIO2_TRI, GIER, IP_ISR, IP_IER
- Support DTB-based discovery via `compatible = "xlnx,xps-gpio-1.00.a"` string

## Capability Integration

### REQ-GPIO-020: Capability Gating

- A new capability type `CAP_GPIO` SHALL be defined in the `security` crate
- WHEN a process calls `sys_dev_open()` for a GPIO controller, THEN the kernel SHALL verify `CAP_GPIO`
- `CAP_GPIO` SHALL be granular to individual GPIO controller instances and optionally to pin ranges

### REQ-GPIO-021: Syscall Interface

GPIO devices SHALL be accessible via the existing device syscall interface:
- `sys_dev_enumerate()` — lists discovered GPIO controllers
- `sys_dev_open(id)` — opens a GPIO controller (checks `CAP_GPIO`)
- `sys_dev_ioctl(handle, GPIO_CONFIGURE, &pin_config)` — configure a pin
- `sys_dev_ioctl(handle, GPIO_READ, pin)` — read pin level
- `sys_dev_ioctl(handle, GPIO_WRITE, &pin_value)` — set pin level
- `sys_dev_ioctl(handle, GPIO_SET_MASK, &mask)` — atomic multi-pin set/clear
- `sys_dev_ioctl(handle, GPIO_READ_ALL, &buf)` — read all pins
- `sys_dev_ioctl(handle, GPIO_ENABLE_IRQ, pin)` — enable pin interrupt
- `sys_dev_close(handle)` — close device

## Safety and Verification

### REQ-GPIO-030: Formal Verification

A TLA+ model SHALL verify:
- No pin mode conflict (two processes configuring same pin)
- Interrupt acknowledgment completes before re-enabling
- Atomic set_mask correctness (no intermediate states visible)

### REQ-GPIO-031: Test Coverage

- Unit tests for all trait methods with mock hardware
- Integration tests for each platform driver
- Concurrent access tests (multiple tasks using set_mask)
- MC/DC coverage on interrupt edge detection logic

### REQ-GPIO-032: Safe Defaults

- On `reset()`, all pins SHALL be configured as Input with no pull (high-impedance)
- This is the safest default — no output driving, no current draw
- Boot-time GPIO state SHALL match `reset()` default
