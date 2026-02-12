# Peripheral UART Specification

## Overview

UART (Universal Asynchronous Receiver/Transmitter) controller for SmallAIOS. Provides a platform-agnostic HAL trait for serial communication with configurable baud rates, framing, flow control, and both interrupt-driven and DMA modes. Concrete drivers for ARM PL011, NS16550A-compatible, SiFive UART, and Xilinx AXI UART Lite FPGA soft-IP. Gated by the `uart` feature flag (default OFF). All UART operations require the `CAP_UART` capability.

Primary use cases: serial data ingestion from radar modules (TI AWR/IWR UART output for processed detections), LiDAR sensors (Benewake TFmini, Livox), GPS/GNSS receivers (NMEA 0183, UBX binary protocol), and debug/diagnostic consoles.

Reference: 16550 UART specification, ARM PL011 Technical Reference Manual (DDI0183), Xilinx AXI UART Lite Product Guide (PG142).

## Feature Gate

- **Cargo feature:** `uart` (default OFF)
- WHEN the `uart` feature is disabled, THEN zero UART peripheral code SHALL be compiled into the binary
- NOTE: Existing boot console UARTs (PL011 in aarch64 crate, NS16550A in riscv64 crate) are NOT affected by this flag — they are separate, minimal boot-time drivers. This feature controls the full-featured UART HAL for application-level serial I/O.

## HAL Trait: `UartController`

### REQ-UART-001: Baud Rate Configuration

The system SHALL support the following standard baud rates:
- 9600, 19200, 38400, 57600, 115200, 230400, 460800, 921600, 1000000, 2000000, 3000000

WHEN a non-standard baud rate is requested, the driver SHALL attempt to configure the nearest achievable rate. WHEN the error exceeds 2%, the driver SHALL return `HalError::InvalidConfig`.

### REQ-UART-002: Frame Format

```rust
pub enum UartParity {
    None,
    Even,
    Odd,
}

pub enum UartStopBits {
    One,
    Two,
}

pub enum UartFlowControl {
    None,
    RtsCts,     // Hardware flow control using RTS/CTS lines
}

pub enum UartRxMode {
    /// Interrupt-driven: IRQ raised when RX FIFO threshold reached.
    Interrupt { fifo_threshold: u8 },
    /// DMA: received data written directly to memory ring buffer.
    Dma { buffer_addr: u64, buffer_size: u32 },
    /// Polling: caller explicitly polls for data (no IRQ/DMA).
    Polling,
}
```

### REQ-UART-003: UartConfig Structure

```rust
pub struct UartConfig {
    pub baud_rate: u32,
    pub data_bits: u8,              // 7 or 8
    pub parity: UartParity,
    pub stop_bits: UartStopBits,
    pub flow_control: UartFlowControl,
    pub rx_mode: UartRxMode,
    pub tx_fifo_size: u8,           // TX FIFO depth (hardware-dependent, informational)
    pub timeout_us: u32,            // Read timeout (0 = non-blocking)
}
```

### REQ-UART-004: Core UartController Trait

```rust
pub trait UartController {
    /// Initialize the UART controller with the given configuration.
    fn init(&mut self, config: UartConfig) -> Result<(), HalError>;

    /// Write bytes to the UART TX FIFO.
    /// Returns the number of bytes actually written (may be less than data.len()
    /// if TX FIFO is full).
    fn write(&mut self, data: &[u8]) -> Result<usize, HalError>;

    /// Write all bytes, blocking until TX FIFO drains as needed.
    /// Returns HalError::Timeout if timeout_us is exceeded.
    fn write_all(&mut self, data: &[u8]) -> Result<(), HalError>;

    /// Read available bytes from the RX FIFO into buf.
    /// Returns the number of bytes read (may be 0 if no data available).
    /// Non-blocking in Polling mode; blocks up to timeout_us in Interrupt mode.
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, HalError>;

    /// Read until buf is full or timeout_us expires.
    /// Returns the number of bytes actually read.
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<usize, HalError>;

    /// Read until a delimiter byte is encountered or buf is full.
    /// Useful for line-oriented protocols (NMEA sentences end with \n).
    /// Returns number of bytes read including the delimiter.
    fn read_until(&mut self, delimiter: u8, buf: &mut [u8]) -> Result<usize, HalError>;

    /// Flush the TX FIFO (block until all pending bytes are transmitted).
    fn flush(&mut self) -> Result<(), HalError>;

    /// Return the number of bytes available in the RX FIFO.
    fn rx_available(&self) -> usize;

    /// Handle a UART interrupt (RX data available, TX empty, error).
    fn irq_handler(&mut self) -> Result<UartIrqSource, HalError>;

    /// Clear error flags and reset the UART controller.
    fn reset(&mut self) -> Result<(), HalError>;
}

pub enum UartIrqSource {
    /// RX FIFO reached threshold (data available for reading).
    RxDataAvailable,
    /// RX FIFO timeout (data sitting in FIFO without reaching threshold).
    RxTimeout,
    /// TX FIFO empty (can accept more data).
    TxEmpty,
    /// Framing error (stop bit not detected).
    FramingError,
    /// Parity error.
    ParityError,
    /// RX FIFO overrun (data lost).
    Overrun,
    /// Break condition detected.
    Break,
}
```

### REQ-UART-005: Error Handling

New `HalError` variants SHALL be added:

- `FramingError` — stop bit not detected at expected position
- `ParityError` — received parity bit does not match calculated parity
- `Overrun` — RX FIFO overflow, data lost

WHEN a framing or parity error occurs, THEN the affected byte SHALL be discarded and the error reported.
WHEN an overrun occurs, THEN the driver SHALL clear the overrun flag, log the event, and continue receiving.

### REQ-UART-006: Flow Control

WHEN `RtsCts` flow control is enabled:
- The UART SHALL deassert RTS when the RX FIFO is nearly full (threshold configurable)
- The UART SHALL pause TX when CTS is deasserted by the remote device
- WHEN CTS remains deasserted longer than `timeout_us`, THEN `write()` SHALL return `HalError::Timeout`

### REQ-UART-007: DMA Mode

WHEN `UartRxMode::Dma` is configured:
- Received bytes SHALL be written directly to the specified memory buffer via DMA
- The DMA SHALL operate in circular/ring-buffer mode
- The driver SHALL track the DMA write pointer to determine how many new bytes are available
- `read()` SHALL copy from the DMA buffer to the caller's buffer
- DMA mode is preferred for high baud rates (>= 1 Mbps) to avoid interrupt overhead

## Platform Drivers

### REQ-UART-010: ARM PL011 Driver

WHEN running on ARM64 platforms with PL011 UART controllers, THEN the driver SHALL:
- Support the full PL011 register set (DR, RSR, FR, IBRD, FBRD, LCR_H, CR, IFLS, IMSC, RIS, MIS, ICR, DMACR)
- Support 16-entry TX and RX FIFOs
- Support fractional baud rate divisor for precise rate generation
- Support RTS/CTS hardware flow control
- Support DMA via PL011's DMA request signals
- Support DTB-based discovery via `compatible = "arm,pl011"` string

NOTE: This is a separate, full-featured PL011 driver from the minimal boot console in `arch/aarch64`. The boot console only uses TX polling; this driver supports full bidirectional interrupt/DMA operation.

### REQ-UART-011: NS16550A-Compatible Driver

WHEN running on platforms with 16550-compatible UART controllers, THEN the driver SHALL:
- Support the standard 16550A register set (RBR, THR, IER, IIR, FCR, LCR, MCR, LSR, MSR, SCR, DLL, DLM)
- Support 16-byte TX and RX FIFOs
- Support divisor latch for baud rate configuration
- Support DTB-based discovery via `compatible = "ns16550a"` or `"ns16550"` strings

Supported platforms: x86 COM ports (I/O port or MMIO), various ARM SoCs with 16550-compatible UARTs.

### REQ-UART-012: SiFive UART Driver

WHEN running on RISC-V SiFive platforms, THEN the driver SHALL:
- Support the SiFive UART register layout (txdata, rxdata, txctrl, rxctrl, ie, ip, div)
- Support 8-entry TX and RX FIFOs
- Support DTB-based discovery via `compatible = "sifive,uart0"` string
- Integrate with PLIC for interrupt delivery

### REQ-UART-013: Xilinx AXI UART Lite FPGA Driver

WHEN running on FPGA platforms with Xilinx AXI UART Lite soft-IP, THEN the driver SHALL:
- Access registers via the existing `FpgaFabric` trait
- Support the AXI UART Lite register map: RX_FIFO, TX_FIFO, STAT_REG, CTRL_REG
- Note: AXI UART Lite has fixed 8N1 framing and fixed baud rate (configured at synthesis time)
- Support DTB-based discovery via `compatible = "xlnx,xps-uartlite-1.00.a"` string

For configurable baud rate on FPGA, the AXI UART 16550 IP should be used instead (compatible with REQ-UART-011).

## Capability Integration

### REQ-UART-020: Capability Gating

- A new capability type `CAP_UART` SHALL be defined in the `security` crate
- WHEN a process calls `sys_dev_open()` for a UART device, THEN the kernel SHALL verify `CAP_UART`
- `CAP_UART` SHALL be granular to individual UART instances (e.g., `CAP_UART(port=0)`)
- Debug console UART MAY have a separate policy (always accessible to kernel, restricted for userspace)

### REQ-UART-021: Syscall Interface

UART devices SHALL be accessible via the existing device syscall interface:
- `sys_dev_enumerate()` — lists discovered UART controllers
- `sys_dev_open(id)` — opens a UART (checks `CAP_UART`)
- `sys_dev_ioctl(handle, UART_SET_CONFIG, &config)` — configure baud/framing/flow control
- `sys_dev_ioctl(handle, UART_WRITE, &data)` — write bytes
- `sys_dev_ioctl(handle, UART_READ, &read_req)` — read bytes
- `sys_dev_ioctl(handle, UART_READ_LINE, &buf)` — read until newline (NMEA)
- `sys_dev_ioctl(handle, UART_FLUSH, 0)` — flush TX
- `sys_dev_ioctl(handle, UART_RX_AVAILABLE, &count)` — query available bytes
- `sys_dev_close(handle)` — close device

## Sensor Protocol Helpers

### REQ-UART-030: NMEA 0183 Parser (Optional)

WHEN the `uart` feature is enabled, a lightweight NMEA 0183 parser MAY be included:
- Parse standard NMEA sentences ($GPGGA, $GPRMC, $GPGLL)
- Validate checksum (XOR of bytes between $ and *)
- Extract latitude, longitude, altitude, speed, timestamp
- This is a convenience for GPS/GNSS receivers; raw UART access is always available

### REQ-UART-031: Radar Detection Protocol Helpers (Optional)

Common radar UART output formats MAY have lightweight parsers:
- TI radar output protocol (TLV format: frame header, detected objects, range/velocity/azimuth)
- Parsers are opt-in and do not add to binary size unless used

## Safety and Verification

### REQ-UART-040: Formal Verification

A TLA+ model SHALL verify:
- RX FIFO overflow detection and data integrity after overrun recovery
- Flow control handshake (RTS/CTS) prevents data loss
- DMA ring buffer wrap-around correctness

### REQ-UART-041: Test Coverage

- Unit tests for all trait methods with mock UART hardware
- Baud rate divisor calculation accuracy tests for all supported rates
- Flow control scenario tests (RTS/CTS assertion/deassertion timing)
- Overrun recovery tests
- MC/DC coverage on error detection paths (framing, parity, overrun)

### REQ-UART-042: WCET Bounds

- `write(data)`: WCET = overhead + min(data.len(), fifo_size) × byte_time
- `read(buf)`: WCET = overhead + min(rx_available, buf.len()) × copy_time
- `read_until(delim, buf)`: WCET = timeout_us (bounded by configuration)
- `flush()`: WCET = fifo_size × byte_time (worst case: full FIFO draining)
