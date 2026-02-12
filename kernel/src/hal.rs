// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Hardware Abstraction Layer (HAL) traits.
//!
//! Defines architecture-agnostic interfaces for bus peripheral controllers,
//! FPGA fabric access, and platform-level operations. Concrete implementations
//! live in the architecture crates (`smallaios-arch-*`).

#![allow(dead_code)]

use core::fmt;

// ---------------------------------------------------------------------------
// Common HAL error
// ---------------------------------------------------------------------------

/// Errors returned by HAL operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalError {
    /// Controller initialization failed or timed out.
    InitFailed,
    /// Transmit FIFO or buffer is full.
    TxFifoFull,
    /// No data available in the receive FIFO.
    RxEmpty,
    /// Bus controller is in Bus Off state.
    BusOff,
    /// Invalid configuration parameter.
    InvalidConfig,
    /// Timeout waiting for hardware response.
    Timeout,
    /// DMA transfer failed (bus error or decode error).
    DmaError,
    /// Memory-mapped register access failed.
    MmioError,
    /// Feature not supported by this hardware.
    NotSupported,
    /// Interrupt handling error.
    InterruptError,
    /// Address or offset out of range.
    OutOfRange,
    /// Hardware reported an internal error.
    HardwareError,
    /// I2C: target device did not acknowledge (address or data NACK).
    NackReceived,
    /// I2C: another master won bus arbitration.
    ArbitrationLost,
    /// UART: stop bit not detected at expected position.
    FramingError,
    /// UART: received parity bit does not match calculated parity.
    ParityError,
    /// UART/I2S: receive FIFO overflow, data lost.
    Overrun,
}

impl fmt::Display for HalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HalError::InitFailed => write!(f, "initialization failed"),
            HalError::TxFifoFull => write!(f, "transmit FIFO full"),
            HalError::RxEmpty => write!(f, "receive FIFO empty"),
            HalError::BusOff => write!(f, "bus off"),
            HalError::InvalidConfig => write!(f, "invalid configuration"),
            HalError::Timeout => write!(f, "timeout"),
            HalError::DmaError => write!(f, "DMA error"),
            HalError::MmioError => write!(f, "MMIO error"),
            HalError::NotSupported => write!(f, "not supported"),
            HalError::InterruptError => write!(f, "interrupt error"),
            HalError::OutOfRange => write!(f, "address out of range"),
            HalError::HardwareError => write!(f, "hardware error"),
            HalError::NackReceived => write!(f, "NACK received"),
            HalError::ArbitrationLost => write!(f, "arbitration lost"),
            HalError::FramingError => write!(f, "framing error"),
            HalError::ParityError => write!(f, "parity error"),
            HalError::Overrun => write!(f, "receive overrun"),
        }
    }
}

// ---------------------------------------------------------------------------
// Bus Peripheral HAL — Section 27.1
// ---------------------------------------------------------------------------

/// Bus protocol type identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusProtocol {
    /// CAN 2.0A/B or CAN FD.
    Can,
    /// ARINC 429.
    Arinc429,
    /// ARINC 664 (AFDX).
    Arinc664,
    /// MIL-STD-1553B.
    Mil1553,
    /// SpaceWire (ECSS-E-ST-50-12C).
    SpaceWire,
    /// CCSDS Space Packet Protocol.
    Ccsds,
}

/// CAN baud rate configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanBaudRate {
    /// 125 Kbps.
    Kbps125,
    /// 250 Kbps.
    Kbps250,
    /// 500 Kbps.
    Kbps500,
    /// 1 Mbps.
    Mbps1,
}

/// CAN FD data-phase baud rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanFdDataRate {
    /// 2 Mbps.
    Mbps2,
    /// 4 Mbps.
    Mbps4,
    /// 8 Mbps.
    Mbps8,
}

/// ARINC 429 channel speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arinc429Speed {
    /// Low speed: 12.5 Kbps.
    Low,
    /// High speed: 100 Kbps.
    High,
}

/// MIL-STD-1553 operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mil1553Mode {
    /// Bus Controller mode.
    BusController,
    /// Remote Terminal mode.
    RemoteTerminal,
    /// Bus Monitor (passive listen).
    BusMonitor,
}

/// Bus controller configuration for initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusConfig {
    /// CAN configuration with baud rate and optional FD data rate.
    Can {
        baud_rate: CanBaudRate,
        fd_data_rate: Option<CanFdDataRate>,
    },
    /// ARINC 429 configuration with channel speed.
    Arinc429 { speed: Arinc429Speed },
    /// ARINC 664 Virtual Link configuration.
    Arinc664 { max_vl_count: u16 },
    /// MIL-STD-1553 configuration.
    Mil1553 { mode: Mil1553Mode, rt_address: u8 },
    /// SpaceWire link configuration.
    SpaceWire { link_speed_mbps: u16 },
    /// CCSDS configuration.
    Ccsds { max_packet_len: u16 },
}

/// A raw frame received from or to be sent to a bus controller.
#[derive(Debug, Clone)]
pub struct BusFrame {
    /// Protocol-specific address/ID (CAN ID, ARINC label, RT/SA, etc.).
    pub address: u32,
    /// Frame payload data.
    pub data: [u8; 64],
    /// Number of valid bytes in `data`.
    pub data_len: u8,
    /// Timestamp in microseconds (monotonic).
    pub timestamp_us: u64,
}

impl Default for BusFrame {
    fn default() -> Self {
        Self {
            address: 0,
            data: [0u8; 64],
            data_len: 0,
            timestamp_us: 0,
        }
    }
}

/// Bus controller error status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusStatus {
    /// Controller is operating normally.
    Ok,
    /// Warning threshold reached (e.g., high error count).
    Warning,
    /// Controller entered error passive state.
    ErrorPassive,
    /// Controller is in bus-off state.
    BusOff,
    /// Controller is not initialized.
    NotInitialized,
}

/// Interrupt source identifier for bus controllers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BusIrqSource {
    /// Bus controller index.
    pub controller: u8,
    /// Interrupt type (0 = RX, 1 = TX complete, 2 = error, 3 = bus off).
    pub irq_type: u8,
}

/// Hardware abstraction trait for bus peripheral controllers.
///
/// Each bus protocol implementation provides a concrete type implementing
/// this trait. The trait covers controller lifecycle, frame I/O, interrupt
/// handling, and error reporting.
pub trait BusController {
    /// Initialize the bus controller with the given configuration.
    ///
    /// Returns `Ok(())` when the controller enters operational state.
    /// Returns `Err(HalError::InitFailed)` if the controller does not
    /// respond within the timeout.
    fn init(&mut self, config: BusConfig) -> Result<(), HalError>;

    /// Submit a frame for transmission.
    ///
    /// Returns `Ok(())` when the frame is accepted into the transmit FIFO.
    /// Returns `Err(HalError::TxFifoFull)` if no transmit slots are available.
    fn tx_frame(&mut self, frame: &BusFrame) -> Result<(), HalError>;

    /// Retrieve a received frame from the receive FIFO.
    ///
    /// Returns `Ok(Some(frame))` if a frame is available, `Ok(None)` if
    /// the receive FIFO is empty.
    fn rx_frame(&mut self) -> Result<Option<BusFrame>, HalError>;

    /// Handle an interrupt from this bus controller.
    ///
    /// The top-half handler acknowledges the interrupt and returns the
    /// interrupt source for bottom-half processing.
    fn irq_handler(&mut self) -> Result<BusIrqSource, HalError>;

    /// Query the current error status of the controller.
    fn error_status(&self) -> BusStatus;

    /// Return the bus protocol type.
    fn protocol(&self) -> BusProtocol;

    /// Reset the bus controller to a known-good state.
    fn reset(&mut self) -> Result<(), HalError>;
}

// ---------------------------------------------------------------------------
// FPGA Fabric HAL — Section 27.2
// ---------------------------------------------------------------------------

/// DMA transfer direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmaDirection {
    /// Transfer from FPGA/device to system memory.
    FpgaToMemory,
    /// Transfer from system memory to FPGA/device.
    MemoryToFpga,
}

/// Token representing an in-flight DMA transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DmaToken {
    /// DMA channel number.
    pub channel: u8,
    /// Unique transfer ID within the channel.
    pub transfer_id: u32,
    /// Whether the transfer has completed.
    pub complete: bool,
    /// Whether the transfer encountered an error.
    pub error: bool,
}

/// DMA transfer descriptor for scatter-gather operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DmaDescriptor {
    /// Source physical address.
    pub src_addr: u64,
    /// Destination physical address.
    pub dst_addr: u64,
    /// Transfer length in bytes.
    pub length: u32,
    /// Transfer direction.
    pub direction: DmaDirection,
}

/// Hardware abstraction trait for FPGA fabric bus access.
///
/// Provides AXI/AXI-Lite memory-mapped register operations and AXI DMA
/// transfer management. All register accesses use volatile reads/writes
/// with appropriate memory barriers.
pub trait FpgaFabric {
    /// Read a 32-bit register from an FPGA soft-IP peripheral.
    ///
    /// Performs a volatile MMIO read at `base_addr + offset` and issues
    /// a read memory barrier after the access.
    fn read_reg(&self, base_addr: u64, offset: u32) -> Result<u32, HalError>;

    /// Write a 32-bit register to an FPGA soft-IP peripheral.
    ///
    /// Issues a write memory barrier before performing a volatile MMIO
    /// write at `base_addr + offset`.
    fn write_reg(&mut self, base_addr: u64, offset: u32, value: u32) -> Result<(), HalError>;

    /// Initiate an AXI DMA transfer.
    ///
    /// Programs the DMA controller with the given descriptor and starts
    /// the transfer. Returns a `DmaToken` for polling or awaiting completion.
    fn dma_start(&mut self, desc: DmaDescriptor) -> Result<DmaToken, HalError>;

    /// Poll a DMA transfer for completion.
    ///
    /// Returns the updated `DmaToken` with `complete` set to true when
    /// the transfer finishes, or `error` set to true on failure.
    fn dma_poll(&self, token: DmaToken) -> Result<DmaToken, HalError>;

    /// Acknowledge a DMA completion interrupt and return the channel number.
    fn dma_irq_ack(&mut self) -> Result<u8, HalError>;
}

// ---------------------------------------------------------------------------
// Peripheral HAL — I2C, SPI, GPIO, UART, CSI Camera, I2S Audio
// ---------------------------------------------------------------------------

/// I2C bus speed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum I2cSpeed {
    /// Standard mode: 100 kHz.
    Standard,
    /// Fast mode: 400 kHz.
    Fast,
    /// Fast mode plus: 1 MHz.
    FastPlus,
}

/// I2C addressing mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum I2cAddressMode {
    /// 7-bit addressing (0x00–0x7F).
    SevenBit,
    /// 10-bit addressing (0x000–0x3FF).
    TenBit,
}

/// I2C controller configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct I2cConfig {
    /// Bus speed selection.
    pub speed: I2cSpeed,
    /// Addressing mode.
    pub address_mode: I2cAddressMode,
    /// Enable clock stretching by targets.
    pub clock_stretching: bool,
    /// Timeout for a single transaction in microseconds.
    pub timeout_us: u32,
}

/// Hardware abstraction trait for I2C controllers.
pub trait I2cController {
    /// Initialize the I2C controller with the given configuration.
    fn init(&mut self, config: I2cConfig) -> Result<(), HalError>;

    /// Write bytes to a target device.
    fn write(&mut self, addr: u16, data: &[u8]) -> Result<(), HalError>;

    /// Read bytes from a target device.
    fn read(&mut self, addr: u16, buf: &mut [u8]) -> Result<(), HalError>;

    /// Write then read in a single transaction (repeated START).
    fn write_read(
        &mut self,
        addr: u16,
        write_data: &[u8],
        read_buf: &mut [u8],
    ) -> Result<(), HalError>;

    /// Attempt I2C bus recovery (9 clock pulses + STOP).
    fn bus_recovery(&mut self) -> Result<(), HalError>;

    /// Reset the I2C controller.
    fn reset(&mut self) -> Result<(), HalError>;
}

/// SPI clock polarity and phase mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpiMode {
    /// CPOL=0, CPHA=0: clock idle low, sample on leading edge.
    Mode0,
    /// CPOL=0, CPHA=1: clock idle low, sample on trailing edge.
    Mode1,
    /// CPOL=1, CPHA=0: clock idle high, sample on leading edge.
    Mode2,
    /// CPOL=1, CPHA=1: clock idle high, sample on trailing edge.
    Mode3,
}

/// SPI bit order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpiBitOrder {
    /// Most significant bit first (default).
    MsbFirst,
    /// Least significant bit first.
    LsbFirst,
}

/// SPI controller configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpiConfig {
    /// Clock polarity/phase mode.
    pub mode: SpiMode,
    /// Requested clock frequency in Hz.
    pub clock_hz: u32,
    /// Bit order.
    pub bit_order: SpiBitOrder,
    /// Bits per word: 8, 16, or 32.
    pub word_size: u8,
    /// Chip select polarity (true = active low, default).
    pub cs_active_low: bool,
    /// Transaction timeout in microseconds.
    pub timeout_us: u32,
}

/// Hardware abstraction trait for SPI controllers.
pub trait SpiController {
    /// Initialize the SPI controller.
    fn init(&mut self, config: SpiConfig) -> Result<(), HalError>;

    /// Full-duplex transfer: simultaneously write `tx_data` and read into `rx_buf`.
    fn transfer(&mut self, cs: u8, tx_data: &[u8], rx_buf: &mut [u8]) -> Result<(), HalError>;

    /// Write-only transfer (MOSI only, MISO data discarded).
    fn write(&mut self, cs: u8, data: &[u8]) -> Result<(), HalError>;

    /// Read-only transfer (sends zeros on MOSI, captures MISO).
    fn read(&mut self, cs: u8, buf: &mut [u8]) -> Result<(), HalError>;

    /// Start a DMA-based transfer for bulk data.
    fn transfer_dma(
        &mut self,
        cs: u8,
        tx_addr: u64,
        rx_addr: u64,
        len: u32,
    ) -> Result<DmaToken, HalError>;

    /// Manually assert chip select.
    fn cs_assert(&mut self, cs: u8) -> Result<(), HalError>;

    /// Manually deassert chip select.
    fn cs_deassert(&mut self, cs: u8) -> Result<(), HalError>;

    /// Reset the SPI controller.
    fn reset(&mut self) -> Result<(), HalError>;
}

/// GPIO pin mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpioPinMode {
    /// High-impedance input.
    Input,
    /// Push-pull output.
    Output,
    /// Open-drain output.
    OpenDrain,
    /// Alternate function (peripheral-controlled).
    AlternateFunction(u8),
    /// Analog (disable digital input buffer).
    Analog,
}

/// GPIO pull resistor configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpioPull {
    /// No pull resistor (floating).
    None,
    /// Internal pull-up resistor.
    PullUp,
    /// Internal pull-down resistor.
    PullDown,
}

/// GPIO interrupt edge configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpioInterruptEdge {
    /// Trigger on rising edge.
    Rising,
    /// Trigger on falling edge.
    Falling,
    /// Trigger on both edges.
    Both,
    /// Trigger while level is high.
    LevelHigh,
    /// Trigger while level is low.
    LevelLow,
}

/// GPIO pin configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpioConfig {
    /// Pin mode.
    pub mode: GpioPinMode,
    /// Pull resistor.
    pub pull: GpioPull,
    /// Interrupt edge (None = no interrupt).
    pub interrupt: Option<GpioInterruptEdge>,
    /// Debounce time in microseconds (0 = no debounce).
    pub debounce_us: u32,
}

/// Hardware abstraction trait for GPIO controllers.
pub trait GpioController {
    /// Configure a single GPIO pin.
    fn configure(&mut self, pin: u8, config: GpioConfig) -> Result<(), HalError>;

    /// Read the current level of an input pin (true = high).
    fn read(&self, pin: u8) -> Result<bool, HalError>;

    /// Set an output pin high.
    fn set_high(&mut self, pin: u8) -> Result<(), HalError>;

    /// Set an output pin low.
    fn set_low(&mut self, pin: u8) -> Result<(), HalError>;

    /// Toggle an output pin.
    fn toggle(&mut self, pin: u8) -> Result<(), HalError>;

    /// Atomically set multiple pins using a bitmask.
    fn set_mask(&mut self, set_mask: u32, clear_mask: u32) -> Result<(), HalError>;

    /// Read all pins as a bitmask (up to 32 pins per controller).
    fn read_all(&self) -> Result<u32, HalError>;

    /// Enable interrupt for a pin.
    fn enable_interrupt(&mut self, pin: u8) -> Result<(), HalError>;

    /// Disable interrupt for a pin.
    fn disable_interrupt(&mut self, pin: u8) -> Result<(), HalError>;

    /// Acknowledge and clear a pending interrupt. Returns the triggering pin.
    fn irq_handler(&mut self) -> Result<u8, HalError>;

    /// Return the number of GPIO pins on this controller.
    fn pin_count(&self) -> u8;

    /// Reset all pins to input/floating.
    fn reset(&mut self) -> Result<(), HalError>;
}

/// UART parity configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UartParity {
    /// No parity bit.
    None,
    /// Even parity.
    Even,
    /// Odd parity.
    Odd,
}

/// UART stop bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UartStopBits {
    /// 1 stop bit.
    One,
    /// 2 stop bits.
    Two,
}

/// UART flow control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UartFlowControl {
    /// No flow control.
    None,
    /// Hardware flow control using RTS/CTS lines.
    RtsCts,
}

/// UART receive mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UartRxMode {
    /// Interrupt-driven with FIFO threshold.
    Interrupt { fifo_threshold: u8 },
    /// DMA ring-buffer mode.
    Dma { buffer_addr: u64, buffer_size: u32 },
    /// Polling mode.
    Polling,
}

/// UART controller configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UartConfig {
    /// Baud rate in bits per second.
    pub baud_rate: u32,
    /// Data bits: 7 or 8.
    pub data_bits: u8,
    /// Parity configuration.
    pub parity: UartParity,
    /// Stop bits.
    pub stop_bits: UartStopBits,
    /// Flow control.
    pub flow_control: UartFlowControl,
    /// Receive mode.
    pub rx_mode: UartRxMode,
    /// Read timeout in microseconds (0 = non-blocking).
    pub timeout_us: u32,
}

/// UART interrupt source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UartIrqSource {
    /// RX FIFO reached threshold.
    RxDataAvailable,
    /// RX FIFO timeout (partial data sitting in FIFO).
    RxTimeout,
    /// TX FIFO empty.
    TxEmpty,
    /// Framing error detected.
    FramingError,
    /// Parity error detected.
    ParityError,
    /// RX FIFO overrun.
    Overrun,
    /// Break condition detected.
    Break,
}

/// Hardware abstraction trait for UART controllers.
pub trait UartController {
    /// Initialize the UART controller.
    fn init(&mut self, config: UartConfig) -> Result<(), HalError>;

    /// Write bytes to the TX FIFO. Returns number of bytes written.
    fn write(&mut self, data: &[u8]) -> Result<usize, HalError>;

    /// Write all bytes, blocking until TX FIFO drains as needed.
    fn write_all(&mut self, data: &[u8]) -> Result<(), HalError>;

    /// Read available bytes from RX FIFO. Returns number of bytes read.
    fn read(&mut self, buf: &mut [u8]) -> Result<usize, HalError>;

    /// Read until `buf` is full or timeout expires.
    fn read_exact(&mut self, buf: &mut [u8]) -> Result<usize, HalError>;

    /// Read until delimiter byte or buf is full. Returns bytes read including delimiter.
    fn read_until(&mut self, delimiter: u8, buf: &mut [u8]) -> Result<usize, HalError>;

    /// Flush the TX FIFO (block until all bytes transmitted).
    fn flush(&mut self) -> Result<(), HalError>;

    /// Return the number of bytes available in RX FIFO.
    fn rx_available(&self) -> usize;

    /// Handle a UART interrupt.
    fn irq_handler(&mut self) -> Result<UartIrqSource, HalError>;

    /// Reset the UART controller.
    fn reset(&mut self) -> Result<(), HalError>;
}

/// CSI-2 pixel format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsiPixelFormat {
    /// 8-bit raw Bayer.
    Raw8,
    /// 10-bit raw Bayer.
    Raw10,
    /// 12-bit raw Bayer.
    Raw12,
    /// YUV 4:2:2.
    Yuv422,
    /// 24-bit RGB.
    Rgb888,
}

/// CSI-2 receiver configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsiConfig {
    /// Number of data lanes: 1, 2, or 4.
    pub lanes: u8,
    /// Per-lane data rate in Mbps.
    pub lane_speed_mbps: u16,
    /// Frame width in pixels.
    pub width: u16,
    /// Frame height in pixels.
    pub height: u16,
    /// Pixel format.
    pub pixel_format: CsiPixelFormat,
    /// Target frame rate.
    pub fps: u8,
    /// Number of frame buffers (double/triple buffering).
    pub num_buffers: u8,
}

/// A completed CSI frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsiFrame {
    /// Buffer index of the completed frame.
    pub buffer_index: u8,
    /// Timestamp of frame capture (monotonic microseconds).
    pub timestamp_us: u64,
    /// Frame sequence number.
    pub sequence: u32,
    /// Actual bytes written.
    pub bytes_used: u32,
}

/// CSI interrupt source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsiIrqSource {
    /// Frame capture complete (buffer index).
    FrameComplete(u8),
    /// Receiver FIFO overflow.
    Overflow,
    /// CRC mismatch.
    CrcError,
    /// ECC error in packet header.
    EccError,
}

/// Hardware abstraction trait for CSI-2 camera receivers.
pub trait CsiReceiver {
    /// Initialize the CSI-2 receiver and allocate frame buffers.
    fn init(&mut self, config: CsiConfig) -> Result<(), HalError>;

    /// Start continuous frame capture.
    fn start_capture(&mut self) -> Result<(), HalError>;

    /// Stop frame capture.
    fn stop_capture(&mut self) -> Result<(), HalError>;

    /// Dequeue a completed frame buffer.
    fn dequeue_frame(&mut self) -> Result<CsiFrame, HalError>;

    /// Return a frame buffer to the capture queue.
    fn enqueue_frame(&mut self, buffer_index: u8) -> Result<(), HalError>;

    /// Get the physical address of a frame buffer (for zero-copy ONNX input).
    fn buffer_address(&self, buffer_index: u8) -> Result<u64, HalError>;

    /// Get the byte size of a single frame buffer.
    fn buffer_size(&self) -> usize;

    /// Handle a CSI interrupt.
    fn irq_handler(&mut self) -> Result<CsiIrqSource, HalError>;

    /// Reset the CSI receiver and release all buffers.
    fn reset(&mut self) -> Result<(), HalError>;
}

/// I2S bus mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum I2sMode {
    /// Standard I2S (Philips) — data delayed by 1 BCLK after WS transition.
    Standard,
    /// Left-justified — data starts immediately on WS transition.
    LeftJustified,
    /// Right-justified — data is right-aligned within the WS slot.
    RightJustified,
    /// TDM — multiple channels time-multiplexed on a single data line.
    Tdm { num_slots: u8 },
}

/// I2S controller role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum I2sRole {
    /// Controller generates BCLK and WS.
    Master,
    /// Controller receives BCLK and WS from external source.
    Slave,
}

/// I2S bit depth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum I2sBitDepth {
    /// 16-bit samples.
    Bits16,
    /// 24-bit samples.
    Bits24,
    /// 32-bit samples.
    Bits32,
}

/// I2S controller configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct I2sConfig {
    /// I2S bus mode.
    pub mode: I2sMode,
    /// Controller role (master/slave).
    pub role: I2sRole,
    /// Sample rate in Hz.
    pub sample_rate_hz: u32,
    /// Bit depth per sample.
    pub bit_depth: I2sBitDepth,
    /// Number of channels (1=mono, 2=stereo, up to 8 for TDM).
    pub channels: u8,
    /// Number of audio frames per DMA buffer.
    pub buffer_frames: u32,
    /// Number of ring buffers (2-4).
    pub num_buffers: u8,
}

/// A completed audio buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioBuffer {
    /// Buffer index.
    pub buffer_index: u8,
    /// Timestamp of first sample (monotonic microseconds).
    pub timestamp_us: u64,
    /// Number of samples (frames x channels).
    pub sample_count: u32,
    /// Actual bytes in buffer.
    pub bytes_used: u32,
}

/// I2S interrupt source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum I2sIrqSource {
    /// Buffer capture complete (buffer index).
    BufferComplete(u8),
    /// DMA overrun (data lost).
    Overrun,
    /// DMA underrun.
    Underrun,
    /// BCLK/WS clock error.
    ClockError,
}

/// Hardware abstraction trait for I2S/TDM audio controllers.
pub trait I2sController {
    /// Initialize the I2S controller and allocate DMA ring buffers.
    fn init(&mut self, config: I2sConfig) -> Result<(), HalError>;

    /// Start audio capture (continuous DMA streaming).
    fn start_capture(&mut self) -> Result<(), HalError>;

    /// Stop audio capture.
    fn stop_capture(&mut self) -> Result<(), HalError>;

    /// Dequeue a completed audio buffer.
    fn dequeue_buffer(&mut self) -> Result<AudioBuffer, HalError>;

    /// Return a buffer to the capture ring.
    fn enqueue_buffer(&mut self, buffer_index: u8) -> Result<(), HalError>;

    /// Get the physical address of an audio buffer (for zero-copy ONNX input).
    fn buffer_address(&self, buffer_index: u8) -> Result<u64, HalError>;

    /// Get the byte size of a single audio buffer.
    fn buffer_size(&self) -> usize;

    /// Handle an I2S interrupt.
    fn irq_handler(&mut self) -> Result<I2sIrqSource, HalError>;

    /// Reset the I2S controller and release all buffers.
    fn reset(&mut self) -> Result<(), HalError>;
}

// ---------------------------------------------------------------------------
// RISC-V HAL stubs — Section 27.3
// ---------------------------------------------------------------------------

pub mod riscv {
    //! RISC-V HAL stub types and functions.
    //!
    //! Provides data structures for SV48 paging, PLIC interrupt controller,
    //! CLINT timer, and SBI wrappers. These are platform-independent type
    //! definitions; actual hardware access requires the `smallaios-arch-riscv64`
    //! crate.

    use super::HalError;

    /// SBI call return value.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct SbiResult {
        /// Error code (0 = success).
        pub error: i64,
        /// Return value.
        pub value: i64,
    }

    impl SbiResult {
        /// Returns true if the SBI call succeeded.
        pub fn is_success(&self) -> bool {
            self.error == 0
        }

        /// Create a success result.
        pub fn success(value: i64) -> Self {
            Self { error: 0, value }
        }

        /// Create an error result.
        pub fn error(code: i64) -> Self {
            Self {
                error: code,
                value: 0,
            }
        }
    }

    /// SBI extension IDs.
    pub mod sbi_ext {
        /// Base extension.
        pub const BASE: u64 = 0x10;
        /// Timer extension.
        pub const TIMER: u64 = 0x54494D45;
        /// IPI extension.
        pub const IPI: u64 = 0x735049;
        /// RFENCE extension.
        pub const RFENCE: u64 = 0x52464E43;
        /// Hart State Management.
        pub const HSM: u64 = 0x48534D;
        /// Debug Console.
        pub const DBCN: u64 = 0x4442434E;
    }

    /// Hart state as reported by SBI HSM extension.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum HartState {
        /// Hart is currently executing.
        Started,
        /// Hart is not executing and waiting for start.
        Stopped,
        /// Hart start request is pending.
        StartPending,
        /// Hart stop request is pending.
        StopPending,
        /// Hart is suspended.
        Suspended,
        /// Hart suspend request is pending.
        SuspendPending,
        /// Hart resume request is pending.
        ResumePending,
    }

    impl HartState {
        /// Parse a hart state from the SBI return value.
        pub fn from_value(v: i64) -> Option<Self> {
            match v {
                0 => Some(HartState::Started),
                1 => Some(HartState::Stopped),
                2 => Some(HartState::StartPending),
                3 => Some(HartState::StopPending),
                4 => Some(HartState::Suspended),
                5 => Some(HartState::SuspendPending),
                6 => Some(HartState::ResumePending),
                _ => None,
            }
        }
    }

    // -- SV48 Paging -------------------------------------------------------

    /// Page table entry permission flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PteFlags(pub u64);

    impl PteFlags {
        /// Valid bit.
        pub const V: u64 = 1 << 0;
        /// Read permission.
        pub const R: u64 = 1 << 1;
        /// Write permission.
        pub const W: u64 = 1 << 2;
        /// Execute permission.
        pub const X: u64 = 1 << 3;
        /// User-mode accessible.
        pub const U: u64 = 1 << 4;
        /// Global mapping.
        pub const G: u64 = 1 << 5;
        /// Accessed (set by hardware).
        pub const A: u64 = 1 << 6;
        /// Dirty (set by hardware on write).
        pub const D: u64 = 1 << 7;

        /// Create flags from raw value.
        pub fn new(flags: u64) -> Self {
            Self(flags)
        }

        /// Create read-only flags.
        pub fn read_only() -> Self {
            Self(Self::V | Self::R | Self::A)
        }

        /// Create read-write flags.
        pub fn read_write() -> Self {
            Self(Self::V | Self::R | Self::W | Self::A | Self::D)
        }

        /// Create read-execute flags.
        pub fn read_execute() -> Self {
            Self(Self::V | Self::R | Self::X | Self::A)
        }

        /// Create read-write-execute flags.
        pub fn read_write_execute() -> Self {
            Self(Self::V | Self::R | Self::W | Self::X | Self::A | Self::D)
        }

        /// Check if a specific flag is set.
        pub fn has(&self, flag: u64) -> bool {
            self.0 & flag != 0
        }
    }

    /// SV48 page table entry.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Sv48Pte(pub u64);

    impl Sv48Pte {
        /// Number of levels in the SV48 page table.
        pub const LEVELS: usize = 4;
        /// Number of entries per page table page.
        pub const ENTRIES_PER_PAGE: usize = 512;
        /// Page size in bytes (4 KiB).
        pub const PAGE_SIZE: usize = 4096;
        /// Mega-page size (2 MiB).
        pub const MEGA_PAGE_SIZE: usize = 2 * 1024 * 1024;
        /// Giga-page size (1 GiB).
        pub const GIGA_PAGE_SIZE: usize = 1024 * 1024 * 1024;

        /// Physical page number shift.
        const PPN_SHIFT: u64 = 10;
        /// Physical page number mask (44 bits).
        const PPN_MASK: u64 = (1 << 44) - 1;

        /// Create a new PTE from a physical page number and flags.
        pub fn new(ppn: u64, flags: PteFlags) -> Self {
            Self((ppn << Self::PPN_SHIFT) | flags.0)
        }

        /// Create an invalid (zero) PTE.
        pub fn invalid() -> Self {
            Self(0)
        }

        /// Check if this PTE is valid.
        pub fn is_valid(&self) -> bool {
            self.0 & PteFlags::V != 0
        }

        /// Check if this PTE is a leaf (has R, W, or X set).
        pub fn is_leaf(&self) -> bool {
            self.0 & (PteFlags::R | PteFlags::W | PteFlags::X) != 0
        }

        /// Extract the physical page number.
        pub fn ppn(&self) -> u64 {
            (self.0 >> Self::PPN_SHIFT) & Self::PPN_MASK
        }

        /// Extract the flags.
        pub fn flags(&self) -> PteFlags {
            PteFlags(self.0 & 0x3FF)
        }

        /// Convert PPN to a physical address.
        pub fn physical_address(&self) -> u64 {
            self.ppn() << 12
        }
    }

    /// SV48 page table manager (stub implementation for host-side testing).
    ///
    /// In a real RISC-V kernel, this would manage actual page table memory
    /// and write the satp CSR. This stub maintains a simple mapping table
    /// for testing the interface.
    pub struct Sv48PageTable {
        /// Mapping entries: (virtual_addr, physical_addr, flags).
        mappings: [(u64, u64, PteFlags); 256],
        /// Number of active mappings.
        count: usize,
    }

    impl Default for Sv48PageTable {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Sv48PageTable {
        /// Maximum number of mappings in the stub.
        const MAX_MAPPINGS: usize = 256;

        /// Create a new empty page table.
        pub fn new() -> Self {
            Self {
                mappings: [(0, 0, PteFlags(0)); 256],
                count: 0,
            }
        }

        /// Map a virtual page to a physical page with the given permissions.
        pub fn map(&mut self, vaddr: u64, paddr: u64, flags: PteFlags) -> Result<(), HalError> {
            // Check for existing mapping at this virtual address.
            for i in 0..self.count {
                if self.mappings[i].0 == vaddr {
                    // Update existing mapping.
                    self.mappings[i] = (vaddr, paddr, flags);
                    return Ok(());
                }
            }
            if self.count >= Self::MAX_MAPPINGS {
                return Err(HalError::OutOfRange);
            }
            self.mappings[self.count] = (vaddr, paddr, flags);
            self.count += 1;
            Ok(())
        }

        /// Unmap a virtual page. In a real implementation this would issue sfence.vma.
        pub fn unmap(&mut self, vaddr: u64) -> Result<(), HalError> {
            for i in 0..self.count {
                if self.mappings[i].0 == vaddr {
                    // Shift remaining entries.
                    for j in i..self.count - 1 {
                        self.mappings[j] = self.mappings[j + 1];
                    }
                    self.count -= 1;
                    return Ok(());
                }
            }
            Err(HalError::NotSupported)
        }

        /// Change permissions on an existing mapping without changing the physical address.
        pub fn protect(&mut self, vaddr: u64, flags: PteFlags) -> Result<(), HalError> {
            for i in 0..self.count {
                if self.mappings[i].0 == vaddr {
                    self.mappings[i].2 = flags;
                    return Ok(());
                }
            }
            Err(HalError::NotSupported)
        }

        /// Look up the physical address for a virtual address.
        pub fn translate(&self, vaddr: u64) -> Option<(u64, PteFlags)> {
            for i in 0..self.count {
                if self.mappings[i].0 == vaddr {
                    return Some((self.mappings[i].1, self.mappings[i].2));
                }
            }
            None
        }

        /// Return the number of active mappings.
        pub fn mapping_count(&self) -> usize {
            self.count
        }
    }

    // -- PLIC ---------------------------------------------------------------

    /// PLIC interrupt priority (0 = disabled, 1-7 = ascending priority).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PlicPriority(pub u8);

    impl PlicPriority {
        /// Disabled (priority 0).
        pub const DISABLED: Self = Self(0);
        /// Maximum priority.
        pub const MAX: Self = Self(7);

        /// Check if priority is valid (0-7).
        pub fn is_valid(&self) -> bool {
            self.0 <= 7
        }
    }

    /// PLIC context identifier (hart + privilege mode).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PlicContext(pub u32);

    /// Stub PLIC driver.
    pub struct Plic {
        /// Per-source priorities (up to 1024 sources).
        priorities: [u8; 1024],
        /// Per-context enable bits (simplified: 1 context, 1024 sources).
        enabled: [bool; 1024],
        /// Per-context threshold.
        threshold: u8,
        /// Currently claimed source (0 = none).
        claimed: u32,
        /// Pending interrupt sources.
        pending: [bool; 1024],
    }

    impl Default for Plic {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Plic {
        /// Create a new PLIC with all sources disabled.
        pub fn new() -> Self {
            Self {
                priorities: [0; 1024],
                enabled: [false; 1024],
                threshold: 0,
                claimed: 0,
                pending: [false; 1024],
            }
        }

        /// Set the priority for an interrupt source.
        pub fn set_priority(
            &mut self,
            source: u32,
            priority: PlicPriority,
        ) -> Result<(), HalError> {
            if source == 0 || source >= 1024 || !priority.is_valid() {
                return Err(HalError::InvalidConfig);
            }
            self.priorities[source as usize] = priority.0;
            Ok(())
        }

        /// Enable an interrupt source for the given context.
        pub fn enable(&mut self, _context: PlicContext, source: u32) -> Result<(), HalError> {
            if source == 0 || source >= 1024 {
                return Err(HalError::InvalidConfig);
            }
            self.enabled[source as usize] = true;
            Ok(())
        }

        /// Disable an interrupt source.
        pub fn disable(&mut self, _context: PlicContext, source: u32) -> Result<(), HalError> {
            if source == 0 || source >= 1024 {
                return Err(HalError::InvalidConfig);
            }
            self.enabled[source as usize] = false;
            Ok(())
        }

        /// Set the priority threshold for a context.
        pub fn set_threshold(
            &mut self,
            _context: PlicContext,
            threshold: u8,
        ) -> Result<(), HalError> {
            if threshold > 7 {
                return Err(HalError::InvalidConfig);
            }
            self.threshold = threshold;
            Ok(())
        }

        /// Claim the highest-priority pending interrupt.
        pub fn claim(&mut self, _context: PlicContext) -> u32 {
            let mut best_source = 0u32;
            let mut best_priority = 0u8;
            for i in 1..1024u32 {
                let idx = i as usize;
                if self.pending[idx]
                    && self.enabled[idx]
                    && self.priorities[idx] > self.threshold
                    && self.priorities[idx] > best_priority
                {
                    best_source = i;
                    best_priority = self.priorities[idx];
                }
            }
            if best_source != 0 {
                self.pending[best_source as usize] = false;
                self.claimed = best_source;
            }
            best_source
        }

        /// Complete interrupt handling for the claimed source.
        pub fn complete(&mut self, _context: PlicContext, source: u32) {
            if self.claimed == source {
                self.claimed = 0;
            }
        }

        /// Set a source as pending (simulates hardware interrupt arrival).
        pub fn set_pending(&mut self, source: u32) {
            if source > 0 && source < 1024 {
                self.pending[source as usize] = true;
            }
        }
    }

    // -- CLINT --------------------------------------------------------------

    /// Stub CLINT timer driver.
    pub struct Clint {
        /// Current mtime value (shared across harts).
        mtime: u64,
        /// Per-hart mtimecmp values (up to 8 harts).
        mtimecmp: [u64; 8],
        /// Timer frequency in Hz.
        frequency: u64,
    }

    impl Default for Clint {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Clint {
        /// Default timer frequency: 10 MHz (QEMU virt default).
        const DEFAULT_FREQ: u64 = 10_000_000;

        /// Create a new CLINT with default frequency.
        pub fn new() -> Self {
            Self {
                mtime: 0,
                mtimecmp: [u64::MAX; 8],
                frequency: Self::DEFAULT_FREQ,
            }
        }

        /// Read the current mtime register.
        pub fn read_mtime(&self) -> u64 {
            self.mtime
        }

        /// Set the mtime value (for testing; in hardware this is read-only from S-mode).
        pub fn set_mtime(&mut self, value: u64) {
            self.mtime = value;
        }

        /// Set the mtimecmp for a hart to schedule a timer interrupt.
        pub fn set_timer(&mut self, hart_id: usize, deadline: u64) -> Result<(), HalError> {
            if hart_id >= 8 {
                return Err(HalError::OutOfRange);
            }
            self.mtimecmp[hart_id] = deadline;
            Ok(())
        }

        /// Check if the timer interrupt is pending for a hart.
        pub fn is_timer_pending(&self, hart_id: usize) -> bool {
            hart_id < 8 && self.mtime >= self.mtimecmp[hart_id]
        }

        /// Program the next periodic tick.
        pub fn schedule_tick(&mut self, hart_id: usize, interval: u64) -> Result<(), HalError> {
            if hart_id >= 8 {
                return Err(HalError::OutOfRange);
            }
            self.mtimecmp[hart_id] = self.mtime.saturating_add(interval);
            Ok(())
        }

        /// Return the timer frequency in Hz.
        pub fn frequency(&self) -> u64 {
            self.frequency
        }

        /// Convert microseconds to timer ticks.
        pub fn us_to_ticks(&self, us: u64) -> u64 {
            us * self.frequency / 1_000_000
        }
    }

    // -- RISC-V Feature Detection ------------------------------------------

    /// Detected RISC-V ISA extensions.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct RiscvFeatures {
        /// Base integer ISA (always true for RV64I).
        pub has_i: bool,
        /// Multiply/Divide extension.
        pub has_m: bool,
        /// Atomic extension (required for SMP).
        pub has_a: bool,
        /// Single-precision floating point.
        pub has_f: bool,
        /// Double-precision floating point.
        pub has_d: bool,
        /// Compressed instructions.
        pub has_c: bool,
        /// XLEN (64 for RV64).
        pub xlen: u8,
    }

    impl RiscvFeatures {
        /// Create features for the standard RV64GC configuration.
        pub fn rv64gc() -> Self {
            Self {
                has_i: true,
                has_m: true,
                has_a: true,
                has_f: true,
                has_d: true,
                has_c: true,
                xlen: 64,
            }
        }

        /// Create minimal features (RV64I only).
        pub fn rv64i_only() -> Self {
            Self {
                has_i: true,
                has_m: false,
                has_a: false,
                has_f: false,
                has_d: false,
                has_c: false,
                xlen: 64,
            }
        }

        /// Check if the configuration meets the RV64GC requirement.
        pub fn is_gc(&self) -> bool {
            self.has_i && self.has_m && self.has_a && self.has_f && self.has_d && self.has_c
        }

        /// Parse features from a misa CSR value (RV64).
        pub fn from_misa(misa: u64) -> Self {
            Self {
                has_i: misa & (1 << 8) != 0,
                has_m: misa & (1 << 12) != 0,
                has_a: misa & (1 << 0) != 0,
                has_f: misa & (1 << 5) != 0,
                has_d: misa & (1 << 3) != 0,
                has_c: misa & (1 << 2) != 0,
                xlen: 64,
            }
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::riscv::*;
    use super::*;
    use alloc::format;

    // -- HalError -----------------------------------------------------------

    #[test]
    fn test_hal_error_display() {
        assert_eq!(format!("{}", HalError::InitFailed), "initialization failed");
        assert_eq!(format!("{}", HalError::TxFifoFull), "transmit FIFO full");
        assert_eq!(format!("{}", HalError::RxEmpty), "receive FIFO empty");
        assert_eq!(format!("{}", HalError::BusOff), "bus off");
        assert_eq!(
            format!("{}", HalError::InvalidConfig),
            "invalid configuration"
        );
        assert_eq!(format!("{}", HalError::Timeout), "timeout");
        assert_eq!(format!("{}", HalError::DmaError), "DMA error");
        assert_eq!(format!("{}", HalError::MmioError), "MMIO error");
        assert_eq!(format!("{}", HalError::NotSupported), "not supported");
        assert_eq!(format!("{}", HalError::InterruptError), "interrupt error");
        assert_eq!(format!("{}", HalError::OutOfRange), "address out of range");
        assert_eq!(format!("{}", HalError::HardwareError), "hardware error");
        assert_eq!(format!("{}", HalError::NackReceived), "NACK received");
        assert_eq!(format!("{}", HalError::ArbitrationLost), "arbitration lost");
        assert_eq!(format!("{}", HalError::FramingError), "framing error");
        assert_eq!(format!("{}", HalError::ParityError), "parity error");
        assert_eq!(format!("{}", HalError::Overrun), "receive overrun");
    }

    #[test]
    fn test_hal_error_eq() {
        assert_eq!(HalError::InitFailed, HalError::InitFailed);
        assert_ne!(HalError::InitFailed, HalError::Timeout);
    }

    // -- BusConfig ----------------------------------------------------------

    #[test]
    fn test_bus_config_can() {
        let cfg = BusConfig::Can {
            baud_rate: CanBaudRate::Mbps1,
            fd_data_rate: Some(CanFdDataRate::Mbps8),
        };
        match cfg {
            BusConfig::Can {
                baud_rate,
                fd_data_rate,
            } => {
                assert_eq!(baud_rate, CanBaudRate::Mbps1);
                assert_eq!(fd_data_rate, Some(CanFdDataRate::Mbps8));
            }
            _ => panic!("expected CAN config"),
        }
    }

    #[test]
    fn test_bus_config_arinc429() {
        let cfg = BusConfig::Arinc429 {
            speed: Arinc429Speed::High,
        };
        match cfg {
            BusConfig::Arinc429 { speed } => assert_eq!(speed, Arinc429Speed::High),
            _ => panic!("expected ARINC 429 config"),
        }
    }

    #[test]
    fn test_bus_config_mil1553() {
        let cfg = BusConfig::Mil1553 {
            mode: Mil1553Mode::BusController,
            rt_address: 5,
        };
        match cfg {
            BusConfig::Mil1553 { mode, rt_address } => {
                assert_eq!(mode, Mil1553Mode::BusController);
                assert_eq!(rt_address, 5);
            }
            _ => panic!("expected MIL-1553 config"),
        }
    }

    // -- BusFrame -----------------------------------------------------------

    #[test]
    fn test_bus_frame_default() {
        let frame = BusFrame::default();
        assert_eq!(frame.address, 0);
        assert_eq!(frame.data_len, 0);
        assert_eq!(frame.timestamp_us, 0);
    }

    // -- BusStatus ----------------------------------------------------------

    #[test]
    fn test_bus_status_variants() {
        assert_ne!(BusStatus::Ok, BusStatus::BusOff);
        assert_ne!(BusStatus::Warning, BusStatus::ErrorPassive);
        assert_eq!(BusStatus::NotInitialized, BusStatus::NotInitialized);
    }

    // -- SV48 PTE -----------------------------------------------------------

    #[test]
    fn test_pte_flags_read_only() {
        let flags = PteFlags::read_only();
        assert!(flags.has(PteFlags::V));
        assert!(flags.has(PteFlags::R));
        assert!(!flags.has(PteFlags::W));
        assert!(!flags.has(PteFlags::X));
    }

    #[test]
    fn test_pte_flags_read_write() {
        let flags = PteFlags::read_write();
        assert!(flags.has(PteFlags::V));
        assert!(flags.has(PteFlags::R));
        assert!(flags.has(PteFlags::W));
        assert!(!flags.has(PteFlags::X));
    }

    #[test]
    fn test_pte_flags_read_execute() {
        let flags = PteFlags::read_execute();
        assert!(flags.has(PteFlags::R));
        assert!(flags.has(PteFlags::X));
        assert!(!flags.has(PteFlags::W));
    }

    #[test]
    fn test_pte_flags_read_write_execute() {
        let flags = PteFlags::read_write_execute();
        assert!(flags.has(PteFlags::R));
        assert!(flags.has(PteFlags::W));
        assert!(flags.has(PteFlags::X));
    }

    #[test]
    fn test_sv48_pte_new() {
        let pte = Sv48Pte::new(0x12345, PteFlags::read_write());
        assert!(pte.is_valid());
        assert!(pte.is_leaf());
        assert_eq!(pte.ppn(), 0x12345);
        assert_eq!(pte.physical_address(), 0x12345 << 12);
    }

    #[test]
    fn test_sv48_pte_invalid() {
        let pte = Sv48Pte::invalid();
        assert!(!pte.is_valid());
        assert!(!pte.is_leaf());
    }

    #[test]
    fn test_sv48_pte_non_leaf() {
        // Valid but no R/W/X — points to next-level page table.
        let pte = Sv48Pte::new(0xABCDE, PteFlags::new(PteFlags::V));
        assert!(pte.is_valid());
        assert!(!pte.is_leaf());
    }

    #[test]
    fn test_sv48_constants() {
        assert_eq!(Sv48Pte::LEVELS, 4);
        assert_eq!(Sv48Pte::ENTRIES_PER_PAGE, 512);
        assert_eq!(Sv48Pte::PAGE_SIZE, 4096);
        assert_eq!(Sv48Pte::MEGA_PAGE_SIZE, 2 * 1024 * 1024);
        assert_eq!(Sv48Pte::GIGA_PAGE_SIZE, 1024 * 1024 * 1024);
    }

    // -- SV48 Page Table stub -----------------------------------------------

    #[test]
    fn test_sv48_page_table_map_translate() {
        let mut pt = Sv48PageTable::new();
        let flags = PteFlags::read_write();
        pt.map(0x1000, 0x8000_0000, flags).unwrap();
        let (paddr, pflags) = pt.translate(0x1000).unwrap();
        assert_eq!(paddr, 0x8000_0000);
        assert_eq!(pflags, flags);
        assert_eq!(pt.mapping_count(), 1);
    }

    #[test]
    fn test_sv48_page_table_unmap() {
        let mut pt = Sv48PageTable::new();
        pt.map(0x1000, 0x8000_0000, PteFlags::read_only()).unwrap();
        pt.unmap(0x1000).unwrap();
        assert!(pt.translate(0x1000).is_none());
        assert_eq!(pt.mapping_count(), 0);
    }

    #[test]
    fn test_sv48_page_table_protect() {
        let mut pt = Sv48PageTable::new();
        pt.map(0x1000, 0x8000_0000, PteFlags::read_only()).unwrap();
        pt.protect(0x1000, PteFlags::read_write()).unwrap();
        let (_, flags) = pt.translate(0x1000).unwrap();
        assert!(flags.has(PteFlags::W));
    }

    #[test]
    fn test_sv48_page_table_update_existing() {
        let mut pt = Sv48PageTable::new();
        pt.map(0x1000, 0x8000_0000, PteFlags::read_only()).unwrap();
        pt.map(0x1000, 0x9000_0000, PteFlags::read_write()).unwrap();
        let (paddr, _) = pt.translate(0x1000).unwrap();
        assert_eq!(paddr, 0x9000_0000);
        assert_eq!(pt.mapping_count(), 1);
    }

    #[test]
    fn test_sv48_page_table_translate_unmapped() {
        let pt = Sv48PageTable::new();
        assert!(pt.translate(0xDEAD).is_none());
    }

    #[test]
    fn test_sv48_page_table_unmap_nonexistent() {
        let mut pt = Sv48PageTable::new();
        assert_eq!(pt.unmap(0x1000), Err(HalError::NotSupported));
    }

    #[test]
    fn test_sv48_page_table_protect_nonexistent() {
        let mut pt = Sv48PageTable::new();
        assert_eq!(
            pt.protect(0x1000, PteFlags::read_write()),
            Err(HalError::NotSupported)
        );
    }

    // -- PLIC ---------------------------------------------------------------

    #[test]
    fn test_plic_priority() {
        let mut plic = Plic::new();
        plic.set_priority(1, PlicPriority(5)).unwrap();
        assert!(PlicPriority(7).is_valid());
        assert!(!PlicPriority(8).is_valid());
    }

    #[test]
    fn test_plic_priority_invalid_source() {
        let mut plic = Plic::new();
        assert_eq!(
            plic.set_priority(0, PlicPriority(1)),
            Err(HalError::InvalidConfig)
        );
    }

    #[test]
    fn test_plic_enable_disable() {
        let mut plic = Plic::new();
        let ctx = PlicContext(0);
        plic.enable(ctx, 1).unwrap();
        plic.disable(ctx, 1).unwrap();
    }

    #[test]
    fn test_plic_claim_complete() {
        let mut plic = Plic::new();
        let ctx = PlicContext(0);
        plic.set_priority(5, PlicPriority(3)).unwrap();
        plic.enable(ctx, 5).unwrap();
        plic.set_pending(5);

        let source = plic.claim(ctx);
        assert_eq!(source, 5);

        plic.complete(ctx, 5);
    }

    #[test]
    fn test_plic_claim_no_pending() {
        let mut plic = Plic::new();
        let ctx = PlicContext(0);
        assert_eq!(plic.claim(ctx), 0);
    }

    #[test]
    fn test_plic_claim_below_threshold() {
        let mut plic = Plic::new();
        let ctx = PlicContext(0);
        plic.set_priority(5, PlicPriority(2)).unwrap();
        plic.enable(ctx, 5).unwrap();
        plic.set_threshold(ctx, 3).unwrap();
        plic.set_pending(5);
        // Priority 2 is below threshold 3, so no claim.
        assert_eq!(plic.claim(ctx), 0);
    }

    #[test]
    fn test_plic_claim_highest_priority() {
        let mut plic = Plic::new();
        let ctx = PlicContext(0);
        plic.set_priority(1, PlicPriority(2)).unwrap();
        plic.set_priority(2, PlicPriority(5)).unwrap();
        plic.enable(ctx, 1).unwrap();
        plic.enable(ctx, 2).unwrap();
        plic.set_pending(1);
        plic.set_pending(2);
        // Source 2 has higher priority.
        assert_eq!(plic.claim(ctx), 2);
    }

    // -- CLINT --------------------------------------------------------------

    #[test]
    fn test_clint_new() {
        let clint = Clint::new();
        assert_eq!(clint.read_mtime(), 0);
        assert_eq!(clint.frequency(), 10_000_000);
    }

    #[test]
    fn test_clint_set_timer() {
        let mut clint = Clint::new();
        clint.set_timer(0, 1000).unwrap();
        assert!(!clint.is_timer_pending(0));
        clint.set_mtime(1000);
        assert!(clint.is_timer_pending(0));
    }

    #[test]
    fn test_clint_set_timer_invalid_hart() {
        let mut clint = Clint::new();
        assert_eq!(clint.set_timer(8, 1000), Err(HalError::OutOfRange));
    }

    #[test]
    fn test_clint_schedule_tick() {
        let mut clint = Clint::new();
        clint.set_mtime(5000);
        clint.schedule_tick(0, 1000).unwrap();
        assert!(!clint.is_timer_pending(0));
        clint.set_mtime(6000);
        assert!(clint.is_timer_pending(0));
    }

    #[test]
    fn test_clint_us_to_ticks() {
        let clint = Clint::new();
        // 10 MHz: 1 us = 10 ticks.
        assert_eq!(clint.us_to_ticks(1), 10);
        assert_eq!(clint.us_to_ticks(1000), 10_000);
    }

    // -- SBI ----------------------------------------------------------------

    #[test]
    fn test_sbi_result_success() {
        let result = SbiResult::success(42);
        assert!(result.is_success());
        assert_eq!(result.value, 42);
    }

    #[test]
    fn test_sbi_result_error() {
        let result = SbiResult::error(-1);
        assert!(!result.is_success());
        assert_eq!(result.error, -1);
    }

    // -- Hart State ---------------------------------------------------------

    #[test]
    fn test_hart_state_from_value() {
        assert_eq!(HartState::from_value(0), Some(HartState::Started));
        assert_eq!(HartState::from_value(1), Some(HartState::Stopped));
        assert_eq!(HartState::from_value(2), Some(HartState::StartPending));
        assert_eq!(HartState::from_value(3), Some(HartState::StopPending));
        assert_eq!(HartState::from_value(7), None);
    }

    // -- RISC-V Feature Detection -------------------------------------------

    #[test]
    fn test_riscv_features_rv64gc() {
        let f = RiscvFeatures::rv64gc();
        assert!(f.is_gc());
        assert!(f.has_i);
        assert!(f.has_m);
        assert!(f.has_a);
        assert!(f.has_f);
        assert!(f.has_d);
        assert!(f.has_c);
        assert_eq!(f.xlen, 64);
    }

    #[test]
    fn test_riscv_features_rv64i_only() {
        let f = RiscvFeatures::rv64i_only();
        assert!(!f.is_gc());
        assert!(f.has_i);
        assert!(!f.has_m);
    }

    #[test]
    fn test_riscv_features_from_misa() {
        // Build a misa value with I, M, A, F, D, C set.
        let misa = (1u64 << 8) | (1 << 12) | (1 << 0) | (1 << 5) | (1 << 3) | (1 << 2);
        let f = RiscvFeatures::from_misa(misa);
        assert!(f.is_gc());
    }

    #[test]
    fn test_riscv_features_from_misa_partial() {
        // Only I and M.
        let misa = (1u64 << 8) | (1 << 12);
        let f = RiscvFeatures::from_misa(misa);
        assert!(f.has_i);
        assert!(f.has_m);
        assert!(!f.has_a);
        assert!(!f.is_gc());
    }

    // -- DMA types ----------------------------------------------------------

    #[test]
    fn test_dma_token_default() {
        let token = DmaToken {
            channel: 0,
            transfer_id: 1,
            complete: false,
            error: false,
        };
        assert!(!token.complete);
        assert!(!token.error);
    }

    #[test]
    fn test_dma_descriptor() {
        let desc = DmaDescriptor {
            src_addr: 0x1000_0000,
            dst_addr: 0x2000_0000,
            length: 4096,
            direction: DmaDirection::FpgaToMemory,
        };
        assert_eq!(desc.direction, DmaDirection::FpgaToMemory);
        assert_eq!(desc.length, 4096);
    }

    // -- BusProtocol --------------------------------------------------------

    #[test]
    fn test_bus_protocol_variants() {
        assert_ne!(BusProtocol::Can, BusProtocol::Arinc429);
        assert_ne!(BusProtocol::Mil1553, BusProtocol::SpaceWire);
        assert_eq!(BusProtocol::Ccsds, BusProtocol::Ccsds);
    }

    // -- I2C config ---------------------------------------------------------

    #[test]
    fn test_i2c_config() {
        let cfg = I2cConfig {
            speed: I2cSpeed::Fast,
            address_mode: I2cAddressMode::SevenBit,
            clock_stretching: true,
            timeout_us: 1000,
        };
        assert_eq!(cfg.speed, I2cSpeed::Fast);
        assert_eq!(cfg.address_mode, I2cAddressMode::SevenBit);
        assert!(cfg.clock_stretching);
        assert_eq!(cfg.timeout_us, 1000);
    }

    #[test]
    fn test_i2c_speed_variants() {
        assert_ne!(I2cSpeed::Standard, I2cSpeed::Fast);
        assert_ne!(I2cSpeed::Fast, I2cSpeed::FastPlus);
    }

    #[test]
    fn test_i2c_address_mode_variants() {
        assert_ne!(I2cAddressMode::SevenBit, I2cAddressMode::TenBit);
    }

    // -- SPI config ---------------------------------------------------------

    #[test]
    fn test_spi_config() {
        let cfg = SpiConfig {
            mode: SpiMode::Mode0,
            clock_hz: 1_000_000,
            bit_order: SpiBitOrder::MsbFirst,
            word_size: 8,
            cs_active_low: true,
            timeout_us: 5000,
        };
        assert_eq!(cfg.mode, SpiMode::Mode0);
        assert_eq!(cfg.clock_hz, 1_000_000);
        assert_eq!(cfg.word_size, 8);
        assert!(cfg.cs_active_low);
    }

    #[test]
    fn test_spi_mode_variants() {
        assert_ne!(SpiMode::Mode0, SpiMode::Mode1);
        assert_ne!(SpiMode::Mode2, SpiMode::Mode3);
        assert_eq!(SpiMode::Mode0, SpiMode::Mode0);
    }

    // -- GPIO config --------------------------------------------------------

    #[test]
    fn test_gpio_config() {
        let cfg = GpioConfig {
            mode: GpioPinMode::Input,
            pull: GpioPull::PullUp,
            interrupt: Some(GpioInterruptEdge::Rising),
            debounce_us: 50,
        };
        assert_eq!(cfg.mode, GpioPinMode::Input);
        assert_eq!(cfg.pull, GpioPull::PullUp);
        assert_eq!(cfg.interrupt, Some(GpioInterruptEdge::Rising));
        assert_eq!(cfg.debounce_us, 50);
    }

    #[test]
    fn test_gpio_pin_mode_alternate() {
        let mode = GpioPinMode::AlternateFunction(3);
        match mode {
            GpioPinMode::AlternateFunction(n) => assert_eq!(n, 3),
            _ => panic!("expected AlternateFunction"),
        }
    }

    #[test]
    fn test_gpio_interrupt_edge_variants() {
        assert_ne!(GpioInterruptEdge::Rising, GpioInterruptEdge::Falling);
        assert_ne!(GpioInterruptEdge::Both, GpioInterruptEdge::LevelHigh);
    }

    // -- UART config --------------------------------------------------------

    #[test]
    fn test_uart_config() {
        let cfg = UartConfig {
            baud_rate: 115200,
            data_bits: 8,
            parity: UartParity::None,
            stop_bits: UartStopBits::One,
            flow_control: UartFlowControl::None,
            rx_mode: UartRxMode::Polling,
            timeout_us: 0,
        };
        assert_eq!(cfg.baud_rate, 115200);
        assert_eq!(cfg.data_bits, 8);
        assert_eq!(cfg.parity, UartParity::None);
    }

    #[test]
    fn test_uart_rx_mode_interrupt() {
        let mode = UartRxMode::Interrupt { fifo_threshold: 8 };
        match mode {
            UartRxMode::Interrupt { fifo_threshold } => assert_eq!(fifo_threshold, 8),
            _ => panic!("expected Interrupt mode"),
        }
    }

    #[test]
    fn test_uart_rx_mode_dma() {
        let mode = UartRxMode::Dma {
            buffer_addr: 0x1000,
            buffer_size: 4096,
        };
        match mode {
            UartRxMode::Dma {
                buffer_addr,
                buffer_size,
            } => {
                assert_eq!(buffer_addr, 0x1000);
                assert_eq!(buffer_size, 4096);
            }
            _ => panic!("expected Dma mode"),
        }
    }

    #[test]
    fn test_uart_irq_source_variants() {
        assert_ne!(UartIrqSource::RxDataAvailable, UartIrqSource::TxEmpty);
        assert_ne!(UartIrqSource::FramingError, UartIrqSource::ParityError);
        assert_eq!(UartIrqSource::Overrun, UartIrqSource::Overrun);
    }

    // -- CSI config ---------------------------------------------------------

    #[test]
    fn test_csi_config() {
        let cfg = CsiConfig {
            lanes: 2,
            lane_speed_mbps: 1500,
            width: 1920,
            height: 1080,
            pixel_format: CsiPixelFormat::Yuv422,
            fps: 30,
            num_buffers: 3,
        };
        assert_eq!(cfg.lanes, 2);
        assert_eq!(cfg.width, 1920);
        assert_eq!(cfg.height, 1080);
        assert_eq!(cfg.pixel_format, CsiPixelFormat::Yuv422);
        assert_eq!(cfg.fps, 30);
    }

    #[test]
    fn test_csi_frame() {
        let frame = CsiFrame {
            buffer_index: 1,
            timestamp_us: 123456,
            sequence: 42,
            bytes_used: 1920 * 1080 * 2,
        };
        assert_eq!(frame.buffer_index, 1);
        assert_eq!(frame.sequence, 42);
    }

    #[test]
    fn test_csi_irq_source() {
        let src = CsiIrqSource::FrameComplete(2);
        match src {
            CsiIrqSource::FrameComplete(idx) => assert_eq!(idx, 2),
            _ => panic!("expected FrameComplete"),
        }
        assert_ne!(CsiIrqSource::Overflow, CsiIrqSource::CrcError);
    }

    #[test]
    fn test_csi_pixel_format_variants() {
        assert_ne!(CsiPixelFormat::Raw8, CsiPixelFormat::Raw10);
        assert_ne!(CsiPixelFormat::Yuv422, CsiPixelFormat::Rgb888);
    }

    // -- I2S config ---------------------------------------------------------

    #[test]
    fn test_i2s_config() {
        let cfg = I2sConfig {
            mode: I2sMode::Standard,
            role: I2sRole::Master,
            sample_rate_hz: 16000,
            bit_depth: I2sBitDepth::Bits16,
            channels: 1,
            buffer_frames: 1024,
            num_buffers: 2,
        };
        assert_eq!(cfg.mode, I2sMode::Standard);
        assert_eq!(cfg.role, I2sRole::Master);
        assert_eq!(cfg.sample_rate_hz, 16000);
        assert_eq!(cfg.channels, 1);
    }

    #[test]
    fn test_i2s_mode_tdm() {
        let mode = I2sMode::Tdm { num_slots: 8 };
        match mode {
            I2sMode::Tdm { num_slots } => assert_eq!(num_slots, 8),
            _ => panic!("expected TDM mode"),
        }
    }

    #[test]
    fn test_audio_buffer() {
        let buf = AudioBuffer {
            buffer_index: 0,
            timestamp_us: 999,
            sample_count: 1024,
            bytes_used: 2048,
        };
        assert_eq!(buf.buffer_index, 0);
        assert_eq!(buf.sample_count, 1024);
        assert_eq!(buf.bytes_used, 2048);
    }

    #[test]
    fn test_i2s_irq_source() {
        let src = I2sIrqSource::BufferComplete(1);
        match src {
            I2sIrqSource::BufferComplete(idx) => assert_eq!(idx, 1),
            _ => panic!("expected BufferComplete"),
        }
        assert_ne!(I2sIrqSource::Overrun, I2sIrqSource::Underrun);
    }

    #[test]
    fn test_i2s_bit_depth_variants() {
        assert_ne!(I2sBitDepth::Bits16, I2sBitDepth::Bits24);
        assert_ne!(I2sBitDepth::Bits24, I2sBitDepth::Bits32);
    }
}
