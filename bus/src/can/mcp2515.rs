// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! MCP2515 SPI CAN controller driver.
//!
//! Implements the [`CanController`] trait for the Microchip MCP2515 standalone
//! CAN controller, commonly used on embedded boards connected via SPI.
//!
//! # Hardware interface
//!
//! The MCP2515 communicates via a 4-wire SPI bus (SCK, MOSI, MISO, CS).
//! This driver depends on an [`SpiDevice`] trait abstraction provided by the
//! platform HAL. The driver handles register-level communication per the
//! MCP2515 datasheet (DS20001801).
//!
//! # References
//!
//! - Microchip MCP2515 datasheet (DS20001801)
//! - ISO 11898-1 CAN data link layer

use crate::can::controller::{CanBitTiming, CanController, CanMode, CanStats};
use crate::can::frame::{CanFrame, FrameType};
use crate::can::state::BusState;
use crate::BusError;

// ---------------------------------------------------------------------------
// SPI abstraction
// ---------------------------------------------------------------------------

/// SPI device abstraction for the MCP2515 driver.
///
/// Platform HALs implement this trait to provide byte-level SPI access.
/// The driver manages chip-select (CS) via `transfer_frame` which asserts
/// CS for the duration of the entire SPI transaction.
pub trait SpiDevice {
    /// Perform a full-duplex SPI transfer within a single CS assertion.
    ///
    /// Writes `tx` bytes while simultaneously reading into `rx`.
    /// Both slices must have the same length.
    fn transfer(&mut self, tx: &[u8], rx: &mut [u8]) -> Result<(), BusError>;

    /// Write bytes to SPI (ignore read data). CS is asserted for the transfer.
    fn write(&mut self, data: &[u8]) -> Result<(), BusError>;
}

// ---------------------------------------------------------------------------
// MCP2515 register definitions (from DS20001801)
// ---------------------------------------------------------------------------

/// MCP2515 SPI instructions.
mod spi_cmd {
    pub const RESET: u8 = 0xC0;
    pub const READ: u8 = 0x03;
    pub const WRITE: u8 = 0x02;
    pub const BIT_MODIFY: u8 = 0x05;
    pub const READ_RX_BUFFER_0: u8 = 0x90;
    pub const LOAD_TX_BUFFER_0: u8 = 0x40;
    pub const RTS_TXB0: u8 = 0x81;
}

/// MCP2515 register addresses.
mod reg {
    pub const CANSTAT: u8 = 0x0E;
    pub const CANCTRL: u8 = 0x0F;
    pub const CNF1: u8 = 0x2A;
    pub const CNF2: u8 = 0x29;
    pub const CNF3: u8 = 0x28;
    pub const CANINTF: u8 = 0x2C;
    pub const RXB0CTRL: u8 = 0x60;
}

/// CANCTRL mode bits (bits 7:5).
mod mode_bits {
    pub const NORMAL: u8 = 0x00;
    pub const SLEEP: u8 = 0x20;
    pub const LOOPBACK: u8 = 0x40;
    pub const LISTEN_ONLY: u8 = 0x60;
    pub const CONFIGURATION: u8 = 0x80;
    pub const MASK: u8 = 0xE0;
}

/// CANINTF interrupt flag bits.
mod int_flag {
    pub const RX0IF: u8 = 0x01;
}

// ---------------------------------------------------------------------------
// MCP2515 CAN controller
// ---------------------------------------------------------------------------

/// MCP2515 SPI CAN controller driver.
///
/// Wraps an [`SpiDevice`] to provide [`CanController`] access to an MCP2515.
/// The controller supports CAN 2.0A/B standard and extended frames with up
/// to 8 bytes of payload.
pub struct Mcp2515<S: SpiDevice> {
    spi: S,
    mode: CanMode,
    stats: CanStats,
    initialized: bool,
}

impl<S: SpiDevice> Mcp2515<S> {
    /// Create a new MCP2515 driver wrapping the given SPI device.
    pub fn new(spi: S) -> Self {
        Self {
            spi,
            mode: CanMode::Configuration,
            stats: CanStats::default(),
            initialized: false,
        }
    }

    /// Read a single register.
    fn read_reg(&mut self, addr: u8) -> Result<u8, BusError> {
        let tx = [spi_cmd::READ, addr, 0x00];
        let mut rx = [0u8; 3];
        self.spi.transfer(&tx, &mut rx)?;
        Ok(rx[2])
    }

    /// Write a single register.
    fn write_reg(&mut self, addr: u8, value: u8) -> Result<(), BusError> {
        self.spi.write(&[spi_cmd::WRITE, addr, value])
    }

    /// Modify specific bits in a register (read-modify-write via SPI command).
    fn bit_modify(&mut self, addr: u8, mask: u8, value: u8) -> Result<(), BusError> {
        self.spi.write(&[spi_cmd::BIT_MODIFY, addr, mask, value])
    }

    /// Send the hardware reset SPI command.
    fn hw_reset(&mut self) -> Result<(), BusError> {
        self.spi.write(&[spi_cmd::RESET])
    }

    /// Get the current CANSTAT operating mode bits.
    fn read_mode_bits(&mut self) -> Result<u8, BusError> {
        let canstat = self.read_reg(reg::CANSTAT)?;
        Ok(canstat & mode_bits::MASK)
    }

    /// Set the CANCTRL operating mode bits and wait for the mode to take effect.
    fn set_mode_bits(&mut self, mode: u8) -> Result<(), BusError> {
        self.bit_modify(reg::CANCTRL, mode_bits::MASK, mode)?;
        // Verify mode changed (in real hardware, may need retries).
        let actual = self.read_mode_bits()?;
        if actual != mode {
            return Err(BusError::Timeout);
        }
        Ok(())
    }

    /// Read an RX buffer (buffer 0) and return the decoded CAN frame.
    fn read_rx_buffer(&mut self) -> Result<Option<CanFrame>, BusError> {
        // Check if RX0 interrupt flag is set.
        let intf = self.read_reg(reg::CANINTF)?;
        if intf & int_flag::RX0IF == 0 {
            return Ok(None);
        }

        // Read 13 bytes from RX buffer 0 (SIDH, SIDL, EID8, EID0, DLC, D0-D7).
        let mut tx = [0u8; 14];
        let mut rx = [0u8; 14];
        tx[0] = spi_cmd::READ_RX_BUFFER_0;
        self.spi.transfer(&tx, &mut rx)?;

        let sidh = rx[1];
        let sidl = rx[2];
        let eid8 = rx[3];
        let eid0 = rx[4];
        let dlc = rx[5] & 0x0F;

        let is_extended = sidl & 0x08 != 0;

        let (id, frame_type) = if is_extended {
            let id = ((sidh as u32) << 21)
                | (((sidl as u32) & 0xE0) << 13)
                | (((sidl as u32) & 0x03) << 16)
                | ((eid8 as u32) << 8)
                | (eid0 as u32);
            (id, FrameType::Extended)
        } else {
            let id = ((sidh as u32) << 3) | ((sidl as u32) >> 5);
            (id, FrameType::Standard)
        };

        let data_len = core::cmp::min(dlc as usize, 8);
        let data = &rx[6..6 + data_len];

        // Clear the RX0 interrupt flag.
        self.bit_modify(reg::CANINTF, int_flag::RX0IF, 0x00)?;

        let frame = match frame_type {
            FrameType::Standard => CanFrame::new_standard(id, data)?,
            FrameType::Extended => CanFrame::new_extended(id, data)?,
        };

        self.stats.rx_frames += 1;
        Ok(Some(frame))
    }

    /// Load a CAN frame into TX buffer 0 and request to send.
    fn write_tx_buffer(&mut self, frame: &CanFrame) -> Result<(), BusError> {
        // Build the TX buffer data: SIDH, SIDL, EID8, EID0, DLC, D0-D7
        let mut buf = [0u8; 14];
        buf[0] = spi_cmd::LOAD_TX_BUFFER_0;

        match frame.frame_type {
            FrameType::Standard => {
                buf[1] = ((frame.id >> 3) & 0xFF) as u8; // SIDH
                buf[2] = ((frame.id & 0x07) << 5) as u8; // SIDL (IDE=0)
            }
            FrameType::Extended => {
                buf[1] = ((frame.id >> 21) & 0xFF) as u8; // SIDH
                buf[2] = (((frame.id >> 13) & 0xE0) as u8)
                    | 0x08 // IDE bit
                    | (((frame.id >> 16) & 0x03) as u8); // SIDL
                buf[3] = ((frame.id >> 8) & 0xFF) as u8; // EID8
                buf[4] = (frame.id & 0xFF) as u8; // EID0
            }
        }

        let dlc = core::cmp::min(frame.data.len(), 8);
        buf[5] = dlc as u8;
        buf[6..6 + dlc].copy_from_slice(&frame.data[..dlc]);

        self.spi.write(&buf[..6 + dlc + 1])?;

        // Request to send TX buffer 0.
        self.spi.write(&[spi_cmd::RTS_TXB0])?;

        self.stats.tx_frames += 1;
        Ok(())
    }
}

impl<S: SpiDevice> CanController for Mcp2515<S> {
    fn init(&mut self) -> Result<(), BusError> {
        // Hardware reset.
        self.hw_reset()?;
        // Verify we're in configuration mode after reset.
        let mode = self.read_mode_bits()?;
        if mode != mode_bits::CONFIGURATION {
            return Err(BusError::NotSupported);
        }
        // Configure RXB0CTRL to accept all messages.
        self.write_reg(reg::RXB0CTRL, 0x60)?;
        self.mode = CanMode::Configuration;
        self.initialized = true;
        Ok(())
    }

    fn set_mode(&mut self, mode: CanMode) -> Result<(), BusError> {
        if !self.initialized {
            return Err(BusError::NotSupported);
        }
        let bits = match mode {
            CanMode::Normal => mode_bits::NORMAL,
            CanMode::ListenOnly => mode_bits::LISTEN_ONLY,
            CanMode::Loopback => mode_bits::LOOPBACK,
            CanMode::Configuration => mode_bits::CONFIGURATION,
            CanMode::Sleep => mode_bits::SLEEP,
        };
        self.set_mode_bits(bits)?;
        self.mode = mode;
        Ok(())
    }

    fn mode(&self) -> CanMode {
        self.mode
    }

    fn set_bit_timing(&mut self, timing: &CanBitTiming) -> Result<(), BusError> {
        if self.mode != CanMode::Configuration {
            return Err(BusError::NotSupported);
        }
        // CNF1: SJW(2) + BRP(6)
        let cnf1 = ((timing.sjw.saturating_sub(1) & 0x03) << 6)
            | (timing.prescaler.saturating_sub(1) as u8 & 0x3F);
        // CNF2: BTLMODE(1)=1 + SAM(1)=0 + PHSEG1(3) + PRSEG(3)
        let prseg = (timing.tseg1 / 2).saturating_sub(1) & 0x07;
        let phseg1 = (timing.tseg1 - timing.tseg1 / 2).saturating_sub(1) & 0x07;
        let cnf2 = 0x80 | (phseg1 << 3) | prseg;
        // CNF3: PHSEG2(3)
        let cnf3 = timing.tseg2.saturating_sub(1) & 0x07;

        self.write_reg(reg::CNF1, cnf1)?;
        self.write_reg(reg::CNF2, cnf2)?;
        self.write_reg(reg::CNF3, cnf3)?;
        Ok(())
    }

    fn transmit(&mut self, frame: &CanFrame) -> Result<(), BusError> {
        if self.mode != CanMode::Normal && self.mode != CanMode::Loopback {
            return Err(BusError::NotSupported);
        }
        self.write_tx_buffer(frame)
    }

    fn receive(&mut self) -> Result<Option<CanFrame>, BusError> {
        if self.mode == CanMode::Configuration || self.mode == CanMode::Sleep {
            return Err(BusError::NotSupported);
        }
        self.read_rx_buffer()
    }

    fn bus_state(&self) -> BusState {
        // Return cached state; real driver would read from hardware.
        BusState::ErrorActive
    }

    fn stats(&self) -> CanStats {
        self.stats
    }

    fn reset(&mut self) -> Result<(), BusError> {
        self.hw_reset()?;
        self.mode = CanMode::Configuration;
        self.stats = CanStats::default();
        self.initialized = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    /// Mock SPI device for testing the MCP2515 driver.
    struct MockSpi {
        /// Expected SPI transactions: (tx_bytes, rx_bytes) pairs.
        responses: Vec<Vec<u8>>,
        call_idx: usize,
    }

    impl MockSpi {
        fn new() -> Self {
            Self {
                responses: Vec::new(),
                call_idx: 0,
            }
        }

        fn push_response(&mut self, data: &[u8]) {
            self.responses.push(Vec::from(data));
        }
    }

    impl SpiDevice for MockSpi {
        fn transfer(&mut self, _tx: &[u8], rx: &mut [u8]) -> Result<(), BusError> {
            if self.call_idx < self.responses.len() {
                let resp = &self.responses[self.call_idx];
                let len = core::cmp::min(rx.len(), resp.len());
                rx[..len].copy_from_slice(&resp[..len]);
                self.call_idx += 1;
            }
            Ok(())
        }

        fn write(&mut self, _data: &[u8]) -> Result<(), BusError> {
            self.call_idx += 1;
            Ok(())
        }
    }

    #[test]
    fn test_mcp2515_new() {
        let spi = MockSpi::new();
        let drv = Mcp2515::new(spi);
        assert_eq!(drv.mode(), CanMode::Configuration);
        assert!(!drv.initialized);
    }

    #[test]
    fn test_mcp2515_init() {
        let mut spi = MockSpi::new();
        // Reset response (write, no rx)
        spi.push_response(&[]);
        // read_mode_bits: READ CANSTAT -> returns CONFIGURATION mode
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]);
        // write_reg RXB0CTRL
        spi.push_response(&[]);

        let mut drv = Mcp2515::new(spi);
        drv.init().unwrap();
        assert!(drv.initialized);
        assert_eq!(drv.mode(), CanMode::Configuration);
    }

    #[test]
    fn test_mcp2515_set_mode_loopback() {
        let mut spi = MockSpi::new();
        // init responses
        spi.push_response(&[]); // reset
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]); // read CANSTAT
        spi.push_response(&[]); // write RXB0CTRL
                                // set_mode_bits: bit_modify
        spi.push_response(&[]);
        // read_mode_bits: verify
        spi.push_response(&[0, 0, mode_bits::LOOPBACK]);

        let mut drv = Mcp2515::new(spi);
        drv.init().unwrap();
        drv.set_mode(CanMode::Loopback).unwrap();
        assert_eq!(drv.mode(), CanMode::Loopback);
    }

    #[test]
    fn test_mcp2515_set_bit_timing() {
        let mut spi = MockSpi::new();
        // init responses
        spi.push_response(&[]); // reset
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]); // read CANSTAT
        spi.push_response(&[]); // write RXB0CTRL
                                // set_bit_timing: 3 write_reg calls
        spi.push_response(&[]); // CNF1
        spi.push_response(&[]); // CNF2
        spi.push_response(&[]); // CNF3

        let mut drv = Mcp2515::new(spi);
        drv.init().unwrap();
        let timing = CanBitTiming::standard_500kbps();
        drv.set_bit_timing(&timing).unwrap();
    }

    #[test]
    fn test_mcp2515_stats_default() {
        let spi = MockSpi::new();
        let drv = Mcp2515::new(spi);
        assert_eq!(drv.stats().tx_frames, 0);
        assert_eq!(drv.stats().rx_frames, 0);
    }

    #[test]
    fn test_mcp2515_bus_state() {
        let spi = MockSpi::new();
        let drv = Mcp2515::new(spi);
        assert_eq!(drv.bus_state(), BusState::ErrorActive);
    }

    #[test]
    fn test_mcp2515_transmit_wrong_mode() {
        let mut spi = MockSpi::new();
        spi.push_response(&[]); // reset
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]);
        spi.push_response(&[]);

        let mut drv = Mcp2515::new(spi);
        drv.init().unwrap();
        let frame = CanFrame::new_standard(0x100, &[0x01]).unwrap();
        assert_eq!(drv.transmit(&frame).err(), Some(BusError::NotSupported));
    }

    #[test]
    fn test_mcp2515_receive_wrong_mode() {
        let mut spi = MockSpi::new();
        spi.push_response(&[]); // reset
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]);
        spi.push_response(&[]);

        let mut drv = Mcp2515::new(spi);
        drv.init().unwrap();
        assert_eq!(drv.receive().err(), Some(BusError::NotSupported));
    }

    #[test]
    fn test_mcp2515_receive_empty() {
        let mut spi = MockSpi::new();
        spi.push_response(&[]); // reset
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]);
        spi.push_response(&[]);
        // set_mode to Normal
        spi.push_response(&[]); // bit_modify
        spi.push_response(&[0, 0, mode_bits::NORMAL]); // verify
                                                       // receive: read CANINTF -> no RX0IF
        spi.push_response(&[0, 0, 0x00]);

        let mut drv = Mcp2515::new(spi);
        drv.init().unwrap();
        drv.set_mode(CanMode::Normal).unwrap();
        assert!(drv.receive().unwrap().is_none());
    }

    #[test]
    fn test_mcp2515_reset() {
        let mut spi = MockSpi::new();
        spi.push_response(&[]); // init reset
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]);
        spi.push_response(&[]);
        // reset
        spi.push_response(&[]); // hw_reset

        let mut drv = Mcp2515::new(spi);
        drv.init().unwrap();
        drv.reset().unwrap();
        assert!(!drv.initialized);
        assert_eq!(drv.mode(), CanMode::Configuration);
    }

    // -------------------------------------------------------------------
    // set_mode coverage: Normal, ListenOnly, Configuration, Sleep
    // -------------------------------------------------------------------

    #[test]
    fn test_mcp2515_set_mode_normal() {
        let mut spi = MockSpi::new();
        // init: reset + read CANSTAT + write RXB0CTRL
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]);
        spi.push_response(&[]);
        // set_mode(Normal): bit_modify + read_mode_bits verify
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::NORMAL]);

        let mut drv = Mcp2515::new(spi);
        drv.init().unwrap();
        drv.set_mode(CanMode::Normal).unwrap();
        assert_eq!(drv.mode(), CanMode::Normal);
    }

    #[test]
    fn test_mcp2515_set_mode_listen_only() {
        let mut spi = MockSpi::new();
        // init
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]);
        spi.push_response(&[]);
        // set_mode(ListenOnly): bit_modify + verify
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::LISTEN_ONLY]);

        let mut drv = Mcp2515::new(spi);
        drv.init().unwrap();
        drv.set_mode(CanMode::ListenOnly).unwrap();
        assert_eq!(drv.mode(), CanMode::ListenOnly);
    }

    #[test]
    fn test_mcp2515_set_mode_configuration() {
        let mut spi = MockSpi::new();
        // init
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]);
        spi.push_response(&[]);
        // set_mode(Configuration): bit_modify + verify
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]);

        let mut drv = Mcp2515::new(spi);
        drv.init().unwrap();
        drv.set_mode(CanMode::Configuration).unwrap();
        assert_eq!(drv.mode(), CanMode::Configuration);
    }

    #[test]
    fn test_mcp2515_set_mode_sleep() {
        let mut spi = MockSpi::new();
        // init
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]);
        spi.push_response(&[]);
        // set_mode(Sleep): bit_modify + verify
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::SLEEP]);

        let mut drv = Mcp2515::new(spi);
        drv.init().unwrap();
        drv.set_mode(CanMode::Sleep).unwrap();
        assert_eq!(drv.mode(), CanMode::Sleep);
    }

    // -------------------------------------------------------------------
    // set_mode before init should fail
    // -------------------------------------------------------------------

    #[test]
    fn test_mcp2515_set_mode_not_initialized() {
        let spi = MockSpi::new();
        let mut drv = Mcp2515::new(spi);
        // Not initialized, set_mode should return NotSupported
        assert_eq!(
            drv.set_mode(CanMode::Normal).err(),
            Some(BusError::NotSupported)
        );
    }

    // -------------------------------------------------------------------
    // transmit standard frame in Normal mode
    // -------------------------------------------------------------------

    #[test]
    fn test_mcp2515_transmit_standard_normal() {
        let mut spi = MockSpi::new();
        // init
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]);
        spi.push_response(&[]);
        // set_mode(Normal)
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::NORMAL]);
        // transmit: write_tx_buffer (LOAD_TX_BUFFER_0) + RTS_TXB0
        spi.push_response(&[]); // LOAD_TX_BUFFER_0
        spi.push_response(&[]); // RTS_TXB0

        let mut drv = Mcp2515::new(spi);
        drv.init().unwrap();
        drv.set_mode(CanMode::Normal).unwrap();

        let frame = CanFrame::new_standard(0x100, &[0xAA, 0xBB]).unwrap();
        drv.transmit(&frame).unwrap();
        assert_eq!(drv.stats().tx_frames, 1);
    }

    // -------------------------------------------------------------------
    // transmit extended frame in Loopback mode
    // -------------------------------------------------------------------

    #[test]
    fn test_mcp2515_transmit_extended_loopback() {
        let mut spi = MockSpi::new();
        // init
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]);
        spi.push_response(&[]);
        // set_mode(Loopback)
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::LOOPBACK]);
        // transmit: LOAD_TX_BUFFER_0 + RTS_TXB0
        spi.push_response(&[]);
        spi.push_response(&[]);

        let mut drv = Mcp2515::new(spi);
        drv.init().unwrap();
        drv.set_mode(CanMode::Loopback).unwrap();

        let frame = CanFrame::new_extended(0x1234_5678, &[0x01, 0x02, 0x03]).unwrap();
        drv.transmit(&frame).unwrap();
        assert_eq!(drv.stats().tx_frames, 1);
    }

    // -------------------------------------------------------------------
    // receive standard frame (RX0IF set)
    // -------------------------------------------------------------------

    #[test]
    fn test_mcp2515_receive_standard_frame() {
        let mut spi = MockSpi::new();
        // init
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]);
        spi.push_response(&[]);
        // set_mode(Normal)
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::NORMAL]);

        // receive: read_reg(CANINTF) -> RX0IF set
        spi.push_response(&[0, 0, int_flag::RX0IF]);

        // receive: READ_RX_BUFFER_0 transfer (14 bytes)
        // Standard frame id=0x100, data=[0xDE, 0xAD]
        // SIDH = (0x100 >> 3) & 0xFF = 0x20
        // SIDL = (0x100 & 0x07) << 5 = 0x00 (IDE bit=0)
        // EID8 = 0, EID0 = 0, DLC = 2
        let mut rx_buf = [0u8; 14];
        rx_buf[1] = 0x20; // SIDH
        rx_buf[2] = 0x00; // SIDL (standard, IDE=0)
        rx_buf[3] = 0x00; // EID8
        rx_buf[4] = 0x00; // EID0
        rx_buf[5] = 0x02; // DLC
        rx_buf[6] = 0xDE; // D0
        rx_buf[7] = 0xAD; // D1
        spi.push_response(&rx_buf);

        // receive: bit_modify to clear RX0IF
        spi.push_response(&[]);

        let mut drv = Mcp2515::new(spi);
        drv.init().unwrap();
        drv.set_mode(CanMode::Normal).unwrap();

        let frame = drv.receive().unwrap().unwrap();
        assert_eq!(frame.frame_type, FrameType::Standard);
        assert_eq!(frame.id, 0x100);
        assert_eq!(frame.data.len(), 2);
        assert_eq!(frame.data[0], 0xDE);
        assert_eq!(frame.data[1], 0xAD);
        assert_eq!(drv.stats().rx_frames, 1);
    }

    // -------------------------------------------------------------------
    // receive extended frame (RX0IF set)
    // -------------------------------------------------------------------

    #[test]
    fn test_mcp2515_receive_extended_frame() {
        let mut spi = MockSpi::new();
        // init
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]);
        spi.push_response(&[]);
        // set_mode(Normal)
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::NORMAL]);

        // receive: read_reg(CANINTF) -> RX0IF set
        spi.push_response(&[0, 0, int_flag::RX0IF]);

        // receive: READ_RX_BUFFER_0 transfer (14 bytes)
        // Extended frame id=0x0001_ABCD, data=[0x42]
        //
        // Encode into SIDH/SIDL/EID8/EID0:
        //   SIDH = (id >> 21) & 0xFF = (0x0001_ABCD >> 21) & 0xFF = 0x00
        //   SIDL = ((id >> 13) & 0xE0) | 0x08 | ((id >> 16) & 0x03)
        //        = ((0x0001_ABCD >> 13) & 0xE0) | 0x08 | ((0x0001_ABCD >> 16) & 0x03)
        //        = (0xD5E0 & 0xE0 = 0xC0) ... wait, let me compute step by step
        //
        // id = 0x0001_ABCD = 0b0000_0000_0000_0001_1010_1011_1100_1101
        //   bits [28:21] -> SIDH: (id >> 21) & 0xFF = 0 >> ... = 0x00
        //   bits [20:18] -> SIDL[7:5]: (id >> 13) & 0xE0 = (0x0001_ABCD >> 13) = 0xD5
        //          0xD5 & 0xE0 = 0xC0... Hmm, let me use a simpler ID.
        //
        // Use id = 0x0000_1234, data = [0x42]
        //   SIDH = (0x1234 >> 21) & 0xFF = 0x00
        //   SIDL = ((0x1234 >> 13) & 0xE0) | 0x08 | ((0x1234 >> 16) & 0x03)
        //        = (0 & 0xE0) | 0x08 | (0 & 0x03)
        //        = 0x08
        //   EID8 = (0x1234 >> 8) & 0xFF = 0x12
        //   EID0 = 0x1234 & 0xFF = 0x34
        //
        // Verify decode: is_extended = SIDL & 0x08 != 0 -> 0x08 & 0x08 = 0x08, yes
        //   id = (SIDH << 21) | ((SIDL & 0xE0) << 13) | ((SIDL & 0x03) << 16) | (EID8 << 8) | EID0
        //      = (0 << 21) | (0 << 13) | (0 << 16) | (0x12 << 8) | 0x34
        //      = 0x1234  -> correct
        let mut rx_buf = [0u8; 14];
        rx_buf[1] = 0x00; // SIDH
        rx_buf[2] = 0x08; // SIDL (IDE=1)
        rx_buf[3] = 0x12; // EID8
        rx_buf[4] = 0x34; // EID0
        rx_buf[5] = 0x01; // DLC
        rx_buf[6] = 0x42; // D0
        spi.push_response(&rx_buf);

        // receive: bit_modify to clear RX0IF
        spi.push_response(&[]);

        let mut drv = Mcp2515::new(spi);
        drv.init().unwrap();
        drv.set_mode(CanMode::Normal).unwrap();

        let frame = drv.receive().unwrap().unwrap();
        assert_eq!(frame.frame_type, FrameType::Extended);
        assert_eq!(frame.id, 0x1234);
        assert_eq!(frame.data.len(), 1);
        assert_eq!(frame.data[0], 0x42);
        assert_eq!(drv.stats().rx_frames, 1);
    }

    // -------------------------------------------------------------------
    // receive in Sleep mode should fail
    // -------------------------------------------------------------------

    #[test]
    fn test_mcp2515_receive_sleep_mode() {
        let mut spi = MockSpi::new();
        // init
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]);
        spi.push_response(&[]);
        // set_mode(Sleep)
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::SLEEP]);

        let mut drv = Mcp2515::new(spi);
        drv.init().unwrap();
        drv.set_mode(CanMode::Sleep).unwrap();
        assert_eq!(drv.receive().err(), Some(BusError::NotSupported));
    }

    // -------------------------------------------------------------------
    // transmit in ListenOnly mode should fail
    // -------------------------------------------------------------------

    #[test]
    fn test_mcp2515_transmit_listen_only() {
        let mut spi = MockSpi::new();
        // init
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]);
        spi.push_response(&[]);
        // set_mode(ListenOnly)
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::LISTEN_ONLY]);

        let mut drv = Mcp2515::new(spi);
        drv.init().unwrap();
        drv.set_mode(CanMode::ListenOnly).unwrap();

        let frame = CanFrame::new_standard(0x100, &[0x01]).unwrap();
        assert_eq!(drv.transmit(&frame).err(), Some(BusError::NotSupported));
    }

    // -------------------------------------------------------------------
    // receive in ListenOnly mode should succeed (not Config/Sleep)
    // -------------------------------------------------------------------

    #[test]
    fn test_mcp2515_receive_listen_only_empty() {
        let mut spi = MockSpi::new();
        // init
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]);
        spi.push_response(&[]);
        // set_mode(ListenOnly)
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::LISTEN_ONLY]);
        // receive: read CANINTF -> no RX0IF
        spi.push_response(&[0, 0, 0x00]);

        let mut drv = Mcp2515::new(spi);
        drv.init().unwrap();
        drv.set_mode(CanMode::ListenOnly).unwrap();
        assert!(drv.receive().unwrap().is_none());
    }

    // -------------------------------------------------------------------
    // receive in Loopback mode should succeed
    // -------------------------------------------------------------------

    #[test]
    fn test_mcp2515_receive_loopback_empty() {
        let mut spi = MockSpi::new();
        // init
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]);
        spi.push_response(&[]);
        // set_mode(Loopback)
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::LOOPBACK]);
        // receive: read CANINTF -> no RX0IF
        spi.push_response(&[0, 0, 0x00]);

        let mut drv = Mcp2515::new(spi);
        drv.init().unwrap();
        drv.set_mode(CanMode::Loopback).unwrap();
        assert!(drv.receive().unwrap().is_none());
    }

    // -------------------------------------------------------------------
    // set_mode_bits fails when verify returns wrong mode (Timeout)
    // -------------------------------------------------------------------

    #[test]
    fn test_mcp2515_set_mode_timeout() {
        let mut spi = MockSpi::new();
        // init
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]);
        spi.push_response(&[]);
        // set_mode(Normal): bit_modify succeeds but verify returns wrong mode
        spi.push_response(&[]); // bit_modify
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]); // verify returns CONFIG instead of NORMAL

        let mut drv = Mcp2515::new(spi);
        drv.init().unwrap();
        assert_eq!(drv.set_mode(CanMode::Normal).err(), Some(BusError::Timeout));
    }

    // -------------------------------------------------------------------
    // set_bit_timing in wrong mode should fail
    // -------------------------------------------------------------------

    #[test]
    fn test_mcp2515_set_bit_timing_wrong_mode() {
        let mut spi = MockSpi::new();
        // init
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]);
        spi.push_response(&[]);
        // set_mode(Normal)
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::NORMAL]);

        let mut drv = Mcp2515::new(spi);
        drv.init().unwrap();
        drv.set_mode(CanMode::Normal).unwrap();

        let timing = CanBitTiming::standard_500kbps();
        assert_eq!(
            drv.set_bit_timing(&timing).err(),
            Some(BusError::NotSupported)
        );
    }

    // -------------------------------------------------------------------
    // init fails when CANSTAT does not report Configuration mode
    // -------------------------------------------------------------------

    #[test]
    fn test_mcp2515_init_wrong_mode() {
        let mut spi = MockSpi::new();
        // reset
        spi.push_response(&[]);
        // read_mode_bits: returns NORMAL instead of CONFIGURATION
        spi.push_response(&[0, 0, mode_bits::NORMAL]);

        let mut drv = Mcp2515::new(spi);
        assert_eq!(drv.init().err(), Some(BusError::NotSupported));
        assert!(!drv.initialized);
    }

    // -------------------------------------------------------------------
    // transmit and receive multiple frames to verify stats
    // -------------------------------------------------------------------

    #[test]
    fn test_mcp2515_stats_after_tx_rx() {
        let mut spi = MockSpi::new();
        // init
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::CONFIGURATION]);
        spi.push_response(&[]);
        // set_mode(Normal)
        spi.push_response(&[]);
        spi.push_response(&[0, 0, mode_bits::NORMAL]);

        // First transmit
        spi.push_response(&[]); // LOAD_TX_BUFFER_0
        spi.push_response(&[]); // RTS_TXB0

        // Second transmit
        spi.push_response(&[]); // LOAD_TX_BUFFER_0
        spi.push_response(&[]); // RTS_TXB0

        // Receive a standard frame
        spi.push_response(&[0, 0, int_flag::RX0IF]); // CANINTF
        let mut rx_buf = [0u8; 14];
        rx_buf[1] = 0x20; // SIDH for id=0x100
        rx_buf[2] = 0x00; // SIDL
        rx_buf[5] = 0x01; // DLC=1
        rx_buf[6] = 0xFF; // D0
        spi.push_response(&rx_buf); // READ_RX_BUFFER_0
        spi.push_response(&[]); // bit_modify clear RX0IF

        let mut drv = Mcp2515::new(spi);
        drv.init().unwrap();
        drv.set_mode(CanMode::Normal).unwrap();

        let frame1 = CanFrame::new_standard(0x100, &[0xAA]).unwrap();
        drv.transmit(&frame1).unwrap();
        let frame2 = CanFrame::new_standard(0x200, &[0xBB]).unwrap();
        drv.transmit(&frame2).unwrap();

        let received = drv.receive().unwrap().unwrap();
        assert_eq!(received.id, 0x100);

        assert_eq!(drv.stats().tx_frames, 2);
        assert_eq!(drv.stats().rx_frames, 1);
    }
}
