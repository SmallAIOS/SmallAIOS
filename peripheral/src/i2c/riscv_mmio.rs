// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! RISC-V SiFive/OpenCores I2C MMIO controller driver.
//!
//! Implements the [`I2cController`] trait for RISC-V platforms using the
//! OpenCores I2C controller IP, as integrated in SiFive FU540/FU740 SoCs.
//!
//! # Hardware interface
//!
//! The OpenCores I2C controller uses a simple 5-register interface.
//! The prescaler divides the input clock to produce the desired SCL frequency.
//!
//! # References
//!
//! - OpenCores I2C Controller Core Specification
//! - SiFive FU740-C000 Manual

#![allow(dead_code)]

use smallaios_kernel::hal::{HalError, I2cConfig, I2cController, I2cSpeed};

// ---------------------------------------------------------------------------
// OpenCores I2C register offsets
// ---------------------------------------------------------------------------

/// PRER_LO — Clock Prescale Register (low byte).
const PRER_LO: u32 = 0x00;
/// PRER_HI — Clock Prescale Register (high byte).
const PRER_HI: u32 = 0x04;
/// CTR — Control Register.
const CTR: u32 = 0x08;
/// TXR — Transmit Register (write) / RXR — Receive Register (read).
const TXR_RXR: u32 = 0x0C;
/// CR — Command Register (write) / SR — Status Register (read).
const CR_SR: u32 = 0x10;

// ---------------------------------------------------------------------------
// Control register (CTR) bits
// ---------------------------------------------------------------------------

/// Core enable bit.
const CTR_EN: u32 = 1 << 7;
/// Interrupt enable bit.
const CTR_IEN: u32 = 1 << 6;

// ---------------------------------------------------------------------------
// Command register (CR) bits
// ---------------------------------------------------------------------------

/// Generate START condition.
const CR_STA: u32 = 1 << 7;
/// Generate STOP condition.
const CR_STO: u32 = 1 << 6;
/// Read from slave.
const CR_RD: u32 = 1 << 5;
/// Write to slave.
const CR_WR: u32 = 1 << 4;
/// Send NACK (when reading, to signal end of read).
const CR_NACK: u32 = 1 << 3;
/// Clear interrupt flag.
const CR_IACK: u32 = 1 << 0;

// ---------------------------------------------------------------------------
// Status register (SR) bits
// ---------------------------------------------------------------------------

/// Received ACK from slave (0 = ACK, 1 = NACK).
const SR_RXACK: u32 = 1 << 7;
/// I2C bus busy.
const SR_BUSY: u32 = 1 << 6;
/// Arbitration lost.
const SR_AL: u32 = 1 << 5;
/// Transfer in progress.
const SR_TIP: u32 = 1 << 1;
/// Interrupt flag.
const SR_IF: u32 = 1 << 0;

// ---------------------------------------------------------------------------
// Default constants
// ---------------------------------------------------------------------------

/// Default SiFive I2C input clock: 500 MHz (typical PRCI output).
const DEFAULT_INPUT_CLOCK_HZ: u32 = 500_000_000;

// ---------------------------------------------------------------------------
// Driver implementation
// ---------------------------------------------------------------------------

/// OpenCores I2C controller driver for RISC-V platforms.
///
/// Provides I2C master mode with standard and fast speed support.
/// The controller uses a 16-bit prescaler to divide the input clock.
pub struct OpenCoresI2c {
    /// MMIO base address.
    base_addr: u64,
    /// IRQ number.
    irq: u32,
    /// Whether the controller has been initialized.
    configured: bool,
    /// Stored configuration.
    config: Option<I2cConfig>,
    /// Input reference clock in Hz.
    input_clock_hz: u32,
}

impl OpenCoresI2c {
    /// Create a new OpenCores I2C driver at the given MMIO base address.
    pub fn new(base_addr: u64, irq: u32) -> Self {
        Self {
            base_addr,
            irq,
            configured: false,
            config: None,
            input_clock_hz: DEFAULT_INPUT_CLOCK_HZ,
        }
    }

    /// Create a new driver with a custom input clock frequency.
    pub fn with_clock(base_addr: u64, irq: u32, input_clock_hz: u32) -> Self {
        Self {
            base_addr,
            irq,
            configured: false,
            config: None,
            input_clock_hz,
        }
    }

    /// Return the MMIO base address.
    pub fn base_addr(&self) -> u64 {
        self.base_addr
    }

    /// Return whether the controller is configured.
    pub fn is_configured(&self) -> bool {
        self.configured
    }

    /// Compute the 16-bit prescaler value for the desired I2C speed.
    ///
    /// prescale = (input_clock / (5 * SCL_freq)) - 1
    fn compute_prescaler(&self, speed: I2cSpeed) -> u16 {
        let scl_freq = match speed {
            I2cSpeed::Standard => 100_000u32,
            I2cSpeed::Fast => 400_000u32,
            I2cSpeed::FastPlus => 1_000_000u32,
        };
        if self.input_clock_hz == 0 || scl_freq == 0 {
            return 0xFFFF;
        }
        let prescale = (self.input_clock_hz / (5 * scl_freq)).saturating_sub(1);
        prescale.min(0xFFFF) as u16
    }

    /// Read a 32-bit register (stub: returns MmioError on host).
    fn read_reg(&self, _offset: u32) -> Result<u32, HalError> {
        Err(HalError::MmioError)
    }

    /// Write a 32-bit register (stub: returns MmioError on host).
    fn write_reg(&mut self, _offset: u32, _value: u32) -> Result<(), HalError> {
        Err(HalError::MmioError)
    }

    /// Wait for a transfer to complete (TIP bit clears).
    fn wait_tip(&self) -> Result<u32, HalError> {
        // In real hardware: poll SR until TIP clears.
        let sr = self.read_reg(CR_SR)?;
        if sr & SR_TIP != 0 {
            return Err(HalError::Timeout);
        }
        Ok(sr)
    }

    /// Send a START condition with the address byte.
    fn send_start(&mut self, addr_byte: u8) -> Result<(), HalError> {
        self.write_reg(TXR_RXR, addr_byte as u32)?;
        self.write_reg(CR_SR, CR_STA | CR_WR)?;
        let sr = self.wait_tip()?;
        if sr & SR_AL != 0 {
            return Err(HalError::ArbitrationLost);
        }
        if sr & SR_RXACK != 0 {
            // NACK received — send STOP.
            self.write_reg(CR_SR, CR_STO)?;
            return Err(HalError::NackReceived);
        }
        Ok(())
    }

    /// Write a data byte (no START, no STOP).
    fn send_byte(&mut self, byte: u8) -> Result<(), HalError> {
        self.write_reg(TXR_RXR, byte as u32)?;
        self.write_reg(CR_SR, CR_WR)?;
        let sr = self.wait_tip()?;
        if sr & SR_RXACK != 0 {
            return Err(HalError::NackReceived);
        }
        Ok(())
    }

    /// Read a byte, sending ACK or NACK.
    fn recv_byte(&mut self, last: bool) -> Result<u8, HalError> {
        let cmd = if last { CR_RD | CR_NACK } else { CR_RD };
        self.write_reg(CR_SR, cmd)?;
        self.wait_tip()?;
        let val = self.read_reg(TXR_RXR)?;
        Ok((val & 0xFF) as u8)
    }

    /// Send a STOP condition.
    fn send_stop(&mut self) -> Result<(), HalError> {
        self.write_reg(CR_SR, CR_STO)
    }

    /// Build the 7-bit address byte (addr << 1 | rw).
    fn addr_byte_7bit(addr: u16, read: bool) -> u8 {
        ((addr as u8) << 1) | (read as u8)
    }
}

impl I2cController for OpenCoresI2c {
    fn init(&mut self, config: I2cConfig) -> Result<(), HalError> {
        if config.timeout_us == 0 {
            return Err(HalError::InvalidConfig);
        }

        // Disable core.
        self.write_reg(CTR, 0)?;

        // Program prescaler.
        let prescale = self.compute_prescaler(config.speed);
        self.write_reg(PRER_LO, (prescale & 0xFF) as u32)?;
        self.write_reg(PRER_HI, ((prescale >> 8) & 0xFF) as u32)?;

        // Enable core.
        self.write_reg(CTR, CTR_EN)?;

        self.config = Some(config);
        self.configured = true;
        Ok(())
    }

    fn write(&mut self, addr: u16, data: &[u8]) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        let config = self.config.as_ref().unwrap();
        crate::i2c::validate_addr(addr, config.address_mode)?;

        // START + address (write mode).
        self.send_start(Self::addr_byte_7bit(addr, false))?;

        // Write data bytes.
        for &byte in data {
            self.send_byte(byte)?;
        }

        // STOP.
        self.send_stop()?;
        Ok(())
    }

    fn read(&mut self, addr: u16, buf: &mut [u8]) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        let config = self.config.as_ref().unwrap();
        crate::i2c::validate_addr(addr, config.address_mode)?;

        // START + address (read mode).
        self.send_start(Self::addr_byte_7bit(addr, true))?;

        // Read bytes.
        for i in 0..buf.len() {
            let last = i == buf.len() - 1;
            buf[i] = self.recv_byte(last)?;
        }

        // STOP.
        self.send_stop()?;
        Ok(())
    }

    fn write_read(
        &mut self,
        addr: u16,
        write_data: &[u8],
        read_buf: &mut [u8],
    ) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        let config = self.config.as_ref().unwrap();
        crate::i2c::validate_addr(addr, config.address_mode)?;

        // START + address (write mode).
        self.send_start(Self::addr_byte_7bit(addr, false))?;
        for &byte in write_data {
            self.send_byte(byte)?;
        }

        // Repeated START + address (read mode).
        self.send_start(Self::addr_byte_7bit(addr, true))?;
        for i in 0..read_buf.len() {
            let last = i == read_buf.len() - 1;
            read_buf[i] = self.recv_byte(last)?;
        }

        self.send_stop()?;
        Ok(())
    }

    fn bus_recovery(&mut self) -> Result<(), HalError> {
        // OpenCores: generate 9 clocks by performing dummy reads.
        for _ in 0..crate::i2c::bus_recovery_clock_count() {
            let _ = self.write_reg(CR_SR, CR_RD | CR_NACK);
            let _ = self.wait_tip();
        }
        self.send_stop()
    }

    fn reset(&mut self) -> Result<(), HalError> {
        self.write_reg(CTR, 0)?;
        self.configured = false;
        self.config = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallaios_kernel::hal::I2cAddressMode;

    #[test]
    fn test_opencores_new() {
        let drv = OpenCoresI2c::new(0x1002_0000, 52);
        assert_eq!(drv.base_addr(), 0x1002_0000);
        assert!(!drv.is_configured());
    }

    #[test]
    fn test_compute_prescaler_standard() {
        let drv = OpenCoresI2c::with_clock(0, 0, 100_000_000);
        let prescale = drv.compute_prescaler(I2cSpeed::Standard);
        // 100_000_000 / (5 * 100_000) - 1 = 199
        assert_eq!(prescale, 199);
    }

    #[test]
    fn test_compute_prescaler_fast() {
        let drv = OpenCoresI2c::with_clock(0, 0, 100_000_000);
        let prescale = drv.compute_prescaler(I2cSpeed::Fast);
        // 100_000_000 / (5 * 400_000) - 1 = 49
        assert_eq!(prescale, 49);
    }

    #[test]
    fn test_compute_prescaler_fast_plus() {
        let drv = OpenCoresI2c::with_clock(0, 0, 100_000_000);
        let prescale = drv.compute_prescaler(I2cSpeed::FastPlus);
        // 100_000_000 / (5 * 1_000_000) - 1 = 19
        assert_eq!(prescale, 19);
    }

    #[test]
    fn test_compute_prescaler_zero_clock() {
        let drv = OpenCoresI2c::with_clock(0, 0, 0);
        let prescale = drv.compute_prescaler(I2cSpeed::Standard);
        assert_eq!(prescale, 0xFFFF);
    }

    #[test]
    fn test_addr_byte_7bit_write() {
        assert_eq!(OpenCoresI2c::addr_byte_7bit(0x50, false), 0xA0);
    }

    #[test]
    fn test_addr_byte_7bit_read() {
        assert_eq!(OpenCoresI2c::addr_byte_7bit(0x50, true), 0xA1);
    }

    #[test]
    fn test_init_returns_mmio_error() {
        let mut drv = OpenCoresI2c::new(0x1002_0000, 52);
        let config = I2cConfig {
            speed: I2cSpeed::Standard,
            address_mode: I2cAddressMode::SevenBit,
            clock_stretching: false,
            timeout_us: 10_000,
        };
        assert_eq!(drv.init(config), Err(HalError::MmioError));
    }

    #[test]
    fn test_init_zero_timeout() {
        let mut drv = OpenCoresI2c::new(0x1002_0000, 52);
        let config = I2cConfig {
            speed: I2cSpeed::Standard,
            address_mode: I2cAddressMode::SevenBit,
            clock_stretching: false,
            timeout_us: 0,
        };
        assert_eq!(drv.init(config), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_write_not_configured() {
        let mut drv = OpenCoresI2c::new(0x1002_0000, 52);
        assert_eq!(drv.write(0x50, &[0x00]), Err(HalError::InitFailed));
    }

    #[test]
    fn test_read_not_configured() {
        let mut drv = OpenCoresI2c::new(0x1002_0000, 52);
        let mut buf = [0u8; 4];
        assert_eq!(drv.read(0x50, &mut buf), Err(HalError::InitFailed));
    }

    #[test]
    fn test_reset_returns_mmio_error() {
        let mut drv = OpenCoresI2c::new(0x1002_0000, 52);
        assert_eq!(drv.reset(), Err(HalError::MmioError));
    }
}
