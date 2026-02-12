// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Xilinx AXI IIC FPGA I2C controller driver.
//!
//! Implements the [`I2cController`] trait for the Xilinx AXI IIC soft-IP core
//! (PG090), used on Zynq UltraScale+ and similar SoC FPGAs.
//!
//! # Hardware interface
//!
//! The AXI IIC IP is accessed through AXI4-Lite memory-mapped registers
//! via the [`FpgaFabric`] trait. The controller supports standard, fast,
//! and dynamic multi-master operation.
//!
//! # References
//!
//! - Xilinx PG090: AXI IIC Bus Interface v2.1

#![allow(dead_code)]

use smallaios_kernel::hal::{FpgaFabric, HalError, I2cConfig, I2cController, I2cSpeed};

// ---------------------------------------------------------------------------
// AXI IIC register offsets (from PG090)
// ---------------------------------------------------------------------------

/// GIE — Global Interrupt Enable Register.
const GIE: u32 = 0x01C;
/// ISR — Interrupt Status Register.
const ISR: u32 = 0x020;
/// IER — Interrupt Enable Register.
const IER: u32 = 0x028;
/// SOFTR — Soft Reset Register.
const SOFTR: u32 = 0x040;
/// CR — Control Register.
const CR: u32 = 0x100;
/// SR — Status Register.
const SR: u32 = 0x104;
/// TX_FIFO — Transmit FIFO Register.
const TX_FIFO: u32 = 0x108;
/// RX_FIFO — Receive FIFO Register.
const RX_FIFO: u32 = 0x10C;
/// ADR — Slave Address Register.
const ADR: u32 = 0x110;
/// TX_FIFO_OCY — Transmit FIFO Occupancy Register.
const TX_FIFO_OCY: u32 = 0x114;
/// RX_FIFO_OCY — Receive FIFO Occupancy Register.
const RX_FIFO_OCY: u32 = 0x118;
/// TEN_ADR — 10-bit Slave Address Register.
const TEN_ADR: u32 = 0x11C;
/// RX_FIFO_PIRQ — Receive FIFO Programmable Depth Interrupt Register.
const RX_FIFO_PIRQ: u32 = 0x120;
/// GPO — General Purpose Output Register.
const GPO: u32 = 0x124;
/// TSUSTA — Timing Parameter (setup time for START).
const TSUSTA: u32 = 0x128;
/// TSUSTO — Timing Parameter (setup time for STOP).
const TSUSTO: u32 = 0x12C;
/// THDSTA — Timing Parameter (hold time for START).
const THDSTA: u32 = 0x130;
/// TSUDAT — Timing Parameter (data setup time).
const TSUDAT: u32 = 0x134;
/// TBUF — Timing Parameter (bus free time).
const TBUF: u32 = 0x138;
/// THIGH — Timing Parameter (SCL high time).
const THIGH: u32 = 0x13C;
/// TLOW — Timing Parameter (SCL low time).
const TLOW: u32 = 0x140;
/// THDDAT — Timing Parameter (data hold time).
const THDDAT: u32 = 0x144;

// ---------------------------------------------------------------------------
// Control Register (CR) bits
// ---------------------------------------------------------------------------

/// General call enable.
const CR_GC_EN: u32 = 1 << 6;
/// Repeated START.
const CR_RSTA: u32 = 1 << 5;
/// TX ACK enable (0 = ACK, 1 = NACK).
const CR_TXAK: u32 = 1 << 4;
/// Transmit mode (1 = TX, 0 = RX).
const CR_TX: u32 = 1 << 3;
/// Master/slave mode select (1 = master).
const CR_MSMS: u32 = 1 << 2;
/// TX FIFO reset.
const CR_TX_FIFO_RST: u32 = 1 << 1;
/// I2C core enable.
const CR_EN: u32 = 1 << 0;

// ---------------------------------------------------------------------------
// Status Register (SR) bits
// ---------------------------------------------------------------------------

/// TX FIFO empty.
const SR_TX_FIFO_EMPTY: u32 = 1 << 7;
/// RX FIFO empty.
const SR_RX_FIFO_EMPTY: u32 = 1 << 6;
/// RX FIFO full.
const SR_RX_FIFO_FULL: u32 = 1 << 5;
/// TX FIFO full.
const SR_TX_FIFO_FULL: u32 = 1 << 4;
/// Slave Read/Write (SRW).
const SR_SRW: u32 = 1 << 3;
/// Bus busy.
const SR_BB: u32 = 1 << 2;
/// Addressed as slave.
const SR_AAS: u32 = 1 << 1;
/// ABGC — addressed by general call.
const SR_ABGC: u32 = 1 << 0;

// ---------------------------------------------------------------------------
// Interrupt bits (ISR/IER)
// ---------------------------------------------------------------------------

/// Arbitration lost.
const INT_ARB_LOST: u32 = 1 << 0;
/// TX error (NACK during address or data).
const INT_TX_ERR: u32 = 1 << 1;
/// TX FIFO empty.
const INT_TX_EMPTY: u32 = 1 << 2;
/// RX FIFO full.
const INT_RX_FULL: u32 = 1 << 3;
/// Bus not busy.
const INT_BNB: u32 = 1 << 4;
/// Addressed as slave.
const INT_AAS: u32 = 1 << 5;
/// Not addressed as slave.
const INT_NAAS: u32 = 1 << 6;
/// TX FIFO half-empty.
const INT_TX_HALF_EMPTY: u32 = 1 << 7;

/// Soft reset key.
const SOFTR_KEY: u32 = 0x0000_000A;

// ---------------------------------------------------------------------------
// TX FIFO data bits
// ---------------------------------------------------------------------------

/// Start bit in TX FIFO data word.
const TX_FIFO_START: u32 = 1 << 8;
/// Stop bit in TX FIFO data word.
const TX_FIFO_STOP: u32 = 1 << 9;

/// FIFO depth for the standard AXI IIC configuration.
const FIFO_DEPTH: usize = 16;

// ---------------------------------------------------------------------------
// Driver implementation
// ---------------------------------------------------------------------------

/// Xilinx AXI IIC FPGA I2C controller driver.
///
/// Generic over an [`FpgaFabric`] implementation that provides register
/// access to the AXI4-Lite address space.
pub struct AxiIic<F: FpgaFabric> {
    /// FPGA fabric accessor.
    fabric: F,
    /// AXI base address of the IIC IP core.
    base_addr: u64,
    /// Whether the controller has been initialized.
    configured: bool,
    /// Stored configuration.
    config: Option<I2cConfig>,
}

impl<F: FpgaFabric> AxiIic<F> {
    /// Create a new AXI IIC driver.
    pub fn new(fabric: F, base_addr: u64) -> Self {
        Self {
            fabric,
            base_addr,
            configured: false,
            config: None,
        }
    }

    /// Return the AXI base address.
    pub fn base_addr(&self) -> u64 {
        self.base_addr
    }

    /// Return whether the controller is configured.
    pub fn is_configured(&self) -> bool {
        self.configured
    }

    /// Read a register.
    fn read_reg(&self, offset: u32) -> Result<u32, HalError> {
        self.fabric.read_reg(self.base_addr, offset)
    }

    /// Write a register.
    fn write_reg(&mut self, offset: u32, value: u32) -> Result<(), HalError> {
        self.fabric.write_reg(self.base_addr, offset, value)
    }

    /// Perform a soft reset of the IIC core.
    fn soft_reset(&mut self) -> Result<(), HalError> {
        self.write_reg(SOFTR, SOFTR_KEY)
    }

    /// Wait for the TX FIFO to empty, indicating the bus transaction completed.
    fn wait_tx_empty(&self) -> Result<(), HalError> {
        let sr = self.read_reg(SR)?;
        if sr & SR_TX_FIFO_EMPTY == 0 {
            return Err(HalError::Timeout);
        }
        Ok(())
    }

    /// Check for bus errors (arbitration lost, TX error).
    fn check_errors(&self) -> Result<(), HalError> {
        let isr = self.read_reg(ISR)?;
        if isr & INT_ARB_LOST != 0 {
            return Err(HalError::ArbitrationLost);
        }
        if isr & INT_TX_ERR != 0 {
            return Err(HalError::NackReceived);
        }
        Ok(())
    }

    /// Compute AXI IIC timing parameters for the given speed and input clock.
    fn compute_timing(speed: I2cSpeed) -> (u32, u32) {
        // Returns (thigh, tlow) as register values.
        // These are placeholder values scaled for a 100 MHz AXI clock.
        match speed {
            I2cSpeed::Standard => (500, 500), // 100 kHz
            I2cSpeed::Fast => (125, 250),     // 400 kHz
            I2cSpeed::FastPlus => (50, 100),  // 1 MHz
        }
    }
}

impl<F: FpgaFabric> I2cController for AxiIic<F> {
    fn init(&mut self, config: I2cConfig) -> Result<(), HalError> {
        if config.timeout_us == 0 {
            return Err(HalError::InvalidConfig);
        }

        // Soft reset.
        self.soft_reset()?;

        // Disable interrupts.
        self.write_reg(GIE, 0)?;
        self.write_reg(IER, 0)?;

        // Clear any pending interrupt status.
        let isr = self.read_reg(ISR)?;
        self.write_reg(ISR, isr)?;

        // Set timing parameters.
        let (thigh, tlow) = Self::compute_timing(config.speed);
        self.write_reg(THIGH, thigh)?;
        self.write_reg(TLOW, tlow)?;

        // Set RX FIFO depth interrupt level.
        self.write_reg(RX_FIFO_PIRQ, (FIFO_DEPTH - 1) as u32)?;

        // Enable I2C core.
        self.write_reg(CR, CR_EN | CR_TX_FIFO_RST)?;
        // Release TX FIFO reset.
        self.write_reg(CR, CR_EN)?;

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

        // Reset TX FIFO.
        self.write_reg(CR, CR_EN | CR_TX_FIFO_RST)?;
        self.write_reg(CR, CR_EN)?;

        // Write address byte with START.
        let addr_byte = (addr as u32) << 1; // Write mode (R/W=0)
        self.write_reg(TX_FIFO, addr_byte | TX_FIFO_START)?;

        // Write data bytes; last byte gets STOP.
        for (i, &byte) in data.iter().enumerate() {
            let mut word = byte as u32;
            if i == data.len() - 1 {
                word |= TX_FIFO_STOP;
            }
            self.write_reg(TX_FIFO, word)?;
        }

        // Set master transmit mode.
        self.write_reg(CR, CR_EN | CR_MSMS | CR_TX)?;

        // Wait for completion.
        self.wait_tx_empty()?;
        self.check_errors()
    }

    fn read(&mut self, addr: u16, buf: &mut [u8]) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        let config = self.config.as_ref().unwrap();
        crate::i2c::validate_addr(addr, config.address_mode)?;

        if buf.is_empty() {
            return Ok(());
        }

        // Reset TX FIFO.
        self.write_reg(CR, CR_EN | CR_TX_FIFO_RST)?;
        self.write_reg(CR, CR_EN)?;

        // Write address byte with START and read bit.
        let addr_byte = ((addr as u32) << 1) | 1;
        self.write_reg(TX_FIFO, addr_byte | TX_FIFO_START)?;

        // Set RX FIFO depth for the number of bytes to receive.
        self.write_reg(RX_FIFO_PIRQ, (buf.len().saturating_sub(1)) as u32)?;

        // Set master receive mode.
        self.write_reg(CR, CR_EN | CR_MSMS)?;

        // Read data from RX FIFO.
        for byte in buf.iter_mut() {
            let val = self.read_reg(RX_FIFO)?;
            *byte = (val & 0xFF) as u8;
        }

        self.check_errors()
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

        // Write phase.
        self.write_reg(CR, CR_EN | CR_TX_FIFO_RST)?;
        self.write_reg(CR, CR_EN)?;

        let addr_byte_w = (addr as u32) << 1;
        self.write_reg(TX_FIFO, addr_byte_w | TX_FIFO_START)?;

        for &byte in write_data {
            self.write_reg(TX_FIFO, byte as u32)?;
        }

        self.write_reg(CR, CR_EN | CR_MSMS | CR_TX)?;
        self.wait_tx_empty()?;

        // Repeated START for read phase.
        let addr_byte_r = ((addr as u32) << 1) | 1;
        self.write_reg(TX_FIFO, addr_byte_r | TX_FIFO_START)?;
        self.write_reg(RX_FIFO_PIRQ, (read_buf.len().saturating_sub(1)) as u32)?;
        self.write_reg(CR, CR_EN | CR_MSMS | CR_RSTA)?;

        for byte in read_buf.iter_mut() {
            let val = self.read_reg(RX_FIFO)?;
            *byte = (val & 0xFF) as u8;
        }

        self.check_errors()
    }

    fn bus_recovery(&mut self) -> Result<(), HalError> {
        // AXI IIC: perform a soft reset to recover the bus.
        self.soft_reset()?;
        // Re-init if we were configured.
        if let Some(config) = self.config {
            self.configured = false;
            self.init(config)?;
        }
        Ok(())
    }

    fn reset(&mut self) -> Result<(), HalError> {
        self.soft_reset()?;
        self.configured = false;
        self.config = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallaios_kernel::hal::{DmaDescriptor, DmaToken, I2cAddressMode};

    /// Mock FPGA fabric backed by a register array.
    struct MockFpgaFabric {
        regs: [u32; 128],
    }

    impl MockFpgaFabric {
        fn new() -> Self {
            Self { regs: [0u32; 128] }
        }
    }

    impl FpgaFabric for MockFpgaFabric {
        fn read_reg(&self, _base_addr: u64, offset: u32) -> Result<u32, HalError> {
            let idx = (offset / 4) as usize;
            if idx < self.regs.len() {
                Ok(self.regs[idx])
            } else {
                Err(HalError::OutOfRange)
            }
        }

        fn write_reg(&mut self, _base_addr: u64, offset: u32, value: u32) -> Result<(), HalError> {
            let idx = (offset / 4) as usize;
            if idx < self.regs.len() {
                self.regs[idx] = value;
                Ok(())
            } else {
                Err(HalError::OutOfRange)
            }
        }

        fn dma_start(&mut self, _desc: DmaDescriptor) -> Result<DmaToken, HalError> {
            Err(HalError::NotSupported)
        }

        fn dma_poll(&self, token: DmaToken) -> Result<DmaToken, HalError> {
            Ok(token)
        }

        fn dma_irq_ack(&mut self) -> Result<u8, HalError> {
            Ok(0)
        }
    }

    fn make_config() -> I2cConfig {
        I2cConfig {
            speed: I2cSpeed::Standard,
            address_mode: I2cAddressMode::SevenBit,
            clock_stretching: false,
            timeout_us: 10_000,
        }
    }

    #[test]
    fn test_axi_iic_new() {
        let fab = MockFpgaFabric::new();
        let drv = AxiIic::new(fab, 0x4060_0000);
        assert_eq!(drv.base_addr(), 0x4060_0000);
        assert!(!drv.is_configured());
    }

    #[test]
    fn test_axi_iic_init() {
        let mut fab = MockFpgaFabric::new();
        // Pre-set SR with TX_FIFO_EMPTY so wait_tx_empty passes.
        fab.regs[(SR / 4) as usize] = SR_TX_FIFO_EMPTY;

        let mut drv = AxiIic::new(fab, 0x4060_0000);
        assert!(drv.init(make_config()).is_ok());
        assert!(drv.is_configured());
    }

    #[test]
    fn test_axi_iic_init_zero_timeout() {
        let fab = MockFpgaFabric::new();
        let mut drv = AxiIic::new(fab, 0x4060_0000);
        let mut config = make_config();
        config.timeout_us = 0;
        assert_eq!(drv.init(config), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_axi_iic_write_not_configured() {
        let fab = MockFpgaFabric::new();
        let mut drv = AxiIic::new(fab, 0x4060_0000);
        assert_eq!(drv.write(0x50, &[0x00]), Err(HalError::InitFailed));
    }

    #[test]
    fn test_axi_iic_read_not_configured() {
        let fab = MockFpgaFabric::new();
        let mut drv = AxiIic::new(fab, 0x4060_0000);
        let mut buf = [0u8; 4];
        assert_eq!(drv.read(0x50, &mut buf), Err(HalError::InitFailed));
    }

    #[test]
    fn test_axi_iic_write_initialized() {
        let mut fab = MockFpgaFabric::new();
        fab.regs[(SR / 4) as usize] = SR_TX_FIFO_EMPTY;

        let mut drv = AxiIic::new(fab, 0x4060_0000);
        drv.init(make_config()).unwrap();
        // Write should succeed (mock returns TX_FIFO_EMPTY, no errors in ISR).
        assert!(drv.write(0x50, &[0xAB, 0xCD]).is_ok());
    }

    #[test]
    fn test_axi_iic_read_initialized() {
        let mut fab = MockFpgaFabric::new();
        fab.regs[(SR / 4) as usize] = SR_TX_FIFO_EMPTY;

        let mut drv = AxiIic::new(fab, 0x4060_0000);
        drv.init(make_config()).unwrap();
        let mut buf = [0u8; 2];
        assert!(drv.read(0x50, &mut buf).is_ok());
    }

    #[test]
    fn test_axi_iic_write_invalid_addr() {
        let mut fab = MockFpgaFabric::new();
        fab.regs[(SR / 4) as usize] = SR_TX_FIFO_EMPTY;

        let mut drv = AxiIic::new(fab, 0x4060_0000);
        drv.init(make_config()).unwrap();
        // 0x00 is reserved in 7-bit mode.
        assert_eq!(drv.write(0x00, &[0xAB]), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_axi_iic_reset() {
        let mut fab = MockFpgaFabric::new();
        fab.regs[(SR / 4) as usize] = SR_TX_FIFO_EMPTY;

        let mut drv = AxiIic::new(fab, 0x4060_0000);
        drv.init(make_config()).unwrap();
        assert!(drv.reset().is_ok());
        assert!(!drv.is_configured());
    }

    #[test]
    fn test_axi_iic_bus_recovery() {
        let mut fab = MockFpgaFabric::new();
        fab.regs[(SR / 4) as usize] = SR_TX_FIFO_EMPTY;

        let mut drv = AxiIic::new(fab, 0x4060_0000);
        drv.init(make_config()).unwrap();
        assert!(drv.bus_recovery().is_ok());
        assert!(drv.is_configured());
    }

    #[test]
    fn test_compute_timing_standard() {
        let (thigh, tlow) = AxiIic::<MockFpgaFabric>::compute_timing(I2cSpeed::Standard);
        assert_eq!(thigh, 500);
        assert_eq!(tlow, 500);
    }

    #[test]
    fn test_compute_timing_fast() {
        let (thigh, tlow) = AxiIic::<MockFpgaFabric>::compute_timing(I2cSpeed::Fast);
        assert!(thigh < tlow); // Fast mode: tHIGH < tLOW
    }
}
