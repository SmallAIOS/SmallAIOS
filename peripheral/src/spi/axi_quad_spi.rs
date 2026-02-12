// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Xilinx AXI Quad SPI FPGA controller driver.
//!
//! Implements the [`SpiController`] trait for the Xilinx AXI Quad SPI
//! soft-IP core (PG153), used on Zynq UltraScale+ and similar SoC FPGAs.
//!
//! # References
//!
//! - Xilinx PG153: AXI Quad SPI v3.2

#![allow(dead_code)]

use smallaios_kernel::hal::{DmaToken, FpgaFabric, HalError, SpiConfig, SpiController};

// ---------------------------------------------------------------------------
// AXI Quad SPI register offsets (from PG153)
// ---------------------------------------------------------------------------

/// SRR — Software Reset Register.
const SRR: u32 = 0x40;
/// SPICR — SPI Control Register.
const SPICR: u32 = 0x60;
/// SPISR — SPI Status Register.
const SPISR: u32 = 0x64;
/// SPI_DTR — SPI Data Transmit Register.
const SPI_DTR: u32 = 0x68;
/// SPI_DRR — SPI Data Receive Register.
const SPI_DRR: u32 = 0x6C;
/// SPISSR — SPI Slave Select Register.
const SPISSR: u32 = 0x70;
/// TX_FIFO_OCY — Transmit FIFO Occupancy Register.
const TX_FIFO_OCY: u32 = 0x74;
/// RX_FIFO_OCY — Receive FIFO Occupancy Register.
const RX_FIFO_OCY: u32 = 0x78;
/// DGIER — Device Global Interrupt Enable Register.
const DGIER: u32 = 0x1C;
/// IPISR — IP Interrupt Status Register.
const IPISR: u32 = 0x20;
/// IPIER — IP Interrupt Enable Register.
const IPIER: u32 = 0x28;

// ---------------------------------------------------------------------------
// SRR constants
// ---------------------------------------------------------------------------

/// Software reset key.
const SRR_RESET_KEY: u32 = 0x0000_000A;

// ---------------------------------------------------------------------------
// SPICR bits
// ---------------------------------------------------------------------------

/// Loop mode (internal loopback).
const SPICR_LOOP: u32 = 1 << 0;
/// SPI system enable.
const SPICR_SPE: u32 = 1 << 1;
/// Master mode (0=slave, 1=master).
const SPICR_MASTER: u32 = 1 << 2;
/// Clock polarity.
const SPICR_CPOL: u32 = 1 << 3;
/// Clock phase.
const SPICR_CPHA: u32 = 1 << 4;
/// TX FIFO reset.
const SPICR_TX_RST: u32 = 1 << 5;
/// RX FIFO reset.
const SPICR_RX_RST: u32 = 1 << 6;
/// Manual slave select assertion enable.
const SPICR_MANUAL_SS: u32 = 1 << 7;
/// Master transaction inhibit.
const SPICR_MTI: u32 = 1 << 8;
/// LSB first.
const SPICR_LSB_FIRST: u32 = 1 << 9;

// ---------------------------------------------------------------------------
// SPISR bits
// ---------------------------------------------------------------------------

/// RX FIFO empty.
const SPISR_RX_EMPTY: u32 = 1 << 0;
/// RX FIFO full.
const SPISR_RX_FULL: u32 = 1 << 1;
/// TX FIFO empty.
const SPISR_TX_EMPTY: u32 = 1 << 2;
/// TX FIFO full.
const SPISR_TX_FULL: u32 = 1 << 3;
/// Mode fault error.
const SPISR_MODF: u32 = 1 << 4;
/// Slave mode select.
const SPISR_SLAVE_MODE: u32 = 1 << 5;
/// Command error.
const SPISR_CMD_ERR: u32 = 1 << 7;

// ---------------------------------------------------------------------------
// FIFO depth
// ---------------------------------------------------------------------------

const FIFO_DEPTH: usize = 256;

// ---------------------------------------------------------------------------
// Driver implementation
// ---------------------------------------------------------------------------

/// Xilinx AXI Quad SPI FPGA controller driver.
pub struct AxiQuadSpi<F: FpgaFabric> {
    /// FPGA fabric accessor.
    fabric: F,
    /// AXI base address.
    base_addr: u64,
    /// Whether configured.
    configured: bool,
    /// Stored configuration.
    config: Option<SpiConfig>,
}

impl<F: FpgaFabric> AxiQuadSpi<F> {
    /// Create a new AXI Quad SPI driver.
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

    /// Return whether configured.
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

    /// Perform a soft reset.
    fn soft_reset(&mut self) -> Result<(), HalError> {
        self.write_reg(SRR, SRR_RESET_KEY)
    }

    /// Build the SPICR value from configuration.
    fn build_spicr(config: &SpiConfig) -> u32 {
        let mut cr = SPICR_MASTER | SPICR_SPE | SPICR_MANUAL_SS;

        let (cpol, cpha) = crate::spi::mode_to_cpol_cpha(config.mode);
        if cpol {
            cr |= SPICR_CPOL;
        }
        if cpha {
            cr |= SPICR_CPHA;
        }

        if config.bit_order == smallaios_kernel::hal::SpiBitOrder::LsbFirst {
            cr |= SPICR_LSB_FIRST;
        }

        cr
    }

    /// Assert a specific chip select (active low).
    fn assert_cs(&mut self, cs: u8) -> Result<(), HalError> {
        // SPISSR: active-low bitmask. Assert CS by clearing the bit.
        let mask = !(1u32 << cs);
        self.write_reg(SPISSR, mask)
    }

    /// Deassert all chip selects.
    fn deassert_all_cs(&mut self) -> Result<(), HalError> {
        self.write_reg(SPISSR, 0xFFFF_FFFF)
    }
}

impl<F: FpgaFabric> SpiController for AxiQuadSpi<F> {
    fn init(&mut self, config: SpiConfig) -> Result<(), HalError> {
        if !crate::spi::validate_word_size(config.word_size) {
            return Err(HalError::InvalidConfig);
        }
        if config.timeout_us == 0 {
            return Err(HalError::InvalidConfig);
        }

        // Soft reset.
        self.soft_reset()?;

        // Disable global interrupts.
        self.write_reg(DGIER, 0)?;
        self.write_reg(IPIER, 0)?;

        // Clear interrupt status.
        let isr = self.read_reg(IPISR)?;
        self.write_reg(IPISR, isr)?;

        // Configure SPICR: master, enable, reset FIFOs.
        let cr = Self::build_spicr(&config) | SPICR_TX_RST | SPICR_RX_RST;
        self.write_reg(SPICR, cr)?;

        // Release FIFO reset.
        let cr = Self::build_spicr(&config);
        self.write_reg(SPICR, cr)?;

        // Deassert all CS.
        self.deassert_all_cs()?;

        self.config = Some(config);
        self.configured = true;
        Ok(())
    }

    fn transfer(&mut self, cs: u8, tx_data: &[u8], rx_buf: &mut [u8]) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        crate::spi::validate_cs(cs)?;

        self.assert_cs(cs)?;

        // Inhibit master transactions while loading TX FIFO.
        let cr = self.read_reg(SPICR)?;
        self.write_reg(SPICR, cr | SPICR_MTI)?;

        // Load TX FIFO.
        let len = tx_data.len().min(rx_buf.len());
        for byte in &tx_data[..len] {
            self.write_reg(SPI_DTR, *byte as u32)?;
        }

        // Release inhibit to start transfer.
        self.write_reg(SPICR, cr & !SPICR_MTI)?;

        // Read RX FIFO.
        for byte in &mut rx_buf[..len] {
            let val = self.read_reg(SPI_DRR)?;
            *byte = (val & 0xFF) as u8;
        }

        self.deassert_all_cs()?;
        Ok(())
    }

    fn write(&mut self, cs: u8, data: &[u8]) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        crate::spi::validate_cs(cs)?;

        self.assert_cs(cs)?;

        let cr = self.read_reg(SPICR)?;
        self.write_reg(SPICR, cr | SPICR_MTI)?;

        for &byte in data {
            self.write_reg(SPI_DTR, byte as u32)?;
        }

        self.write_reg(SPICR, cr & !SPICR_MTI)?;

        // Drain RX FIFO.
        for _ in 0..data.len() {
            let _ = self.read_reg(SPI_DRR);
        }

        self.deassert_all_cs()?;
        Ok(())
    }

    fn read(&mut self, cs: u8, buf: &mut [u8]) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        crate::spi::validate_cs(cs)?;

        self.assert_cs(cs)?;

        let cr = self.read_reg(SPICR)?;
        self.write_reg(SPICR, cr | SPICR_MTI)?;

        for _ in 0..buf.len() {
            self.write_reg(SPI_DTR, 0)?;
        }

        self.write_reg(SPICR, cr & !SPICR_MTI)?;

        for byte in buf.iter_mut() {
            let val = self.read_reg(SPI_DRR)?;
            *byte = (val & 0xFF) as u8;
        }

        self.deassert_all_cs()?;
        Ok(())
    }

    fn transfer_dma(
        &mut self,
        _cs: u8,
        _tx_addr: u64,
        _rx_addr: u64,
        _len: u32,
    ) -> Result<DmaToken, HalError> {
        Err(HalError::NotSupported)
    }

    fn cs_assert(&mut self, cs: u8) -> Result<(), HalError> {
        crate::spi::validate_cs(cs)?;
        self.assert_cs(cs)
    }

    fn cs_deassert(&mut self, _cs: u8) -> Result<(), HalError> {
        self.deassert_all_cs()
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
    use smallaios_kernel::hal::{DmaDescriptor, SpiBitOrder, SpiMode};

    struct MockFpgaFabric {
        regs: [u32; 64],
    }

    impl MockFpgaFabric {
        fn new() -> Self {
            Self { regs: [0u32; 64] }
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

    fn make_config() -> SpiConfig {
        SpiConfig {
            mode: SpiMode::Mode0,
            clock_hz: 10_000_000,
            bit_order: SpiBitOrder::MsbFirst,
            word_size: 8,
            cs_active_low: true,
            timeout_us: 1000,
        }
    }

    #[test]
    fn test_axi_quad_spi_new() {
        let fab = MockFpgaFabric::new();
        let drv = AxiQuadSpi::new(fab, 0x4400_0000);
        assert_eq!(drv.base_addr(), 0x4400_0000);
        assert!(!drv.is_configured());
    }

    #[test]
    fn test_axi_quad_spi_init() {
        let fab = MockFpgaFabric::new();
        let mut drv = AxiQuadSpi::new(fab, 0x4400_0000);
        assert!(drv.init(make_config()).is_ok());
        assert!(drv.is_configured());
    }

    #[test]
    fn test_axi_quad_spi_init_invalid_word_size() {
        let fab = MockFpgaFabric::new();
        let mut drv = AxiQuadSpi::new(fab, 0x4400_0000);
        let mut config = make_config();
        config.word_size = 12;
        assert_eq!(drv.init(config), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_axi_quad_spi_write_not_configured() {
        let fab = MockFpgaFabric::new();
        let mut drv = AxiQuadSpi::new(fab, 0x4400_0000);
        assert_eq!(drv.write(0, &[0x01]), Err(HalError::InitFailed));
    }

    #[test]
    fn test_axi_quad_spi_transfer() {
        let fab = MockFpgaFabric::new();
        let mut drv = AxiQuadSpi::new(fab, 0x4400_0000);
        drv.init(make_config()).unwrap();
        let mut rx = [0u8; 2];
        assert!(drv.transfer(0, &[0xAB, 0xCD], &mut rx).is_ok());
    }

    #[test]
    fn test_axi_quad_spi_cs_assert_deassert() {
        let fab = MockFpgaFabric::new();
        let mut drv = AxiQuadSpi::new(fab, 0x4400_0000);
        drv.init(make_config()).unwrap();
        assert!(drv.cs_assert(0).is_ok());
        assert!(drv.cs_deassert(0).is_ok());
    }

    #[test]
    fn test_axi_quad_spi_reset() {
        let fab = MockFpgaFabric::new();
        let mut drv = AxiQuadSpi::new(fab, 0x4400_0000);
        drv.init(make_config()).unwrap();
        assert!(drv.reset().is_ok());
        assert!(!drv.is_configured());
    }

    #[test]
    fn test_build_spicr_mode0_msb() {
        let config = make_config();
        let cr = AxiQuadSpi::<MockFpgaFabric>::build_spicr(&config);
        assert_ne!(cr & SPICR_MASTER, 0);
        assert_ne!(cr & SPICR_SPE, 0);
        assert_eq!(cr & SPICR_CPOL, 0);
        assert_eq!(cr & SPICR_CPHA, 0);
        assert_eq!(cr & SPICR_LSB_FIRST, 0);
    }

    #[test]
    fn test_build_spicr_mode3_lsb() {
        let mut config = make_config();
        config.mode = SpiMode::Mode3;
        config.bit_order = SpiBitOrder::LsbFirst;
        let cr = AxiQuadSpi::<MockFpgaFabric>::build_spicr(&config);
        assert_ne!(cr & SPICR_CPOL, 0);
        assert_ne!(cr & SPICR_CPHA, 0);
        assert_ne!(cr & SPICR_LSB_FIRST, 0);
    }
}
