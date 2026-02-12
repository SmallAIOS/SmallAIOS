// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SiFive SPI MMIO controller driver.
//!
//! Implements the [`SpiController`] trait for RISC-V platforms using the
//! SiFive SPI controller IP, as found on FU540/FU740 SoCs.
//!
//! # References
//!
//! - SiFive FU740-C000 Manual, Chapter 19: SPI

#![allow(dead_code)]

use smallaios_kernel::hal::{DmaToken, HalError, SpiConfig, SpiController, SpiMode};

// ---------------------------------------------------------------------------
// SiFive SPI register offsets
// ---------------------------------------------------------------------------

/// sckdiv — Serial clock divisor.
const SCKDIV: u32 = 0x00;
/// sckmode — Serial clock mode.
const SCKMODE: u32 = 0x04;
/// csid — Chip select ID.
const CSID: u32 = 0x10;
/// csdef — Chip select default value.
const CSDEF: u32 = 0x14;
/// csmode — Chip select mode.
const CSMODE: u32 = 0x18;
/// delay0 — Delay control 0.
const DELAY0: u32 = 0x28;
/// delay1 — Delay control 1.
const DELAY1: u32 = 0x2C;
/// fmt — Frame format.
const FMT: u32 = 0x40;
/// txdata — Transmit data register.
const TXDATA: u32 = 0x48;
/// rxdata — Receive data register.
const RXDATA: u32 = 0x4C;
/// txmark — TX watermark.
const TXMARK: u32 = 0x50;
/// rxmark — RX watermark.
const RXMARK: u32 = 0x54;
/// fctrl — Flash interface control.
const FCTRL: u32 = 0x60;
/// ffmt — Flash instruction format.
const FFMT: u32 = 0x64;
/// ie — Interrupt enable.
const IE: u32 = 0x70;
/// ip — Interrupt pending.
const IP: u32 = 0x74;

// ---------------------------------------------------------------------------
// sckmode bits
// ---------------------------------------------------------------------------

/// Clock polarity bit.
const SCKMODE_POL: u32 = 1 << 1;
/// Clock phase bit.
const SCKMODE_PHA: u32 = 1 << 0;

// ---------------------------------------------------------------------------
// csmode values
// ---------------------------------------------------------------------------

/// Auto mode: CS asserted/deasserted around each frame.
const CSMODE_AUTO: u32 = 0;
/// Hold mode: CS stays asserted after frame.
const CSMODE_HOLD: u32 = 2;
/// Off mode: CS deasserted.
const CSMODE_OFF: u32 = 3;

// ---------------------------------------------------------------------------
// fmt register bits
// ---------------------------------------------------------------------------

/// Protocol: single (bits 1:0 = 0).
const FMT_PROTO_SINGLE: u32 = 0;
/// Protocol: dual (bits 1:0 = 1).
const FMT_PROTO_DUAL: u32 = 1;
/// Protocol: quad (bits 1:0 = 2).
const FMT_PROTO_QUAD: u32 = 2;
/// Endianness: MSB first (bit 2 = 0).
const FMT_ENDIAN_MSB: u32 = 0;
/// Endianness: LSB first (bit 2 = 1).
const FMT_ENDIAN_LSB: u32 = 1 << 2;
/// I/O direction: TX only (bit 3 = 1).
const FMT_DIR_TX: u32 = 1 << 3;
/// Length field shift (bits 19:16).
const FMT_LEN_SHIFT: u32 = 16;

// ---------------------------------------------------------------------------
// txdata/rxdata bits
// ---------------------------------------------------------------------------

/// TX FIFO full flag (bit 31 of txdata).
const TXDATA_FULL: u32 = 1 << 31;
/// RX FIFO empty flag (bit 31 of rxdata).
const RXDATA_EMPTY: u32 = 1 << 31;

/// Default SiFive SPI input clock: 500 MHz.
const DEFAULT_INPUT_CLOCK_HZ: u32 = 500_000_000;

// ---------------------------------------------------------------------------
// Driver implementation
// ---------------------------------------------------------------------------

/// SiFive SPI controller driver.
pub struct SiFiveSpi {
    /// MMIO base address.
    base_addr: u64,
    /// IRQ number.
    irq: u32,
    /// Whether configured.
    configured: bool,
    /// Stored configuration.
    config: Option<SpiConfig>,
    /// Input reference clock.
    input_clock_hz: u32,
}

impl SiFiveSpi {
    /// Create a new SiFive SPI driver.
    pub fn new(base_addr: u64, irq: u32) -> Self {
        Self {
            base_addr,
            irq,
            configured: false,
            config: None,
            input_clock_hz: DEFAULT_INPUT_CLOCK_HZ,
        }
    }

    /// Create with custom input clock.
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

    /// Read a register (stub).
    fn read_reg(&self, _offset: u32) -> Result<u32, HalError> {
        Err(HalError::MmioError)
    }

    /// Write a register (stub).
    fn write_reg(&mut self, _offset: u32, _value: u32) -> Result<(), HalError> {
        Err(HalError::MmioError)
    }

    /// Compute the sckdiv value for the target clock.
    ///
    /// SiFive: `SPICLK = input_clock / (2 * (sckdiv + 1))`
    /// So `sckdiv = input_clock / (2 * target) - 1`
    fn compute_sckdiv(input_hz: u32, target_hz: u32) -> u32 {
        if target_hz == 0 || input_hz == 0 {
            return 0;
        }
        let div = input_hz / (2 * target_hz);
        if div > 0 {
            div - 1
        } else {
            0
        }
    }

    /// Build the sckmode register value.
    fn build_sckmode(mode: SpiMode) -> u32 {
        let (cpol, cpha) = crate::spi::mode_to_cpol_cpha(mode);
        let mut val = 0u32;
        if cpol {
            val |= SCKMODE_POL;
        }
        if cpha {
            val |= SCKMODE_PHA;
        }
        val
    }

    /// Build the fmt register value.
    fn build_fmt(config: &SpiConfig) -> u32 {
        let len = (config.word_size.clamp(1, 8) as u32) << FMT_LEN_SHIFT;
        let endian = match config.bit_order {
            smallaios_kernel::hal::SpiBitOrder::MsbFirst => FMT_ENDIAN_MSB,
            smallaios_kernel::hal::SpiBitOrder::LsbFirst => FMT_ENDIAN_LSB,
        };
        FMT_PROTO_SINGLE | endian | len
    }
}

impl SpiController for SiFiveSpi {
    fn init(&mut self, config: SpiConfig) -> Result<(), HalError> {
        if !crate::spi::validate_word_size(config.word_size) {
            return Err(HalError::InvalidConfig);
        }
        if config.timeout_us == 0 {
            return Err(HalError::InvalidConfig);
        }

        // Set clock divider.
        let sckdiv = Self::compute_sckdiv(self.input_clock_hz, config.clock_hz);
        self.write_reg(SCKDIV, sckdiv)?;

        // Set clock mode.
        let sckmode = Self::build_sckmode(config.mode);
        self.write_reg(SCKMODE, sckmode)?;

        // Set CS default (all high = deasserted).
        self.write_reg(CSDEF, 0xFFFF_FFFF)?;

        // Set CS mode to auto.
        self.write_reg(CSMODE, CSMODE_AUTO)?;

        // Set frame format.
        let fmt = Self::build_fmt(&config);
        self.write_reg(FMT, fmt)?;

        // Set watermarks.
        self.write_reg(TXMARK, 1)?;
        self.write_reg(RXMARK, 0)?;

        // Disable interrupts.
        self.write_reg(IE, 0)?;

        // Disable memory-mapped flash mode.
        self.write_reg(FCTRL, 0)?;

        self.config = Some(config);
        self.configured = true;
        Ok(())
    }

    fn transfer(&mut self, cs: u8, tx_data: &[u8], rx_buf: &mut [u8]) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        crate::spi::validate_cs(cs)?;

        self.write_reg(CSID, cs as u32)?;

        let len = tx_data.len().min(rx_buf.len());
        for i in 0..len {
            self.write_reg(TXDATA, tx_data[i] as u32)?;
            let val = self.read_reg(RXDATA)?;
            rx_buf[i] = (val & 0xFF) as u8;
        }
        Ok(())
    }

    fn write(&mut self, cs: u8, data: &[u8]) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        crate::spi::validate_cs(cs)?;

        self.write_reg(CSID, cs as u32)?;

        for &byte in data {
            self.write_reg(TXDATA, byte as u32)?;
        }
        Ok(())
    }

    fn read(&mut self, cs: u8, buf: &mut [u8]) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        crate::spi::validate_cs(cs)?;

        self.write_reg(CSID, cs as u32)?;

        for byte in buf.iter_mut() {
            self.write_reg(TXDATA, 0)?;
            let val = self.read_reg(RXDATA)?;
            *byte = (val & 0xFF) as u8;
        }
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
        self.write_reg(CSID, cs as u32)?;
        self.write_reg(CSMODE, CSMODE_HOLD)
    }

    fn cs_deassert(&mut self, _cs: u8) -> Result<(), HalError> {
        self.write_reg(CSMODE, CSMODE_OFF)
    }

    fn reset(&mut self) -> Result<(), HalError> {
        self.write_reg(IE, 0)?;
        self.write_reg(CSDEF, 0xFFFF_FFFF)?;
        self.write_reg(CSMODE, CSMODE_AUTO)?;
        self.configured = false;
        self.config = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallaios_kernel::hal::SpiBitOrder;

    #[test]
    fn test_sifive_spi_new() {
        let drv = SiFiveSpi::new(0x1004_0000, 51);
        assert_eq!(drv.base_addr(), 0x1004_0000);
        assert!(!drv.is_configured());
    }

    #[test]
    fn test_compute_sckdiv() {
        // 500 MHz / (2 * 25 MHz) - 1 = 9
        assert_eq!(SiFiveSpi::compute_sckdiv(500_000_000, 25_000_000), 9);
    }

    #[test]
    fn test_compute_sckdiv_1mhz() {
        // 500 MHz / (2 * 1 MHz) - 1 = 249
        assert_eq!(SiFiveSpi::compute_sckdiv(500_000_000, 1_000_000), 249);
    }

    #[test]
    fn test_compute_sckdiv_zero() {
        assert_eq!(SiFiveSpi::compute_sckdiv(0, 1_000_000), 0);
        assert_eq!(SiFiveSpi::compute_sckdiv(500_000_000, 0), 0);
    }

    #[test]
    fn test_build_sckmode() {
        assert_eq!(SiFiveSpi::build_sckmode(SpiMode::Mode0), 0);
        assert_eq!(SiFiveSpi::build_sckmode(SpiMode::Mode1), SCKMODE_PHA);
        assert_eq!(SiFiveSpi::build_sckmode(SpiMode::Mode2), SCKMODE_POL);
        assert_eq!(
            SiFiveSpi::build_sckmode(SpiMode::Mode3),
            SCKMODE_POL | SCKMODE_PHA
        );
    }

    #[test]
    fn test_build_fmt_8bit_msb() {
        let config = SpiConfig {
            mode: SpiMode::Mode0,
            clock_hz: 1_000_000,
            bit_order: SpiBitOrder::MsbFirst,
            word_size: 8,
            cs_active_low: true,
            timeout_us: 1000,
        };
        let fmt = SiFiveSpi::build_fmt(&config);
        assert_eq!((fmt >> FMT_LEN_SHIFT) & 0xF, 8);
        assert_eq!(fmt & FMT_ENDIAN_LSB, 0);
    }

    #[test]
    fn test_build_fmt_lsb_first() {
        let config = SpiConfig {
            mode: SpiMode::Mode0,
            clock_hz: 1_000_000,
            bit_order: SpiBitOrder::LsbFirst,
            word_size: 8,
            cs_active_low: true,
            timeout_us: 1000,
        };
        let fmt = SiFiveSpi::build_fmt(&config);
        assert_ne!(fmt & FMT_ENDIAN_LSB, 0);
    }

    #[test]
    fn test_init_returns_mmio_error() {
        let mut drv = SiFiveSpi::new(0x1004_0000, 51);
        let config = SpiConfig {
            mode: SpiMode::Mode0,
            clock_hz: 1_000_000,
            bit_order: SpiBitOrder::MsbFirst,
            word_size: 8,
            cs_active_low: true,
            timeout_us: 1000,
        };
        assert_eq!(drv.init(config), Err(HalError::MmioError));
    }

    #[test]
    fn test_write_not_configured() {
        let mut drv = SiFiveSpi::new(0x1004_0000, 51);
        assert_eq!(drv.write(0, &[0x01]), Err(HalError::InitFailed));
    }
}
