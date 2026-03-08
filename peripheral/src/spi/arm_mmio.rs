// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARM PL022 SSP/SPI MMIO controller driver.
//!
//! Implements the [`SpiController`] trait for ARM platforms using the
//! PrimeCell PL022 Synchronous Serial Port (SSP) IP, found on many
//! ARM-based SoCs (e.g., Versatile, RealView, STM32).
//!
//! # References
//!
//! - ARM PL022 TRM (DDI0194)

#![allow(dead_code)]

use smallaios_kernel::hal::{DmaToken, HalError, SpiConfig, SpiController};

// ---------------------------------------------------------------------------
// PL022 register offsets (from DDI0194)
// ---------------------------------------------------------------------------

/// SSPCR0 — Control Register 0.
const SSPCR0: u32 = 0x000;
/// SSPCR1 — Control Register 1.
const SSPCR1: u32 = 0x004;
/// SSPDR — Data Register (16-bit).
const SSPDR: u32 = 0x008;
/// SSPSR — Status Register.
const SSPSR: u32 = 0x00C;
/// SSPCPSR — Clock Prescale Register.
const SSPCPSR: u32 = 0x010;
/// SSPIMSC — Interrupt Mask Set/Clear Register.
const SSPIMSC: u32 = 0x014;
/// SSPRIS — Raw Interrupt Status Register.
const SSPRIS: u32 = 0x018;
/// SSPMIS — Masked Interrupt Status Register.
const SSPMIS: u32 = 0x01C;
/// SSPICR — Interrupt Clear Register.
const SSPICR: u32 = 0x020;
/// SSPDMACR — DMA Control Register.
const SSPDMACR: u32 = 0x024;

// ---------------------------------------------------------------------------
// SSPCR0 bits
// ---------------------------------------------------------------------------

/// Data Size Select mask (bits 3:0). Encoded as (bits_per_frame - 1).
const CR0_DSS_MASK: u32 = 0x0F;
/// Frame Format: SPI (bits 5:4 = 00).
const CR0_FRF_SPI: u32 = 0x00;
/// Frame Format: SSI (bits 5:4 = 01).
const CR0_FRF_SSI: u32 = 0x10;
/// Frame Format: Microwire (bits 5:4 = 10).
const CR0_FRF_MW: u32 = 0x20;
/// Clock polarity (bit 6).
const CR0_SPO: u32 = 1 << 6;
/// Clock phase (bit 7).
const CR0_SPH: u32 = 1 << 7;
/// Serial Clock Rate mask (bits 15:8).
const CR0_SCR_SHIFT: u32 = 8;

// ---------------------------------------------------------------------------
// SSPCR1 bits
// ---------------------------------------------------------------------------

/// Loopback mode.
const CR1_LBM: u32 = 1 << 0;
/// SSP enable.
const CR1_SSE: u32 = 1 << 1;
/// Master/slave select (0=master, 1=slave).
const CR1_MS: u32 = 1 << 2;
/// Slave output disable.
const CR1_SOD: u32 = 1 << 3;

// ---------------------------------------------------------------------------
// SSPSR (Status Register) bits
// ---------------------------------------------------------------------------

/// TX FIFO empty.
const SR_TFE: u32 = 1 << 0;
/// TX FIFO not full.
const SR_TNF: u32 = 1 << 1;
/// RX FIFO not empty.
const SR_RNE: u32 = 1 << 2;
/// RX FIFO full.
const SR_RFF: u32 = 1 << 3;
/// SSP busy.
const SR_BSY: u32 = 1 << 4;

// ---------------------------------------------------------------------------
// SSPCPSR constraints
// ---------------------------------------------------------------------------

/// Minimum prescaler value (must be even, >= 2).
const CPSR_MIN: u32 = 2;
/// Maximum prescaler value.
const CPSR_MAX: u32 = 254;

/// Default PL022 input clock: 48 MHz.
const DEFAULT_INPUT_CLOCK_HZ: u32 = 48_000_000;

// ---------------------------------------------------------------------------
// Driver implementation
// ---------------------------------------------------------------------------

/// ARM PL022 SPI controller driver.
pub struct Pl022Spi {
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

impl Pl022Spi {
    /// Create a new PL022 SPI driver.
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

    /// Build SSPCR0 from config.
    fn build_cr0(config: &SpiConfig, scr: u8) -> u32 {
        let dss = (config.word_size.clamp(4, 16) - 1) as u32;
        let (cpol, cpha) = crate::spi::mode_to_cpol_cpha(config.mode);
        let mut cr0 = (dss & CR0_DSS_MASK) | CR0_FRF_SPI;
        if cpol {
            cr0 |= CR0_SPO;
        }
        if cpha {
            cr0 |= CR0_SPH;
        }
        cr0 |= (scr as u32) << CR0_SCR_SHIFT;
        cr0
    }

    /// Compute CPSR and SCR from input clock and target clock.
    ///
    /// PL022 clock: `SPICLK = input_clock / (CPSR * (1 + SCR))`
    /// CPSR must be even, 2..=254. SCR is 0..=255.
    fn compute_clock_params(input_hz: u32, target_hz: u32) -> (u8, u8) {
        if target_hz == 0 || input_hz == 0 {
            return (CPSR_MIN as u8, 0);
        }

        let mut best_cpsr = CPSR_MIN as u8;
        let mut best_scr = 0u8;
        let mut best_error = u32::MAX;

        let mut cpsr = CPSR_MIN;
        while cpsr <= CPSR_MAX {
            let divisor = (cpsr as u64) * (target_hz as u64);
            let scr_needed = (input_hz as u64).checked_div(divisor).unwrap_or(0) as u32;
            let scr = if scr_needed > 0 { scr_needed - 1 } else { 0 };
            let scr = scr.min(255) as u8;

            let actual = input_hz / (cpsr * (1 + scr as u32));
            let error = actual.abs_diff(target_hz);

            if error < best_error {
                best_error = error;
                best_cpsr = cpsr as u8;
                best_scr = scr;
            }

            if error == 0 {
                break;
            }

            cpsr += 2;
        }

        (best_cpsr, best_scr)
    }
}

impl SpiController for Pl022Spi {
    fn init(&mut self, config: SpiConfig) -> Result<(), HalError> {
        if !crate::spi::validate_word_size(config.word_size) {
            return Err(HalError::InvalidConfig);
        }
        if config.timeout_us == 0 {
            return Err(HalError::InvalidConfig);
        }

        // Disable SSP.
        self.write_reg(SSPCR1, 0)?;

        // Compute clock parameters.
        let (cpsr, scr) = Self::compute_clock_params(self.input_clock_hz, config.clock_hz);
        self.write_reg(SSPCPSR, cpsr as u32)?;

        // Configure CR0.
        let cr0 = Self::build_cr0(&config, scr);
        self.write_reg(SSPCR0, cr0)?;

        // Mask all interrupts.
        self.write_reg(SSPIMSC, 0)?;

        // Enable SSP in master mode.
        self.write_reg(SSPCR1, CR1_SSE)?;

        self.config = Some(config);
        self.configured = true;
        Ok(())
    }

    fn transfer(&mut self, cs: u8, tx_data: &[u8], rx_buf: &mut [u8]) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        crate::spi::validate_cs(cs)?;

        let len = tx_data.len().min(rx_buf.len());
        for i in 0..len {
            self.write_reg(SSPDR, tx_data[i] as u32)?;
            let val = self.read_reg(SSPDR)?;
            rx_buf[i] = (val & 0xFF) as u8;
        }
        Ok(())
    }

    fn write(&mut self, cs: u8, data: &[u8]) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        crate::spi::validate_cs(cs)?;

        for &byte in data {
            self.write_reg(SSPDR, byte as u32)?;
            // Drain RX FIFO.
            let _ = self.read_reg(SSPDR);
        }
        Ok(())
    }

    fn read(&mut self, cs: u8, buf: &mut [u8]) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        crate::spi::validate_cs(cs)?;

        for byte in buf.iter_mut() {
            self.write_reg(SSPDR, 0)?; // Send zeros.
            let val = self.read_reg(SSPDR)?;
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
        // PL022 does not have hardware CS control; CS is managed by GPIO.
        Err(HalError::NotSupported)
    }

    fn cs_deassert(&mut self, cs: u8) -> Result<(), HalError> {
        crate::spi::validate_cs(cs)?;
        Err(HalError::NotSupported)
    }

    fn reset(&mut self) -> Result<(), HalError> {
        self.write_reg(SSPCR1, 0)?;
        self.write_reg(SSPIMSC, 0)?;
        let _ = self.read_reg(SSPICR);
        self.configured = false;
        self.config = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallaios_kernel::hal::{SpiBitOrder, SpiMode};

    #[test]
    fn test_pl022_new() {
        let drv = Pl022Spi::new(0x4003_0000, 25);
        assert_eq!(drv.base_addr(), 0x4003_0000);
        assert!(!drv.is_configured());
    }

    #[test]
    fn test_compute_clock_params_exact() {
        // 48 MHz / (2 * (1 + 11)) = 48M / 24 = 2 MHz
        let (cpsr, scr) = Pl022Spi::compute_clock_params(48_000_000, 2_000_000);
        let actual = 48_000_000 / (cpsr as u32 * (1 + scr as u32));
        assert_eq!(actual, 2_000_000);
    }

    #[test]
    fn test_compute_clock_params_max_speed() {
        // Minimum divider = CPSR=2, SCR=0 → input/2
        let (cpsr, scr) = Pl022Spi::compute_clock_params(48_000_000, 48_000_000);
        assert_eq!(cpsr, 2);
        assert_eq!(scr, 0);
    }

    #[test]
    fn test_compute_clock_params_zero_target() {
        let (cpsr, _scr) = Pl022Spi::compute_clock_params(48_000_000, 0);
        assert_eq!(cpsr, CPSR_MIN as u8);
    }

    #[test]
    fn test_build_cr0_mode0() {
        let config = SpiConfig {
            mode: SpiMode::Mode0,
            clock_hz: 1_000_000,
            bit_order: SpiBitOrder::MsbFirst,
            word_size: 8,
            cs_active_low: true,
            timeout_us: 1000,
        };
        let cr0 = Pl022Spi::build_cr0(&config, 0);
        assert_eq!(cr0 & CR0_DSS_MASK, 7); // 8-1
        assert_eq!(cr0 & CR0_SPO, 0); // CPOL=0
        assert_eq!(cr0 & CR0_SPH, 0); // CPHA=0
    }

    #[test]
    fn test_build_cr0_mode3_16bit() {
        let config = SpiConfig {
            mode: SpiMode::Mode3,
            clock_hz: 1_000_000,
            bit_order: SpiBitOrder::MsbFirst,
            word_size: 16,
            cs_active_low: true,
            timeout_us: 1000,
        };
        let cr0 = Pl022Spi::build_cr0(&config, 5);
        assert_eq!(cr0 & CR0_DSS_MASK, 15); // 16-1
        assert_ne!(cr0 & CR0_SPO, 0); // CPOL=1
        assert_ne!(cr0 & CR0_SPH, 0); // CPHA=1
        assert_eq!((cr0 >> CR0_SCR_SHIFT) & 0xFF, 5);
    }

    #[test]
    fn test_init_returns_mmio_error() {
        let mut drv = Pl022Spi::new(0x4003_0000, 25);
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
    fn test_init_invalid_word_size() {
        let mut drv = Pl022Spi::new(0x4003_0000, 25);
        let config = SpiConfig {
            mode: SpiMode::Mode0,
            clock_hz: 1_000_000,
            bit_order: SpiBitOrder::MsbFirst,
            word_size: 12, // invalid
            cs_active_low: true,
            timeout_us: 1000,
        };
        assert_eq!(drv.init(config), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_transfer_not_configured() {
        let mut drv = Pl022Spi::new(0x4003_0000, 25);
        let mut rx = [0u8; 4];
        assert_eq!(
            drv.transfer(0, &[0x01, 0x02], &mut rx),
            Err(HalError::InitFailed)
        );
    }

    #[test]
    fn test_cs_assert_not_supported() {
        let mut drv = Pl022Spi::new(0x4003_0000, 25);
        // PL022 doesn't have hardware CS.
        assert_eq!(drv.cs_assert(0), Err(HalError::NotSupported));
    }

    #[test]
    fn test_constructor_fields() {
        let drv = Pl022Spi::new(0xABCD_0000, 42);
        assert_eq!(drv.base_addr(), 0xABCD_0000);
        assert_eq!(drv.irq, 42);
        assert!(!drv.is_configured());
        assert!(drv.config.is_none());
        assert_eq!(drv.input_clock_hz, DEFAULT_INPUT_CLOCK_HZ);
    }

    #[test]
    fn test_with_clock_constructor() {
        let drv = Pl022Spi::with_clock(0x5000_0000, 15, 96_000_000);
        assert_eq!(drv.base_addr(), 0x5000_0000);
        assert_eq!(drv.irq, 15);
        assert_eq!(drv.input_clock_hz, 96_000_000);
        assert!(!drv.is_configured());
    }

    #[test]
    fn test_write_not_configured() {
        let mut drv = Pl022Spi::new(0x4003_0000, 25);
        assert_eq!(drv.write(0, &[0x01]), Err(HalError::InitFailed));
    }

    #[test]
    fn test_read_not_configured() {
        let mut drv = Pl022Spi::new(0x4003_0000, 25);
        let mut buf = [0u8; 4];
        assert_eq!(drv.read(0, &mut buf), Err(HalError::InitFailed));
    }

    #[test]
    fn test_transfer_dma_not_supported() {
        let mut drv = Pl022Spi::new(0x4003_0000, 25);
        assert_eq!(
            drv.transfer_dma(0, 0x8000_0000, 0x8001_0000, 256),
            Err(HalError::NotSupported)
        );
    }

    #[test]
    fn test_cs_deassert_not_supported() {
        let mut drv = Pl022Spi::new(0x4003_0000, 25);
        assert_eq!(drv.cs_deassert(0), Err(HalError::NotSupported));
    }

    #[test]
    fn test_cs_assert_invalid_cs() {
        let mut drv = Pl022Spi::new(0x4003_0000, 25);
        // CS >= 8 should fail with InvalidConfig before NotSupported.
        assert_eq!(drv.cs_assert(8), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_init_zero_timeout() {
        let mut drv = Pl022Spi::new(0x4003_0000, 25);
        let config = SpiConfig {
            mode: SpiMode::Mode0,
            clock_hz: 1_000_000,
            bit_order: SpiBitOrder::MsbFirst,
            word_size: 8,
            cs_active_low: true,
            timeout_us: 0,
        };
        assert_eq!(drv.init(config), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_build_cr0_mode1() {
        let config = SpiConfig {
            mode: SpiMode::Mode1,
            clock_hz: 1_000_000,
            bit_order: SpiBitOrder::MsbFirst,
            word_size: 8,
            cs_active_low: true,
            timeout_us: 1000,
        };
        let cr0 = Pl022Spi::build_cr0(&config, 0);
        assert_eq!(cr0 & CR0_SPO, 0); // CPOL=0
        assert_ne!(cr0 & CR0_SPH, 0); // CPHA=1
    }

    #[test]
    fn test_build_cr0_mode2() {
        let config = SpiConfig {
            mode: SpiMode::Mode2,
            clock_hz: 1_000_000,
            bit_order: SpiBitOrder::MsbFirst,
            word_size: 8,
            cs_active_low: true,
            timeout_us: 1000,
        };
        let cr0 = Pl022Spi::build_cr0(&config, 0);
        assert_ne!(cr0 & CR0_SPO, 0); // CPOL=1
        assert_eq!(cr0 & CR0_SPH, 0); // CPHA=0
    }

    #[test]
    fn test_build_cr0_4bit_word() {
        let config = SpiConfig {
            mode: SpiMode::Mode0,
            clock_hz: 1_000_000,
            bit_order: SpiBitOrder::MsbFirst,
            word_size: 4,
            cs_active_low: true,
            timeout_us: 1000,
        };
        let cr0 = Pl022Spi::build_cr0(&config, 0);
        assert_eq!(cr0 & CR0_DSS_MASK, 3); // 4-1
    }

    #[test]
    fn test_build_cr0_scr_value() {
        let config = SpiConfig {
            mode: SpiMode::Mode0,
            clock_hz: 1_000_000,
            bit_order: SpiBitOrder::MsbFirst,
            word_size: 8,
            cs_active_low: true,
            timeout_us: 1000,
        };
        let cr0 = Pl022Spi::build_cr0(&config, 200);
        assert_eq!((cr0 >> CR0_SCR_SHIFT) & 0xFF, 200);
    }

    #[test]
    fn test_compute_clock_params_low_speed() {
        // Very low target speed should produce large CPSR and SCR values.
        let (cpsr, scr) = Pl022Spi::compute_clock_params(48_000_000, 100_000);
        let actual = 48_000_000 / (cpsr as u32 * (1 + scr as u32));
        // Should be close to 100 kHz.
        assert!(actual <= 100_000 + 5_000);
        assert!(actual >= 100_000 - 5_000);
    }

    #[test]
    fn test_compute_clock_params_zero_input() {
        let (cpsr, _scr) = Pl022Spi::compute_clock_params(0, 1_000_000);
        assert_eq!(cpsr, CPSR_MIN as u8);
    }

    #[test]
    fn test_reset_returns_mmio_error() {
        let mut drv = Pl022Spi::new(0x4003_0000, 25);
        assert_eq!(drv.reset(), Err(HalError::MmioError));
    }

    #[test]
    fn test_init_valid_word_sizes() {
        // Word sizes 4, 8, 16, 32 are valid. Only 4, 8, 16 fit PL022 (clamped).
        for ws in [4u8, 8, 16, 32] {
            let mut drv = Pl022Spi::new(0x4003_0000, 25);
            let config = SpiConfig {
                mode: SpiMode::Mode0,
                clock_hz: 1_000_000,
                bit_order: SpiBitOrder::MsbFirst,
                word_size: ws,
                cs_active_low: true,
                timeout_us: 1000,
            };
            // Should fail at MmioError (past validation), not InvalidConfig.
            assert_eq!(drv.init(config), Err(HalError::MmioError));
        }
    }
}
