// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARM Designware I2C MMIO controller driver.
//!
//! Implements the [`I2cController`] trait for ARM platforms using the Synopsys
//! Designware I2C IP core, commonly found on Cortex-A/R SoCs (e.g., RK3399,
//! i.MX, Allwinner).
//!
//! # Hardware interface
//!
//! The controller is memory-mapped and accessed through 32-bit register
//! reads/writes. This driver provides the complete register layout and logic;
//! actual MMIO access is stubbed for host testing (returns `HalError::MmioError`).
//!
//! # References
//!
//! - Synopsys DesignWare DW_apb_i2c Databook

#![allow(dead_code)]

use smallaios_kernel::hal::{HalError, I2cAddressMode, I2cConfig, I2cController, I2cSpeed};

// ---------------------------------------------------------------------------
// Designware I2C register offsets
// ---------------------------------------------------------------------------

/// IC_CON — I2C Control Register.
const IC_CON: u32 = 0x00;
/// IC_TAR — I2C Target Address Register.
const IC_TAR: u32 = 0x04;
/// IC_SAR — I2C Slave Address Register.
const IC_SAR: u32 = 0x08;
/// IC_DATA_CMD — I2C Data Buffer and Command Register.
const IC_DATA_CMD: u32 = 0x10;
/// IC_SS_SCL_HCNT — Standard Speed SCL High Count.
const IC_SS_SCL_HCNT: u32 = 0x14;
/// IC_SS_SCL_LCNT — Standard Speed SCL Low Count.
const IC_SS_SCL_LCNT: u32 = 0x18;
/// IC_FS_SCL_HCNT — Fast Speed SCL High Count.
const IC_FS_SCL_HCNT: u32 = 0x1C;
/// IC_FS_SCL_LCNT — Fast Speed SCL Low Count.
const IC_FS_SCL_LCNT: u32 = 0x20;
/// IC_HS_SCL_HCNT — High Speed SCL High Count.
const IC_HS_SCL_HCNT: u32 = 0x24;
/// IC_HS_SCL_LCNT — High Speed SCL Low Count.
const IC_HS_SCL_LCNT: u32 = 0x28;
/// IC_INTR_STAT — Interrupt Status Register (read-only).
const IC_INTR_STAT: u32 = 0x2C;
/// IC_INTR_MASK — Interrupt Mask Register.
const IC_INTR_MASK: u32 = 0x30;
/// IC_RAW_INTR_STAT — Raw Interrupt Status.
const IC_RAW_INTR_STAT: u32 = 0x34;
/// IC_RX_TL — Receive FIFO Threshold Level.
const IC_RX_TL: u32 = 0x38;
/// IC_TX_TL — Transmit FIFO Threshold Level.
const IC_TX_TL: u32 = 0x3C;
/// IC_CLR_INTR — Clear Combined and Individual Interrupt.
const IC_CLR_INTR: u32 = 0x40;
/// IC_CLR_TX_ABRT — Clear TX_ABRT Interrupt.
const IC_CLR_TX_ABRT: u32 = 0x54;
/// IC_ENABLE — I2C Enable Register.
const IC_ENABLE: u32 = 0x6C;
/// IC_STATUS — I2C Status Register.
const IC_STATUS: u32 = 0x70;
/// IC_TXFLR — Transmit FIFO Level Register.
const IC_TXFLR: u32 = 0x74;
/// IC_RXFLR — Receive FIFO Level Register.
const IC_RXFLR: u32 = 0x78;
/// IC_SDA_HOLD — SDA Hold Time Length Register.
const IC_SDA_HOLD: u32 = 0x7C;
/// IC_TX_ABRT_SOURCE — Transmit Abort Source Register.
const IC_TX_ABRT_SOURCE: u32 = 0x80;
/// IC_ENABLE_STATUS — I2C Enable Status Register.
const IC_ENABLE_STATUS: u32 = 0x9C;
/// IC_COMP_PARAM_1 — Component Parameter Register 1.
const IC_COMP_PARAM_1: u32 = 0xF4;
/// IC_COMP_VERSION — Component Version.
const IC_COMP_VERSION: u32 = 0xF8;

// ---------------------------------------------------------------------------
// IC_CON register bits
// ---------------------------------------------------------------------------

/// Master mode enable.
const IC_CON_MASTER_MODE: u32 = 1 << 0;
/// Standard speed (1-2 in bits 2:1).
const IC_CON_SPEED_STD: u32 = 1 << 1;
/// Fast speed.
const IC_CON_SPEED_FAST: u32 = 2 << 1;
/// High speed.
const IC_CON_SPEED_HIGH: u32 = 3 << 1;
/// Speed mask.
const IC_CON_SPEED_MASK: u32 = 0x06;
/// 10-bit addressing for master.
const IC_CON_10BIT_ADDR_MASTER: u32 = 1 << 4;
/// Restart enable.
const IC_CON_RESTART_EN: u32 = 1 << 5;
/// Slave disable.
const IC_CON_SLAVE_DISABLE: u32 = 1 << 6;

// ---------------------------------------------------------------------------
// IC_DATA_CMD register bits
// ---------------------------------------------------------------------------

/// Read command bit.
const IC_DATA_CMD_READ: u32 = 1 << 8;
/// Stop bit.
const IC_DATA_CMD_STOP: u32 = 1 << 9;
/// Restart bit.
const IC_DATA_CMD_RESTART: u32 = 1 << 10;

// ---------------------------------------------------------------------------
// IC_STATUS register bits
// ---------------------------------------------------------------------------

/// Activity bit.
const IC_STATUS_ACTIVITY: u32 = 1 << 0;
/// TX FIFO not full.
const IC_STATUS_TFNF: u32 = 1 << 1;
/// TX FIFO empty.
const IC_STATUS_TFE: u32 = 1 << 2;
/// RX FIFO not empty.
const IC_STATUS_RFNE: u32 = 1 << 3;
/// RX FIFO full.
const IC_STATUS_RFF: u32 = 1 << 4;
/// Master FSM activity.
const IC_STATUS_MST_ACTIVITY: u32 = 1 << 5;

// ---------------------------------------------------------------------------
// IC_RAW_INTR_STAT / IC_INTR_STAT bits
// ---------------------------------------------------------------------------

/// TX abort interrupt.
const IC_INTR_TX_ABRT: u32 = 1 << 6;
/// TX empty interrupt.
const IC_INTR_TX_EMPTY: u32 = 1 << 4;
/// RX full interrupt.
const IC_INTR_RX_FULL: u32 = 1 << 2;

// ---------------------------------------------------------------------------
// IC_ENABLE bits
// ---------------------------------------------------------------------------

/// Enable the controller.
const IC_ENABLE_ENABLE: u32 = 1 << 0;
/// Abort current transfer.
const IC_ENABLE_ABORT: u32 = 1 << 1;

// ---------------------------------------------------------------------------
// TX abort source bits (selected)
// ---------------------------------------------------------------------------

/// 7-bit address NACK.
const ABRT_7B_ADDR_NOACK: u32 = 1 << 0;
/// 10-bit address byte 1 NACK.
const ABRT_10ADDR1_NOACK: u32 = 1 << 1;
/// 10-bit address byte 2 NACK.
const ABRT_10ADDR2_NOACK: u32 = 1 << 2;
/// Data NACK.
const ABRT_TXDATA_NOACK: u32 = 1 << 3;
/// Lost arbitration.
const ABRT_LOST: u32 = 1 << 12;

// ---------------------------------------------------------------------------
// Default constants
// ---------------------------------------------------------------------------

/// Default Designware I2C input clock: 48 MHz (common on ARM SoCs).
const DEFAULT_INPUT_CLOCK_HZ: u32 = 48_000_000;

/// FIFO depth for standard Designware configuration.
const FIFO_DEPTH: usize = 64;

// ---------------------------------------------------------------------------
// Driver implementation
// ---------------------------------------------------------------------------

/// ARM Designware I2C MMIO controller driver.
///
/// Wraps a base MMIO address for accessing the Designware I2C register block.
/// Implements [`I2cController`] for use on ARM-based SoCs.
pub struct DesignwareI2c {
    /// MMIO base address of the I2C register block.
    base_addr: u64,
    /// IRQ number for interrupt-driven operation.
    irq: u32,
    /// Whether the controller has been initialized.
    configured: bool,
    /// Stored configuration.
    config: Option<I2cConfig>,
    /// Input reference clock in Hz.
    input_clock_hz: u32,
}

impl DesignwareI2c {
    /// Create a new Designware I2C driver at the given MMIO base address.
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

    /// Return the IRQ number.
    pub fn irq(&self) -> u32 {
        self.irq
    }

    /// Return whether the controller is configured.
    pub fn is_configured(&self) -> bool {
        self.configured
    }

    /// Read a 32-bit register (stub: returns MmioError on host).
    fn read_reg(&self, _offset: u32) -> Result<u32, HalError> {
        // In a real kernel:
        //   let addr = (self.base_addr + offset as u64) as *const u32;
        //   Ok(unsafe { core::ptr::read_volatile(addr) })
        Err(HalError::MmioError)
    }

    /// Write a 32-bit register (stub: returns MmioError on host).
    fn write_reg(&mut self, _offset: u32, _value: u32) -> Result<(), HalError> {
        // In a real kernel:
        //   let addr = (self.base_addr + offset as u64) as *mut u32;
        //   unsafe { core::ptr::write_volatile(addr, value) }
        //   Ok(())
        Err(HalError::MmioError)
    }

    /// Disable the I2C controller (IC_ENABLE = 0).
    fn disable(&mut self) -> Result<(), HalError> {
        self.write_reg(IC_ENABLE, 0)
    }

    /// Enable the I2C controller (IC_ENABLE = 1).
    fn enable(&mut self) -> Result<(), HalError> {
        self.write_reg(IC_ENABLE, IC_ENABLE_ENABLE)
    }

    /// Build the IC_CON register value from the configuration.
    fn build_ic_con(config: &I2cConfig) -> u32 {
        let mut con = IC_CON_MASTER_MODE | IC_CON_SLAVE_DISABLE | IC_CON_RESTART_EN;

        con |= match config.speed {
            I2cSpeed::Standard => IC_CON_SPEED_STD,
            I2cSpeed::Fast | I2cSpeed::FastPlus => IC_CON_SPEED_FAST,
        };

        if config.address_mode == I2cAddressMode::TenBit {
            con |= IC_CON_10BIT_ADDR_MASTER;
        }

        con
    }

    /// Compute SCL timing register values for the configured speed.
    fn compute_timing(&self, speed: I2cSpeed) -> (u16, u16) {
        crate::i2c::compute_scl_counts(self.input_clock_hz, speed)
    }

    /// Check the TX abort source register and map to HalError.
    fn check_abort_source(&self, abrt_src: u32) -> HalError {
        if abrt_src & ABRT_LOST != 0 {
            HalError::ArbitrationLost
        } else if abrt_src
            & (ABRT_7B_ADDR_NOACK | ABRT_10ADDR1_NOACK | ABRT_10ADDR2_NOACK | ABRT_TXDATA_NOACK)
            != 0
        {
            HalError::NackReceived
        } else {
            HalError::HardwareError
        }
    }

    /// Validate the I2C configuration parameters.
    fn validate_config(config: &I2cConfig) -> Result<(), HalError> {
        if config.timeout_us == 0 {
            return Err(HalError::InvalidConfig);
        }
        Ok(())
    }
}

impl I2cController for DesignwareI2c {
    fn init(&mut self, config: I2cConfig) -> Result<(), HalError> {
        Self::validate_config(&config)?;

        // Disable controller before configuring.
        self.disable()?;

        // Set IC_CON.
        let con = Self::build_ic_con(&config);
        self.write_reg(IC_CON, con)?;

        // Set SCL timing based on speed.
        let (hcnt, lcnt) = self.compute_timing(config.speed);
        match config.speed {
            I2cSpeed::Standard => {
                self.write_reg(IC_SS_SCL_HCNT, hcnt as u32)?;
                self.write_reg(IC_SS_SCL_LCNT, lcnt as u32)?;
            }
            I2cSpeed::Fast | I2cSpeed::FastPlus => {
                self.write_reg(IC_FS_SCL_HCNT, hcnt as u32)?;
                self.write_reg(IC_FS_SCL_LCNT, lcnt as u32)?;
            }
        }

        // Set SDA hold time (use default of 1 us worth of clocks).
        let sda_hold = self.input_clock_hz / 1_000_000;
        self.write_reg(IC_SDA_HOLD, sda_hold)?;

        // Set FIFO thresholds.
        self.write_reg(IC_RX_TL, 0)?;
        self.write_reg(IC_TX_TL, 0)?;

        // Mask all interrupts initially.
        self.write_reg(IC_INTR_MASK, 0)?;

        // Enable controller.
        self.enable()?;

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

        // Disable controller to set target address.
        self.disable()?;
        self.write_reg(IC_TAR, addr as u32)?;
        self.enable()?;

        // Write each byte via IC_DATA_CMD.
        for (i, &byte) in data.iter().enumerate() {
            let mut cmd = byte as u32;
            if i == data.len() - 1 {
                cmd |= IC_DATA_CMD_STOP;
            }
            self.write_reg(IC_DATA_CMD, cmd)?;
        }

        Ok(())
    }

    fn read(&mut self, addr: u16, buf: &mut [u8]) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        let config = self.config.as_ref().unwrap();
        crate::i2c::validate_addr(addr, config.address_mode)?;

        self.disable()?;
        self.write_reg(IC_TAR, addr as u32)?;
        self.enable()?;

        // Issue read commands.
        for i in 0..buf.len() {
            let mut cmd = IC_DATA_CMD_READ;
            if i == buf.len() - 1 {
                cmd |= IC_DATA_CMD_STOP;
            }
            self.write_reg(IC_DATA_CMD, cmd)?;
        }

        // Read data from RX FIFO.
        for byte in buf.iter_mut() {
            let val = self.read_reg(IC_DATA_CMD)?;
            *byte = (val & 0xFF) as u8;
        }

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

        self.disable()?;
        self.write_reg(IC_TAR, addr as u32)?;
        self.enable()?;

        // Write phase (no STOP — will use repeated START).
        for &byte in write_data {
            self.write_reg(IC_DATA_CMD, byte as u32)?;
        }

        // Read phase with RESTART.
        for i in 0..read_buf.len() {
            let mut cmd = IC_DATA_CMD_READ;
            if i == 0 {
                cmd |= IC_DATA_CMD_RESTART;
            }
            if i == read_buf.len() - 1 {
                cmd |= IC_DATA_CMD_STOP;
            }
            self.write_reg(IC_DATA_CMD, cmd)?;
        }

        for byte in read_buf.iter_mut() {
            let val = self.read_reg(IC_DATA_CMD)?;
            *byte = (val & 0xFF) as u8;
        }

        Ok(())
    }

    fn bus_recovery(&mut self) -> Result<(), HalError> {
        // Designware I2C does not have a dedicated bus recovery register.
        // In a real implementation, we would use GPIO bitbang fallback.
        // Here we attempt an abort and re-enable.
        self.write_reg(IC_ENABLE, IC_ENABLE_ENABLE | IC_ENABLE_ABORT)?;

        // Wait for abort to complete then re-enable normally.
        self.write_reg(IC_ENABLE, IC_ENABLE_ENABLE)?;

        // Clear any pending interrupts.
        let _ = self.read_reg(IC_CLR_INTR);
        let _ = self.read_reg(IC_CLR_TX_ABRT);
        Ok(())
    }

    fn reset(&mut self) -> Result<(), HalError> {
        self.disable()?;
        self.write_reg(IC_INTR_MASK, 0)?;
        let _ = self.read_reg(IC_CLR_INTR);
        self.configured = false;
        self.config = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_designware_new() {
        let drv = DesignwareI2c::new(0x4000_5400, 31);
        assert_eq!(drv.base_addr(), 0x4000_5400);
        assert_eq!(drv.irq(), 31);
        assert!(!drv.is_configured());
    }

    #[test]
    fn test_designware_with_clock() {
        let drv = DesignwareI2c::with_clock(0x4000_5400, 31, 100_000_000);
        assert_eq!(drv.input_clock_hz, 100_000_000);
    }

    #[test]
    fn test_build_ic_con_standard_7bit() {
        let config = I2cConfig {
            speed: I2cSpeed::Standard,
            address_mode: I2cAddressMode::SevenBit,
            clock_stretching: false,
            timeout_us: 10_000,
        };
        let con = DesignwareI2c::build_ic_con(&config);
        assert_ne!(con & IC_CON_MASTER_MODE, 0);
        assert_ne!(con & IC_CON_SLAVE_DISABLE, 0);
        assert_ne!(con & IC_CON_RESTART_EN, 0);
        assert_eq!(con & IC_CON_SPEED_MASK, IC_CON_SPEED_STD);
        assert_eq!(con & IC_CON_10BIT_ADDR_MASTER, 0);
    }

    #[test]
    fn test_build_ic_con_fast_10bit() {
        let config = I2cConfig {
            speed: I2cSpeed::Fast,
            address_mode: I2cAddressMode::TenBit,
            clock_stretching: true,
            timeout_us: 5_000,
        };
        let con = DesignwareI2c::build_ic_con(&config);
        assert_eq!(con & IC_CON_SPEED_MASK, IC_CON_SPEED_FAST);
        assert_ne!(con & IC_CON_10BIT_ADDR_MASTER, 0);
    }

    #[test]
    fn test_build_ic_con_fast_plus() {
        let config = I2cConfig {
            speed: I2cSpeed::FastPlus,
            address_mode: I2cAddressMode::SevenBit,
            clock_stretching: false,
            timeout_us: 5_000,
        };
        let con = DesignwareI2c::build_ic_con(&config);
        // FastPlus also uses FAST speed bits in Designware
        assert_eq!(con & IC_CON_SPEED_MASK, IC_CON_SPEED_FAST);
    }

    #[test]
    fn test_validate_config_zero_timeout() {
        let config = I2cConfig {
            speed: I2cSpeed::Standard,
            address_mode: I2cAddressMode::SevenBit,
            clock_stretching: false,
            timeout_us: 0,
        };
        assert_eq!(
            DesignwareI2c::validate_config(&config),
            Err(HalError::InvalidConfig)
        );
    }

    #[test]
    fn test_validate_config_valid() {
        let config = I2cConfig {
            speed: I2cSpeed::Standard,
            address_mode: I2cAddressMode::SevenBit,
            clock_stretching: false,
            timeout_us: 10_000,
        };
        assert!(DesignwareI2c::validate_config(&config).is_ok());
    }

    #[test]
    fn test_init_returns_mmio_error() {
        // On host, all MMIO access returns MmioError.
        let mut drv = DesignwareI2c::new(0x4000_5400, 31);
        let config = I2cConfig {
            speed: I2cSpeed::Standard,
            address_mode: I2cAddressMode::SevenBit,
            clock_stretching: false,
            timeout_us: 10_000,
        };
        assert_eq!(drv.init(config), Err(HalError::MmioError));
    }

    #[test]
    fn test_write_not_configured() {
        let mut drv = DesignwareI2c::new(0x4000_5400, 31);
        assert_eq!(drv.write(0x50, &[0x00]), Err(HalError::InitFailed));
    }

    #[test]
    fn test_read_not_configured() {
        let mut drv = DesignwareI2c::new(0x4000_5400, 31);
        let mut buf = [0u8; 4];
        assert_eq!(drv.read(0x50, &mut buf), Err(HalError::InitFailed));
    }

    #[test]
    fn test_write_read_not_configured() {
        let mut drv = DesignwareI2c::new(0x4000_5400, 31);
        let mut buf = [0u8; 2];
        assert_eq!(
            drv.write_read(0x50, &[0x00], &mut buf),
            Err(HalError::InitFailed)
        );
    }

    #[test]
    fn test_check_abort_source_arb_lost() {
        let drv = DesignwareI2c::new(0x4000_5400, 31);
        assert_eq!(drv.check_abort_source(ABRT_LOST), HalError::ArbitrationLost);
    }

    #[test]
    fn test_check_abort_source_addr_nack() {
        let drv = DesignwareI2c::new(0x4000_5400, 31);
        assert_eq!(
            drv.check_abort_source(ABRT_7B_ADDR_NOACK),
            HalError::NackReceived
        );
        assert_eq!(
            drv.check_abort_source(ABRT_10ADDR1_NOACK),
            HalError::NackReceived
        );
    }

    #[test]
    fn test_check_abort_source_data_nack() {
        let drv = DesignwareI2c::new(0x4000_5400, 31);
        assert_eq!(
            drv.check_abort_source(ABRT_TXDATA_NOACK),
            HalError::NackReceived
        );
    }

    #[test]
    fn test_check_abort_source_unknown() {
        let drv = DesignwareI2c::new(0x4000_5400, 31);
        assert_eq!(drv.check_abort_source(0), HalError::HardwareError);
    }

    #[test]
    fn test_reset_returns_mmio_error() {
        let mut drv = DesignwareI2c::new(0x4000_5400, 31);
        assert_eq!(drv.reset(), Err(HalError::MmioError));
    }

    #[test]
    fn test_constructor_fields() {
        let drv = DesignwareI2c::new(0xBEEF_0000, 77);
        assert_eq!(drv.base_addr(), 0xBEEF_0000);
        assert_eq!(drv.irq(), 77);
        assert!(!drv.is_configured());
        assert!(drv.config.is_none());
        assert_eq!(drv.input_clock_hz, DEFAULT_INPUT_CLOCK_HZ);
    }

    #[test]
    fn test_with_clock_constructor() {
        let drv = DesignwareI2c::with_clock(0x5000_0000, 12, 96_000_000);
        assert_eq!(drv.base_addr(), 0x5000_0000);
        assert_eq!(drv.irq(), 12);
        assert_eq!(drv.input_clock_hz, 96_000_000);
        assert!(!drv.is_configured());
    }

    #[test]
    fn test_7bit_address_validation_write() {
        let mut drv = DesignwareI2c::new(0x4000_5400, 31);
        drv.configured = true;
        drv.config = Some(I2cConfig {
            speed: I2cSpeed::Standard,
            address_mode: I2cAddressMode::SevenBit,
            clock_stretching: false,
            timeout_us: 10_000,
        });
        // Address 0x00 (reserved) should fail.
        assert_eq!(drv.write(0x00, &[0xAB]), Err(HalError::InvalidConfig));
        // Address 0x78 (reserved) should fail.
        assert_eq!(drv.write(0x78, &[0xAB]), Err(HalError::InvalidConfig));
        // Address 0x50 (valid) should proceed to MMIO → MmioError.
        assert_eq!(drv.write(0x50, &[0xAB]), Err(HalError::MmioError));
    }

    #[test]
    fn test_7bit_address_validation_read() {
        let mut drv = DesignwareI2c::new(0x4000_5400, 31);
        drv.configured = true;
        drv.config = Some(I2cConfig {
            speed: I2cSpeed::Standard,
            address_mode: I2cAddressMode::SevenBit,
            clock_stretching: false,
            timeout_us: 10_000,
        });
        let mut buf = [0u8; 2];
        assert_eq!(drv.read(0x00, &mut buf), Err(HalError::InvalidConfig));
        assert_eq!(drv.read(0x50, &mut buf), Err(HalError::MmioError));
    }

    #[test]
    fn test_10bit_address_build_ic_con() {
        let config = I2cConfig {
            speed: I2cSpeed::Standard,
            address_mode: I2cAddressMode::TenBit,
            clock_stretching: false,
            timeout_us: 10_000,
        };
        let con = DesignwareI2c::build_ic_con(&config);
        assert_ne!(con & IC_CON_10BIT_ADDR_MASTER, 0);
    }

    #[test]
    fn test_check_abort_source_10bit_addr2_nack() {
        let drv = DesignwareI2c::new(0x4000_5400, 31);
        assert_eq!(
            drv.check_abort_source(ABRT_10ADDR2_NOACK),
            HalError::NackReceived
        );
    }

    #[test]
    fn test_check_abort_source_combined_arb_and_nack() {
        let drv = DesignwareI2c::new(0x4000_5400, 31);
        // When both arbitration lost and NACK are set, arbitration lost takes priority.
        assert_eq!(
            drv.check_abort_source(ABRT_LOST | ABRT_7B_ADDR_NOACK),
            HalError::ArbitrationLost
        );
    }

    #[test]
    fn test_validate_config_valid_timeout() {
        let config = I2cConfig {
            speed: I2cSpeed::Fast,
            address_mode: I2cAddressMode::SevenBit,
            clock_stretching: true,
            timeout_us: 1,
        };
        assert!(DesignwareI2c::validate_config(&config).is_ok());
    }

    #[test]
    fn test_init_zero_timeout_returns_invalid_config() {
        let mut drv = DesignwareI2c::new(0x4000_5400, 31);
        let config = I2cConfig {
            speed: I2cSpeed::Standard,
            address_mode: I2cAddressMode::SevenBit,
            clock_stretching: false,
            timeout_us: 0,
        };
        assert_eq!(drv.init(config), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_bus_recovery_returns_mmio_error() {
        let mut drv = DesignwareI2c::new(0x4000_5400, 31);
        assert_eq!(drv.bus_recovery(), Err(HalError::MmioError));
    }
}
