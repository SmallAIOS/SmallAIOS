// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! I2C peripheral drivers.
//!
//! Provides platform-specific I2C controller implementations for:
//! - ARM Designware I2C (MMIO)
//! - RISC-V SiFive/OpenCores I2C (MMIO)
//! - Xilinx AXI IIC (FPGA)
//! - GPIO bitbang fallback
//!
//! All drivers implement the [`I2cController`] trait from `smallaios_kernel::hal`.

#![allow(dead_code)]

pub mod arm_mmio;
pub mod axi_iic;
pub mod bitbang;
pub mod riscv_mmio;

pub use smallaios_kernel::hal::{HalError, I2cAddressMode, I2cConfig, I2cController, I2cSpeed};

// ---------------------------------------------------------------------------
// Shared address validation helpers
// ---------------------------------------------------------------------------

/// Maximum valid 7-bit I2C address (0x00–0x77, excluding reserved ranges).
pub const I2C_7BIT_MAX: u16 = 0x77;

/// Maximum valid 10-bit I2C address.
pub const I2C_10BIT_MAX: u16 = 0x3FF;

/// Reserved 7-bit addresses per I2C specification.
/// 0x00–0x07 and 0x78–0x7F are reserved for special purposes.
const RESERVED_LOW: u16 = 0x08;
const RESERVED_HIGH: u16 = 0x77;

/// Validate a 7-bit I2C address.
///
/// Returns `true` if the address is in the valid range 0x08..=0x77
/// (excludes reserved addresses 0x00-0x07 and 0x78-0x7F per the I2C spec).
///
/// # Examples
///
/// ```ignore
/// assert!(validate_7bit_addr(0x50)); // EEPROM
/// assert!(!validate_7bit_addr(0x00)); // General call (reserved)
/// assert!(!validate_7bit_addr(0x78)); // 10-bit prefix (reserved)
/// ```
pub fn validate_7bit_addr(addr: u16) -> bool {
    (RESERVED_LOW..=RESERVED_HIGH).contains(&addr)
}

/// Validate a 10-bit I2C address.
///
/// Returns `true` if the address is in the valid range 0x000..=0x3FF.
pub fn validate_10bit_addr(addr: u16) -> bool {
    addr <= I2C_10BIT_MAX
}

/// Validate an address against the configured addressing mode.
///
/// Returns `Ok(())` if the address is valid for the given mode, or
/// `Err(HalError::InvalidConfig)` if out of range.
pub fn validate_addr(addr: u16, mode: I2cAddressMode) -> Result<(), HalError> {
    let valid = match mode {
        I2cAddressMode::SevenBit => validate_7bit_addr(addr),
        I2cAddressMode::TenBit => validate_10bit_addr(addr),
    };
    if valid {
        Ok(())
    } else {
        Err(HalError::InvalidConfig)
    }
}

/// I2C bus recovery sequence description.
///
/// When the SDA line is stuck low (e.g., a target device is holding it after
/// a partially-completed transaction), the controller must:
///
/// 1. Drive SCL high (release).
/// 2. Generate 9 clock pulses on SCL while monitoring SDA.
/// 3. After each rising edge of SCL, check if SDA is released (high).
/// 4. If SDA is released, generate a STOP condition (SDA low->high while SCL high).
/// 5. If SDA is still stuck after 9 clocks, the bus is unrecoverable without reset.
///
/// This is implemented in the bitbang driver and can be called by MMIO drivers
/// as a fallback when their hardware bus recovery mechanism fails.
pub fn bus_recovery_clock_count() -> u8 {
    9
}

/// Compute the SCL high/low count values for a given input clock and target speed.
///
/// Returns `(high_count, low_count)` for the timer registers. The SCL period
/// is `(high_count + low_count) * input_clock_period`.
///
/// For standard mode (100 kHz): tHIGH >= 4.0 us, tLOW >= 4.7 us
/// For fast mode (400 kHz):     tHIGH >= 0.6 us, tLOW >= 1.3 us
/// For fast-mode plus (1 MHz):  tHIGH >= 0.26 us, tLOW >= 0.5 us
pub fn compute_scl_counts(input_clock_hz: u32, speed: I2cSpeed) -> (u16, u16) {
    let (target_hz, high_ratio_pct) = match speed {
        I2cSpeed::Standard => (100_000u32, 46), // 4.0/(4.0+4.7) ~ 46%
        I2cSpeed::Fast => (400_000u32, 32),     // 0.6/(0.6+1.3) ~ 32%
        I2cSpeed::FastPlus => (1_000_000u32, 34), // 0.26/(0.26+0.5) ~ 34%
    };

    if input_clock_hz == 0 || target_hz == 0 {
        return (1, 1);
    }

    let total_count = input_clock_hz / target_hz;
    let high_count = (total_count * high_ratio_pct / 100).max(1) as u16;
    let low_count = (total_count - high_count as u32).max(1) as u16;
    (high_count, low_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // 7-bit address validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_7bit_addr_typical_devices() {
        // Common I2C device addresses
        assert!(validate_7bit_addr(0x50)); // EEPROM (AT24C)
        assert!(validate_7bit_addr(0x68)); // MPU-6050 IMU
        assert!(validate_7bit_addr(0x76)); // BME280 sensor
        assert!(validate_7bit_addr(0x27)); // PCF8574 expander
        assert!(validate_7bit_addr(0x3C)); // SSD1306 OLED
        assert!(validate_7bit_addr(0x48)); // TMP102 temp sensor
    }

    #[test]
    fn test_validate_7bit_addr_boundaries() {
        assert!(!validate_7bit_addr(0x00)); // General call
        assert!(!validate_7bit_addr(0x01)); // CBUS
        assert!(!validate_7bit_addr(0x07)); // Reserved
        assert!(validate_7bit_addr(0x08)); // First valid
        assert!(validate_7bit_addr(0x77)); // Last valid
        assert!(!validate_7bit_addr(0x78)); // 10-bit prefix start
        assert!(!validate_7bit_addr(0x7F)); // Reserved
    }

    #[test]
    fn test_validate_7bit_addr_reserved_ranges() {
        for addr in 0x00..RESERVED_LOW {
            assert!(
                !validate_7bit_addr(addr),
                "0x{:02X} should be reserved",
                addr
            );
        }
        for addr in (RESERVED_HIGH + 1)..=0xFF {
            assert!(
                !validate_7bit_addr(addr),
                "0x{:02X} should be reserved",
                addr
            );
        }
    }

    // -----------------------------------------------------------------------
    // 10-bit address validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_10bit_addr_valid() {
        assert!(validate_10bit_addr(0x000));
        assert!(validate_10bit_addr(0x100));
        assert!(validate_10bit_addr(0x3FF));
    }

    #[test]
    fn test_validate_10bit_addr_invalid() {
        assert!(!validate_10bit_addr(0x400));
        assert!(!validate_10bit_addr(0xFFFF));
    }

    // -----------------------------------------------------------------------
    // Combined address validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_addr_7bit_mode() {
        assert!(validate_addr(0x50, I2cAddressMode::SevenBit).is_ok());
        assert!(validate_addr(0x00, I2cAddressMode::SevenBit).is_err());
        assert!(validate_addr(0x78, I2cAddressMode::SevenBit).is_err());
    }

    #[test]
    fn test_validate_addr_10bit_mode() {
        assert!(validate_addr(0x3FF, I2cAddressMode::TenBit).is_ok());
        assert!(validate_addr(0x400, I2cAddressMode::TenBit).is_err());
    }

    // -----------------------------------------------------------------------
    // SCL count computation
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_scl_counts_standard_48mhz() {
        let (hcnt, lcnt) = compute_scl_counts(48_000_000, I2cSpeed::Standard);
        // 48 MHz / 100 kHz = 480 total
        // High ~ 46% of 480 = 220, Low ~ 260
        assert!(hcnt > 0);
        assert!(lcnt > 0);
        assert_eq!(hcnt as u32 + lcnt as u32, 480);
    }

    #[test]
    fn test_compute_scl_counts_fast_48mhz() {
        let (hcnt, lcnt) = compute_scl_counts(48_000_000, I2cSpeed::Fast);
        // 48 MHz / 400 kHz = 120 total
        assert!(hcnt > 0);
        assert!(lcnt > 0);
        assert_eq!(hcnt as u32 + lcnt as u32, 120);
    }

    #[test]
    fn test_compute_scl_counts_zero_clock() {
        let (hcnt, lcnt) = compute_scl_counts(0, I2cSpeed::Standard);
        assert_eq!(hcnt, 1);
        assert_eq!(lcnt, 1);
    }

    #[test]
    fn test_bus_recovery_clock_count() {
        assert_eq!(bus_recovery_clock_count(), 9);
    }
}
