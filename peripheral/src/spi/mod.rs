// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SPI peripheral drivers.
//!
//! Provides platform-specific SPI controller implementations for:
//! - ARM PL022/Designware SSI (MMIO)
//! - RISC-V SiFive SPI (MMIO)
//! - Xilinx AXI Quad SPI (FPGA)
//!
//! All drivers implement the [`SpiController`] trait from `smallaios_kernel::hal`.

#![allow(dead_code)]

pub mod arm_mmio;
pub mod axi_quad_spi;
pub mod riscv_mmio;

pub use smallaios_kernel::hal::{
    DmaToken, HalError, SpiBitOrder, SpiConfig, SpiController, SpiMode,
};

// ---------------------------------------------------------------------------
// Shared SPI helpers
// ---------------------------------------------------------------------------

/// Maximum number of chip-select lines typically supported.
pub const MAX_CS_LINES: u8 = 8;

/// Compute the clock divider to achieve the target SPI clock frequency.
///
/// The actual SPI clock will be `input_hz / divider`. The divider is rounded
/// up so the actual clock does not exceed `target_hz`.
///
/// Returns the divider value (minimum 2, as most SPI controllers require
/// an even divider >= 2).
///
/// # Examples
///
/// ```ignore
/// assert_eq!(compute_clock_divider(48_000_000, 12_000_000), 4);
/// assert_eq!(compute_clock_divider(48_000_000, 10_000_000), 6);
/// ```
pub fn compute_clock_divider(input_hz: u32, target_hz: u32) -> u32 {
    if target_hz == 0 || input_hz == 0 {
        return 2;
    }

    // Round up: divider = ceil(input / target)
    let divider = input_hz.div_ceil(target_hz);

    // Most SPI controllers require even divider >= 2.
    let divider = divider.max(2);

    // Round up to even.
    if !divider.is_multiple_of(2) {
        divider + 1
    } else {
        divider
    }
}

/// Validate the SPI word size.
///
/// Returns `true` if the word size is a supported value (4, 8, 16, or 32 bits).
pub fn validate_word_size(bits: u8) -> bool {
    matches!(bits, 4 | 8 | 16 | 32)
}

/// Validate a chip-select index.
pub fn validate_cs(cs: u8) -> Result<(), HalError> {
    if cs >= MAX_CS_LINES {
        Err(HalError::InvalidConfig)
    } else {
        Ok(())
    }
}

/// Convert SPI mode to CPOL/CPHA tuple.
///
/// Returns `(cpol, cpha)` as booleans.
pub fn mode_to_cpol_cpha(mode: SpiMode) -> (bool, bool) {
    match mode {
        SpiMode::Mode0 => (false, false),
        SpiMode::Mode1 => (false, true),
        SpiMode::Mode2 => (true, false),
        SpiMode::Mode3 => (true, true),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Clock divider computation
    // -----------------------------------------------------------------------

    #[test]
    fn test_compute_clock_divider_exact() {
        // 48 MHz / 12 MHz = 4 (exact, even)
        assert_eq!(compute_clock_divider(48_000_000, 12_000_000), 4);
    }

    #[test]
    fn test_compute_clock_divider_round_up() {
        // 48 MHz / 10 MHz = 4.8 → round up to 5 → round to even = 6
        assert_eq!(compute_clock_divider(48_000_000, 10_000_000), 6);
    }

    #[test]
    fn test_compute_clock_divider_minimum() {
        // Target higher than input → divider should be minimum 2.
        assert_eq!(compute_clock_divider(1_000_000, 100_000_000), 2);
    }

    #[test]
    fn test_compute_clock_divider_zero_target() {
        assert_eq!(compute_clock_divider(48_000_000, 0), 2);
    }

    #[test]
    fn test_compute_clock_divider_zero_input() {
        assert_eq!(compute_clock_divider(0, 10_000_000), 2);
    }

    #[test]
    fn test_compute_clock_divider_1mhz() {
        // 48 MHz / 1 MHz = 48 (exact, even)
        assert_eq!(compute_clock_divider(48_000_000, 1_000_000), 48);
    }

    // -----------------------------------------------------------------------
    // Word size validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_word_size_valid() {
        assert!(validate_word_size(4));
        assert!(validate_word_size(8));
        assert!(validate_word_size(16));
        assert!(validate_word_size(32));
    }

    #[test]
    fn test_validate_word_size_invalid() {
        assert!(!validate_word_size(0));
        assert!(!validate_word_size(1));
        assert!(!validate_word_size(7));
        assert!(!validate_word_size(12));
        assert!(!validate_word_size(24));
        assert!(!validate_word_size(64));
    }

    // -----------------------------------------------------------------------
    // CS validation
    // -----------------------------------------------------------------------

    #[test]
    fn test_validate_cs_valid() {
        for cs in 0..MAX_CS_LINES {
            assert!(validate_cs(cs).is_ok());
        }
    }

    #[test]
    fn test_validate_cs_invalid() {
        assert_eq!(validate_cs(8), Err(HalError::InvalidConfig));
        assert_eq!(validate_cs(255), Err(HalError::InvalidConfig));
    }

    // -----------------------------------------------------------------------
    // Mode conversion
    // -----------------------------------------------------------------------

    #[test]
    fn test_mode_to_cpol_cpha() {
        assert_eq!(mode_to_cpol_cpha(SpiMode::Mode0), (false, false));
        assert_eq!(mode_to_cpol_cpha(SpiMode::Mode1), (false, true));
        assert_eq!(mode_to_cpol_cpha(SpiMode::Mode2), (true, false));
        assert_eq!(mode_to_cpol_cpha(SpiMode::Mode3), (true, true));
    }
}
