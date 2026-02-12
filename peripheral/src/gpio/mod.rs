// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GPIO peripheral drivers.
//!
//! Provides platform-specific GPIO controller implementations for:
//! - ARM PL061 (MMIO)
//! - RISC-V SiFive GPIO (MMIO)
//! - Xilinx AXI GPIO (FPGA)
//!
//! All drivers implement the [`GpioController`] trait from `smallaios_kernel::hal`.

#![allow(dead_code)]

pub mod arm_pl061;
pub mod axi_gpio;
pub mod riscv_mmio;

pub use smallaios_kernel::hal::{
    GpioConfig, GpioController, GpioInterruptEdge, GpioPinMode, GpioPull, HalError,
};

// ---------------------------------------------------------------------------
// Shared GPIO helpers
// ---------------------------------------------------------------------------

/// Validate a pin number against the controller's pin count.
pub fn validate_pin(pin: u8, pin_count: u8) -> Result<(), HalError> {
    if pin >= pin_count {
        Err(HalError::OutOfRange)
    } else {
        Ok(())
    }
}

/// Compute a bitmask for a single pin.
pub fn pin_mask(pin: u8) -> u32 {
    1u32 << pin
}

/// Count the number of set bits in a 32-bit mask (population count).
pub fn popcount(mask: u32) -> u8 {
    let mut n = mask;
    let mut count = 0u8;
    while n != 0 {
        count += 1;
        n &= n - 1; // Clear lowest set bit.
    }
    count
}

/// Check if a GPIO pin mode requires output capability.
pub fn is_output_mode(mode: GpioPinMode) -> bool {
    matches!(mode, GpioPinMode::Output | GpioPinMode::OpenDrain)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_pin_valid() {
        assert!(validate_pin(0, 8).is_ok());
        assert!(validate_pin(7, 8).is_ok());
        assert!(validate_pin(0, 32).is_ok());
        assert!(validate_pin(31, 32).is_ok());
    }

    #[test]
    fn test_validate_pin_invalid() {
        assert_eq!(validate_pin(8, 8), Err(HalError::OutOfRange));
        assert_eq!(validate_pin(32, 32), Err(HalError::OutOfRange));
        assert_eq!(validate_pin(0, 0), Err(HalError::OutOfRange));
    }

    #[test]
    fn test_pin_mask() {
        assert_eq!(pin_mask(0), 0x0000_0001);
        assert_eq!(pin_mask(7), 0x0000_0080);
        assert_eq!(pin_mask(31), 0x8000_0000);
    }

    #[test]
    fn test_popcount() {
        assert_eq!(popcount(0), 0);
        assert_eq!(popcount(1), 1);
        assert_eq!(popcount(0xFF), 8);
        assert_eq!(popcount(0xFFFF_FFFF), 32);
        assert_eq!(popcount(0x8000_0001), 2);
    }

    #[test]
    fn test_is_output_mode() {
        assert!(is_output_mode(GpioPinMode::Output));
        assert!(is_output_mode(GpioPinMode::OpenDrain));
        assert!(!is_output_mode(GpioPinMode::Input));
        assert!(!is_output_mode(GpioPinMode::Analog));
        assert!(!is_output_mode(GpioPinMode::AlternateFunction(0)));
    }
}
