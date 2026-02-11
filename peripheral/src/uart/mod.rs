// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! UART peripheral drivers.
//!
//! Platform-specific UART controller implementations:
//! - ARM PL011 (MMIO)
//! - NS16550A (MMIO)
//! - SiFive UART (MMIO)
//! - Xilinx AXI UART Lite (FPGA)
//! - NMEA 0183 sentence parser (protocol layer)

#![allow(dead_code)]

pub mod axi_uart_lite;
pub mod nmea;
pub mod ns16550a;
pub mod pl011;
pub mod sifive;

pub use smallaios_kernel::hal::{
    HalError, UartConfig, UartController, UartFlowControl, UartIrqSource, UartParity, UartRxMode,
    UartStopBits,
};

/// Compute integer and fractional baud rate divisors.
///
/// For a UART clock of `clock_hz` and desired `baud` rate:
/// - Integer divisor: `IBRD = clock_hz / (16 * baud)`
/// - Fractional divisor: `FBRD = round((frac_part * 64) + 0.5)`
///
/// Returns `(ibrd, fbrd)`.
pub fn compute_baud_divisor(clock_hz: u32, baud: u32) -> (u16, u16) {
    if baud == 0 || clock_hz == 0 {
        return (1, 0);
    }
    let div_times_64 = (clock_hz as u64 * 4) / baud as u64; // = clock / (16*baud) * 64
    let ibrd = (div_times_64 / 64) as u16;
    let fbrd = (div_times_64 % 64) as u16;
    (if ibrd == 0 { 1 } else { ibrd }, fbrd)
}

/// Validate baud rate is within reasonable range.
pub fn validate_baud(baud: u32) -> bool {
    (300..=4_000_000).contains(&baud)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_baud_divisor_115200() {
        let (ibrd, fbrd) = compute_baud_divisor(48_000_000, 115200);
        // 48M / (16*115200) = 26.0416... → IBRD=26, FBRD≈3
        assert_eq!(ibrd, 26);
        assert!(fbrd <= 5);
    }

    #[test]
    fn test_compute_baud_divisor_9600() {
        let (ibrd, _fbrd) = compute_baud_divisor(48_000_000, 9600);
        // 48M / (16*9600) = 312.5
        assert_eq!(ibrd, 312);
    }

    #[test]
    fn test_compute_baud_divisor_zero() {
        let (ibrd, fbrd) = compute_baud_divisor(48_000_000, 0);
        assert_eq!(ibrd, 1);
        assert_eq!(fbrd, 0);
    }

    #[test]
    fn test_validate_baud() {
        assert!(validate_baud(9600));
        assert!(validate_baud(115200));
        assert!(validate_baud(1_000_000));
        assert!(!validate_baud(0));
        assert!(!validate_baud(100));
        assert!(!validate_baud(5_000_000));
    }
}
