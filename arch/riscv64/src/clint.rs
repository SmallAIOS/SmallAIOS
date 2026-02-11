// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CLINT (Core Local Interruptor) timer and IPI driver (23.5).
//!
//! QEMU virt machine CLINT base: 0x0200_0000
//!
//! Memory map:
//! - 0x0000: MSIP registers (4 bytes per hart) — software interrupts
//! - 0x4000: MTIMECMP registers (8 bytes per hart) — timer compare
//! - 0xBFF8: MTIME register (8 bytes) — shared monotonic counter
//!
//! Note: mtime and mtimecmp are M-mode registers. In S-mode, we access
//! the timer via SBI calls (sbi_set_timer). Direct CLINT access is used
//! only when running with M-mode privileges or if OpenSBI exposes them.

/// CLINT base address (QEMU virt).
const CLINT_BASE: usize = 0x0200_0000;

/// MSIP register offset (per hart: 4 bytes each).
const MSIP_BASE: usize = 0x0000;
/// MTIMECMP register offset (per hart: 8 bytes each).
const MTIMECMP_BASE: usize = 0x4000;
/// MTIME register offset (shared).
const MTIME_OFFSET: usize = 0xBFF8;

/// Default timer frequency on QEMU virt: 10 MHz.
pub const DEFAULT_FREQUENCY: u64 = 10_000_000;

/// Read the mtime register (monotonic counter shared across all harts).
///
/// # Safety
/// CLINT must be identity-mapped.
pub unsafe fn read_mtime() -> u64 {
    let ptr = (CLINT_BASE + MTIME_OFFSET) as *const u64;
    core::ptr::read_volatile(ptr)
}

/// Read the mtimecmp register for a specific hart.
///
/// # Safety
/// CLINT must be identity-mapped.
pub unsafe fn read_mtimecmp(hart_id: usize) -> u64 {
    let ptr = (CLINT_BASE + MTIMECMP_BASE + hart_id * 8) as *const u64;
    core::ptr::read_volatile(ptr)
}

/// Write the mtimecmp register for a specific hart.
///
/// The timer interrupt fires when mtime >= mtimecmp.
///
/// # Safety
/// CLINT must be identity-mapped.
pub unsafe fn write_mtimecmp(hart_id: usize, value: u64) {
    let ptr = (CLINT_BASE + MTIMECMP_BASE + hart_id * 8) as *mut u64;
    // Write u64::MAX first to prevent spurious interrupts during update.
    core::ptr::write_volatile(ptr, u64::MAX);
    core::arch::asm!("fence w,w", options(nostack));
    core::ptr::write_volatile(ptr, value);
}

/// Schedule the next timer interrupt at mtime + interval.
///
/// # Safety
/// CLINT must be identity-mapped.
pub unsafe fn schedule_tick(hart_id: usize, interval: u64) {
    let now = read_mtime();
    write_mtimecmp(hart_id, now.wrapping_add(interval));
}

/// Send an IPI (inter-processor interrupt) to a target hart by setting its MSIP bit.
///
/// # Safety
/// CLINT must be identity-mapped.
pub unsafe fn send_ipi(hart_id: usize) {
    let ptr = (CLINT_BASE + MSIP_BASE + hart_id * 4) as *mut u32;
    core::ptr::write_volatile(ptr, 1);
}

/// Clear the IPI pending bit for the current hart.
///
/// # Safety
/// CLINT must be identity-mapped.
pub unsafe fn clear_ipi(hart_id: usize) {
    let ptr = (CLINT_BASE + MSIP_BASE + hart_id * 4) as *mut u32;
    core::ptr::write_volatile(ptr, 0);
}

/// Convert microseconds to timer ticks at the default frequency.
pub const fn us_to_ticks(us: u64) -> u64 {
    us * DEFAULT_FREQUENCY / 1_000_000
}

/// Convert timer ticks to microseconds at the default frequency.
pub const fn ticks_to_us(ticks: u64) -> u64 {
    ticks * 1_000_000 / DEFAULT_FREQUENCY
}

/// Default scheduler tick interval: 10ms (100 Hz).
pub const TICK_INTERVAL: u64 = DEFAULT_FREQUENCY / 100;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_us_to_ticks() {
        // 10 MHz: 1 us = 10 ticks.
        assert_eq!(us_to_ticks(1), 10);
        assert_eq!(us_to_ticks(1000), 10_000);
        assert_eq!(us_to_ticks(1_000_000), 10_000_000);
    }

    #[test]
    fn test_ticks_to_us() {
        assert_eq!(ticks_to_us(10), 1);
        assert_eq!(ticks_to_us(10_000), 1000);
        assert_eq!(ticks_to_us(10_000_000), 1_000_000);
    }

    #[test]
    fn test_tick_interval() {
        // 10ms at 10 MHz = 100,000 ticks.
        assert_eq!(TICK_INTERVAL, 100_000);
    }

    #[test]
    fn test_constants() {
        assert_eq!(DEFAULT_FREQUENCY, 10_000_000);
        assert_eq!(CLINT_BASE, 0x0200_0000);
    }
}
