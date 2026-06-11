// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Timer abstraction for scheduler tick and time measurement.
//!
//! Architecture-specific timer drivers:
//! - x86-64: Local APIC timer + TSC for high-resolution timestamps
//! - ARM64: Generic Timer (CNTPCT_EL0 / CNTP_TVAL_EL0)
//!
//! The timer provides:
//! - Periodic tick for scheduler preemption checks
//! - Monotonic timestamps for operator budget measurement
//! - Watchdog tick integration

use core::sync::atomic::{AtomicU64, Ordering};

/// Global monotonic tick counter (incremented by timer interrupt).
static TICK_COUNT: AtomicU64 = AtomicU64::new(0);

/// Timer frequency in Hz (set during calibration).
static TIMER_FREQ_HZ: AtomicU64 = AtomicU64::new(0);

/// Get the current tick count.
pub fn tick_count() -> u64 {
    TICK_COUNT.load(Ordering::Acquire)
}

/// Increment the tick count (called from timer interrupt handler).
pub fn tick() {
    TICK_COUNT.fetch_add(1, Ordering::Release);
}

/// Set the timer frequency (called during calibration).
pub fn set_frequency(hz: u64) {
    TIMER_FREQ_HZ.store(hz, Ordering::Release);
}

/// Get the timer frequency in Hz.
pub fn frequency() -> u64 {
    TIMER_FREQ_HZ.load(Ordering::Acquire)
}

/// Convert ticks to microseconds.
pub fn ticks_to_us(ticks: u64) -> u64 {
    let freq = frequency();
    if freq == 0 {
        return 0;
    }
    ticks * 1_000_000 / freq
}

/// Convert microseconds to ticks.
pub fn us_to_ticks(us: u64) -> u64 {
    let freq = frequency();
    us * freq / 1_000_000
}

/// A timestamp from the high-resolution timer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Timestamp(pub u64);

impl Timestamp {
    /// Get the current timestamp.
    ///
    /// On x86-64, this reads TSC. On ARM64, this reads CNTPCT_EL0.
    /// In test mode (under Miri, or on other architectures), it falls
    /// back to the tick counter — Miri cannot interpret inline assembly.
    pub fn now() -> Self {
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        {
            Self(read_tsc())
        }
        #[cfg(all(target_arch = "aarch64", not(miri)))]
        {
            Self(read_cntpct())
        }
        #[cfg(any(miri, not(any(target_arch = "x86_64", target_arch = "aarch64"))))]
        {
            // Fallback for host testing and Miri (no inline-asm support)
            Self(tick_count())
        }
    }

    /// Get the raw value.
    pub fn as_u64(self) -> u64 {
        self.0
    }

    /// Duration since another timestamp, in raw timer ticks.
    pub fn elapsed_since(self, earlier: Timestamp) -> u64 {
        self.0.saturating_sub(earlier.0)
    }

    /// Duration since another timestamp, in microseconds.
    pub fn elapsed_us_since(self, earlier: Timestamp) -> u64 {
        ticks_to_us(self.elapsed_since(earlier))
    }
}

/// Read x86-64 TSC (Time Stamp Counter).
///
/// Not compiled under Miri: Miri cannot interpret inline assembly, so
/// [`Timestamp::now`] falls back to [`tick_count`] there instead.
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[inline]
fn read_tsc() -> u64 {
    let lo: u32;
    let hi: u32;
    unsafe {
        core::arch::asm!(
            "rdtsc",
            out("eax") lo,
            out("edx") hi,
            options(nomem, nostack),
        );
    }
    ((hi as u64) << 32) | (lo as u64)
}

/// Read ARM64 generic timer counter.
///
/// Not compiled under Miri: Miri cannot interpret inline assembly, so
/// [`Timestamp::now`] falls back to [`tick_count`] there instead.
#[cfg(all(target_arch = "aarch64", not(miri)))]
#[inline]
fn read_cntpct() -> u64 {
    let val: u64;
    unsafe {
        core::arch::asm!(
            "mrs {}, cntpct_el0",
            out(reg) val,
            options(nomem, nostack),
        );
    }
    val
}

/// Read ARM64 generic timer frequency.
#[cfg(target_arch = "aarch64")]
pub fn read_cntfrq() -> u64 {
    let val: u64;
    unsafe {
        core::arch::asm!(
            "mrs {}, cntfrq_el0",
            out(reg) val,
            options(nomem, nostack),
        );
    }
    val
}

/// Configure a one-shot timer interrupt after `ticks` timer ticks.
///
/// # x86-64 (Local APIC Timer)
/// Programs the APIC timer in one-shot mode.
///
/// # ARM64 (Generic Timer)
/// Programs CNTP_TVAL_EL0 and enables the timer.
pub fn set_oneshot(_ticks: u64) {
    // Architecture-specific implementation provided by arch crates.
    // This is a placeholder that will be called by the arch-specific
    // timer initialization code.
}

/// Scheduler tick rate target (1000 Hz = 1 ms ticks).
pub const TICK_RATE_HZ: u64 = 1000;

/// Scheduler tick period in microseconds (1000 us = 1 ms).
pub const TICK_PERIOD_US: u64 = 1_000_000 / TICK_RATE_HZ;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tick_count() {
        let initial = tick_count();
        tick();
        tick();
        assert_eq!(tick_count(), initial + 2);
    }

    #[test]
    fn test_frequency() {
        set_frequency(1_000_000); // 1 MHz
        assert_eq!(frequency(), 1_000_000);
    }

    #[test]
    fn test_ticks_to_us() {
        set_frequency(1_000_000); // 1 MHz = 1 tick per us
        assert_eq!(ticks_to_us(1000), 1000);

        set_frequency(2_000_000); // 2 MHz = 0.5 us per tick
        assert_eq!(ticks_to_us(2000), 1000);
    }

    #[test]
    fn test_us_to_ticks() {
        set_frequency(1_000_000);
        assert_eq!(us_to_ticks(1000), 1000);
    }

    #[test]
    fn test_timestamp_elapsed() {
        let t1 = Timestamp(100);
        let t2 = Timestamp(250);
        assert_eq!(t2.elapsed_since(t1), 150);
    }

    #[test]
    fn test_timestamp_elapsed_us() {
        set_frequency(1_000_000);
        let t1 = Timestamp(0);
        let t2 = Timestamp(5000);
        assert_eq!(t2.elapsed_us_since(t1), 5000);
    }

    #[test]
    fn test_zero_frequency() {
        set_frequency(0);
        assert_eq!(ticks_to_us(100), 0);
    }

    #[test]
    fn test_tick_constants() {
        assert_eq!(TICK_RATE_HZ, 1000);
        assert_eq!(TICK_PERIOD_US, 1000);
    }

    #[test]
    fn test_timestamp_saturating_sub() {
        // elapsed_since uses saturating_sub, so earlier > later = 0
        let t1 = Timestamp(100);
        let t2 = Timestamp(50);
        assert_eq!(t1.elapsed_since(t2), 50);
        assert_eq!(t2.elapsed_since(t1), 0); // saturating
    }

    #[test]
    fn test_timestamp_raw_value() {
        let ts = Timestamp(12345);
        assert_eq!(ts.as_u64(), 12345);
    }

    #[test]
    fn test_timestamp_now() {
        // Timestamp::now() should return a non-zero value on any platform.
        // On x86_64 it reads TSC, on aarch64 it reads CNTPCT; under Miri
        // (no inline-asm support) it falls back to tick_count().
        let ts = Timestamp::now();
        // On x86_64 host, TSC is always > 0 after boot; on the tick-counter
        // fallback (Miri / other arches), the count may be 0.
        #[cfg(all(target_arch = "x86_64", not(miri)))]
        assert!(ts.as_u64() > 0);
        #[cfg(any(miri, not(target_arch = "x86_64")))]
        {
            let _ = ts; // just verify it doesn't panic
        }
    }

    #[test]
    fn test_timestamp_ordering() {
        let t1 = Timestamp(10);
        let t2 = Timestamp(20);
        let t3 = Timestamp(10);
        assert!(t1 < t2);
        assert!(t2 > t1);
        assert_eq!(t1, t3);
    }

    #[test]
    fn test_us_to_ticks_zero_freq() {
        set_frequency(0);
        // us * 0 / 1_000_000 = 0
        assert_eq!(us_to_ticks(1000), 0);
    }

    #[test]
    fn test_large_frequency() {
        set_frequency(3_600_000_000); // 3.6 GHz TSC
        let ticks = us_to_ticks(1000); // 1 ms
        assert_eq!(ticks, 3_600_000);
        let us = ticks_to_us(3_600_000);
        assert_eq!(us, 1000);
    }
}
