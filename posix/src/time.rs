// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! POSIX time operations.
//!
//! Provides `clock_gettime`, `clock_getres`, `nanosleep`, and `getrandom`
//! (Linux groups getrandom with time-related syscalls in the vDSO).
//!
//! All operations are currently stubs returning `ENOSYS` after validation.
//! `clock_getres` for `Monotonic` returns 1 ns to advertise the intended
//! resolution once the hardware timer driver is wired up.

use crate::errno::Errno;

// ─── Constants ───────────────────────────────────────────────────────────────

/// Nanoseconds per second.
pub const NSEC_PER_SEC: i64 = 1_000_000_000;

/// `getrandom` flag: use /dev/random (blocking) pool.
pub const GRND_RANDOM: u32 = 2;

/// `getrandom` flag: return `EAGAIN` instead of blocking.
pub const GRND_NONBLOCK: u32 = 1;

// ─── Clock ID ───────────────────────────────────────────────────────────────

/// POSIX clock identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum ClockId {
    /// Wall-clock time (may jump due to NTP / manual set).
    Realtime = 0,
    /// Monotonically increasing clock; unaffected by NTP.
    Monotonic = 1,
    /// Per-process CPU time consumed.
    ProcessCputime = 2,
    /// Per-thread CPU time consumed.
    ThreadCputime = 3,
}

impl ClockId {
    /// Try to convert a raw integer to a `ClockId`.
    pub fn from_raw(raw: i32) -> Option<Self> {
        match raw {
            0 => Some(Self::Realtime),
            1 => Some(Self::Monotonic),
            2 => Some(Self::ProcessCputime),
            3 => Some(Self::ThreadCputime),
            _ => None,
        }
    }
}

// ─── Timespec ───────────────────────────────────────────────────────────────

/// POSIX `struct timespec` equivalent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Timespec {
    /// Whole seconds.
    pub tv_sec: i64,
    /// Nanosecond fraction (must be in `0..NSEC_PER_SEC`).
    pub tv_nsec: i64,
}

impl Timespec {
    /// Create a new `Timespec` with the given seconds and nanoseconds.
    pub const fn new(tv_sec: i64, tv_nsec: i64) -> Self {
        Self { tv_sec, tv_nsec }
    }

    /// A zero-valued timespec.
    pub const fn zero() -> Self {
        Self {
            tv_sec: 0,
            tv_nsec: 0,
        }
    }

    /// Check whether the nanosecond field is in the valid range
    /// `[0, NSEC_PER_SEC)`.
    pub const fn is_valid(&self) -> bool {
        self.tv_nsec >= 0 && self.tv_nsec < NSEC_PER_SEC
    }

    /// Return the total duration in nanoseconds (may overflow for very
    /// large values; intended for short durations).
    pub const fn to_nanos(&self) -> i64 {
        self.tv_sec * NSEC_PER_SEC + self.tv_nsec
    }
}

// ─── Clock Operations ───────────────────────────────────────────────────────

/// Read the current value of a POSIX clock.
///
/// All clocks are currently stubs and return `ENOSYS`.
/// Invalid clock IDs yield `EINVAL`.
pub fn clock_gettime(clock_id: ClockId) -> Result<Timespec, Errno> {
    // Validate that the clock_id is one we recognise (it always is if the
    // caller used the enum, but the from_raw path may produce any variant).
    match clock_id {
        ClockId::Realtime
        | ClockId::Monotonic
        | ClockId::ProcessCputime
        | ClockId::ThreadCputime => {}
    }
    // Stub: needs hardware timer / TSC integration.
    Err(Errno::ENOSYS)
}

/// Query the resolution of a POSIX clock.
///
/// `Monotonic` advertises 1 ns resolution. All other clocks return `ENOSYS`.
pub fn clock_getres(clock_id: ClockId) -> Result<Timespec, Errno> {
    match clock_id {
        ClockId::Monotonic => Ok(Timespec::new(0, 1)),
        ClockId::Realtime | ClockId::ProcessCputime | ClockId::ThreadCputime => {
            // Stub: resolution not yet determined for these clocks.
            Err(Errno::ENOSYS)
        }
    }
}

/// Suspend the calling thread for at least the specified duration.
///
/// Returns `EINVAL` if the duration has a negative field or an out-of-range
/// nanosecond value. Otherwise returns `ENOSYS` (stub).
pub fn nanosleep(duration: &Timespec) -> Result<Timespec, Errno> {
    if duration.tv_sec < 0 || duration.tv_nsec < 0 {
        return Err(Errno::EINVAL);
    }
    if !duration.is_valid() {
        return Err(Errno::EINVAL);
    }
    // Stub: needs scheduler timer integration.
    Err(Errno::ENOSYS)
}

/// Fill a buffer with cryptographically-secure random bytes.
///
/// Modelled after Linux `getrandom(2)`. Validates that the buffer pointer
/// is non-null and `len` is within bounds. Flags may include `GRND_RANDOM`
/// and/or `GRND_NONBLOCK`.
///
/// Currently a stub — returns `ENOSYS` after validation. Will be backed
/// by a PQC-grade CSPRNG once the security crate is integrated.
pub fn getrandom(buf: &mut [u8], len: usize, _flags: u32) -> Result<usize, Errno> {
    if len == 0 {
        return Ok(0);
    }
    if len > buf.len() {
        return Err(Errno::EINVAL);
    }
    // Stub: needs CSPRNG integration from the security crate.
    Err(Errno::ENOSYS)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Timespec ─────────────────────────────────────────────────────────

    #[test]
    fn timespec_new() {
        let ts = Timespec::new(42, 123_456_789);
        assert_eq!(ts.tv_sec, 42);
        assert_eq!(ts.tv_nsec, 123_456_789);
    }

    #[test]
    fn timespec_zero() {
        let ts = Timespec::zero();
        assert_eq!(ts.tv_sec, 0);
        assert_eq!(ts.tv_nsec, 0);
        assert!(ts.is_valid());
    }

    #[test]
    fn timespec_is_valid() {
        assert!(Timespec::new(0, 0).is_valid());
        assert!(Timespec::new(0, NSEC_PER_SEC - 1).is_valid());
        assert!(!Timespec::new(0, NSEC_PER_SEC).is_valid());
        assert!(!Timespec::new(0, -1).is_valid());
    }

    #[test]
    fn timespec_to_nanos() {
        let ts = Timespec::new(1, 500_000_000);
        assert_eq!(ts.to_nanos(), 1_500_000_000);
    }

    // ── ClockId ──────────────────────────────────────────────────────────

    #[test]
    fn clock_id_from_raw() {
        assert_eq!(ClockId::from_raw(0), Some(ClockId::Realtime));
        assert_eq!(ClockId::from_raw(1), Some(ClockId::Monotonic));
        assert_eq!(ClockId::from_raw(2), Some(ClockId::ProcessCputime));
        assert_eq!(ClockId::from_raw(3), Some(ClockId::ThreadCputime));
        assert_eq!(ClockId::from_raw(-1), None);
        assert_eq!(ClockId::from_raw(99), None);
    }

    // ── clock_gettime ────────────────────────────────────────────────────

    #[test]
    fn clock_gettime_stubs() {
        assert_eq!(clock_gettime(ClockId::Realtime), Err(Errno::ENOSYS));
        assert_eq!(clock_gettime(ClockId::Monotonic), Err(Errno::ENOSYS));
        assert_eq!(clock_gettime(ClockId::ProcessCputime), Err(Errno::ENOSYS));
        assert_eq!(clock_gettime(ClockId::ThreadCputime), Err(Errno::ENOSYS));
    }

    // ── clock_getres ─────────────────────────────────────────────────────

    #[test]
    fn clock_getres_monotonic_1ns() {
        let res = clock_getres(ClockId::Monotonic).unwrap();
        assert_eq!(res, Timespec::new(0, 1));
    }

    #[test]
    fn clock_getres_others_stub() {
        assert_eq!(clock_getres(ClockId::Realtime), Err(Errno::ENOSYS));
        assert_eq!(clock_getres(ClockId::ProcessCputime), Err(Errno::ENOSYS));
        assert_eq!(clock_getres(ClockId::ThreadCputime), Err(Errno::ENOSYS));
    }

    // ── nanosleep ────────────────────────────────────────────────────────

    #[test]
    fn nanosleep_rejects_negative_sec() {
        let dur = Timespec::new(-1, 0);
        assert_eq!(nanosleep(&dur), Err(Errno::EINVAL));
    }

    #[test]
    fn nanosleep_rejects_negative_nsec() {
        let dur = Timespec::new(0, -1);
        assert_eq!(nanosleep(&dur), Err(Errno::EINVAL));
    }

    #[test]
    fn nanosleep_rejects_invalid_nsec() {
        let dur = Timespec::new(0, NSEC_PER_SEC);
        assert_eq!(nanosleep(&dur), Err(Errno::EINVAL));
    }

    #[test]
    fn nanosleep_valid_returns_stub() {
        let dur = Timespec::new(0, 100);
        assert_eq!(nanosleep(&dur), Err(Errno::ENOSYS));
    }

    // ── getrandom ────────────────────────────────────────────────────────

    #[test]
    fn getrandom_zero_len() {
        let mut buf = [0u8; 32];
        assert_eq!(getrandom(&mut buf, 0, 0), Ok(0));
    }

    #[test]
    fn getrandom_len_exceeds_buf() {
        let mut buf = [0u8; 8];
        assert_eq!(getrandom(&mut buf, 16, 0), Err(Errno::EINVAL));
    }

    #[test]
    fn getrandom_valid_returns_stub() {
        let mut buf = [0u8; 32];
        assert_eq!(getrandom(&mut buf, 32, 0), Err(Errno::ENOSYS));
    }

    #[test]
    fn getrandom_with_flags() {
        let mut buf = [0u8; 16];
        // Flags should be accepted but operation is still a stub.
        assert_eq!(
            getrandom(&mut buf, 16, GRND_RANDOM | GRND_NONBLOCK),
            Err(Errno::ENOSYS)
        );
    }
}
