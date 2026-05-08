// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Per-op timeout + bounded-retry policy for block I/O.
//!
//! Per the spec (`embedded-filesystem-v1::fs-block-device`):
//!
//! - **Read** default per-op timeout is **250 ms**, retry schedule is
//!   **500 ms / 1 s / 2 s** (3 retries), then `Err(Timeout)`.
//! - **Write** default per-op timeout is **1 s**, same retry schedule
//!   (3 retries), then `Err(Timeout)`.
//! - Configurable via `mgmt/policy.toml` keys
//!   `fs.block.read_timeout_ms`, `fs.block.write_timeout_ms`,
//!   `fs.block.retry_count`, `fs.block.retry_backoff_ms`.
//! - **Fail-after-retries is non-configurable.** An unattended
//!   appliance must NEVER block indefinitely on a single bad sector;
//!   "infinite retry" sentinels (`u32::MAX`, `0xFFFF_FFFF`) are
//!   rejected by [`RetryPolicy::validate`] with [`PolicyError::Indefinite`]
//!   and surfaced to userspace as `-EINVAL`.
//!
//! Wave 0's `mgmt/` is empty — the `from_config()` helper here is a
//! stub that returns `default()` for now; `management-login-v1`
//! Phase 5 wires it to the real `Config` struct.

use super::BlockError;

/// Default read per-op timeout, in milliseconds.
pub const DEFAULT_READ_TIMEOUT_MS: u32 = 250;

/// Default write per-op timeout, in milliseconds.
pub const DEFAULT_WRITE_TIMEOUT_MS: u32 = 1000;

/// Default retry count (3 attempts beyond the initial one).
pub const DEFAULT_RETRY_COUNT: u32 = 3;

/// Default exponential-backoff schedule, in milliseconds.
///
/// `[500, 1000, 2000]` = first retry waits 500 ms, second 1 s, third
/// 2 s. The schedule has at most [`DEFAULT_RETRY_COUNT`] entries; if
/// `retry_count` is configured higher than the schedule length, the
/// last value is reused for the additional retries.
pub const DEFAULT_RETRY_BACKOFF_MS: [u32; 3] = [500, 1000, 2000];

/// Sentinel value rejected by [`RetryPolicy::validate`].
///
/// The spec is explicit: an unattended appliance must never block
/// indefinitely on a single bad sector. Configuration writes
/// containing this value (or any retry count above
/// [`MAX_ALLOWED_RETRY_COUNT`]) are rejected with `-EINVAL`.
pub const RETRY_COUNT_INFINITE_SENTINEL: u32 = u32::MAX;

/// Hard upper bound on `retry_count`. Higher values are rejected at
/// configuration time as effectively-infinite (an unattended appliance
/// must never wedge).
///
/// Sized so that with the maximum allowed backoff, a single failed I/O
/// can stall for at most a few minutes — well short of the watchdog.
pub const MAX_ALLOWED_RETRY_COUNT: u32 = 16;

/// Hard upper bound on `retry_backoff_ms` entries.
///
/// 30 s per backoff step + 16 retries = ~8 min worst case. Beyond this
/// the operator should investigate hardware rather than configure
/// longer backoff.
pub const MAX_ALLOWED_BACKOFF_MS: u32 = 30_000;

/// Hard upper bound on per-op timeout. 30 s per attempt is already
/// extreme for any reasonable transport.
pub const MAX_ALLOWED_TIMEOUT_MS: u32 = 30_000;

/// Errors returned by [`RetryPolicy::validate`].
///
/// Each variant maps to `-EINVAL` at the syscall boundary
/// (configuration write).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyError {
    /// `retry_count` is `u32::MAX` or otherwise an "infinite retry"
    /// sentinel. Rejected to keep an unattended appliance from
    /// wedging.
    Indefinite,
    /// `retry_count` exceeds [`MAX_ALLOWED_RETRY_COUNT`].
    RetryCountTooHigh,
    /// A backoff entry exceeds [`MAX_ALLOWED_BACKOFF_MS`], or the
    /// list is empty when `retry_count > 0`.
    BackoffOutOfRange,
    /// Per-op timeout exceeds [`MAX_ALLOWED_TIMEOUT_MS`] or is zero.
    TimeoutOutOfRange,
}

/// Per-op timeout + retry schedule for a block-I/O operation.
///
/// Two well-known constructors give the spec defaults:
/// [`RetryPolicy::default_read`] (250 ms timeout) and
/// [`RetryPolicy::default_write`] (1 s timeout). The retry schedule
/// is the same for both (`500 / 1000 / 2000` ms).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Per-attempt timeout, in milliseconds. Zero is invalid.
    pub timeout_ms: u32,
    /// Number of retries beyond the initial attempt. The total number
    /// of attempts is `retry_count + 1`. Capped by
    /// [`MAX_ALLOWED_RETRY_COUNT`].
    pub retry_count: u32,
    /// Exponential-backoff schedule, in milliseconds. If
    /// `retry_count > backoff_ms.len()`, the last value is reused for
    /// the additional retries.
    pub backoff_ms: alloc::vec::Vec<u32>,
}

impl RetryPolicy {
    /// Spec defaults for **read** operations: 250 ms per-op timeout,
    /// 3 retries with `[500, 1000, 2000]` ms backoff.
    pub fn default_read() -> Self {
        Self {
            timeout_ms: DEFAULT_READ_TIMEOUT_MS,
            retry_count: DEFAULT_RETRY_COUNT,
            backoff_ms: DEFAULT_RETRY_BACKOFF_MS.to_vec(),
        }
    }

    /// Spec defaults for **write** operations: 1 s per-op timeout,
    /// 3 retries with `[500, 1000, 2000]` ms backoff.
    pub fn default_write() -> Self {
        Self {
            timeout_ms: DEFAULT_WRITE_TIMEOUT_MS,
            retry_count: DEFAULT_RETRY_COUNT,
            backoff_ms: DEFAULT_RETRY_BACKOFF_MS.to_vec(),
        }
    }

    /// Stub for the `mgmt::Config` integration. Wave 0's `mgmt` crate
    /// is empty; once `management-login-v1` Phase 5 lands, this
    /// reads the four `fs.block.*` keys and applies them on top of
    /// the spec defaults.
    ///
    /// For now: returns the spec defaults for the requested operation
    /// kind.
    pub fn from_config(kind: PolicyKind) -> Self {
        match kind {
            PolicyKind::Read => Self::default_read(),
            PolicyKind::Write => Self::default_write(),
        }
    }

    /// Validate this policy against the non-configurable safety
    /// invariants. Called by `mgmt`'s config validator on every
    /// `policy.toml` write.
    pub fn validate(&self) -> Result<(), PolicyError> {
        if self.retry_count == RETRY_COUNT_INFINITE_SENTINEL {
            return Err(PolicyError::Indefinite);
        }
        if self.retry_count > MAX_ALLOWED_RETRY_COUNT {
            return Err(PolicyError::RetryCountTooHigh);
        }
        if self.timeout_ms == 0 || self.timeout_ms > MAX_ALLOWED_TIMEOUT_MS {
            return Err(PolicyError::TimeoutOutOfRange);
        }
        if self.retry_count > 0 && self.backoff_ms.is_empty() {
            return Err(PolicyError::BackoffOutOfRange);
        }
        for &b in &self.backoff_ms {
            if b > MAX_ALLOWED_BACKOFF_MS {
                return Err(PolicyError::BackoffOutOfRange);
            }
        }
        Ok(())
    }

    /// Backoff to wait before retry attempt `n` (1-indexed).
    ///
    /// `n=1` is the first retry; for `n >= backoff_ms.len()` returns
    /// the last entry. Returns `0` if `backoff_ms` is empty.
    pub fn backoff_for_retry(&self, n: u32) -> u32 {
        if self.backoff_ms.is_empty() {
            return 0;
        }
        if n == 0 {
            return 0;
        }
        let idx = ((n - 1) as usize).min(self.backoff_ms.len() - 1);
        self.backoff_ms[idx]
    }
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self::default_read()
    }
}

/// Whether to use the read or write defaults in
/// [`RetryPolicy::from_config`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyKind {
    Read,
    Write,
}

/// Outcome of running a single attempt inside [`run_with_retries`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttemptOutcome {
    /// Success — return immediately.
    Success,
    /// Failure that should be retried (transient transport error,
    /// timeout, busy).
    Retry,
    /// Failure that should NOT be retried (alignment, bounds,
    /// permanent media error).
    Permanent(BlockError),
}

/// Run a block-I/O attempt with bounded retries.
///
/// `attempt` is invoked up to `1 + policy.retry_count` times. Between
/// attempts it sleeps for `policy.backoff_for_retry(n)` ms via the
/// caller-provided `sleep_ms` callback (no global timer dependency).
///
/// On exhaustion the helper returns `Err(BlockError::Timeout)`,
/// regardless of which transient error variant the underlying device
/// reported, because the spec's contract with userspace is
/// "fail-after-retries === ETIMEDOUT".
pub fn run_with_retries<F, S>(
    policy: &RetryPolicy,
    mut attempt: F,
    mut sleep_ms: S,
) -> Result<(), BlockError>
where
    F: FnMut(u32) -> AttemptOutcome,
    S: FnMut(u32),
{
    let total_attempts = policy.retry_count.saturating_add(1);
    for n in 0..total_attempts {
        match attempt(n) {
            AttemptOutcome::Success => return Ok(()),
            AttemptOutcome::Permanent(err) => return Err(err),
            AttemptOutcome::Retry => {
                // After the last attempt, fall through to Timeout.
                if n + 1 < total_attempts {
                    let backoff = policy.backoff_for_retry(n + 1);
                    if backoff > 0 {
                        sleep_ms(backoff);
                    }
                }
            }
        }
    }
    Err(BlockError::Timeout)
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::Cell;

    #[test]
    fn read_defaults_match_spec() {
        let p = RetryPolicy::default_read();
        assert_eq!(p.timeout_ms, 250);
        assert_eq!(p.retry_count, 3);
        assert_eq!(p.backoff_ms, [500, 1000, 2000]);
    }

    #[test]
    fn write_defaults_match_spec() {
        let p = RetryPolicy::default_write();
        assert_eq!(p.timeout_ms, 1000);
        assert_eq!(p.retry_count, 3);
        assert_eq!(p.backoff_ms, [500, 1000, 2000]);
    }

    #[test]
    fn from_config_returns_defaults_for_now() {
        assert_eq!(
            RetryPolicy::from_config(PolicyKind::Read),
            RetryPolicy::default_read()
        );
        assert_eq!(
            RetryPolicy::from_config(PolicyKind::Write),
            RetryPolicy::default_write()
        );
    }

    #[test]
    fn validate_accepts_defaults() {
        assert!(RetryPolicy::default_read().validate().is_ok());
        assert!(RetryPolicy::default_write().validate().is_ok());
    }

    #[test]
    fn validate_rejects_infinite_sentinel() {
        let p = RetryPolicy {
            timeout_ms: 250,
            retry_count: RETRY_COUNT_INFINITE_SENTINEL,
            backoff_ms: alloc::vec![500],
        };
        assert_eq!(p.validate(), Err(PolicyError::Indefinite));
    }

    #[test]
    fn validate_rejects_too_many_retries() {
        let p = RetryPolicy {
            timeout_ms: 250,
            retry_count: MAX_ALLOWED_RETRY_COUNT + 1,
            backoff_ms: alloc::vec![500],
        };
        assert_eq!(p.validate(), Err(PolicyError::RetryCountTooHigh));
    }

    #[test]
    fn validate_rejects_zero_timeout() {
        let p = RetryPolicy {
            timeout_ms: 0,
            retry_count: 0,
            backoff_ms: alloc::vec![],
        };
        assert_eq!(p.validate(), Err(PolicyError::TimeoutOutOfRange));
    }

    #[test]
    fn validate_rejects_huge_timeout() {
        let p = RetryPolicy {
            timeout_ms: MAX_ALLOWED_TIMEOUT_MS + 1,
            retry_count: 0,
            backoff_ms: alloc::vec![],
        };
        assert_eq!(p.validate(), Err(PolicyError::TimeoutOutOfRange));
    }

    #[test]
    fn validate_rejects_huge_backoff() {
        let p = RetryPolicy {
            timeout_ms: 250,
            retry_count: 1,
            backoff_ms: alloc::vec![MAX_ALLOWED_BACKOFF_MS + 1],
        };
        assert_eq!(p.validate(), Err(PolicyError::BackoffOutOfRange));
    }

    #[test]
    fn validate_rejects_empty_backoff_with_retries() {
        let p = RetryPolicy {
            timeout_ms: 250,
            retry_count: 3,
            backoff_ms: alloc::vec![],
        };
        assert_eq!(p.validate(), Err(PolicyError::BackoffOutOfRange));
    }

    #[test]
    fn backoff_schedule_clamps_to_last_entry() {
        let p = RetryPolicy::default_read();
        assert_eq!(p.backoff_for_retry(1), 500);
        assert_eq!(p.backoff_for_retry(2), 1000);
        assert_eq!(p.backoff_for_retry(3), 2000);
        // beyond the schedule reuses the last
        assert_eq!(p.backoff_for_retry(4), 2000);
        assert_eq!(p.backoff_for_retry(99), 2000);
    }

    #[test]
    fn run_with_retries_succeeds_on_first_try() {
        let p = RetryPolicy::default_read();
        let calls = Cell::new(0u32);
        let res = run_with_retries(
            &p,
            |_| {
                calls.set(calls.get() + 1);
                AttemptOutcome::Success
            },
            |_| panic!("must not sleep"),
        );
        assert!(res.is_ok());
        assert_eq!(calls.get(), 1);
    }

    #[test]
    fn run_with_retries_three_retries_then_timeout() {
        let p = RetryPolicy::default_read();
        let calls = Cell::new(0u32);
        let sleeps: core::cell::RefCell<alloc::vec::Vec<u32>> =
            core::cell::RefCell::new(alloc::vec::Vec::new());
        let res = run_with_retries(
            &p,
            |_| {
                calls.set(calls.get() + 1);
                AttemptOutcome::Retry
            },
            |ms| sleeps.borrow_mut().push(ms),
        );
        // 1 initial + 3 retries = 4 calls
        assert_eq!(calls.get(), 4);
        // Sleeps happen between attempts: 3 of them
        assert_eq!(*sleeps.borrow(), alloc::vec![500u32, 1000, 2000]);
        assert_eq!(res, Err(BlockError::Timeout));
    }

    #[test]
    fn run_with_retries_permanent_error_returned_immediately() {
        let p = RetryPolicy::default_read();
        let calls = Cell::new(0u32);
        let res = run_with_retries(
            &p,
            |_| {
                calls.set(calls.get() + 1);
                AttemptOutcome::Permanent(BlockError::OutOfRange)
            },
            |_| panic!("must not sleep on permanent error"),
        );
        assert_eq!(calls.get(), 1);
        assert_eq!(res, Err(BlockError::OutOfRange));
    }

    #[test]
    fn run_with_retries_success_on_third_attempt() {
        let p = RetryPolicy::default_read();
        let calls = Cell::new(0u32);
        let res = run_with_retries(
            &p,
            |_| {
                let n = calls.get() + 1;
                calls.set(n);
                if n < 3 {
                    AttemptOutcome::Retry
                } else {
                    AttemptOutcome::Success
                }
            },
            |_| {},
        );
        assert!(res.is_ok());
        assert_eq!(calls.get(), 3);
    }

    #[test]
    fn run_with_retries_zero_retries_one_attempt() {
        let p = RetryPolicy {
            timeout_ms: 100,
            retry_count: 0,
            backoff_ms: alloc::vec![],
        };
        let calls = Cell::new(0u32);
        let res = run_with_retries(
            &p,
            |_| {
                calls.set(calls.get() + 1);
                AttemptOutcome::Retry
            },
            |_| panic!("must not sleep when retry_count == 0"),
        );
        assert_eq!(calls.get(), 1);
        assert_eq!(res, Err(BlockError::Timeout));
    }
}
