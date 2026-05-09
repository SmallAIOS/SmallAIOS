// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Default values for the `fs.*` config-field surface.
//!
//! These constants are the single source of truth for the
//! `fs.cache.*`, `fs.block.*`, and `fs.boot.*` field defaults declared
//! by the `mgmt-config-layout` spec. The `mgmt` crate's Phase 5 config
//! loader pulls these values directly so the runtime defaults and the
//! spec stay in lock-step. Operators can override them via
//! `policy.toml`; the validator in `mgmt` rejects values that fall
//! below the floors documented in the `embedded-filesystem-v1` change.
//!
//! Owned by `embedded-filesystem-v1` Phase 6.

// ─── Block-cache budgets ─────────────────────────────────────────────────────

/// Default LRU block-cache budget for the `/models/` (squashfs) mount,
/// in bytes. 16 MiB matches the Q10 answer in the
/// `fs-block-device` design clarifications.
pub const FS_CACHE_MODELS_BYTES: u64 = 16 * 1024 * 1024;

/// Default LRU block-cache budget for the `/data/` (F2FS) mount, in
/// bytes. 4 MiB is sized for short-lived metadata working sets.
pub const FS_CACHE_DATA_BYTES: u64 = 4 * 1024 * 1024;

/// Floor for `fs.cache.models_bytes`. Configurations below this floor
/// are rejected by the `mgmt` validator. Keeps the cache large enough
/// to hold the squashfs root + a working set of metadata blocks.
pub const FS_CACHE_MODELS_FLOOR_BYTES: u64 = 4 * 1024 * 1024;

/// Floor for `fs.cache.data_bytes`. One MiB holds the `/data/auth/`
/// shadow file plus a small audit-log working set.
pub const FS_CACHE_DATA_FLOOR_BYTES: u64 = 1024 * 1024;

/// Default size of the per-mount write-coalescing buffer used by the
/// `/data/` audit-log writer to batch short metadata appends, in bytes.
pub const FS_WRITE_COALESCE_BYTES: u64 = 256 * 1024;

// ─── Block-device timeouts and retry policy ──────────────────────────────────

/// Default per-read timeout (milliseconds).
pub const FS_BLOCK_READ_TIMEOUT_MS: u64 = 250;

/// Default per-write timeout (milliseconds). Larger than reads because
/// flush-completion paths on eMMC can stall briefly.
pub const FS_BLOCK_WRITE_TIMEOUT_MS: u64 = 1000;

/// Default number of retries for a transient block-device error before
/// giving up and surfacing the failure.
pub const FS_BLOCK_RETRY_COUNT: u32 = 3;

/// Default backoff (milliseconds) inserted between retries.
pub const FS_BLOCK_RETRY_BACKOFF_MS: u64 = 500;

// ─── Boot watchdog ───────────────────────────────────────────────────────────

/// Default boot watchdog window, in seconds. If `boot_success` is not
/// asserted within this window the bootloader rolls back to the
/// previously-known-good slot.
pub const FS_BOOT_WATCHDOG_SECONDS: u32 = 60;

#[cfg(test)]
mod tests {
    // These tests are intentional compile-time documentation assertions
    // on `const` values — they read like runtime tests but verify
    // constant invariants. clippy's `assertions_on_constants` and
    // `bool_assert_comparison` lints object; opted out for the module.
    #![allow(clippy::assertions_on_constants)]
    #![allow(clippy::bool_assert_comparison)]

    use super::*;

    #[test]
    fn cache_defaults_above_floor() {
        assert!(FS_CACHE_MODELS_BYTES >= FS_CACHE_MODELS_FLOOR_BYTES);
        assert!(FS_CACHE_DATA_BYTES >= FS_CACHE_DATA_FLOOR_BYTES);
    }

    #[test]
    fn write_coalesce_buffer_at_least_one_block() {
        // Must hold at least one 4 KiB block; 256 KiB easily clears
        // that bar but the assertion documents intent.
        assert!(FS_WRITE_COALESCE_BYTES >= 4096);
    }

    #[test]
    fn timeouts_are_non_zero() {
        assert!(FS_BLOCK_READ_TIMEOUT_MS > 0);
        assert!(FS_BLOCK_WRITE_TIMEOUT_MS > 0);
    }

    #[test]
    fn retry_policy_is_finite() {
        assert!(FS_BLOCK_RETRY_COUNT > 0);
        assert!(FS_BLOCK_RETRY_BACKOFF_MS > 0);
    }

    #[test]
    fn boot_watchdog_window_is_positive() {
        assert!(FS_BOOT_WATCHDOG_SECONDS > 0);
    }

    #[test]
    fn write_timeout_at_least_read_timeout() {
        // Documented invariant: write paths can stall longer than reads
        // (eMMC flush completion), so the default write timeout must be
        // at least the read timeout.
        assert!(FS_BLOCK_WRITE_TIMEOUT_MS >= FS_BLOCK_READ_TIMEOUT_MS);
    }
}
