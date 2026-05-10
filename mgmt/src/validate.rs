// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Validation pipeline for [`crate::Config`] writes.
//!
//! Spec:
//! `openspec/changes/management-login-v1/specs/mgmt-config-model/spec.md`
//! Requirement "Apply lifecycle" — step (2) "validate per-field rules
//! and cross-field constraints".
//!
//! Validation runs **before** any staging or in-memory mutation:
//!
//! - [`validate_field`] — per-field rules (range, length, alphabet)
//!   on a single proposed value.
//! - [`validate_cross_field`] — invariants that touch multiple
//!   fields, evaluated against the candidate config that *would*
//!   exist if the proposed value were applied.
//! - [`validate_all`] — runs every cross-field invariant against an
//!   already-fully-populated [`crate::Config`]; used by the loader on
//!   first read.
//!
//! All errors are [`crate::surface::Error::Validation`] with a
//! `&'static str` reason. Surfaces print the reason verbatim — keep it
//! short, lower-case, no trailing punctuation.

use crate::config::Config;
use crate::surface::{ConfigPath, Error, PasswordClassSet, RoleSet, Value};

// ─── Bounds (named so tests can pin them down) ───────────────────────────────

/// Minimum value for any `idle.*_minutes` field.
pub const IDLE_MIN_MINUTES: u32 = 1;
/// Maximum value for any `idle.*_minutes` field — 24 h.
pub const IDLE_MAX_MINUTES: u32 = 1440;

/// Minimum allowed `password.min_length`. Enforces a basic floor
/// regardless of operator preference.
pub const PASSWORD_MIN_LENGTH_FLOOR: u32 = 8;
/// Maximum allowed `password.max_length`. Caps argon2 input cost.
pub const PASSWORD_MAX_LENGTH_CEILING: u32 = 256;

/// Minimum sample interval for any `metrics.*_interval_ms`. Below
/// this, the cooperative sweeper starves other Core 0 work.
pub const METRICS_MIN_INTERVAL_MS: u32 = 100;
/// Maximum sample interval — 60 s.
pub const METRICS_MAX_INTERVAL_MS: u32 = 60_000;

/// Minimum CPU adaptive threshold percent (0..=100).
pub const ADAPTIVE_PCT_MIN: u32 = 0;
/// Maximum CPU adaptive threshold percent.
pub const ADAPTIVE_PCT_MAX: u32 = 100;

/// Minimum filesystem block read timeout — 100 ms.
pub const FS_MIN_READ_TIMEOUT_MS: u32 = 100;
/// Maximum filesystem block read timeout — 60 s.
pub const FS_MAX_READ_TIMEOUT_MS: u32 = 60_000;

/// Maximum block-device retry count. Higher than this risks the
/// boot watchdog firing.
pub const FS_MAX_RETRY_COUNT: u32 = 16;

/// Floor on cache budgets (1 MiB). Smaller values are useless to the
/// allocator.
pub const FS_CACHE_MIN_BYTES: u64 = 1024 * 1024;

/// Floor on audit `rotate_size_bytes` (64 KiB). Smaller values cause
/// pathological rotation cycles.
pub const AUDIT_ROTATE_MIN_BYTES: u64 = 64 * 1024;

/// Maximum number of retained audit archives. The on-disk audit ring
/// is bounded; this caps the bookkeeping vector.
pub const AUDIT_MAX_KEEP_ARCHIVES: u32 = 256;

/// Minimum `audit.signed_checkpoint_interval`. Below this, the
/// signed-checkpoint commit cost dominates the audit pipeline.
pub const AUDIT_MIN_CHECKPOINT_INTERVAL: u32 = 16;
/// Maximum `audit.signed_checkpoint_interval`. Above this, a
/// long-running system risks never reaching a checkpoint, defeating
/// off-box tamper-detection.
pub const AUDIT_MAX_CHECKPOINT_INTERVAL: u32 = 1_048_576;

/// Minimum `mgmt.zenoh_per_identity_max` — at least one session.
pub const ZENOH_PER_IDENTITY_MIN: u32 = 1;
/// Maximum `mgmt.zenoh_total_max` — caps in-kernel session table.
pub const ZENOH_TOTAL_MAX_CEILING: u32 = 1024;

/// Minimum overlay upper-layer cap (1 MiB).
pub const OVERLAY_MIN_BYTES: u64 = 1024 * 1024;

// ─── Per-field validation ────────────────────────────────────────────────────

/// Validate a single field's proposed value against its per-field
/// rules. Does **not** consult the rest of [`Config`] — for that see
/// [`validate_cross_field`].
///
/// Returns [`Error::Validation`] with a short reason on failure,
/// [`Error::TypeMismatch`] if `value` is the wrong variant for the
/// field, [`Error::NotFound`] for unknown paths.
pub fn validate_field(path: &ConfigPath, value: &Value) -> Result<(), Error> {
    match path.as_str() {
        // idle.*
        "idle.root_minutes" | "idle.operator_minutes" | "idle.viewer_minutes" => {
            let n = value.as_u32()?;
            check_range_u32(
                n,
                IDLE_MIN_MINUTES,
                IDLE_MAX_MINUTES,
                "idle minutes out of range",
            )
        }

        // password.*
        "password.min_length" => {
            let n = value.as_u32()?;
            check_range_u32(
                n,
                PASSWORD_MIN_LENGTH_FLOOR,
                PASSWORD_MAX_LENGTH_CEILING,
                "password.min_length out of range",
            )
        }
        "password.max_length" => {
            let n = value.as_u32()?;
            check_range_u32(
                n,
                PASSWORD_MIN_LENGTH_FLOOR,
                PASSWORD_MAX_LENGTH_CEILING,
                "password.max_length out of range",
            )
        }
        "password.require_classes" => {
            let p = value.as_password_classes()?;
            // Reject empty class set: at least one class must be
            // required, otherwise the policy is meaningless.
            if p.0 == 0 {
                return Err(Error::Validation(
                    "password.require_classes must require at least one class",
                ));
            }
            // Reject unknown bits — `from_mask` already drops them,
            // but if a caller hand-constructs `PasswordClassSet(_)`
            // we want a defensive check.
            if p.0 & !PasswordClassSet::ALL != 0 {
                return Err(Error::Validation(
                    "password.require_classes contains unknown bits",
                ));
            }
            Ok(())
        }
        "password.dictionary_check" => {
            let _ = value.as_bool()?;
            Ok(())
        }

        // audit.*
        "audit.rotate_size_bytes" => {
            let n = value.as_u64()?;
            check_range_u64(
                n,
                AUDIT_ROTATE_MIN_BYTES,
                u64::MAX,
                "audit.rotate_size_bytes too small",
            )
        }
        "audit.rotate_age_hours" => {
            let n = value.as_u32()?;
            check_range_u32(n, 1, 24 * 365, "audit.rotate_age_hours out of range")
        }
        "audit.keep_archives" => {
            let n = value.as_u32()?;
            check_range_u32(
                n,
                1,
                AUDIT_MAX_KEEP_ARCHIVES,
                "audit.keep_archives out of range",
            )
        }
        "audit.max_total_disk_bytes" => {
            let n = value.as_u64()?;
            check_range_u64(
                n,
                AUDIT_ROTATE_MIN_BYTES,
                u64::MAX,
                "audit.max_total_disk_bytes too small",
            )
        }
        "audit.signed_checkpoints" => {
            let _ = value.as_bool()?;
            Ok(())
        }
        "audit.signed_checkpoint_interval" => {
            let n = value.as_u32()?;
            check_range_u32(
                n,
                AUDIT_MIN_CHECKPOINT_INTERVAL,
                AUDIT_MAX_CHECKPOINT_INTERVAL,
                "audit.signed_checkpoint_interval out of range",
            )
        }

        // metrics.*
        "metrics.cpu_interval_ms"
        | "metrics.memory_interval_ms"
        | "metrics.network_interval_ms"
        | "metrics.inference_interval_ms" => {
            let n = value.as_u32()?;
            check_range_u32(
                n,
                METRICS_MIN_INTERVAL_MS,
                METRICS_MAX_INTERVAL_MS,
                "metrics interval out of range",
            )
        }
        "metrics.adaptive_cpu_high_pct" | "metrics.adaptive_cpu_low_pct" => {
            let n = value.as_u32()?;
            check_range_u32(
                n,
                ADAPTIVE_PCT_MIN,
                ADAPTIVE_PCT_MAX,
                "metrics adaptive percent out of range",
            )
        }

        // fs.*
        "fs.block_read_timeout_ms" => {
            let n = value.as_u32()?;
            check_range_u32(
                n,
                FS_MIN_READ_TIMEOUT_MS,
                FS_MAX_READ_TIMEOUT_MS,
                "fs.block_read_timeout_ms out of range",
            )
        }
        "fs.block_retry_count" => {
            let n = value.as_u32()?;
            check_range_u32(
                n,
                0,
                FS_MAX_RETRY_COUNT,
                "fs.block_retry_count out of range",
            )
        }
        "fs.cache_models_bytes" | "fs.cache_data_bytes" => {
            let n = value.as_u64()?;
            check_range_u64(n, FS_CACHE_MIN_BYTES, u64::MAX, "fs cache budget too small")
        }
        "fs.boot_watchdog_seconds" => {
            // 0 disables the watchdog; otherwise must fit in a
            // reasonable bound.
            let n = value.as_u32()?;
            if n > 3600 {
                return Err(Error::Validation("fs.boot_watchdog_seconds out of range"));
            }
            Ok(())
        }

        // mgmt.*
        "mgmt.zenoh_per_identity_max" => {
            let n = value.as_u32()?;
            check_range_u32(
                n,
                ZENOH_PER_IDENTITY_MIN,
                ZENOH_TOTAL_MAX_CEILING,
                "mgmt.zenoh_per_identity_max out of range",
            )
        }
        "mgmt.zenoh_total_max" => {
            let n = value.as_u32()?;
            check_range_u32(
                n,
                ZENOH_PER_IDENTITY_MIN,
                ZENOH_TOTAL_MAX_CEILING,
                "mgmt.zenoh_total_max out of range",
            )
        }
        "mgmt.token_two_tier" => {
            let _ = value.as_bool()?;
            Ok(())
        }

        // overlay.*
        "overlay.upper_max_bytes" => {
            let n = value.as_u64()?;
            check_range_u64(
                n,
                OVERLAY_MIN_BYTES,
                u64::MAX,
                "overlay.upper_max_bytes too small",
            )
        }
        "overlay.require_signed" | "overlay.allow_operator_unhide" => {
            let _ = value.as_bool()?;
            Ok(())
        }

        // flash.*
        "flash.prefer_flash_for_models" | "flash.readback_verify" => {
            let _ = value.as_bool()?;
            Ok(())
        }

        // totp.* (Phase 9)
        "totp.enforced_for_roles" => {
            let r = value.as_roles()?;
            // Reject unknown bits — defensive, since `from_mask`
            // already drops them but a hand-constructed `RoleSet(_)`
            // can carry garbage.
            if r.0 & !RoleSet::ALL != 0 {
                return Err(Error::Validation(
                    "totp.enforced_for_roles contains unknown bits",
                ));
            }
            Ok(())
        }

        _ => Err(Error::NotFound),
    }
}

// ─── Cross-field validation ──────────────────────────────────────────────────

/// Validate that the proposed write does not violate any cross-field
/// invariant. Evaluated against a hypothetical [`Config`] in which
/// `path` already holds `value`.
pub fn validate_cross_field(
    current: &Config,
    path: &ConfigPath,
    value: &Value,
) -> Result<(), Error> {
    // Build a candidate config: copy current, overwrite the field
    // being validated. Cross-field validators run against the
    // candidate so dependent invariants see the proposed value.
    let mut candidate = current.clone();
    apply_value_in_memory(&mut candidate, path, value)?;
    validate_all(&candidate)
}

/// Validate every cross-field invariant against an already-populated
/// [`Config`]. Used by the loader on first read and by
/// [`validate_cross_field`] internally.
pub fn validate_all(c: &Config) -> Result<(), Error> {
    // Password invariants.
    if c.password.min_length > c.password.max_length {
        return Err(Error::Validation(
            "password.min_length must be <= password.max_length",
        ));
    }
    let class_count = c.password.require_classes.count();
    if class_count > c.password.min_length {
        return Err(Error::Validation(
            "password.require_classes count must be <= password.min_length",
        ));
    }

    // Idle invariants — Root SHALL NOT be more permissive than
    // Operator, Operator SHALL NOT be more permissive than Viewer.
    // (More permissive = larger timeout.)
    if c.idle.root_minutes > c.idle.operator_minutes {
        return Err(Error::Validation(
            "idle.root_minutes must be <= idle.operator_minutes",
        ));
    }
    if c.idle.operator_minutes > c.idle.viewer_minutes {
        return Err(Error::Validation(
            "idle.operator_minutes must be <= idle.viewer_minutes",
        ));
    }

    // Metrics adaptive thresholds — `low_pct` SHALL be < `high_pct`.
    if c.metrics.adaptive_cpu_low_pct >= c.metrics.adaptive_cpu_high_pct {
        return Err(Error::Validation(
            "metrics.adaptive_cpu_low_pct must be < metrics.adaptive_cpu_high_pct",
        ));
    }

    // Audit invariants.
    let max_audit_use = c
        .audit
        .rotate_size_bytes
        .saturating_mul((c.audit.keep_archives as u64) + 1);
    if max_audit_use > c.audit.max_total_disk_bytes {
        return Err(Error::Validation(
            "audit.rotate_size_bytes * (keep_archives + 1) must be <= max_total_disk_bytes",
        ));
    }

    // Mgmt invariants.
    if c.mgmt.zenoh_per_identity_max > c.mgmt.zenoh_total_max {
        return Err(Error::Validation(
            "mgmt.zenoh_per_identity_max must be <= mgmt.zenoh_total_max",
        ));
    }

    Ok(())
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Apply a [`Value`] to [`Config`] **without** validation. Used by
/// [`validate_cross_field`] to build a candidate config; never call
/// directly from outside the validator.
fn apply_value_in_memory(c: &mut Config, path: &ConfigPath, value: &Value) -> Result<(), Error> {
    match path.as_str() {
        "idle.root_minutes" => c.idle.root_minutes = value.as_u32()?,
        "idle.operator_minutes" => c.idle.operator_minutes = value.as_u32()?,
        "idle.viewer_minutes" => c.idle.viewer_minutes = value.as_u32()?,

        "password.min_length" => c.password.min_length = value.as_u32()?,
        "password.max_length" => c.password.max_length = value.as_u32()?,
        "password.require_classes" => c.password.require_classes = value.as_password_classes()?,
        "password.dictionary_check" => c.password.dictionary_check = value.as_bool()?,

        "audit.rotate_size_bytes" => c.audit.rotate_size_bytes = value.as_u64()?,
        "audit.rotate_age_hours" => c.audit.rotate_age_hours = value.as_u32()?,
        "audit.keep_archives" => c.audit.keep_archives = value.as_u32()?,
        "audit.max_total_disk_bytes" => c.audit.max_total_disk_bytes = value.as_u64()?,
        "audit.signed_checkpoints" => c.audit.signed_checkpoints = value.as_bool()?,
        "audit.signed_checkpoint_interval" => {
            c.audit.signed_checkpoint_interval = value.as_u32()?
        }

        "metrics.cpu_interval_ms" => c.metrics.cpu_interval_ms = value.as_u32()?,
        "metrics.memory_interval_ms" => c.metrics.memory_interval_ms = value.as_u32()?,
        "metrics.network_interval_ms" => c.metrics.network_interval_ms = value.as_u32()?,
        "metrics.inference_interval_ms" => c.metrics.inference_interval_ms = value.as_u32()?,
        "metrics.adaptive_cpu_high_pct" => c.metrics.adaptive_cpu_high_pct = value.as_u32()?,
        "metrics.adaptive_cpu_low_pct" => c.metrics.adaptive_cpu_low_pct = value.as_u32()?,

        "fs.block_read_timeout_ms" => c.fs.block_read_timeout_ms = value.as_u32()?,
        "fs.block_retry_count" => c.fs.block_retry_count = value.as_u32()?,
        "fs.cache_models_bytes" => c.fs.cache_models_bytes = value.as_u64()?,
        "fs.cache_data_bytes" => c.fs.cache_data_bytes = value.as_u64()?,
        "fs.boot_watchdog_seconds" => c.fs.boot_watchdog_seconds = value.as_u32()?,

        "mgmt.zenoh_per_identity_max" => c.mgmt.zenoh_per_identity_max = value.as_u32()?,
        "mgmt.zenoh_total_max" => c.mgmt.zenoh_total_max = value.as_u32()?,
        "mgmt.token_two_tier" => c.mgmt.token_two_tier = value.as_bool()?,

        "overlay.upper_max_bytes" => c.overlay.upper_max_bytes = value.as_u64()?,
        "overlay.require_signed" => c.overlay.require_signed = value.as_bool()?,
        "overlay.allow_operator_unhide" => c.overlay.allow_operator_unhide = value.as_bool()?,

        "flash.prefer_flash_for_models" => c.flash.prefer_flash_for_models = value.as_bool()?,
        "flash.readback_verify" => c.flash.readback_verify = value.as_bool()?,

        "totp.enforced_for_roles" => c.totp.enforced_for_roles = value.as_roles()?,

        _ => return Err(Error::NotFound),
    }
    Ok(())
}

/// Verify `n` lies in `[min, max]` inclusive; otherwise yield
/// `Error::Validation(reason)`.
fn check_range_u32(n: u32, min: u32, max: u32, reason: &'static str) -> Result<(), Error> {
    if n < min || n > max {
        return Err(Error::Validation(reason));
    }
    Ok(())
}

/// Verify `n` lies in `[min, max]` inclusive; otherwise yield
/// `Error::Validation(reason)`.
fn check_range_u64(n: u64, min: u64, max: u64, reason: &'static str) -> Result<(), Error> {
    if n < min || n > max {
        return Err(Error::Validation(reason));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;
    use crate::surface::{ConfigPath, PasswordClassSet, Value};

    fn p(s: &str) -> ConfigPath {
        ConfigPath::new(s).unwrap()
    }

    // ─── per-field tests ─────────────────────────────────────────────────────

    #[test]
    fn idle_minutes_accept_boundary_values() {
        validate_field(&p("idle.root_minutes"), &Value::U32(1)).unwrap();
        validate_field(&p("idle.root_minutes"), &Value::U32(1440)).unwrap();
    }

    #[test]
    fn idle_minutes_reject_zero() {
        let err = validate_field(&p("idle.root_minutes"), &Value::U32(0)).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn idle_minutes_reject_above_max() {
        let err = validate_field(&p("idle.root_minutes"), &Value::U32(1441)).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn idle_minutes_reject_type_mismatch() {
        let err = validate_field(&p("idle.root_minutes"), &Value::Bool(true)).unwrap_err();
        assert!(matches!(err, Error::TypeMismatch { .. }));
    }

    #[test]
    fn password_min_length_floor() {
        validate_field(&p("password.min_length"), &Value::U32(8)).unwrap();
        let err = validate_field(&p("password.min_length"), &Value::U32(7)).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn password_max_length_ceiling() {
        validate_field(&p("password.max_length"), &Value::U32(256)).unwrap();
        let err = validate_field(&p("password.max_length"), &Value::U32(257)).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn password_classes_reject_empty_set() {
        let empty = PasswordClassSet::from_mask(0);
        let err = validate_field(
            &p("password.require_classes"),
            &Value::PasswordClasses(empty),
        )
        .unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn password_classes_accept_single_class() {
        let one = PasswordClassSet::from_mask(PasswordClassSet::LOWER);
        validate_field(&p("password.require_classes"), &Value::PasswordClasses(one)).unwrap();
    }

    #[test]
    fn password_classes_accept_all_classes() {
        let all = PasswordClassSet::from_mask(PasswordClassSet::ALL);
        validate_field(&p("password.require_classes"), &Value::PasswordClasses(all)).unwrap();
    }

    #[test]
    fn password_classes_reject_unknown_bits() {
        // Hand-constructed mask with bit 0x80 set — `from_mask`
        // wouldn't produce this but a hand-crafted struct can.
        let raw = PasswordClassSet(0x80);
        let err = validate_field(&p("password.require_classes"), &Value::PasswordClasses(raw))
            .unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn audit_rotate_size_floor() {
        validate_field(&p("audit.rotate_size_bytes"), &Value::U64(64 * 1024)).unwrap();
        let err =
            validate_field(&p("audit.rotate_size_bytes"), &Value::U64(63 * 1024)).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn audit_rotate_age_hours_range() {
        validate_field(&p("audit.rotate_age_hours"), &Value::U32(1)).unwrap();
        validate_field(&p("audit.rotate_age_hours"), &Value::U32(24 * 365)).unwrap();
        let err = validate_field(&p("audit.rotate_age_hours"), &Value::U32(0)).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn audit_keep_archives_range() {
        validate_field(&p("audit.keep_archives"), &Value::U32(1)).unwrap();
        validate_field(
            &p("audit.keep_archives"),
            &Value::U32(AUDIT_MAX_KEEP_ARCHIVES),
        )
        .unwrap();
        let err = validate_field(&p("audit.keep_archives"), &Value::U32(0)).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn metrics_interval_floor_and_ceiling() {
        for path_str in [
            "metrics.cpu_interval_ms",
            "metrics.memory_interval_ms",
            "metrics.network_interval_ms",
            "metrics.inference_interval_ms",
        ] {
            validate_field(&p(path_str), &Value::U32(METRICS_MIN_INTERVAL_MS)).unwrap();
            validate_field(&p(path_str), &Value::U32(METRICS_MAX_INTERVAL_MS)).unwrap();
            let err = validate_field(&p(path_str), &Value::U32(50)).unwrap_err();
            assert!(matches!(err, Error::Validation(_)));
            let err = validate_field(&p(path_str), &Value::U32(60_001)).unwrap_err();
            assert!(matches!(err, Error::Validation(_)));
        }
    }

    #[test]
    fn metrics_adaptive_pct_range() {
        validate_field(&p("metrics.adaptive_cpu_high_pct"), &Value::U32(0)).unwrap();
        validate_field(&p("metrics.adaptive_cpu_high_pct"), &Value::U32(100)).unwrap();
        let err =
            validate_field(&p("metrics.adaptive_cpu_high_pct"), &Value::U32(101)).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn fs_read_timeout_range() {
        validate_field(&p("fs.block_read_timeout_ms"), &Value::U32(100)).unwrap();
        validate_field(&p("fs.block_read_timeout_ms"), &Value::U32(60_000)).unwrap();
        let err = validate_field(&p("fs.block_read_timeout_ms"), &Value::U32(99)).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn fs_retry_count_zero_allowed() {
        validate_field(&p("fs.block_retry_count"), &Value::U32(0)).unwrap();
        validate_field(&p("fs.block_retry_count"), &Value::U32(FS_MAX_RETRY_COUNT)).unwrap();
        let err = validate_field(
            &p("fs.block_retry_count"),
            &Value::U32(FS_MAX_RETRY_COUNT + 1),
        )
        .unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn fs_cache_size_floor() {
        validate_field(&p("fs.cache_models_bytes"), &Value::U64(FS_CACHE_MIN_BYTES)).unwrap();
        let err = validate_field(&p("fs.cache_models_bytes"), &Value::U64(0)).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn fs_boot_watchdog_zero_disables() {
        // Spec: 0 is a sentinel that disables the watchdog.
        validate_field(&p("fs.boot_watchdog_seconds"), &Value::U32(0)).unwrap();
        validate_field(&p("fs.boot_watchdog_seconds"), &Value::U32(60)).unwrap();
        let err = validate_field(&p("fs.boot_watchdog_seconds"), &Value::U32(3601)).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn mgmt_zenoh_caps() {
        validate_field(&p("mgmt.zenoh_per_identity_max"), &Value::U32(1)).unwrap();
        validate_field(
            &p("mgmt.zenoh_total_max"),
            &Value::U32(ZENOH_TOTAL_MAX_CEILING),
        )
        .unwrap();
        let err = validate_field(&p("mgmt.zenoh_per_identity_max"), &Value::U32(0)).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn overlay_upper_max_floor() {
        validate_field(
            &p("overlay.upper_max_bytes"),
            &Value::U64(OVERLAY_MIN_BYTES),
        )
        .unwrap();
        let err = validate_field(&p("overlay.upper_max_bytes"), &Value::U64(0)).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn unknown_path_returns_not_found() {
        let err = validate_field(&p("nope.nada"), &Value::U32(0)).unwrap_err();
        assert!(matches!(err, Error::NotFound));
    }

    // ─── cross-field tests ───────────────────────────────────────────────────

    #[test]
    fn defaults_pass_validate_all() {
        validate_all(&Config::default()).unwrap();
    }

    #[test]
    fn password_min_above_max_rejected() {
        let mut c = Config::default();
        c.password.min_length = 200;
        c.password.max_length = 100;
        let err = validate_all(&c).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn password_classes_count_above_min_length_rejected() {
        let mut c = Config::default();
        c.password.min_length = 2; // require 4 classes but only 2 chars.
        c.password.require_classes = PasswordClassSet::from_mask(PasswordClassSet::ALL);
        // Need to also re-allow min_length=2 path; but per-field
        // validator floor=8 catches that. validate_all only checks
        // cross-field, so directly:
        let err = validate_all(&c).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn idle_root_above_operator_rejected() {
        let mut c = Config::default();
        c.idle.root_minutes = 60;
        c.idle.operator_minutes = 30;
        let err = validate_all(&c).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn idle_operator_above_viewer_rejected() {
        let mut c = Config::default();
        c.idle.operator_minutes = 120;
        c.idle.viewer_minutes = 60;
        let err = validate_all(&c).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn metrics_low_above_high_rejected() {
        let mut c = Config::default();
        c.metrics.adaptive_cpu_low_pct = 80;
        c.metrics.adaptive_cpu_high_pct = 80;
        let err = validate_all(&c).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn audit_disk_budget_too_small_rejected() {
        let mut c = Config::default();
        // 16 MiB rotate * (8 + 1) = 144 MiB; cap to 64 MiB → fail.
        c.audit.max_total_disk_bytes = 64 * 1024 * 1024;
        let err = validate_all(&c).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn mgmt_per_identity_above_total_rejected() {
        let mut c = Config::default();
        c.mgmt.zenoh_per_identity_max = 100;
        c.mgmt.zenoh_total_max = 50;
        let err = validate_all(&c).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    // ─── validate_cross_field driver ─────────────────────────────────────────

    #[test]
    fn cross_field_uses_candidate_not_current() {
        // Current config has root=15, operator=60 → fine.
        // Try to write root=90 → would violate root <= operator=60.
        let c = Config::default();
        let err = validate_cross_field(&c, &p("idle.root_minutes"), &Value::U32(90)).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn cross_field_pass_when_invariants_hold() {
        let c = Config::default();
        // root 15 → 30 still <= operator 60.
        validate_cross_field(&c, &p("idle.root_minutes"), &Value::U32(30)).unwrap();
    }

    #[test]
    fn cross_field_rejects_unknown_path() {
        let c = Config::default();
        let err = validate_cross_field(&c, &p("nope.nada"), &Value::U32(0)).unwrap_err();
        assert!(matches!(err, Error::NotFound));
    }

    // ─── Phase 9: TOTP enforcement field ────────────────────────────────

    #[test]
    fn totp_enforced_for_roles_accepts_empty() {
        let empty = RoleSet::from_mask(0);
        validate_field(&p("totp.enforced_for_roles"), &Value::Roles(empty)).unwrap();
    }

    #[test]
    fn totp_enforced_for_roles_accepts_all_known_roles() {
        let all = RoleSet::from_mask(RoleSet::ALL);
        validate_field(&p("totp.enforced_for_roles"), &Value::Roles(all)).unwrap();
    }

    #[test]
    fn totp_enforced_for_roles_rejects_unknown_bits() {
        let bogus = RoleSet(0x80);
        let err = validate_field(&p("totp.enforced_for_roles"), &Value::Roles(bogus)).unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn totp_enforced_for_roles_rejects_type_mismatch() {
        let err = validate_field(&p("totp.enforced_for_roles"), &Value::Bool(true)).unwrap_err();
        assert!(matches!(err, Error::TypeMismatch { .. }));
    }
}
