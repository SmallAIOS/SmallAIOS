// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Typed [`Config`] source-of-truth for every operator-tunable
//! setting in SmallAIOS, plus the path registry that drives the
//! universal-exposure walker.
//!
//! Spec:
//! `openspec/changes/management-login-v1/specs/mgmt-config-model/spec.md`.
//!
//! ## Single source of truth
//!
//! Every reachable management surface — TOML loader, TTY console,
//! Zenoh admin, future UDS-on-CAN — is a thin (de)serialiser over the
//! same [`Config`] value. There SHALL NOT be any surface-specific
//! configuration knob held outside [`Config`].
//!
//! ## v1 schema
//!
//! The v1 schema in this module covers the fields named in the
//! management-login-v1 design plus the `embedded-filesystem-v1` and
//! `embedded-overlay-v1` config-field additions. Each field is a
//! `Copy`-friendly Rust type wrapped in a typed namespace struct so
//! `Config` itself is a tree of plain data with no internal locks.
//!
//! ## Path registry
//!
//! [`registry::CONFIG_FIELDS`] is the const slice walked by the
//! universal-exposure gate. Every entry pairs a path with its typed
//! [`crate::surface::Value`] tag, [`crate::surface::ReloadKind`], and
//! [`crate::surface::SurfaceScope`]. The registry — not the field
//! initialisers — is what the build-time test in
//! `every_config_field_has_a_path` walks.
//!
//! ## On `#[reload(...)]` proc-macro
//!
//! The spec calls for a `#[reload("live"|"boot")]` attribute on each
//! field. For v1 we ship the equivalent **const-based registry**: each
//! field's reload kind is encoded once in [`registry::CONFIG_FIELDS`]
//! rather than scattered across attributes. The registry-based design
//! is functionally equivalent (build-time gate, identical runtime
//! semantics) and keeps `mgmt` free of an extra `mgmt-macros/`
//! proc-macro crate. A future phase MAY add the proc-macro and
//! generate the registry from it; the public surface SHALL NOT change.

use crate::surface::{ConfigPath, PasswordClassSet, ReloadKind, RoleSet, SurfaceScope, Value};

// ─── Top-level Config ────────────────────────────────────────────────────────

/// Single typed `Config` source-of-truth for every operator-tunable
/// setting in SmallAIOS.
///
/// `Config` is intentionally a tree of plain `Copy` data — no `Vec`,
/// no `Rc`, no interior mutability. State that needs interior
/// mutability (subscriber rings, audit appenders) lives in
/// [`crate::ApplyEngine`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Config {
    /// Per-role idle-timeout policy.
    pub idle: IdleConfig,
    /// Password complexity / length policy.
    pub password: PasswordPolicy,
    /// Audit-log rotation + checkpoint policy.
    pub audit: AuditConfig,
    /// Metrics emission cadence + adaptive thresholds.
    pub metrics: MetricsConfig,
    /// Filesystem block, cache, and boot watchdog policy.
    pub fs: FsConfig,
    /// Management-surface (Zenoh + tokens) policy.
    pub mgmt: MgmtConfig,
    /// Overlay (`/data/overlay`) tunables.
    pub overlay: OverlayConfig,
    /// Flash (`/flash` vs `/data`) routing knobs.
    pub flash: FlashConfig,
    /// TOTP (RFC 6238) second-factor enforcement policy. Phase 9.
    pub totp: TotpConfig,
}

impl Default for Config {
    fn default() -> Self {
        Self::v1_defaults()
    }
}

impl Config {
    /// Conservative v1 defaults appropriate for a fresh first-boot.
    /// Concrete numbers come from the management-login-v1 design
    /// document; values inside per-namespace defaults below.
    pub const fn v1_defaults() -> Self {
        Self {
            idle: IdleConfig::v1_defaults(),
            password: PasswordPolicy::v1_defaults(),
            audit: AuditConfig::v1_defaults(),
            metrics: MetricsConfig::v1_defaults(),
            fs: FsConfig::v1_defaults(),
            mgmt: MgmtConfig::v1_defaults(),
            overlay: OverlayConfig::v1_defaults(),
            flash: FlashConfig::v1_defaults(),
            totp: TotpConfig::v1_defaults(),
        }
    }
}

// ─── Idle-timeout policy ─────────────────────────────────────────────────────

/// Per-role idle-timeout policy in minutes.
///
/// `Root` defaults to the most aggressive timeout (least time-on);
/// `Viewer` to the most permissive. The kernel idle sweeper enforces
/// these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdleConfig {
    /// Idle timeout for `Root` sessions, in minutes (≥ 1, ≤ 1440).
    pub root_minutes: u32,
    /// Idle timeout for `Operator` sessions, in minutes (≥ 1, ≤ 1440).
    pub operator_minutes: u32,
    /// Idle timeout for `Viewer` sessions, in minutes (≥ 1, ≤ 1440).
    pub viewer_minutes: u32,
}

impl IdleConfig {
    /// v1 defaults: 15 / 30 / 60 minutes.
    pub const fn v1_defaults() -> Self {
        Self {
            root_minutes: 15,
            operator_minutes: 60,
            viewer_minutes: 60,
        }
    }
}

// ─── Password policy ─────────────────────────────────────────────────────────

/// Password complexity policy. Used by the first-boot prompt and the
/// `auth_change_password` syscall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PasswordPolicy {
    /// Minimum password length, in characters (≥ 8, ≤ `max_length`).
    pub min_length: u32,
    /// Maximum password length, in characters (≤ 256).
    pub max_length: u32,
    /// Required character classes (lower, upper, digit, symbol).
    pub require_classes: PasswordClassSet,
    /// Reject passwords that match the in-kernel weak-password
    /// dictionary.
    pub dictionary_check: bool,
}

impl PasswordPolicy {
    /// v1 defaults: 12..=128 chars, lower+upper+digit, dictionary on.
    pub const fn v1_defaults() -> Self {
        Self {
            min_length: 12,
            max_length: 128,
            require_classes: PasswordClassSet::from_mask(
                PasswordClassSet::LOWER | PasswordClassSet::UPPER | PasswordClassSet::DIGIT,
            ),
            dictionary_check: true,
        }
    }
}

// ─── Audit-log policy ────────────────────────────────────────────────────────

/// Audit-log rotation, retention, and signed-checkpoint policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuditConfig {
    /// Rotate the active log when its size exceeds this many bytes.
    pub rotate_size_bytes: u64,
    /// Rotate the active log when it's older than this many hours.
    pub rotate_age_hours: u32,
    /// Number of rotated archives to keep before deleting the oldest.
    pub keep_archives: u32,
    /// Hard cap on bytes used by the log + archives combined.
    pub max_total_disk_bytes: u64,
    /// Emit ML-DSA-65 signed checkpoints. Off in builds without
    /// `signed-checkpoints`; configurable when on. When `true` the
    /// kernel signs every [`Self::signed_checkpoint_interval`]-th
    /// chain head with the kernel-held ML-DSA-65 key.
    pub signed_checkpoints: bool,
    /// Append a signed-checkpoint record every `N` committed records
    /// when [`Self::signed_checkpoints`] is `true`. Per
    /// `management-login-v1` Q24 the default is 1024.
    pub signed_checkpoint_interval: u32,
}

impl AuditConfig {
    /// v1 defaults: 16 MiB / 24 h / 8 archives / 256 MiB total /
    /// signed-checkpoint interval 1024.
    pub const fn v1_defaults() -> Self {
        Self {
            rotate_size_bytes: 16 * 1024 * 1024,
            rotate_age_hours: 24,
            keep_archives: 8,
            max_total_disk_bytes: 256 * 1024 * 1024,
            signed_checkpoints: true,
            signed_checkpoint_interval: 1024,
        }
    }
}

// ─── Metrics policy ──────────────────────────────────────────────────────────

/// Metrics emission cadence + adaptive thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetricsConfig {
    /// CPU-usage sample interval, in milliseconds (≥ 100, ≤ 60_000).
    pub cpu_interval_ms: u32,
    /// Memory-usage sample interval, in milliseconds.
    pub memory_interval_ms: u32,
    /// Network counters sample interval, in milliseconds.
    pub network_interval_ms: u32,
    /// Inference-latency sample interval, in milliseconds.
    pub inference_interval_ms: u32,
    /// Adaptive threshold: emit "load high" if CPU > X%.
    pub adaptive_cpu_high_pct: u32,
    /// Adaptive threshold: emit "load low" if CPU < X%.
    pub adaptive_cpu_low_pct: u32,
}

impl MetricsConfig {
    /// v1 defaults: 1 s / 5 s / 1 s / 500 ms ; 80% / 20%.
    pub const fn v1_defaults() -> Self {
        Self {
            cpu_interval_ms: 1000,
            memory_interval_ms: 5000,
            network_interval_ms: 1000,
            inference_interval_ms: 500,
            adaptive_cpu_high_pct: 80,
            adaptive_cpu_low_pct: 20,
        }
    }
}

// ─── Filesystem policy ───────────────────────────────────────────────────────

/// Filesystem block-device, cache, and boot-watchdog policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsConfig {
    /// Block-device read timeout, in milliseconds.
    pub block_read_timeout_ms: u32,
    /// Block-device retry count on transient errors.
    pub block_retry_count: u32,
    /// Cache budget for `/models/`, in bytes.
    pub cache_models_bytes: u64,
    /// Cache budget for `/data/`, in bytes.
    pub cache_data_bytes: u64,
    /// Boot watchdog, in seconds. `0` disables the watchdog.
    pub boot_watchdog_seconds: u32,
}

impl FsConfig {
    /// v1 defaults: 5 s read timeout / 3 retries / 64 MiB models /
    /// 64 MiB data / 60 s watchdog.
    pub const fn v1_defaults() -> Self {
        Self {
            block_read_timeout_ms: 5000,
            block_retry_count: 3,
            cache_models_bytes: 64 * 1024 * 1024,
            cache_data_bytes: 64 * 1024 * 1024,
            boot_watchdog_seconds: 60,
        }
    }
}

// ─── Management policy ───────────────────────────────────────────────────────

/// Management-surface (Zenoh + tokens) policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MgmtConfig {
    /// Maximum concurrent Zenoh admin sessions per identity.
    pub zenoh_per_identity_max: u32,
    /// Maximum total concurrent Zenoh admin sessions.
    pub zenoh_total_max: u32,
    /// Use the two-tier (long + short) token model.
    pub token_two_tier: bool,
}

impl MgmtConfig {
    /// v1 defaults: 4 per-identity / 32 total / two-tier on.
    pub const fn v1_defaults() -> Self {
        Self {
            zenoh_per_identity_max: 4,
            zenoh_total_max: 32,
            token_two_tier: true,
        }
    }
}

// ─── Overlay policy ──────────────────────────────────────────────────────────

/// Overlay (`/data/overlay`) tunables, owned by `embedded-overlay-v1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OverlayConfig {
    /// Overlay upper-layer hard cap, in bytes.
    pub upper_max_bytes: u64,
    /// Reject unsigned overlay manifests.
    pub require_signed: bool,
    /// Allow `Operator` to un-hide kernel-hidden paths.
    pub allow_operator_unhide: bool,
}

impl OverlayConfig {
    /// v1 defaults: 256 MiB / require signed / Operator may un-hide.
    pub const fn v1_defaults() -> Self {
        Self {
            upper_max_bytes: 256 * 1024 * 1024,
            require_signed: true,
            allow_operator_unhide: true,
        }
    }
}

// ─── Flash policy ────────────────────────────────────────────────────────────

/// Flash routing knobs. The path resolver picks `/flash` (read-only,
/// signed) vs `/data` (read-write) using these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FlashConfig {
    /// Prefer `/flash` for read-only model loads (vs falling back to
    /// `/models` on R/W).
    pub prefer_flash_for_models: bool,
    /// Read-back verify after every flash-region write (slower, more
    /// reliable).
    pub readback_verify: bool,
}

impl FlashConfig {
    /// v1 defaults: prefer flash for models; read-back verify on.
    pub const fn v1_defaults() -> Self {
        Self {
            prefer_flash_for_models: true,
            readback_verify: true,
        }
    }
}

// ─── TOTP policy (Phase 9) ──────────────────────────────────────────────────

/// TOTP (RFC 6238) second-factor enforcement policy.
///
/// `enforced_for_roles` is a [`RoleSet`] bitmask: when a role's bit is
/// set, every user holding that role MUST present a valid TOTP code as
/// `factor2` to `auth_login` and (Phase 10) every privileged
/// `auth_change_password`. The kernel reads this field through the
/// universal-exposure registry; it is reload-kind `Live`, so an
/// operator can dial enforcement up or down without rebooting.
///
/// Defaults: empty set (TOTP is opt-in per user via
/// `auth_totp_setup` + the shadow `totp_required=true` field). The
/// operator opts roles in by setting the bitmask in
/// `mgmt/policy.toml`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TotpConfig {
    /// Bitmask of [`RoleSet`] flags whose users MUST present a TOTP
    /// code at login. Empty (0) ⇒ TOTP enforcement is opt-in per
    /// user via the shadow record.
    pub enforced_for_roles: RoleSet,
}

impl TotpConfig {
    /// v1 defaults: empty (opt-in via shadow).
    pub const fn v1_defaults() -> Self {
        Self {
            enforced_for_roles: RoleSet::from_mask(0),
        }
    }
}

// ─── Path → field projection ─────────────────────────────────────────────────

impl Config {
    /// Read the field at `path` as a [`Value`]. Returns
    /// [`crate::surface::Error::NotFound`] if the path doesn't match
    /// any known field.
    pub fn read_path(&self, path: &ConfigPath) -> Result<Value, crate::surface::Error> {
        match path.as_str() {
            // idle.*
            "idle.root_minutes" => Ok(Value::U32(self.idle.root_minutes)),
            "idle.operator_minutes" => Ok(Value::U32(self.idle.operator_minutes)),
            "idle.viewer_minutes" => Ok(Value::U32(self.idle.viewer_minutes)),

            // password.*
            "password.min_length" => Ok(Value::U32(self.password.min_length)),
            "password.max_length" => Ok(Value::U32(self.password.max_length)),
            "password.require_classes" => Ok(Value::PasswordClasses(self.password.require_classes)),
            "password.dictionary_check" => Ok(Value::Bool(self.password.dictionary_check)),

            // audit.*
            "audit.rotate_size_bytes" => Ok(Value::U64(self.audit.rotate_size_bytes)),
            "audit.rotate_age_hours" => Ok(Value::U32(self.audit.rotate_age_hours)),
            "audit.keep_archives" => Ok(Value::U32(self.audit.keep_archives)),
            "audit.max_total_disk_bytes" => Ok(Value::U64(self.audit.max_total_disk_bytes)),
            "audit.signed_checkpoints" => Ok(Value::Bool(self.audit.signed_checkpoints)),
            "audit.signed_checkpoint_interval" => {
                Ok(Value::U32(self.audit.signed_checkpoint_interval))
            }

            // metrics.*
            "metrics.cpu_interval_ms" => Ok(Value::U32(self.metrics.cpu_interval_ms)),
            "metrics.memory_interval_ms" => Ok(Value::U32(self.metrics.memory_interval_ms)),
            "metrics.network_interval_ms" => Ok(Value::U32(self.metrics.network_interval_ms)),
            "metrics.inference_interval_ms" => Ok(Value::U32(self.metrics.inference_interval_ms)),
            "metrics.adaptive_cpu_high_pct" => Ok(Value::U32(self.metrics.adaptive_cpu_high_pct)),
            "metrics.adaptive_cpu_low_pct" => Ok(Value::U32(self.metrics.adaptive_cpu_low_pct)),

            // fs.*
            "fs.block_read_timeout_ms" => Ok(Value::U32(self.fs.block_read_timeout_ms)),
            "fs.block_retry_count" => Ok(Value::U32(self.fs.block_retry_count)),
            "fs.cache_models_bytes" => Ok(Value::U64(self.fs.cache_models_bytes)),
            "fs.cache_data_bytes" => Ok(Value::U64(self.fs.cache_data_bytes)),
            "fs.boot_watchdog_seconds" => Ok(Value::U32(self.fs.boot_watchdog_seconds)),

            // mgmt.*
            "mgmt.zenoh_per_identity_max" => Ok(Value::U32(self.mgmt.zenoh_per_identity_max)),
            "mgmt.zenoh_total_max" => Ok(Value::U32(self.mgmt.zenoh_total_max)),
            "mgmt.token_two_tier" => Ok(Value::Bool(self.mgmt.token_two_tier)),

            // overlay.*
            "overlay.upper_max_bytes" => Ok(Value::U64(self.overlay.upper_max_bytes)),
            "overlay.require_signed" => Ok(Value::Bool(self.overlay.require_signed)),
            "overlay.allow_operator_unhide" => Ok(Value::Bool(self.overlay.allow_operator_unhide)),

            // flash.*
            "flash.prefer_flash_for_models" => Ok(Value::Bool(self.flash.prefer_flash_for_models)),
            "flash.readback_verify" => Ok(Value::Bool(self.flash.readback_verify)),

            // totp.* (Phase 9)
            "totp.enforced_for_roles" => Ok(Value::Roles(self.totp.enforced_for_roles)),

            _ => Err(crate::surface::Error::NotFound),
        }
    }

    /// Write the field at `path` from `value` after validation. The
    /// caller [`crate::ApplyEngine`] is responsible for applying the
    /// six-step lifecycle around this; this method only mutates the
    /// in-memory struct and is **not** crash-safe by itself.
    ///
    /// Returns [`crate::surface::Error::TypeMismatch`] if `value`
    /// doesn't match the field's variant, [`Error::NotFound`] if the
    /// path is unknown, [`Error::Validation`] if a per-field
    /// validator rejects the new value.
    ///
    /// [`Error::NotFound`]: crate::surface::Error::NotFound
    /// [`Error::Validation`]: crate::surface::Error::Validation
    pub fn write_path(
        &mut self,
        path: &ConfigPath,
        value: Value,
    ) -> Result<(), crate::surface::Error> {
        // Per-field validation lives in `validate.rs`; we call into it
        // before the write so on-disk and in-memory state are
        // synchronised.
        crate::validate::validate_field(path, &value)?;
        crate::validate::validate_cross_field(self, path, &value)?;

        match path.as_str() {
            "idle.root_minutes" => self.idle.root_minutes = value.as_u32()?,
            "idle.operator_minutes" => self.idle.operator_minutes = value.as_u32()?,
            "idle.viewer_minutes" => self.idle.viewer_minutes = value.as_u32()?,

            "password.min_length" => self.password.min_length = value.as_u32()?,
            "password.max_length" => self.password.max_length = value.as_u32()?,
            "password.require_classes" => {
                self.password.require_classes = value.as_password_classes()?
            }
            "password.dictionary_check" => self.password.dictionary_check = value.as_bool()?,

            "audit.rotate_size_bytes" => self.audit.rotate_size_bytes = value.as_u64()?,
            "audit.rotate_age_hours" => self.audit.rotate_age_hours = value.as_u32()?,
            "audit.keep_archives" => self.audit.keep_archives = value.as_u32()?,
            "audit.max_total_disk_bytes" => self.audit.max_total_disk_bytes = value.as_u64()?,
            "audit.signed_checkpoints" => self.audit.signed_checkpoints = value.as_bool()?,
            "audit.signed_checkpoint_interval" => {
                self.audit.signed_checkpoint_interval = value.as_u32()?
            }

            "metrics.cpu_interval_ms" => self.metrics.cpu_interval_ms = value.as_u32()?,
            "metrics.memory_interval_ms" => self.metrics.memory_interval_ms = value.as_u32()?,
            "metrics.network_interval_ms" => self.metrics.network_interval_ms = value.as_u32()?,
            "metrics.inference_interval_ms" => {
                self.metrics.inference_interval_ms = value.as_u32()?
            }
            "metrics.adaptive_cpu_high_pct" => {
                self.metrics.adaptive_cpu_high_pct = value.as_u32()?
            }
            "metrics.adaptive_cpu_low_pct" => self.metrics.adaptive_cpu_low_pct = value.as_u32()?,

            "fs.block_read_timeout_ms" => self.fs.block_read_timeout_ms = value.as_u32()?,
            "fs.block_retry_count" => self.fs.block_retry_count = value.as_u32()?,
            "fs.cache_models_bytes" => self.fs.cache_models_bytes = value.as_u64()?,
            "fs.cache_data_bytes" => self.fs.cache_data_bytes = value.as_u64()?,
            "fs.boot_watchdog_seconds" => self.fs.boot_watchdog_seconds = value.as_u32()?,

            "mgmt.zenoh_per_identity_max" => self.mgmt.zenoh_per_identity_max = value.as_u32()?,
            "mgmt.zenoh_total_max" => self.mgmt.zenoh_total_max = value.as_u32()?,
            "mgmt.token_two_tier" => self.mgmt.token_two_tier = value.as_bool()?,

            "overlay.upper_max_bytes" => self.overlay.upper_max_bytes = value.as_u64()?,
            "overlay.require_signed" => self.overlay.require_signed = value.as_bool()?,
            "overlay.allow_operator_unhide" => {
                self.overlay.allow_operator_unhide = value.as_bool()?
            }

            "flash.prefer_flash_for_models" => {
                self.flash.prefer_flash_for_models = value.as_bool()?
            }
            "flash.readback_verify" => self.flash.readback_verify = value.as_bool()?,

            "totp.enforced_for_roles" => self.totp.enforced_for_roles = value.as_roles()?,

            _ => return Err(crate::surface::Error::NotFound),
        }
        Ok(())
    }
}

// ─── Path registry (universal-exposure walker input) ─────────────────────────

pub mod registry {
    //! Const slice describing every known [`super::Config`] field.
    //!
    //! The universal-exposure walker (build-time test) iterates this
    //! slice and asserts every entry maps to a writable field on
    //! `Config` and (when surfaces land) a handler on every required
    //! surface.

    use crate::surface::{ReloadKind, SurfaceScope};

    /// Type tag for the typed [`crate::surface::Value`] expected at
    /// the path.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum FieldType {
        /// `Value::Bool`.
        Bool,
        /// `Value::U32`.
        U32,
        /// `Value::U64`.
        U64,
        /// `Value::String`.
        String,
        /// `Value::PasswordClasses`.
        PasswordClasses,
        /// `Value::Roles` — bitmask of [`crate::surface::RoleSet`].
        Roles,
    }

    impl FieldType {
        /// Variant tag used for diagnostics (`"bool"`, `"u32"` …).
        pub const fn as_value_tag(self) -> &'static str {
            match self {
                Self::Bool => "bool",
                Self::U32 => "u32",
                Self::U64 => "u64",
                Self::String => "string",
                Self::PasswordClasses => "password_classes",
                Self::Roles => "roles",
            }
        }
    }

    /// One row of the registry — a single field's metadata.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct FieldDescriptor {
        /// Canonical dotted path (matches [`super::Config::read_path`]).
        pub path: &'static str,
        /// Typed value tag — drives `Value` variant selection.
        pub kind: FieldType,
        /// Reload semantics — live or boot.
        pub reload: ReloadKind,
        /// Surface-exposure scope — drives the universal-exposure
        /// walker.
        pub scope: SurfaceScope,
    }

    /// Every field in the v1 [`super::Config`]. Order does not matter
    /// for correctness but is grouped by namespace for readability.
    ///
    /// **Adding a new `Config` field?** Append a row here. The
    /// build-time test in `lib.rs` will fail compilation if a field
    /// exists in the struct but not in this registry, and vice versa.
    pub const CONFIG_FIELDS: &[FieldDescriptor] = &[
        // idle.*
        FieldDescriptor {
            path: "idle.root_minutes",
            kind: FieldType::U32,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "idle.operator_minutes",
            kind: FieldType::U32,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "idle.viewer_minutes",
            kind: FieldType::U32,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        // password.*
        FieldDescriptor {
            path: "password.min_length",
            kind: FieldType::U32,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "password.max_length",
            kind: FieldType::U32,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "password.require_classes",
            kind: FieldType::PasswordClasses,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "password.dictionary_check",
            kind: FieldType::Bool,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        // audit.*
        FieldDescriptor {
            path: "audit.rotate_size_bytes",
            kind: FieldType::U64,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "audit.rotate_age_hours",
            kind: FieldType::U32,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "audit.keep_archives",
            kind: FieldType::U32,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "audit.max_total_disk_bytes",
            kind: FieldType::U64,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "audit.signed_checkpoints",
            kind: FieldType::Bool,
            // Toggling signed checkpoints alters the on-disk audit
            // format mid-stream, so it's a boot-time field.
            reload: ReloadKind::Boot,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "audit.signed_checkpoint_interval",
            kind: FieldType::U32,
            // The checkpoint cadence is read fresh on every commit by
            // the audit ring, so live reload is safe.
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        // metrics.*
        FieldDescriptor {
            path: "metrics.cpu_interval_ms",
            kind: FieldType::U32,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "metrics.memory_interval_ms",
            kind: FieldType::U32,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "metrics.network_interval_ms",
            kind: FieldType::U32,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "metrics.inference_interval_ms",
            kind: FieldType::U32,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "metrics.adaptive_cpu_high_pct",
            kind: FieldType::U32,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "metrics.adaptive_cpu_low_pct",
            kind: FieldType::U32,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        // fs.*
        FieldDescriptor {
            path: "fs.block_read_timeout_ms",
            kind: FieldType::U32,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "fs.block_retry_count",
            kind: FieldType::U32,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "fs.cache_models_bytes",
            kind: FieldType::U64,
            // Cache budgets re-allocate at boot.
            reload: ReloadKind::Boot,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "fs.cache_data_bytes",
            kind: FieldType::U64,
            reload: ReloadKind::Boot,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "fs.boot_watchdog_seconds",
            kind: FieldType::U32,
            // Watchdog is armed once at boot.
            reload: ReloadKind::Boot,
            scope: SurfaceScope::Universal,
        },
        // mgmt.*
        FieldDescriptor {
            path: "mgmt.zenoh_per_identity_max",
            kind: FieldType::U32,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "mgmt.zenoh_total_max",
            kind: FieldType::U32,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "mgmt.token_two_tier",
            kind: FieldType::Bool,
            // Token-format change requires fresh tokens.
            reload: ReloadKind::Boot,
            scope: SurfaceScope::Universal,
        },
        // overlay.*
        FieldDescriptor {
            path: "overlay.upper_max_bytes",
            kind: FieldType::U64,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "overlay.require_signed",
            kind: FieldType::Bool,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "overlay.allow_operator_unhide",
            kind: FieldType::Bool,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        // flash.*
        FieldDescriptor {
            path: "flash.prefer_flash_for_models",
            kind: FieldType::Bool,
            reload: ReloadKind::Boot,
            scope: SurfaceScope::Universal,
        },
        FieldDescriptor {
            path: "flash.readback_verify",
            kind: FieldType::Bool,
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
        // totp.* (Phase 9)
        FieldDescriptor {
            path: "totp.enforced_for_roles",
            kind: FieldType::Roles,
            // The kernel reads this on every login attempt, so a
            // change takes effect on the next login without a reboot.
            reload: ReloadKind::Live,
            scope: SurfaceScope::Universal,
        },
    ];

    /// Look up a field by canonical path. `O(n)` over the registry —
    /// the registry is small and reads are infrequent compared to
    /// hot-path inference work, so a hash is overkill.
    pub fn lookup(path: &str) -> Option<&'static FieldDescriptor> {
        CONFIG_FIELDS.iter().find(|f| f.path == path)
    }

    /// Reload kind for `path`, or `None` if unknown.
    pub fn reload_kind(path: &str) -> Option<ReloadKind> {
        lookup(path).map(|f| f.reload)
    }

    /// Surface scope for `path`, or `None` if unknown.
    pub fn surface_scope(path: &str) -> Option<SurfaceScope> {
        lookup(path).map(|f| f.scope)
    }

    /// Number of registered fields. Stable across feature flags in v1
    /// (no cargo-feature-gated fields yet).
    pub const fn field_count() -> usize {
        CONFIG_FIELDS.len()
    }
}

/// Re-export the descriptor + registry types under the public
/// [`crate::config`] namespace for downstream callers.
pub use registry::{FieldDescriptor, FieldType};

/// Wrapper around [`registry::reload_kind`] returning the spec-defined
/// default ([`ReloadKind::Live`]) for unknown paths. Used by surfaces
/// that need to ask "would a write here be live or boot-deferred?"
/// without bubbling the error.
pub fn reload_kind_of(path: &ConfigPath) -> ReloadKind {
    registry::reload_kind(path.as_str()).unwrap_or(ReloadKind::Live)
}

/// Wrapper around [`registry::surface_scope`] returning the spec
/// default ([`SurfaceScope::Universal`]) for unknown paths.
pub fn surface_scope_of(path: &ConfigPath) -> SurfaceScope {
    registry::surface_scope(path.as_str()).unwrap_or(SurfaceScope::Universal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::{ConfigPath, Value};

    #[test]
    fn defaults_are_internally_consistent() {
        let c = Config::default();
        // Sanity: defaults should pass cross-field validation.
        crate::validate::validate_all(&c).expect("v1 defaults validate");
    }

    #[test]
    fn read_round_trips_every_namespace_default() {
        let c = Config::default();

        assert_eq!(
            c.read_path(&ConfigPath::new("idle.root_minutes").unwrap()),
            Ok(Value::U32(15))
        );
        assert_eq!(
            c.read_path(&ConfigPath::new("password.min_length").unwrap()),
            Ok(Value::U32(12))
        );
        assert_eq!(
            c.read_path(&ConfigPath::new("audit.rotate_size_bytes").unwrap()),
            Ok(Value::U64(16 * 1024 * 1024))
        );
        assert_eq!(
            c.read_path(&ConfigPath::new("metrics.cpu_interval_ms").unwrap()),
            Ok(Value::U32(1000))
        );
        assert_eq!(
            c.read_path(&ConfigPath::new("fs.boot_watchdog_seconds").unwrap()),
            Ok(Value::U32(60))
        );
        assert_eq!(
            c.read_path(&ConfigPath::new("mgmt.zenoh_total_max").unwrap()),
            Ok(Value::U32(32))
        );
        assert_eq!(
            c.read_path(&ConfigPath::new("overlay.require_signed").unwrap()),
            Ok(Value::Bool(true))
        );
        assert_eq!(
            c.read_path(&ConfigPath::new("flash.readback_verify").unwrap()),
            Ok(Value::Bool(true))
        );
    }

    #[test]
    fn read_unknown_path_returns_not_found() {
        let c = Config::default();
        let p = ConfigPath::new("does.not.exist").unwrap();
        assert!(matches!(
            c.read_path(&p),
            Err(crate::surface::Error::NotFound)
        ));
    }

    #[test]
    fn write_path_updates_in_memory_field() {
        let mut c = Config::default();
        let p = ConfigPath::new("idle.root_minutes").unwrap();
        c.write_path(&p, Value::U32(45)).unwrap();
        assert_eq!(c.idle.root_minutes, 45);
        assert_eq!(c.read_path(&p), Ok(Value::U32(45)));
    }

    #[test]
    fn write_path_validates_range() {
        let mut c = Config::default();
        // Out of range: idle.root_minutes must be ≥ 1.
        let p = ConfigPath::new("idle.root_minutes").unwrap();
        let err = c.write_path(&p, Value::U32(0)).unwrap_err();
        assert!(matches!(err, crate::surface::Error::Validation(_)));
        // In-memory unchanged.
        assert_eq!(c.idle.root_minutes, 15);
    }

    #[test]
    fn write_path_rejects_type_mismatch() {
        let mut c = Config::default();
        let p = ConfigPath::new("idle.root_minutes").unwrap();
        let err = c.write_path(&p, Value::Bool(true)).unwrap_err();
        assert!(matches!(err, crate::surface::Error::TypeMismatch { .. }));
    }

    #[test]
    fn registry_covers_every_known_path_and_vice_versa() {
        // Walk the registry: each path must be readable.
        let c = Config::default();
        for f in registry::CONFIG_FIELDS {
            let p = ConfigPath::new(f.path).expect("registry path is valid");
            let v = c.read_path(&p).expect("registry path is readable");
            // And the variant tag must agree.
            assert_eq!(
                v.type_tag(),
                f.kind.as_value_tag(),
                "type tag mismatch for {}",
                f.path
            );
        }
    }

    #[test]
    fn registry_has_no_duplicates() {
        let mut seen: alloc::vec::Vec<&str> = alloc::vec::Vec::new();
        for f in registry::CONFIG_FIELDS {
            assert!(
                !seen.contains(&f.path),
                "duplicate path in registry: {}",
                f.path
            );
            seen.push(f.path);
        }
    }

    #[test]
    fn registry_field_count_matches_const() {
        assert_eq!(registry::field_count(), registry::CONFIG_FIELDS.len());
    }

    #[test]
    fn reload_kind_of_returns_registered_value() {
        let live = ConfigPath::new("idle.root_minutes").unwrap();
        let boot = ConfigPath::new("fs.boot_watchdog_seconds").unwrap();
        assert_eq!(reload_kind_of(&live), ReloadKind::Live);
        assert_eq!(reload_kind_of(&boot), ReloadKind::Boot);
    }

    #[test]
    fn reload_kind_of_unknown_defaults_live() {
        let unknown = ConfigPath::new("nope.nada").unwrap();
        assert_eq!(reload_kind_of(&unknown), ReloadKind::Live);
    }

    #[test]
    fn surface_scope_of_universal_for_v1_fields() {
        for f in registry::CONFIG_FIELDS {
            assert_eq!(f.scope, SurfaceScope::Universal, "scope for {}", f.path);
        }
    }

    #[test]
    fn registry_lookup_returns_descriptor() {
        let f = registry::lookup("metrics.cpu_interval_ms").expect("known path");
        assert_eq!(f.kind, FieldType::U32);
        assert_eq!(f.reload, ReloadKind::Live);
    }

    #[test]
    fn registry_lookup_returns_none_for_unknown() {
        assert!(registry::lookup("nope.nada").is_none());
    }

    #[test]
    fn config_clone_independent_of_original() {
        let mut a = Config::default();
        let b = a.clone();
        let p = ConfigPath::new("idle.root_minutes").unwrap();
        a.write_path(&p, Value::U32(30)).unwrap();
        assert_eq!(a.idle.root_minutes, 30);
        assert_eq!(b.idle.root_minutes, 15);
    }

    #[test]
    fn boolean_fields_round_trip() {
        let mut c = Config::default();
        let p = ConfigPath::new("password.dictionary_check").unwrap();
        assert_eq!(c.read_path(&p), Ok(Value::Bool(true)));
        c.write_path(&p, Value::Bool(false)).unwrap();
        assert_eq!(c.read_path(&p), Ok(Value::Bool(false)));
    }

    #[test]
    fn u64_fields_round_trip() {
        let mut c = Config::default();
        let p = ConfigPath::new("audit.rotate_size_bytes").unwrap();
        c.write_path(&p, Value::U64(8 * 1024 * 1024)).unwrap();
        assert_eq!(c.read_path(&p), Ok(Value::U64(8 * 1024 * 1024)));
    }

    #[test]
    fn password_classes_round_trip() {
        let mut c = Config::default();
        let p = ConfigPath::new("password.require_classes").unwrap();
        let new_set = PasswordClassSet::from_mask(
            PasswordClassSet::LOWER
                | PasswordClassSet::UPPER
                | PasswordClassSet::DIGIT
                | PasswordClassSet::SYMBOL,
        );
        c.write_path(&p, Value::PasswordClasses(new_set)).unwrap();
        assert_eq!(c.read_path(&p), Ok(Value::PasswordClasses(new_set)));
    }

    // ─── Phase 9: TOTP enforcement field ────────────────────────────────

    #[test]
    fn totp_default_is_empty_role_set() {
        let c = Config::default();
        let v = c
            .read_path(&ConfigPath::new("totp.enforced_for_roles").unwrap())
            .unwrap();
        assert_eq!(v, Value::Roles(crate::surface::RoleSet::from_mask(0)));
    }

    #[test]
    fn totp_enforced_for_roles_round_trip() {
        // Spec: every role bit can be flipped on/off independently.
        let mut c = Config::default();
        let p = ConfigPath::new("totp.enforced_for_roles").unwrap();
        let mask = crate::surface::RoleSet::ROOT | crate::surface::RoleSet::OPERATOR;
        let new_set = crate::surface::RoleSet::from_mask(mask);
        c.write_path(&p, Value::Roles(new_set)).unwrap();
        assert_eq!(c.read_path(&p), Ok(Value::Roles(new_set)));
    }

    #[test]
    fn totp_field_is_registered_with_live_reload() {
        // The kernel reads `totp.enforced_for_roles` on every login,
        // so the registry SHALL declare it `Live` (no reboot needed).
        let f = registry::lookup("totp.enforced_for_roles").expect("registered");
        assert_eq!(f.kind, FieldType::Roles);
        assert_eq!(f.reload, ReloadKind::Live);
        assert_eq!(f.scope, SurfaceScope::Universal);
    }

    #[test]
    fn totp_field_rejects_type_mismatch_on_write() {
        let mut c = Config::default();
        let p = ConfigPath::new("totp.enforced_for_roles").unwrap();
        let err = c.write_path(&p, Value::U32(7)).unwrap_err();
        assert!(matches!(err, crate::surface::Error::TypeMismatch { .. }));
    }

    #[test]
    fn totp_field_rejects_unknown_role_bits() {
        let mut c = Config::default();
        let p = ConfigPath::new("totp.enforced_for_roles").unwrap();
        // Hand-crafted set with an unknown bit (bit 7).
        let bogus = crate::surface::RoleSet(0x80);
        let err = c.write_path(&p, Value::Roles(bogus)).unwrap_err();
        assert!(matches!(err, crate::surface::Error::Validation(_)));
    }
}
