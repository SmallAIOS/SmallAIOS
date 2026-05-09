// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! `ConfigSurface` trait + typed [`ConfigPath`] / [`Value`] / [`Error`].
//!
//! Spec:
//! `openspec/changes/management-login-v1/specs/mgmt-config-surface-trait/spec.md`.
//!
//! ## What lives here
//!
//! - [`ConfigPath`] — typed dotted path through the [`crate::Config`]
//!   schema, e.g. `idle.root_minutes` or `password.require_classes`.
//!   Validated at construction; cheap to clone and compare.
//! - [`Value`] — sum type carrying the typed value at a path. v1
//!   variants cover the field types in `mgmt-config-model`.
//! - [`Error`] — surface-level errors: `NotFound`, `TypeMismatch`,
//!   `Validation`, `Io`, `Permission`. Surfaces map their concrete
//!   transport errors onto these variants.
//! - [`ConfigSurface`] — every active management surface (TOML, TTY,
//!   Zenoh admin, future UDS-on-CAN, future REST proxy) implements
//!   this trait over the same in-memory [`crate::Config`]. The
//!   `subscribe` method returns a [`crate::notify::Subscriber`] so a
//!   subscriber pulls notifications cooperatively.
//!
//! ## Universal-exposure invariant
//!
//! Every option in [`crate::Config`] SHALL be reachable from **all**
//! active surfaces. The build-time walker in [`crate::config::registry`]
//! drives the gate; this module only defines the trait surface.

use alloc::borrow::ToOwned;
use alloc::string::String;
#[cfg(test)]
use alloc::string::ToString;
use alloc::vec::Vec;
use core::fmt;

use crate::notify::Subscriber;

/// Maximum depth of a [`ConfigPath`]. The v1 schema only nests two
/// levels (`namespace.field`), but a future namespace (e.g.
/// `network.eth0.mtu`) needs three; round up to four for headroom.
pub const MAX_PATH_DEPTH: usize = 4;

/// Maximum length of any single segment within a [`ConfigPath`]. Keeps
/// path comparisons predictable on cooperative tasks.
pub const MAX_PATH_SEGMENT_LEN: usize = 64;

// ─── ConfigPath ──────────────────────────────────────────────────────────────

/// A typed dotted path through the [`crate::Config`] schema.
///
/// Constructed via [`ConfigPath::new`] (validates) or
/// [`ConfigPath::from_segments`] (pre-split). Paths are always
/// normalised to lowercase ASCII; segments may contain `[a-z0-9_]`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ConfigPath {
    /// Canonical dotted form, e.g. `"idle.root_minutes"`. Owned so the
    /// path is `'static`-free and can be embedded in audit records.
    canonical: String,
    /// Pre-split segments — small `Vec` of small `String`s. Doing this
    /// at construction makes prefix checks (subscribers, surfaces)
    /// allocation-free.
    segments: Vec<String>,
}

impl ConfigPath {
    /// Parse `"a.b.c"` into a [`ConfigPath`]. Validates each segment
    /// against the alphabet `[a-z0-9_]` and the depth/length bounds.
    pub fn new(s: &str) -> Result<Self, Error> {
        if s.is_empty() {
            return Err(Error::InvalidPath("empty path"));
        }
        let mut segments: Vec<String> = Vec::new();
        for raw in s.split('.') {
            if raw.is_empty() {
                return Err(Error::InvalidPath("empty segment"));
            }
            if raw.len() > MAX_PATH_SEGMENT_LEN {
                return Err(Error::InvalidPath("segment too long"));
            }
            for ch in raw.chars() {
                if !is_path_char(ch) {
                    return Err(Error::InvalidPath("bad character"));
                }
            }
            segments.push(raw.to_owned());
            if segments.len() > MAX_PATH_DEPTH {
                return Err(Error::InvalidPath("path too deep"));
            }
        }
        Ok(Self {
            canonical: s.to_owned(),
            segments,
        })
    }

    /// Construct from a pre-split list of segments. Each segment is
    /// validated independently.
    pub fn from_segments(parts: &[&str]) -> Result<Self, Error> {
        if parts.is_empty() {
            return Err(Error::InvalidPath("empty path"));
        }
        if parts.len() > MAX_PATH_DEPTH {
            return Err(Error::InvalidPath("path too deep"));
        }
        let mut segments: Vec<String> = Vec::with_capacity(parts.len());
        let mut canonical = String::new();
        for (i, raw) in parts.iter().enumerate() {
            if raw.is_empty() {
                return Err(Error::InvalidPath("empty segment"));
            }
            if raw.len() > MAX_PATH_SEGMENT_LEN {
                return Err(Error::InvalidPath("segment too long"));
            }
            for ch in raw.chars() {
                if !is_path_char(ch) {
                    return Err(Error::InvalidPath("bad character"));
                }
            }
            if i > 0 {
                canonical.push('.');
            }
            canonical.push_str(raw);
            segments.push((*raw).to_owned());
        }
        Ok(Self {
            canonical,
            segments,
        })
    }

    /// Canonical dotted form.
    pub fn as_str(&self) -> &str {
        &self.canonical
    }

    /// Pre-split segment slice.
    pub fn segments(&self) -> &[String] {
        &self.segments
    }

    /// Number of segments — `1` for a top-level field.
    pub fn depth(&self) -> usize {
        self.segments.len()
    }

    /// Top-level namespace (`"idle"` for `"idle.root_minutes"`).
    pub fn namespace(&self) -> &str {
        &self.segments[0]
    }

    /// Returns true iff `prefix` is a prefix of this path.
    pub fn starts_with(&self, prefix: &ConfigPath) -> bool {
        if prefix.segments.len() > self.segments.len() {
            return false;
        }
        prefix
            .segments
            .iter()
            .zip(self.segments.iter())
            .all(|(a, b)| a == b)
    }
}

impl fmt::Display for ConfigPath {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.canonical)
    }
}

fn is_path_char(ch: char) -> bool {
    ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_'
}

// ─── Value ───────────────────────────────────────────────────────────────────

/// Typed value carrier crossing the [`ConfigSurface`] boundary.
///
/// Variants match the field types in `mgmt-config-model`. `String` is
/// the only owning variant; the rest are `Copy` and free to pass by
/// value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Boolean — feature toggles, `dictionary_check`, etc.
    Bool(bool),
    /// 32-bit unsigned — small bounded counts (e.g. minute timeouts,
    /// retry counts, password lengths).
    U32(u32),
    /// 64-bit unsigned — byte sizes, periods in milliseconds.
    U64(u64),
    /// UTF-8 string — paths, identifiers.
    String(String),
    /// Set of [`PasswordClass`] flags.
    PasswordClasses(PasswordClassSet),
    /// Set of [`auth::Role`] flags — Phase 9 TOTP-enforcement roles.
    ///
    /// [`auth::Role`]: https://docs.rs/smallaios-auth/latest/smallaios_auth/role/enum.Role.html
    Roles(RoleSet),
}

/// Bitmask of [`auth::Role`] values, used by Phase 9's
/// `mgmt.totp.enforced_for_roles` field.
///
/// Mirrors `auth::Role` byte-for-byte: bit 0 = `Root`, bit 1 =
/// `Operator`, bit 2 = `Viewer`. Empty mask (`0`) means TOTP
/// enforcement is opt-in (no role is forced into the second-factor
/// gate). Operators add bits to require TOTP for the listed roles at
/// `auth_login` time.
///
/// [`auth::Role`]: https://docs.rs/smallaios-auth/latest/smallaios_auth/role/enum.Role.html
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoleSet(pub u8);

impl RoleSet {
    /// Bit position for `Role::Root`.
    pub const ROOT: u8 = 0b0000_0001;
    /// Bit position for `Role::Operator`.
    pub const OPERATOR: u8 = 0b0000_0010;
    /// Bit position for `Role::Viewer`.
    pub const VIEWER: u8 = 0b0000_0100;
    /// Bitwise OR of all known role bits. Useful for masking input.
    pub const ALL: u8 = Self::ROOT | Self::OPERATOR | Self::VIEWER;

    /// Construct from a raw mask, dropping unknown bits.
    pub const fn from_mask(mask: u8) -> Self {
        Self(mask & Self::ALL)
    }

    /// Returns true if `flag` is set.
    pub const fn contains(self, flag: u8) -> bool {
        self.0 & flag == flag
    }

    /// Number of selected roles (popcount).
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// Returns true if no role is selected.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Bitmask of password character-class requirements.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PasswordClassSet(pub u8);

impl PasswordClassSet {
    /// At least one lowercase letter required.
    pub const LOWER: u8 = 0b0000_0001;
    /// At least one uppercase letter required.
    pub const UPPER: u8 = 0b0000_0010;
    /// At least one digit required.
    pub const DIGIT: u8 = 0b0000_0100;
    /// At least one symbol (`!@#$%^&*` …) required.
    pub const SYMBOL: u8 = 0b0000_1000;

    /// Bitwise OR of all known class flags. Useful for masking input.
    pub const ALL: u8 = Self::LOWER | Self::UPPER | Self::DIGIT | Self::SYMBOL;

    /// Construct from a raw mask, dropping unknown bits.
    pub const fn from_mask(mask: u8) -> Self {
        Self(mask & Self::ALL)
    }

    /// Returns true if this set requires `flag`.
    pub const fn contains(self, flag: u8) -> bool {
        self.0 & flag == flag
    }

    /// Number of required classes (popcount).
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }
}

impl Value {
    /// Variant tag string for diagnostics (`"bool"`, `"u32"`, etc.).
    pub fn type_tag(&self) -> &'static str {
        match self {
            Self::Bool(_) => "bool",
            Self::U32(_) => "u32",
            Self::U64(_) => "u64",
            Self::String(_) => "string",
            Self::PasswordClasses(_) => "password_classes",
            Self::Roles(_) => "roles",
        }
    }

    /// Try to extract a `bool`.
    pub fn as_bool(&self) -> Result<bool, Error> {
        if let Self::Bool(b) = self {
            Ok(*b)
        } else {
            Err(Error::TypeMismatch {
                expected: "bool",
                got: self.type_tag(),
            })
        }
    }

    /// Try to extract a `u32`.
    pub fn as_u32(&self) -> Result<u32, Error> {
        if let Self::U32(n) = self {
            Ok(*n)
        } else {
            Err(Error::TypeMismatch {
                expected: "u32",
                got: self.type_tag(),
            })
        }
    }

    /// Try to extract a `u64`.
    pub fn as_u64(&self) -> Result<u64, Error> {
        if let Self::U64(n) = self {
            Ok(*n)
        } else {
            Err(Error::TypeMismatch {
                expected: "u64",
                got: self.type_tag(),
            })
        }
    }

    /// Try to extract a string slice.
    pub fn as_str(&self) -> Result<&str, Error> {
        if let Self::String(s) = self {
            Ok(s.as_str())
        } else {
            Err(Error::TypeMismatch {
                expected: "string",
                got: self.type_tag(),
            })
        }
    }

    /// Try to extract a [`PasswordClassSet`].
    pub fn as_password_classes(&self) -> Result<PasswordClassSet, Error> {
        if let Self::PasswordClasses(p) = self {
            Ok(*p)
        } else {
            Err(Error::TypeMismatch {
                expected: "password_classes",
                got: self.type_tag(),
            })
        }
    }

    /// Try to extract a [`RoleSet`] — used for Phase 9
    /// `mgmt.totp.enforced_for_roles`.
    pub fn as_roles(&self) -> Result<RoleSet, Error> {
        if let Self::Roles(r) = self {
            Ok(*r)
        } else {
            Err(Error::TypeMismatch {
                expected: "roles",
                got: self.type_tag(),
            })
        }
    }
}

// ─── Error ───────────────────────────────────────────────────────────────────

/// Errors raised by [`ConfigSurface`] implementations.
///
/// Variants are deliberately coarse — surfaces map their concrete
/// transport errors onto these. The detailed reason for a validation
/// failure is carried in the inline `&'static str` so log output
/// retains the diagnostic without forcing every caller to depend on
/// the validator types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// Path doesn't correspond to any field in the active schema.
    NotFound,
    /// Type of the supplied [`Value`] doesn't match the field's type.
    TypeMismatch {
        /// Expected variant tag.
        expected: &'static str,
        /// Variant tag actually supplied.
        got: &'static str,
    },
    /// Per-field or cross-field validation failed. The reason is a
    /// fixed string from the validator.
    Validation(&'static str),
    /// The path syntactically isn't a valid [`ConfigPath`].
    InvalidPath(&'static str),
    /// Backing I/O surface (TOML loader, Zenoh transport) failed.
    Io(&'static str),
    /// Caller does not have the role required for the operation.
    Permission(&'static str),
    /// Unsupported on this surface (e.g., subscribe on a one-shot
    /// loader that only supports `read`/`write`).
    Unsupported(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotFound => write!(f, "mgmt::surface: path not found"),
            Self::TypeMismatch { expected, got } => write!(
                f,
                "mgmt::surface: type mismatch, expected {expected}, got {got}"
            ),
            Self::Validation(reason) => write!(f, "mgmt::surface: validation: {reason}"),
            Self::InvalidPath(reason) => write!(f, "mgmt::surface: invalid path: {reason}"),
            Self::Io(reason) => write!(f, "mgmt::surface: io: {reason}"),
            Self::Permission(reason) => write!(f, "mgmt::surface: permission: {reason}"),
            Self::Unsupported(reason) => write!(f, "mgmt::surface: unsupported: {reason}"),
        }
    }
}

// ─── Reload semantics ────────────────────────────────────────────────────────

/// Per-field reload semantics. Drives whether [`crate::ApplyEngine`]
/// makes a successful write visible immediately (live) or defers to
/// next boot (boot).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReloadKind {
    /// Live — the in-memory `Config` is updated on commit and
    /// subscribers receive a notify event so they can re-read at
    /// their own pace.
    Live,
    /// Boot — the file is rewritten with the new value, but the
    /// in-memory `Config` retains the old value until next boot.
    /// [`crate::ApplyEngine::pending_reboot`] lists the deferred path.
    Boot,
}

/// Surface scope for the universal-exposure walker. A field annotated
/// `#[surface(only = "tty")]` (or whatever surface mix) restricts the
/// walker to only require handlers on the named surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceScope {
    /// All active surfaces SHALL provide a handler — the default.
    Universal,
    /// Only the listed surfaces SHALL provide a handler.
    OnlyTty,
    /// Only the TOML loader needs to provide a handler. Used for
    /// fields that only make sense at file-load time.
    OnlyToml,
}

/// All surfaces recognised by the universal-exposure walker. New
/// surfaces SHALL extend this enum **and** the
/// [`crate::config::registry::CONFIG_FIELDS`] schema gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceKind {
    /// `/data/*.toml` loader/saver.
    Toml,
    /// Console TTY login surface.
    Tty,
    /// Zenoh admin transport (`smallaios/admin/**`).
    Zenoh,
}

impl SurfaceKind {
    /// Lower-case canonical name used in `#[surface(only = ...)]`
    /// attributes.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Toml => "toml",
            Self::Tty => "tty",
            Self::Zenoh => "zenoh",
        }
    }

    /// Parse the canonical name. Returns `None` for unknown names —
    /// the walker uses this to flag mistyped surface annotations.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "toml" => Some(Self::Toml),
            "tty" => Some(Self::Tty),
            "zenoh" => Some(Self::Zenoh),
            _ => None,
        }
    }
}

// ─── ConfigSurface trait ─────────────────────────────────────────────────────

/// Every management surface implements this trait over the same
/// in-memory [`crate::Config`].
///
/// The shape mirrors the spec exactly. `subscribe` returns a
/// [`Subscriber`] (the `Stream` from the spec) so callers pull
/// notifications cooperatively — no callbacks, no preemption. The
/// `path` argument acts as a filter; passing the root namespace
/// subscribes to every field within it.
pub trait ConfigSurface {
    /// Read the current value at `path`.
    fn read(&self, path: &ConfigPath) -> Result<Value, Error>;

    /// Write `value` at `path` through the apply lifecycle. Returns
    /// `Ok(())` only after step (4) — `fsync` + atomic `rename` —
    /// completes. Validation failures, IO failures, and permission
    /// failures all leave both the on-disk file and the in-memory
    /// `Config` unchanged.
    fn write(&mut self, path: &ConfigPath, value: Value, who: &str) -> Result<(), Error>;

    /// Subscribe to notify events filtered by `path`. The subscriber
    /// observes events published **after** the call.
    fn subscribe(&self, path: &ConfigPath) -> Result<Subscriber, Error>;
}

/// Identity carried through the apply lifecycle into the audit ring.
///
/// This is a thin wrapper rather than `String` so callers can record
/// a stable origin without building a `(role, name, source)` tuple at
/// every call site. The audit ring is the canonical authority — the
/// notify channel only sees the rendered `who` string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditIdentity {
    /// Role taking the action: `"root"`, `"operator"`, or `"viewer"`.
    pub role: &'static str,
    /// Optional username. `None` means "the role's bootstrap account"
    /// (e.g., `Root` at first boot before names exist).
    pub user: Option<String>,
    /// Originating surface that produced the call.
    pub surface: SurfaceKind,
}

impl AuditIdentity {
    /// Render this identity to the `who` string used in
    /// [`crate::notify::NotifyEvent`] and audit records.
    pub fn render(&self) -> String {
        let mut s = String::new();
        s.push_str(self.role);
        if let Some(user) = &self.user {
            s.push(':');
            s.push_str(user);
        }
        s.push('@');
        s.push_str(self.surface.name());
        s
    }
}

impl fmt::Display for AuditIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.render())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_path() {
        let p = ConfigPath::new("idle.root_minutes").unwrap();
        assert_eq!(p.depth(), 2);
        assert_eq!(p.namespace(), "idle");
        assert_eq!(p.as_str(), "idle.root_minutes");
        assert_eq!(p.segments()[1], "root_minutes");
    }

    #[test]
    fn parse_single_segment() {
        let p = ConfigPath::new("flash").unwrap();
        assert_eq!(p.depth(), 1);
        assert_eq!(p.namespace(), "flash");
    }

    #[test]
    fn rejects_empty_path() {
        assert!(matches!(ConfigPath::new(""), Err(Error::InvalidPath(_))));
    }

    #[test]
    fn rejects_empty_segment() {
        assert!(matches!(
            ConfigPath::new("idle..root_minutes"),
            Err(Error::InvalidPath(_))
        ));
        assert!(matches!(
            ConfigPath::new(".idle"),
            Err(Error::InvalidPath(_))
        ));
        assert!(matches!(
            ConfigPath::new("idle."),
            Err(Error::InvalidPath(_))
        ));
    }

    #[test]
    fn rejects_uppercase() {
        assert!(matches!(
            ConfigPath::new("Idle.root_minutes"),
            Err(Error::InvalidPath(_))
        ));
    }

    #[test]
    fn rejects_punctuation() {
        assert!(matches!(
            ConfigPath::new("idle-config"),
            Err(Error::InvalidPath(_))
        ));
        assert!(matches!(
            ConfigPath::new("idle/cfg"),
            Err(Error::InvalidPath(_))
        ));
    }

    #[test]
    fn rejects_too_deep() {
        // MAX_PATH_DEPTH = 4 — five segments must fail.
        assert!(matches!(
            ConfigPath::new("a.b.c.d.e"),
            Err(Error::InvalidPath(_))
        ));
        // Four segments must succeed.
        assert!(ConfigPath::new("a.b.c.d").is_ok());
    }

    #[test]
    fn rejects_too_long_segment() {
        let big = "a".repeat(MAX_PATH_SEGMENT_LEN + 1);
        assert!(matches!(ConfigPath::new(&big), Err(Error::InvalidPath(_))));
    }

    #[test]
    fn from_segments_round_trip() {
        let p = ConfigPath::from_segments(&["idle", "root_minutes"]).unwrap();
        assert_eq!(p.as_str(), "idle.root_minutes");
        assert_eq!(p, ConfigPath::new("idle.root_minutes").unwrap());
    }

    #[test]
    fn from_segments_rejects_empty_input() {
        assert!(matches!(
            ConfigPath::from_segments(&[]),
            Err(Error::InvalidPath(_))
        ));
        assert!(matches!(
            ConfigPath::from_segments(&[""]),
            Err(Error::InvalidPath(_))
        ));
    }

    #[test]
    fn starts_with_prefix() {
        let parent = ConfigPath::new("idle").unwrap();
        let child = ConfigPath::new("idle.root_minutes").unwrap();
        let unrelated = ConfigPath::new("password.min_length").unwrap();
        assert!(child.starts_with(&parent));
        assert!(parent.starts_with(&parent));
        assert!(!parent.starts_with(&child));
        assert!(!child.starts_with(&unrelated));
    }

    #[test]
    fn value_type_tags() {
        assert_eq!(Value::Bool(true).type_tag(), "bool");
        assert_eq!(Value::U32(0).type_tag(), "u32");
        assert_eq!(Value::U64(0).type_tag(), "u64");
        assert_eq!(Value::String("x".to_string()).type_tag(), "string");
        assert_eq!(
            Value::PasswordClasses(PasswordClassSet::from_mask(0)).type_tag(),
            "password_classes"
        );
    }

    #[test]
    fn value_extractors_succeed_on_match() {
        assert_eq!(Value::Bool(true).as_bool(), Ok(true));
        assert_eq!(Value::U32(7).as_u32(), Ok(7));
        assert_eq!(Value::U64(7).as_u64(), Ok(7));
        assert_eq!(Value::String("hi".to_string()).as_str(), Ok("hi"));
    }

    #[test]
    fn value_extractors_fail_on_mismatch_with_diagnostic() {
        match Value::Bool(true).as_u32() {
            Err(Error::TypeMismatch { expected, got }) => {
                assert_eq!(expected, "u32");
                assert_eq!(got, "bool");
            }
            other => panic!("expected TypeMismatch, got {:?}", other),
        }
    }

    #[test]
    fn password_classes_count() {
        let p = PasswordClassSet::from_mask(
            PasswordClassSet::LOWER | PasswordClassSet::UPPER | PasswordClassSet::DIGIT,
        );
        assert_eq!(p.count(), 3);
        assert!(p.contains(PasswordClassSet::LOWER));
        assert!(!p.contains(PasswordClassSet::SYMBOL));
    }

    #[test]
    fn password_classes_drops_unknown_bits() {
        let p = PasswordClassSet::from_mask(0xFF);
        assert_eq!(p.0, PasswordClassSet::ALL);
    }

    #[test]
    fn surface_kind_round_trip() {
        for s in [SurfaceKind::Toml, SurfaceKind::Tty, SurfaceKind::Zenoh] {
            assert_eq!(SurfaceKind::from_name(s.name()), Some(s));
        }
    }

    #[test]
    fn surface_kind_rejects_unknown_name() {
        assert_eq!(SurfaceKind::from_name("stty"), None);
        assert_eq!(SurfaceKind::from_name(""), None);
        assert_eq!(SurfaceKind::from_name("Tty"), None); // case-sensitive
    }

    #[test]
    fn audit_identity_render() {
        let id = AuditIdentity {
            role: "root",
            user: None,
            surface: SurfaceKind::Tty,
        };
        assert_eq!(id.render(), "root@tty");

        let id = AuditIdentity {
            role: "operator",
            user: Some("alice".to_string()),
            surface: SurfaceKind::Zenoh,
        };
        assert_eq!(id.render(), "operator:alice@zenoh");
    }

    #[test]
    fn error_display_includes_diagnostic() {
        let e = Error::Validation("idle.root_minutes out of range");
        let s = alloc::format!("{}", e);
        assert!(s.contains("idle.root_minutes"));
    }
}
