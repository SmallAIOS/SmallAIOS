// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Typed [`AuditRecord`] and its JSON / JSONL projections.
//!
//! Spec:
//! `openspec/changes/management-login-v1/specs/mgmt-audit-log/spec.md`
//! Requirement "Audit record fields".
//!
//! Each record has the exact field set the spec calls for —
//! `{ ts, who, surface, action, before, after, code, prev_hash, hash }`.
//! `before` / `after` are typed [`JsonValue`]s so config writes can
//! capture the old/new value without rendering through the wider
//! [`crate::surface_zenoh::json`] codec at the audit boundary. `code`
//! is a signed 32-bit POSIX errno (negative on failure, `0` on
//! success).
//!
//! ## Hash domain
//!
//! `hash` is the SHA-3-256 digest of [`serialize_for_hash`] over every
//! field except `hash`. `prev_hash` is the immediately previous
//! record's `hash` (32 zero bytes for the first record). The chain
//! verifier replays the same serialiser to detect tampering.

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use super::chain::ChainHead;

// ─── Field-name constants (used by the JSON projection + tests) ──────────────

/// JSON key for the `before` field on a config-write record.
pub const AUDIT_FIELD_BEFORE: &str = "before";
/// JSON key for the `after` field on a config-write record.
pub const AUDIT_FIELD_AFTER: &str = "after";
/// JSON key carrying the SHA-3-256 hash of this record.
pub const AUDIT_FIELD_HASH: &str = "hash";

// ─── Surface ────────────────────────────────────────────────────────────────

/// Originating surface for an audit record. Mirrors
/// [`crate::surface::SurfaceKind`] plus a kernel-internal variant for
/// system-issued events (boot, idle sweep, signed checkpoints).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditSurface {
    /// Console TTY login surface.
    Tty,
    /// Zenoh admin transport.
    Zenoh,
    /// `/data/*.toml` loader/saver.
    Toml,
    /// Kernel-internal event (sweeper, signed checkpoint, boot
    /// success commit, etc.).
    Kernel,
}

impl AuditSurface {
    /// Lower-case canonical name used on the JSONL wire.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tty => "tty",
            Self::Zenoh => "zenoh",
            Self::Toml => "toml",
            Self::Kernel => "kernel",
        }
    }

    /// Parse the canonical name. Returns `None` on unknown input —
    /// the chain verifier uses this to flag malformed JSONL.
    pub fn from_name(s: &str) -> Option<Self> {
        match s {
            "tty" => Some(Self::Tty),
            "zenoh" => Some(Self::Zenoh),
            "toml" => Some(Self::Toml),
            "kernel" => Some(Self::Kernel),
            _ => None,
        }
    }
}

impl fmt::Display for AuditSurface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ─── Action ─────────────────────────────────────────────────────────────────

/// Canonical action verbs. The spec keeps the set open-ended (any
/// "verb" string is acceptable) so the typed enum carries the
/// well-known actions plus an [`AuditAction::Custom`] tail for
/// caller-extended verbs (e.g. denial syscall names, `model_load`
/// variants).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AuditAction {
    /// `auth_login` syscall.
    AuthLogin,
    /// `auth_logout` syscall.
    AuthLogout,
    /// Idle-sweeper auto-logout.
    AutoLogout,
    /// `auth_create_user` syscall.
    AuthCreateUser,
    /// `auth_change_password` syscall.
    AuthChangePassword,
    /// First-boot completion — Root password set, shadow finalised.
    FirstbootComplete,
    /// `audit_chain_committed` — emitted on every successful
    /// `Ring::append` and used by integration tests to confirm the
    /// chain head moved.
    AuditChainCommitted,
    /// Periodic / out-of-band signed checkpoint record.
    Checkpoint,
    /// Per-record `min_role` denial — `syscall` is the rejected
    /// syscall name (e.g. `system_power`). Wire form is `DENY:<name>`.
    Deny(String),
    /// Coalesced denial burst — the rate limiter coalesces excess
    /// denials beyond [`super::DENIAL_RATE_PER_SECOND`] into one
    /// record with `count = N`.
    DenyBurst {
        /// Number of denials coalesced.
        count: u32,
    },
    /// Config write — the `before` / `after` fields capture the typed
    /// old / new values.
    ConfigWrite {
        /// Dotted path that was written, e.g. `metrics.cpu_interval_ms`.
        path: String,
    },
    /// Model-load notification — emitted when ONNX runtime mmaps a
    /// new model. `name` is the model basename.
    ModelLoad {
        /// Model basename (matches `/models/<name>.onnx`).
        name: String,
    },
    /// Caller-supplied verb. The string SHALL be `[A-Za-z0-9_:]+`.
    Custom(String),
}

impl AuditAction {
    /// Render this action to its canonical wire form, e.g.
    /// `auth_login`, `DENY:system_power`, `DENY_BURST`.
    pub fn render(&self) -> String {
        match self {
            Self::AuthLogin => "auth_login".to_string(),
            Self::AuthLogout => "auth_logout".to_string(),
            Self::AutoLogout => "auto_logout".to_string(),
            Self::AuthCreateUser => "auth_create_user".to_string(),
            Self::AuthChangePassword => "auth_change_password".to_string(),
            Self::FirstbootComplete => "firstboot_complete".to_string(),
            Self::AuditChainCommitted => "audit_chain_committed".to_string(),
            Self::Checkpoint => "checkpoint".to_string(),
            Self::Deny(syscall) => {
                let mut s = String::with_capacity(5 + syscall.len());
                s.push_str("DENY:");
                s.push_str(syscall);
                s
            }
            Self::DenyBurst { .. } => "DENY_BURST".to_string(),
            Self::ConfigWrite { path: _ } => "config_write".to_string(),
            Self::ModelLoad { name: _ } => "model_load".to_string(),
            Self::Custom(s) => s.clone(),
        }
    }

    /// Returns `true` if this action is one of the **immediate-flush**
    /// auth events. The flusher uses this to skip the 1-second timer
    /// and write the record to disk before the originating syscall
    /// returns.
    pub fn is_auth_event(&self) -> bool {
        matches!(
            self,
            Self::AuthLogin
                | Self::AuthLogout
                | Self::AutoLogout
                | Self::AuthCreateUser
                | Self::AuthChangePassword
        )
    }

    /// Returns `true` if this action is a denial — used by the rate
    /// limiter to short-circuit before constructing the record.
    pub fn is_denial(&self) -> bool {
        matches!(self, Self::Deny(_) | Self::DenyBurst { .. })
    }
}

// ─── JSON value (audit-local minimal subset) ────────────────────────────────

/// Minimal `#![no_std]` JSON value for `before` / `after`. Mirrors
/// [`crate::surface_zenoh::json::JsonValue`] but lives here so the
/// audit subsystem doesn't drag in the full admin codec — config
/// writes typically capture scalar `before`/`after` and a small
/// number of strings; arrays and nested objects are supported for
/// completeness.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    /// JSON `null`.
    Null,
    /// JSON `true` / `false`.
    Bool(bool),
    /// Signed integer (i64, mirroring the admin codec's range).
    Int(i64),
    /// Unsigned integer (u64) — used by `audit.*_bytes` config fields.
    UInt(u64),
    /// String value.
    String(String),
    /// Array.
    Array(Vec<JsonValue>),
    /// Object — keys SHALL be unique, iteration order is sorted by
    /// the `BTreeMap`.
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    /// Construct a [`JsonValue::String`] without an explicit
    /// `.to_string()` at the call site.
    pub fn string(s: impl Into<String>) -> Self {
        Self::String(s.into())
    }

    /// Render to canonical form. Keys are emitted in sorted order
    /// (BTreeMap iteration), strings are double-quoted with the
    /// admin-codec's eight standard escapes, integers are decimal,
    /// `null` / `true` / `false` are lower-case literals.
    pub fn render(&self) -> String {
        let mut out = String::new();
        Self::render_into(self, &mut out);
        out
    }

    fn render_into(v: &JsonValue, out: &mut String) {
        match v {
            JsonValue::Null => out.push_str("null"),
            JsonValue::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            JsonValue::Int(n) => {
                let mut buf = [0u8; 24];
                let s = i64_to_str(*n, &mut buf);
                out.push_str(s);
            }
            JsonValue::UInt(n) => {
                let mut buf = [0u8; 24];
                let s = u64_to_str(*n, &mut buf);
                out.push_str(s);
            }
            JsonValue::String(s) => {
                out.push('"');
                escape_string_into(s, out);
                out.push('"');
            }
            JsonValue::Array(arr) => {
                out.push('[');
                for (i, item) in arr.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    Self::render_into(item, out);
                }
                out.push(']');
            }
            JsonValue::Object(obj) => {
                out.push('{');
                for (i, (k, vv)) in obj.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    out.push('"');
                    escape_string_into(k, out);
                    out.push_str("\":");
                    Self::render_into(vv, out);
                }
                out.push('}');
            }
        }
    }
}

fn escape_string_into(s: &str, out: &mut String) {
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                let n = c as u32;
                let mut buf = [0u8; 6];
                buf[0] = b'\\';
                buf[1] = b'u';
                buf[2] = b'0';
                buf[3] = b'0';
                buf[4] = HEX[((n >> 4) & 0xf) as usize];
                buf[5] = HEX[(n & 0xf) as usize];
                out.push_str(core::str::from_utf8(&buf).expect("ascii"));
            }
            c => out.push(c),
        }
    }
}

const HEX: &[u8; 16] = b"0123456789abcdef";

fn i64_to_str(n: i64, buf: &mut [u8; 24]) -> &str {
    if n == 0 {
        buf[0] = b'0';
        return core::str::from_utf8(&buf[..1]).expect("ascii digits");
    }
    let neg = n < 0;
    // Avoid `n.unsigned_abs()` to keep `#![no_std]` compat across MSRV.
    let mut x: u64 = if neg {
        (!(n as u64)).wrapping_add(1)
    } else {
        n as u64
    };
    let mut len = 0usize;
    while x > 0 {
        buf[len] = b'0' + (x % 10) as u8;
        x /= 10;
        len += 1;
    }
    if neg {
        buf[len] = b'-';
        len += 1;
    }
    buf[..len].reverse();
    core::str::from_utf8(&buf[..len]).expect("ascii digits")
}

fn u64_to_str(mut n: u64, buf: &mut [u8; 24]) -> &str {
    if n == 0 {
        buf[0] = b'0';
        return core::str::from_utf8(&buf[..1]).expect("ascii digits");
    }
    let mut len = 0usize;
    while n > 0 {
        buf[len] = b'0' + (n % 10) as u8;
        n /= 10;
        len += 1;
    }
    buf[..len].reverse();
    core::str::from_utf8(&buf[..len]).expect("ascii digits")
}

// ─── AuditRecord ────────────────────────────────────────────────────────────

/// One immutable entry in the audit chain.
#[derive(Debug, Clone, PartialEq)]
pub struct AuditRecord {
    /// UNIX timestamp in nanoseconds. `Ring::append` reads from a
    /// caller-supplied clock, so timestamp monotonicity is the
    /// caller's responsibility.
    pub ts_ns: u64,
    /// Identity rendering — `root@tty`, `op:alice@zenoh`, `kernel`,
    /// etc. Free-form string; the chain verifier compares verbatim.
    pub who: String,
    /// Originating surface.
    pub surface: AuditSurface,
    /// Action verb.
    pub action: AuditAction,
    /// Optional `before` value for config writes; `JsonValue::Null`
    /// for non-write actions. Captured before validation.
    pub before: JsonValue,
    /// Optional `after` value for config writes; `JsonValue::Null`
    /// for non-write actions.
    pub after: JsonValue,
    /// POSIX errno result code — `0` on success, negative on failure.
    pub code: i32,
    /// SHA-3-256 hash of the immediately previous record (32 zero
    /// bytes for the first record).
    pub prev_hash: ChainHead,
    /// SHA-3-256 hash over `serialize_for_hash`.
    pub hash: ChainHead,
    /// ML-DSA-65 signature bytes — only populated on
    /// [`AuditAction::Checkpoint`] records when signed checkpoints
    /// are enabled. Empty for every other record.
    pub signature: Vec<u8>,
}

impl AuditRecord {
    /// Render the canonical JSONL line (no trailing newline).
    /// `BTreeMap` iteration keeps the field order stable so the chain
    /// verifier can replay byte-identically.
    pub fn to_jsonl(&self) -> String {
        let mut obj: BTreeMap<String, JsonValue> = BTreeMap::new();
        obj.insert("ts".to_string(), JsonValue::UInt(self.ts_ns));
        obj.insert("who".to_string(), JsonValue::string(self.who.clone()));
        obj.insert(
            "surface".to_string(),
            JsonValue::string(self.surface.as_str()),
        );
        obj.insert(
            "action".to_string(),
            JsonValue::string(self.action.render()),
        );
        // DENY_BURST records carry their `count` alongside; emit it
        // in a way the verifier can reconstruct verbatim.
        if let AuditAction::DenyBurst { count } = &self.action {
            obj.insert("count".to_string(), JsonValue::UInt(*count as u64));
        }
        if let AuditAction::ConfigWrite { path } = &self.action {
            obj.insert("path".to_string(), JsonValue::string(path.clone()));
        }
        if let AuditAction::ModelLoad { name } = &self.action {
            obj.insert("model".to_string(), JsonValue::string(name.clone()));
        }
        obj.insert(AUDIT_FIELD_BEFORE.to_string(), self.before.clone());
        obj.insert(AUDIT_FIELD_AFTER.to_string(), self.after.clone());
        obj.insert("code".to_string(), JsonValue::Int(self.code as i64));
        obj.insert(
            "prev_hash".to_string(),
            JsonValue::string(fingerprint_hex(&self.prev_hash)),
        );
        obj.insert(
            AUDIT_FIELD_HASH.to_string(),
            JsonValue::string(fingerprint_hex(&self.hash)),
        );
        if !self.signature.is_empty() {
            obj.insert(
                "signature".to_string(),
                JsonValue::string(hex_lower(&self.signature)),
            );
        }
        JsonValue::Object(obj).render()
    }
}

/// Lowercase hex encoding of a 32-byte [`ChainHead`]. Used by the
/// JSONL projection and the `audit_fingerprint` telemetry payload.
pub fn fingerprint_hex(head: &ChainHead) -> String {
    hex_lower(head)
}

/// Lowercase hex encoding for arbitrary-length byte slices (signed
/// checkpoint signatures are 3 309 bytes; we still hex-encode to keep
/// the JSONL line ASCII-safe).
pub fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(HEX[((b >> 4) & 0xf) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::audit_test_vectors as fixtures;
    use super::*;

    #[test]
    fn surface_round_trip() {
        for s in [
            AuditSurface::Tty,
            AuditSurface::Zenoh,
            AuditSurface::Toml,
            AuditSurface::Kernel,
        ] {
            assert_eq!(AuditSurface::from_name(s.as_str()), Some(s));
        }
    }

    #[test]
    fn surface_unknown_rejects() {
        assert_eq!(AuditSurface::from_name("smoke"), None);
        assert_eq!(AuditSurface::from_name(""), None);
        assert_eq!(AuditSurface::from_name("Tty"), None);
    }

    #[test]
    fn action_render_matches_spec_form() {
        assert_eq!(AuditAction::AuthLogin.render(), "auth_login");
        assert_eq!(AuditAction::AuthLogout.render(), "auth_logout");
        assert_eq!(AuditAction::AutoLogout.render(), "auto_logout");
        assert_eq!(
            AuditAction::FirstbootComplete.render(),
            "firstboot_complete"
        );
        assert_eq!(
            AuditAction::Deny("system_power".to_string()).render(),
            "DENY:system_power"
        );
        assert_eq!(AuditAction::DenyBurst { count: 1 }.render(), "DENY_BURST");
    }

    #[test]
    fn auth_event_classification() {
        assert!(AuditAction::AuthLogin.is_auth_event());
        assert!(AuditAction::AuthLogout.is_auth_event());
        assert!(AuditAction::AutoLogout.is_auth_event());
        assert!(AuditAction::AuthCreateUser.is_auth_event());
        assert!(AuditAction::AuthChangePassword.is_auth_event());
        assert!(!AuditAction::AuditChainCommitted.is_auth_event());
        assert!(!AuditAction::Checkpoint.is_auth_event());
        assert!(!AuditAction::Deny("x".to_string()).is_auth_event());
    }

    #[test]
    fn denial_classification() {
        assert!(AuditAction::Deny("x".to_string()).is_denial());
        assert!(AuditAction::DenyBurst { count: 1 }.is_denial());
        assert!(!AuditAction::AuthLogin.is_denial());
    }

    #[test]
    fn jsonvalue_render_scalars() {
        assert_eq!(JsonValue::Null.render(), "null");
        assert_eq!(JsonValue::Bool(true).render(), "true");
        assert_eq!(JsonValue::Bool(false).render(), "false");
        assert_eq!(JsonValue::Int(0).render(), "0");
        assert_eq!(JsonValue::Int(-1).render(), "-1");
        assert_eq!(JsonValue::Int(i64::MIN).render(), "-9223372036854775808");
        assert_eq!(JsonValue::UInt(0).render(), "0");
        assert_eq!(JsonValue::UInt(u64::MAX).render(), "18446744073709551615");
    }

    #[test]
    fn jsonvalue_render_string_escapes() {
        assert_eq!(JsonValue::string("abc").render(), "\"abc\"");
        assert_eq!(JsonValue::string("a\"b\\c").render(), "\"a\\\"b\\\\c\"");
        assert_eq!(JsonValue::string("a\nb").render(), "\"a\\nb\"");
        assert_eq!(JsonValue::string("a\tb").render(), "\"a\\tb\"");
        // Control character below 0x20 falls back to \u00XX.
        assert_eq!(JsonValue::string("a\x01b").render(), "\"a\\u0001b\"");
    }

    #[test]
    fn jsonvalue_render_array_object() {
        let arr = JsonValue::Array(alloc::vec![JsonValue::Int(1), JsonValue::Int(2)]);
        assert_eq!(arr.render(), "[1,2]");
        let mut obj = BTreeMap::new();
        obj.insert("b".to_string(), JsonValue::Int(2));
        obj.insert("a".to_string(), JsonValue::Int(1));
        assert_eq!(JsonValue::Object(obj).render(), "{\"a\":1,\"b\":2}");
    }

    #[test]
    fn fingerprint_hex_zero_chain() {
        let h: ChainHead = [0u8; 32];
        assert_eq!(fingerprint_hex(&h), "0".repeat(64));
    }

    #[test]
    fn fingerprint_hex_full_byte_pattern() {
        let mut h: ChainHead = [0u8; 32];
        for (i, b) in h.iter_mut().enumerate() {
            *b = i as u8;
        }
        let hex = fingerprint_hex(&h);
        assert!(hex.starts_with("000102030405"));
        assert_eq!(hex.len(), 64);
    }

    #[test]
    fn record_to_jsonl_sorts_keys() {
        let r = fixtures::sample_login_record();
        let line = r.to_jsonl();
        // Key order is BTreeMap-sorted: action,after,before,code,hash,
        // prev_hash,surface,ts,who.
        let action_pos = line.find("\"action\"").unwrap();
        let ts_pos = line.find("\"ts\"").unwrap();
        let who_pos = line.find("\"who\"").unwrap();
        assert!(action_pos < ts_pos);
        assert!(ts_pos < who_pos);
    }

    #[test]
    fn record_to_jsonl_includes_required_fields() {
        let r = fixtures::sample_login_record();
        let line = r.to_jsonl();
        for required in [
            "\"ts\"",
            "\"who\"",
            "\"surface\"",
            "\"action\"",
            "\"before\"",
            "\"after\"",
            "\"code\"",
            "\"prev_hash\"",
            "\"hash\"",
        ] {
            assert!(line.contains(required), "missing field {}", required);
        }
    }

    #[test]
    fn record_to_jsonl_emits_count_for_burst() {
        let r = fixtures::sample_burst_record(190);
        let line = r.to_jsonl();
        assert!(line.contains("\"count\":190"));
        assert!(line.contains("\"DENY_BURST\""));
    }

    #[test]
    fn record_to_jsonl_emits_path_for_config_write() {
        let r = fixtures::sample_config_write_record();
        let line = r.to_jsonl();
        assert!(line.contains("\"path\":\"metrics.cpu_interval_ms\""));
        assert!(line.contains("\"action\":\"config_write\""));
    }

    #[test]
    fn record_to_jsonl_includes_signature_only_when_present() {
        let r = fixtures::sample_login_record();
        assert!(!r.to_jsonl().contains("signature"));
        let r2 = fixtures::sample_checkpoint_record();
        assert!(r2.to_jsonl().contains("\"signature\":"));
    }

    #[test]
    fn hex_lower_round_trip() {
        let bytes = [0x00u8, 0x0fu8, 0xf0u8, 0xffu8];
        assert_eq!(hex_lower(&bytes), "000ff0ff");
    }
}
