// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Shadow file format — parser, serializer, file-mode check.
//!
//! Spec: `openspec/changes/management-login-v1/specs/auth-shadow/spec.md`
//!
//! ## Canonical record format
//!
//! ```text
//! username:$argon2id$v=19$m=<m>,t=<t>,p=<p>$<base64-salt>$<base64-tag>:role=<root|operator|viewer>:flags=<u32>:last_changed=<unix-day>:totp_secret=<base32-or-empty>:lockout_until=<unix-or-zero>
//! ```
//!
//! ## Forward-compatibility rule
//!
//! Parsers SHALL ignore unknown trailing fields. Any additional `key=value`
//! field after `lockout_until=` is preserved verbatim in
//! [`ShadowEntry::trailing_unknown_fields`] and round-tripped on rewrite,
//! so a future feature flag (e.g. `mfa_kind=fido2`) does not break older
//! readers.
//!
//! ## File-mode defense in depth
//!
//! [`verify_file_permissions`] rejects any mode laxer than 0600. The
//! caller (kernel boot path) supplies the mode bits read from the
//! filesystem; this module performs the policy check so the boot logic
//! can stay independent of the `fs/` crate's metadata API.

use crate::role::{Role, RoleError};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

// ─── Bit flags ────────────────────────────────────────────────────────────────

/// Bit 0 of [`ShadowEntry::flags`] — when set, the kernel SHALL only honor
/// `auth_change_password`, `auth_whoami`, and `auth_logout` until the user
/// rotates the password. See `auth-roles` spec "must-change-password
/// gating".
pub const FLAG_MUST_CHANGE_PASSWORD: u32 = 0x0000_0001;

// ─── Errors ───────────────────────────────────────────────────────────────────

/// Errors raised by shadow-file parsing/serialization or permission checks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShadowError {
    /// Record had fewer mandatory colon-separated segments than required.
    Truncated,
    /// PHC string did not start with `$argon2id$` or otherwise failed
    /// shape validation.
    MalformedPhc,
    /// `role=…` field carried an unknown role string.
    UnknownRole,
    /// A mandatory key=value field was missing or its key was wrong.
    MissingField(&'static str),
    /// A numeric field (`flags`, `last_changed`, `lockout_until`) failed
    /// to parse as the expected integer type.
    InvalidInteger(&'static str),
    /// A duplicate `key=` was seen in the trailing-fields section.
    DuplicateField(&'static str),
    /// Username was empty.
    EmptyUsername,
    /// Username contained a forbidden character (`:`, NUL, or `\n`).
    InvalidUsername,
    /// File mode was laxer than 0600 — the loader SHALL refuse to read it.
    PermissionTooLax(u32),
    /// File contained a record whose line was empty (other than trailing
    /// blank lines).
    EmptyRecord,
}

impl fmt::Display for ShadowError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Truncated => write!(f, "auth/shadow: truncated record"),
            Self::MalformedPhc => write!(f, "auth/shadow: malformed PHC string"),
            Self::UnknownRole => write!(f, "auth/shadow: unknown role name"),
            Self::MissingField(k) => write!(f, "auth/shadow: missing field `{k}`"),
            Self::InvalidInteger(k) => write!(f, "auth/shadow: invalid integer for `{k}`"),
            Self::DuplicateField(k) => write!(f, "auth/shadow: duplicate field `{k}`"),
            Self::EmptyUsername => write!(f, "auth/shadow: empty username"),
            Self::InvalidUsername => write!(f, "auth/shadow: invalid character in username"),
            Self::PermissionTooLax(m) => {
                write!(f, "auth/shadow: mode 0o{m:o} laxer than 0600")
            }
            Self::EmptyRecord => write!(f, "auth/shadow: empty record"),
        }
    }
}

impl From<RoleError> for ShadowError {
    fn from(_: RoleError) -> Self {
        ShadowError::UnknownRole
    }
}

// ─── Record ───────────────────────────────────────────────────────────────────

/// A single parsed shadow-file record.
///
/// Round-trip invariant: after `serialize_record(parse_record(line))?` the
/// resulting string SHALL equal `line` exactly, including any unknown
/// trailing fields. This is verified by tests in this module and is
/// required by the "Round-trip parse and serialize" spec scenario.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShadowEntry {
    /// Username — printable, no `:` / `\n` / NUL.
    pub username: String,
    /// PHC-encoded Argon2id hash — the entire `$argon2id$…` string.
    pub phc: String,
    /// Role for capability gating.
    pub role: Role,
    /// Bit-flag set: bit 0 = `must_change_password_on_login`, others
    /// reserved.
    pub flags: u32,
    /// Day on which the password was last changed (Unix days since epoch,
    /// `floor(unix_seconds / 86_400)`).
    pub last_changed_unix_day: u32,
    /// Optional TOTP shared secret. `None` means TOTP not enrolled. The
    /// field stores the *decoded* secret bytes; the on-disk format is
    /// RFC 4648 base32 without padding (or empty).
    pub totp_secret: Option<Vec<u8>>,
    /// Lockout deadline as Unix seconds (`0` ⇒ not locked out).
    pub lockout_until_unix: u64,
    /// Forward-compat: any `key=value` field appearing after
    /// `lockout_until=` is preserved verbatim and rewritten on
    /// serialization. Each entry is the raw `key=value` string (no
    /// surrounding colons).
    pub trailing_unknown_fields: Vec<String>,
}

// ─── Permission check ─────────────────────────────────────────────────────────

/// Refuse to read a shadow file whose mode is laxer than 0600.
///
/// Per spec "Shadow file permission defense": only the owner-read and
/// owner-write bits MAY be set. Any group or world bits, or non-owner
/// special bits, SHALL trigger [`ShadowError::PermissionTooLax`].
///
/// `mode` is the POSIX mode field as seen by `fstat(2)`'s `st_mode &
/// 0o777`. Callers SHALL strip the file-type nibble before passing.
pub fn verify_file_permissions(mode: u32) -> Result<(), ShadowError> {
    // Reject any group or world bits. Bits 077 cover g_rwx + o_rwx.
    if mode & 0o077 != 0 {
        return Err(ShadowError::PermissionTooLax(mode));
    }
    Ok(())
}

// ─── Parsing ──────────────────────────────────────────────────────────────────

/// Parse a single shadow-file record (one line, no trailing newline).
pub fn parse_record(line: &str) -> Result<ShadowEntry, ShadowError> {
    if line.is_empty() {
        return Err(ShadowError::EmptyRecord);
    }

    // Layout:
    //   username : phc : role=… : flags=… : last_changed=… : totp_secret=… : lockout_until=… [: extra]*
    //
    // The PHC string itself has *internal* `$`s but no `:`s, so `:` is a
    // safe separator.
    let mut parts = line.split(':');

    let username = parts.next().ok_or(ShadowError::Truncated)?;
    validate_username(username)?;

    let phc = parts.next().ok_or(ShadowError::Truncated)?;
    if !phc.starts_with("$argon2id$") {
        return Err(ShadowError::MalformedPhc);
    }
    // Sanity-check the canonical PHC shape: 5 `$` segments after the
    // leading `$`. We don't run the argon2id verifier here — that is
    // applied at login time. We just require the string be parseable.
    let phc_dollar_segments = phc.matches('$').count();
    if phc_dollar_segments != 5 {
        return Err(ShadowError::MalformedPhc);
    }

    // Mandatory key=value fields, in canonical order. We accept any order
    // for forward-compat readers, but enforce all four are present.
    let mut role: Option<Role> = None;
    let mut flags: Option<u32> = None;
    let mut last_changed: Option<u32> = None;
    let mut totp_secret: Option<Option<Vec<u8>>> = None;
    let mut lockout_until: Option<u64> = None;
    let mut trailing: Vec<String> = Vec::new();

    for kv in parts {
        let (k, v) = split_key_value(kv)?;
        match k {
            "role" => {
                if role.is_some() {
                    return Err(ShadowError::DuplicateField("role"));
                }
                role = Some(Role::from_name(v)?);
            }
            "flags" => {
                if flags.is_some() {
                    return Err(ShadowError::DuplicateField("flags"));
                }
                flags = Some(parse_u32(v).ok_or(ShadowError::InvalidInteger("flags"))?);
            }
            "last_changed" => {
                if last_changed.is_some() {
                    return Err(ShadowError::DuplicateField("last_changed"));
                }
                last_changed =
                    Some(parse_u32(v).ok_or(ShadowError::InvalidInteger("last_changed"))?);
            }
            "totp_secret" => {
                if totp_secret.is_some() {
                    return Err(ShadowError::DuplicateField("totp_secret"));
                }
                totp_secret = Some(if v.is_empty() {
                    None
                } else {
                    Some(base32_decode(v)?)
                });
            }
            "lockout_until" => {
                if lockout_until.is_some() {
                    return Err(ShadowError::DuplicateField("lockout_until"));
                }
                lockout_until =
                    Some(parse_u64(v).ok_or(ShadowError::InvalidInteger("lockout_until"))?);
            }
            _ => {
                // Forward-compat: preserve verbatim.
                trailing.push(kv.to_string());
            }
        }
    }

    Ok(ShadowEntry {
        username: username.to_string(),
        phc: phc.to_string(),
        role: role.ok_or(ShadowError::MissingField("role"))?,
        flags: flags.ok_or(ShadowError::MissingField("flags"))?,
        last_changed_unix_day: last_changed.ok_or(ShadowError::MissingField("last_changed"))?,
        totp_secret: totp_secret.ok_or(ShadowError::MissingField("totp_secret"))?,
        lockout_until_unix: lockout_until.ok_or(ShadowError::MissingField("lockout_until"))?,
        trailing_unknown_fields: trailing,
    })
}

/// Parse an entire shadow file into a vector of entries. Empty lines are
/// silently skipped (so a trailing newline is harmless).
pub fn parse_file(content: &str) -> Result<Vec<ShadowEntry>, ShadowError> {
    let mut out = Vec::new();
    for line in content.split('\n') {
        if line.is_empty() {
            continue;
        }
        out.push(parse_record(line)?);
    }
    Ok(out)
}

// ─── Serialization ────────────────────────────────────────────────────────────

/// Serialize a single record back to the canonical line form. The trailing
/// `\n` is not included.
pub fn serialize_record(entry: &ShadowEntry) -> String {
    let mut out = String::new();
    out.push_str(&entry.username);
    out.push(':');
    out.push_str(&entry.phc);
    out.push_str(":role=");
    out.push_str(entry.role.name());
    out.push_str(":flags=");
    push_u32(&mut out, entry.flags);
    out.push_str(":last_changed=");
    push_u32(&mut out, entry.last_changed_unix_day);
    out.push_str(":totp_secret=");
    if let Some(secret) = entry.totp_secret.as_ref() {
        push_base32(&mut out, secret);
    }
    out.push_str(":lockout_until=");
    push_u64(&mut out, entry.lockout_until_unix);
    for extra in &entry.trailing_unknown_fields {
        out.push(':');
        out.push_str(extra);
    }
    out
}

/// Serialize a slice of entries into a shadow-file body. A trailing
/// newline is appended so the file is terminated per Unix convention.
pub fn serialize_file(entries: &[ShadowEntry]) -> String {
    let mut out = String::new();
    for entry in entries {
        out.push_str(&serialize_record(entry));
        out.push('\n');
    }
    out
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

fn validate_username(u: &str) -> Result<(), ShadowError> {
    if u.is_empty() {
        return Err(ShadowError::EmptyUsername);
    }
    for b in u.bytes() {
        if b == b':' || b == 0 || b == b'\n' {
            return Err(ShadowError::InvalidUsername);
        }
    }
    Ok(())
}

fn split_key_value(kv: &str) -> Result<(&str, &str), ShadowError> {
    let idx = kv.find('=').ok_or(ShadowError::Truncated)?;
    let (k, rest) = kv.split_at(idx);
    if k.is_empty() {
        return Err(ShadowError::Truncated);
    }
    Ok((k, &rest[1..]))
}

fn parse_u32(s: &str) -> Option<u32> {
    if s.is_empty() {
        return None;
    }
    let mut acc: u32 = 0;
    for b in s.bytes() {
        if !b.is_ascii_digit() {
            return None;
        }
        acc = acc.checked_mul(10)?.checked_add((b - b'0') as u32)?;
    }
    Some(acc)
}

fn parse_u64(s: &str) -> Option<u64> {
    if s.is_empty() {
        return None;
    }
    let mut acc: u64 = 0;
    for b in s.bytes() {
        if !b.is_ascii_digit() {
            return None;
        }
        acc = acc.checked_mul(10)?.checked_add((b - b'0') as u64)?;
    }
    Some(acc)
}

fn push_u32(out: &mut String, n: u32) {
    if n == 0 {
        out.push('0');
        return;
    }
    let mut buf = [0u8; 10];
    let mut i = buf.len();
    let mut m = n;
    while m > 0 {
        i -= 1;
        buf[i] = b'0' + (m % 10) as u8;
        m /= 10;
    }
    // SAFETY: only ASCII digits written.
    out.push_str(core::str::from_utf8(&buf[i..]).unwrap());
}

fn push_u64(out: &mut String, n: u64) {
    if n == 0 {
        out.push('0');
        return;
    }
    let mut buf = [0u8; 20];
    let mut i = buf.len();
    let mut m = n;
    while m > 0 {
        i -= 1;
        buf[i] = b'0' + (m % 10) as u8;
        m /= 10;
    }
    out.push_str(core::str::from_utf8(&buf[i..]).unwrap());
}

// ─── Base32 (RFC 4648, no padding) ────────────────────────────────────────────

// RFC 4648 §6 base32 alphabet. The 32 letters A..=Z plus digits 2..=7,
// in canonical order. Used both for the on-disk TOTP secret encoding and
// for round-tripping; this is a public-facing alphabet, not a secret.
//
// lgtm[rust/hard-coded-cryptographic-value] — RFC 4648 §6 base32 alphabet, not a secret
const BASE32_ALPHABET: &[u8; 32] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567";

fn base32_decode(s: &str) -> Result<Vec<u8>, ShadowError> {
    // Reject padding `=` to keep the field length canonical.
    if s.contains('=') {
        return Err(ShadowError::InvalidInteger("totp_secret"));
    }
    let mut out: Vec<u8> = Vec::with_capacity(s.len() * 5 / 8);
    let mut buffer: u32 = 0;
    let mut bits: u32 = 0;
    for ch in s.bytes() {
        let v: u32 = match ch {
            b'A'..=b'Z' => (ch - b'A') as u32,
            b'a'..=b'z' => (ch - b'a') as u32,
            b'2'..=b'7' => (ch - b'2' + 26) as u32,
            _ => return Err(ShadowError::InvalidInteger("totp_secret")),
        };
        buffer = (buffer << 5) | v;
        bits += 5;
        if bits >= 8 {
            bits -= 8;
            out.push(((buffer >> bits) & 0xFF) as u8);
        }
    }
    // Any leftover `bits` are zero padding from a non-multiple-of-8 secret
    // length, which is allowed by RFC 4648 §6 when no `=` padding is used.
    Ok(out)
}

fn push_base32(out: &mut String, bytes: &[u8]) {
    let mut buffer: u32 = 0;
    let mut bits: u32 = 0;
    for &b in bytes {
        buffer = (buffer << 8) | (b as u32);
        bits += 8;
        while bits >= 5 {
            bits -= 5;
            let idx = ((buffer >> bits) & 0x1F) as usize;
            out.push(BASE32_ALPHABET[idx] as char);
        }
    }
    if bits > 0 {
        let idx = ((buffer << (5 - bits)) & 0x1F) as usize;
        out.push(BASE32_ALPHABET[idx] as char);
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic, fixture-only PHC string. The salt and tag bytes here are
    // *not* secret — they exist solely as a parser oracle.
    //
    // lgtm[rust/hard-coded-cryptographic-value] — synthetic test fixture, not a real credential
    const SAMPLE_PHC: &str =
        "$argon2id$v=19$m=8192,t=3,p=1$YWFhYWFhYWFhYWFhYWFhYQ$YWJjZGVmZ2hpamtsbW5vcA";

    fn sample_entry() -> ShadowEntry {
        ShadowEntry {
            username: "root".to_string(),
            phc: SAMPLE_PHC.to_string(),
            role: Role::Root,
            flags: FLAG_MUST_CHANGE_PASSWORD,
            last_changed_unix_day: 19_852,
            totp_secret: None,
            lockout_until_unix: 0,
            trailing_unknown_fields: Vec::new(),
        }
    }

    #[test]
    fn round_trip_basic_record() {
        let entry = sample_entry();
        let line = serialize_record(&entry);
        let parsed = parse_record(&line).unwrap();
        assert_eq!(parsed, entry);
    }

    #[test]
    fn round_trip_with_totp_secret() {
        // Synthetic 20-byte secret (RFC 6238 typical length). Not used to
        // exfiltrate any real credential — purely a round-trip oracle.
        //
        // lgtm[rust/hard-coded-cryptographic-value] — synthetic test fixture, not a real credential
        let secret: Vec<u8> = (0u8..20).collect();
        let mut entry = sample_entry();
        entry.totp_secret = Some(secret.clone());
        let line = serialize_record(&entry);
        let parsed = parse_record(&line).unwrap();
        assert_eq!(parsed.totp_secret, Some(secret));
        assert_eq!(parsed, entry);
    }

    #[test]
    fn forward_compatible_unknown_trailing_field_preserved() {
        // Spec scenario: "Forward-compatible parse"
        let entry = sample_entry();
        let mut line = serialize_record(&entry);
        line.push_str(":mfa_kind=fido2");
        line.push_str(":hardened=true");

        let parsed = parse_record(&line).unwrap();
        assert_eq!(
            parsed.trailing_unknown_fields,
            ["mfa_kind=fido2", "hardened=true"]
        );

        // Round-trip preserves them verbatim.
        let reserialized = serialize_record(&parsed);
        assert_eq!(reserialized, line);
    }

    #[test]
    fn reject_malformed_phc_missing_prefix() {
        // Spec scenario: "Reject malformed PHC string"
        let bad = "root:$bcrypt$abc:role=root:flags=0:last_changed=0:totp_secret=:lockout_until=0";
        assert_eq!(parse_record(bad), Err(ShadowError::MalformedPhc));
    }

    #[test]
    fn reject_malformed_phc_truncated() {
        let bad =
            "root:$argon2id$abc:role=root:flags=0:last_changed=0:totp_secret=:lockout_until=0";
        assert_eq!(parse_record(bad), Err(ShadowError::MalformedPhc));
    }

    #[test]
    fn reject_unknown_role() {
        let line = alloc::format!(
            "root:{}:role=admin:flags=0:last_changed=0:totp_secret=:lockout_until=0",
            SAMPLE_PHC
        );
        assert_eq!(parse_record(&line), Err(ShadowError::UnknownRole));
    }

    #[test]
    fn reject_truncated_record() {
        // Only the username, no PHC.
        assert_eq!(parse_record("root"), Err(ShadowError::Truncated));
    }

    #[test]
    fn reject_missing_required_field_lockout_until() {
        let line = alloc::format!(
            "root:{}:role=root:flags=0:last_changed=0:totp_secret=",
            SAMPLE_PHC
        );
        assert_eq!(
            parse_record(&line),
            Err(ShadowError::MissingField("lockout_until"))
        );
    }

    #[test]
    fn reject_invalid_integer() {
        let line = alloc::format!(
            "root:{}:role=root:flags=notanumber:last_changed=0:totp_secret=:lockout_until=0",
            SAMPLE_PHC
        );
        assert_eq!(
            parse_record(&line),
            Err(ShadowError::InvalidInteger("flags"))
        );
    }

    #[test]
    fn reject_empty_username() {
        let line = alloc::format!(
            ":{}:role=root:flags=0:last_changed=0:totp_secret=:lockout_until=0",
            SAMPLE_PHC
        );
        assert_eq!(parse_record(&line), Err(ShadowError::EmptyUsername));
    }

    #[test]
    fn reject_duplicate_field() {
        let line = alloc::format!(
            "root:{}:role=root:flags=0:flags=1:last_changed=0:totp_secret=:lockout_until=0",
            SAMPLE_PHC
        );
        assert_eq!(
            parse_record(&line),
            Err(ShadowError::DuplicateField("flags"))
        );
    }

    #[test]
    fn parse_file_handles_multiple_records_and_blank_lines() {
        let entry = sample_entry();
        let mut file = serialize_record(&entry);
        file.push('\n');
        file.push('\n'); // blank line in the middle
        let mut other = entry.clone();
        other.username = "operator-1".to_string();
        other.role = Role::Operator;
        file.push_str(&serialize_record(&other));
        file.push('\n');

        let parsed = parse_file(&file).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].username, "root");
        assert_eq!(parsed[1].username, "operator-1");
        assert_eq!(parsed[1].role, Role::Operator);
    }

    #[test]
    fn serialize_file_round_trips() {
        let entry_a = sample_entry();
        let mut entry_b = entry_a.clone();
        entry_b.username = "viewer-bob".to_string();
        entry_b.role = Role::Viewer;
        entry_b.flags = 0;
        entry_b.lockout_until_unix = 1_700_000_000;

        let body = serialize_file(&[entry_a.clone(), entry_b.clone()]);
        let parsed = parse_file(&body).unwrap();
        assert_eq!(parsed, alloc::vec![entry_a, entry_b]);
    }

    #[test]
    fn verify_file_permissions_accepts_0600() {
        // Spec scenario: "Mode 0600 shadow accepted"
        assert!(verify_file_permissions(0o600).is_ok());
        // Read-only is also fine — the loader doesn't write.
        assert!(verify_file_permissions(0o400).is_ok());
    }

    #[test]
    fn verify_file_permissions_rejects_0644() {
        // Spec scenario: "Mode 0644 shadow rejected"
        assert_eq!(
            verify_file_permissions(0o644),
            Err(ShadowError::PermissionTooLax(0o644))
        );
    }

    #[test]
    fn verify_file_permissions_rejects_any_group_or_world_bit() {
        for laxer in [0o604, 0o620, 0o640, 0o660, 0o666, 0o777] {
            assert_eq!(
                verify_file_permissions(laxer),
                Err(ShadowError::PermissionTooLax(laxer)),
                "mode 0o{laxer:o} must be rejected"
            );
        }
    }

    #[test]
    fn invalid_username_rejected() {
        let line = alloc::format!(
            "ro\not:{}:role=root:flags=0:last_changed=0:totp_secret=:lockout_until=0",
            SAMPLE_PHC
        );
        assert_eq!(parse_record(&line), Err(ShadowError::InvalidUsername));
    }

    #[test]
    fn flag_bit_zero_is_must_change() {
        // Cross-check the named constant against the spec wording.
        assert_eq!(FLAG_MUST_CHANGE_PASSWORD, 0x0000_0001);
    }

    #[test]
    fn base32_round_trip_smoke() {
        // RFC 4648 §10 test vectors, base32 form (no padding).
        let cases: &[(&[u8], &str)] = &[
            (b"", ""),
            (b"f", "MY"),
            (b"fo", "MZXQ"),
            (b"foo", "MZXW6"),
            (b"foob", "MZXW6YQ"),
            (b"fooba", "MZXW6YTB"),
            (b"foobar", "MZXW6YTBOI"),
        ];
        for (raw, encoded) in cases {
            let mut got = String::new();
            push_base32(&mut got, raw);
            assert_eq!(&got, encoded, "encode failed for {raw:?}");
            assert_eq!(
                base32_decode(encoded).unwrap(),
                raw.to_vec(),
                "decode failed for {encoded}"
            );
        }
    }

    #[test]
    fn base32_decode_rejects_padding_and_garbage() {
        assert!(base32_decode("MY=").is_err());
        assert!(base32_decode("M!Y").is_err());
        assert!(base32_decode("0189").is_err()); // '0' / '1' not in alphabet
    }

    #[test]
    fn empty_record_rejected() {
        assert_eq!(parse_record(""), Err(ShadowError::EmptyRecord));
    }
}
