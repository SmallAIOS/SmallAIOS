// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SHA-3-256 fingerprint sidecar + optional ML-DSA-65 signature sidecar.
//!
//! Spec: `openspec/changes/embedded-overlay-v1/specs/fs-overlay-integrity/spec.md`.
//!
//! Each overlay-upper model file `<name>` is accompanied by:
//!
//! - `<name>.sha3` — required. Hex-encoded SHA-3-256 of the file's bytes,
//!   produced atomically alongside the model on every successful
//!   `model_add`. Reads from `/models/` hash-verify the upper-layer
//!   bytes against this sidecar before the bytes flow into ONNX
//!   runtime; mismatch fails closed with `-EIO`.
//! - `<name>.sig` — optional. ML-DSA-65 signature over the SHA-3-256
//!   fingerprint. Required when `mgmt::Config::overlay.require_signed`
//!   is `true`; optional otherwise. Missing/invalid → `-EAUTH`.
//!
//! The sidecar suffixes (`.sha3`, `.sig`) are reserved per
//! [`super::reserved`]; operator-supplied names ending in those
//! suffixes are rejected at `model_add` with `-EINVAL`.
//!
//! The hash check is **fail-closed**: any I/O failure reading the
//! sidecar, any sidecar parse failure, and any mismatch all surface as
//! [`IntegrityError::HashMismatch`]/[`IntegrityError::SidecarMissing`]
//! at the verifier. Zero bytes of unverified upper-layer content
//! reach the caller.

extern crate alloc;

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use smallaios_security::crypto::sha3::{Sha3_256, Sha3_256Digest, SHA3_256_DIGEST_LEN};

use super::upper_layer::{UpperEntryKind, UpperError, UpperLayer};

/// Suffix for the SHA-3-256 fingerprint sidecar.
pub const SHA3_SIDECAR_SUFFIX: &str = ".sha3";

/// Suffix for the ML-DSA-65 signature sidecar.
pub const SIG_SIDECAR_SUFFIX: &str = ".sig";

/// Errors raised by the integrity layer.
///
/// Every variant is a fail-closed condition: the caller MUST surface
/// `-EIO` or `-EAUTH` to the syscall layer and emit the audit record
/// named in the variant's docstring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegrityError {
    /// `<name>.sha3` is absent for an upper-layer file. Audit record
    /// `model_hash_mismatch` with reason `missing fingerprint sidecar`.
    /// Surfaces as `-EIO`.
    SidecarMissing,
    /// `<name>.sha3` could not be parsed (wrong length, non-hex byte,
    /// trailing junk). Audit record `model_hash_mismatch` with reason
    /// `unparseable sidecar`. Surfaces as `-EIO`.
    SidecarMalformed,
    /// File bytes' SHA-3-256 disagrees with `<name>.sha3`. Audit record
    /// `model_hash_mismatch`. Surfaces as `-EIO`.
    HashMismatch,
    /// `<name>.sig` is absent and policy requires it. Audit record
    /// `model_load_unsigned`. Surfaces as `-EAUTH`.
    SignatureMissing,
    /// `<name>.sig` is malformed (wrong length, etc). Audit record
    /// `model_signature_invalid`. Surfaces as `-EAUTH`.
    SignatureMalformed,
    /// `<name>.sig` failed ML-DSA-65 verification. Audit record
    /// `model_signature_invalid`. Surfaces as `-EAUTH`.
    SignatureInvalid,
    /// Upper-layer I/O failed while reading the file or sidecar.
    /// Surfaces as `-EIO`.
    Io(UpperError),
}

impl fmt::Display for IntegrityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SidecarMissing => write!(f, "integrity: missing fingerprint sidecar"),
            Self::SidecarMalformed => write!(f, "integrity: malformed fingerprint sidecar"),
            Self::HashMismatch => write!(f, "integrity: SHA-3-256 hash mismatch"),
            Self::SignatureMissing => write!(f, "integrity: missing signature sidecar"),
            Self::SignatureMalformed => write!(f, "integrity: malformed signature sidecar"),
            Self::SignatureInvalid => write!(f, "integrity: invalid ML-DSA-65 signature"),
            Self::Io(e) => write!(f, "integrity: I/O error: {:?}", e),
        }
    }
}

impl From<UpperError> for IntegrityError {
    fn from(e: UpperError) -> Self {
        Self::Io(e)
    }
}

/// Compose `<name>.sha3` from `<name>` (upper-relative path).
pub fn sha3_sidecar_path(name: &str) -> String {
    let mut s = name.to_string();
    s.push_str(SHA3_SIDECAR_SUFFIX);
    s
}

/// Compose `<name>.sig` from `<name>` (upper-relative path).
pub fn sig_sidecar_path(name: &str) -> String {
    let mut s = name.to_string();
    s.push_str(SIG_SIDECAR_SUFFIX);
    s
}

/// Encode a 32-byte digest as a 64-character lowercase hex string. The
/// sidecar format is exactly this — no leading `0x`, no trailing
/// newline, no whitespace.
pub fn encode_hex(digest: &Sha3_256Digest) -> String {
    let bytes = digest.as_bytes();
    let mut out = String::with_capacity(SHA3_256_DIGEST_LEN * 2);
    for b in bytes {
        // Manual hex emission — `core::fmt`'s lowerhex padding requires
        // an allocation per byte and we want a single contiguous push.
        const HEX: &[u8; 16] = b"0123456789abcdef";
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// Parse a 64-character lowercase-hex sidecar back into a 32-byte
/// digest. Returns `None` if the input has the wrong length or contains
/// any non-hex byte. Whitespace is NOT tolerated — sidecar files are
/// emitted exactly 64 bytes long with no terminator.
pub fn decode_hex(s: &str) -> Option<Sha3_256Digest> {
    if s.len() != SHA3_256_DIGEST_LEN * 2 {
        return None;
    }
    let bytes = s.as_bytes();
    let mut out = [0u8; SHA3_256_DIGEST_LEN];
    for (i, chunk) in bytes.chunks_exact(2).enumerate() {
        let hi = decode_nibble(chunk[0])?;
        let lo = decode_nibble(chunk[1])?;
        out[i] = (hi << 4) | lo;
    }
    Some(Sha3_256Digest::from_bytes(out))
}

fn decode_nibble(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(10 + b - b'a'),
        // Reject uppercase — sidecar emission is strictly lowercase
        // and accepting both opens a tamper path on case-insensitive
        // comparisons.
        _ => None,
    }
}

/// Read the entire contents of `path` into a freshly allocated `Vec`,
/// using the upper layer's chunked `read_file` API. Returns
/// [`IntegrityError::Io`] on any underlying error.
pub fn read_full<U: UpperLayer>(upper: &U, path: &str) -> Result<Vec<u8>, IntegrityError> {
    let entry = match upper.lookup(path)? {
        Some(e) => e,
        None => return Err(IntegrityError::Io(UpperError::NotFound)),
    };
    if entry.kind != UpperEntryKind::RegularFile {
        return Err(IntegrityError::Io(UpperError::NotARegularFile));
    }
    let size = entry.size as usize;
    let mut out = alloc::vec![0u8; size];
    let mut filled = 0usize;
    while filled < size {
        let n = upper.read_file(path, filled as u64, &mut out[filled..])?;
        if n == 0 {
            break;
        }
        filled += n;
    }
    out.truncate(filled);
    Ok(out)
}

/// Compute the SHA-3-256 of `data` (one-shot wrapper that hides the
/// `Sha3_256` builder boilerplate from the rest of the overlay).
pub fn hash_bytes(data: &[u8]) -> Sha3_256Digest {
    let mut hasher = Sha3_256::new();
    hasher
        .update(data)
        .expect("fresh hasher cannot fail on first update");
    hasher
        .finalize()
        .expect("fresh hasher cannot fail on first finalize")
}

/// Verify the upper-layer file at `name` against its `<name>.sha3`
/// sidecar.
///
/// Returns the verified bytes on success. On any error returns the
/// fail-closed [`IntegrityError`] — callers MUST NOT touch the bytes
/// surfaced through any other path.
///
/// Implementation order (each step fail-closed):
///
/// 1. Locate `<name>.sha3` on the upper. Missing → [`IntegrityError::SidecarMissing`].
/// 2. Read the sidecar's 64 hex chars. Malformed → [`IntegrityError::SidecarMalformed`].
/// 3. Read the file's bytes. I/O error → [`IntegrityError::Io`].
/// 4. Compute SHA-3-256. Compare. Mismatch → [`IntegrityError::HashMismatch`].
pub fn verify_fingerprint<U: UpperLayer>(upper: &U, name: &str) -> Result<Vec<u8>, IntegrityError> {
    let sidecar = sha3_sidecar_path(name);
    // Step 1+2: read the sidecar.
    let expected = read_sidecar_digest(upper, &sidecar)?;

    // Step 3: read the file.
    let bytes = read_full(upper, name)?;

    // Step 4: compute and compare.
    let actual = hash_bytes(&bytes);
    if actual.as_bytes() != expected.as_bytes() {
        return Err(IntegrityError::HashMismatch);
    }
    Ok(bytes)
}

/// Read and decode the SHA-3 sidecar at `sidecar_path` (which is
/// `<name>.sha3`). Returns the decoded digest or a fail-closed error.
pub fn read_sidecar_digest<U: UpperLayer>(
    upper: &U,
    sidecar_path: &str,
) -> Result<Sha3_256Digest, IntegrityError> {
    match upper.lookup(sidecar_path)? {
        Some(e) if e.kind == UpperEntryKind::RegularFile => {
            let raw = read_full(upper, sidecar_path)?;
            let s = match core::str::from_utf8(&raw) {
                Ok(s) => s,
                Err(_) => return Err(IntegrityError::SidecarMalformed),
            };
            decode_hex(s).ok_or(IntegrityError::SidecarMalformed)
        }
        Some(_) => Err(IntegrityError::SidecarMalformed),
        None => Err(IntegrityError::SidecarMissing),
    }
}

/// Render a digest as the exact bytes the sidecar holds on disk: 64
/// lowercase hex characters, NO trailing newline. Used by the `add`
/// path so the sidecar layout round-trips through [`decode_hex`].
pub fn encode_sidecar_bytes(digest: &Sha3_256Digest) -> Vec<u8> {
    encode_hex(digest).into_bytes()
}

// ─── Signature sidecar ─────────────────────────────────────────────────────────

/// Verify the ML-DSA-65 signature sidecar `<name>.sig` against the
/// model's SHA-3-256 fingerprint. The signed message is exactly the
/// 32 raw fingerprint bytes (NOT the hex encoding).
///
/// Returns `Ok(())` if the signature is present, well-formed, and
/// verifies; otherwise a fail-closed [`IntegrityError`].
///
/// The caller passes the policy's trust anchor (an `MlDsaPublicKey`).
/// In integration the public key comes from the verified-boot
/// measurement, identical to what the squashfs manifest verifier uses.
pub fn verify_signature<U: UpperLayer>(
    upper: &U,
    name: &str,
    fingerprint: &Sha3_256Digest,
    public_key: &smallaios_security::crypto::ml_dsa::MlDsaPublicKey,
) -> Result<(), IntegrityError> {
    use smallaios_security::crypto::ml_dsa::{ml_dsa_65_verify, MlDsaSignature, ML_DSA_65_SIG_LEN};

    let sig_path = sig_sidecar_path(name);
    let raw = match upper.lookup(&sig_path)? {
        Some(e) if e.kind == UpperEntryKind::RegularFile => read_full(upper, &sig_path)?,
        Some(_) => return Err(IntegrityError::SignatureMalformed),
        None => return Err(IntegrityError::SignatureMissing),
    };
    if raw.len() != ML_DSA_65_SIG_LEN {
        return Err(IntegrityError::SignatureMalformed);
    }
    let sig = MlDsaSignature::from_slice(&raw).map_err(|_| IntegrityError::SignatureMalformed)?;
    ml_dsa_65_verify(public_key, fingerprint.as_bytes(), &sig)
        .map_err(|_| IntegrityError::SignatureInvalid)
}

/// Quickly check whether a `.sig` sidecar exists for `name` (without
/// actually verifying it). Used by the `require_signed = false` path,
/// which still verifies a signature if one happens to be present (per
/// spec scenario "Default-off allows unsigned" + the spec text "if
/// present they're verified anyway").
pub fn signature_present<U: UpperLayer>(upper: &U, name: &str) -> Result<bool, IntegrityError> {
    let sig_path = sig_sidecar_path(name);
    match upper.lookup(&sig_path)? {
        Some(e) if e.kind == UpperEntryKind::RegularFile => Ok(true),
        _ => Ok(false),
    }
}

#[cfg(test)]
mod tests {
    use super::super::upper_layer::MockUpperLayer;
    use super::*;

    fn b(s: &str) -> &[u8] {
        s.as_bytes()
    }

    #[test]
    fn sha3_sidecar_path_appends_suffix() {
        assert_eq!(sha3_sidecar_path("foo.onnx"), "foo.onnx.sha3");
        assert_eq!(sha3_sidecar_path("nested/bar"), "nested/bar.sha3");
    }

    #[test]
    fn sig_sidecar_path_appends_suffix() {
        assert_eq!(sig_sidecar_path("foo.onnx"), "foo.onnx.sig");
    }

    #[test]
    fn encode_hex_round_trips_through_decode_hex() {
        let digest = hash_bytes(b("hello world"));
        let s = encode_hex(&digest);
        assert_eq!(s.len(), 64);
        let parsed = decode_hex(&s).unwrap();
        assert_eq!(parsed.as_bytes(), digest.as_bytes());
    }

    #[test]
    fn encode_hex_emits_lowercase() {
        let digest = hash_bytes(b("any-bytes"));
        let s = encode_hex(&digest);
        // No uppercase hex letter SHALL appear.
        assert!(!s.chars().any(|c| c.is_ascii_uppercase()));
        // Only [0-9a-f].
        assert!(s.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn decode_hex_rejects_uppercase() {
        // The lowercase form of any-bytes' digest, then ALL-CAPS.
        let digest = hash_bytes(b("any-bytes"));
        let lower = encode_hex(&digest);
        let upper = lower.to_uppercase();
        assert!(decode_hex(&upper).is_none());
    }

    #[test]
    fn decode_hex_rejects_wrong_length() {
        assert!(decode_hex("aa").is_none());
        assert!(decode_hex("").is_none());
        let too_long = "a".repeat(65);
        assert!(decode_hex(&too_long).is_none());
    }

    #[test]
    fn decode_hex_rejects_non_hex() {
        let mut s = String::with_capacity(64);
        for _ in 0..63 {
            s.push('a');
        }
        s.push('z');
        assert!(decode_hex(&s).is_none());
    }

    #[test]
    fn read_full_returns_file_bytes() {
        let mut u = MockUpperLayer::new();
        u.add_file("foo", b"hello");
        let bytes = read_full(&u, "foo").unwrap();
        assert_eq!(bytes, b"hello".to_vec());
    }

    #[test]
    fn read_full_errors_on_missing() {
        let u = MockUpperLayer::new();
        assert_eq!(
            read_full(&u, "missing"),
            Err(IntegrityError::Io(UpperError::NotFound))
        );
    }

    #[test]
    fn verify_fingerprint_succeeds_on_match() {
        let mut u = MockUpperLayer::new();
        let payload = b"hello model bytes";
        u.add_file("foo", payload);
        let digest = hash_bytes(payload);
        u.add_file("foo.sha3", &encode_sidecar_bytes(&digest));
        let ok = verify_fingerprint(&u, "foo").unwrap();
        assert_eq!(ok, payload.to_vec());
    }

    #[test]
    fn verify_fingerprint_fails_on_corrupted_bytes() {
        let mut u = MockUpperLayer::new();
        let payload = b"hello";
        u.add_file("foo", payload);
        // Sidecar pinned to the *correct* digest.
        let digest = hash_bytes(payload);
        u.add_file("foo.sha3", &encode_sidecar_bytes(&digest));
        // Now corrupt the file bytes (the sidecar still says `payload`).
        u.add_file("foo", b"ATTACKER!");
        assert_eq!(
            verify_fingerprint(&u, "foo"),
            Err(IntegrityError::HashMismatch)
        );
    }

    #[test]
    fn verify_fingerprint_fails_on_missing_sidecar() {
        let mut u = MockUpperLayer::new();
        u.add_file("foo", b"x");
        assert_eq!(
            verify_fingerprint(&u, "foo"),
            Err(IntegrityError::SidecarMissing)
        );
    }

    #[test]
    fn verify_fingerprint_fails_on_malformed_sidecar() {
        let mut u = MockUpperLayer::new();
        u.add_file("foo", b"x");
        u.add_file("foo.sha3", b"not-hex");
        assert_eq!(
            verify_fingerprint(&u, "foo"),
            Err(IntegrityError::SidecarMalformed)
        );
    }

    #[test]
    fn signature_present_detects_sig_file() {
        let mut u = MockUpperLayer::new();
        assert!(!signature_present(&u, "foo").unwrap());
        u.add_file("foo.sig", b"signature-bytes");
        assert!(signature_present(&u, "foo").unwrap());
    }
}
