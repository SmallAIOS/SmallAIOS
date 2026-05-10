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

// ─── Phase 5: Load-time signature policy enforcement ──────────────────────────

/// Outcome of [`verify_load`] — granular enough that callers can emit
/// the right audit record for each path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadVerifyOutcome {
    /// Verification succeeded; no signature was present and policy did
    /// not require one. Caller MAY emit no audit (or an INFO-level
    /// `model_load` if the audit policy wants positive coverage).
    OkUnsigned,
    /// Verification succeeded; a signature was present and verified.
    /// Audit record `model_signature_verified` SHOULD be emitted at
    /// INFO level for forensic coverage.
    OkSigned,
}

/// Verify a model's signed-fingerprint chain against an in-memory
/// public key, given the raw sidecar contents.
///
/// This is the "pure" form — no upper-layer I/O. It signs is `sha3_sidecar`
/// is the bytes the on-disk `<name>.sha3` sidecar would hold (64 lower-
/// case hex chars), `sig_sidecar` is the raw bytes the on-disk `<name>.sig`
/// would hold (an [`smallaios_security::crypto::ml_dsa::ML_DSA_65_SIG_LEN`]-
/// byte serialised signature), and `public_key` is the operator-deployed
/// trust anchor (an `ML_DSA_65_PK_LEN`-byte key). The signed message
/// is the **raw 32-byte SHA-3-256 fingerprint** decoded from
/// `sha3_sidecar` (NOT the hex string — operators pre-sign the binary
/// hash so they don't need to re-encode tools).
///
/// This wrapper performs:
///
/// 1. Verify `contents`' SHA-3-256 matches `sha3_sidecar`. Mismatch →
///    [`IntegrityError::HashMismatch`].
/// 2. Decode the public key (length check). Bad key → [`IntegrityError::SignatureMalformed`].
/// 3. Decode the signature (length check). Bad sig → [`IntegrityError::SignatureMalformed`].
/// 4. ML-DSA-65 verify (pubkey, sha3_raw, sig). Mismatch → [`IntegrityError::SignatureInvalid`].
///
/// Returns `Ok(())` on full chain success; otherwise the fail-closed
/// error. The caller MUST surface the matching audit tag and `-EAUTH` /
/// `-EIO` per the load policy in [`verify_load`].
pub fn verify_signed(
    contents: &[u8],
    sha3_sidecar: &[u8],
    sig_sidecar: &[u8],
    public_key: &[u8],
) -> Result<(), IntegrityError> {
    use smallaios_security::crypto::ml_dsa::{
        ml_dsa_65_verify, MlDsaPublicKey, MlDsaSignature, ML_DSA_65_PK_LEN, ML_DSA_65_SIG_LEN,
    };

    // 1. SHA-3 sidecar parse + content match.
    let sidecar_str =
        core::str::from_utf8(sha3_sidecar).map_err(|_| IntegrityError::SidecarMalformed)?;
    let expected_digest = decode_hex(sidecar_str).ok_or(IntegrityError::SidecarMalformed)?;
    let actual_digest = hash_bytes(contents);
    if actual_digest.as_bytes() != expected_digest.as_bytes() {
        return Err(IntegrityError::HashMismatch);
    }

    // 2. Public-key length check.
    if public_key.len() != ML_DSA_65_PK_LEN {
        return Err(IntegrityError::SignatureMalformed);
    }
    let pk =
        MlDsaPublicKey::from_slice(public_key).map_err(|_| IntegrityError::SignatureMalformed)?;

    // 3. Signature length check.
    if sig_sidecar.len() != ML_DSA_65_SIG_LEN {
        return Err(IntegrityError::SignatureMalformed);
    }
    let sig =
        MlDsaSignature::from_slice(sig_sidecar).map_err(|_| IntegrityError::SignatureMalformed)?;

    // 4. ML-DSA-65 verify over the raw 32-byte fingerprint.
    ml_dsa_65_verify(&pk, expected_digest.as_bytes(), &sig)
        .map_err(|_| IntegrityError::SignatureInvalid)
}

/// Apply the operator policy at load time and return the outcome.
///
/// `require_signed`:
///
/// - `false` (permissive default): if `sig_sidecar` is `None` the load
///   succeeds (returns [`LoadVerifyOutcome::OkUnsigned`]) after the
///   SHA-3 fingerprint check on `contents` against `sha3_sidecar`. If
///   `sig_sidecar` is `Some` the signature MUST verify (defense in
///   depth — never accept a signature that fails to verify, even when
///   policy is permissive). Invalid → [`IntegrityError::SignatureInvalid`].
///   Valid → [`LoadVerifyOutcome::OkSigned`].
/// - `true` (strict): `sig_sidecar` is mandatory. `None` →
///   [`IntegrityError::SignatureMissing`]. Otherwise both the SHA-3
///   and the ML-DSA-65 verify must pass.
///
/// `public_key` is the operator-deployed trust anchor. When
/// `public_key.is_empty()` and `require_signed = true` this returns
/// [`IntegrityError::SignatureMalformed`] immediately — defense in
/// depth, refusing to silently load unsigned in the absence of a
/// trust anchor (the operator must explicitly install
/// `/data/mgmt/keys/overlay-pubkey.pub`).
///
/// In the permissive path with an empty `public_key` slice, a signature
/// IF present is treated as an attempted spoof — return
/// [`IntegrityError::SignatureInvalid`]. (Permissive policy ignores
/// missing signatures, but any signature that exists must verify, and
/// nothing can verify against no trust anchor.)
///
/// On `Ok(LoadVerifyOutcome::*)` the caller MUST emit the matching
/// audit tag (or none for `OkUnsigned` if the operator-set audit
/// policy elects not to). On `Err(...)` the caller MUST emit the
/// fail-closed audit tag and surface `-EAUTH` (or `-EIO` for sha3
/// errors).
pub fn verify_load(
    contents: &[u8],
    sha3_sidecar: &[u8],
    sig_sidecar: Option<&[u8]>,
    public_key: &[u8],
    require_signed: bool,
) -> Result<LoadVerifyOutcome, IntegrityError> {
    // Strict-policy gate FIRST: a missing trust anchor in strict mode
    // SHALL fail before we touch any sidecar bytes — operators expect
    // "no pubkey configured" to be visible as a configuration error,
    // not a runtime parse error on the sidecar.
    if require_signed && public_key.is_empty() {
        return Err(IntegrityError::SignatureMalformed);
    }

    // Always do the SHA-3 fingerprint check — it gates BOTH the
    // strict and permissive paths.
    let sidecar_str =
        core::str::from_utf8(sha3_sidecar).map_err(|_| IntegrityError::SidecarMalformed)?;
    let expected_digest = decode_hex(sidecar_str).ok_or(IntegrityError::SidecarMalformed)?;
    let actual_digest = hash_bytes(contents);
    if actual_digest.as_bytes() != expected_digest.as_bytes() {
        return Err(IntegrityError::HashMismatch);
    }

    match (require_signed, sig_sidecar) {
        // Strict + missing → fail with SignatureMissing.
        (true, None) => Err(IntegrityError::SignatureMissing),
        // Strict + present → must verify.
        (true, Some(sig)) => {
            verify_signed(contents, sha3_sidecar, sig, public_key)?;
            Ok(LoadVerifyOutcome::OkSigned)
        }
        // Permissive + missing → OK (sha3 already checked above).
        (false, None) => Ok(LoadVerifyOutcome::OkUnsigned),
        // Permissive + present → defense in depth: signature MUST verify.
        (false, Some(sig)) => {
            verify_signed(contents, sha3_sidecar, sig, public_key)?;
            Ok(LoadVerifyOutcome::OkSigned)
        }
    }
}

#[cfg(test)]
#[path = "integrity_test_vectors.rs"]
mod test_vectors;

#[cfg(test)]
mod tests {
    use super::super::upper_layer::MockUpperLayer;
    use super::test_vectors as tv;
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

    // ─── Phase 5: verify_signed (pure-byte form) ────────────────────────────

    /// Build a `(sha3_sidecar_bytes, sig_sidecar_bytes, pubkey_bytes)`
    /// triple for a fresh primary-keypair signing of `payload`. Used
    /// by every Phase 5 test to keep the per-test boilerplate small.
    fn signed_fixture(
        payload: &[u8],
    ) -> (
        alloc::vec::Vec<u8>,
        alloc::vec::Vec<u8>,
        alloc::vec::Vec<u8>,
    ) {
        let kp = tv::primary_keypair();
        let digest = hash_bytes(payload);
        let sha3_bytes = encode_sidecar_bytes(&digest);
        let sig_bytes = tv::sign_message_bytes(&kp, digest.as_bytes());
        let pk_bytes = kp.public_key.as_bytes().to_vec();
        (sha3_bytes, sig_bytes, pk_bytes)
    }

    #[test]
    fn verify_signed_succeeds_on_valid_chain() {
        let payload = tv::MODEL_BYTES;
        let (sha3, sig, pk) = signed_fixture(payload);
        verify_signed(payload, &sha3, &sig, &pk).expect("signed chain must verify");
    }

    #[test]
    fn verify_signed_fails_on_corrupted_payload() {
        let payload = tv::MODEL_BYTES;
        let (sha3, sig, pk) = signed_fixture(payload);
        // SHA-3 fingerprint check should fire before the ML-DSA verify.
        let corrupt = b"different bytes";
        assert_eq!(
            verify_signed(corrupt, &sha3, &sig, &pk),
            Err(IntegrityError::HashMismatch)
        );
    }

    #[test]
    fn verify_signed_fails_on_wrong_pubkey() {
        let payload = tv::MODEL_BYTES;
        let (sha3, sig, _pk) = signed_fixture(payload);
        // Primary signed but we supply the attacker pubkey.
        let attacker = tv::attacker_keypair();
        assert_eq!(
            verify_signed(payload, &sha3, &sig, attacker.public_key.as_bytes()),
            Err(IntegrityError::SignatureInvalid)
        );
    }

    #[test]
    fn verify_signed_fails_on_corrupt_signature() {
        let payload = tv::MODEL_BYTES;
        let (sha3, mut sig, pk) = signed_fixture(payload);
        // Flip a byte in the middle of the signature.
        sig[100] ^= 0xFF;
        let res = verify_signed(payload, &sha3, &sig, &pk);
        assert!(matches!(
            res,
            Err(IntegrityError::SignatureInvalid) | Err(IntegrityError::SignatureMalformed)
        ));
    }

    #[test]
    fn verify_signed_fails_on_truncated_signature() {
        let payload = tv::MODEL_BYTES;
        let (sha3, mut sig, pk) = signed_fixture(payload);
        sig.truncate(10);
        assert_eq!(
            verify_signed(payload, &sha3, &sig, &pk),
            Err(IntegrityError::SignatureMalformed)
        );
    }

    #[test]
    fn verify_signed_fails_on_oversized_signature() {
        let payload = tv::MODEL_BYTES;
        let (sha3, mut sig, pk) = signed_fixture(payload);
        sig.push(0x00);
        assert_eq!(
            verify_signed(payload, &sha3, &sig, &pk),
            Err(IntegrityError::SignatureMalformed)
        );
    }

    #[test]
    fn verify_signed_fails_on_truncated_pubkey() {
        let payload = tv::MODEL_BYTES;
        let (sha3, sig, pk) = signed_fixture(payload);
        let short = &pk[..pk.len() - 1];
        assert_eq!(
            verify_signed(payload, &sha3, &sig, short),
            Err(IntegrityError::SignatureMalformed)
        );
    }

    #[test]
    fn verify_signed_fails_on_oversized_pubkey() {
        let payload = tv::MODEL_BYTES;
        let (sha3, sig, mut pk) = signed_fixture(payload);
        pk.push(0x00);
        assert_eq!(
            verify_signed(payload, &sha3, &sig, &pk),
            Err(IntegrityError::SignatureMalformed)
        );
    }

    #[test]
    fn verify_signed_fails_on_malformed_sha3_non_utf8() {
        let payload = tv::MODEL_BYTES;
        let (_sha3, sig, pk) = signed_fixture(payload);
        // 0xC3 0x28 is invalid UTF-8 (continuation byte without lead).
        let bogus_sha3 = [0xC3u8, 0x28];
        assert_eq!(
            verify_signed(payload, &bogus_sha3, &sig, &pk),
            Err(IntegrityError::SidecarMalformed)
        );
    }

    #[test]
    fn verify_signed_fails_on_malformed_sha3_wrong_length() {
        let payload = tv::MODEL_BYTES;
        let (_sha3, sig, pk) = signed_fixture(payload);
        let short_sha3 = b"abcd";
        assert_eq!(
            verify_signed(payload, short_sha3, &sig, &pk),
            Err(IntegrityError::SidecarMalformed)
        );
    }

    #[test]
    fn verify_signed_fails_on_malformed_sha3_non_hex() {
        let payload = tv::MODEL_BYTES;
        let (_sha3, sig, pk) = signed_fixture(payload);
        // 64-char string but with a non-hex char.
        let mut bad = alloc::vec![b'a'; 63];
        bad.push(b'z');
        assert_eq!(
            verify_signed(payload, &bad, &sig, &pk),
            Err(IntegrityError::SidecarMalformed)
        );
    }

    #[test]
    fn verify_signed_signs_raw_fingerprint_not_hex() {
        // Defensive: confirm the verifier signs the 32-byte raw digest,
        // NOT the 64-char hex string. If someone changed the impl to
        // hex-as-message this test would fail because the signature
        // was made on the raw bytes.
        let payload = tv::MODEL_BYTES;
        let kp = tv::primary_keypair();
        let digest = hash_bytes(payload);
        let sha3_bytes = encode_sidecar_bytes(&digest);
        // Sign the raw 32 bytes (matches verify_signed's expectation).
        let sig_raw = tv::sign_message_bytes(&kp, digest.as_bytes());
        verify_signed(payload, &sha3_bytes, &sig_raw, kp.public_key.as_bytes()).unwrap();

        // Sign the HEX (64 bytes) — verify_signed should reject because
        // it always signs the raw fingerprint.
        let sig_hex = tv::sign_message_bytes(&kp, sha3_bytes.as_slice());
        assert_eq!(
            verify_signed(payload, &sha3_bytes, &sig_hex, kp.public_key.as_bytes()),
            Err(IntegrityError::SignatureInvalid)
        );
    }

    // ─── Phase 5: verify_load (policy enforcement) ──────────────────────────

    #[test]
    fn verify_load_permissive_no_sig_succeeds() {
        let payload = tv::MODEL_BYTES;
        let (sha3, _sig, pk) = signed_fixture(payload);
        let outcome = verify_load(payload, &sha3, None, &pk, false).unwrap();
        assert_eq!(outcome, LoadVerifyOutcome::OkUnsigned);
    }

    #[test]
    fn verify_load_permissive_valid_sig_succeeds() {
        let payload = tv::MODEL_BYTES;
        let (sha3, sig, pk) = signed_fixture(payload);
        let outcome = verify_load(payload, &sha3, Some(&sig), &pk, false).unwrap();
        assert_eq!(outcome, LoadVerifyOutcome::OkSigned);
    }

    #[test]
    fn verify_load_permissive_invalid_sig_fails_defense_in_depth() {
        // Spec: "Permissive policy, invalid sig → fails (defense in depth)".
        let payload = tv::MODEL_BYTES;
        let (sha3, sig, _pk) = signed_fixture(payload);
        let attacker = tv::attacker_keypair();
        let res = verify_load(
            payload,
            &sha3,
            Some(&sig),
            attacker.public_key.as_bytes(),
            false,
        );
        assert_eq!(res, Err(IntegrityError::SignatureInvalid));
    }

    #[test]
    fn verify_load_permissive_corrupted_sha3_fails() {
        let payload = tv::MODEL_BYTES;
        let (_sha3, sig, pk) = signed_fixture(payload);
        // Different content (so its sha3 differs) — sha3_sidecar still
        // points at MODEL_BYTES' sha3.
        let other_payload = tv::MODEL_BYTES_OTHER;
        let (sha3_other, _, _) = signed_fixture(other_payload);
        // sha3_other is for OTHER, but we hand it `payload` bytes. Hash
        // mismatch fires before signature check.
        assert_eq!(
            verify_load(payload, &sha3_other, Some(&sig), &pk, false),
            Err(IntegrityError::HashMismatch)
        );
    }

    #[test]
    fn verify_load_permissive_malformed_sha3_fails() {
        let payload = tv::MODEL_BYTES;
        let (_sha3, sig, pk) = signed_fixture(payload);
        let bad = b"not-hex";
        assert_eq!(
            verify_load(payload, bad, Some(&sig), &pk, false),
            Err(IntegrityError::SidecarMalformed)
        );
    }

    #[test]
    fn verify_load_strict_no_sig_returns_signature_missing() {
        let payload = tv::MODEL_BYTES;
        let (sha3, _sig, pk) = signed_fixture(payload);
        assert_eq!(
            verify_load(payload, &sha3, None, &pk, true),
            Err(IntegrityError::SignatureMissing)
        );
    }

    #[test]
    fn verify_load_strict_valid_sig_succeeds() {
        let payload = tv::MODEL_BYTES;
        let (sha3, sig, pk) = signed_fixture(payload);
        let outcome = verify_load(payload, &sha3, Some(&sig), &pk, true).unwrap();
        assert_eq!(outcome, LoadVerifyOutcome::OkSigned);
    }

    #[test]
    fn verify_load_strict_invalid_sig_fails() {
        let payload = tv::MODEL_BYTES;
        let (sha3, sig, _pk) = signed_fixture(payload);
        let attacker = tv::attacker_keypair();
        assert_eq!(
            verify_load(
                payload,
                &sha3,
                Some(&sig),
                attacker.public_key.as_bytes(),
                true
            ),
            Err(IntegrityError::SignatureInvalid)
        );
    }

    #[test]
    fn verify_load_strict_malformed_sig_fails() {
        let payload = tv::MODEL_BYTES;
        let (sha3, mut sig, pk) = signed_fixture(payload);
        sig.truncate(5);
        assert_eq!(
            verify_load(payload, &sha3, Some(&sig), &pk, true),
            Err(IntegrityError::SignatureMalformed)
        );
    }

    #[test]
    fn verify_load_strict_empty_pubkey_fails_with_clear_error() {
        // Spec: "Pubkey path empty + strict → fails with clear error".
        let payload = tv::MODEL_BYTES;
        let (sha3, sig, _pk) = signed_fixture(payload);
        let empty_pk: &[u8] = &[];
        assert_eq!(
            verify_load(payload, &sha3, Some(&sig), empty_pk, true),
            Err(IntegrityError::SignatureMalformed)
        );
    }

    #[test]
    fn verify_load_strict_empty_pubkey_no_sig_still_fails_pubkey_first() {
        // Defense in depth: a strict policy with no pubkey rejects
        // BEFORE noticing the missing signature — the operator's
        // misconfiguration is the real problem.
        let payload = tv::MODEL_BYTES;
        let (sha3, _sig, _pk) = signed_fixture(payload);
        let empty_pk: &[u8] = &[];
        assert_eq!(
            verify_load(payload, &sha3, None, empty_pk, true),
            Err(IntegrityError::SignatureMalformed)
        );
    }

    #[test]
    fn verify_load_permissive_empty_pubkey_no_sig_succeeds() {
        // No trust anchor + no sig + permissive = OK (sha3 only).
        let payload = tv::MODEL_BYTES;
        let (sha3, _sig, _pk) = signed_fixture(payload);
        let empty_pk: &[u8] = &[];
        let outcome = verify_load(payload, &sha3, None, empty_pk, false).unwrap();
        assert_eq!(outcome, LoadVerifyOutcome::OkUnsigned);
    }

    #[test]
    fn verify_load_permissive_empty_pubkey_with_sig_fails() {
        // No trust anchor but a sig is present — permissive policy still
        // requires the sig to verify (defense in depth). Empty pubkey
        // → fail.
        let payload = tv::MODEL_BYTES;
        let (sha3, sig, _pk) = signed_fixture(payload);
        let empty_pk: &[u8] = &[];
        assert_eq!(
            verify_load(payload, &sha3, Some(&sig), empty_pk, false),
            Err(IntegrityError::SignatureMalformed)
        );
    }

    #[test]
    fn verify_load_strict_corrupt_payload_fails_hash_first() {
        // Hash check fires before signature check.
        let payload = tv::MODEL_BYTES;
        let (sha3, sig, pk) = signed_fixture(payload);
        let corrupt = b"corrupted";
        assert_eq!(
            verify_load(corrupt, &sha3, Some(&sig), &pk, true),
            Err(IntegrityError::HashMismatch)
        );
    }

    #[test]
    fn verify_load_policy_flip_simulates_existing_unsigned_now_failing() {
        // Spec scenario: unsigned model exists in upper. Policy was
        // false → flips to true. Subsequent loads MUST fail with
        // SignatureMissing. The on-disk file is not invalidated; the
        // operator can re-upload with sig.
        let payload = tv::MODEL_BYTES;
        let kp = tv::primary_keypair();
        let digest = hash_bytes(payload);
        let sha3_bytes = encode_sidecar_bytes(&digest);
        let pk_bytes = kp.public_key.as_bytes();

        // Permissive load: succeeds.
        let outcome =
            verify_load(payload, &sha3_bytes, None, pk_bytes, false).expect("permissive ok");
        assert_eq!(outcome, LoadVerifyOutcome::OkUnsigned);

        // Operator flips policy to strict. Same on-disk content; same
        // call site; but now MUST fail.
        let res = verify_load(payload, &sha3_bytes, None, pk_bytes, true);
        assert_eq!(res, Err(IntegrityError::SignatureMissing));

        // Now operator re-uploads with a sig; strict load succeeds.
        let sig = tv::sign_message_bytes(&kp, digest.as_bytes());
        let outcome2 =
            verify_load(payload, &sha3_bytes, Some(&sig), pk_bytes, true).expect("strict ok");
        assert_eq!(outcome2, LoadVerifyOutcome::OkSigned);
    }

    #[test]
    fn verify_load_policy_flip_to_permissive_keeps_signed_models_loadable() {
        // Spec text: "When flipped true → false, the policy relaxes —
        // existing signed models keep their sidecars; future uploads
        // MAY omit sig."
        let payload = tv::MODEL_BYTES;
        let (sha3, sig, pk) = signed_fixture(payload);
        // Strict load passes.
        verify_load(payload, &sha3, Some(&sig), &pk, true).unwrap();
        // Operator flips policy off; signed models keep working.
        let outcome = verify_load(payload, &sha3, Some(&sig), &pk, false).unwrap();
        assert_eq!(outcome, LoadVerifyOutcome::OkSigned);
    }

    #[test]
    fn verify_load_streaming_does_not_retain_contents_beyond_call() {
        // The test asserts the API takes `&[u8]` and returns a small
        // outcome — no internal `Vec` of contents leaks to the caller
        // and no static buffer holds a reference. We verify by
        // dropping the contents after the call returns.
        let payload = tv::MODEL_BYTES.to_vec();
        let (sha3, sig, pk) = signed_fixture(&payload);
        let outcome = verify_load(&payload, &sha3, Some(&sig), &pk, true).unwrap();
        assert_eq!(outcome, LoadVerifyOutcome::OkSigned);
        // Drop the input — nothing in `outcome` should depend on it.
        drop(payload);
        // Only the small Copy outcome remains.
        let _: LoadVerifyOutcome = outcome;
    }

    #[test]
    fn verify_load_permissive_reuploaded_with_attacker_sig_rejected() {
        // Defense-in-depth scenario: operator was permissive, model
        // had no sig, then a malicious party UPLOADS a new sig file
        // signed by a key the operator does NOT trust. Permissive
        // policy STILL rejects because sigs that don't verify fail
        // closed.
        let payload = tv::MODEL_BYTES;
        let kp = tv::primary_keypair();
        let attacker = tv::attacker_keypair();
        let digest = hash_bytes(payload);
        let sha3_bytes = encode_sidecar_bytes(&digest);
        let attacker_sig = tv::sign_message_bytes(&attacker, digest.as_bytes());
        // Operator's trust anchor is the primary key.
        let pk_bytes = kp.public_key.as_bytes();
        let res = verify_load(payload, &sha3_bytes, Some(&attacker_sig), pk_bytes, false);
        assert_eq!(res, Err(IntegrityError::SignatureInvalid));
    }

    #[test]
    fn verify_load_returns_correct_outcome_variants_distinct() {
        // Sanity: the two Ok variants are not the same, so a caller
        // emitting `model_signature_verified` only on `OkSigned` will
        // not also emit on `OkUnsigned`.
        assert_ne!(LoadVerifyOutcome::OkSigned, LoadVerifyOutcome::OkUnsigned);
    }

    #[test]
    fn verify_load_strict_with_correct_sig_other_payload() {
        // Cross-check: signing one payload's digest does NOT verify
        // against another payload, even with the right key.
        let payload_a = tv::MODEL_BYTES;
        let payload_b = tv::MODEL_BYTES_OTHER;
        let (sha3_a, sig_a, pk) = signed_fixture(payload_a);
        let (sha3_b, _sig_b, _) = signed_fixture(payload_b);
        // Trying to load B with A's sha3+sig should fail at hash check.
        assert_eq!(
            verify_load(payload_b, &sha3_a, Some(&sig_a), &pk, true),
            Err(IntegrityError::HashMismatch)
        );
        // Or with B's sha3 + A's sig → sig fails to verify (digest
        // mismatch under the hood).
        let sig_a_for_b_attempt = sig_a.clone();
        assert_eq!(
            verify_load(payload_b, &sha3_b, Some(&sig_a_for_b_attempt), &pk, true),
            Err(IntegrityError::SignatureInvalid)
        );
    }

    #[test]
    fn verify_load_permissive_corrupt_then_no_pubkey_falls_through_correctly() {
        // Permissive + sig present + empty pk → SignatureMalformed.
        // Should NOT silently treat empty pk as "no trust anchor, skip
        // sig check".
        let payload = tv::MODEL_BYTES;
        let (sha3, sig, _pk) = signed_fixture(payload);
        let empty: &[u8] = &[];
        let res = verify_load(payload, &sha3, Some(&sig), empty, false);
        assert_eq!(res, Err(IntegrityError::SignatureMalformed));
    }

    #[test]
    fn verify_load_strict_signature_invalid_takes_precedence_over_missing_pubkey() {
        // If pubkey is fine but signature is wrong, fail with
        // SignatureInvalid — not SignatureMalformed.
        let payload = tv::MODEL_BYTES;
        let kp = tv::primary_keypair();
        let attacker = tv::attacker_keypair();
        let digest = hash_bytes(payload);
        let sha3_bytes = encode_sidecar_bytes(&digest);
        let sig = tv::sign_message_bytes(&attacker, digest.as_bytes());
        let res = verify_load(
            payload,
            &sha3_bytes,
            Some(&sig),
            kp.public_key.as_bytes(),
            true,
        );
        assert_eq!(res, Err(IntegrityError::SignatureInvalid));
    }

    #[test]
    fn integrity_error_displays_have_useful_message() {
        // Each variant Display's distinct human-readable reason.
        use alloc::string::ToString;
        let cases = [
            IntegrityError::SignatureMissing,
            IntegrityError::SignatureMalformed,
            IntegrityError::SignatureInvalid,
        ];
        for e in cases {
            let s = e.to_string();
            assert!(!s.is_empty());
            assert!(s.starts_with("integrity:"));
        }
    }

    #[test]
    fn verify_load_outcome_is_copy() {
        // Confirm the outcome is a small Copy enum (no heap, no lifetime).
        fn assert_copy<T: Copy>() {}
        assert_copy::<LoadVerifyOutcome>();
    }

    #[test]
    fn verify_signed_stable_under_repeated_verify() {
        // Verify is pure — calling it twice on the same input is
        // deterministic.
        let payload = tv::MODEL_BYTES;
        let (sha3, sig, pk) = signed_fixture(payload);
        for _ in 0..3 {
            verify_signed(payload, &sha3, &sig, &pk).unwrap();
        }
    }

    #[test]
    fn verify_load_handles_zero_byte_sig_sidecar() {
        let payload = tv::MODEL_BYTES;
        let (sha3, _sig, pk) = signed_fixture(payload);
        let empty_sig: &[u8] = &[];
        // Empty sig sidecar is malformed (length != ML_DSA_65_SIG_LEN).
        assert_eq!(
            verify_load(payload, &sha3, Some(empty_sig), &pk, true),
            Err(IntegrityError::SignatureMalformed)
        );
    }

    #[test]
    fn verify_load_handles_oversize_sig_sidecar() {
        let payload = tv::MODEL_BYTES;
        let (sha3, mut sig, pk) = signed_fixture(payload);
        sig.extend_from_slice(&[0xAAu8; 64]);
        assert_eq!(
            verify_load(payload, &sha3, Some(&sig), &pk, true),
            Err(IntegrityError::SignatureMalformed)
        );
    }

    #[test]
    fn verify_load_strict_short_pubkey_fails_pubkey_check() {
        let payload = tv::MODEL_BYTES;
        let (sha3, sig, pk) = signed_fixture(payload);
        let short_pk = &pk[..pk.len() - 10];
        assert_eq!(
            verify_load(payload, &sha3, Some(&sig), short_pk, true),
            Err(IntegrityError::SignatureMalformed)
        );
    }

    #[test]
    fn verify_load_strict_short_pubkey_no_sig_fails_pubkey_check() {
        // With missing sig + short pubkey, strict policy: the
        // SHA3 still passes; pubkey-len check passes (we didn't
        // declare it empty), and the missing-sig path fires.
        let payload = tv::MODEL_BYTES;
        let (sha3, _sig, pk) = signed_fixture(payload);
        let short_pk = &pk[..pk.len() - 1];
        // verify_load only validates pubkey length when actually
        // verifying a signature; missing-sig in strict mode is
        // SignatureMissing.
        let res = verify_load(payload, &sha3, None, short_pk, true);
        assert_eq!(res, Err(IntegrityError::SignatureMissing));
    }

    #[test]
    fn verify_load_permissive_and_strict_agree_on_valid_signed() {
        let payload = tv::MODEL_BYTES;
        let (sha3, sig, pk) = signed_fixture(payload);
        let r1 = verify_load(payload, &sha3, Some(&sig), &pk, false).unwrap();
        let r2 = verify_load(payload, &sha3, Some(&sig), &pk, true).unwrap();
        assert_eq!(r1, LoadVerifyOutcome::OkSigned);
        assert_eq!(r2, LoadVerifyOutcome::OkSigned);
    }

    #[test]
    fn verify_load_distinguishes_okunsigned_from_oksigned() {
        let payload = tv::MODEL_BYTES;
        let (sha3, sig, pk) = signed_fixture(payload);
        let none = verify_load(payload, &sha3, None, &pk, false).unwrap();
        let some = verify_load(payload, &sha3, Some(&sig), &pk, false).unwrap();
        assert_eq!(none, LoadVerifyOutcome::OkUnsigned);
        assert_eq!(some, LoadVerifyOutcome::OkSigned);
    }

    #[test]
    #[allow(clippy::type_complexity)]
    fn verify_load_audit_pure_function_no_io() {
        // verify_load takes only byte slices — no `UpperLayer` at all,
        // so it has no I/O to fail on. The streaming-verify spec calls
        // for the verify to not hold the bytes beyond the verification
        // window; this test is the type-level proof.
        // (Compilation alone is the assertion.)
        let _ptr: fn(
            &[u8],
            &[u8],
            Option<&[u8]>,
            &[u8],
            bool,
        ) -> Result<LoadVerifyOutcome, IntegrityError> = verify_load;
    }
}
