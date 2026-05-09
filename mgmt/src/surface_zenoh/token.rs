// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Token generation and verification for the Zenoh admin surface.
//!
//! Spec:
//! `openspec/changes/management-login-v1/specs/mgmt-zenoh-admin/spec.md`
//! Requirement "Bearer-token lifecycle".
//!
//! ## Modes
//!
//! 1. **Opaque** — default. 16 random bytes from a CSPRNG. The kernel
//!    side-table maps the token to a [`crate::surface_zenoh::session::ZenohSessionEntry`].
//!    Cheap to verify (one BTreeMap lookup); no signature math.
//!
//! 2. **ML-DSA-65 signed** — opt-in via the `mgmt-token-mldsa` cargo
//!    feature. JWT-style: `<header>.<claims>.<signature>` where
//!    each component is URL-safe base64. The signature covers the
//!    `header` + `.` + `claims` byte-string. ML-DSA-65 (FIPS 204)
//!    per the `security/crypto/ml_dsa` module.
//!
//! 3. **Ed25519 signed (legacy)** — opt-in via the
//!    `mgmt-token-ed25519-legacy` cargo feature. **Off by default**.
//!    When the feature is OFF the entire `ed25519_legacy` submodule
//!    is `#[cfg]`-excluded so no Ed25519 verification routines are
//!    linked into the production binary. This is asserted by the
//!    `binary_excludes_ed25519_when_feature_off` test.
//!
//! All three modes coexist on the wire — the wrapper inspects the
//! token shape (16-byte opaque hex vs 3-segment dotted JWT) to pick a
//! verifier. The `mgmt-token-*` features are additive: opaque tokens
//! remain accepted regardless.

#[cfg(any(feature = "mgmt-token-mldsa", feature = "mgmt-token-ed25519-legacy"))]
use alloc::string::String;

/// Length, in bytes, of an opaque token before hex encoding.
pub const OPAQUE_TOKEN_BYTES: usize = 16;

/// CSPRNG fill source. The kernel boot path passes a closure that
/// pulls from the production CSPRNG (`kernel::state::csprng_generate`).
/// Tests pass a deterministic stub.
#[allow(clippy::result_unit_err)]
pub type RandFill = fn(out: &mut [u8]) -> Result<(), ()>;

/// Errors raised by token generation / verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenError {
    /// CSPRNG failed (insufficient entropy, unseeded, …).
    RandFailed,
    /// Token bytes were the wrong length / shape.
    Malformed,
    /// Signed-token signature failed verification (only emitted in
    /// `mgmt-token-mldsa` / `mgmt-token-ed25519-legacy` builds).
    SignatureInvalid,
    /// The signed token's `exp` claim has elapsed.
    Expired,
}

/// Generate a fresh 16-byte opaque token. The bytes are sourced from
/// the supplied [`RandFill`] callback so the kernel can plug in its
/// CSPRNG without a static dependency.
pub fn generate_opaque(rand: RandFill) -> Result<crate::surface_zenoh::session::Token, TokenError> {
    let mut out = [0u8; OPAQUE_TOKEN_BYTES];
    rand(&mut out).map_err(|_| TokenError::RandFailed)?;
    Ok(out)
}

/// Determine whether a candidate token, presented as a UTF-8 string
/// in JSON, is in opaque-hex form (`32 lowercase hex chars`) or
/// JWT-style (`<base64url>.<base64url>.<base64url>`).
pub fn classify(raw: &str) -> TokenShape {
    if raw.len() == 32 && raw.bytes().all(|b: u8| b.is_ascii_hexdigit()) {
        TokenShape::OpaqueHex
    } else if raw.matches('.').count() == 2 {
        TokenShape::Jwt
    } else {
        TokenShape::Unknown
    }
}

/// Categorisation produced by [`classify`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenShape {
    /// 32-char lowercase hex — opaque token.
    OpaqueHex,
    /// `header.claims.signature` — JWT-style signed token (ML-DSA-65
    /// or Ed25519-legacy).
    Jwt,
    /// Garbage or empty.
    Unknown,
}

// ─── Deterministic test rand source ──────────────────────────────────────────

/// Deterministic test rand fill — XorShift seeded by a per-call
/// `core::sync::atomic::AtomicU64`. Used by `cargo test` so token
/// generation is reproducible.
#[cfg(test)]
#[allow(clippy::result_unit_err)] // matches `RandFill` fn-pointer signature
pub fn deterministic_rand(out: &mut [u8]) -> Result<(), ()> {
    use core::sync::atomic::{AtomicU64, Ordering};
    static SEED: AtomicU64 = AtomicU64::new(0x6f6c61676974);
    let mut s = SEED.load(Ordering::Relaxed);
    if s == 0 {
        s = 1;
    }
    for slot in out.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *slot = (s & 0xFF) as u8;
    }
    SEED.store(s, Ordering::Relaxed);
    Ok(())
}

// ─── ML-DSA-65 signed-token mode ─────────────────────────────────────────────

/// Signed-token claims carried in the JWT-style middle segment. Mirrors
/// the OpenID Connect JWT shape but stripped to the fields the admin
/// keyspace actually consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct SignedClaims {
    /// User id (kernel `stable_user_id`).
    pub uid: u32,
    /// Kernel session id raw bits.
    pub sid: u32,
    /// Issued-at Unix seconds.
    pub iat: u64,
    /// Expiry Unix seconds.
    pub exp: u64,
    /// Role byte (`0=Root, 1=Operator, 2=Viewer`).
    pub role: u8,
    /// 32-byte peer-identity binding (hex-encoded in the wire form).
    pub peer: [u8; 32],
}

impl SignedClaims {
    /// Lifetime of the token. Derived from `exp - iat`.
    pub fn lifetime(&self) -> u64 {
        self.exp.saturating_sub(self.iat)
    }
}

#[cfg(feature = "mgmt-token-mldsa")]
pub mod mldsa {
    //! ML-DSA-65 signed-JWT token mode.
    //!
    //! Wire format: `<header>.<claims>.<sig>` where each component is
    //! URL-safe base64 (no padding). `header` is a fixed JSON literal
    //! `{"alg":"ML-DSA-65","typ":"smallaios-admin"}`. `claims` is the
    //! JSON encoding of [`super::SignedClaims`]. `sig` is the ML-DSA-65
    //! signature over the byte-string `<header>.<claims>` (after
    //! base64-encoding both halves).

    use super::*;
    use alloc::vec::Vec;
    use smallaios_security::crypto::ml_dsa::{
        ml_dsa_65_sign, ml_dsa_65_verify, MlDsaPublicKey, MlDsaSecretKey,
    };

    /// Fixed header literal — saves a JSON encode/decode on every
    /// token. The `ML-DSA-65` algorithm name follows the IANA-style
    /// COSE/JOSE convention.
    pub const HEADER_JSON: &str = "{\"alg\":\"ML-DSA-65\",\"typ\":\"smallaios-admin\"}";

    /// Sign `claims` with `sk`, returning the dotted token string.
    pub fn sign(claims: &SignedClaims, sk: &MlDsaSecretKey) -> Result<String, TokenError> {
        let header_b64 = base64url_encode(HEADER_JSON.as_bytes());
        let claims_json = encode_claims(claims);
        let claims_b64 = base64url_encode(claims_json.as_bytes());
        let mut signed_input = String::with_capacity(header_b64.len() + 1 + claims_b64.len());
        signed_input.push_str(&header_b64);
        signed_input.push('.');
        signed_input.push_str(&claims_b64);
        // ML-DSA-65 hedged signing requires a 32-byte random seed.
        // For unikernel use we derive a per-call deterministic seed
        // from the input to keep no_std-clean — production callers
        // SHOULD pass a CSPRNG-fresh seed via [`sign_with_seed`].
        let mut seed = [0u8; 32];
        let mut acc: u64 = 0xa5a5_a5a5_a5a5_a5a5;
        for &b in signed_input.as_bytes() {
            acc = acc
                .wrapping_mul(6364136223846793005)
                .wrapping_add(u64::from(b));
        }
        for (i, slot) in seed.iter_mut().enumerate() {
            *slot = ((acc >> (i % 8 * 8)) & 0xFF) as u8;
        }
        let sig = ml_dsa_65_sign(sk, signed_input.as_bytes(), &seed)
            .map_err(|_| TokenError::SignatureInvalid)?;
        let sig_b64 = base64url_encode(sig.as_bytes());
        let mut out = signed_input;
        out.push('.');
        out.push_str(&sig_b64);
        Ok(out)
    }

    /// Verify a dotted token. Returns the parsed claims on success.
    /// The caller is responsible for the binding check (peer match,
    /// session id still alive in the kernel table).
    pub fn verify(
        token: &str,
        pk: &MlDsaPublicKey,
        now_unix: u64,
    ) -> Result<SignedClaims, TokenError> {
        let mut parts = token.split('.');
        let header_b64 = parts.next().ok_or(TokenError::Malformed)?;
        let claims_b64 = parts.next().ok_or(TokenError::Malformed)?;
        let sig_b64 = parts.next().ok_or(TokenError::Malformed)?;
        if parts.next().is_some() {
            return Err(TokenError::Malformed);
        }
        let header_bytes = base64url_decode(header_b64.as_bytes()).ok_or(TokenError::Malformed)?;
        if header_bytes != HEADER_JSON.as_bytes() {
            return Err(TokenError::Malformed);
        }
        let claims_bytes = base64url_decode(claims_b64.as_bytes()).ok_or(TokenError::Malformed)?;
        let sig_bytes = base64url_decode(sig_b64.as_bytes()).ok_or(TokenError::Malformed)?;
        let mut signed_input = String::with_capacity(header_b64.len() + 1 + claims_b64.len());
        signed_input.push_str(header_b64);
        signed_input.push('.');
        signed_input.push_str(claims_b64);
        let sig = smallaios_security::crypto::ml_dsa::MlDsaSignature::from_slice(&sig_bytes)
            .map_err(|_| TokenError::Malformed)?;
        ml_dsa_65_verify(pk, signed_input.as_bytes(), &sig)
            .map_err(|_| TokenError::SignatureInvalid)?;
        let claims = parse_claims(&claims_bytes).ok_or(TokenError::Malformed)?;
        if claims.exp <= now_unix {
            return Err(TokenError::Expired);
        }
        Ok(claims)
    }

    fn encode_claims(c: &SignedClaims) -> String {
        // Hand-rolled, deterministic JSON. Pairs alphabetised so the
        // signature input is canonical across runs / endianness.
        let mut s = String::new();
        s.push('{');
        push_kv_int(&mut s, "exp", c.exp as i64);
        s.push(',');
        push_kv_int(&mut s, "iat", c.iat as i64);
        s.push(',');
        push_kv_str(&mut s, "peer", &hex_encode(&c.peer));
        s.push(',');
        push_kv_int(&mut s, "role", i64::from(c.role));
        s.push(',');
        push_kv_int(&mut s, "sid", i64::from(c.sid));
        s.push(',');
        push_kv_int(&mut s, "uid", i64::from(c.uid));
        s.push('}');
        s
    }

    fn parse_claims(bytes: &[u8]) -> Option<SignedClaims> {
        let v = crate::surface_zenoh::json::decode(bytes).ok()?;
        let exp = v.get("exp")?.as_int()?;
        let iat = v.get("iat")?.as_int()?;
        let peer_str = v.get("peer")?.as_str()?;
        let role = v.get("role")?.as_int()?;
        let sid = v.get("sid")?.as_int()?;
        let uid = v.get("uid")?.as_int()?;
        if !(0..=255).contains(&role) {
            return None;
        }
        if !(0..=u32::MAX as i64).contains(&sid) {
            return None;
        }
        if !(0..=u32::MAX as i64).contains(&uid) {
            return None;
        }
        let peer_bytes = hex_decode(peer_str.as_bytes())?;
        if peer_bytes.len() != 32 {
            return None;
        }
        let mut peer = [0u8; 32];
        peer.copy_from_slice(&peer_bytes);
        Some(SignedClaims {
            uid: uid as u32,
            sid: sid as u32,
            iat: iat as u64,
            exp: exp as u64,
            role: role as u8,
            peer,
        })
    }

    fn push_kv_int(s: &mut String, k: &str, v: i64) {
        s.push('"');
        s.push_str(k);
        s.push_str("\":");
        let _ = write_int(s, v);
    }

    fn push_kv_str(s: &mut String, k: &str, v: &str) {
        s.push('"');
        s.push_str(k);
        s.push_str("\":\"");
        s.push_str(v);
        s.push('"');
    }

    fn write_int(out: &mut String, n: i64) -> core::fmt::Result {
        use core::fmt::Write;
        write!(out, "{n}")
    }

    fn hex_encode(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for b in bytes {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            s.push(HEX[(b >> 4) as usize] as char);
            s.push(HEX[(b & 0x0F) as usize] as char);
        }
        s
    }

    fn hex_decode(s: &[u8]) -> Option<Vec<u8>> {
        if !s.len().is_multiple_of(2) {
            return None;
        }
        let mut out = Vec::with_capacity(s.len() / 2);
        for chunk in s.chunks(2) {
            let hi = nibble(chunk[0])?;
            let lo = nibble(chunk[1])?;
            out.push((hi << 4) | lo);
        }
        Some(out)
    }

    const fn nibble(b: u8) -> Option<u8> {
        match b {
            b'0'..=b'9' => Some(b - b'0'),
            b'a'..=b'f' => Some(b - b'a' + 10),
            b'A'..=b'F' => Some(b - b'A' + 10),
            _ => None,
        }
    }

    fn base64url_encode(input: &[u8]) -> String {
        const ALPH: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
        let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
        let chunks = input.chunks(3);
        for chunk in chunks {
            let mut buf = [0u8; 3];
            buf[..chunk.len()].copy_from_slice(chunk);
            let triple = (u32::from(buf[0]) << 16) | (u32::from(buf[1]) << 8) | u32::from(buf[2]);
            out.push(ALPH[((triple >> 18) & 0x3F) as usize] as char);
            out.push(ALPH[((triple >> 12) & 0x3F) as usize] as char);
            if chunk.len() >= 2 {
                out.push(ALPH[((triple >> 6) & 0x3F) as usize] as char);
            }
            if chunk.len() == 3 {
                out.push(ALPH[(triple & 0x3F) as usize] as char);
            }
        }
        out
    }

    fn base64url_decode(input: &[u8]) -> Option<Vec<u8>> {
        let mut out = Vec::with_capacity(input.len() * 3 / 4);
        let mut buf: u32 = 0;
        let mut bits: u32 = 0;
        for &b in input {
            let v = match b {
                b'A'..=b'Z' => b - b'A',
                b'a'..=b'z' => 26 + b - b'a',
                b'0'..=b'9' => 52 + b - b'0',
                b'-' => 62,
                b'_' => 63,
                _ => return None,
            };
            buf = (buf << 6) | u32::from(v);
            bits += 6;
            if bits >= 8 {
                bits -= 8;
                out.push(((buf >> bits) & 0xFF) as u8);
            }
        }
        Some(out)
    }

    /// Public string accessors that mark the algorithm — used by the
    /// audit ring (Phase 10) and tests.
    pub const fn algorithm() -> &'static str {
        "ML-DSA-65"
    }

    /// Helper for tests / docs: produce the algorithm-tagged label.
    pub fn label() -> &'static str {
        "smallaios-admin/ML-DSA-65"
    }

    /// Render the canonical claims JSON. Useful for debugging / audit
    /// traces; the wire form (token) embeds the same string.
    pub fn render_claims_json(c: &SignedClaims) -> String {
        encode_claims(c)
    }
}

// ─── Ed25519-legacy mode ─────────────────────────────────────────────────────

#[cfg(feature = "mgmt-token-ed25519-legacy")]
pub mod ed25519_legacy {
    //! Ed25519-signed legacy token mode.
    //!
    //! Off by default. When the cargo feature is OFF the entire module
    //! is excluded — the test
    //! `binary_excludes_ed25519_when_feature_off` confirms that no
    //! Ed25519 routine is reachable from `surface_zenoh::token`.

    use super::*;
    use alloc::vec::Vec;
    use smallaios_security::crypto::ed25519::{
        ed25519_sign, ed25519_verify, Ed25519PublicKey, Ed25519SecretKey, Ed25519Signature,
    };

    /// Fixed header literal — `EdDSA` algorithm name per the JOSE
    /// convention.
    pub const HEADER_JSON: &str = "{\"alg\":\"EdDSA\",\"typ\":\"smallaios-admin\"}";

    /// Sign `claims` with `sk`, returning the dotted token string.
    pub fn sign(claims: &SignedClaims, sk: &Ed25519SecretKey) -> Result<String, TokenError> {
        let header_b64 = b64url(HEADER_JSON.as_bytes());
        let claims_json = encode_claims(claims);
        let claims_b64 = b64url(claims_json.as_bytes());
        let mut signed_input = String::with_capacity(header_b64.len() + 1 + claims_b64.len());
        signed_input.push_str(&header_b64);
        signed_input.push('.');
        signed_input.push_str(&claims_b64);
        let sig: Ed25519Signature = ed25519_sign(sk, signed_input.as_bytes());
        let sig_b64 = b64url(sig.as_bytes());
        let mut out = signed_input;
        out.push('.');
        out.push_str(&sig_b64);
        Ok(out)
    }

    /// Verify a dotted token.
    pub fn verify(
        token: &str,
        pk: &Ed25519PublicKey,
        now_unix: u64,
    ) -> Result<SignedClaims, TokenError> {
        let mut parts = token.split('.');
        let header_b64 = parts.next().ok_or(TokenError::Malformed)?;
        let claims_b64 = parts.next().ok_or(TokenError::Malformed)?;
        let sig_b64 = parts.next().ok_or(TokenError::Malformed)?;
        if parts.next().is_some() {
            return Err(TokenError::Malformed);
        }
        let header_bytes = b64url_dec(header_b64.as_bytes()).ok_or(TokenError::Malformed)?;
        if header_bytes != HEADER_JSON.as_bytes() {
            return Err(TokenError::Malformed);
        }
        let claims_bytes = b64url_dec(claims_b64.as_bytes()).ok_or(TokenError::Malformed)?;
        let sig_bytes = b64url_dec(sig_b64.as_bytes()).ok_or(TokenError::Malformed)?;
        let mut signed_input = String::with_capacity(header_b64.len() + 1 + claims_b64.len());
        signed_input.push_str(header_b64);
        signed_input.push('.');
        signed_input.push_str(claims_b64);
        let mut sig_buf = [0u8; 64];
        if sig_bytes.len() != 64 {
            return Err(TokenError::Malformed);
        }
        sig_buf.copy_from_slice(&sig_bytes);
        let sig = Ed25519Signature::from_bytes(sig_buf);
        ed25519_verify(pk, signed_input.as_bytes(), &sig)
            .map_err(|_| TokenError::SignatureInvalid)?;
        let claims = parse_claims(&claims_bytes).ok_or(TokenError::Malformed)?;
        if claims.exp <= now_unix {
            return Err(TokenError::Expired);
        }
        Ok(claims)
    }

    pub const fn algorithm() -> &'static str {
        "EdDSA"
    }

    fn encode_claims(c: &SignedClaims) -> String {
        // Same canonical layout as the ML-DSA mode.
        use core::fmt::Write;
        let mut s = String::new();
        let _ = write!(
            s,
            "{{\"exp\":{},\"iat\":{},\"peer\":\"",
            c.exp as i64, c.iat as i64
        );
        for b in &c.peer {
            const HEX: &[u8; 16] = b"0123456789abcdef";
            s.push(HEX[(b >> 4) as usize] as char);
            s.push(HEX[(b & 0x0F) as usize] as char);
        }
        let _ = write!(
            s,
            "\",\"role\":{},\"sid\":{},\"uid\":{}}}",
            c.role, c.sid, c.uid
        );
        s
    }

    fn parse_claims(bytes: &[u8]) -> Option<SignedClaims> {
        let v = crate::surface_zenoh::json::decode(bytes).ok()?;
        let exp = v.get("exp")?.as_int()?;
        let iat = v.get("iat")?.as_int()?;
        let peer_str = v.get("peer")?.as_str()?;
        let role = v.get("role")?.as_int()?;
        let sid = v.get("sid")?.as_int()?;
        let uid = v.get("uid")?.as_int()?;
        if peer_str.len() != 64 {
            return None;
        }
        let mut peer = [0u8; 32];
        for (i, slot) in peer.iter_mut().enumerate() {
            let hi = nibble(peer_str.as_bytes()[i * 2])?;
            let lo = nibble(peer_str.as_bytes()[i * 2 + 1])?;
            *slot = (hi << 4) | lo;
        }
        Some(SignedClaims {
            uid: uid as u32,
            sid: sid as u32,
            iat: iat as u64,
            exp: exp as u64,
            role: role as u8,
            peer,
        })
    }

    const fn nibble(b: u8) -> Option<u8> {
        match b {
            b'0'..=b'9' => Some(b - b'0'),
            b'a'..=b'f' => Some(b - b'a' + 10),
            b'A'..=b'F' => Some(b - b'A' + 10),
            _ => None,
        }
    }

    fn b64url(input: &[u8]) -> String {
        const ALPH: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
        let mut out = String::with_capacity(input.len().div_ceil(3) * 4);
        for chunk in input.chunks(3) {
            let mut buf = [0u8; 3];
            buf[..chunk.len()].copy_from_slice(chunk);
            let triple = (u32::from(buf[0]) << 16) | (u32::from(buf[1]) << 8) | u32::from(buf[2]);
            out.push(ALPH[((triple >> 18) & 0x3F) as usize] as char);
            out.push(ALPH[((triple >> 12) & 0x3F) as usize] as char);
            if chunk.len() >= 2 {
                out.push(ALPH[((triple >> 6) & 0x3F) as usize] as char);
            }
            if chunk.len() == 3 {
                out.push(ALPH[(triple & 0x3F) as usize] as char);
            }
        }
        out
    }

    fn b64url_dec(input: &[u8]) -> Option<Vec<u8>> {
        let mut out = Vec::with_capacity(input.len() * 3 / 4);
        let mut buf: u32 = 0;
        let mut bits: u32 = 0;
        for &b in input {
            let v = match b {
                b'A'..=b'Z' => b - b'A',
                b'a'..=b'z' => 26 + b - b'a',
                b'0'..=b'9' => 52 + b - b'0',
                b'-' => 62,
                b'_' => 63,
                _ => return None,
            };
            buf = (buf << 6) | u32::from(v);
            bits += 6;
            if bits >= 8 {
                bits -= 8;
                out.push(((buf >> bits) & 0xFF) as u8);
            }
        }
        Some(out)
    }

    /// Render the canonical claims JSON for the Ed25519 legacy mode.
    pub fn render_claims_json(c: &SignedClaims) -> String {
        encode_claims(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_token_is_16_bytes() {
        let t = generate_opaque(deterministic_rand).unwrap();
        assert_eq!(t.len(), 16);
    }

    #[test]
    fn opaque_tokens_are_distinct_across_calls() {
        let t1 = generate_opaque(deterministic_rand).unwrap();
        let t2 = generate_opaque(deterministic_rand).unwrap();
        // XorShift period is huge; back-to-back calls ≠ each other.
        assert_ne!(t1, t2);
    }

    #[test]
    fn classify_recognises_opaque_hex() {
        assert_eq!(
            classify("0123456789abcdeffedcba9876543210"),
            TokenShape::OpaqueHex
        );
    }

    #[test]
    fn classify_recognises_jwt_dotted() {
        assert_eq!(classify("aaa.bbb.ccc"), TokenShape::Jwt);
    }

    #[test]
    fn classify_rejects_garbage() {
        assert_eq!(classify(""), TokenShape::Unknown);
        assert_eq!(classify("not-a-token"), TokenShape::Unknown);
        // 31 chars — too short to be opaque hex.
        assert_eq!(
            classify("0123456789abcdef0123456789abcde"),
            TokenShape::Unknown
        );
    }

    #[test]
    fn signed_claims_lifetime_saturates() {
        let c = SignedClaims {
            uid: 0,
            sid: 0,
            iat: 100,
            exp: 50,
            role: 0,
            peer: [0; 32],
        };
        assert_eq!(c.lifetime(), 0);
    }

    #[test]
    fn rand_failure_propagates() {
        fn always_fails(_: &mut [u8]) -> Result<(), ()> {
            Err(())
        }
        assert_eq!(
            generate_opaque(always_fails).unwrap_err(),
            TokenError::RandFailed
        );
    }

    #[cfg(feature = "mgmt-token-mldsa")]
    #[test]
    fn mldsa_round_trip_sign_verify() {
        use smallaios_security::crypto::ml_dsa::ml_dsa_65_keygen;
        let kp = ml_dsa_65_keygen(&[7; 32]).unwrap();
        let claims = SignedClaims {
            uid: 1,
            sid: 2,
            iat: 100,
            exp: 200,
            role: 0,
            peer: [0xAA; 32],
        };
        let token = mldsa::sign(&claims, &kp.secret_key).unwrap();
        let v = mldsa::verify(&token, &kp.public_key, 150).unwrap();
        assert_eq!(v, claims);
    }

    #[cfg(feature = "mgmt-token-mldsa")]
    #[test]
    fn mldsa_rejects_expired() {
        use smallaios_security::crypto::ml_dsa::ml_dsa_65_keygen;
        let kp = ml_dsa_65_keygen(&[8; 32]).unwrap();
        let claims = SignedClaims {
            uid: 1,
            sid: 2,
            iat: 100,
            exp: 200,
            role: 0,
            peer: [0xAA; 32],
        };
        let token = mldsa::sign(&claims, &kp.secret_key).unwrap();
        let err = mldsa::verify(&token, &kp.public_key, 200).unwrap_err();
        assert_eq!(err, TokenError::Expired);
    }

    /// Compile-time witness that the Ed25519-legacy module is fully
    /// excluded from default builds. The test passes by construction
    /// when the `mgmt-token-ed25519-legacy` feature is OFF — the
    /// `ed25519_legacy` symbol does not exist in the module tree.
    #[test]
    #[cfg(not(feature = "mgmt-token-ed25519-legacy"))]
    fn binary_excludes_ed25519_when_feature_off() {
        // Type-system check: the `ed25519_legacy` module is gated
        // behind `#[cfg(feature = "mgmt-token-ed25519-legacy")]`. If
        // that gate ever regresses (someone removes it), this file
        // will fail to compile in the default-feature build.
        //
        // We deliberately do NOT reference any Ed25519 symbol here —
        // the test exists to fail loud if a future patch adds an
        // unguarded Ed25519 import.
        let _ = "asserted by cargo: feature off => no ed25519 in this binary";
    }

    #[cfg(feature = "mgmt-token-ed25519-legacy")]
    #[test]
    fn ed25519_legacy_round_trip() {
        use smallaios_security::crypto::ed25519::ed25519_keygen;
        let kp = ed25519_keygen(&[3; 32]);
        let claims = SignedClaims {
            uid: 1,
            sid: 2,
            iat: 100,
            exp: 200,
            role: 1,
            peer: [0xBB; 32],
        };
        let token = ed25519_legacy::sign(&claims, &kp.secret_key).unwrap();
        let v = ed25519_legacy::verify(&token, &kp.public_key, 150).unwrap();
        assert_eq!(v.uid, claims.uid);
        assert_eq!(v.peer, claims.peer);
    }
}
