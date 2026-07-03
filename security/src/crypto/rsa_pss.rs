// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! RSASSA-PSS signature verification (RFC 8017 §8.1.2 / §9.1.2).
//!
//! Implements the verify side of the TLS 1.3 `rsa_pss_rsae_sha256/384/512`
//! signature schemes. Verify-only: no signing, no key generation, no
//! RSA-OAEP, and no RSA-PKCS#1 v1.5 in either direction (Bleichenbacher
//! legacy — deliberately absent). See
//! `openspec/changes/security-rsa-pss-v1/proposal.md`.
//!
//! - Public keys parse from a strict DER `SubjectPublicKeyInfo`
//!   (`rsaEncryption`, `RSAPublicKey ::= SEQUENCE { INTEGER n, INTEGER e }`);
//!   moduli shorter than 2048 bits are refused at parse time
//!   (NIST SP 800-131A).
//! - EMSA-PSS-VERIFY uses salt length equal to the hash length
//!   (`sLen == hLen`, the standard configuration TLS servers emit).
//! - The final `H == H'` digest comparison is constant-time; every parse
//!   or structural failure is an `Err`, never a panic.
//!
//! **Timing posture.** Verification handles no secret; variable-time
//! big-integer work on the public signature/modulus/exponent is acceptable
//! (documented in [`super::big_int`]). The modular exponentiation is still
//! a constant-time-in-the-exponent Montgomery ladder for reuse.

extern crate alloc;

use alloc::vec;

use super::big_int::BigUint;
use super::constant_time::ct_eq;
use super::mgf1::{mgf1, PssHash};

/// Minimum accepted RSA modulus size (NIST SP 800-131A retires RSA < 2048).
const MIN_MODULUS_BITS: usize = 2048;

/// `rsaEncryption` OID (1.2.840.113549.1.1.1), full DER tag+len+value.
const OID_RSA_ENCRYPTION: [u8; 11] = [
    0x06, 0x09, 0x2A, 0x86, 0x48, 0x86, 0xF7, 0x0D, 0x01, 0x01, 0x01,
];

/// Errors from RSA-PSS parsing and verification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RsaPssError {
    /// Structurally invalid DER in the SubjectPublicKeyInfo or its INTEGERs.
    MalformedDer,
    /// Algorithm OID is not `rsaEncryption`, or the SPKI shape is wrong.
    NotRsaKey,
    /// Modulus is shorter than the 2048-bit floor.
    ModulusTooSmall,
    /// Signature length, range (`s >= n`), or PSS structure is invalid, or
    /// the recomputed digest does not match.
    InvalidSignature,
}

/// A validated RSA public key: modulus `n`, public exponent `e`, and the
/// modulus bit length (≥ 2048, checked at construction).
#[derive(Clone, Debug)]
pub struct RsaPublicKey {
    n: BigUint,
    e: BigUint,
    mod_bits: usize,
}

impl RsaPublicKey {
    /// Parse a DER `SubjectPublicKeyInfo` carrying an RSA public key.
    ///
    /// Shape (RFC 5280 / PKCS#1):
    /// `SEQUENCE { SEQUENCE { rsaEncryption, NULL }, BIT STRING wrapping
    /// SEQUENCE { INTEGER n, INTEGER e } }`. Refuses non-RSA algorithms,
    /// malformed DER, and moduli under 2048 bits.
    pub fn from_spki_der(der: &[u8]) -> Result<Self, RsaPssError> {
        let mut top = Der::new(der);
        let spki = top.seq()?;
        if !top.at_end() {
            return Err(RsaPssError::MalformedDer);
        }
        let mut spki = Der::new(spki);

        // AlgorithmIdentifier ::= SEQUENCE { OID rsaEncryption, params }
        let alg = spki.seq()?;
        if alg.len() < OID_RSA_ENCRYPTION.len()
            || alg[..OID_RSA_ENCRYPTION.len()] != OID_RSA_ENCRYPTION
        {
            return Err(RsaPssError::NotRsaKey);
        }
        // Parameters are conventionally NULL for rsaEncryption; accept NULL
        // or absent, reject anything else non-empty and non-NULL.
        let params = &alg[OID_RSA_ENCRYPTION.len()..];
        if !(params.is_empty() || params == [0x05, 0x00]) {
            return Err(RsaPssError::NotRsaKey);
        }

        // subjectPublicKey BIT STRING wrapping the RSAPublicKey DER.
        let bitstring = spki.bit_string()?;
        if !spki.at_end() {
            return Err(RsaPssError::MalformedDer);
        }

        // RSAPublicKey ::= SEQUENCE { INTEGER n, INTEGER e }
        let mut rsa = Der::new(bitstring);
        let inner = rsa.seq()?;
        if !rsa.at_end() {
            return Err(RsaPssError::MalformedDer);
        }
        let (n, used_n) =
            BigUint::parse_der_integer(inner).map_err(|_| RsaPssError::MalformedDer)?;
        let (e, used_e) =
            BigUint::parse_der_integer(&inner[used_n..]).map_err(|_| RsaPssError::MalformedDer)?;
        if used_n + used_e != inner.len() {
            return Err(RsaPssError::MalformedDer);
        }

        let mod_bits = n.bit_len();
        if mod_bits < MIN_MODULUS_BITS {
            return Err(RsaPssError::ModulusTooSmall);
        }
        if e.is_zero() {
            return Err(RsaPssError::MalformedDer);
        }
        Ok(RsaPublicKey { n, e, mod_bits })
    }

    /// Construct directly from `(n, e)` big-endian byte strings (the form a
    /// certificate parser already holds). Enforces the 2048-bit floor.
    pub fn from_components(n_be: &[u8], e_be: &[u8]) -> Result<Self, RsaPssError> {
        let n = BigUint::from_bytes_be(n_be);
        let e = BigUint::from_bytes_be(e_be);
        let mod_bits = n.bit_len();
        if mod_bits < MIN_MODULUS_BITS {
            return Err(RsaPssError::ModulusTooSmall);
        }
        if e.is_zero() {
            return Err(RsaPssError::MalformedDer);
        }
        Ok(RsaPublicKey { n, e, mod_bits })
    }

    fn modulus_bytes(&self) -> usize {
        self.mod_bits.div_ceil(8)
    }
}

/// Verify an RSASSA-PSS signature (RFC 8017 §8.1.2, `sLen == hLen`).
///
/// Returns `Ok(())` iff `signature` is a valid PSS signature of `message`
/// under `pk` with `hash`. Any length, range, structural, or digest
/// mismatch is `Err(RsaPssError::InvalidSignature)`; no input panics.
pub fn rsa_pss_verify(
    pk: &RsaPublicKey,
    hash: PssHash,
    message: &[u8],
    signature: &[u8],
) -> Result<(), RsaPssError> {
    let k = pk.modulus_bytes();
    // 1. Length check.
    if signature.len() != k {
        return Err(RsaPssError::InvalidSignature);
    }
    // 2. RSAVP1: s = OS2IP(signature); reject s >= n; m = s^e mod n.
    let s = BigUint::from_bytes_be(signature);
    if s.cmp(&pk.n) != core::cmp::Ordering::Less {
        return Err(RsaPssError::InvalidSignature);
    }
    let m = s.mod_exp(&pk.e, &pk.n);

    // 3. EM = I2OSP(m, emLen), emLen = ceil((modBits - 1) / 8).
    let em_bits = pk.mod_bits - 1;
    let em_len = em_bits.div_ceil(8);
    let em = m
        .to_bytes_be_fixed(em_len)
        .ok_or(RsaPssError::InvalidSignature)?;

    // 4. EMSA-PSS-VERIFY.
    emsa_pss_verify(hash, message, &em, em_bits)
}

/// EMSA-PSS-VERIFY per RFC 8017 §9.1.2 with `sLen == hLen`.
fn emsa_pss_verify(
    hash: PssHash,
    message: &[u8],
    em: &[u8],
    em_bits: usize,
) -> Result<(), RsaPssError> {
    let h_len = hash.hlen();
    let s_len = h_len; // sLen == hLen (standard TLS configuration)
    let em_len = em.len();

    // Step 3: emLen < hLen + sLen + 2 → inconsistent.
    if em_len < h_len + s_len + 2 {
        return Err(RsaPssError::InvalidSignature);
    }
    // Step 4: trailer byte.
    if em[em_len - 1] != 0xBC {
        return Err(RsaPssError::InvalidSignature);
    }
    // Step 5: split maskedDB || H.
    let db_len = em_len - h_len - 1;
    let masked_db = &em[..db_len];
    let h = &em[db_len..em_len - 1];

    // Step 6: leftmost (8*emLen - emBits) bits of maskedDB[0] must be 0.
    let mask_bits = 8 * em_len - em_bits;
    if mask_bits > 0 && (masked_db[0] >> (8 - mask_bits)) != 0 {
        return Err(RsaPssError::InvalidSignature);
    }

    // Steps 7-8: DB = maskedDB XOR MGF1(H, dbLen).
    let db_mask = mgf1(h, db_len, hash);
    let mut db = vec![0u8; db_len];
    for i in 0..db_len {
        db[i] = masked_db[i] ^ db_mask[i];
    }
    // Step 9: zero the leftmost mask_bits of DB[0].
    if mask_bits > 0 {
        db[0] &= 0xFF >> mask_bits;
    }

    // Step 10: DB = PS(0x00…) || 0x01 || salt, with |PS| = emLen-hLen-sLen-2.
    let ps_len = em_len - h_len - s_len - 2;
    if db[..ps_len].iter().any(|&b| b != 0) {
        return Err(RsaPssError::InvalidSignature);
    }
    if db[ps_len] != 0x01 {
        return Err(RsaPssError::InvalidSignature);
    }
    // Step 11: salt = last sLen bytes of DB.
    let salt = &db[db_len - s_len..];

    // Steps 12-13: M' = (0x00)*8 || mHash || salt ; H' = Hash(M').
    let m_hash = hash.digest(message);
    let h_prime = hash.digest_concat(&[&[0u8; 8], &m_hash, salt]);

    // Step 14: constant-time H == H'.
    if ct_eq(h, &h_prime).to_bool() {
        Ok(())
    } else {
        Err(RsaPssError::InvalidSignature)
    }
}

// ── strict DER reader (SEQUENCE / BIT STRING) ───────────────────────────────

struct Der<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Der<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Der { buf, pos: 0 }
    }

    fn at_end(&self) -> bool {
        self.pos == self.buf.len()
    }

    fn read_tag_len(&mut self, tag: u8) -> Result<&'a [u8], RsaPssError> {
        let t = *self.buf.get(self.pos).ok_or(RsaPssError::MalformedDer)?;
        if t != tag {
            return Err(RsaPssError::MalformedDer);
        }
        self.pos += 1;
        let l0 = *self.buf.get(self.pos).ok_or(RsaPssError::MalformedDer)?;
        self.pos += 1;
        let len = match l0 {
            0x00..=0x7F => l0 as usize,
            0x81 => {
                let v = *self.buf.get(self.pos).ok_or(RsaPssError::MalformedDer)? as usize;
                self.pos += 1;
                if v < 0x80 {
                    return Err(RsaPssError::MalformedDer);
                }
                v
            }
            0x82 => {
                let hi = *self.buf.get(self.pos).ok_or(RsaPssError::MalformedDer)? as usize;
                let lo = *self
                    .buf
                    .get(self.pos + 1)
                    .ok_or(RsaPssError::MalformedDer)? as usize;
                self.pos += 2;
                let v = (hi << 8) | lo;
                if v < 0x100 {
                    return Err(RsaPssError::MalformedDer);
                }
                v
            }
            _ => return Err(RsaPssError::MalformedDer),
        };
        if self.pos + len > self.buf.len() {
            return Err(RsaPssError::MalformedDer);
        }
        let content = &self.buf[self.pos..self.pos + len];
        self.pos += len;
        Ok(content)
    }

    fn seq(&mut self) -> Result<&'a [u8], RsaPssError> {
        self.read_tag_len(0x30)
    }

    /// Read a BIT STRING and strip its leading "unused bits" octet, which
    /// must be zero for a key encoding.
    fn bit_string(&mut self) -> Result<&'a [u8], RsaPssError> {
        let content = self.read_tag_len(0x03)?;
        if content.first() != Some(&0x00) {
            return Err(RsaPssError::MalformedDer);
        }
        Ok(&content[1..])
    }
}

#[cfg(test)]
mod tests {
    use alloc::vec::Vec;

    use super::super::rsa_pss_test_vectors::{HashId, RsaPssCase, CASES, KEYS_SPKI_DER};
    use super::*;

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
            .collect()
    }

    fn pss_hash(h: HashId) -> PssHash {
        match h {
            HashId::Sha256 => PssHash::Sha256,
            HashId::Sha384 => PssHash::Sha384,
            HashId::Sha512 => PssHash::Sha512,
        }
    }

    fn run_case(c: &RsaPssCase) -> Result<(), RsaPssError> {
        let pk = RsaPublicKey::from_spki_der(&unhex(KEYS_SPKI_DER[c.key]))?;
        rsa_pss_verify(&pk, pss_hash(c.hash), &unhex(c.msg), &unhex(c.sig))
    }

    #[test]
    fn corpus_replays_in_full() {
        let mut executed = 0usize;
        let mut good = 0usize;
        for c in CASES {
            let ok = run_case(c).is_ok();
            assert_eq!(
                ok, c.valid,
                "RSA-PSS case id {} expected valid={} got {}",
                c.id, c.valid, ok
            );
            executed += 1;
            good += c.valid as usize;
        }
        assert_eq!(executed, CASES.len(), "no vector may be skipped");
        // Sanity: the corpus must actually span the good/bad space and all
        // nine (size × hash) combinations (2 messages each → 18 good).
        assert_eq!(good, 18, "expected 18 good vectors (9 combos × 2 msgs)");
        assert!(CASES.len() >= 30, "corpus must contain >= 30 vectors");
        assert_eq!(KEYS_SPKI_DER.len(), 3, "expected RSA-2048/3072/4096 keys");
    }

    // Hand-built structural checks: from_components enforces the 2048-bit
    // floor; the corpus above proves end-to-end verification.

    #[test]
    fn from_components_rejects_small_modulus() {
        // 1024-bit modulus (128 bytes of 0xFF, top bit set → 1024 bits).
        let n = alloc::vec![0xFFu8; 128];
        let e = [0x01, 0x00, 0x01];
        assert_eq!(
            RsaPublicKey::from_components(&n, &e).unwrap_err(),
            RsaPssError::ModulusTooSmall
        );
    }

    #[test]
    fn from_components_accepts_2048() {
        let n = alloc::vec![0xFFu8; 256]; // 2048-bit
        let e = [0x01, 0x00, 0x01];
        let pk = RsaPublicKey::from_components(&n, &e).unwrap();
        assert_eq!(pk.mod_bits, 2048);
        assert_eq!(pk.modulus_bytes(), 256);
    }

    #[test]
    fn spki_rejects_non_rsa_oid() {
        // SEQUENCE { SEQUENCE { OID id-ecPublicKey }, BIT STRING {} }
        let der = [
            0x30, 0x12, 0x30, 0x0B, 0x06, 0x07, 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x02, 0x01, 0x05,
            0x00, 0x03, 0x03, 0x00, 0x30, 0x00,
        ];
        assert_eq!(
            RsaPublicKey::from_spki_der(&der).unwrap_err(),
            RsaPssError::NotRsaKey
        );
    }

    #[test]
    fn verify_rejects_wrong_length_signature() {
        let n = alloc::vec![0xFFu8; 256];
        let e = [0x01, 0x00, 0x01];
        let pk = RsaPublicKey::from_components(&n, &e).unwrap();
        // Signature must be exactly k = 256 bytes.
        assert_eq!(
            rsa_pss_verify(&pk, PssHash::Sha256, b"msg", &[0u8; 128]).unwrap_err(),
            RsaPssError::InvalidSignature
        );
        // s >= n (all 0xFF == n) rejects before exponentiation, no panic.
        assert_eq!(
            rsa_pss_verify(&pk, PssHash::Sha256, b"msg", &[0xFFu8; 256]).unwrap_err(),
            RsaPssError::InvalidSignature
        );
        // Zero-length rejects.
        assert_eq!(
            rsa_pss_verify(&pk, PssHash::Sha256, b"msg", &[]).unwrap_err(),
            RsaPssError::InvalidSignature
        );
    }

    #[test]
    fn from_components_rejects_zero_exponent() {
        let n = alloc::vec![0xFFu8; 256];
        assert_eq!(
            RsaPublicKey::from_components(&n, &[0x00]).unwrap_err(),
            RsaPssError::MalformedDer
        );
    }

    #[test]
    fn spki_error_paths() {
        let base = unhex(KEYS_SPKI_DER[0]);
        // Trailing bytes after the outer SEQUENCE.
        let mut trailing = base.clone();
        trailing.push(0x00);
        assert_eq!(
            RsaPublicKey::from_spki_der(&trailing).unwrap_err(),
            RsaPssError::MalformedDer
        );
        // Truncated input.
        assert!(RsaPublicKey::from_spki_der(&base[..base.len() - 1]).is_err());
        // Empty / garbage.
        assert!(RsaPublicKey::from_spki_der(&[]).is_err());
        assert!(RsaPublicKey::from_spki_der(&[0x30, 0x02, 0x05, 0x00]).is_err());
    }

    #[test]
    fn spki_roundtrips_corpus_keys() {
        // Every corpus key parses and carries a ≥2048-bit modulus.
        for (i, k) in KEYS_SPKI_DER.iter().enumerate() {
            let pk = RsaPublicKey::from_spki_der(&unhex(k)).unwrap();
            assert!(pk.mod_bits >= 2048, "key {i} modulus too small");
            assert!(!pk.e.is_zero());
        }
    }

    // ── EMSA-PSS-VERIFY branch coverage (crafted encoded messages) ──────────

    #[test]
    fn emsa_rejects_inconsistent_length() {
        // emLen < hLen + sLen + 2 (SHA-256 needs ≥ 66).
        let em = [0u8; 60];
        assert_eq!(
            emsa_pss_verify(PssHash::Sha256, b"m", &em, 8 * 60 - 1).unwrap_err(),
            RsaPssError::InvalidSignature
        );
    }

    #[test]
    fn emsa_rejects_bad_trailer() {
        let mut em = [0u8; 256];
        em[255] = 0xAA; // not 0xBC
        assert_eq!(
            emsa_pss_verify(PssHash::Sha256, b"m", &em, 2047).unwrap_err(),
            RsaPssError::InvalidSignature
        );
    }

    #[test]
    fn emsa_rejects_nonzero_leftmost_bits() {
        let mut em = [0u8; 256];
        em[255] = 0xBC;
        em[0] = 0x80; // top mask bit set while emBits = 2047 (mask_bits = 1)
        assert_eq!(
            emsa_pss_verify(PssHash::Sha256, b"m", &em, 2047).unwrap_err(),
            RsaPssError::InvalidSignature
        );
    }

    #[test]
    fn emsa_rejects_bad_db_padding() {
        // Valid trailer and mask bits, but the recovered DB will not have
        // the 0x00…0x01 separator, so the PS/0x01 check must reject.
        let mut em = [0u8; 256];
        em[255] = 0xBC;
        assert_eq!(
            emsa_pss_verify(PssHash::Sha256, b"m", &em, 2047).unwrap_err(),
            RsaPssError::InvalidSignature
        );
    }

    // ── SPKI parser error-branch coverage (crafted DER) ─────────────────────

    fn der_len(len: usize) -> Vec<u8> {
        if len < 0x80 {
            alloc::vec![len as u8]
        } else if len < 0x100 {
            alloc::vec![0x81, len as u8]
        } else {
            alloc::vec![0x82, (len >> 8) as u8, (len & 0xFF) as u8]
        }
    }

    fn der_tlv(tag: u8, body: &[u8]) -> Vec<u8> {
        let mut v = alloc::vec![tag];
        v.extend_from_slice(&der_len(body.len()));
        v.extend_from_slice(body);
        v
    }

    fn der_int(bytes: &[u8]) -> Vec<u8> {
        // Minimal unsigned INTEGER: strip leading zeros, add a 0x00 pad when
        // the high bit is set.
        let mut b = bytes;
        while b.len() > 1 && b[0] == 0 {
            b = &b[1..];
        }
        let mut body = Vec::new();
        if b[0] & 0x80 != 0 {
            body.push(0x00);
        }
        body.extend_from_slice(b);
        der_tlv(0x02, &body)
    }

    /// Build an RSA SubjectPublicKeyInfo from raw `(n, e)`, with an optional
    /// override of the algorithm parameters (default NULL).
    fn build_rsa_spki(n: &[u8], e: &[u8], params: &[u8]) -> Vec<u8> {
        let mut rsapub = der_int(n);
        rsapub.extend(der_int(e));
        let rsapub = der_tlv(0x30, &rsapub);
        let mut bit_body = alloc::vec![0x00];
        bit_body.extend_from_slice(&rsapub);
        let bitstring = der_tlv(0x03, &bit_body);
        let mut alg_body = OID_RSA_ENCRYPTION.to_vec();
        alg_body.extend_from_slice(params);
        let alg = der_tlv(0x30, &alg_body);
        let mut spki = alg;
        spki.extend(bitstring);
        der_tlv(0x30, &spki)
    }

    #[test]
    fn spki_builder_roundtrips() {
        // A well-formed 2048-bit key built by the helper parses.
        let n = alloc::vec![0xC1u8; 256];
        let der = build_rsa_spki(&n, &[0x01, 0x00, 0x01], &[0x05, 0x00]);
        let pk = RsaPublicKey::from_spki_der(&der).unwrap();
        assert_eq!(pk.mod_bits, 2048);
    }

    #[test]
    fn spki_rejects_small_modulus_at_parse() {
        // 1024-bit modulus reaches the size check.
        let n = alloc::vec![0xC1u8; 128];
        let der = build_rsa_spki(&n, &[0x01, 0x00, 0x01], &[0x05, 0x00]);
        assert_eq!(
            RsaPublicKey::from_spki_der(&der).unwrap_err(),
            RsaPssError::ModulusTooSmall
        );
    }

    #[test]
    fn spki_rejects_zero_exponent() {
        let n = alloc::vec![0xC1u8; 256];
        let der = build_rsa_spki(&n, &[0x00], &[0x05, 0x00]);
        assert_eq!(
            RsaPublicKey::from_spki_der(&der).unwrap_err(),
            RsaPssError::MalformedDer
        );
    }

    #[test]
    fn spki_rejects_non_null_params() {
        let n = alloc::vec![0xC1u8; 256];
        // Params = INTEGER 1 instead of NULL.
        let der = build_rsa_spki(&n, &[0x01, 0x00, 0x01], &[0x02, 0x01, 0x01]);
        assert_eq!(
            RsaPublicKey::from_spki_der(&der).unwrap_err(),
            RsaPssError::NotRsaKey
        );
    }

    #[test]
    fn spki_rejects_bad_bitstring_unused_bits() {
        // Non-zero "unused bits" octet in the subjectPublicKey BIT STRING.
        let n = alloc::vec![0xC1u8; 256];
        let mut der = build_rsa_spki(&n, &[0x01, 0x00, 0x01], &[0x05, 0x00]);
        // Find the BIT STRING (0x03) tag and corrupt its unused-bits byte.
        // It is the tag after the AlgorithmIdentifier SEQUENCE; locate the
        // first 0x03 whose following length byte plausibly starts a key.
        let pos = der.windows(1).position(|w| w[0] == 0x03).unwrap();
        // unused-bits octet is 2 bytes after 0x03 for a 0x82 length form.
        let ub = pos + 4;
        der[ub] = 0x01;
        assert_eq!(
            RsaPublicKey::from_spki_der(&der).unwrap_err(),
            RsaPssError::MalformedDer
        );
    }

    #[test]
    fn spki_rejects_trailing_in_rsapublickey() {
        // Extra byte inside the RSAPublicKey SEQUENCE (after e) → used_n +
        // used_e != inner.len().
        let n = alloc::vec![0xC1u8; 256];
        let mut rsapub = der_int(&n);
        rsapub.extend(der_int(&[0x01, 0x00, 0x01]));
        rsapub.push(0x00); // trailing garbage inside the SEQUENCE
        let rsapub = der_tlv(0x30, &rsapub);
        let mut bit_body = alloc::vec![0x00];
        bit_body.extend_from_slice(&rsapub);
        let bitstring = der_tlv(0x03, &bit_body);
        let mut alg_body = OID_RSA_ENCRYPTION.to_vec();
        alg_body.extend_from_slice(&[0x05, 0x00]);
        let alg = der_tlv(0x30, &alg_body);
        let mut spki = alg;
        spki.extend(bitstring);
        let der = der_tlv(0x30, &spki);
        assert_eq!(
            RsaPublicKey::from_spki_der(&der).unwrap_err(),
            RsaPssError::MalformedDer
        );
    }

    #[test]
    fn spki_truncations_error_without_panic() {
        // Every prefix of a real key must error (never panic), exercising
        // the DER reader's truncation branches at each parse step.
        let base = unhex(KEYS_SPKI_DER[0]);
        for cut in 0..base.len() {
            assert!(RsaPublicKey::from_spki_der(&base[..cut]).is_err());
        }
    }
}
