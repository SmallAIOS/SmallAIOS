// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ECDSA signature verification on NIST P-256 with SHA-256.
//!
//! Implements the verify side of the TLS 1.3 `ecdsa_secp256r1_sha256`
//! signature scheme (ANSI X9.62 / FIPS 186-5). Verify-only: no signing,
//! no key generation — see
//! `openspec/changes/security-ecdsa-p256-v1/proposal.md`.
//!
//! Parsing is strict DER, fail-closed:
//!
//! - Signatures must be exactly `SEQUENCE { INTEGER r, INTEGER s }` with
//!   minimally-encoded, positive integers in `[1, n-1]` and no trailing
//!   bytes.
//! - Public keys are accepted as either a `SubjectPublicKeyInfo`
//!   (`id-ecPublicKey` + `secp256r1`, uncompressed point) or a raw
//!   65-byte uncompressed point; the point is validated against the
//!   curve equation before use.
//!
//! ECDSA is inherently malleable (`(r, s)` and `(r, n-s)` both verify);
//! no low-s rule is enforced because public CAs emit high-s signatures
//! (proposal, Threat model).
//!
//! Message hashing uses the crate's own SHA-256 (`crate::sha2`); no
//! external dependencies are introduced.

use super::p256::{AffinePoint, ProjectivePoint, Scalar};
use crate::sha2::sha256;

/// Errors from ECDSA-P256 parsing and verification. All parse failures
/// are error returns — malformed input never panics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EcdsaP256Error {
    /// Structurally invalid DER (bad tag, bad length, truncation,
    /// non-minimal encoding, negative INTEGER, trailing bytes).
    MalformedDer,
    /// `r` or `s` is zero or not below the group order `n`.
    ScalarOutOfRange,
    /// The public key's DER shape, algorithm OID, curve OID, or point
    /// format is not the supported `secp256r1` uncompressed form.
    InvalidPublicKey,
    /// The decoded public-key coordinates do not satisfy the P-256
    /// curve equation.
    PointNotOnCurve,
    /// The signature does not verify for this message and key.
    InvalidSignature,
}

/// A validated P-256 public key (guaranteed on-curve).
#[derive(Clone, Copy, Debug)]
pub struct EcdsaP256PublicKey {
    point: AffinePoint,
}

/// `id-ecPublicKey` (1.2.840.10045.2.1), DER tag + length + value.
const OID_EC_PUBLIC_KEY: [u8; 9] = [0x06, 0x07, 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x02, 0x01];
/// `secp256r1` / `prime256v1` (1.2.840.10045.3.1.7), DER tag + length + value.
const OID_SECP256R1: [u8; 10] = [0x06, 0x08, 0x2A, 0x86, 0x48, 0xCE, 0x3D, 0x03, 0x01, 0x07];

impl EcdsaP256PublicKey {
    /// Parse a raw uncompressed SEC 1 point: `0x04 || X || Y` (65 bytes).
    /// This is the form `tls-client`'s X.509 parser extracts from
    /// certificates. Rejects non-canonical coordinates and off-curve
    /// points.
    pub fn from_uncompressed(bytes: &[u8; 65]) -> Result<Self, EcdsaP256Error> {
        if bytes[0] != 0x04 {
            return Err(EcdsaP256Error::InvalidPublicKey);
        }
        let x: [u8; 32] = bytes[1..33].try_into().unwrap();
        let y: [u8; 32] = bytes[33..65].try_into().unwrap();
        let point = AffinePoint::new_checked(&x, &y).ok_or(EcdsaP256Error::PointNotOnCurve)?;
        Ok(EcdsaP256PublicKey { point })
    }

    /// Parse a DER `SubjectPublicKeyInfo` whose algorithm is
    /// `id-ecPublicKey` with the `secp256r1` named curve and whose
    /// subjectPublicKey BIT STRING carries an uncompressed point — the
    /// standard TLS-emitted form. Anything else is rejected.
    pub fn from_spki_der(der: &[u8]) -> Result<Self, EcdsaP256Error> {
        let mut r = Der::new(der);
        let outer_len = r
            .tag_len(0x30)
            .map_err(|_| EcdsaP256Error::InvalidPublicKey)?;
        if outer_len != r.remaining() {
            // Outer SEQUENCE must span the entire input (no trailing bytes).
            return Err(EcdsaP256Error::InvalidPublicKey);
        }

        // AlgorithmIdentifier ::= SEQUENCE { id-ecPublicKey, secp256r1 }
        let alg_len = r
            .tag_len(0x30)
            .map_err(|_| EcdsaP256Error::InvalidPublicKey)?;
        if alg_len != OID_EC_PUBLIC_KEY.len() + OID_SECP256R1.len() {
            return Err(EcdsaP256Error::InvalidPublicKey);
        }
        let alg = r
            .take(alg_len)
            .map_err(|_| EcdsaP256Error::InvalidPublicKey)?;
        if alg[..OID_EC_PUBLIC_KEY.len()] != OID_EC_PUBLIC_KEY
            || alg[OID_EC_PUBLIC_KEY.len()..] != OID_SECP256R1
        {
            return Err(EcdsaP256Error::InvalidPublicKey);
        }

        // subjectPublicKey BIT STRING: zero unused bits + 65-byte point.
        let bits_len = r
            .tag_len(0x03)
            .map_err(|_| EcdsaP256Error::InvalidPublicKey)?;
        if bits_len != 1 + 65 {
            return Err(EcdsaP256Error::InvalidPublicKey);
        }
        let bits = r
            .take(bits_len)
            .map_err(|_| EcdsaP256Error::InvalidPublicKey)?;
        if bits[0] != 0x00 || r.remaining() != 0 {
            return Err(EcdsaP256Error::InvalidPublicKey);
        }
        let point: [u8; 65] = bits[1..].try_into().unwrap();
        Self::from_uncompressed(&point)
    }
}

// ── Strict DER reader ───────────────────────────────────────────────────────

struct Der<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Der<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Der { buf, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    fn byte(&mut self) -> Result<u8, EcdsaP256Error> {
        let b = *self.buf.get(self.pos).ok_or(EcdsaP256Error::MalformedDer)?;
        self.pos += 1;
        Ok(b)
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], EcdsaP256Error> {
        if self.remaining() < n {
            return Err(EcdsaP256Error::MalformedDer);
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    /// Read an expected tag and its definite length, enforcing DER
    /// minimal-length rules. Both shapes this module parses (signatures
    /// ≤ ~72 bytes, SPKI = 91 bytes) fit in one length byte or the
    /// `0x81` long form, so anything longer is rejected outright.
    fn tag_len(&mut self, expected_tag: u8) -> Result<usize, EcdsaP256Error> {
        if self.byte()? != expected_tag {
            return Err(EcdsaP256Error::MalformedDer);
        }
        let l = self.byte()?;
        let len = match l {
            0x00..=0x7F => l as usize,
            0x81 => {
                let v = self.byte()? as usize;
                if v < 0x80 {
                    // Long form used where short form suffices: non-minimal.
                    return Err(EcdsaP256Error::MalformedDer);
                }
                v
            }
            _ => return Err(EcdsaP256Error::MalformedDer),
        };
        if len > self.remaining() {
            return Err(EcdsaP256Error::MalformedDer);
        }
        Ok(len)
    }

    /// Read a DER INTEGER as an unsigned scalar in `[1, n-1]`,
    /// enforcing: positive (no high bit without pad), minimal encoding
    /// (a `0x00` pad only when required), and the range check before
    /// any curve arithmetic can see the value.
    fn scalar(&mut self) -> Result<Scalar, EcdsaP256Error> {
        let len = self.tag_len(0x02)?;
        if len == 0 {
            return Err(EcdsaP256Error::MalformedDer);
        }
        let body = self.take(len)?;
        if body[0] & 0x80 != 0 {
            // Negative INTEGER: never a valid ECDSA scalar; must not be
            // reinterpreted as a large positive value.
            return Err(EcdsaP256Error::MalformedDer);
        }
        let digits = if body[0] == 0x00 {
            if len == 1 {
                // INTEGER 0 — in DER shape, but out of scalar range.
                return Err(EcdsaP256Error::ScalarOutOfRange);
            }
            if body[1] & 0x80 == 0 {
                // Redundant leading zero: non-minimal encoding.
                return Err(EcdsaP256Error::MalformedDer);
            }
            &body[1..]
        } else {
            body
        };
        if digits.len() > 32 {
            // Wider than 256 bits: necessarily ≥ n.
            return Err(EcdsaP256Error::ScalarOutOfRange);
        }
        let mut be = [0u8; 32];
        be[32 - digits.len()..].copy_from_slice(digits);
        let s = Scalar::from_bytes_be(&be).ok_or(EcdsaP256Error::ScalarOutOfRange)?;
        if s.is_zero() == 1 {
            return Err(EcdsaP256Error::ScalarOutOfRange);
        }
        Ok(s)
    }
}

/// Parse a DER ECDSA signature `SEQUENCE { INTEGER r, INTEGER s }`.
fn parse_der_signature(sig: &[u8]) -> Result<(Scalar, Scalar), EcdsaP256Error> {
    let mut rd = Der::new(sig);
    let seq_len = rd.tag_len(0x30)?;
    if seq_len != rd.remaining() {
        // Trailing garbage after the SEQUENCE, or length mismatch.
        return Err(EcdsaP256Error::MalformedDer);
    }
    let r = rd.scalar()?;
    let s = rd.scalar()?;
    if rd.remaining() != 0 {
        // Trailing bytes inside the SEQUENCE.
        return Err(EcdsaP256Error::MalformedDer);
    }
    Ok((r, s))
}

/// Verify an ECDSA-P256/SHA-256 signature per ANSI X9.62 §7.4.1 /
/// FIPS 186-5 §6.4.2:
///
/// 1. Parse `(r, s)` from DER; require both in `[1, n-1]`.
/// 2. `e` = SHA-256(message), interpreted as an integer mod `n`.
/// 3. `u1 = e·s⁻¹ mod n`, `u2 = r·s⁻¹ mod n`.
/// 4. `R = u1·G + u2·Q`.
/// 5. Reject if `R` is the point at infinity.
/// 6. Accept iff `R.x mod n == r`.
pub fn ecdsa_p256_verify(
    pk: &EcdsaP256PublicKey,
    message: &[u8],
    der_signature: &[u8],
) -> Result<(), EcdsaP256Error> {
    let (r, s) = parse_der_signature(der_signature)?;

    let e = Scalar::from_bytes_be_reduced(&sha256(message));
    let s_inv = s.invert();
    let u1 = e.mul(&s_inv);
    let u2 = r.mul(&s_inv);

    let g = ProjectivePoint::from_affine(&AffinePoint::generator());
    let q = ProjectivePoint::from_affine(&pk.point);
    let big_r = g.mul(&u1.to_raw_limbs()).add(&q.mul(&u2.to_raw_limbs()));

    // Point at infinity has no x-coordinate to compare (spec scenario:
    // never attempt to read one).
    let affine = big_r.to_affine().ok_or(EcdsaP256Error::InvalidSignature)?;

    // R.x is an element of GF(p); reduce its integer value mod n before
    // comparing (covers the r < p - n wraparound cases correctly).
    let x_mod_n = Scalar::from_bytes_be_reduced(&affine.x.to_bytes_be());
    if x_mod_n.ct_eq(&r) == 1 {
        Ok(())
    } else {
        Err(EcdsaP256Error::InvalidSignature)
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec::Vec;

    use super::super::ecdsa_p256_test_vectors::{CASES, KEYS_DER};
    use super::*;

    fn unhex(s: &str) -> Vec<u8> {
        assert!(s.len().is_multiple_of(2), "odd hex length");
        (0..s.len() / 2)
            .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).unwrap())
            .collect()
    }

    /// A known-good corpus entry (tc 3): key 0, msg "313233343030".
    const GOOD: usize = 2;

    fn good_key() -> EcdsaP256PublicKey {
        EcdsaP256PublicKey::from_spki_der(&unhex(KEYS_DER[CASES[GOOD].key])).unwrap()
    }

    // ── Wycheproof corpus replay ────────────────────────────────────────────

    #[test]
    fn wycheproof_corpus_replays_in_full() {
        assert_eq!(CASES.len(), 484, "corpus must not be truncated");
        let mut executed = 0usize;
        for case in CASES {
            // A key that fails to parse rejects every signature under it.
            let outcome = EcdsaP256PublicKey::from_spki_der(&unhex(KEYS_DER[case.key]))
                .and_then(|pk| ecdsa_p256_verify(&pk, &unhex(case.msg), &unhex(case.sig)));
            assert_eq!(
                outcome.is_ok(),
                case.valid,
                "Wycheproof tc_id {} expected valid={} got {:?}",
                case.tc_id,
                case.valid,
                outcome
            );
            executed += 1;
        }
        assert_eq!(executed, CASES.len(), "no vector may be skipped");
    }

    // ── Verify seams beyond the corpus ──────────────────────────────────────

    #[test]
    fn known_good_vector_verifies() {
        let case = &CASES[GOOD];
        assert!(case.valid);
        assert_eq!(
            ecdsa_p256_verify(&good_key(), &unhex(case.msg), &unhex(case.sig)),
            Ok(())
        );
    }

    #[test]
    fn different_message_fails() {
        let case = &CASES[GOOD];
        assert_eq!(
            ecdsa_p256_verify(&good_key(), b"a different message", &unhex(case.sig)),
            Err(EcdsaP256Error::InvalidSignature)
        );
    }

    #[test]
    fn high_s_counterpart_verifies() {
        // Flip a known-good signature's s to n - s (its high-s twin) and
        // re-encode; ECDSA malleability means it must still verify (the
        // proposal explicitly declines a low-s rule).
        let case = &CASES[GOOD];
        let sig = unhex(case.sig);
        let (_, s) = parse_der_signature(&sig).unwrap();
        let neg_s_bytes = Scalar::ZERO.sub(&s).to_bytes_be();
        let high = reencode_with_s(&sig, &neg_s_bytes);
        assert_eq!(
            ecdsa_p256_verify(&good_key(), &unhex(case.msg), &high),
            Ok(()),
            "high-s signature must not be rejected"
        );
    }

    /// Rebuild `SEQUENCE { r (copied), s (minimal DER of `s_be`) }`.
    fn reencode_with_s(orig: &[u8], s_be: &[u8; 32]) -> Vec<u8> {
        // Original layout: 30 LL 02 RL <r...> 02 SL <s...> (short lengths
        // are guaranteed for corpus entry GOOD).
        let r_len = orig[3] as usize;
        let r_tlv = &orig[2..4 + r_len];
        let digits = {
            let first = s_be.iter().position(|&b| b != 0).unwrap();
            &s_be[first..]
        };
        let pad = (digits[0] & 0x80) != 0;
        let s_len = digits.len() + pad as usize;
        let mut out = Vec::new();
        out.push(0x30);
        out.push((r_tlv.len() + 2 + s_len) as u8);
        out.extend_from_slice(r_tlv);
        out.push(0x02);
        out.push(s_len as u8);
        if pad {
            out.push(0x00);
        }
        out.extend_from_slice(digits);
        out
    }

    // ── DER signature parser refusal rules (task 3.5) ───────────────────────

    fn assert_parse_err(hex_sig: &str, err: EcdsaP256Error) {
        assert_eq!(
            parse_der_signature(&unhex(hex_sig)).unwrap_err(),
            err,
            "{hex_sig}"
        );
    }

    #[test]
    fn der_rejects_zero_scalars() {
        // r = 0, s = 1 and r = 1, s = 0.
        assert_parse_err("3006020100020101", EcdsaP256Error::ScalarOutOfRange);
        assert_parse_err("3006020101020100", EcdsaP256Error::ScalarOutOfRange);
    }

    #[test]
    fn der_rejects_scalars_at_or_above_n() {
        // r = n (33-byte padded INTEGER), s = 1.
        assert_parse_err(
            "3026022100ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551020101",
            EcdsaP256Error::ScalarOutOfRange,
        );
        // r = 2^256 - 1 (unpadded would be negative; padded form, > n).
        assert_parse_err(
            "3026022100ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff020101",
            EcdsaP256Error::ScalarOutOfRange,
        );
        // 33 significant digits: wider than any valid scalar.
        assert_parse_err(
            "302702220100ffffffff00000000ffffffffffffffffbce6faada7179e84f3b9cac2fc632551020101",
            EcdsaP256Error::ScalarOutOfRange,
        );
    }

    #[test]
    fn der_rejects_negative_integers() {
        // r's first content byte has the high bit set (negative value).
        assert_parse_err("3006020180020101", EcdsaP256Error::MalformedDer);
    }

    #[test]
    fn der_rejects_non_minimal_encodings() {
        // Redundant 0x00 pad before a byte without its high bit set.
        assert_parse_err("300702020001020101", EcdsaP256Error::MalformedDer);
        // Long-form length where short form suffices (Wycheproof tc 8 shape).
        assert_parse_err("308106020101020101", EcdsaP256Error::MalformedDer);
    }

    #[test]
    fn der_rejects_structural_damage() {
        for (sig, why) in [
            ("", "empty input"),
            ("30", "truncated header"),
            ("3006020101", "sequence shorter than declared"),
            ("31060201010 20101", "wrong outer tag"),
            ("3006030101020101", "wrong integer tag"),
            ("300602010102010100", "trailing garbage after sequence"),
            ("30070201010201 0100", "trailing byte inside sequence"),
            ("30020200", "empty integer"),
        ] {
            let cleaned: alloc::string::String = sig.split_whitespace().collect();
            assert!(
                parse_der_signature(&unhex(&cleaned)).is_err(),
                "must reject: {why}"
            );
        }
    }

    // ── SPKI parser refusal rules ───────────────────────────────────────────

    #[test]
    fn spki_parses_corpus_keys() {
        for (i, key) in KEYS_DER.iter().enumerate() {
            assert!(
                EcdsaP256PublicKey::from_spki_der(&unhex(key)).is_ok(),
                "corpus key group {i} must parse"
            );
        }
    }

    #[test]
    fn spki_rejects_wrong_oids() {
        let good = unhex(KEYS_DER[0]);
        // Corrupt the algorithm OID's last byte (id-ecPublicKey → bogus).
        let mut wrong_alg = good.clone();
        wrong_alg[12] ^= 0x01;
        assert_eq!(
            EcdsaP256PublicKey::from_spki_der(&wrong_alg).unwrap_err(),
            EcdsaP256Error::InvalidPublicKey
        );
        // Corrupt the curve OID's last byte (secp256r1 → another curve).
        let mut wrong_curve = good.clone();
        wrong_curve[22] ^= 0x01;
        assert_eq!(
            EcdsaP256PublicKey::from_spki_der(&wrong_curve).unwrap_err(),
            EcdsaP256Error::InvalidPublicKey
        );
    }

    #[test]
    fn spki_rejects_compressed_or_missized_points() {
        let good = unhex(KEYS_DER[0]);
        // 0x04 uncompressed marker → 0x02/0x03 compressed prefixes.
        for prefix in [0x02u8, 0x03u8] {
            let mut compressed = good.clone();
            compressed[26] = prefix;
            assert_eq!(
                EcdsaP256PublicKey::from_spki_der(&compressed).unwrap_err(),
                EcdsaP256Error::InvalidPublicKey
            );
        }
        // Truncated point (BIT STRING shortened).
        let truncated = &good[..good.len() - 1];
        assert!(EcdsaP256PublicKey::from_spki_der(truncated).is_err());
        // Raw-point API: wrong prefix byte.
        let mut raw = [0u8; 65];
        raw.copy_from_slice(&good[26..]);
        raw[0] = 0x02;
        assert_eq!(
            EcdsaP256PublicKey::from_uncompressed(&raw).unwrap_err(),
            EcdsaP256Error::InvalidPublicKey
        );
    }

    #[test]
    fn spki_rejects_off_curve_point() {
        // Corrupt one byte of Y so (X, Y) leaves the curve.
        let mut bad = unhex(KEYS_DER[0]);
        let last = bad.len() - 1;
        bad[last] ^= 0x01;
        assert_eq!(
            EcdsaP256PublicKey::from_spki_der(&bad).unwrap_err(),
            EcdsaP256Error::PointNotOnCurve
        );
    }

    #[test]
    fn spki_rejects_trailing_bytes() {
        let mut padded = unhex(KEYS_DER[0]);
        padded.push(0x00);
        assert!(EcdsaP256PublicKey::from_spki_der(&padded).is_err());
    }
}
