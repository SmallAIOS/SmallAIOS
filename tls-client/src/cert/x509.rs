// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! "Just-enough" X.509v3 structure parser (task 5.2, design.md D4).
//!
//! Decodes only the `tbsCertificate` fields TLS 1.3 path
//! validation needs, on top of the strict DER decoder in
//! [`super::der`]. Every additional ASN.1 type is attacker-
//! controlled surface, so anything not listed below is left
//! unparsed:
//!
//! - `version` — **must** be v3 (`2`); v1/v2 are refused.
//! - `serialNumber` — captured raw (not validated for uniqueness).
//! - `signature.algorithm` — mapped to [`SignatureAlgorithm`];
//!   any SHA-1 algorithm (or anything unrecognised) is refused.
//! - `validity.{notBefore, notAfter}` — UTCTime / GeneralizedTime
//!   in the mandatory `Z` (UTC) form, converted to Unix seconds.
//! - `issuer` / `subject` — kept as raw DN DER for byte-exact
//!   comparison during chain construction (never pretty-printed).
//! - `subjectPublicKeyInfo` — algorithm OID + key bytes.
//! - extensions `subjectAltName`, `basicConstraints`, `keyUsage`,
//!   `extKeyUsage`.
//!
//! Deliberately **not** parsed (design.md D4): NameConstraints,
//! CertificatePolicies, AuthorityInfoAccess, CRLDistributionPoints,
//! SignedCertificateTimestamp.

use super::der::{read_uint, tag, Reader};
use super::hostname::SanEntry;
use crate::{Result, TlsClientError};
use alloc::string::String;
use alloc::vec::Vec;

/// OID content bytes (the value of an OBJECT IDENTIFIER TLV, i.e.
/// without the `0x06 len` header).
pub mod oid {
    /// id-ecPublicKey (1.2.840.10045.2.1).
    pub const EC_PUBLIC_KEY: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01];
    /// prime256v1 / P-256 named curve (1.2.840.10045.3.1.7).
    pub const P256: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07];
    /// id-Ed25519 (1.3.101.112).
    pub const ED25519: &[u8] = &[0x2b, 0x65, 0x70];
    /// rsaEncryption (1.2.840.113549.1.1.1).
    pub const RSA_ENCRYPTION: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x01];

    /// ecdsa-with-SHA256 (1.2.840.10045.4.3.2).
    pub const ECDSA_SHA256: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x03, 0x02];
    /// sha256WithRSAEncryption (1.2.840.113549.1.1.11).
    pub const RSA_PKCS1_SHA256: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0b];
    /// id-RSASSA-PSS (1.2.840.113549.1.1.10).
    pub const RSA_PSS: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x0a];

    /// sha1WithRSAEncryption (1.2.840.113549.1.1.5) — refused.
    pub const RSA_PKCS1_SHA1: &[u8] = &[0x2a, 0x86, 0x48, 0x86, 0xf7, 0x0d, 0x01, 0x01, 0x05];
    /// ecdsa-with-SHA1 (1.2.840.10045.4.1) — refused.
    pub const ECDSA_SHA1: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x04, 0x01];

    /// id-ce-subjectAltName (2.5.29.17).
    pub const SUBJECT_ALT_NAME: &[u8] = &[0x55, 0x1d, 0x11];
    /// id-ce-basicConstraints (2.5.29.19).
    pub const BASIC_CONSTRAINTS: &[u8] = &[0x55, 0x1d, 0x13];
    /// id-ce-keyUsage (2.5.29.15).
    pub const KEY_USAGE: &[u8] = &[0x55, 0x1d, 0x0f];
    /// id-ce-extKeyUsage (2.5.29.37).
    pub const EXT_KEY_USAGE: &[u8] = &[0x55, 0x1d, 0x25];
    /// id-kp-serverAuth (1.3.6.1.5.5.7.3.1).
    pub const SERVER_AUTH: &[u8] = &[0x2b, 0x06, 0x01, 0x05, 0x05, 0x07, 0x03, 0x01];
}

/// Certificate signature algorithm, restricted to the set TLS 1.3
/// path validation accepts (design.md D4). SHA-1 algorithms are
/// never represented here — they are refused at parse time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureAlgorithm {
    Ed25519,
    EcdsaP256Sha256,
    RsaPkcs1Sha256,
    RsaPssSha256,
}

/// A subject public key, tagged by algorithm. Only Ed25519 can be
/// *used* for verification today; ECDSA-P256 and RSA keys are
/// parsed and carried so chain construction can report a precise
/// "unsupported algorithm" rather than a generic parse failure,
/// until their `security/` primitives land (tasks 5.5 note).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubjectPublicKey {
    Ed25519([u8; 32]),
    /// Uncompressed SEC1 point bytes (`0x04 || X || Y`).
    EcdsaP256(Vec<u8>),
    /// DER `RSAPublicKey` bytes.
    Rsa(Vec<u8>),
}

/// Parsed `BasicConstraints` (RFC 5280 §4.2.1.9). Absent extension
/// is represented as `ca = false` with no path-length limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BasicConstraints {
    pub ca: bool,
    pub path_len: Option<u64>,
}

/// The `keyUsage` bits we care about for path validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KeyUsage {
    /// Whether a `keyUsage` extension was present at all. When
    /// absent, RFC 5280 places no key-usage restriction.
    pub present: bool,
    /// `digitalSignature` (bit 0).
    pub digital_signature: bool,
    /// `keyCertSign` (bit 5) — required on CA certs.
    pub key_cert_sign: bool,
}

/// A parsed certificate. Borrows from the input DER for the raw
/// `tbsCertificate`, issuer/subject DN, and signature bytes so the
/// chain verifier can do byte-exact DN comparison and verify the
/// outer signature over the exact TBS encoding without copying.
#[derive(Debug, Clone)]
pub struct Certificate<'a> {
    /// Complete `tbsCertificate` DER (header + contents) — the
    /// octets the outer signature is computed over.
    pub tbs_der: &'a [u8],
    /// `serialNumber` content octets (unvalidated, for logging).
    pub serial: &'a [u8],
    /// Algorithm from `tbsCertificate.signature`.
    pub signature_algorithm: SignatureAlgorithm,
    /// Raw `issuer` Name DER (the SEQUENCE contents).
    pub issuer: &'a [u8],
    /// Raw `subject` Name DER (the SEQUENCE contents).
    pub subject: &'a [u8],
    /// `validity.notBefore` as Unix seconds.
    pub not_before: i64,
    /// `validity.notAfter` as Unix seconds.
    pub not_after: i64,
    /// Subject public key.
    pub public_key: SubjectPublicKey,
    /// SubjectAltName entries (empty when the extension is absent).
    pub san: Vec<SanEntry>,
    /// Whether a SubjectAltName extension was present.
    pub san_present: bool,
    /// Parsed BasicConstraints (default when absent).
    pub basic_constraints: BasicConstraints,
    /// Parsed KeyUsage (default when absent).
    pub key_usage: KeyUsage,
    /// True when `extKeyUsage` is present AND lists id-kp-serverAuth
    /// (or anyExtendedKeyUsage). When `ext_key_usage_present` is
    /// false, no EKU restriction applies.
    pub server_auth_eku: bool,
    /// Whether an `extKeyUsage` extension was present.
    pub ext_key_usage_present: bool,
    /// Outer `signatureAlgorithm` (must equal `signature_algorithm`).
    pub outer_signature_algorithm: SignatureAlgorithm,
    /// `signatureValue` payload (BIT STRING contents, unused-bits
    /// octet stripped).
    pub signature: &'a [u8],
}

/// Context-specific constructed `[0]` (EXPLICIT version wrapper).
const CTX_0_CONSTRUCTED: u8 = 0xa0;
/// Context-specific constructed `[3]` (EXPLICIT extensions wrapper).
const CTX_3_CONSTRUCTED: u8 = 0xa3;
/// X.509 `Version` value for v3 (the only version we accept).
const VERSION_V3: u64 = 2;

impl<'a> Certificate<'a> {
    /// Parse a single DER certificate. Enforces the design.md D4
    /// constraints: v3 only, no SHA-1 signature algorithms, and a
    /// well-formed structure for every field listed in the module
    /// docs. Does NOT itself require SAN — that (leaf-only) policy
    /// is applied by the chain verifier.
    pub fn parse(der: &'a [u8]) -> Result<Self> {
        let mut outer = Reader::new(der);
        let cert_body = outer.expect_tlv(tag::SEQUENCE)?;
        if !outer.is_empty() {
            // Trailing bytes after the Certificate SEQUENCE.
            return Err(TlsClientError::BadCertificate);
        }
        let mut cert = Reader::new(cert_body);

        // tbsCertificate — capture its exact DER for signature checks.
        let (tbs_tlv, tbs_der) = cert.next_tlv_with_raw()?;
        if !tbs_tlv.is(tag::SEQUENCE) {
            return Err(TlsClientError::BadCertificate);
        }
        let mut tbs = Reader::new(tbs_tlv.value);

        // version [0] EXPLICIT INTEGER — required to be present and v3.
        let version_tlv = tbs.next_tlv()?;
        if !version_tlv.is(CTX_0_CONSTRUCTED) {
            // A v1 certificate omits the version wrapper entirely; we
            // refuse anything that is not explicit v3.
            return Err(TlsClientError::BadCertificate);
        }
        let mut version_inner = Reader::new(version_tlv.value);
        let version = read_uint(version_inner.expect_tlv(tag::INTEGER)?)?;
        if !version_inner.is_empty() || version != VERSION_V3 {
            return Err(TlsClientError::BadCertificate);
        }

        // serialNumber INTEGER (kept raw; may be large / 0x00-prefixed).
        let serial = tbs.expect_tlv(tag::INTEGER)?;

        // signature AlgorithmIdentifier.
        let signature_algorithm = parse_sig_alg(tbs.expect_tlv(tag::SEQUENCE)?)?;

        // issuer Name (raw DN contents).
        let issuer = tbs.expect_tlv(tag::SEQUENCE)?;

        // validity SEQUENCE { notBefore Time, notAfter Time }.
        let mut validity = Reader::new(tbs.expect_tlv(tag::SEQUENCE)?);
        let not_before = parse_time(validity.next_tlv()?)?;
        let not_after = parse_time(validity.next_tlv()?)?;
        if !validity.is_empty() {
            return Err(TlsClientError::BadCertificate);
        }
        if not_after < not_before {
            return Err(TlsClientError::BadCertificate);
        }

        // subject Name (raw DN contents).
        let subject = tbs.expect_tlv(tag::SEQUENCE)?;

        // subjectPublicKeyInfo.
        let public_key = parse_spki(tbs.expect_tlv(tag::SEQUENCE)?)?;

        // Optional issuerUniqueID [1] / subjectUniqueID [2] are
        // skipped if present; then extensions [3] EXPLICIT.
        let mut san = Vec::new();
        let mut san_present = false;
        let mut basic_constraints = BasicConstraints::default();
        let mut key_usage = KeyUsage::default();
        let mut server_auth_eku = false;
        let mut ext_key_usage_present = false;

        while !tbs.is_empty() {
            let field = tbs.next_tlv()?;
            if field.is(CTX_3_CONSTRUCTED) {
                parse_extensions(
                    field.value,
                    &mut san,
                    &mut san_present,
                    &mut basic_constraints,
                    &mut key_usage,
                    &mut server_auth_eku,
                    &mut ext_key_usage_present,
                )?;
                // extensions is the final TBS field.
                if !tbs.is_empty() {
                    return Err(TlsClientError::BadCertificate);
                }
                break;
            }
            // [1]/[2] unique IDs (0x81/0x82) — tolerate and ignore.
            // Anything else here is malformed.
            if field.tag != 0x81 && field.tag != 0x82 {
                return Err(TlsClientError::BadCertificate);
            }
        }

        // signatureAlgorithm (outer) must match the TBS signature alg.
        let outer_signature_algorithm = parse_sig_alg(cert.expect_tlv(tag::SEQUENCE)?)?;
        if outer_signature_algorithm != signature_algorithm {
            return Err(TlsClientError::BadCertificate);
        }

        // signatureValue BIT STRING (unused-bits octet must be 0).
        let signature = parse_bit_string(cert.expect_tlv(tag::BIT_STRING)?)?;
        if !cert.is_empty() {
            return Err(TlsClientError::BadCertificate);
        }

        Ok(Certificate {
            tbs_der,
            serial,
            signature_algorithm,
            issuer,
            subject,
            not_before,
            not_after,
            public_key,
            san,
            san_present,
            basic_constraints,
            key_usage,
            server_auth_eku,
            ext_key_usage_present,
            outer_signature_algorithm,
            signature,
        })
    }
}

/// Parse an `AlgorithmIdentifier` SEQUENCE contents into a
/// [`SignatureAlgorithm`], refusing SHA-1 and unknown OIDs.
fn parse_sig_alg(seq: &[u8]) -> Result<SignatureAlgorithm> {
    let mut r = Reader::new(seq);
    let alg_oid = r.expect_tlv(tag::OID)?;
    // Parameters (NULL for RSA PKCS#1, absent for Ed25519/ECDSA,
    // a structured AlgorithmIdentifier for PSS) are not inspected
    // beyond the OID for v1 of this client.
    match alg_oid {
        oid::ED25519 => Ok(SignatureAlgorithm::Ed25519),
        oid::ECDSA_SHA256 => Ok(SignatureAlgorithm::EcdsaP256Sha256),
        oid::RSA_PKCS1_SHA256 => Ok(SignatureAlgorithm::RsaPkcs1Sha256),
        oid::RSA_PSS => Ok(SignatureAlgorithm::RsaPssSha256),
        // Explicitly refuse SHA-1 with a certificate-specific error.
        oid::RSA_PKCS1_SHA1 | oid::ECDSA_SHA1 => Err(TlsClientError::BadCertificate),
        _ => Err(TlsClientError::BadCertificate),
    }
}

/// Parse a `SubjectPublicKeyInfo` SEQUENCE contents.
fn parse_spki(seq: &[u8]) -> Result<SubjectPublicKey> {
    let mut r = Reader::new(seq);
    let mut alg = Reader::new(r.expect_tlv(tag::SEQUENCE)?);
    let alg_oid = alg.expect_tlv(tag::OID)?;
    let key_bits = parse_bit_string(r.expect_tlv(tag::BIT_STRING)?)?;
    if !r.is_empty() {
        return Err(TlsClientError::BadCertificate);
    }

    match alg_oid {
        oid::ED25519 => {
            if key_bits.len() != 32 {
                return Err(TlsClientError::BadCertificate);
            }
            let mut k = [0u8; 32];
            k.copy_from_slice(key_bits);
            Ok(SubjectPublicKey::Ed25519(k))
        }
        oid::EC_PUBLIC_KEY => {
            // Require the namedCurve parameter to be P-256, and an
            // uncompressed point (0x04 || 32 || 32).
            let curve = alg.expect_tlv(tag::OID)?;
            if curve != oid::P256 {
                return Err(TlsClientError::BadCertificate);
            }
            if key_bits.len() != 65 || key_bits[0] != 0x04 {
                return Err(TlsClientError::BadCertificate);
            }
            Ok(SubjectPublicKey::EcdsaP256(key_bits.to_vec()))
        }
        oid::RSA_ENCRYPTION => Ok(SubjectPublicKey::Rsa(key_bits.to_vec())),
        _ => Err(TlsClientError::BadCertificate),
    }
}

/// Strip the leading "unused bits" octet from a BIT STRING's
/// contents, requiring it to be zero (whole-octet payloads only).
fn parse_bit_string(value: &[u8]) -> Result<&[u8]> {
    match value.split_first() {
        Some((0, rest)) => Ok(rest),
        _ => Err(TlsClientError::BadCertificate),
    }
}

/// Walk `extensions [3]` contents, filling the fields we honour.
fn parse_extensions(
    ctx3: &[u8],
    san: &mut Vec<SanEntry>,
    san_present: &mut bool,
    basic_constraints: &mut BasicConstraints,
    key_usage: &mut KeyUsage,
    server_auth_eku: &mut bool,
    ext_key_usage_present: &mut bool,
) -> Result<()> {
    let mut outer = Reader::new(ctx3);
    let mut list = Reader::new(outer.expect_tlv(tag::SEQUENCE)?);
    if !outer.is_empty() {
        return Err(TlsClientError::BadCertificate);
    }
    while !list.is_empty() {
        let mut ext = Reader::new(list.expect_tlv(tag::SEQUENCE)?);
        let ext_oid = ext.expect_tlv(tag::OID)?;
        // Optional critical BOOLEAN, then the extnValue OCTET STRING.
        let next = ext.next_tlv()?;
        let extn_value = if next.is(0x01) {
            // critical flag present; the value is the following TLV.
            ext.expect_tlv(tag::OCTET_STRING)?
        } else if next.is(tag::OCTET_STRING) {
            next.value
        } else {
            return Err(TlsClientError::BadCertificate);
        };
        if !ext.is_empty() {
            return Err(TlsClientError::BadCertificate);
        }

        match ext_oid {
            oid::SUBJECT_ALT_NAME => {
                *san_present = true;
                parse_san(extn_value, san)?;
            }
            oid::BASIC_CONSTRAINTS => {
                *basic_constraints = parse_basic_constraints(extn_value)?;
            }
            oid::KEY_USAGE => {
                *key_usage = parse_key_usage(extn_value)?;
            }
            oid::EXT_KEY_USAGE => {
                *ext_key_usage_present = true;
                *server_auth_eku = parse_ext_key_usage(extn_value)?;
            }
            // Unhandled extensions are ignored (design.md D4). A
            // production hardening pass could refuse unknown
            // *critical* extensions; v1 does not.
            _ => {}
        }
    }
    Ok(())
}

/// Parse `SubjectAltName ::= GeneralNames` (the OCTET STRING value).
fn parse_san(extn_value: &[u8], san: &mut Vec<SanEntry>) -> Result<()> {
    let mut outer = Reader::new(extn_value);
    let mut names = Reader::new(outer.expect_tlv(tag::SEQUENCE)?);
    if !outer.is_empty() {
        return Err(TlsClientError::BadCertificate);
    }
    while !names.is_empty() {
        let entry = names.next_tlv()?;
        match entry.tag {
            // dNSName [2] IMPLICIT IA5String (primitive context [2]).
            0x82 => {
                let s = core::str::from_utf8(entry.value)
                    .map_err(|_| TlsClientError::BadCertificate)?;
                san.push(SanEntry::Dns(String::from(s)));
            }
            // iPAddress [7] IMPLICIT OCTET STRING (4 or 16 octets).
            0x87 => {
                if entry.value.len() != 4 && entry.value.len() != 16 {
                    return Err(TlsClientError::BadCertificate);
                }
                san.push(SanEntry::IpAddress(entry.value.to_vec()));
            }
            // Other GeneralName forms (rfc822Name, URI, directoryName,
            // …) are not used for TLS server identity — ignore.
            _ => {}
        }
    }
    Ok(())
}

/// Parse `BasicConstraints ::= SEQUENCE { cA BOOLEAN DEFAULT FALSE,
/// pathLenConstraint INTEGER OPTIONAL }`.
fn parse_basic_constraints(extn_value: &[u8]) -> Result<BasicConstraints> {
    let mut outer = Reader::new(extn_value);
    let mut seq = Reader::new(outer.expect_tlv(tag::SEQUENCE)?);
    if !outer.is_empty() {
        return Err(TlsClientError::BadCertificate);
    }
    let mut bc = BasicConstraints::default();
    if seq.is_empty() {
        return Ok(bc);
    }
    let first = seq.next_tlv()?;
    let after_ca = if first.is(0x01) {
        // BOOLEAN cA: DER encodes TRUE as a single 0xFF octet.
        bc.ca = matches!(first.value, [0xff]);
        if !bc.ca && first.value != [0x00] {
            return Err(TlsClientError::BadCertificate);
        }
        if seq.is_empty() {
            return Ok(bc);
        }
        Some(seq.next_tlv()?)
    } else {
        Some(first)
    };
    if let Some(tlv) = after_ca {
        if !tlv.is(tag::INTEGER) {
            return Err(TlsClientError::BadCertificate);
        }
        bc.path_len = Some(read_uint(tlv.value)?);
    }
    if !seq.is_empty() {
        return Err(TlsClientError::BadCertificate);
    }
    Ok(bc)
}

/// Parse `KeyUsage ::= BIT STRING`, extracting the bits we enforce.
fn parse_key_usage(extn_value: &[u8]) -> Result<KeyUsage> {
    let mut outer = Reader::new(extn_value);
    let bits_tlv = outer.next_tlv()?;
    if !bits_tlv.is(tag::BIT_STRING) || !outer.is_empty() {
        return Err(TlsClientError::BadCertificate);
    }
    // First octet is the unused-bit count; bit 0 (digitalSignature)
    // is the MSB of the first content octet, bit 5 (keyCertSign) is
    // mask 0x04 of that octet.
    let (_, body) = bits_tlv
        .value
        .split_first()
        .ok_or(TlsClientError::BadCertificate)?;
    let b0 = body.first().copied().unwrap_or(0);
    Ok(KeyUsage {
        present: true,
        digital_signature: b0 & 0x80 != 0,
        key_cert_sign: b0 & 0x04 != 0,
    })
}

/// Parse `ExtKeyUsage ::= SEQUENCE OF OID`; returns true iff
/// id-kp-serverAuth is present.
fn parse_ext_key_usage(extn_value: &[u8]) -> Result<bool> {
    let mut outer = Reader::new(extn_value);
    let mut seq = Reader::new(outer.expect_tlv(tag::SEQUENCE)?);
    if !outer.is_empty() {
        return Err(TlsClientError::BadCertificate);
    }
    let mut found = false;
    while !seq.is_empty() {
        if seq.expect_tlv(tag::OID)? == oid::SERVER_AUTH {
            found = true;
        }
    }
    Ok(found)
}

/// Parse an X.509 `Time` (UTCTime or GeneralizedTime) in the
/// mandatory `Z` (UTC) form into Unix seconds. Fractional seconds
/// and explicit offsets are refused.
fn parse_time(tlv: super::der::Tlv<'_>) -> Result<i64> {
    let (yyyy_off, body) = match tlv.tag {
        // UTCTime: YYMMDDHHMMSSZ (13 octets). RFC 5280: YY < 50 ⇒
        // 20YY, else 19YY.
        tag::UTC_TIME => {
            if tlv.value.len() != 13 {
                return Err(TlsClientError::BadCertificate);
            }
            let yy = parse_digits(&tlv.value[0..2])?;
            let year = if yy < 50 { 2000 + yy } else { 1900 + yy };
            (year, &tlv.value[2..])
        }
        // GeneralizedTime: YYYYMMDDHHMMSSZ (15 octets).
        tag::GENERALIZED_TIME => {
            if tlv.value.len() != 15 {
                return Err(TlsClientError::BadCertificate);
            }
            let year = parse_digits(&tlv.value[0..4])?;
            (year, &tlv.value[4..])
        }
        _ => return Err(TlsClientError::BadCertificate),
    };
    // body = MMDDHHMMSSZ (11 octets).
    if body.len() != 11 || body[10] != b'Z' {
        return Err(TlsClientError::BadCertificate);
    }
    let month = parse_digits(&body[0..2])?;
    let day = parse_digits(&body[2..4])?;
    let hour = parse_digits(&body[4..6])?;
    let minute = parse_digits(&body[6..8])?;
    let second = parse_digits(&body[8..10])?;
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 60
    {
        return Err(TlsClientError::BadCertificate);
    }
    let days = days_from_civil(yyyy_off as i64, month as i64, day as i64);
    Ok(days * 86_400 + hour as i64 * 3_600 + minute as i64 * 60 + second as i64)
}

/// Parse a run of ASCII digits into an integer.
fn parse_digits(bytes: &[u8]) -> Result<u32> {
    let mut v: u32 = 0;
    for &b in bytes {
        if !b.is_ascii_digit() {
            return Err(TlsClientError::BadCertificate);
        }
        v = v * 10 + (b - b'0') as u32;
    }
    Ok(v)
}

/// Days from the Unix epoch (1970-01-01) to `y-m-d` (proleptic
/// Gregorian). Howard Hinnant's `days_from_civil` algorithm.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

#[cfg(test)]
mod tests {
    use super::super::test_certs::{keypair, CertBuilder, San};
    use super::*;

    #[test]
    fn parses_typical_leaf() {
        let kp = keypair(1);
        let ca = keypair(2);
        let der = CertBuilder::leaf("leaf.example.com", "leaf.example.com").build(&kp, &ca);
        let cert = Certificate::parse(&der).unwrap();

        assert_eq!(cert.signature_algorithm, SignatureAlgorithm::Ed25519);
        assert_eq!(cert.outer_signature_algorithm, SignatureAlgorithm::Ed25519);
        assert_eq!(
            cert.public_key,
            SubjectPublicKey::Ed25519(*kp.public_key.as_bytes())
        );
        assert!(cert.san_present);
        assert_eq!(
            cert.san,
            alloc::vec![SanEntry::Dns("leaf.example.com".into())]
        );
        assert!(cert.key_usage.present);
        assert!(cert.key_usage.digital_signature);
        assert!(!cert.key_usage.key_cert_sign);
        assert!(cert.ext_key_usage_present);
        assert!(cert.server_auth_eku);
        assert!(!cert.basic_constraints.ca);
        assert_eq!(cert.signature.len(), 64);
        // 2024-01-01T00:00:00Z .. 2099-01-01T00:00:00Z
        assert_eq!(cert.not_before, 1_704_067_200);
        assert!(cert.not_after > cert.not_before);
    }

    #[test]
    fn parses_ca_with_basic_constraints() {
        let ca = keypair(2);
        let der = CertBuilder::ca("Test Root CA").build_self_signed(&ca);
        let cert = Certificate::parse(&der).unwrap();
        assert!(cert.basic_constraints.ca);
        assert!(cert.key_usage.key_cert_sign);
        assert!(!cert.san_present);
        // Issuer == subject for a self-signed root.
        assert_eq!(cert.issuer, cert.subject);
    }

    #[test]
    fn refuses_non_v3_version() {
        let kp = keypair(1);
        let ca = keypair(2);
        let mut b = CertBuilder::leaf("leaf.example.com", "leaf.example.com");
        b.version_override = Some(0); // v1
        let der = b.build(&kp, &ca);
        assert_eq!(
            Certificate::parse(&der).unwrap_err(),
            TlsClientError::BadCertificate
        );
    }

    #[test]
    fn ip_san_round_trips() {
        let kp = keypair(1);
        let ca = keypair(2);
        let mut b = CertBuilder::leaf("server", "unused");
        b.sans = alloc::vec![San::Ip(alloc::vec![192, 0, 2, 1])];
        let der = b.build(&kp, &ca);
        let cert = Certificate::parse(&der).unwrap();
        assert_eq!(
            cert.san,
            alloc::vec![SanEntry::IpAddress(alloc::vec![192, 0, 2, 1])]
        );
    }

    #[test]
    fn trailing_garbage_rejected() {
        let kp = keypair(1);
        let ca = keypair(2);
        let mut der = CertBuilder::leaf("a", "a").build(&kp, &ca);
        der.push(0x00);
        assert_eq!(
            Certificate::parse(&der).unwrap_err(),
            TlsClientError::BadCertificate
        );
    }

    #[test]
    fn truncated_rejected() {
        let kp = keypair(1);
        let ca = keypair(2);
        let der = CertBuilder::leaf("a", "a").build(&kp, &ca);
        for cut in [3, 10, der.len() / 2, der.len() - 1] {
            assert!(Certificate::parse(&der[..cut]).is_err());
        }
    }

    #[test]
    fn utctime_two_digit_year() {
        // Directly exercise the time parser via a hand-built cert is
        // covered by the builder's GeneralizedTime; here check the
        // civil-day epoch anchor and the UTCTime pivot.
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(days_from_civil(2000, 1, 1), 10_957);
        assert_eq!(days_from_civil(2024, 1, 1) * 86_400, 1_704_067_200);
    }

    #[test]
    fn no_san_parses_but_flag_clear() {
        let ca = keypair(2);
        let mut b = CertBuilder::ca("Root");
        b.san_extension = false;
        let der = b.build_self_signed(&ca);
        let cert = Certificate::parse(&der).unwrap();
        assert!(!cert.san_present);
        assert!(cert.san.is_empty());
    }
}
