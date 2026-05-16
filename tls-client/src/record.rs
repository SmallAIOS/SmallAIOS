// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! TLS 1.3 record layer (RFC 8446 § 5).
//!
//! A TLS record is the framing unit on the wire: 5-byte
//! header (type + legacy_version + length) followed by a
//! body whose interpretation depends on whether the
//! handshake-traffic keys have been derived yet.
//!
//! Before the keys exist, the body is a plaintext fragment
//! (TLSPlaintext). After, every body is an AEAD-sealed
//! `TLSInnerPlaintext` (TLSCiphertext) — the inner
//! type byte travels inside the AEAD plaintext so a
//! middlebox cannot tell whether a record is handshake,
//! alert, or application data without breaking the AEAD.
//!
//! Phase 3 of `tls-tcp-client-v1`. Phase 4 drives this
//! module via the handshake state machine.

use crate::{Result, TlsClientError};
use alloc::vec;
use alloc::vec::Vec;

/// Length of the fixed TLS record header.
pub const RECORD_HEADER_LEN: usize = 5;

/// Maximum plaintext fragment length (RFC 8446 § 5.1).
pub const MAX_PLAINTEXT_LEN: usize = 16_384; // 2^14

/// Maximum ciphertext fragment length (RFC 8446 § 5.2):
/// 2^14 plaintext + 1 inner-type byte + ≤ 255 padding +
/// 16-byte AEAD tag. We cap at 2^14 + 256.
pub const MAX_CIPHERTEXT_LEN: usize = 16_640; // 2^14 + 256

/// On-the-wire legacy record version: TLS 1.2 (`0x0303`)
/// per RFC 8446 § 5.1 compatibility rule.
pub const LEGACY_VERSION: u16 = 0x0303;

/// Length of the AEAD tag for every cipher suite this
/// crate supports.
pub const AEAD_TAG_LEN: usize = 16;

/// Length of the per-record nonce / IV (RFC 8446 § 5.3).
pub const NONCE_LEN: usize = 12;

/// RFC 8446 § 5.1 ContentType byte values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ContentType {
    /// Sentinel for "no content type seen yet"; never on the wire.
    Invalid = 0,
    /// Legacy ChangeCipherSpec; ignored after handshake.
    ChangeCipherSpec = 20,
    /// TLS alert record.
    Alert = 21,
    /// Handshake protocol record.
    Handshake = 22,
    /// Application data record (post-handshake).
    ApplicationData = 23,
}

impl ContentType {
    /// Decode the on-the-wire byte. Unknown values are
    /// refused with `BadRecord`.
    pub fn from_u8(b: u8) -> Result<Self> {
        match b {
            0 => Ok(Self::Invalid),
            20 => Ok(Self::ChangeCipherSpec),
            21 => Ok(Self::Alert),
            22 => Ok(Self::Handshake),
            23 => Ok(Self::ApplicationData),
            _ => Err(TlsClientError::BadRecord),
        }
    }
}

/// Parsed TLS record header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecordHeader {
    pub content_type: ContentType,
    pub legacy_version: u16,
    pub length: u16,
}

impl RecordHeader {
    /// Parse a 5-byte record header. Enforces the
    /// `length` cap BEFORE the caller allocates any
    /// buffer keyed by it.
    pub fn parse(buf: &[u8]) -> Result<Self> {
        if buf.len() < RECORD_HEADER_LEN {
            return Err(TlsClientError::BadRecord);
        }
        let content_type = ContentType::from_u8(buf[0])?;
        let legacy_version = u16::from_be_bytes([buf[1], buf[2]]);
        let length = u16::from_be_bytes([buf[3], buf[4]]);
        // RFC 8446 § 5.1 caps:
        //   plaintext fragments: ≤ 2^14
        //   ciphertext fragments: ≤ 2^14 + 256
        // We use the looser cap here; the per-direction caller
        // tightens for known plaintext records.
        if (length as usize) > MAX_CIPHERTEXT_LEN {
            return Err(TlsClientError::BadRecord);
        }
        Ok(Self {
            content_type,
            legacy_version,
            length,
        })
    }

    /// Serialize the 5-byte header.
    pub fn encode(&self) -> [u8; RECORD_HEADER_LEN] {
        let mut out = [0u8; RECORD_HEADER_LEN];
        out[0] = self.content_type as u8;
        let v = self.legacy_version.to_be_bytes();
        out[1] = v[0];
        out[2] = v[1];
        let l = self.length.to_be_bytes();
        out[3] = l[0];
        out[4] = l[1];
        out
    }

    /// Reject any record whose `legacy_version` is not the
    /// TLS 1.3 compatibility value `0x0303`. Per RFC 8446
    /// § 5.1, every post-handshake record MUST carry this.
    pub fn enforce_legacy_version(&self) -> Result<()> {
        if self.legacy_version != LEGACY_VERSION {
            return Err(TlsClientError::Version);
        }
        Ok(())
    }
}

/// Cipher suite negotiated for the connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CipherSuite {
    /// `TLS_AES_256_GCM_SHA384` (RFC 8446 § B.4). Strongest
    /// classical AEAD; uses
    /// `security::crypto::aes_gcm` (AES-256 + GCM).
    Aes256GcmSha384,
    /// `TLS_CHACHA20_POLY1305_SHA256` (RFC 8446 § B.4).
    /// Constant-time on any 32-bit machine; preferred when
    /// AES hardware is unavailable.
    ChaCha20Poly1305Sha256,
}

impl CipherSuite {
    /// On-the-wire 2-byte code.
    pub fn wire_value(self) -> u16 {
        match self {
            Self::Aes256GcmSha384 => 0x1302,
            Self::ChaCha20Poly1305Sha256 => 0x1303,
        }
    }

    /// Decode the on-the-wire code. Refuses any suite the
    /// client did not advertise (per `tls-client-handshake`
    /// spec).
    pub fn from_wire(code: u16) -> Result<Self> {
        match code {
            0x1302 => Ok(Self::Aes256GcmSha384),
            0x1303 => Ok(Self::ChaCha20Poly1305Sha256),
            _ => Err(TlsClientError::BadHandshake),
        }
    }
}

/// Derive the per-record nonce by XOR-ing the static IV
/// with the BE-encoded 64-bit sequence number, left-padded
/// to 12 bytes. RFC 8446 § 5.3.
pub fn per_record_nonce(static_iv: &[u8; NONCE_LEN], seq: u64) -> [u8; NONCE_LEN] {
    let seq_bytes = seq.to_be_bytes();
    let mut nonce = *static_iv;
    // The sequence number's 8 bytes XOR into the LAST 8
    // bytes of the IV; the first 4 stay as-is.
    for i in 0..8 {
        nonce[NONCE_LEN - 8 + i] ^= seq_bytes[i];
    }
    nonce
}

/// Build the AEAD additional_data per RFC 8446 § 5.2: the
/// 5-byte record header verbatim (`opaque_type ||
/// legacy_record_version || length`).
fn aead_additional_data(length: u16) -> [u8; RECORD_HEADER_LEN] {
    RecordHeader {
        content_type: ContentType::ApplicationData,
        legacy_version: LEGACY_VERSION,
        length,
    }
    .encode()
}

/// Build a plaintext record (pre-handshake-keys).
/// Used for ClientHello and the optional `ChangeCipherSpec`
/// echo only.
pub fn build_plaintext_record(content_type: ContentType, fragment: &[u8]) -> Result<Vec<u8>> {
    if fragment.len() > MAX_PLAINTEXT_LEN {
        return Err(TlsClientError::BadRecord);
    }
    let mut out = Vec::with_capacity(RECORD_HEADER_LEN + fragment.len());
    let header = RecordHeader {
        content_type,
        legacy_version: LEGACY_VERSION,
        length: fragment.len() as u16,
    };
    out.extend_from_slice(&header.encode());
    out.extend_from_slice(fragment);
    Ok(out)
}

/// Seal a TLSInnerPlaintext into a TLSCiphertext record.
///
/// The inner plaintext is built per RFC 8446 § 5.2:
/// `fragment || ContentType || zeros`. We do not emit any
/// padding bytes in v1 — that's an optional feature.
///
/// Output: full record bytes (5-byte header + encrypted
/// fragment including AEAD tag).
pub fn seal(
    suite: CipherSuite,
    key: &[u8],
    static_iv: &[u8; NONCE_LEN],
    seq: u64,
    inner_type: ContentType,
    inner_plaintext: &[u8],
) -> Result<Vec<u8>> {
    if inner_plaintext.len() > MAX_PLAINTEXT_LEN {
        return Err(TlsClientError::BadRecord);
    }
    // TLSInnerPlaintext = fragment || ContentType (we emit
    // zero padding bytes).
    let inner_len = inner_plaintext.len() + 1;
    let cipher_len = inner_len + AEAD_TAG_LEN;
    if cipher_len > MAX_CIPHERTEXT_LEN {
        return Err(TlsClientError::BadRecord);
    }
    let mut buf = Vec::with_capacity(inner_len);
    buf.extend_from_slice(inner_plaintext);
    buf.push(inner_type as u8);

    let aad = aead_additional_data(cipher_len as u16);
    let nonce = per_record_nonce(static_iv, seq);
    let tag = match suite {
        CipherSuite::ChaCha20Poly1305Sha256 => seal_chacha20(key, &nonce, &aad, &mut buf)?,
        CipherSuite::Aes256GcmSha384 => seal_aes256(key, &nonce, &aad, &mut buf)?,
    };

    let mut out = Vec::with_capacity(RECORD_HEADER_LEN + cipher_len);
    out.extend_from_slice(&aad);
    out.extend_from_slice(&buf);
    out.extend_from_slice(&tag);
    Ok(out)
}

/// Open a TLSCiphertext record. Input is the full record
/// bytes starting at the 5-byte header. Returns
/// `(inner_type, plaintext_fragment)` — the trailing
/// ContentType byte and any padding have been stripped.
pub fn open(
    suite: CipherSuite,
    key: &[u8],
    static_iv: &[u8; NONCE_LEN],
    seq: u64,
    record_bytes: &[u8],
) -> Result<(ContentType, Vec<u8>)> {
    if record_bytes.len() < RECORD_HEADER_LEN {
        return Err(TlsClientError::BadRecord);
    }
    let header = RecordHeader::parse(&record_bytes[..RECORD_HEADER_LEN])?;
    header.enforce_legacy_version()?;
    // RFC 8446 § 5.2: encrypted records carry the outer
    // content_type = application_data.
    if header.content_type != ContentType::ApplicationData {
        return Err(TlsClientError::BadRecord);
    }
    let total = RECORD_HEADER_LEN + header.length as usize;
    if record_bytes.len() < total {
        return Err(TlsClientError::BadRecord);
    }
    if (header.length as usize) < AEAD_TAG_LEN + 1 {
        return Err(TlsClientError::BadRecord);
    }
    let body_len = header.length as usize - AEAD_TAG_LEN;
    let aad = aead_additional_data(header.length);
    let nonce = per_record_nonce(static_iv, seq);
    let mut body = record_bytes[RECORD_HEADER_LEN..RECORD_HEADER_LEN + body_len].to_vec();
    let mut tag = [0u8; AEAD_TAG_LEN];
    tag.copy_from_slice(
        &record_bytes[RECORD_HEADER_LEN + body_len..RECORD_HEADER_LEN + body_len + AEAD_TAG_LEN],
    );
    match suite {
        CipherSuite::ChaCha20Poly1305Sha256 => open_chacha20(key, &nonce, &aad, &mut body, &tag)?,
        CipherSuite::Aes256GcmSha384 => open_aes256(key, &nonce, &aad, &mut body, &tag)?,
    }
    // Strip trailing zero padding to find the inner ContentType.
    let mut end = body.len();
    while end > 0 && body[end - 1] == 0 {
        end -= 1;
    }
    if end == 0 {
        return Err(TlsClientError::BadRecord);
    }
    let inner_type = ContentType::from_u8(body[end - 1])?;
    body.truncate(end - 1);
    Ok((inner_type, body))
}

fn seal_chacha20(
    key: &[u8],
    nonce: &[u8; NONCE_LEN],
    aad: &[u8],
    buf: &mut [u8],
) -> Result<[u8; AEAD_TAG_LEN]> {
    use smallaios_security::crypto::chacha20_poly1305 as aead;
    if key.len() != aead::KEY_LEN {
        return Err(TlsClientError::KeyExchange);
    }
    let mut key_arr = [0u8; aead::KEY_LEN];
    key_arr.copy_from_slice(key);
    Ok(aead::seal(&key_arr, nonce, aad, buf))
}

fn open_chacha20(
    key: &[u8],
    nonce: &[u8; NONCE_LEN],
    aad: &[u8],
    buf: &mut [u8],
    tag: &[u8; AEAD_TAG_LEN],
) -> Result<()> {
    use smallaios_security::crypto::chacha20_poly1305 as aead;
    if key.len() != aead::KEY_LEN {
        return Err(TlsClientError::KeyExchange);
    }
    let mut key_arr = [0u8; aead::KEY_LEN];
    key_arr.copy_from_slice(key);
    aead::open(&key_arr, nonce, aad, buf, tag).map_err(|_| TlsClientError::Aead)
}

fn seal_aes256(
    key: &[u8],
    nonce: &[u8; NONCE_LEN],
    aad: &[u8],
    buf: &mut [u8],
) -> Result<[u8; AEAD_TAG_LEN]> {
    use smallaios_security::crypto::aes_gcm::{Aes256Gcm, AesKey, GcmNonce};
    if key.len() != 32 {
        return Err(TlsClientError::KeyExchange);
    }
    let mut key_arr = [0u8; 32];
    key_arr.copy_from_slice(key);
    let cipher = Aes256Gcm::new(AesKey::from_bytes(key_arr));
    let gcm_nonce = GcmNonce::from_bytes(*nonce);
    let mut ciphertext = vec![0u8; buf.len()];
    let tag = cipher
        .encrypt(&gcm_nonce, aad, buf, &mut ciphertext)
        .map_err(|_| TlsClientError::Aead)?;
    buf.copy_from_slice(&ciphertext);
    Ok(*tag.as_bytes())
}

fn open_aes256(
    key: &[u8],
    nonce: &[u8; NONCE_LEN],
    aad: &[u8],
    buf: &mut [u8],
    tag: &[u8; AEAD_TAG_LEN],
) -> Result<()> {
    use smallaios_security::crypto::aes_gcm::{Aes256Gcm, AesKey, GcmNonce, GcmTag};
    if key.len() != 32 {
        return Err(TlsClientError::KeyExchange);
    }
    let mut key_arr = [0u8; 32];
    key_arr.copy_from_slice(key);
    let cipher = Aes256Gcm::new(AesKey::from_bytes(key_arr));
    let gcm_nonce = GcmNonce::from_bytes(*nonce);
    let gcm_tag = GcmTag::from_bytes(*tag);
    let mut plaintext = vec![0u8; buf.len()];
    cipher
        .decrypt(&gcm_nonce, aad, buf, &gcm_tag, &mut plaintext)
        .map_err(|_| TlsClientError::Aead)?;
    buf.copy_from_slice(&plaintext);
    Ok(())
}

#[cfg(test)]
#[path = "record_test_vectors.rs"]
mod test_vectors;

#[cfg(test)]
mod tests {
    use super::test_vectors::*;
    use super::*;

    #[test]
    fn header_parse_round_trip() {
        let h = RecordHeader {
            content_type: ContentType::ApplicationData,
            legacy_version: LEGACY_VERSION,
            length: 1024,
        };
        let bytes = h.encode();
        let parsed = RecordHeader::parse(&bytes).unwrap();
        assert_eq!(parsed, h);
    }

    #[test]
    fn header_too_short_rejected() {
        assert_eq!(
            RecordHeader::parse(&[0; 4]).unwrap_err(),
            TlsClientError::BadRecord
        );
    }

    #[test]
    fn unknown_content_type_rejected() {
        let buf = [99u8, 0x03, 0x03, 0, 0];
        assert_eq!(
            RecordHeader::parse(&buf).unwrap_err(),
            TlsClientError::BadRecord
        );
    }

    #[test]
    fn oversized_length_rejected_before_alloc() {
        // length = 0xFFFF; cap is MAX_CIPHERTEXT_LEN = 16640.
        let buf = [23, 0x03, 0x03, 0xff, 0xff];
        assert_eq!(
            RecordHeader::parse(&buf).unwrap_err(),
            TlsClientError::BadRecord
        );
    }

    #[test]
    fn legacy_version_enforcement() {
        let h = RecordHeader {
            content_type: ContentType::ApplicationData,
            legacy_version: 0x0302,
            length: 0,
        };
        assert_eq!(
            h.enforce_legacy_version().unwrap_err(),
            TlsClientError::Version
        );

        let h = RecordHeader {
            legacy_version: LEGACY_VERSION,
            ..h
        };
        h.enforce_legacy_version().unwrap();
    }

    #[test]
    fn cipher_suite_wire_round_trip() {
        for s in [
            CipherSuite::Aes256GcmSha384,
            CipherSuite::ChaCha20Poly1305Sha256,
        ] {
            let w = s.wire_value();
            assert_eq!(CipherSuite::from_wire(w).unwrap(), s);
        }
        assert!(CipherSuite::from_wire(0x1301).is_err()); // AES-128 not advertised
        assert!(CipherSuite::from_wire(0x00ff).is_err());
    }

    #[test]
    fn per_record_nonce_xors_seq() {
        let iv = ZERO_IV;
        let nonce = per_record_nonce(&iv, 1);
        // seq=1 BE = [0,0,0,0,0,0,0,1]; XOR'd into last 8 bytes.
        assert_eq!(nonce, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]);
        let nonce = per_record_nonce(&iv, 0x100);
        assert_eq!(nonce[NONCE_LEN - 1], 0x00);
        assert_eq!(nonce[NONCE_LEN - 2], 0x01);
    }

    #[test]
    fn plaintext_record_round_trip() {
        let frag = b"hello world";
        let rec = build_plaintext_record(ContentType::Handshake, frag).unwrap();
        assert_eq!(&rec[..5], &[22, 0x03, 0x03, 0, 11]);
        assert_eq!(&rec[5..], frag);
    }

    #[test]
    fn plaintext_oversized_rejected() {
        let big = vec![0u8; MAX_PLAINTEXT_LEN + 1];
        assert!(build_plaintext_record(ContentType::Handshake, &big).is_err());
    }

    #[test]
    fn seal_open_round_trip_chacha20() {
        let key = FILL_KEY_42;
        let iv = FILL_IV_9E;
        let inner = b"transcript material";
        let rec = seal(
            CipherSuite::ChaCha20Poly1305Sha256,
            &key,
            &iv,
            7,
            ContentType::Handshake,
            inner,
        )
        .unwrap();
        let (t, pt) = open(CipherSuite::ChaCha20Poly1305Sha256, &key, &iv, 7, &rec).unwrap();
        assert_eq!(t, ContentType::Handshake);
        assert_eq!(&pt, inner);
    }

    #[test]
    fn seal_open_round_trip_aes256() {
        let key = FILL_KEY_42;
        let iv = FILL_IV_9E;
        let inner = b"some application bytes";
        let rec = seal(
            CipherSuite::Aes256GcmSha384,
            &key,
            &iv,
            42,
            ContentType::ApplicationData,
            inner,
        )
        .unwrap();
        let (t, pt) = open(CipherSuite::Aes256GcmSha384, &key, &iv, 42, &rec).unwrap();
        assert_eq!(t, ContentType::ApplicationData);
        assert_eq!(&pt, inner);
    }

    #[test]
    fn tampered_ciphertext_rejected() {
        let key = ZERO_KEY;
        let iv = ZERO_IV;
        let mut rec = seal(
            CipherSuite::ChaCha20Poly1305Sha256,
            &key,
            &iv,
            0,
            ContentType::Handshake,
            b"abc",
        )
        .unwrap();
        rec[RECORD_HEADER_LEN] ^= 0x01;
        assert_eq!(
            open(CipherSuite::ChaCha20Poly1305Sha256, &key, &iv, 0, &rec).unwrap_err(),
            TlsClientError::Aead
        );
    }

    #[test]
    fn wrong_seq_rejected() {
        let key = ZERO_KEY;
        let iv = ZERO_IV;
        let rec = seal(
            CipherSuite::ChaCha20Poly1305Sha256,
            &key,
            &iv,
            5,
            ContentType::Handshake,
            b"x",
        )
        .unwrap();
        // Open with a different seq → AEAD authentication fails
        // because the nonce differs.
        assert_eq!(
            open(CipherSuite::ChaCha20Poly1305Sha256, &key, &iv, 6, &rec).unwrap_err(),
            TlsClientError::Aead
        );
    }

    #[test]
    fn open_with_legacy_version_mismatch_rejected() {
        // Build a valid record then corrupt the legacy_version
        // byte.
        let key = ZERO_KEY;
        let iv = ZERO_IV;
        let mut rec = seal(
            CipherSuite::ChaCha20Poly1305Sha256,
            &key,
            &iv,
            0,
            ContentType::Handshake,
            b"abc",
        )
        .unwrap();
        rec[1] = 0x03;
        rec[2] = 0x02; // TLS 1.1
        assert_eq!(
            open(CipherSuite::ChaCha20Poly1305Sha256, &key, &iv, 0, &rec).unwrap_err(),
            TlsClientError::Version
        );
    }

    #[test]
    fn ciphertext_too_short_rejected() {
        // Header says length = 5 but tag alone is 16 bytes —
        // any length < AEAD_TAG_LEN + 1 must be refused.
        let bad = [23, 0x03, 0x03, 0, 5, 0, 0, 0, 0, 0];
        let key = ZERO_KEY;
        let iv = ZERO_IV;
        assert_eq!(
            open(CipherSuite::ChaCha20Poly1305Sha256, &key, &iv, 0, &bad).unwrap_err(),
            TlsClientError::BadRecord
        );
    }

    #[test]
    fn open_with_wrong_outer_content_type_rejected() {
        // After-handshake records MUST have outer
        // content_type = application_data (23). Build a
        // record with outer = Handshake (22) and confirm
        // we refuse it.
        let key = ZERO_KEY;
        let iv = ZERO_IV;
        let mut rec = seal(
            CipherSuite::ChaCha20Poly1305Sha256,
            &key,
            &iv,
            0,
            ContentType::Handshake,
            b"x",
        )
        .unwrap();
        rec[0] = ContentType::Handshake as u8;
        assert_eq!(
            open(CipherSuite::ChaCha20Poly1305Sha256, &key, &iv, 0, &rec).unwrap_err(),
            TlsClientError::BadRecord
        );
    }
}
