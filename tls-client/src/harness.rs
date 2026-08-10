// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Mock TLS 1.3 server harness — **test use only, never production**.
//!
//! One copy of the synthetic server flight used to exercise the
//! client end-to-end: the wire-message builders, a server-side key
//! schedule that answers a real ClientHello, and the data-phase
//! seal/open helpers a mock peer needs after the handshake.
//!
//! Compiled in two situations only:
//! - `cfg(test)` — this crate's own unit tests (`handshake::driver`
//!   drives the full client state machine against it), and
//! - the `test-harness` feature — so `container`'s transport tests
//!   can run a full handshake through the production
//!   `TcpTlsStream` bridge against a mock server on a loopback
//!   socket (`tls-tcp-client-v1` task 8.3). Only dev-dependencies
//!   may enable it; a production build that turns it on ships dead
//!   test code at best.
//!
//! The server here is deliberately minimal: fixed seeds, a
//! synthetic (non-X.509) leaf certificate, and no negotiation
//! logic beyond echoing what the test asked for. Chain
//! verification against real DER chains is covered separately by
//! the `cert` corpus tests.

use crate::cert::{LeafPublicKey, ServerCertVerifier};
use crate::handshake::certificate_msg::certificate_verify_content;
use crate::handshake::extensions::{ext_type, named_group, sig_scheme, TLS_1_3};
use crate::handshake::finished::{build_finished, check_verify_data, parse_finished};
use crate::handshake::key_schedule::{HashAlg, KeySchedule, TranscriptHash};
use crate::handshake::{HandshakeHeader, HandshakeType};
use crate::record::{self, build_plaintext_record, CipherSuite, ContentType, RECORD_HEADER_LEN};
use crate::{Result, TlsClientError};
use alloc::vec::Vec;
use smallaios_security::crypto::ed25519::{ed25519_keygen, ed25519_sign, Ed25519KeyPair};
use smallaios_security::crypto::ml_kem::{ml_kem_768_encaps, MlKemPublicKey, ML_KEM_768_PK_LEN};
use smallaios_security::crypto::x25519::{x25519_dh, x25519_keygen, X25519PublicKey};

// ── Wire-message builders ─────────────────────────────────────

/// Prepend the 4-byte handshake header to `body`.
pub fn wrap_handshake(msg_type: HandshakeType, body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(HandshakeHeader::LEN + body.len());
    HandshakeHeader {
        msg_type,
        length: body.len() as u32,
    }
    .encode(&mut out)
    .unwrap();
    out.extend_from_slice(body);
    out
}

/// Build a synthetic ServerHello message (header included).
/// `selected_version = None` omits the supported_versions extension.
pub fn build_server_hello(
    cipher_suite_code: u16,
    selected_version: Option<u16>,
    key_share_group: u16,
    key_share_pk: &[u8],
    random: [u8; 32],
) -> Vec<u8> {
    build_server_hello_with_session_id(
        cipher_suite_code,
        selected_version,
        key_share_group,
        key_share_pk,
        random,
        &[],
    )
}

/// [`build_server_hello`] with a caller-chosen
/// `legacy_session_id_echo`. The client under test always sends an
/// empty `legacy_session_id`, so tests pass a non-empty echo here
/// to provoke the RFC 8446 §4.1.3 mismatch abort.
pub fn build_server_hello_with_session_id(
    cipher_suite_code: u16,
    selected_version: Option<u16>,
    key_share_group: u16,
    key_share_pk: &[u8],
    random: [u8; 32],
    session_id_echo: &[u8],
) -> Vec<u8> {
    let mut body = Vec::new();
    // legacy_version
    body.extend_from_slice(&[0x03, 0x03]);
    body.extend_from_slice(&random);
    // session_id_echo
    body.push(session_id_echo.len() as u8);
    body.extend_from_slice(session_id_echo);
    body.extend_from_slice(&cipher_suite_code.to_be_bytes());
    // legacy_compression_method
    body.push(0);
    let mut exts = Vec::new();
    if let Some(v) = selected_version {
        // supported_versions extension (server form: bare u16)
        exts.extend_from_slice(&ext_type::SUPPORTED_VERSIONS.to_be_bytes());
        exts.extend_from_slice(&2u16.to_be_bytes());
        exts.extend_from_slice(&v.to_be_bytes());
    }
    let mut ks_body = Vec::new();
    ks_body.extend_from_slice(&key_share_group.to_be_bytes());
    ks_body.extend_from_slice(&(key_share_pk.len() as u16).to_be_bytes());
    ks_body.extend_from_slice(key_share_pk);
    exts.extend_from_slice(&ext_type::KEY_SHARE.to_be_bytes());
    exts.extend_from_slice(&(ks_body.len() as u16).to_be_bytes());
    exts.extend_from_slice(&ks_body);
    body.extend_from_slice(&(exts.len() as u16).to_be_bytes());
    body.extend_from_slice(&exts);
    wrap_handshake(HandshakeType::ServerHello, &body)
}

/// Build an EncryptedExtensions message carrying the given raw
/// `(ext_type, data)` pairs.
pub fn build_encrypted_extensions(exts: &[(u16, &[u8])]) -> Vec<u8> {
    let mut block = Vec::new();
    for (t, data) in exts {
        block.extend_from_slice(&t.to_be_bytes());
        block.extend_from_slice(&(data.len() as u16).to_be_bytes());
        block.extend_from_slice(data);
    }
    let mut body = Vec::new();
    body.extend_from_slice(&(block.len() as u16).to_be_bytes());
    body.extend_from_slice(&block);
    wrap_handshake(HandshakeType::EncryptedExtensions, &body)
}

/// Build a Certificate message from raw DER blobs (no per-entry
/// extensions, empty certificate_request_context).
pub fn build_certificate(certs: &[&[u8]]) -> Vec<u8> {
    let mut list = Vec::new();
    for c in certs {
        list.extend_from_slice(&(c.len() as u32).to_be_bytes()[1..]); // u24
        list.extend_from_slice(c);
        list.extend_from_slice(&[0, 0]); // no per-entry extensions
    }
    let mut body = Vec::new();
    body.push(0); // empty certificate_request_context
    body.extend_from_slice(&(list.len() as u32).to_be_bytes()[1..]); // u24
    body.extend_from_slice(&list);
    wrap_handshake(HandshakeType::Certificate, &body)
}

/// Build a CertificateVerify message from a scheme code and raw
/// signature bytes.
pub fn build_certificate_verify(scheme: u16, sig: &[u8]) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend_from_slice(&scheme.to_be_bytes());
    body.extend_from_slice(&(sig.len() as u16).to_be_bytes());
    body.extend_from_slice(sig);
    wrap_handshake(HandshakeType::CertificateVerify, &body)
}

// ── Synthetic identity ────────────────────────────────────────

/// The mock server's "certificate": an opaque blob, not X.509.
/// [`StaticLeafVerifier`] accepts exactly this leaf; real-chain
/// verification is the `cert` corpus tests' job.
pub const LEAF_DER: &[u8] = b"synthetic-leaf-der-for-driver-tests";

/// The mock server's fixed Ed25519 signing identity.
pub fn server_cert_ed25519() -> Ed25519KeyPair {
    ed25519_keygen(&[9u8; 32])
}

/// A [`ServerCertVerifier`] that accepts one exact leaf blob and
/// returns a fixed Ed25519 key — the harness stand-in for the
/// production `TrustStoreVerifier`.
pub struct StaticLeafVerifier {
    pub expected_leaf: &'static [u8],
    pub pubkey: [u8; 32],
}

impl ServerCertVerifier for StaticLeafVerifier {
    fn verify_chain(&self, certs: &[Vec<u8>], _server_name: Option<&str>) -> Result<LeafPublicKey> {
        if certs.first().map(Vec::as_slice) != Some(self.expected_leaf) {
            return Err(TlsClientError::BadCertificate);
        }
        Ok(LeafPublicKey::Ed25519(self.pubkey))
    }
}

/// [`StaticLeafVerifier`] wired to [`LEAF_DER`] and `kp`'s public
/// key — pairs with a flight from [`server_respond`] using `kp`.
pub fn harness_verifier(kp: &Ed25519KeyPair) -> StaticLeafVerifier {
    StaticLeafVerifier {
        expected_leaf: LEAF_DER,
        pubkey: *kp.public_key.as_bytes(),
    }
}

// ── ClientHello inspection ────────────────────────────────────

/// Extract the first key_share entry `(group, public_key)` from a
/// raw ClientHello *message* (no record header).
///
/// Panics on a malformed ClientHello — this is a test harness; a
/// panic here means the client under test emitted garbage.
pub fn parse_ch_key_share(ch_msg: &[u8]) -> (u16, Vec<u8>) {
    let body = &ch_msg[4..];
    let mut cur = 2 + 32; // legacy_version + random
    let sid = body[cur] as usize;
    cur += 1 + sid;
    let cs_len = u16::from_be_bytes([body[cur], body[cur + 1]]) as usize;
    cur += 2 + cs_len;
    cur += 1 + body[cur] as usize; // compression methods
    let ext_len = u16::from_be_bytes([body[cur], body[cur + 1]]) as usize;
    cur += 2;
    let ext_end = cur + ext_len;
    while cur < ext_end {
        let t = u16::from_be_bytes([body[cur], body[cur + 1]]);
        let l = u16::from_be_bytes([body[cur + 2], body[cur + 3]]) as usize;
        cur += 4;
        if t == ext_type::KEY_SHARE {
            // client form: u16 list length, then entries.
            let mut p = cur + 2;
            let group = u16::from_be_bytes([body[p], body[p + 1]]);
            let klen = u16::from_be_bytes([body[p + 2], body[p + 3]]) as usize;
            p += 4;
            return (group, body[p..p + klen].to_vec());
        }
        cur += l;
    }
    panic!("no key_share in ClientHello");
}

/// Does this ClientHello *record* offer `X25519+ML-KEM-768` as its
/// first key share? Pins the `audit-export-immudb-client` spec
/// scenario "PQC hybrid offered first".
pub fn client_hello_offers_hybrid_first(ch_record: &[u8]) -> bool {
    let (group, _) = parse_ch_key_share(&ch_record[RECORD_HEADER_LEN..]);
    group == named_group::X25519_MLKEM768
}

// ── Server flight ─────────────────────────────────────────────

/// Knobs for [`server_respond`]'s deliberately-broken variants.
#[derive(Debug, Clone, Copy, Default)]
pub struct ServerFlightOptions {
    /// Corrupt the server Finished verify_data.
    pub tamper_finished: bool,
    /// Coalesce trailing junk bytes into the Finished record —
    /// they land beyond the RFC 8446 §5.1 key-change boundary and
    /// must abort the client.
    pub junk_after_finished: bool,
}

/// Everything the mock server produced for its flight, plus the
/// state needed to verify the client Finished and run the data
/// phase.
pub struct ServerFlight {
    /// SH + encrypted {EE+Cert} {CV} {Fin} record bytes.
    pub records: Vec<u8>,
    pub schedule: KeySchedule,
    pub transcript: TranscriptHash,
    pub suite: CipherSuite,
    /// Client→server application-record sequence number.
    read_seq: u64,
    /// Server→client application-record sequence number.
    write_seq: u64,
    app_derived: bool,
}

/// Run the server side: consume the ClientHello record, emit
/// SH + encrypted {EE+Certificate} (coalesced in one record,
/// exercising multi-message records), {CertificateVerify},
/// {Finished} records.
pub fn server_respond(
    ch_record: &[u8],
    suite: CipherSuite,
    cert_kp: &Ed25519KeyPair,
    opts: ServerFlightOptions,
) -> ServerFlight {
    let ch_msg = &ch_record[RECORD_HEADER_LEN..];
    let (group, client_share) = parse_ch_key_share(ch_msg);

    // Server key exchange.
    let server_x = x25519_keygen(&[0x55u8; 32]);
    let (ecdhe, server_share): (Vec<u8>, Vec<u8>) = if group == named_group::X25519_MLKEM768 {
        let client_mlkem = MlKemPublicKey::from_slice(&client_share[..ML_KEM_768_PK_LEN]).unwrap();
        let (ct, mlkem_ss) = ml_kem_768_encaps(&client_mlkem, &[0x66u8; 32]).unwrap();
        let mut client_x = [0u8; 32];
        client_x.copy_from_slice(&client_share[ML_KEM_768_PK_LEN..]);
        let x_ss = x25519_dh(&server_x.secret_key, &X25519PublicKey::from_bytes(client_x)).unwrap();
        let mut ss = Vec::new();
        ss.extend_from_slice(mlkem_ss.as_bytes());
        ss.extend_from_slice(&x_ss);
        let mut share = Vec::new();
        share.extend_from_slice(ct.as_bytes());
        share.extend_from_slice(server_x.public_key.as_bytes());
        (ss, share)
    } else {
        let mut client_x = [0u8; 32];
        client_x.copy_from_slice(&client_share);
        let ss = x25519_dh(&server_x.secret_key, &X25519PublicKey::from_bytes(client_x)).unwrap();
        (ss.to_vec(), server_x.public_key.as_bytes().to_vec())
    };

    let sh_msg = build_server_hello(
        suite.wire_value(),
        Some(TLS_1_3),
        group,
        &server_share,
        [0xaa; 32],
    );
    let alg = HashAlg::for_suite(suite);
    let mut transcript = TranscriptHash::new(alg);
    transcript.update(ch_msg);
    transcript.update(&sh_msg);
    let mut schedule = KeySchedule::new(alg);
    schedule
        .derive_handshake_secrets(&ecdhe, &transcript.current())
        .unwrap();
    let server_keys = schedule
        .traffic_keys(schedule.server_hs_traffic_secret().unwrap())
        .unwrap();

    let mut records = build_plaintext_record(ContentType::Handshake, &sh_msg).unwrap();
    let mut seq = 0u64;
    let seal_hs = |payload: &[u8], seq: &mut u64| {
        let rec = record::seal(
            suite,
            &server_keys.key,
            &server_keys.iv,
            *seq,
            ContentType::Handshake,
            payload,
        )
        .unwrap();
        *seq += 1;
        rec
    };

    // EE + Certificate coalesced into ONE record.
    let ee_msg = build_encrypted_extensions(&[(ext_type::SERVER_NAME, &[])]);
    let cert_msg = build_certificate(&[LEAF_DER]);
    transcript.update(&ee_msg);
    transcript.update(&cert_msg);
    let mut coalesced = ee_msg.clone();
    coalesced.extend_from_slice(&cert_msg);
    records.extend_from_slice(&seal_hs(&coalesced, &mut seq));

    // CertificateVerify over Transcript-Hash(CH..Certificate).
    let content = certificate_verify_content(&transcript.current());
    let sig = ed25519_sign(&cert_kp.secret_key, &content);
    let cv_msg = build_certificate_verify(sig_scheme::ED25519, sig.as_bytes());
    transcript.update(&cv_msg);
    records.extend_from_slice(&seal_hs(&cv_msg, &mut seq));

    // Server Finished over Transcript-Hash(CH..CV).
    let mut vd = schedule
        .finished_verify_data(
            schedule.server_hs_traffic_secret().unwrap(),
            &transcript.current(),
        )
        .unwrap();
    if opts.tamper_finished {
        vd[0] ^= 0xff;
    }
    let fin_msg = build_finished(&vd).unwrap();
    transcript.update(&fin_msg);
    let mut fin_payload = fin_msg;
    if opts.junk_after_finished {
        fin_payload.extend_from_slice(&[0u8; 7]);
    }
    records.extend_from_slice(&seal_hs(&fin_payload, &mut seq));

    ServerFlight {
        records,
        schedule,
        transcript,
        suite,
        read_seq: 0,
        write_seq: 0,
        app_derived: false,
    }
}

/// A ServerHello record that selects TLS 1.2 — the client must
/// abort with `TlsClientError::Version` before any application
/// data moves ("TLS 1.2 handshake rejected" spec scenario).
pub fn build_tls12_server_hello_record() -> Vec<u8> {
    let x = x25519_keygen(&[0x55u8; 32]);
    let sh = build_server_hello(
        0x1302,
        Some(0x0303),
        named_group::X25519,
        x.public_key.as_bytes(),
        [0xaa; 32],
    );
    build_plaintext_record(ContentType::Handshake, &sh).unwrap()
}

/// A TLS 1.3 ServerHello record that answers a hybrid offer with a
/// pure-classical X25519 share — a `require_pqc` client must abort
/// with `TlsClientError::PqcDowngrade` ("PQC hybrid offered first"
/// spec scenario's refusal arm).
pub fn build_classical_server_hello_record() -> Vec<u8> {
    let x = x25519_keygen(&[0x55u8; 32]);
    let sh = build_server_hello(
        0x1303,
        Some(TLS_1_3),
        named_group::X25519,
        x.public_key.as_bytes(),
        [0xaa; 32],
    );
    build_plaintext_record(ContentType::Handshake, &sh).unwrap()
}

impl ServerFlight {
    /// Verify the client Finished record and derive the
    /// application secrets, arming the data-phase helpers.
    pub fn complete(&mut self, client_finished_record: &[u8]) -> Result<()> {
        let client_keys = self
            .schedule
            .traffic_keys(self.schedule.client_hs_traffic_secret()?)?;
        let (inner, fin_plain) = record::open(
            self.suite,
            &client_keys.key,
            &client_keys.iv,
            0,
            client_finished_record,
        )?;
        if inner != ContentType::Handshake {
            return Err(TlsClientError::BadHandshake);
        }
        let alg = HashAlg::for_suite(self.suite);
        let received = parse_finished(&fin_plain, alg.digest_len())?;
        let expected = self.schedule.finished_verify_data(
            self.schedule.client_hs_traffic_secret()?,
            &self.transcript.current(),
        )?;
        check_verify_data(&received, &expected)?;
        self.schedule
            .derive_application_secrets(&self.transcript.current())?;
        self.app_derived = true;
        Ok(())
    }

    /// Decrypt one client→server application record. Only valid
    /// after [`complete`](Self::complete).
    pub fn open_app_record(&mut self, record_bytes: &[u8]) -> Result<(ContentType, Vec<u8>)> {
        if !self.app_derived {
            return Err(TlsClientError::BadHandshake);
        }
        let keys = self
            .schedule
            .traffic_keys(self.schedule.client_ap_traffic_secret()?)?;
        let out = record::open(self.suite, &keys.key, &keys.iv, self.read_seq, record_bytes)?;
        self.read_seq += 1;
        Ok(out)
    }

    /// Seal one server→client application record. Only valid after
    /// [`complete`](Self::complete).
    pub fn seal_app_record(&mut self, payload: &[u8]) -> Result<Vec<u8>> {
        if !self.app_derived {
            return Err(TlsClientError::BadHandshake);
        }
        let keys = self
            .schedule
            .traffic_keys(self.schedule.server_ap_traffic_secret()?)?;
        let rec = record::seal(
            self.suite,
            &keys.key,
            &keys.iv,
            self.write_seq,
            ContentType::ApplicationData,
            payload,
        )?;
        self.write_seq += 1;
        Ok(rec)
    }
}
