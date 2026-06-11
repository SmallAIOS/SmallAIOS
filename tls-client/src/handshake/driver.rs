// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Client handshake state machine (RFC 8446 §4) — Phase 4 of
//! `tls-tcp-client-v1`, design.md D7.
//!
//! The driver is a synchronous, I/O-free state machine. The
//! integration layer (Phase 7's `TcpTlsStream`) owns the socket:
//! it sends [`ClientHandshake::new`]'s initial flight, then feeds
//! every received byte through [`ClientHandshake::push`] and
//! writes back whatever bytes `push` returns, until the status
//! turns [`HandshakeStatus::Complete`]. No `async`, no executor.
//!
//! ```text
//! ExpectServerHello ──► ExpectEncryptedExtensions ──►
//! ExpectCertificate ──► ExpectCertificateVerify ──►
//! ExpectFinished ──► Complete
//! ```
//!
//! Refused mid-handshake (v1 scope): HelloRetryRequest (D7),
//! CertificateRequest (no mTLS — see proposal §"No mTLS in v1"),
//! TLS 1.2 selection anywhere (spec: version pinning).
//!
//! Certificate-chain validation is delegated through the
//! [`ServerCertVerifier`] seam (`crate::cert`) — the wire parsing
//! and CertificateVerify transcript-signature check happen here;
//! X.509 structure/chain/hostname checks are Phase 5's
//! `TrustStoreVerifier`.

use super::certificate_msg::{
    parse_certificate, parse_certificate_verify, verify_certificate_verify,
};
use super::client_hello::{build_client_hello, ClientHelloInput, ClientHelloKeyShare};
use super::encrypted_extensions::parse_encrypted_extensions;
use super::finished::{build_finished, check_verify_data, parse_finished};
use super::key_schedule::{HashAlg, KeySchedule, TrafficKeys, TranscriptHash};
use super::server_hello::parse_server_hello;
use super::{extensions::named_group, HandshakeHeader, HandshakeType};
use crate::cert::{LeafPublicKey, ServerCertVerifier};
use crate::record::{
    self, build_plaintext_record, CipherSuite, ContentType, RecordHeader, RECORD_HEADER_LEN,
};
use crate::{Result, TlsClientError};
use alloc::vec::Vec;
use smallaios_security::crypto::ml_kem::{
    ml_kem_768_decaps, ml_kem_768_keygen, MlKemCiphertext, MlKemKeyPair, ML_KEM_768_CT_LEN,
    ML_KEM_768_PK_LEN,
};
use smallaios_security::crypto::x25519::{
    x25519_dh, x25519_keygen, X25519KeyPair, X25519PublicKey,
};

/// ServerHello.random sentinel that turns the message into a
/// HelloRetryRequest (RFC 8446 §4.1.3) = SHA-256("HelloRetryRequest").
/// v1 refuses HRR per design.md D7; a unit test pins this constant
/// to the live SHA-256 implementation.
pub(crate) const HRR_RANDOM: [u8; 32] = [
    0xcf, 0x21, 0xad, 0x74, 0xe5, 0x9a, 0x61, 0x11, 0xbe, 0x1d, 0x8c, 0x02, 0x1e, 0x65, 0xb8, 0x91,
    0xc2, 0xa2, 0x11, 0x16, 0x7a, 0xbb, 0x8c, 0x5e, 0x07, 0x9e, 0x09, 0xe2, 0xc8, 0xa8, 0x33, 0x9c,
];

/// Maximum number of unencrypted middlebox-compatibility
/// ChangeCipherSpec records tolerated per handshake (RFC 8446 §5
/// says drop them; we additionally bound how much of that junk we
/// tolerate so a peer cannot stream CCS records indefinitely).
const MAX_CCS_RECORDS: u8 = 2;

/// Cap on the claimed 24-bit length of a Certificate message.
/// Real chains can be tens of KiB (several certificates plus
/// per-entry extensions), so this cap is generous.
const MAX_CERTIFICATE_MSG_LEN: usize = 128 * 1024;

/// Cap on the claimed 24-bit length of every other handshake
/// message. ServerHello/EncryptedExtensions/CertificateVerify/
/// Finished are all well under a few KiB in practice; 32 KiB
/// leaves slack without letting a forged header pin multiple
/// records' worth of heap (see `drain_messages`).
const MAX_HANDSHAKE_MSG_LEN: usize = 32 * 1024;

/// Operator-facing handshake policy (subset of the `immudb.toml`
/// `tls.*` surface that Phase 4 consumes).
#[derive(Debug, Clone, Copy)]
pub struct ClientConfig<'a> {
    /// SNI hostname; `None` for IP-literal endpoints (RFC 6066 §3).
    pub server_name: Option<&'a str>,
    /// When true, offer X25519+ML-KEM-768 first and refuse a
    /// classical reply with `PqcDowngrade` (design.md D3).
    pub require_pqc: bool,
}

/// Caller-supplied entropy. The driver is `#![no_std]` and does not
/// bake in a CSPRNG; the integration layer draws these from
/// `security::crypto::csprng`.
pub struct ClientEntropy {
    /// ClientHello.random.
    pub client_random: [u8; 32],
    /// X25519 ephemeral key seed.
    pub x25519_seed: [u8; 32],
    /// ML-KEM-768 keypair seed — consumed only when `require_pqc`.
    pub ml_kem_seed: [u8; 64],
}

/// Where the handshake stands after a `push` call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HandshakeStatus {
    /// Keep reading from the peer and calling `push`.
    InProgress,
    /// Handshake done — application keys are available.
    Complete,
}

/// Output of one `push` call.
#[derive(Debug)]
pub struct Push {
    /// Bytes the integration layer must write to the socket
    /// before reading further (empty when nothing to send).
    pub send: Vec<u8>,
    pub status: HandshakeStatus,
}

/// Application-traffic keys for the data phase, handed to the
/// record loop once the handshake completes.
pub struct AppKeys {
    pub suite: CipherSuite,
    pub client: TrafficKeys,
    pub server: TrafficKeys,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    ExpectServerHello,
    ExpectEncryptedExtensions,
    ExpectCertificate,
    ExpectCertificateVerify,
    ExpectFinished,
    Complete,
    Failed,
}

enum KeyExchange {
    Classical(X25519KeyPair),
    // Boxed: an ML-KEM-768 keypair is ~3.6 KiB.
    Hybrid(X25519KeyPair, alloc::boxed::Box<MlKemKeyPair>),
}

/// The client-side handshake state machine.
pub struct ClientHandshake<'a, V: ServerCertVerifier> {
    state: State,
    require_pqc: bool,
    server_name: Option<&'a str>,
    verifier: &'a V,
    kx: KeyExchange,
    /// Raw ClientHello message (transcript seed — the transcript
    /// hash can only start once the ServerHello fixes the suite).
    client_hello: Vec<u8>,
    /// Unconsumed socket bytes.
    inbuf: Vec<u8>,
    /// Reassembly buffer for handshake messages spanning records.
    hs_buf: Vec<u8>,
    suite: Option<CipherSuite>,
    transcript: Option<TranscriptHash>,
    schedule: Option<KeySchedule>,
    /// Server handshake-traffic keys + record sequence number.
    server_keys: Option<TrafficKeys>,
    read_seq: u64,
    /// Unencrypted CCS records seen so far (bounded by
    /// [`MAX_CCS_RECORDS`]).
    ccs_count: u8,
    /// Leaf public key returned by the chain verifier.
    leaf_key: Option<LeafPublicKey>,
    /// Transcript-Hash(CH..Certificate) for CertificateVerify.
    hash_at_cert: Vec<u8>,
    /// Transcript-Hash(CH..CertificateVerify) for server Finished.
    hash_at_cv: Vec<u8>,
    sni_acked: bool,
}

impl<'a, V: ServerCertVerifier> ClientHandshake<'a, V> {
    /// Start a handshake: returns the driver and the initial
    /// flight (the ClientHello record) to write to the socket.
    pub fn new(
        config: ClientConfig<'a>,
        entropy: &ClientEntropy,
        verifier: &'a V,
    ) -> Result<(Self, Vec<u8>)> {
        let x25519 = x25519_keygen(&entropy.x25519_seed);
        let (kx, groups, share): (KeyExchange, &[u16], Vec<u8>) = if config.require_pqc {
            let ml_kem =
                ml_kem_768_keygen(&entropy.ml_kem_seed).map_err(|_| TlsClientError::KeyExchange)?;
            // draft-ietf-tls-ecdhe-mlkem X25519MLKEM768 ordering:
            // ML-KEM-768 encapsulation key first, then X25519.
            let mut share = Vec::with_capacity(ML_KEM_768_PK_LEN + 32);
            share.extend_from_slice(ml_kem.public_key.as_bytes());
            share.extend_from_slice(x25519.public_key.as_bytes());
            (
                KeyExchange::Hybrid(x25519, alloc::boxed::Box::new(ml_kem)),
                &[named_group::X25519_MLKEM768, named_group::X25519],
                share,
            )
        } else {
            let share = x25519.public_key.as_bytes().to_vec();
            (
                KeyExchange::Classical(x25519),
                &[named_group::X25519],
                share,
            )
        };
        let primary_group = groups[0];
        let suites = [
            CipherSuite::Aes256GcmSha384,
            CipherSuite::ChaCha20Poly1305Sha256,
        ];
        let client_hello = build_client_hello(&ClientHelloInput {
            random: entropy.client_random,
            server_name: config.server_name,
            cipher_suites: &suites,
            supported_groups: groups,
            key_share: ClientHelloKeyShare {
                group: primary_group,
                public_key: &share,
            },
        })?;
        let flight = build_plaintext_record(ContentType::Handshake, &client_hello)?;
        Ok((
            Self {
                state: State::ExpectServerHello,
                require_pqc: config.require_pqc,
                server_name: config.server_name,
                verifier,
                kx,
                client_hello,
                inbuf: Vec::new(),
                hs_buf: Vec::new(),
                suite: None,
                transcript: None,
                schedule: None,
                server_keys: None,
                read_seq: 0,
                ccs_count: 0,
                leaf_key: None,
                hash_at_cert: Vec::new(),
                hash_at_cv: Vec::new(),
                sni_acked: false,
            },
            flight,
        ))
    }

    /// Feed bytes received from the peer. Returns bytes to send
    /// back (the client Finished flight, once the server flight
    /// verifies) and the handshake status.
    pub fn push(&mut self, incoming: &[u8]) -> Result<Push> {
        if self.state == State::Failed {
            return Err(TlsClientError::BadHandshake);
        }
        if self.state == State::Complete {
            return Ok(Push {
                send: Vec::new(),
                status: HandshakeStatus::Complete,
            });
        }
        self.inbuf.extend_from_slice(incoming);
        let mut send = Vec::new();
        let result = self.drain_records(&mut send);
        if let Err(e) = result {
            self.state = State::Failed;
            return Err(e);
        }
        Ok(Push {
            send,
            status: if self.state == State::Complete {
                HandshakeStatus::Complete
            } else {
                HandshakeStatus::InProgress
            },
        })
    }

    /// Negotiated cipher suite (after ServerHello).
    pub fn suite(&self) -> Option<CipherSuite> {
        self.suite
    }

    /// Whether the server acknowledged our SNI in
    /// EncryptedExtensions.
    pub fn sni_acked(&self) -> bool {
        self.sni_acked
    }

    /// Application-traffic keys; only once `Complete`.
    pub fn app_keys(&self) -> Result<AppKeys> {
        if self.state != State::Complete {
            return Err(TlsClientError::BadHandshake);
        }
        let suite = self.suite.ok_or(TlsClientError::BadHandshake)?;
        let ks = self.schedule.as_ref().ok_or(TlsClientError::BadHandshake)?;
        Ok(AppKeys {
            suite,
            client: ks.traffic_keys(ks.client_ap_traffic_secret()?)?,
            server: ks.traffic_keys(ks.server_ap_traffic_secret()?)?,
        })
    }

    /// Bytes received after the handshake finished (e.g. a
    /// NewSessionTicket flight coalesced into the same TCP
    /// segment). The record loop takes ownership of these.
    pub fn take_residual(&mut self) -> Vec<u8> {
        core::mem::take(&mut self.inbuf)
    }

    // ── internals ─────────────────────────────────────────────

    fn drain_records(&mut self, send: &mut Vec<u8>) -> Result<()> {
        loop {
            if self.state == State::Complete {
                return Ok(());
            }
            if self.inbuf.len() < RECORD_HEADER_LEN {
                return Ok(());
            }
            let header = RecordHeader::parse(&self.inbuf[..RECORD_HEADER_LEN])?;
            let total = RECORD_HEADER_LEN + header.length as usize;
            if self.inbuf.len() < total {
                return Ok(());
            }
            let record: Vec<u8> = self.inbuf.drain(..total).collect();
            self.process_record(header, &record, send)?;
            self.drain_messages(send)?;
        }
    }

    fn process_record(
        &mut self,
        header: RecordHeader,
        record: &[u8],
        _send: &mut Vec<u8>,
    ) -> Result<()> {
        match header.content_type {
            // Middlebox-compatibility ChangeCipherSpec — RFC 8446
            // §5: drop unencrypted CCS received during the
            // handshake, but only the genuine compatibility form
            // (exactly one payload byte, value 0x01); any other
            // length or value is a protocol violation. Tolerance is
            // also capped at MAX_CCS_RECORDS so a peer cannot
            // stream junk CCS records at us indefinitely.
            ContentType::ChangeCipherSpec => {
                if record[RECORD_HEADER_LEN..] != [0x01] {
                    return Err(TlsClientError::BadHandshake);
                }
                self.ccs_count += 1;
                if self.ccs_count > MAX_CCS_RECORDS {
                    return Err(TlsClientError::BadHandshake);
                }
                Ok(())
            }
            ContentType::Alert if self.server_keys.is_none() => {
                // Plaintext alert during the first flight (e.g.
                // handshake_failure). Surface as handshake failure.
                Err(TlsClientError::BadHandshake)
            }
            ContentType::Handshake if self.server_keys.is_none() => {
                // Plaintext handshake record (ServerHello flight).
                //
                // Deliberate deviation, on record: RFC 8446 §5.1
                // says receivers MUST NOT pay attention to
                // legacy_record_version ("MUST be ignored for all
                // purposes"), but no conformant TLS 1.3 peer sends
                // anything other than 0x0303 here, so we pin it
                // defensively and reject everything else — it
                // shrinks the unauthenticated parsing surface at
                // zero interop cost.
                header.enforce_legacy_version()?;
                self.hs_buf.extend_from_slice(&record[RECORD_HEADER_LEN..]);
                Ok(())
            }
            ContentType::ApplicationData => {
                // Encrypted record. Keys must exist by now.
                let keys = self
                    .server_keys
                    .as_ref()
                    .ok_or(TlsClientError::BadHandshake)?;
                let suite = self.suite.ok_or(TlsClientError::BadHandshake)?;
                let (inner_type, plaintext) =
                    record::open(suite, &keys.key, &keys.iv, self.read_seq, record)?;
                self.read_seq += 1;
                match inner_type {
                    ContentType::Handshake => {
                        self.hs_buf.extend_from_slice(&plaintext);
                        Ok(())
                    }
                    ContentType::Alert => Err(TlsClientError::BadHandshake),
                    _ => Err(TlsClientError::BadHandshake),
                }
            }
            _ => Err(TlsClientError::BadHandshake),
        }
    }

    fn drain_messages(&mut self, send: &mut Vec<u8>) -> Result<()> {
        loop {
            if self.hs_buf.len() < HandshakeHeader::LEN {
                return Ok(());
            }
            let header = HandshakeHeader::parse(&self.hs_buf)?;
            // DoS bound: cap the claimed 24-bit length BEFORE the
            // reassembly buffer is allowed to grow toward it. A
            // forged header can claim up to 16 MiB and would
            // otherwise make us accumulate records on the no_std
            // kernel heap during the unauthenticated phase.
            // Certificate is the only legitimately large message
            // (a real chain can be tens of KiB), so it gets a
            // wider cap than everything else.
            let cap = match header.msg_type {
                HandshakeType::Certificate => MAX_CERTIFICATE_MSG_LEN,
                _ => MAX_HANDSHAKE_MSG_LEN,
            };
            if header.length as usize > cap {
                return Err(TlsClientError::BadHandshake);
            }
            let total = HandshakeHeader::LEN + header.length as usize;
            if self.hs_buf.len() < total {
                return Ok(());
            }
            let msg: Vec<u8> = self.hs_buf.drain(..total).collect();
            self.process_message(header.msg_type, &msg, send)?;
            if self.state == State::Complete {
                return Ok(());
            }
        }
    }

    fn process_message(
        &mut self,
        msg_type: HandshakeType,
        msg: &[u8],
        send: &mut Vec<u8>,
    ) -> Result<()> {
        match (self.state, msg_type) {
            (State::ExpectServerHello, HandshakeType::ServerHello) => self.on_server_hello(msg),
            (State::ExpectEncryptedExtensions, HandshakeType::EncryptedExtensions) => {
                let ee = parse_encrypted_extensions(msg)?;
                self.sni_acked = ee.sni_acked;
                self.transcript_update(msg);
                self.state = State::ExpectCertificate;
                Ok(())
            }
            (State::ExpectCertificate, HandshakeType::CertificateRequest) => {
                // v1 ships no client certificate (proposal: "No
                // mTLS in v1") — refuse rather than answer with an
                // empty Certificate the operator can't influence.
                Err(TlsClientError::BadHandshake)
            }
            (State::ExpectCertificate, HandshakeType::Certificate) => {
                let cert_msg = parse_certificate(msg)?;
                self.leaf_key = Some(
                    self.verifier
                        .verify_chain(&cert_msg.certs, self.server_name)?,
                );
                self.transcript_update(msg);
                self.hash_at_cert = self.transcript_current();
                self.state = State::ExpectCertificateVerify;
                Ok(())
            }
            (State::ExpectCertificateVerify, HandshakeType::CertificateVerify) => {
                let cv = parse_certificate_verify(msg)?;
                let LeafPublicKey::Ed25519(pk) =
                    self.leaf_key.as_ref().ok_or(TlsClientError::BadHandshake)?;
                verify_certificate_verify(&cv, pk, &self.hash_at_cert)?;
                self.transcript_update(msg);
                self.hash_at_cv = self.transcript_current();
                self.state = State::ExpectFinished;
                Ok(())
            }
            (State::ExpectFinished, HandshakeType::Finished) => self.on_server_finished(msg, send),
            // Anything else is out of order for this state machine
            // (incl. HelloRetryRequest-triggered second ServerHello,
            // NewSessionTicket before Finished, KeyUpdate during
            // handshake).
            _ => Err(TlsClientError::BadHandshake),
        }
    }

    fn on_server_hello(&mut self, msg: &[u8]) -> Result<()> {
        let sh = parse_server_hello(msg)?;
        // RFC 8446 §4.1.3: legacy_session_id_echo MUST echo the
        // ClientHello's legacy_session_id, and a client MUST abort
        // on a mismatch. This client always sends an EMPTY
        // legacy_session_id (see client_hello.rs), so any non-empty
        // echo is a violation.
        if !sh.session_id_echo.is_empty() {
            return Err(TlsClientError::BadHandshake);
        }
        // Downgrade-sentinel note: the RFC 8446 §4.1.3 "DOWNGRD"
        // check on the last 8 bytes of ServerHello.random protects
        // clients that are willing to negotiate TLS 1.2 or below.
        // It is deliberately omitted here: this client pins
        // supported_versions to 0x0304 (parse_server_hello aborts
        // on anything else) and never offers a lower version, which
        // structurally subsumes the sentinel check — a downgraded
        // ServerHello cannot get past the version pinning. Anyone
        // adding TLS 1.2 fallback later MUST add the sentinel check
        // here.
        //
        // HelloRetryRequest is a ServerHello whose random is the
        // fixed SHA-256("HelloRetryRequest") sentinel — refused in
        // v1 (design.md D7).
        if sh.random == HRR_RANDOM {
            return Err(TlsClientError::BadHandshake);
        }
        // Group policy (design.md D3 + spec scenario "Hybrid
        // required and server picks classical → reject").
        let ecdhe: Vec<u8> = match (&self.kx, sh.key_share_group) {
            (KeyExchange::Hybrid(x, mlkem), g) if g == named_group::X25519_MLKEM768 => {
                // Server share: ML-KEM-768 ciphertext || X25519 pk
                // (draft-ietf-tls-ecdhe-mlkem ordering). Shared
                // secret: ML-KEM ss || X25519 ss.
                if sh.key_share_public.len() != ML_KEM_768_CT_LEN + 32 {
                    return Err(TlsClientError::KeyExchange);
                }
                let mut ct = [0u8; ML_KEM_768_CT_LEN];
                ct.copy_from_slice(&sh.key_share_public[..ML_KEM_768_CT_LEN]);
                let ml_kem_ss =
                    ml_kem_768_decaps(&mlkem.secret_key, &MlKemCiphertext::from_bytes(ct))
                        .map_err(|_| TlsClientError::KeyExchange)?;
                let mut server_x = [0u8; 32];
                server_x.copy_from_slice(&sh.key_share_public[ML_KEM_768_CT_LEN..]);
                let x_ss = x25519_dh(&x.secret_key, &X25519PublicKey::from_bytes(server_x))
                    .map_err(|_| TlsClientError::KeyExchange)?;
                let mut ss = Vec::with_capacity(64);
                ss.extend_from_slice(ml_kem_ss.as_bytes());
                ss.extend_from_slice(&x_ss);
                ss
            }
            (KeyExchange::Hybrid(..), _) => {
                // Operator demanded PQC; server picked a classical
                // group.
                return Err(TlsClientError::PqcDowngrade);
            }
            (KeyExchange::Classical(x), g) if g == named_group::X25519 => {
                if sh.key_share_public.len() != 32 {
                    return Err(TlsClientError::KeyExchange);
                }
                let mut server_x = [0u8; 32];
                server_x.copy_from_slice(&sh.key_share_public);
                x25519_dh(&x.secret_key, &X25519PublicKey::from_bytes(server_x))
                    .map_err(|_| TlsClientError::KeyExchange)?
                    .to_vec()
            }
            (KeyExchange::Classical(_), _) => {
                // We never offered that group.
                return Err(TlsClientError::BadHandshake);
            }
        };
        debug_assert!(self.require_pqc == matches!(self.kx, KeyExchange::Hybrid(..)));

        // Suite fixed — start the transcript and the schedule.
        let alg = HashAlg::for_suite(sh.cipher_suite);
        let mut transcript = TranscriptHash::new(alg);
        transcript.update(&self.client_hello);
        transcript.update(msg);
        let mut schedule = KeySchedule::new(alg);
        schedule.derive_handshake_secrets(&ecdhe, &transcript.current())?;
        self.server_keys = Some(schedule.traffic_keys(schedule.server_hs_traffic_secret()?)?);
        self.read_seq = 0;
        self.suite = Some(sh.cipher_suite);
        self.transcript = Some(transcript);
        self.schedule = Some(schedule);
        // RFC 8446 §5.1 key-change boundary: ServerHello is the
        // last plaintext handshake message — everything after it
        // travels under the handshake keys just installed. Any
        // bytes still sitting in the reassembly buffer arrived in
        // plaintext records coalesced past the ServerHello and must
        // not be processed as if they had been protected.
        if !self.hs_buf.is_empty() {
            return Err(TlsClientError::BadHandshake);
        }
        self.state = State::ExpectEncryptedExtensions;
        Ok(())
    }

    fn on_server_finished(&mut self, msg: &[u8], send: &mut Vec<u8>) -> Result<()> {
        let suite = self.suite.ok_or(TlsClientError::BadHandshake)?;
        let ks = self.schedule.as_mut().ok_or(TlsClientError::BadHandshake)?;
        let alg = ks.alg();

        // Task 4.8 — verify the server Finished MAC over
        // Transcript-Hash(CH..CertificateVerify).
        let received = parse_finished(msg, alg.digest_len())?;
        let expected = ks.finished_verify_data(ks.server_hs_traffic_secret()?, &self.hash_at_cv)?;
        check_verify_data(&received, &expected)?;

        // Task 4.9 — application secrets are keyed by the
        // transcript through the server Finished...
        let transcript = self
            .transcript
            .as_mut()
            .ok_or(TlsClientError::BadHandshake)?;
        transcript.update(msg);
        let hash_through_sfin = transcript.current();
        ks.derive_application_secrets(&hash_through_sfin)?;

        // ...and the client Finished is the same transcript point,
        // MAC'd with the *client* handshake-traffic secret and
        // sealed under the client handshake keys (its sequence
        // number is 0 — this is our first encrypted record).
        let client_vd =
            ks.finished_verify_data(ks.client_hs_traffic_secret()?, &hash_through_sfin)?;
        let fin_msg = build_finished(&client_vd)?;
        let client_keys = ks.traffic_keys(ks.client_hs_traffic_secret()?)?;
        let sealed = record::seal(
            suite,
            &client_keys.key,
            &client_keys.iv,
            0,
            ContentType::Handshake,
            &fin_msg,
        )?;
        send.extend_from_slice(&sealed);
        // Same RFC 8446 §5.1 key-change rule at the handshake's
        // end: the server Finished closes the handshake key epoch.
        // Leftover bytes in the reassembly buffer (e.g. junk
        // coalesced into the Finished record) would otherwise be
        // silently retained across the boundary — post-handshake
        // messages must arrive in their own records under the
        // application keys.
        if !self.hs_buf.is_empty() {
            return Err(TlsClientError::BadHandshake);
        }
        self.state = State::Complete;
        Ok(())
    }

    fn transcript_update(&mut self, msg: &[u8]) {
        if let Some(t) = self.transcript.as_mut() {
            t.update(msg);
        }
    }

    fn transcript_current(&self) -> Vec<u8> {
        self.transcript
            .as_ref()
            .map(|t| t.current())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handshake::certificate_msg::certificate_verify_content;
    use crate::handshake::extensions::{ext_type, sig_scheme, TLS_1_3};
    use smallaios_security::crypto::ed25519::{ed25519_keygen, ed25519_sign, Ed25519KeyPair};
    use smallaios_security::crypto::ml_kem::ml_kem_768_encaps;
    use smallaios_security::sha2::sha256;

    #[test]
    fn hrr_sentinel_is_sha256_of_label() {
        assert_eq!(HRR_RANDOM, sha256(b"HelloRetryRequest"));
    }

    // ── In-test TLS 1.3 server ────────────────────────────────
    //
    // A minimal server built from the same primitives, driving
    // the client end-to-end: SH, EE+Certificate (coalesced in one
    // record, exercising multi-message records), CertificateVerify,
    // Finished, then verification of the client Finished.

    struct TestVerifier {
        expected_leaf: &'static [u8],
        pubkey: [u8; 32],
    }

    impl ServerCertVerifier for TestVerifier {
        fn verify_chain(
            &self,
            certs: &[Vec<u8>],
            _server_name: Option<&str>,
        ) -> Result<LeafPublicKey> {
            assert_eq!(certs[0], self.expected_leaf);
            Ok(LeafPublicKey::Ed25519(self.pubkey))
        }
    }

    struct RejectingVerifier;
    impl ServerCertVerifier for RejectingVerifier {
        fn verify_chain(&self, _: &[Vec<u8>], _: Option<&str>) -> Result<LeafPublicKey> {
            Err(TlsClientError::ChainUntrusted)
        }
    }

    const LEAF_DER: &[u8] = b"synthetic-leaf-der-for-driver-tests";

    /// Extract (cipher_suites, key_share group, key_share pubkey)
    /// from a raw ClientHello message.
    fn parse_ch_key_share(ch: &[u8]) -> (u16, Vec<u8>) {
        let body = &ch[4..];
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

    fn build_server_hello_msg(suite: CipherSuite, group: u16, share: &[u8]) -> Vec<u8> {
        build_server_hello_custom(suite.wire_value(), Some(TLS_1_3), group, share, [0xaa; 32])
    }

    use crate::handshake::test_util::build_certificate as build_certificate_raw;
    use crate::handshake::test_util::build_server_hello as build_server_hello_custom;
    use crate::handshake::test_util::{build_certificate_verify, build_encrypted_extensions};

    fn build_ee_msg() -> Vec<u8> {
        // EncryptedExtensions with an SNI ack.
        build_encrypted_extensions(&[(ext_type::SERVER_NAME, &[])])
    }

    fn build_certificate_msg(leaf: &[u8]) -> Vec<u8> {
        build_certificate_raw(&[leaf])
    }

    fn build_cv_msg(kp: &Ed25519KeyPair, transcript_hash: &[u8]) -> Vec<u8> {
        let content = certificate_verify_content(transcript_hash);
        let sig = ed25519_sign(&kp.secret_key, &content);
        build_certificate_verify(sig_scheme::ED25519, sig.as_bytes())
    }

    /// Everything the server produced for its flight, plus the
    /// state needed to check the client Finished.
    struct ServerFlight {
        records: Vec<u8>,
        schedule: KeySchedule,
        transcript: TranscriptHash,
        suite: CipherSuite,
    }

    /// Run the server side: consume the ClientHello record, emit
    /// SH + encrypted {EE+Cert} {CV} {Fin} records.
    ///
    /// `junk_after_finished` coalesces trailing bytes into the
    /// Finished record — they land beyond the RFC 8446 §5.1
    /// key-change boundary and must abort the client.
    fn server_respond(
        ch_record: &[u8],
        suite: CipherSuite,
        cert_kp: &Ed25519KeyPair,
        tamper_finished: bool,
        junk_after_finished: bool,
    ) -> ServerFlight {
        let ch_msg = &ch_record[RECORD_HEADER_LEN..];
        let (group, client_share) = parse_ch_key_share(ch_msg);

        // Server key exchange.
        let server_x = x25519_keygen(&[0x55u8; 32]);
        let (ecdhe, server_share): (Vec<u8>, Vec<u8>) = if group == named_group::X25519_MLKEM768 {
            let client_mlkem = smallaios_security::crypto::ml_kem::MlKemPublicKey::from_slice(
                &client_share[..ML_KEM_768_PK_LEN],
            )
            .unwrap();
            let (ct, mlkem_ss) = ml_kem_768_encaps(&client_mlkem, &[0x66u8; 32]).unwrap();
            let mut client_x = [0u8; 32];
            client_x.copy_from_slice(&client_share[ML_KEM_768_PK_LEN..]);
            let x_ss =
                x25519_dh(&server_x.secret_key, &X25519PublicKey::from_bytes(client_x)).unwrap();
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
            let ss =
                x25519_dh(&server_x.secret_key, &X25519PublicKey::from_bytes(client_x)).unwrap();
            (ss.to_vec(), server_x.public_key.as_bytes().to_vec())
        };

        let sh_msg = build_server_hello_msg(suite, group, &server_share);
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
        let ee_msg = build_ee_msg();
        let cert_msg = build_certificate_msg(LEAF_DER);
        transcript.update(&ee_msg);
        transcript.update(&cert_msg);
        let mut coalesced = ee_msg.clone();
        coalesced.extend_from_slice(&cert_msg);
        records.extend_from_slice(&seal_hs(&coalesced, &mut seq));

        // CertificateVerify over Transcript-Hash(CH..Certificate).
        let cv_msg = build_cv_msg(cert_kp, &transcript.current());
        transcript.update(&cv_msg);
        records.extend_from_slice(&seal_hs(&cv_msg, &mut seq));

        // Server Finished over Transcript-Hash(CH..CV).
        let mut vd = schedule
            .finished_verify_data(
                schedule.server_hs_traffic_secret().unwrap(),
                &transcript.current(),
            )
            .unwrap();
        if tamper_finished {
            vd[0] ^= 0xff;
        }
        let fin_msg = build_finished(&vd).unwrap();
        transcript.update(&fin_msg);
        let mut fin_payload = fin_msg;
        if junk_after_finished {
            fin_payload.extend_from_slice(&[0u8; 7]);
        }
        records.extend_from_slice(&seal_hs(&fin_payload, &mut seq));

        ServerFlight {
            records,
            schedule,
            transcript,
            suite,
        }
    }

    fn entropy() -> ClientEntropy {
        ClientEntropy {
            client_random: [0x11; 32],
            x25519_seed: [0x22; 32],
            ml_kem_seed: [0x33; 64],
        }
    }

    fn run_e2e(suite: CipherSuite, require_pqc: bool, fragment: bool) {
        let cert_kp = ed25519_keygen(&[9u8; 32]);
        let verifier = TestVerifier {
            expected_leaf: LEAF_DER,
            pubkey: *cert_kp.public_key.as_bytes(),
        };
        let (mut client, flight) = ClientHandshake::new(
            ClientConfig {
                server_name: Some("immudb.example.com"),
                require_pqc,
            },
            &entropy(),
            &verifier,
        )
        .unwrap();

        let mut server = server_respond(&flight, suite, &cert_kp, false, false);

        // Feed the server flight to the client — all at once, or
        // byte-by-byte to exercise the reassembly paths.
        let mut sent = Vec::new();
        let mut status = HandshakeStatus::InProgress;
        if fragment {
            for b in server.records.clone() {
                let p = client.push(&[b]).unwrap();
                sent.extend_from_slice(&p.send);
                status = p.status;
            }
        } else {
            let p = client.push(&server.records.clone()).unwrap();
            sent.extend_from_slice(&p.send);
            status = p.status;
        }
        assert_eq!(status, HandshakeStatus::Complete);
        assert_eq!(client.suite(), Some(suite));
        assert!(client.sni_acked());

        // Server verifies the client Finished (over the transcript
        // through the server Finished) and compares app keys.
        let client_keys = server
            .schedule
            .traffic_keys(server.schedule.client_hs_traffic_secret().unwrap())
            .unwrap();
        let (inner, fin_plain) =
            record::open(server.suite, &client_keys.key, &client_keys.iv, 0, &sent).unwrap();
        assert_eq!(inner, ContentType::Handshake);
        let alg = HashAlg::for_suite(suite);
        let received = parse_finished(&fin_plain, alg.digest_len()).unwrap();
        let expected = server
            .schedule
            .finished_verify_data(
                server.schedule.client_hs_traffic_secret().unwrap(),
                &server.transcript.current(),
            )
            .unwrap();
        check_verify_data(&received, &expected).unwrap();

        server
            .schedule
            .derive_application_secrets(&server.transcript.current())
            .unwrap();
        let client_app = client.app_keys().unwrap();
        let server_view_client = server
            .schedule
            .traffic_keys(server.schedule.client_ap_traffic_secret().unwrap())
            .unwrap();
        let server_view_server = server
            .schedule
            .traffic_keys(server.schedule.server_ap_traffic_secret().unwrap())
            .unwrap();
        assert_eq!(client_app.client.key, server_view_client.key);
        assert_eq!(client_app.client.iv, server_view_client.iv);
        assert_eq!(client_app.server.key, server_view_server.key);
        assert_eq!(client_app.server.iv, server_view_server.iv);
    }

    #[test]
    fn e2e_chacha20_classical() {
        run_e2e(CipherSuite::ChaCha20Poly1305Sha256, false, false);
    }

    #[test]
    fn e2e_aes256_classical() {
        run_e2e(CipherSuite::Aes256GcmSha384, false, false);
    }

    #[test]
    fn e2e_hybrid_pqc() {
        run_e2e(CipherSuite::ChaCha20Poly1305Sha256, true, false);
    }

    #[test]
    fn e2e_byte_at_a_time_fragmentation() {
        run_e2e(CipherSuite::Aes256GcmSha384, false, true);
    }

    fn start_classical() -> (ClientHandshake<'static, TestVerifier>, Vec<u8>) {
        static VERIFIER: TestVerifier = TestVerifier {
            expected_leaf: LEAF_DER,
            pubkey: [0u8; 32],
        };
        let (client, flight) = ClientHandshake::new(
            ClientConfig {
                server_name: Some("immudb.example.com"),
                require_pqc: false,
            },
            &entropy(),
            &VERIFIER,
        )
        .unwrap();
        (client, flight)
    }

    #[test]
    fn server_selecting_tls12_aborts() {
        let (mut client, _flight) = start_classical();
        let x = x25519_keygen(&[0x55u8; 32]);
        let sh = build_server_hello_custom(
            0x1302,
            Some(0x0303),
            named_group::X25519,
            x.public_key.as_bytes(),
            [0xaa; 32],
        );
        let rec = build_plaintext_record(ContentType::Handshake, &sh).unwrap();
        assert_eq!(client.push(&rec).unwrap_err(), TlsClientError::Version);
        // Driver latches Failed.
        assert_eq!(client.push(&[]).unwrap_err(), TlsClientError::BadHandshake);
    }

    #[test]
    fn hello_retry_request_refused() {
        let (mut client, _flight) = start_classical();
        let x = x25519_keygen(&[0x55u8; 32]);
        let sh = build_server_hello_custom(
            0x1302,
            Some(TLS_1_3),
            named_group::X25519,
            x.public_key.as_bytes(),
            HRR_RANDOM,
        );
        let rec = build_plaintext_record(ContentType::Handshake, &sh).unwrap();
        assert_eq!(client.push(&rec).unwrap_err(), TlsClientError::BadHandshake);
    }

    #[test]
    fn pqc_downgrade_detected() {
        let cert_kp = ed25519_keygen(&[9u8; 32]);
        let verifier = TestVerifier {
            expected_leaf: LEAF_DER,
            pubkey: *cert_kp.public_key.as_bytes(),
        };
        let (mut client, _flight) = ClientHandshake::new(
            ClientConfig {
                server_name: None,
                require_pqc: true,
            },
            &entropy(),
            &verifier,
        )
        .unwrap();
        // Server replies with a classical x25519 share despite the
        // hybrid offer.
        let x = x25519_keygen(&[0x55u8; 32]);
        let sh = build_server_hello_custom(
            0x1303,
            Some(TLS_1_3),
            named_group::X25519,
            x.public_key.as_bytes(),
            [0xaa; 32],
        );
        let rec = build_plaintext_record(ContentType::Handshake, &sh).unwrap();
        assert_eq!(client.push(&rec).unwrap_err(), TlsClientError::PqcDowngrade);
    }

    #[test]
    fn tampered_server_finished_rejected() {
        let cert_kp = ed25519_keygen(&[9u8; 32]);
        let verifier = TestVerifier {
            expected_leaf: LEAF_DER,
            pubkey: *cert_kp.public_key.as_bytes(),
        };
        let (mut client, flight) = ClientHandshake::new(
            ClientConfig {
                server_name: Some("immudb.example.com"),
                require_pqc: false,
            },
            &entropy(),
            &verifier,
        )
        .unwrap();
        let server = server_respond(
            &flight,
            CipherSuite::ChaCha20Poly1305Sha256,
            &cert_kp,
            true, // tamper the Finished verify_data
            false,
        );
        assert_eq!(
            client.push(&server.records).unwrap_err(),
            TlsClientError::BadHandshake
        );
    }

    #[test]
    fn untrusted_chain_aborts() {
        let cert_kp = ed25519_keygen(&[9u8; 32]);
        let verifier = RejectingVerifier;
        let (mut client, flight) = ClientHandshake::new(
            ClientConfig {
                server_name: Some("immudb.example.com"),
                require_pqc: false,
            },
            &entropy(),
            &verifier,
        )
        .unwrap();
        let server = server_respond(
            &flight,
            CipherSuite::ChaCha20Poly1305Sha256,
            &cert_kp,
            false,
            false,
        );
        assert_eq!(
            client.push(&server.records).unwrap_err(),
            TlsClientError::ChainUntrusted
        );
    }

    #[test]
    fn wrong_cert_verify_signature_rejected() {
        let cert_kp = ed25519_keygen(&[9u8; 32]);
        let wrong_kp = ed25519_keygen(&[10u8; 32]);
        // Verifier returns a pubkey that does NOT match the key
        // the server signed CertificateVerify with.
        let verifier = TestVerifier {
            expected_leaf: LEAF_DER,
            pubkey: *wrong_kp.public_key.as_bytes(),
        };
        let (mut client, flight) = ClientHandshake::new(
            ClientConfig {
                server_name: Some("immudb.example.com"),
                require_pqc: false,
            },
            &entropy(),
            &verifier,
        )
        .unwrap();
        let server = server_respond(
            &flight,
            CipherSuite::ChaCha20Poly1305Sha256,
            &cert_kp,
            false,
            false,
        );
        assert_eq!(
            client.push(&server.records).unwrap_err(),
            TlsClientError::BadCertificate
        );
    }

    #[test]
    fn certificate_request_refused() {
        let cert_kp = ed25519_keygen(&[9u8; 32]);
        let verifier = TestVerifier {
            expected_leaf: LEAF_DER,
            pubkey: *cert_kp.public_key.as_bytes(),
        };
        let (mut client, flight) = ClientHandshake::new(
            ClientConfig {
                server_name: Some("immudb.example.com"),
                require_pqc: false,
            },
            &entropy(),
            &verifier,
        )
        .unwrap();
        // Build a normal flight, then splice a CertificateRequest
        // where the Certificate would go.
        let ch_msg = &flight[RECORD_HEADER_LEN..];
        let (group, client_share) = parse_ch_key_share(ch_msg);
        let server_x = x25519_keygen(&[0x55u8; 32]);
        let mut client_x = [0u8; 32];
        client_x.copy_from_slice(&client_share);
        let ecdhe =
            x25519_dh(&server_x.secret_key, &X25519PublicKey::from_bytes(client_x)).unwrap();
        let suite = CipherSuite::ChaCha20Poly1305Sha256;
        let sh_msg = build_server_hello_msg(suite, group, server_x.public_key.as_bytes());
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
        let ee = build_ee_msg();
        // CertificateRequest: context len 0 + empty extensions.
        let mut cr_body = alloc::vec![0u8];
        cr_body.extend_from_slice(&0u16.to_be_bytes());
        let cr = crate::handshake::test_util::wrap_handshake(
            HandshakeType::CertificateRequest,
            &cr_body,
        );
        let mut payload = ee;
        payload.extend_from_slice(&cr);
        records.extend_from_slice(
            &record::seal(
                suite,
                &server_keys.key,
                &server_keys.iv,
                0,
                ContentType::Handshake,
                &payload,
            )
            .unwrap(),
        );
        assert_eq!(
            client.push(&records).unwrap_err(),
            TlsClientError::BadHandshake
        );
    }

    #[test]
    fn ccs_record_ignored_mid_handshake() {
        let cert_kp = ed25519_keygen(&[9u8; 32]);
        let verifier = TestVerifier {
            expected_leaf: LEAF_DER,
            pubkey: *cert_kp.public_key.as_bytes(),
        };
        let (mut client, flight) = ClientHandshake::new(
            ClientConfig {
                server_name: Some("immudb.example.com"),
                require_pqc: false,
            },
            &entropy(),
            &verifier,
        )
        .unwrap();
        let server = server_respond(
            &flight,
            CipherSuite::ChaCha20Poly1305Sha256,
            &cert_kp,
            false,
            false,
        );
        // Inject a middlebox-compat CCS record between the
        // ServerHello and the encrypted flight.
        let sh_len =
            RECORD_HEADER_LEN + u16::from_be_bytes([server.records[3], server.records[4]]) as usize;
        let mut with_ccs = server.records[..sh_len].to_vec();
        with_ccs.extend_from_slice(&[20, 0x03, 0x03, 0, 1, 1]); // CCS record
        with_ccs.extend_from_slice(&server.records[sh_len..]);
        let p = client.push(&with_ccs).unwrap();
        assert_eq!(p.status, HandshakeStatus::Complete);
    }

    #[test]
    fn app_keys_gated_until_complete() {
        let (client, _flight) = start_classical();
        assert!(client.app_keys().is_err());
    }

    /// A valid classical ServerHello message answering
    /// `start_classical`'s ClientHello (any well-formed x25519
    /// public key works — the client just runs DH with it).
    fn classical_server_hello_msg() -> Vec<u8> {
        let x = x25519_keygen(&[0x55u8; 32]);
        build_server_hello_msg(
            CipherSuite::ChaCha20Poly1305Sha256,
            named_group::X25519,
            x.public_key.as_bytes(),
        )
    }

    #[test]
    fn plaintext_bytes_coalesced_after_server_hello_abort() {
        // RFC 8446 §5.1 key-change boundary: one PLAINTEXT record
        // carrying ServerHello || EncryptedExtensions. The EE bytes
        // sit on the wrong side of the key change and must not be
        // processed as if they had been encrypted.
        let (mut client, _flight) = start_classical();
        let mut payload = classical_server_hello_msg();
        payload.extend_from_slice(&build_ee_msg());
        let rec = build_plaintext_record(ContentType::Handshake, &payload).unwrap();
        assert_eq!(client.push(&rec).unwrap_err(), TlsClientError::BadHandshake);
    }

    #[test]
    fn leftover_bytes_at_server_finished_abort() {
        // Junk coalesced into the (encrypted) server Finished
        // record must abort instead of being silently retained
        // across the handshake-end key boundary.
        let cert_kp = ed25519_keygen(&[9u8; 32]);
        let verifier = TestVerifier {
            expected_leaf: LEAF_DER,
            pubkey: *cert_kp.public_key.as_bytes(),
        };
        let (mut client, flight) = ClientHandshake::new(
            ClientConfig {
                server_name: Some("immudb.example.com"),
                require_pqc: false,
            },
            &entropy(),
            &verifier,
        )
        .unwrap();
        let server = server_respond(
            &flight,
            CipherSuite::ChaCha20Poly1305Sha256,
            &cert_kp,
            false,
            true, // coalesce junk bytes after Finished
        );
        assert_eq!(
            client.push(&server.records).unwrap_err(),
            TlsClientError::BadHandshake
        );
    }

    #[test]
    fn non_empty_session_id_echo_aborts() {
        // RFC 8446 §4.1.3: the echo must match our (always empty)
        // legacy_session_id.
        let (mut client, _flight) = start_classical();
        let x = x25519_keygen(&[0x55u8; 32]);
        let sh = crate::handshake::test_util::build_server_hello_with_session_id(
            CipherSuite::ChaCha20Poly1305Sha256.wire_value(),
            Some(TLS_1_3),
            named_group::X25519,
            x.public_key.as_bytes(),
            [0xaa; 32],
            &[0xde, 0xad, 0xbe, 0xef],
        );
        let rec = build_plaintext_record(ContentType::Handshake, &sh).unwrap();
        assert_eq!(client.push(&rec).unwrap_err(), TlsClientError::BadHandshake);
    }

    #[test]
    fn ccs_with_wrong_payload_byte_aborts() {
        // RFC 8446 §5: the compat CCS is exactly one 0x01 byte.
        let (mut client, _flight) = start_classical();
        assert_eq!(
            client.push(&[20, 0x03, 0x03, 0, 1, 2]).unwrap_err(),
            TlsClientError::BadHandshake
        );
    }

    #[test]
    fn ccs_with_wrong_length_aborts() {
        let (mut client, _flight) = start_classical();
        assert_eq!(
            client.push(&[20, 0x03, 0x03, 0, 2, 1, 1]).unwrap_err(),
            TlsClientError::BadHandshake
        );
    }

    #[test]
    fn ccs_flood_aborts() {
        let (mut client, _flight) = start_classical();
        // Up to MAX_CCS_RECORDS well-formed CCS records are
        // dropped per RFC 8446 §5...
        for _ in 0..MAX_CCS_RECORDS {
            client.push(&[20, 0x03, 0x03, 0, 1, 1]).unwrap();
        }
        // ...but the junk tolerance is bounded.
        assert_eq!(
            client.push(&[20, 0x03, 0x03, 0, 1, 1]).unwrap_err(),
            TlsClientError::BadHandshake
        );
    }

    #[test]
    fn oversized_handshake_length_claim_aborts() {
        // A ServerHello header claiming 2^24 - 1 bytes is rejected
        // from the 4 header bytes alone — the driver never
        // accumulates records toward the bogus claim.
        let (mut client, _flight) = start_classical();
        let rec = build_plaintext_record(ContentType::Handshake, &[2, 0xff, 0xff, 0xff]).unwrap();
        assert_eq!(client.push(&rec).unwrap_err(), TlsClientError::BadHandshake);
    }

    #[test]
    fn certificate_length_cap_wider_but_enforced() {
        // Certificate (type 11) gets the wider cap; one byte over
        // it still aborts.
        let (mut client, _flight) = start_classical();
        let over = ((MAX_CERTIFICATE_MSG_LEN + 1) as u32).to_be_bytes();
        let rec = build_plaintext_record(ContentType::Handshake, &[11, over[1], over[2], over[3]])
            .unwrap();
        assert_eq!(client.push(&rec).unwrap_err(), TlsClientError::BadHandshake);

        // A claim at the cap is not rejected at the header stage —
        // the driver just waits for the rest of the message.
        let (mut client2, _flight2) = start_classical();
        let at_cap = (MAX_CERTIFICATE_MSG_LEN as u32).to_be_bytes();
        let rec2 = build_plaintext_record(
            ContentType::Handshake,
            &[11, at_cap[1], at_cap[2], at_cap[3]],
        )
        .unwrap();
        let p = client2.push(&rec2).unwrap();
        assert_eq!(p.status, HandshakeStatus::InProgress);
    }
}
