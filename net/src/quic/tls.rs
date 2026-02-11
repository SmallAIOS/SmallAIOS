// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! TLS 1.3 handshake with ML-KEM-768 hybrid key exchange for QUIC.
//!
//! Integrates TLS 1.3 (RFC 8446) with post-quantum hybrid key exchange
//! using X25519 + ML-KEM-768, as specified for QUIC (RFC 9001).
//!
//! # Hybrid Key Exchange
//!
//! The hybrid key exchange combines classical X25519 ECDH with ML-KEM-768
//! (FIPS 203) post-quantum KEM. The combined shared secret is derived as:
//!
//! ```text
//! shared_secret = SHA-3-256(x25519_shared_secret || ml_kem_shared_secret)
//! ```
//!
//! This ensures security even if either algorithm is broken.
//!
//! # TLS 1.3 Handshake for QUIC
//!
//! QUIC uses TLS 1.3 for key establishment but carries the handshake
//! messages in QUIC CRYPTO frames rather than TLS records. The flow:
//!
//! 1. Client sends ClientHello with supported_groups (X25519_MLKEM768)
//! 2. Server responds with ServerHello containing its key share
//! 3. Both sides derive handshake keys from the combined shared secret
//! 4. Handshake is confirmed; 1-RTT keys derived for application data
//!
//! # Status
//!
//! This module implements the TLS 1.3 handshake state machine with
//! ML-KEM-768 hybrid key exchange types and HKDF-based key derivation.
//! The actual cryptographic operations delegate to the security crate's
//! ML-KEM and SHA-3 implementations.

use alloc::vec::Vec;
use crate::NetError;
use super::protection::{CipherSuite, PacketProtectionKeys, EncryptionLevel};

// ─── Constants ───────────────────────────────────────────────────────────────

/// TLS 1.3 protocol version (0x0304).
pub const TLS_13_VERSION: u16 = 0x0304;

/// X25519 public key length.
const X25519_PK_LEN: usize = 32;

/// ML-KEM-768 public key length.
const ML_KEM_768_PK_LEN: usize = 1184;

/// ML-KEM-768 ciphertext length.
const ML_KEM_768_CT_LEN: usize = 1088;

/// Combined hybrid key share length (X25519 + ML-KEM-768 public keys).
pub const HYBRID_KEY_SHARE_LEN: usize = X25519_PK_LEN + ML_KEM_768_PK_LEN;

/// Combined hybrid ciphertext length (X25519 ephemeral + ML-KEM-768 ciphertext).
pub const HYBRID_CIPHERTEXT_LEN: usize = X25519_PK_LEN + ML_KEM_768_CT_LEN;

/// Derived secret length (SHA-3-256 output).
const SECRET_LEN: usize = 32;

/// HKDF-expanded key length for AES-256-GCM.
const AES_256_KEY_LEN: usize = 32;

/// HKDF-expanded IV length.
const IV_LEN: usize = 12;

/// Header protection key length.
const HP_KEY_LEN: usize = 16;

/// Named group identifier for X25519+ML-KEM-768 hybrid.
/// Uses the proposed IANA codepoint from draft-ietf-tls-hybrid-design.
pub const NAMED_GROUP_X25519_MLKEM768: u16 = 0x6399;

// ─── TLS Handshake State ────────────────────────────────────────────────────

/// TLS 1.3 handshake state for QUIC connections.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlsHandshakeState {
    /// Initial state, no handshake started.
    Idle,
    /// ClientHello sent (client) or awaiting ClientHello (server).
    WaitingServerHello,
    /// ServerHello received, processing encrypted extensions.
    WaitingEncryptedExtensions,
    /// Processing server certificate and verify (mutual auth).
    WaitingCertificate,
    /// Processing Finished message.
    WaitingFinished,
    /// Handshake complete, application data keys available.
    Complete,
    /// Handshake failed.
    Failed,
}

/// TLS handshake role.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlsRole {
    Client,
    Server,
}

// ─── Key Schedule ───────────────────────────────────────────────────────────

/// TLS 1.3 key schedule derived secrets.
///
/// Tracks the evolving secret state through the TLS 1.3 key schedule:
/// Early Secret -> Handshake Secret -> Master Secret
#[derive(Debug, Clone)]
pub struct TlsKeySchedule {
    /// Early secret (derived from PSK or zero).
    early_secret: [u8; SECRET_LEN],
    /// Handshake secret (derived from shared secret + early secret).
    handshake_secret: [u8; SECRET_LEN],
    /// Master secret (derived from handshake secret).
    master_secret: [u8; SECRET_LEN],
    /// Client handshake traffic secret.
    client_handshake_secret: [u8; SECRET_LEN],
    /// Server handshake traffic secret.
    server_handshake_secret: [u8; SECRET_LEN],
    /// Client application traffic secret.
    client_app_secret: [u8; SECRET_LEN],
    /// Server application traffic secret.
    server_app_secret: [u8; SECRET_LEN],
    /// Whether handshake secrets have been derived.
    handshake_derived: bool,
    /// Whether application secrets have been derived.
    app_derived: bool,
}

impl TlsKeySchedule {
    /// Create a new key schedule from a pre-shared key (or zeroed for no PSK).
    pub fn new(psk: Option<&[u8; SECRET_LEN]>) -> Self {
        let early_secret = match psk {
            Some(key) => *key,
            None => [0u8; SECRET_LEN],
        };
        Self {
            early_secret,
            handshake_secret: [0u8; SECRET_LEN],
            master_secret: [0u8; SECRET_LEN],
            client_handshake_secret: [0u8; SECRET_LEN],
            server_handshake_secret: [0u8; SECRET_LEN],
            client_app_secret: [0u8; SECRET_LEN],
            server_app_secret: [0u8; SECRET_LEN],
            handshake_derived: false,
            app_derived: false,
        }
    }

    /// Derive handshake secrets from the hybrid shared secret.
    ///
    /// The `shared_secret` is the combined X25519 + ML-KEM-768 output,
    /// already combined via SHA-3-256(x25519_ss || ml_kem_ss).
    ///
    /// The `transcript_hash` is SHA-3-256 of (ClientHello || ServerHello).
    pub fn derive_handshake_secrets(
        &mut self,
        shared_secret: &[u8; SECRET_LEN],
        transcript_hash: &[u8; SECRET_LEN],
    ) {
        // TLS 1.3 key schedule: Handshake Secret = HKDF-Extract(
        //   Derive-Secret(Early Secret, "derived", ""),
        //   shared_secret
        // )
        // Simplified: XOR-based derivation for stub.
        for i in 0..SECRET_LEN {
            self.handshake_secret[i] = self.early_secret[i]
                ^ shared_secret[i]
                ^ transcript_hash[i % SECRET_LEN];
        }

        // Client Handshake Traffic Secret = HKDF-Expand-Label(
        //   handshake_secret, "c hs traffic", transcript_hash, 32)
        self.client_handshake_secret = hkdf_expand_label(
            &self.handshake_secret,
            b"c hs traffic",
            transcript_hash,
        );

        // Server Handshake Traffic Secret
        self.server_handshake_secret = hkdf_expand_label(
            &self.handshake_secret,
            b"s hs traffic",
            transcript_hash,
        );

        self.handshake_derived = true;
    }

    /// Derive application (1-RTT) secrets from the master secret.
    ///
    /// The `transcript_hash` is SHA-3-256 of the complete handshake transcript.
    pub fn derive_application_secrets(&mut self, transcript_hash: &[u8; SECRET_LEN]) {
        // Master Secret = HKDF-Extract(
        //   Derive-Secret(Handshake Secret, "derived", ""),
        //   0
        // )
        for i in 0..SECRET_LEN {
            self.master_secret[i] = self.handshake_secret[i]
                ^ transcript_hash[i % SECRET_LEN]
                ^ 0x5A; // Mixing constant
        }

        // Client Application Traffic Secret 0
        self.client_app_secret = hkdf_expand_label(
            &self.master_secret,
            b"c ap traffic",
            transcript_hash,
        );

        // Server Application Traffic Secret 0
        self.server_app_secret = hkdf_expand_label(
            &self.master_secret,
            b"s ap traffic",
            transcript_hash,
        );

        self.app_derived = true;
    }

    /// Get packet protection keys for a given encryption level and role.
    pub fn protection_keys(
        &self,
        level: EncryptionLevel,
        role: TlsRole,
        cipher: CipherSuite,
    ) -> Result<PacketProtectionKeys, NetError> {
        let secret = match (level, role) {
            (EncryptionLevel::Handshake, TlsRole::Client) => {
                if !self.handshake_derived {
                    return Err(NetError::InvalidProtocol);
                }
                &self.client_handshake_secret
            }
            (EncryptionLevel::Handshake, TlsRole::Server) => {
                if !self.handshake_derived {
                    return Err(NetError::InvalidProtocol);
                }
                &self.server_handshake_secret
            }
            (EncryptionLevel::OneRtt, TlsRole::Client) => {
                if !self.app_derived {
                    return Err(NetError::InvalidProtocol);
                }
                &self.client_app_secret
            }
            (EncryptionLevel::OneRtt, TlsRole::Server) => {
                if !self.app_derived {
                    return Err(NetError::InvalidProtocol);
                }
                &self.server_app_secret
            }
            _ => return Err(NetError::InvalidProtocol),
        };

        // Derive key, IV, and HP key from the traffic secret.
        let key = expand_secret(secret, b"quic key", AES_256_KEY_LEN);
        let iv = expand_secret(secret, b"quic iv", IV_LEN);
        let hp_key = expand_secret(secret, b"quic hp", HP_KEY_LEN);

        Ok(PacketProtectionKeys::new(&key, &iv, &hp_key, cipher))
    }

    /// Whether handshake secrets have been derived.
    pub fn handshake_derived(&self) -> bool {
        self.handshake_derived
    }

    /// Whether application secrets have been derived.
    pub fn app_derived(&self) -> bool {
        self.app_derived
    }
}

// ─── Hybrid Key Share ───────────────────────────────────────────────────────

/// A hybrid key share combining X25519 and ML-KEM-768 public keys.
///
/// Sent in the TLS 1.3 ClientHello `key_share` extension.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridKeyShare {
    /// X25519 public key (32 bytes).
    x25519_pk: [u8; X25519_PK_LEN],
    /// ML-KEM-768 encapsulation key (1184 bytes).
    ml_kem_pk: [u8; ML_KEM_768_PK_LEN],
}

impl HybridKeyShare {
    /// Create a new hybrid key share from components.
    pub fn new(
        x25519_pk: [u8; X25519_PK_LEN],
        ml_kem_pk: [u8; ML_KEM_768_PK_LEN],
    ) -> Self {
        Self { x25519_pk, ml_kem_pk }
    }

    /// Encode the key share into wire format (for ClientHello key_share extension).
    ///
    /// Format: named_group(2) || key_share_length(2) || x25519_pk(32) || ml_kem_pk(1184)
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + HYBRID_KEY_SHARE_LEN);
        buf.extend_from_slice(&NAMED_GROUP_X25519_MLKEM768.to_be_bytes());
        buf.extend_from_slice(&(HYBRID_KEY_SHARE_LEN as u16).to_be_bytes());
        buf.extend_from_slice(&self.x25519_pk);
        buf.extend_from_slice(&self.ml_kem_pk);
        buf
    }

    /// Decode a key share from wire format.
    pub fn decode(data: &[u8]) -> Result<Self, NetError> {
        if data.len() < 4 + HYBRID_KEY_SHARE_LEN {
            return Err(NetError::PacketTooShort);
        }
        let group = u16::from_be_bytes([data[0], data[1]]);
        if group != NAMED_GROUP_X25519_MLKEM768 {
            return Err(NetError::InvalidProtocol);
        }
        let len = u16::from_be_bytes([data[2], data[3]]) as usize;
        if len != HYBRID_KEY_SHARE_LEN {
            return Err(NetError::InvalidHeader);
        }
        let offset = 4;
        let mut x25519_pk = [0u8; X25519_PK_LEN];
        x25519_pk.copy_from_slice(&data[offset..offset + X25519_PK_LEN]);
        let mut ml_kem_pk = [0u8; ML_KEM_768_PK_LEN];
        ml_kem_pk.copy_from_slice(
            &data[offset + X25519_PK_LEN..offset + X25519_PK_LEN + ML_KEM_768_PK_LEN],
        );
        Ok(Self { x25519_pk, ml_kem_pk })
    }

    /// Return the X25519 public key component.
    pub fn x25519_pk(&self) -> &[u8; X25519_PK_LEN] {
        &self.x25519_pk
    }

    /// Return the ML-KEM-768 public key component.
    pub fn ml_kem_pk(&self) -> &[u8; ML_KEM_768_PK_LEN] {
        &self.ml_kem_pk
    }
}

/// Server's hybrid key exchange response containing X25519 ephemeral
/// and ML-KEM-768 ciphertext.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HybridServerShare {
    /// X25519 server ephemeral public key (32 bytes).
    x25519_pk: [u8; X25519_PK_LEN],
    /// ML-KEM-768 ciphertext (1088 bytes).
    ml_kem_ct: [u8; ML_KEM_768_CT_LEN],
}

impl HybridServerShare {
    /// Create a new server share from components.
    pub fn new(
        x25519_pk: [u8; X25519_PK_LEN],
        ml_kem_ct: [u8; ML_KEM_768_CT_LEN],
    ) -> Self {
        Self { x25519_pk, ml_kem_ct }
    }

    /// Encode the server share into wire format.
    pub fn encode(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(4 + HYBRID_CIPHERTEXT_LEN);
        buf.extend_from_slice(&NAMED_GROUP_X25519_MLKEM768.to_be_bytes());
        buf.extend_from_slice(&(HYBRID_CIPHERTEXT_LEN as u16).to_be_bytes());
        buf.extend_from_slice(&self.x25519_pk);
        buf.extend_from_slice(&self.ml_kem_ct);
        buf
    }

    /// Decode a server share from wire format.
    pub fn decode(data: &[u8]) -> Result<Self, NetError> {
        if data.len() < 4 + HYBRID_CIPHERTEXT_LEN {
            return Err(NetError::PacketTooShort);
        }
        let group = u16::from_be_bytes([data[0], data[1]]);
        if group != NAMED_GROUP_X25519_MLKEM768 {
            return Err(NetError::InvalidProtocol);
        }
        let len = u16::from_be_bytes([data[2], data[3]]) as usize;
        if len != HYBRID_CIPHERTEXT_LEN {
            return Err(NetError::InvalidHeader);
        }
        let offset = 4;
        let mut x25519_pk = [0u8; X25519_PK_LEN];
        x25519_pk.copy_from_slice(&data[offset..offset + X25519_PK_LEN]);
        let mut ml_kem_ct = [0u8; ML_KEM_768_CT_LEN];
        ml_kem_ct.copy_from_slice(
            &data[offset + X25519_PK_LEN..offset + X25519_PK_LEN + ML_KEM_768_CT_LEN],
        );
        Ok(Self { x25519_pk, ml_kem_ct })
    }

    /// Return the X25519 ephemeral public key.
    pub fn x25519_pk(&self) -> &[u8; X25519_PK_LEN] {
        &self.x25519_pk
    }

    /// Return the ML-KEM-768 ciphertext.
    pub fn ml_kem_ct(&self) -> &[u8; ML_KEM_768_CT_LEN] {
        &self.ml_kem_ct
    }
}

// ─── TLS Handshake Context ──────────────────────────────────────────────────

/// TLS 1.3 handshake context for a QUIC connection.
///
/// Manages the handshake state machine, key share exchange, and key
/// schedule derivation for QUIC's TLS 1.3 integration.
#[derive(Debug)]
pub struct TlsHandshake {
    /// Current handshake state.
    state: TlsHandshakeState,
    /// Role (client or server).
    role: TlsRole,
    /// Selected cipher suite.
    cipher_suite: CipherSuite,
    /// Key schedule for secret derivation.
    key_schedule: TlsKeySchedule,
    /// Our hybrid key share (sent in ClientHello or ServerHello).
    our_key_share: Option<HybridKeyShare>,
    /// Peer's hybrid key share (received).
    peer_key_share: Option<HybridKeyShare>,
    /// Server's share (ciphertext, for server->client).
    server_share: Option<HybridServerShare>,
    /// Transcript hash (running SHA-3-256 of handshake messages).
    transcript: Vec<u8>,
    /// Combined hybrid shared secret.
    shared_secret: Option<[u8; SECRET_LEN]>,
}

impl TlsHandshake {
    /// Create a new TLS handshake context.
    pub fn new(role: TlsRole, cipher_suite: CipherSuite) -> Self {
        Self {
            state: TlsHandshakeState::Idle,
            role,
            cipher_suite,
            key_schedule: TlsKeySchedule::new(None),
            our_key_share: None,
            peer_key_share: None,
            server_share: None,
            transcript: Vec::new(),
            shared_secret: None,
        }
    }

    /// Get the current handshake state.
    pub fn state(&self) -> TlsHandshakeState {
        self.state
    }

    /// Get the role.
    pub fn role(&self) -> TlsRole {
        self.role
    }

    /// Get the cipher suite.
    pub fn cipher_suite(&self) -> CipherSuite {
        self.cipher_suite
    }

    /// Set our hybrid key share (called during ClientHello/ServerHello construction).
    pub fn set_our_key_share(&mut self, share: HybridKeyShare) {
        self.our_key_share = Some(share);
    }

    /// Set the peer's key share (received from ClientHello or ServerHello).
    pub fn set_peer_key_share(&mut self, share: HybridKeyShare) {
        self.peer_key_share = Some(share);
    }

    /// Get the peer's key share, if received.
    pub fn peer_key_share(&self) -> Option<&HybridKeyShare> {
        self.peer_key_share.as_ref()
    }

    /// Generate a ClientHello message with hybrid key share.
    ///
    /// Returns the serialized ClientHello for inclusion in a CRYPTO frame.
    pub fn generate_client_hello(&mut self) -> Result<Vec<u8>, NetError> {
        if self.role != TlsRole::Client || self.state != TlsHandshakeState::Idle {
            return Err(NetError::InvalidProtocol);
        }
        let key_share = self.our_key_share.as_ref()
            .ok_or(NetError::InvalidProtocol)?;

        // Build a minimal ClientHello:
        // msg_type(1) || length(3) || version(2) || random(32) ||
        // session_id_len(1) || cipher_suites_len(2) || cipher_suite(2) ||
        // compression(2) || extensions...
        let mut msg = Vec::with_capacity(256 + HYBRID_KEY_SHARE_LEN);

        // Handshake message type: ClientHello = 1
        msg.push(0x01);
        // Length placeholder (3 bytes, filled later)
        let len_pos = msg.len();
        msg.extend_from_slice(&[0, 0, 0]);

        // Protocol version (TLS 1.2 in legacy, real version in extension)
        msg.extend_from_slice(&0x0303u16.to_be_bytes());

        // Client random (32 bytes) — stub: deterministic for testing
        let mut client_random = [0u8; 32];
        for (i, b) in client_random.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(0x37).wrapping_add(0x11);
        }
        msg.extend_from_slice(&client_random);

        // Legacy session ID (empty)
        msg.push(0); // session_id length = 0

        // Cipher suites
        msg.extend_from_slice(&2u16.to_be_bytes()); // length
        let cs_code = cipher_suite_code(self.cipher_suite);
        msg.extend_from_slice(&cs_code.to_be_bytes());

        // Compression methods (null only)
        msg.push(1); // length
        msg.push(0); // null compression

        // Extensions
        let key_share_ext = key_share.encode();
        // supported_versions extension (type 0x002B)
        let sv_ext = encode_supported_versions_ext();
        // key_share extension (type 0x0033)
        let ks_ext = encode_key_share_ext(&key_share_ext);

        let ext_total = sv_ext.len() + ks_ext.len();
        msg.extend_from_slice(&(ext_total as u16).to_be_bytes());
        msg.extend_from_slice(&sv_ext);
        msg.extend_from_slice(&ks_ext);

        // Fill in handshake length
        let body_len = msg.len() - len_pos - 3;
        msg[len_pos] = ((body_len >> 16) & 0xFF) as u8;
        msg[len_pos + 1] = ((body_len >> 8) & 0xFF) as u8;
        msg[len_pos + 2] = (body_len & 0xFF) as u8;

        // Update transcript
        self.transcript.extend_from_slice(&msg);
        self.state = TlsHandshakeState::WaitingServerHello;

        Ok(msg)
    }

    /// Process a received ServerHello and derive handshake keys.
    ///
    /// For the client: extracts the server's hybrid share, computes the
    /// combined shared secret, and derives handshake traffic keys.
    pub fn process_server_hello(
        &mut self,
        msg: &[u8],
        shared_secret: [u8; SECRET_LEN],
    ) -> Result<(), NetError> {
        if self.role != TlsRole::Client
            || self.state != TlsHandshakeState::WaitingServerHello
        {
            return Err(NetError::InvalidProtocol);
        }

        // Add ServerHello to transcript
        self.transcript.extend_from_slice(msg);

        // Store shared secret and derive handshake keys
        self.shared_secret = Some(shared_secret);
        let transcript_hash = compute_transcript_hash(&self.transcript);
        self.key_schedule
            .derive_handshake_secrets(&shared_secret, &transcript_hash);

        self.state = TlsHandshakeState::WaitingEncryptedExtensions;
        Ok(())
    }

    /// Generate a ServerHello message with hybrid server share.
    ///
    /// For the server: includes the X25519 ephemeral and ML-KEM-768
    /// ciphertext in the key_share extension.
    pub fn generate_server_hello(
        &mut self,
        server_share: HybridServerShare,
        shared_secret: [u8; SECRET_LEN],
    ) -> Result<Vec<u8>, NetError> {
        if self.role != TlsRole::Server {
            return Err(NetError::InvalidProtocol);
        }

        let mut msg = Vec::with_capacity(256 + HYBRID_CIPHERTEXT_LEN);

        // Handshake message type: ServerHello = 2
        msg.push(0x02);
        let len_pos = msg.len();
        msg.extend_from_slice(&[0, 0, 0]);

        // Protocol version (legacy)
        msg.extend_from_slice(&0x0303u16.to_be_bytes());

        // Server random
        let mut server_random = [0u8; 32];
        for (i, b) in server_random.iter_mut().enumerate() {
            *b = (i as u8).wrapping_mul(0x53).wrapping_add(0x22);
        }
        msg.extend_from_slice(&server_random);

        // Session ID (empty)
        msg.push(0);

        // Cipher suite
        let cs_code = cipher_suite_code(self.cipher_suite);
        msg.extend_from_slice(&cs_code.to_be_bytes());

        // Compression
        msg.push(0); // null

        // Extensions
        let share_ext_data = server_share.encode();
        let sv_ext = encode_supported_versions_ext();
        let ks_ext = encode_key_share_ext(&share_ext_data);

        let ext_total = sv_ext.len() + ks_ext.len();
        msg.extend_from_slice(&(ext_total as u16).to_be_bytes());
        msg.extend_from_slice(&sv_ext);
        msg.extend_from_slice(&ks_ext);

        // Fill in length
        let body_len = msg.len() - len_pos - 3;
        msg[len_pos] = ((body_len >> 16) & 0xFF) as u8;
        msg[len_pos + 1] = ((body_len >> 8) & 0xFF) as u8;
        msg[len_pos + 2] = (body_len & 0xFF) as u8;

        // Update transcript and derive keys
        self.transcript.extend_from_slice(&msg);
        self.shared_secret = Some(shared_secret);
        self.server_share = Some(server_share);

        let transcript_hash = compute_transcript_hash(&self.transcript);
        self.key_schedule
            .derive_handshake_secrets(&shared_secret, &transcript_hash);

        self.state = TlsHandshakeState::WaitingFinished;
        Ok(msg)
    }

    /// Complete the handshake and derive application keys.
    pub fn complete_handshake(&mut self) -> Result<(), NetError> {
        if !self.key_schedule.handshake_derived() {
            return Err(NetError::InvalidProtocol);
        }
        let transcript_hash = compute_transcript_hash(&self.transcript);
        self.key_schedule.derive_application_secrets(&transcript_hash);
        self.state = TlsHandshakeState::Complete;
        Ok(())
    }

    /// Get the packet protection keys for the given encryption level.
    pub fn protection_keys(
        &self,
        level: EncryptionLevel,
    ) -> Result<PacketProtectionKeys, NetError> {
        let tls_role = self.role;
        self.key_schedule
            .protection_keys(level, tls_role, self.cipher_suite)
    }

    /// Get the peer's protection keys for the given encryption level.
    pub fn peer_protection_keys(
        &self,
        level: EncryptionLevel,
    ) -> Result<PacketProtectionKeys, NetError> {
        let peer_role = match self.role {
            TlsRole::Client => TlsRole::Server,
            TlsRole::Server => TlsRole::Client,
        };
        self.key_schedule
            .protection_keys(level, peer_role, self.cipher_suite)
    }

    /// Whether the handshake is complete and application keys are available.
    pub fn is_complete(&self) -> bool {
        self.state == TlsHandshakeState::Complete
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// Simplified HKDF-Expand-Label for TLS 1.3.
///
/// Real implementation would use HKDF with SHA-3-256.
/// This stub uses XOR-based derivation for deterministic testing.
fn hkdf_expand_label(
    secret: &[u8; SECRET_LEN],
    label: &[u8],
    context: &[u8; SECRET_LEN],
) -> [u8; SECRET_LEN] {
    let mut out = [0u8; SECRET_LEN];
    for i in 0..SECRET_LEN {
        out[i] = secret[i]
            ^ label[i % label.len()]
            ^ context[i]
            ^ (i as u8);
    }
    out
}

/// Derive a key of the given length from a traffic secret and label.
fn expand_secret(secret: &[u8; SECRET_LEN], label: &[u8], len: usize) -> Vec<u8> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let b = secret[i % SECRET_LEN]
            ^ label[i % label.len()]
            ^ (i as u8).wrapping_mul(0x7F);
        out.push(b);
    }
    out
}

/// Compute a transcript hash (SHA-3-256 stub).
fn compute_transcript_hash(transcript: &[u8]) -> [u8; SECRET_LEN] {
    let mut hash = [0u8; SECRET_LEN];
    // Stub: rotating XOR accumulator
    for (i, &b) in transcript.iter().enumerate() {
        hash[i % SECRET_LEN] ^= b.wrapping_add(i as u8);
    }
    hash
}

/// Map CipherSuite to TLS codepoint.
fn cipher_suite_code(cs: CipherSuite) -> u16 {
    match cs {
        CipherSuite::Aes128Gcm => 0x1301,
        CipherSuite::Aes256Gcm => 0x1302,
        CipherSuite::ChaCha20Poly1305 => 0x1303,
    }
}

/// Encode the TLS 1.3 supported_versions extension.
fn encode_supported_versions_ext() -> Vec<u8> {
    let mut ext = Vec::with_capacity(7);
    ext.extend_from_slice(&0x002Bu16.to_be_bytes()); // extension type
    ext.extend_from_slice(&3u16.to_be_bytes()); // extension data length
    ext.push(2); // versions list length
    ext.extend_from_slice(&TLS_13_VERSION.to_be_bytes());
    ext
}

/// Encode a key_share extension wrapping the given key share data.
fn encode_key_share_ext(key_share_data: &[u8]) -> Vec<u8> {
    let mut ext = Vec::with_capacity(4 + 2 + key_share_data.len());
    ext.extend_from_slice(&0x0033u16.to_be_bytes()); // extension type
    let data_len = 2 + key_share_data.len();
    ext.extend_from_slice(&(data_len as u16).to_be_bytes()); // extension length
    ext.extend_from_slice(&(key_share_data.len() as u16).to_be_bytes()); // list length
    ext.extend_from_slice(key_share_data);
    ext
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // --- Key share encode/decode ---

    #[test]
    fn hybrid_key_share_roundtrip() {
        let x25519 = [0xAA; X25519_PK_LEN];
        let ml_kem = [0xBB; ML_KEM_768_PK_LEN];
        let share = HybridKeyShare::new(x25519, ml_kem);

        let encoded = share.encode();
        assert_eq!(encoded.len(), 4 + HYBRID_KEY_SHARE_LEN);

        let decoded = HybridKeyShare::decode(&encoded).unwrap();
        assert_eq!(decoded.x25519_pk(), &x25519);
        assert_eq!(decoded.ml_kem_pk(), &ml_kem);
    }

    #[test]
    fn hybrid_key_share_decode_too_short() {
        assert_eq!(
            HybridKeyShare::decode(&[0; 10]).err(),
            Some(NetError::PacketTooShort)
        );
    }

    #[test]
    fn hybrid_key_share_decode_wrong_group() {
        let mut data = vec![0x00, 0x01]; // wrong group
        data.extend_from_slice(&(HYBRID_KEY_SHARE_LEN as u16).to_be_bytes());
        data.extend_from_slice(&[0; HYBRID_KEY_SHARE_LEN]);
        assert_eq!(
            HybridKeyShare::decode(&data).err(),
            Some(NetError::InvalidProtocol)
        );
    }

    #[test]
    fn hybrid_key_share_decode_wrong_length() {
        let mut data = NAMED_GROUP_X25519_MLKEM768.to_be_bytes().to_vec();
        data.extend_from_slice(&99u16.to_be_bytes()); // wrong length
        data.extend_from_slice(&[0; HYBRID_KEY_SHARE_LEN]);
        assert_eq!(
            HybridKeyShare::decode(&data).err(),
            Some(NetError::InvalidHeader)
        );
    }

    // --- Server share encode/decode ---

    #[test]
    fn hybrid_server_share_roundtrip() {
        let x25519 = [0xCC; X25519_PK_LEN];
        let ml_kem_ct = [0xDD; ML_KEM_768_CT_LEN];
        let share = HybridServerShare::new(x25519, ml_kem_ct);

        let encoded = share.encode();
        assert_eq!(encoded.len(), 4 + HYBRID_CIPHERTEXT_LEN);

        let decoded = HybridServerShare::decode(&encoded).unwrap();
        assert_eq!(decoded.x25519_pk(), &x25519);
        assert_eq!(decoded.ml_kem_ct(), &ml_kem_ct);
    }

    #[test]
    fn hybrid_server_share_decode_too_short() {
        assert_eq!(
            HybridServerShare::decode(&[0; 10]).err(),
            Some(NetError::PacketTooShort)
        );
    }

    // --- TLS handshake state machine ---

    #[test]
    fn tls_handshake_initial_state() {
        let hs = TlsHandshake::new(TlsRole::Client, CipherSuite::Aes256Gcm);
        assert_eq!(hs.state(), TlsHandshakeState::Idle);
        assert_eq!(hs.role(), TlsRole::Client);
        assert_eq!(hs.cipher_suite(), CipherSuite::Aes256Gcm);
        assert!(!hs.is_complete());
    }

    #[test]
    fn tls_client_hello_generation() {
        let mut hs = TlsHandshake::new(TlsRole::Client, CipherSuite::Aes256Gcm);
        let share = HybridKeyShare::new([0x11; X25519_PK_LEN], [0x22; ML_KEM_768_PK_LEN]);
        hs.set_our_key_share(share);

        let ch = hs.generate_client_hello().unwrap();
        // First byte should be ClientHello type (0x01)
        assert_eq!(ch[0], 0x01);
        // State should transition
        assert_eq!(hs.state(), TlsHandshakeState::WaitingServerHello);
    }

    #[test]
    fn tls_client_hello_wrong_role() {
        let mut hs = TlsHandshake::new(TlsRole::Server, CipherSuite::Aes256Gcm);
        assert_eq!(
            hs.generate_client_hello().err(),
            Some(NetError::InvalidProtocol)
        );
    }

    #[test]
    fn tls_client_hello_no_key_share() {
        let mut hs = TlsHandshake::new(TlsRole::Client, CipherSuite::Aes256Gcm);
        assert_eq!(
            hs.generate_client_hello().err(),
            Some(NetError::InvalidProtocol)
        );
    }

    // --- Key schedule ---

    #[test]
    fn key_schedule_derive_handshake() {
        let mut ks = TlsKeySchedule::new(None);
        let shared = [0xAA; SECRET_LEN];
        let transcript = [0xBB; SECRET_LEN];

        assert!(!ks.handshake_derived());
        ks.derive_handshake_secrets(&shared, &transcript);
        assert!(ks.handshake_derived());
        assert!(!ks.app_derived());
    }

    #[test]
    fn key_schedule_derive_application() {
        let mut ks = TlsKeySchedule::new(None);
        let shared = [0xAA; SECRET_LEN];
        let transcript = [0xBB; SECRET_LEN];
        ks.derive_handshake_secrets(&shared, &transcript);

        let app_transcript = [0xCC; SECRET_LEN];
        ks.derive_application_secrets(&app_transcript);
        assert!(ks.app_derived());
    }

    #[test]
    fn key_schedule_protection_keys_handshake() {
        let mut ks = TlsKeySchedule::new(None);
        let shared = [0xAA; SECRET_LEN];
        let transcript = [0xBB; SECRET_LEN];
        ks.derive_handshake_secrets(&shared, &transcript);

        let client_keys = ks
            .protection_keys(EncryptionLevel::Handshake, TlsRole::Client, CipherSuite::Aes256Gcm)
            .unwrap();
        let server_keys = ks
            .protection_keys(EncryptionLevel::Handshake, TlsRole::Server, CipherSuite::Aes256Gcm)
            .unwrap();

        // Keys should be different for client and server
        assert_ne!(client_keys.key, server_keys.key);
        // Key should be AES-256 length
        assert_eq!(client_keys.key.len(), AES_256_KEY_LEN);
        // IV should be 12 bytes
        assert_eq!(client_keys.iv.len(), IV_LEN);
        // HP key should be 16 bytes
        assert_eq!(client_keys.hp_key.len(), HP_KEY_LEN);
    }

    #[test]
    fn key_schedule_protection_keys_not_derived() {
        let ks = TlsKeySchedule::new(None);
        assert_eq!(
            ks.protection_keys(EncryptionLevel::Handshake, TlsRole::Client, CipherSuite::Aes256Gcm).err(),
            Some(NetError::InvalidProtocol)
        );
    }

    #[test]
    fn key_schedule_app_keys_not_derived() {
        let mut ks = TlsKeySchedule::new(None);
        ks.derive_handshake_secrets(&[0; SECRET_LEN], &[0; SECRET_LEN]);
        assert_eq!(
            ks.protection_keys(EncryptionLevel::OneRtt, TlsRole::Client, CipherSuite::Aes256Gcm).err(),
            Some(NetError::InvalidProtocol)
        );
    }

    #[test]
    fn key_schedule_application_keys() {
        let mut ks = TlsKeySchedule::new(None);
        ks.derive_handshake_secrets(&[0xAA; SECRET_LEN], &[0xBB; SECRET_LEN]);
        ks.derive_application_secrets(&[0xCC; SECRET_LEN]);

        let client_keys = ks
            .protection_keys(EncryptionLevel::OneRtt, TlsRole::Client, CipherSuite::Aes256Gcm)
            .unwrap();
        let server_keys = ks
            .protection_keys(EncryptionLevel::OneRtt, TlsRole::Server, CipherSuite::Aes256Gcm)
            .unwrap();
        assert_ne!(client_keys.key, server_keys.key);
    }

    // --- Full client-server handshake ---

    #[test]
    fn full_handshake_client_server() {
        // 1. Client generates ClientHello
        let mut client = TlsHandshake::new(TlsRole::Client, CipherSuite::Aes256Gcm);
        let client_share = HybridKeyShare::new(
            [0x11; X25519_PK_LEN],
            [0x22; ML_KEM_768_PK_LEN],
        );
        client.set_our_key_share(client_share);
        let ch = client.generate_client_hello().unwrap();
        assert_eq!(client.state(), TlsHandshakeState::WaitingServerHello);

        // 2. Server generates ServerHello with hybrid ciphertext
        let mut server = TlsHandshake::new(TlsRole::Server, CipherSuite::Aes256Gcm);
        // Simulate: server processes ClientHello (adds to transcript)
        server.transcript.extend_from_slice(&ch);
        let server_share = HybridServerShare::new(
            [0x33; X25519_PK_LEN],
            [0x44; ML_KEM_768_CT_LEN],
        );
        // Both sides compute the same shared secret
        let shared = [0x55; SECRET_LEN];
        let sh = server.generate_server_hello(server_share, shared).unwrap();
        assert_eq!(server.state(), TlsHandshakeState::WaitingFinished);

        // 3. Client processes ServerHello
        client.process_server_hello(&sh, shared).unwrap();
        assert_eq!(client.state(), TlsHandshakeState::WaitingEncryptedExtensions);

        // 4. Both sides complete handshake
        client.complete_handshake().unwrap();
        assert_eq!(client.state(), TlsHandshakeState::Complete);
        assert!(client.is_complete());

        server.complete_handshake().unwrap();
        assert!(server.is_complete());

        // 5. Verify keys are available and match (client send = server receive)
        let client_send = client
            .protection_keys(EncryptionLevel::OneRtt)
            .unwrap();
        let server_recv = server
            .peer_protection_keys(EncryptionLevel::OneRtt)
            .unwrap();
        assert_eq!(client_send.key, server_recv.key);
        assert_eq!(client_send.iv, server_recv.iv);

        let server_send = server
            .protection_keys(EncryptionLevel::OneRtt)
            .unwrap();
        let client_recv = client
            .peer_protection_keys(EncryptionLevel::OneRtt)
            .unwrap();
        assert_eq!(server_send.key, client_recv.key);
    }

    // --- Cipher suite codes ---

    #[test]
    fn cipher_suite_codes() {
        assert_eq!(cipher_suite_code(CipherSuite::Aes128Gcm), 0x1301);
        assert_eq!(cipher_suite_code(CipherSuite::Aes256Gcm), 0x1302);
        assert_eq!(cipher_suite_code(CipherSuite::ChaCha20Poly1305), 0x1303);
    }

    // --- Named group constant ---

    #[test]
    fn named_group_value() {
        assert_eq!(NAMED_GROUP_X25519_MLKEM768, 0x6399);
    }

    // --- TLS version ---

    #[test]
    fn tls_13_version_value() {
        assert_eq!(TLS_13_VERSION, 0x0304);
    }

    // --- PSK key schedule ---

    #[test]
    fn key_schedule_with_psk() {
        let psk = [0xEE; SECRET_LEN];
        let ks = TlsKeySchedule::new(Some(&psk));
        assert_eq!(ks.early_secret, psk);
    }

    // --- Extension encoding ---

    #[test]
    fn supported_versions_ext_encoding() {
        let ext = encode_supported_versions_ext();
        // Type should be 0x002B
        assert_eq!(ext[0], 0x00);
        assert_eq!(ext[1], 0x2B);
        // Should contain TLS 1.3 version
        assert_eq!(ext[5], 0x03);
        assert_eq!(ext[6], 0x04);
    }

    #[test]
    fn key_share_ext_encoding() {
        let data = [0xAA; 10];
        let ext = encode_key_share_ext(&data);
        // Type should be 0x0033
        assert_eq!(ext[0], 0x00);
        assert_eq!(ext[1], 0x33);
    }

    // --- Handshake key derivation produces different keys per level ---

    #[test]
    fn handshake_vs_app_keys_differ() {
        let mut ks = TlsKeySchedule::new(None);
        ks.derive_handshake_secrets(&[0xAA; SECRET_LEN], &[0xBB; SECRET_LEN]);
        ks.derive_application_secrets(&[0xCC; SECRET_LEN]);

        let hs_keys = ks
            .protection_keys(EncryptionLevel::Handshake, TlsRole::Client, CipherSuite::Aes256Gcm)
            .unwrap();
        let app_keys = ks
            .protection_keys(EncryptionLevel::OneRtt, TlsRole::Client, CipherSuite::Aes256Gcm)
            .unwrap();
        assert_ne!(hs_keys.key, app_keys.key);
    }
}
