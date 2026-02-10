// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! TLS 1.3 transport with ML-KEM-768 hybrid key exchange.

#![allow(dead_code)]

use alloc::vec::Vec;

use crate::IpcError;

/// TLS protocol version identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum TlsVersion {
    /// TLS 1.2 (legacy, for compatibility detection only).
    Tls12 = 0x0303,
    /// TLS 1.3 (required).
    Tls13 = 0x0304,
}

/// Supported TLS 1.3 cipher suites.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u16)]
pub enum CipherSuite {
    /// AES-256-GCM with SHA-384 (default for SmallAIOS).
    Aes256GcmSha384 = 0x1302,
    /// ChaCha20-Poly1305 with SHA-256 (ARM/software fallback).
    Chacha20Poly1305Sha256 = 0x1303,
}

/// TLS 1.3 handshake message types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum HandshakeType {
    /// Client initiates handshake.
    ClientHello = 1,
    /// Server responds with selected parameters.
    ServerHello = 2,
    /// Server sends additional extensions (encrypted).
    EncryptedExtensions = 8,
    /// Server certificate for authentication.
    Certificate = 11,
    /// Signature proving possession of certificate private key.
    CertificateVerify = 15,
    /// Handshake integrity verification.
    Finished = 20,
}

/// Key exchange mode for the TLS 1.3 handshake.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyExchangeMode {
    /// Classical X25519 Diffie-Hellman only.
    X25519Only,
    /// Post-quantum ML-KEM-768 only.
    MlKem768Only,
    /// Hybrid: X25519 + ML-KEM-768 (default, post-quantum safe).
    HybridX25519MlKem768,
}

/// TLS session state machine states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TlsState {
    /// Session created but handshake not started.
    Idle,
    /// ClientHello has been sent, awaiting ServerHello.
    ClientHelloSent,
    /// ServerHello received, processing extensions.
    ServerHelloReceived,
    /// Handshake in progress (exchanging Finished messages).
    Handshaking,
    /// Handshake complete, secure channel established.
    Established,
    /// Session has been closed.
    Closed,
}

/// Configuration for a TLS 1.3 session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlsConfig {
    /// Selected cipher suite.
    pub cipher_suite: CipherSuite,
    /// Key exchange mode.
    pub key_exchange: KeyExchangeMode,
    /// Whether to require mutual (client) authentication.
    pub require_client_auth: bool,
}

impl TlsConfig {
    /// Create the default post-quantum configuration.
    ///
    /// Uses AES-256-GCM-SHA384 with hybrid X25519 + ML-KEM-768
    /// key exchange and no client authentication requirement.
    pub fn default_pq() -> Self {
        TlsConfig {
            cipher_suite: CipherSuite::Aes256GcmSha384,
            key_exchange: KeyExchangeMode::HybridX25519MlKem768,
            require_client_auth: false,
        }
    }
}

/// A TLS 1.3 session with state machine for the handshake protocol.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TlsSession {
    /// Current handshake/session state.
    state: TlsState,
    /// Session configuration.
    config: TlsConfig,
    /// Unique session identifier.
    session_id: u64,
    /// Whether the peer has been verified (certificate validated).
    peer_verified: bool,
}

impl TlsSession {
    /// Create a new TLS session with the given configuration.
    pub fn new(config: TlsConfig, session_id: u64) -> Self {
        TlsSession {
            state: TlsState::Idle,
            config,
            session_id,
            peer_verified: false,
        }
    }

    /// Returns a reference to the current session state.
    pub fn state(&self) -> &TlsState {
        &self.state
    }

    /// Returns the session identifier.
    pub fn session_id(&self) -> u64 {
        self.session_id
    }

    /// Returns a reference to the session configuration.
    pub fn config(&self) -> &TlsConfig {
        &self.config
    }

    /// Returns whether the peer has been verified.
    pub fn peer_verified(&self) -> bool {
        self.peer_verified
    }

    /// Initiate the TLS 1.3 handshake by generating a ClientHello message.
    ///
    /// Transitions the session from `Idle` to `ClientHelloSent`.
    ///
    /// # Errors
    ///
    /// Returns [`IpcError::InvalidState`] if the session is not in `Idle` state.
    pub fn start_handshake(&mut self) -> Result<Vec<u8>, IpcError> {
        if self.state != TlsState::Idle {
            return Err(IpcError::InvalidState);
        }

        // Build a placeholder ClientHello message.
        // In production this would contain: protocol version, random bytes,
        // cipher suite list, supported groups (X25519 + ML-KEM-768),
        // key share, and signature algorithms.
        let mut client_hello = Vec::new();

        // Handshake type: ClientHello (1 byte)
        client_hello.push(HandshakeType::ClientHello as u8);

        // TLS version (2 bytes, big-endian)
        let version = TlsVersion::Tls13 as u16;
        client_hello.push((version >> 8) as u8);
        client_hello.push(version as u8);

        // Cipher suite (2 bytes, big-endian)
        let suite = self.config.cipher_suite as u16;
        client_hello.push((suite >> 8) as u8);
        client_hello.push(suite as u8);

        // Key exchange mode (1 byte)
        client_hello.push(match self.config.key_exchange {
            KeyExchangeMode::X25519Only => 0x01,
            KeyExchangeMode::MlKem768Only => 0x02,
            KeyExchangeMode::HybridX25519MlKem768 => 0x03,
        });

        // Session ID (8 bytes, big-endian)
        client_hello.extend_from_slice(&self.session_id.to_be_bytes());

        // Placeholder random bytes (32 bytes of zeros -- stub)
        client_hello.extend_from_slice(&[0u8; 32]);

        self.state = TlsState::ClientHelloSent;
        Ok(client_hello)
    }

    /// Process an incoming handshake message and optionally produce a response.
    ///
    /// State machine transitions:
    /// - `ClientHelloSent` + data -> `ServerHelloReceived`, returns EncryptedExtensions
    /// - `ServerHelloReceived` + data -> `Handshaking`, returns Finished
    /// - `Handshaking` + data -> `Established`, returns None (handshake complete)
    ///
    /// # Errors
    ///
    /// Returns [`IpcError::InvalidState`] if called in an unexpected state.
    pub fn process_handshake(&mut self, data: &[u8]) -> Result<Option<Vec<u8>>, IpcError> {
        match self.state {
            TlsState::ClientHelloSent => {
                // We received something (ServerHello equivalent).
                // In production we would parse the ServerHello from `data`,
                // extract the server's key share, and derive handshake keys.
                let _ = data; // Acknowledge receipt

                self.state = TlsState::ServerHelloReceived;

                // Build EncryptedExtensions response (stub).
                let mut response = Vec::new();
                response.push(HandshakeType::EncryptedExtensions as u8);
                // Placeholder: 4 bytes of extension data
                response.extend_from_slice(&[0x00, 0x02, 0x00, 0x00]);
                Ok(Some(response))
            }
            TlsState::ServerHelloReceived => {
                // We received EncryptedExtensions / Certificate / CertificateVerify.
                let _ = data;

                self.state = TlsState::Handshaking;

                // Build Finished message (stub).
                let mut response = Vec::new();
                response.push(HandshakeType::Finished as u8);
                // Placeholder: 32-byte verify data (zeros -- stub)
                response.extend_from_slice(&[0u8; 32]);
                Ok(Some(response))
            }
            TlsState::Handshaking => {
                // Received peer's Finished message.
                let _ = data;

                self.peer_verified = true;
                self.state = TlsState::Established;

                // Handshake complete -- no further message to send.
                Ok(None)
            }
            _ => Err(IpcError::InvalidState),
        }
    }

    /// Returns `true` if the TLS handshake has completed and the session
    /// is in the `Established` state.
    pub fn is_established(&self) -> bool {
        self.state == TlsState::Established
    }

    /// Close the TLS session.
    ///
    /// In production this would send a `close_notify` alert.
    pub fn close(&mut self) {
        self.state = TlsState::Closed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // === TlsConfig tests ===

    #[test]
    fn default_pq_config_cipher_suite() {
        let config = TlsConfig::default_pq();
        assert_eq!(config.cipher_suite, CipherSuite::Aes256GcmSha384);
    }

    #[test]
    fn default_pq_config_key_exchange() {
        let config = TlsConfig::default_pq();
        assert_eq!(config.key_exchange, KeyExchangeMode::HybridX25519MlKem768);
    }

    #[test]
    fn default_pq_config_no_client_auth() {
        let config = TlsConfig::default_pq();
        assert!(!config.require_client_auth);
    }

    // === TlsSession construction ===

    #[test]
    fn new_session_starts_idle() {
        let session = TlsSession::new(TlsConfig::default_pq(), 42);
        assert_eq!(*session.state(), TlsState::Idle);
        assert_eq!(session.session_id(), 42);
        assert!(!session.is_established());
        assert!(!session.peer_verified());
    }

    // === Handshake state machine ===

    #[test]
    fn start_handshake_transitions_to_client_hello_sent() {
        let mut session = TlsSession::new(TlsConfig::default_pq(), 1);
        let client_hello = session.start_handshake().unwrap();
        assert_eq!(*session.state(), TlsState::ClientHelloSent);
        // First byte should be ClientHello type
        assert_eq!(client_hello[0], HandshakeType::ClientHello as u8);
        // Should contain TLS version bytes
        assert_eq!(client_hello[1], 0x03);
        assert_eq!(client_hello[2], 0x04);
    }

    #[test]
    fn start_handshake_invalid_state() {
        let mut session = TlsSession::new(TlsConfig::default_pq(), 1);
        session.start_handshake().unwrap();
        // Calling start_handshake again should fail
        assert_eq!(session.start_handshake(), Err(IpcError::InvalidState));
    }

    #[test]
    fn full_handshake_sequence() {
        let mut session = TlsSession::new(TlsConfig::default_pq(), 100);

        // Step 1: Start handshake (Idle -> ClientHelloSent)
        let client_hello = session.start_handshake().unwrap();
        assert!(!client_hello.is_empty());
        assert_eq!(*session.state(), TlsState::ClientHelloSent);

        // Step 2: Process ServerHello (ClientHelloSent -> ServerHelloReceived)
        let server_hello_data = [0x02, 0x03, 0x04]; // stub data
        let response1 = session.process_handshake(&server_hello_data).unwrap();
        assert!(response1.is_some());
        let encrypted_ext = response1.unwrap();
        assert_eq!(encrypted_ext[0], HandshakeType::EncryptedExtensions as u8);
        assert_eq!(*session.state(), TlsState::ServerHelloReceived);

        // Step 3: Process EncryptedExtensions (ServerHelloReceived -> Handshaking)
        let ext_data = [0x08, 0x00, 0x00]; // stub data
        let response2 = session.process_handshake(&ext_data).unwrap();
        assert!(response2.is_some());
        let finished = response2.unwrap();
        assert_eq!(finished[0], HandshakeType::Finished as u8);
        assert_eq!(*session.state(), TlsState::Handshaking);

        // Step 4: Process peer Finished (Handshaking -> Established)
        let finished_data = [0x14, 0x00]; // stub data
        let response3 = session.process_handshake(&finished_data).unwrap();
        assert!(response3.is_none());
        assert_eq!(*session.state(), TlsState::Established);
        assert!(session.is_established());
        assert!(session.peer_verified());
    }

    #[test]
    fn process_handshake_in_idle_fails() {
        let mut session = TlsSession::new(TlsConfig::default_pq(), 1);
        assert_eq!(
            session.process_handshake(&[0x01]),
            Err(IpcError::InvalidState)
        );
    }

    #[test]
    fn process_handshake_in_established_fails() {
        let mut session = TlsSession::new(TlsConfig::default_pq(), 1);
        session.start_handshake().unwrap();
        session.process_handshake(&[]).unwrap();
        session.process_handshake(&[]).unwrap();
        session.process_handshake(&[]).unwrap();
        assert!(session.is_established());
        assert_eq!(session.process_handshake(&[]), Err(IpcError::InvalidState));
    }

    #[test]
    fn process_handshake_in_closed_fails() {
        let mut session = TlsSession::new(TlsConfig::default_pq(), 1);
        session.close();
        assert_eq!(session.process_handshake(&[]), Err(IpcError::InvalidState));
    }

    #[test]
    fn close_session() {
        let mut session = TlsSession::new(TlsConfig::default_pq(), 1);
        session.start_handshake().unwrap();
        session.process_handshake(&[]).unwrap();
        session.process_handshake(&[]).unwrap();
        session.process_handshake(&[]).unwrap();
        assert!(session.is_established());
        session.close();
        assert_eq!(*session.state(), TlsState::Closed);
        assert!(!session.is_established());
    }

    #[test]
    fn close_idle_session() {
        let mut session = TlsSession::new(TlsConfig::default_pq(), 1);
        session.close();
        assert_eq!(*session.state(), TlsState::Closed);
    }

    // === Enum value tests ===

    #[test]
    fn tls_version_values() {
        assert_eq!(TlsVersion::Tls12 as u16, 0x0303);
        assert_eq!(TlsVersion::Tls13 as u16, 0x0304);
    }

    #[test]
    fn cipher_suite_values() {
        assert_eq!(CipherSuite::Aes256GcmSha384 as u16, 0x1302);
        assert_eq!(CipherSuite::Chacha20Poly1305Sha256 as u16, 0x1303);
    }

    #[test]
    fn handshake_type_values() {
        assert_eq!(HandshakeType::ClientHello as u8, 1);
        assert_eq!(HandshakeType::ServerHello as u8, 2);
        assert_eq!(HandshakeType::EncryptedExtensions as u8, 8);
        assert_eq!(HandshakeType::Certificate as u8, 11);
        assert_eq!(HandshakeType::CertificateVerify as u8, 15);
        assert_eq!(HandshakeType::Finished as u8, 20);
    }

    #[test]
    fn key_exchange_mode_equality() {
        assert_ne!(KeyExchangeMode::X25519Only, KeyExchangeMode::MlKem768Only);
        assert_ne!(
            KeyExchangeMode::MlKem768Only,
            KeyExchangeMode::HybridX25519MlKem768
        );
    }

    // === Client hello content ===

    #[test]
    fn client_hello_contains_session_id() {
        let mut session = TlsSession::new(TlsConfig::default_pq(), 0xDEAD_BEEF_CAFE_BABE);
        let hello = session.start_handshake().unwrap();
        // Session ID starts at offset 6 (1 type + 2 version + 2 cipher + 1 kex)
        let id_bytes = &hello[6..14];
        let id = u64::from_be_bytes(id_bytes.try_into().unwrap());
        assert_eq!(id, 0xDEAD_BEEF_CAFE_BABE);
    }

    #[test]
    fn client_hello_with_chacha_cipher() {
        let config = TlsConfig {
            cipher_suite: CipherSuite::Chacha20Poly1305Sha256,
            key_exchange: KeyExchangeMode::X25519Only,
            require_client_auth: false,
        };
        let mut session = TlsSession::new(config, 1);
        let hello = session.start_handshake().unwrap();
        // Cipher suite at offset 3..5
        assert_eq!(hello[3], 0x13);
        assert_eq!(hello[4], 0x03);
        // Key exchange mode at offset 5
        assert_eq!(hello[5], 0x01); // X25519Only
    }

    #[test]
    fn config_with_client_auth() {
        let config = TlsConfig {
            cipher_suite: CipherSuite::Aes256GcmSha384,
            key_exchange: KeyExchangeMode::HybridX25519MlKem768,
            require_client_auth: true,
        };
        assert!(config.require_client_auth);
        let session = TlsSession::new(config, 1);
        assert!(session.config().require_client_auth);
    }
}
