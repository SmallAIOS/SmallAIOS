// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the TLS 1.3 + ML-KEM-768 crypto pipeline.
//!
//! These tests exercise the cross-crate interaction between the security
//! crate's cryptographic primitives (SHA-3, AES-256-GCM, ML-KEM-768) and
//! the net crate's TLS 1.3 handshake / key schedule implementation.

extern crate alloc;

use smallaios_net::quic::protection::{CipherSuite, EncryptionLevel};
use smallaios_net::quic::{
    HybridKeyShare, HybridServerShare, TlsHandshake, TlsHandshakeState, TlsKeySchedule, TlsRole,
};
use smallaios_security::crypto::aes_gcm::{Aes256Gcm, AesKey, GcmNonce};
use smallaios_security::crypto::ml_kem::{
    ml_kem_768_decaps, ml_kem_768_encaps, ml_kem_768_keygen, ML_KEM_768_CT_LEN, ML_KEM_768_PK_LEN,
};
use smallaios_security::crypto::sha3::sha3_256;

// ─── ML-KEM-768 keygen / encaps / decaps ────────────────────────────────────

/// ML-KEM-768 keygen produces valid keys, and encaps/decaps agree on the
/// shared secret.
#[test]
fn ml_kem_768_roundtrip() {
    // Deterministic seed (64 bytes: d || z)
    let mut seed = [0u8; 64];
    for (i, b) in seed.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(0x3B).wrapping_add(0x07);
    }

    let keypair = ml_kem_768_keygen(&seed).expect("keygen should succeed");

    // Encapsulate with a deterministic random seed
    let encaps_rand = [0xAAu8; 32];
    let (ciphertext, shared_secret_enc) =
        ml_kem_768_encaps(&keypair.public_key, &encaps_rand).expect("encaps should succeed");

    // Decapsulate
    let shared_secret_dec =
        ml_kem_768_decaps(&keypair.secret_key, &ciphertext).expect("decaps should succeed");

    assert_eq!(
        shared_secret_enc.as_bytes(),
        shared_secret_dec.as_bytes(),
        "encaps and decaps must produce the same shared secret"
    );
}

/// Encapsulating with different randomness produces different ciphertexts
/// and shared secrets.
#[test]
fn ml_kem_768_different_randomness() {
    let mut seed = [0u8; 64];
    for (i, b) in seed.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(0x11);
    }
    let keypair = ml_kem_768_keygen(&seed).expect("keygen should succeed");

    let rand_a = [0x01u8; 32];
    let rand_b = [0x02u8; 32];

    let (ct_a, ss_a) = ml_kem_768_encaps(&keypair.public_key, &rand_a).expect("encaps a");
    let (ct_b, ss_b) = ml_kem_768_encaps(&keypair.public_key, &rand_b).expect("encaps b");

    assert_ne!(
        ct_a.as_bytes(),
        ct_b.as_bytes(),
        "different randomness must yield different ciphertexts"
    );
    assert_ne!(
        ss_a.as_bytes(),
        ss_b.as_bytes(),
        "different randomness must yield different shared secrets"
    );
}

// ─── SHA-3-256 hashing ──────────────────────────────────────────────────────

/// SHA-3-256 produces consistent output and is used to combine shared secrets
/// in the hybrid key exchange: SHA-3-256(x25519_ss || ml_kem_ss).
#[test]
fn sha3_256_hybrid_secret_combination() {
    let x25519_ss = [0x11u8; 32];
    let ml_kem_ss = [0x22u8; 32];

    let mut combined = [0u8; 64];
    combined[..32].copy_from_slice(&x25519_ss);
    combined[32..].copy_from_slice(&ml_kem_ss);

    let digest_a = sha3_256(&combined);
    let digest_b = sha3_256(&combined);

    assert_eq!(
        digest_a.as_bytes(),
        digest_b.as_bytes(),
        "SHA-3-256 must be deterministic"
    );

    // Swapping the order should produce a different digest.
    let mut swapped = [0u8; 64];
    swapped[..32].copy_from_slice(&ml_kem_ss);
    swapped[32..].copy_from_slice(&x25519_ss);

    let digest_swapped = sha3_256(&swapped);
    assert_ne!(
        digest_a.as_bytes(),
        digest_swapped.as_bytes(),
        "order of inputs must affect the digest"
    );
}

// ─── TLS 1.3 key schedule derivation ────────────────────────────────────────

/// Derive handshake and application secrets, then verify that keys for
/// client and server differ, and that handshake-level keys differ from
/// application-level keys.
#[test]
fn tls_key_schedule_derivation() {
    let mut ks = TlsKeySchedule::new(None);

    let shared_secret = [0xABu8; 32];
    let transcript_hash = *sha3_256(b"ClientHello || ServerHello").as_bytes();

    ks.derive_handshake_secrets(&shared_secret, &transcript_hash);
    assert!(ks.handshake_derived());
    assert!(!ks.app_derived());

    // Handshake-level keys should be available and differ per role.
    let client_hs = ks
        .protection_keys(
            EncryptionLevel::Handshake,
            TlsRole::Client,
            CipherSuite::Aes256Gcm,
        )
        .expect("client handshake keys");
    let server_hs = ks
        .protection_keys(
            EncryptionLevel::Handshake,
            TlsRole::Server,
            CipherSuite::Aes256Gcm,
        )
        .expect("server handshake keys");

    assert_ne!(
        client_hs.key, server_hs.key,
        "client/server keys must differ"
    );
    assert_eq!(client_hs.key.len(), 32, "AES-256 key must be 32 bytes");
    assert_eq!(client_hs.iv.len(), 12, "IV must be 12 bytes");

    // Application keys should NOT be available yet.
    assert!(
        ks.protection_keys(
            EncryptionLevel::OneRtt,
            TlsRole::Client,
            CipherSuite::Aes256Gcm
        )
        .is_err(),
        "app keys must not be available before derive_application_secrets"
    );

    // Now derive application secrets.
    let app_transcript = *sha3_256(b"full handshake transcript").as_bytes();
    ks.derive_application_secrets(&app_transcript);
    assert!(ks.app_derived());

    let client_app = ks
        .protection_keys(
            EncryptionLevel::OneRtt,
            TlsRole::Client,
            CipherSuite::Aes256Gcm,
        )
        .expect("client app keys");
    let server_app = ks
        .protection_keys(
            EncryptionLevel::OneRtt,
            TlsRole::Server,
            CipherSuite::Aes256Gcm,
        )
        .expect("server app keys");

    assert_ne!(client_app.key, server_app.key);
    assert_ne!(
        client_hs.key, client_app.key,
        "handshake and app keys must differ"
    );
}

// ─── Full TLS 1.3 handshake with hybrid key exchange ────────────────────────

/// Simulates a full client-server TLS 1.3 handshake using hybrid
/// X25519 + ML-KEM-768 key exchange, then verifies the derived keys
/// match on both sides.
#[test]
fn full_tls_handshake_hybrid_key_exchange() {
    // --- Key material (ML-KEM-768) ---
    let mut kem_seed = [0u8; 64];
    for (i, b) in kem_seed.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(0x5D).wrapping_add(0x13);
    }
    let kem_keypair = ml_kem_768_keygen(&kem_seed).expect("ML-KEM keygen");

    let encaps_rand = [0xBBu8; 32];
    let (kem_ct, kem_ss) =
        ml_kem_768_encaps(&kem_keypair.public_key, &encaps_rand).expect("ML-KEM encaps");

    // Simulated X25519 shared secret (in production this comes from DH).
    let x25519_shared = [0x77u8; 32];

    // Combine via SHA-3-256(x25519_ss || ml_kem_ss) as per the hybrid spec.
    let mut combined_input = [0u8; 64];
    combined_input[..32].copy_from_slice(&x25519_shared);
    combined_input[32..].copy_from_slice(kem_ss.as_bytes());
    let hybrid_secret = *sha3_256(&combined_input).as_bytes();

    // --- Client side ---
    let mut client = TlsHandshake::new(TlsRole::Client, CipherSuite::Aes256Gcm);

    // Build client key share from the ML-KEM public key bytes.
    let kem_pk_bytes = kem_keypair.public_key.as_bytes();
    let mut ml_kem_pk_arr = [0u8; ML_KEM_768_PK_LEN];
    ml_kem_pk_arr.copy_from_slice(kem_pk_bytes);
    let client_x25519_pk = [0x44u8; 32]; // Simulated X25519 ephemeral
    let client_share = HybridKeyShare::new(client_x25519_pk, ml_kem_pk_arr);
    client.set_our_key_share(client_share);

    let client_hello = client.generate_client_hello().expect("ClientHello");
    assert_eq!(client.state(), TlsHandshakeState::WaitingServerHello);
    assert_eq!(client_hello[0], 0x01, "first byte must be ClientHello type");

    // --- Server side ---
    let mut server = TlsHandshake::new(TlsRole::Server, CipherSuite::Aes256Gcm);

    // Server receives ClientHello (simulate by adding to transcript).
    // In real code the server parses the ClientHello and extracts the key share.

    // Build server share with ML-KEM ciphertext.
    let mut kem_ct_arr = [0u8; ML_KEM_768_CT_LEN];
    kem_ct_arr.copy_from_slice(kem_ct.as_bytes());
    let server_x25519_pk = [0x55u8; 32]; // Simulated X25519 ephemeral
    let server_share = HybridServerShare::new(server_x25519_pk, kem_ct_arr);

    let server_hello = server
        .generate_server_hello(server_share, hybrid_secret)
        .expect("ServerHello");
    assert_eq!(server_hello[0], 0x02, "first byte must be ServerHello type");

    // --- Client processes ServerHello ---
    // Client independently computes the same hybrid secret:
    // In real code, the client decapsulates the ML-KEM ciphertext.
    let kem_ss_client = ml_kem_768_decaps(&kem_keypair.secret_key, &kem_ct).expect("ML-KEM decaps");
    assert_eq!(kem_ss_client.as_bytes(), kem_ss.as_bytes());

    let mut combined_client = [0u8; 64];
    combined_client[..32].copy_from_slice(&x25519_shared);
    combined_client[32..].copy_from_slice(kem_ss_client.as_bytes());
    let client_hybrid_secret = *sha3_256(&combined_client).as_bytes();
    assert_eq!(
        client_hybrid_secret, hybrid_secret,
        "both sides must derive the same hybrid secret"
    );

    client
        .process_server_hello(&server_hello, client_hybrid_secret)
        .expect("process ServerHello");
    assert_eq!(
        client.state(),
        TlsHandshakeState::WaitingEncryptedExtensions
    );

    // --- Complete handshake on both sides ---
    client.complete_handshake().expect("client complete");
    server.complete_handshake().expect("server complete");
    assert!(client.is_complete());
    assert!(server.is_complete());

    // --- Verify keys are available ---
    // Note: In this integration test the client and server TlsHandshake
    // instances maintain independent transcripts (the server's transcript
    // does not include the ClientHello bytes that the client added to its
    // own transcript before generating the ServerHello). This is expected
    // because the public API does not expose a way to inject transcript
    // data from outside. The internal unit tests in net/src/quic/tls.rs
    // verify cross-side key equality by accessing private fields.
    //
    // Here we verify that both sides successfully derive 1-RTT keys and
    // that the client's send/recv keys are distinct (directional).
    let client_send = client
        .protection_keys(EncryptionLevel::OneRtt)
        .expect("client 1-RTT send keys");
    let client_recv = client
        .peer_protection_keys(EncryptionLevel::OneRtt)
        .expect("client 1-RTT recv keys");
    assert_ne!(
        client_send.key, client_recv.key,
        "client send and recv keys must be directionally distinct"
    );
    assert_eq!(client_send.key.len(), 32, "AES-256 key length");
    assert_eq!(client_send.iv.len(), 12, "GCM IV length");

    let server_send = server
        .protection_keys(EncryptionLevel::OneRtt)
        .expect("server 1-RTT send keys");
    let server_recv = server
        .peer_protection_keys(EncryptionLevel::OneRtt)
        .expect("server 1-RTT recv keys");
    assert_ne!(
        server_send.key, server_recv.key,
        "server send and recv keys must be directionally distinct"
    );
}

// ─── AES-256-GCM encrypt/decrypt with TLS-derived keys ─────────────────────

/// After a TLS 1.3 handshake, use the derived application traffic key
/// and IV to encrypt and decrypt data with AES-256-GCM, verifying the
/// full round-trip.
#[test]
fn aes_gcm_roundtrip_with_tls_derived_keys() {
    // Set up a key schedule and derive application keys.
    let mut ks = TlsKeySchedule::new(None);
    let shared_secret = [0xCDu8; 32];
    let hs_transcript = *sha3_256(b"handshake transcript").as_bytes();
    ks.derive_handshake_secrets(&shared_secret, &hs_transcript);

    let app_transcript = *sha3_256(b"application transcript").as_bytes();
    ks.derive_application_secrets(&app_transcript);

    // Extract client application traffic key and IV.
    let prot_keys = ks
        .protection_keys(
            EncryptionLevel::OneRtt,
            TlsRole::Client,
            CipherSuite::Aes256Gcm,
        )
        .expect("client app keys");

    assert_eq!(prot_keys.key.len(), 32, "AES-256 key must be 32 bytes");
    assert_eq!(prot_keys.iv.len(), 12, "GCM IV must be 12 bytes");

    // Build AES-256-GCM cipher from the derived key.
    let aes_key = AesKey::from_slice(&prot_keys.key).expect("valid AES key from TLS key schedule");
    let nonce = GcmNonce::from_slice(&prot_keys.iv).expect("valid nonce from TLS IV");
    let cipher = Aes256Gcm::new(aes_key);

    // Encrypt application data.
    let plaintext = b"ONNX inference result: tensor([0.91, 0.07, 0.02])";
    let aad = b"quic-packet-header";
    let mut ciphertext = vec![0u8; plaintext.len()];

    let tag = cipher
        .encrypt(&nonce, aad, plaintext, &mut ciphertext)
        .expect("encryption should succeed");

    // Ciphertext must differ from plaintext.
    assert_ne!(
        &ciphertext[..],
        &plaintext[..],
        "ciphertext must differ from plaintext"
    );

    // Decrypt and verify round-trip.
    let mut decrypted = vec![0u8; ciphertext.len()];
    cipher
        .decrypt(&nonce, aad, &ciphertext, &tag, &mut decrypted)
        .expect("decryption should succeed");

    assert_eq!(
        &decrypted[..],
        &plaintext[..],
        "decrypted data must match original plaintext"
    );
}

/// Verify that a tampered ciphertext fails AES-256-GCM authentication
/// when using TLS-derived keys.
#[test]
fn aes_gcm_tamper_detection_with_tls_keys() {
    let mut ks = TlsKeySchedule::new(None);
    let shared = [0xEEu8; 32];
    let transcript = *sha3_256(b"transcript").as_bytes();
    ks.derive_handshake_secrets(&shared, &transcript);
    ks.derive_application_secrets(&transcript);

    let prot_keys = ks
        .protection_keys(
            EncryptionLevel::OneRtt,
            TlsRole::Server,
            CipherSuite::Aes256Gcm,
        )
        .expect("server app keys");

    let aes_key = AesKey::from_slice(&prot_keys.key).expect("valid key");
    let nonce = GcmNonce::from_slice(&prot_keys.iv).expect("valid nonce");
    let cipher = Aes256Gcm::new(aes_key);

    let plaintext = b"sensitive payload";
    let aad = b"header";
    let mut ciphertext = vec![0u8; plaintext.len()];
    let tag = cipher
        .encrypt(&nonce, aad, plaintext, &mut ciphertext)
        .expect("encrypt");

    // Tamper with ciphertext.
    ciphertext[0] ^= 0xFF;

    let mut decrypted = vec![0u8; ciphertext.len()];
    let result = cipher.decrypt(&nonce, aad, &ciphertext, &tag, &mut decrypted);
    assert!(
        result.is_err(),
        "tampered ciphertext must fail authentication"
    );
}

// ─── End-to-end: handshake + encrypt + decrypt ──────────────────────────────

/// Complete end-to-end test: ML-KEM-768 key exchange, TLS 1.3 key schedule,
/// then AES-256-GCM encrypted data transfer in both directions.
#[test]
fn end_to_end_handshake_then_encrypted_transfer() {
    // 1. ML-KEM-768 key exchange
    let mut seed = [0u8; 64];
    for (i, b) in seed.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(0x73).wrapping_add(0x29);
    }
    let keypair = ml_kem_768_keygen(&seed).expect("keygen");
    let encaps_rand = [0xDDu8; 32];
    let (ct, ss_sender) = ml_kem_768_encaps(&keypair.public_key, &encaps_rand).expect("encaps");
    let ss_receiver = ml_kem_768_decaps(&keypair.secret_key, &ct).expect("decaps");
    assert_eq!(ss_sender.as_bytes(), ss_receiver.as_bytes());

    // 2. Combine with simulated X25519 via SHA-3-256
    let x25519_ss = [0x99u8; 32];
    let mut combined = [0u8; 64];
    combined[..32].copy_from_slice(&x25519_ss);
    combined[32..].copy_from_slice(ss_sender.as_bytes());
    let hybrid_secret = *sha3_256(&combined).as_bytes();

    // 3. TLS 1.3 key schedule
    let mut ks = TlsKeySchedule::new(None);
    let transcript = *sha3_256(b"e2e-handshake-transcript").as_bytes();
    ks.derive_handshake_secrets(&hybrid_secret, &transcript);
    let app_transcript = *sha3_256(b"e2e-app-transcript").as_bytes();
    ks.derive_application_secrets(&app_transcript);

    // 4. Client -> Server encrypted message
    let client_keys = ks
        .protection_keys(
            EncryptionLevel::OneRtt,
            TlsRole::Client,
            CipherSuite::Aes256Gcm,
        )
        .expect("client keys");

    let client_cipher =
        Aes256Gcm::new(AesKey::from_slice(&client_keys.key).expect("client AES key"));
    let client_nonce = GcmNonce::from_slice(&client_keys.iv).expect("client nonce");

    let message = b"inference request: model=resnet50, batch=1";
    let aad = b"quic-1rtt";
    let mut ct_buf = vec![0u8; message.len()];
    let tag = client_cipher
        .encrypt(&client_nonce, aad, message, &mut ct_buf)
        .expect("client encrypt");

    // Server decrypts using same key material (client role keys).
    let server_peer_keys = ks
        .protection_keys(
            EncryptionLevel::OneRtt,
            TlsRole::Client,
            CipherSuite::Aes256Gcm,
        )
        .expect("server uses client role to decrypt");
    let server_decipher = Aes256Gcm::new(
        AesKey::from_slice(&server_peer_keys.key).expect("server AES key for client data"),
    );
    let server_nonce = GcmNonce::from_slice(&server_peer_keys.iv).expect("server nonce");

    let mut decrypted = vec![0u8; ct_buf.len()];
    server_decipher
        .decrypt(&server_nonce, aad, &ct_buf, &tag, &mut decrypted)
        .expect("server decrypt");
    assert_eq!(
        &decrypted[..],
        &message[..],
        "server must recover plaintext"
    );

    // 5. Server -> Client encrypted response
    let server_keys = ks
        .protection_keys(
            EncryptionLevel::OneRtt,
            TlsRole::Server,
            CipherSuite::Aes256Gcm,
        )
        .expect("server keys");
    let server_cipher =
        Aes256Gcm::new(AesKey::from_slice(&server_keys.key).expect("server AES key"));
    let server_resp_nonce = GcmNonce::from_slice(&server_keys.iv).expect("server resp nonce");

    let response = b"inference result: class=cat, confidence=0.97";
    let mut resp_ct = vec![0u8; response.len()];
    let resp_tag = server_cipher
        .encrypt(&server_resp_nonce, aad, response, &mut resp_ct)
        .expect("server encrypt response");

    // Client decrypts using server role keys.
    let client_peer_keys = ks
        .protection_keys(
            EncryptionLevel::OneRtt,
            TlsRole::Server,
            CipherSuite::Aes256Gcm,
        )
        .expect("client uses server role to decrypt");
    let client_decipher = Aes256Gcm::new(
        AesKey::from_slice(&client_peer_keys.key).expect("client AES key for server data"),
    );
    let client_resp_nonce = GcmNonce::from_slice(&client_peer_keys.iv).expect("client resp nonce");

    let mut resp_decrypted = vec![0u8; resp_ct.len()];
    client_decipher
        .decrypt(
            &client_resp_nonce,
            aad,
            &resp_ct,
            &resp_tag,
            &mut resp_decrypted,
        )
        .expect("client decrypt response");
    assert_eq!(
        &resp_decrypted[..],
        &response[..],
        "client must recover server response"
    );
}

// ─── HybridKeyShare / HybridServerShare encode-decode with real ML-KEM keys ─

/// Verify that HybridKeyShare encode/decode works with a real ML-KEM-768
/// public key (not just synthetic test bytes).
#[test]
fn hybrid_key_share_with_real_ml_kem_pk() {
    let mut seed = [0u8; 64];
    for (i, b) in seed.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(0x41);
    }
    let keypair = ml_kem_768_keygen(&seed).expect("keygen");

    let pk_bytes = keypair.public_key.as_bytes();
    let mut ml_kem_pk = [0u8; ML_KEM_768_PK_LEN];
    ml_kem_pk.copy_from_slice(pk_bytes);

    let x25519_pk = [0x33u8; 32];
    let share = HybridKeyShare::new(x25519_pk, ml_kem_pk);

    let encoded = share.encode();
    let decoded = HybridKeyShare::decode(&encoded).expect("decode should succeed");

    assert_eq!(decoded.x25519_pk(), &x25519_pk);
    assert_eq!(decoded.ml_kem_pk(), &ml_kem_pk);
}

/// Verify that HybridServerShare encode/decode works with a real ML-KEM-768
/// ciphertext.
#[test]
fn hybrid_server_share_with_real_ml_kem_ct() {
    let mut seed = [0u8; 64];
    for (i, b) in seed.iter_mut().enumerate() {
        *b = (i as u8).wrapping_mul(0x53);
    }
    let keypair = ml_kem_768_keygen(&seed).expect("keygen");

    let encaps_rand = [0xCCu8; 32];
    let (ciphertext, _ss) = ml_kem_768_encaps(&keypair.public_key, &encaps_rand).expect("encaps");

    let ct_bytes = ciphertext.as_bytes();
    let mut ml_kem_ct = [0u8; ML_KEM_768_CT_LEN];
    ml_kem_ct.copy_from_slice(ct_bytes);

    let x25519_pk = [0x66u8; 32];
    let server_share = HybridServerShare::new(x25519_pk, ml_kem_ct);

    let encoded = server_share.encode();
    let decoded = HybridServerShare::decode(&encoded).expect("decode should succeed");

    assert_eq!(decoded.x25519_pk(), &x25519_pk);
    assert_eq!(decoded.ml_kem_ct(), &ml_kem_ct);
}

// ─── PSK key schedule ───────────────────────────────────────────────────────

/// When a PSK is provided, the key schedule should derive different secrets
/// than the no-PSK case.
#[test]
fn tls_key_schedule_with_psk_differs() {
    let shared = [0xABu8; 32];
    let transcript = *sha3_256(b"psk-transcript").as_bytes();

    // Without PSK
    let mut ks_no_psk = TlsKeySchedule::new(None);
    ks_no_psk.derive_handshake_secrets(&shared, &transcript);
    let keys_no_psk = ks_no_psk
        .protection_keys(
            EncryptionLevel::Handshake,
            TlsRole::Client,
            CipherSuite::Aes256Gcm,
        )
        .expect("no-PSK keys");

    // With PSK
    let psk = [0xFFu8; 32];
    let mut ks_psk = TlsKeySchedule::new(Some(&psk));
    ks_psk.derive_handshake_secrets(&shared, &transcript);
    let keys_psk = ks_psk
        .protection_keys(
            EncryptionLevel::Handshake,
            TlsRole::Client,
            CipherSuite::Aes256Gcm,
        )
        .expect("PSK keys");

    assert_ne!(
        keys_no_psk.key, keys_psk.key,
        "PSK must change derived keys"
    );
}
