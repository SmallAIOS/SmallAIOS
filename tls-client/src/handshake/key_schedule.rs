// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! TLS 1.3 key schedule (RFC 8446 §7.1) and transcript hash
//! (§4.4.1), generic over the negotiated cipher suite's hash.
//!
//! ```text
//!             0
//!             |
//!             v
//!   PSK ->  HKDF-Extract = Early Secret
//!             |
//!             +--> Derive-Secret(., "derived", "")
//!             |
//!             v
//! ECDHE -> HKDF-Extract = Handshake Secret
//!             |
//!             +--> Derive-Secret(., "c hs traffic", CH..SH)
//!             +--> Derive-Secret(., "s hs traffic", CH..SH)
//!             +--> Derive-Secret(., "derived", "")
//!             |
//!             v
//!     0 -> HKDF-Extract = Master Secret
//!             |
//!             +--> Derive-Secret(., "c ap traffic", CH..server Fin)
//!             +--> Derive-Secret(., "s ap traffic", CH..server Fin)
//! ```
//!
//! Per-record traffic keys come from the traffic secrets via
//! `HKDF-Expand-Label(secret, "key"/"iv", "", len)` (§7.3); the
//! Finished MAC key via `HKDF-Expand-Label(secret, "finished",
//! "", Hash.length)` (§4.4.4).
//!
//! ## Why not `net::quic::tls::TlsKeySchedule`?
//!
//! The change's tasks file originally pointed Phase 4 at the QUIC
//! module's `TlsKeySchedule`. That type is an XOR-based placeholder
//! ("Simplified: XOR-based derivation for stub" — see
//! `net/src/quic/tls.rs`) wired for QUIC's CRYPTO-frame flow; its
//! output does not interoperate with any RFC 8446 peer. This module
//! is the real HKDF schedule, keyed by the suite hash, validated
//! against vectors cross-checked with two independent oracles
//! (OpenSSL 3.0 `TLS13-KDF` and the Python stdlib HMAC/HKDF
//! construction). See design.md D12.

use crate::record::{CipherSuite, NONCE_LEN};
use crate::{Result, TlsClientError};
use alloc::vec::Vec;
use smallaios_security::hkdf::{
    hkdf_expand_sha256, hkdf_expand_sha384, hkdf_extract_sha256, hkdf_extract_sha384,
};
use smallaios_security::hmac_sha2::{hmac_sha256, hmac_sha384};
use smallaios_security::sha2::{Sha256, Sha384};

/// Maximum digest length across supported suites (SHA-384).
pub const MAX_HASH_LEN: usize = 48;

/// AEAD key length — 32 bytes for both supported suites
/// (AES-256-GCM and ChaCha20-Poly1305).
pub const TRAFFIC_KEY_LEN: usize = 32;

/// The hash algorithm a cipher suite keys its schedule with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HashAlg {
    Sha256,
    Sha384,
}

impl HashAlg {
    /// The suite's transcript/HKDF hash (RFC 8446 §B.4).
    pub fn for_suite(suite: CipherSuite) -> Self {
        match suite {
            CipherSuite::Aes256GcmSha384 => Self::Sha384,
            CipherSuite::ChaCha20Poly1305Sha256 => Self::Sha256,
        }
    }

    /// Digest length in bytes.
    pub fn digest_len(self) -> usize {
        match self {
            Self::Sha256 => 32,
            Self::Sha384 => 48,
        }
    }
}

/// A schedule secret: digest-sized bytes in a max-sized buffer.
#[derive(Clone, Copy)]
pub struct Secret {
    bytes: [u8; MAX_HASH_LEN],
    len: usize,
}

impl Secret {
    fn new(alg: HashAlg) -> Self {
        Self {
            bytes: [0u8; MAX_HASH_LEN],
            len: alg.digest_len(),
        }
    }

    fn from_slice(s: &[u8]) -> Self {
        let mut bytes = [0u8; MAX_HASH_LEN];
        bytes[..s.len()].copy_from_slice(s);
        Self {
            bytes,
            len: s.len(),
        }
    }

    /// The secret bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes[..self.len]
    }
}

impl core::fmt::Debug for Secret {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Never print key material.
        write!(f, "Secret({} bytes)", self.len)
    }
}

/// Streaming transcript hash over every handshake message
/// (header + body), in order (RFC 8446 §4.4.1).
#[derive(Clone)]
pub enum TranscriptHash {
    Sha256(Sha256),
    Sha384(Sha384),
}

impl TranscriptHash {
    /// Fresh transcript for the suite's hash.
    pub fn new(alg: HashAlg) -> Self {
        match alg {
            HashAlg::Sha256 => Self::Sha256(Sha256::new()),
            HashAlg::Sha384 => Self::Sha384(Sha384::new()),
        }
    }

    /// Absorb raw handshake-message bytes (including the 4-byte
    /// message header).
    pub fn update(&mut self, data: &[u8]) {
        match self {
            Self::Sha256(h) => h.update(data),
            Self::Sha384(h) => h.update(data),
        }
    }

    /// Hash of everything absorbed so far, without consuming the
    /// running state.
    pub fn current(&self) -> Vec<u8> {
        match self {
            Self::Sha256(h) => h.clone().finalize().to_vec(),
            Self::Sha384(h) => h.clone().finalize().to_vec(),
        }
    }
}

/// HKDF-Expand-Label (RFC 8446 §7.1):
///
/// ```text
/// struct {
///     uint16 length;
///     opaque label<7..255> = "tls13 " + Label;
///     opaque context<0..255>;
/// } HkdfLabel;
/// ```
fn hkdf_expand_label(
    alg: HashAlg,
    secret: &[u8],
    label: &[u8],
    context: &[u8],
    out: &mut [u8],
) -> Result<()> {
    debug_assert!(label.len() <= 249 && context.len() <= 255);
    let mut info = Vec::with_capacity(2 + 1 + 6 + label.len() + 1 + context.len());
    info.extend_from_slice(&(out.len() as u16).to_be_bytes());
    info.push((6 + label.len()) as u8);
    info.extend_from_slice(b"tls13 ");
    info.extend_from_slice(label);
    info.push(context.len() as u8);
    info.extend_from_slice(context);
    match alg {
        HashAlg::Sha256 => {
            let mut prk = [0u8; 32];
            if secret.len() != 32 {
                return Err(TlsClientError::KeyExchange);
            }
            prk.copy_from_slice(secret);
            hkdf_expand_sha256(&prk, &info, out).map_err(|_| TlsClientError::KeyExchange)
        }
        HashAlg::Sha384 => {
            let mut prk = [0u8; 48];
            if secret.len() != 48 {
                return Err(TlsClientError::KeyExchange);
            }
            prk.copy_from_slice(secret);
            hkdf_expand_sha384(&prk, &info, out).map_err(|_| TlsClientError::KeyExchange)
        }
    }
}

/// Derive-Secret (RFC 8446 §7.1): Expand-Label keyed by an
/// already-computed transcript hash, at digest length.
fn derive_secret(
    alg: HashAlg,
    secret: &[u8],
    label: &[u8],
    transcript_hash: &[u8],
) -> Result<Secret> {
    let mut out = Secret::new(alg);
    let len = out.len;
    hkdf_expand_label(alg, secret, label, transcript_hash, &mut out.bytes[..len])?;
    Ok(out)
}

fn extract(alg: HashAlg, salt: &[u8], ikm: &[u8]) -> Secret {
    match alg {
        HashAlg::Sha256 => Secret::from_slice(&hkdf_extract_sha256(salt, ikm)),
        HashAlg::Sha384 => Secret::from_slice(&hkdf_extract_sha384(salt, ikm)),
    }
}

fn hmac(alg: HashAlg, key: &[u8], message: &[u8]) -> Vec<u8> {
    match alg {
        HashAlg::Sha256 => hmac_sha256(key, message).to_vec(),
        HashAlg::Sha384 => hmac_sha384(key, message).to_vec(),
    }
}

/// AEAD write/read keys for one direction (RFC 8446 §7.3).
#[derive(Clone)]
pub struct TrafficKeys {
    pub key: [u8; TRAFFIC_KEY_LEN],
    pub iv: [u8; NONCE_LEN],
}

impl core::fmt::Debug for TrafficKeys {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "TrafficKeys(redacted)")
    }
}

/// Schedule stage marker — secrets only become readable once the
/// stage that derives them has run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Stage {
    Early,
    Handshake,
    Application,
}

/// The TLS 1.3 key schedule for one connection (client role,
/// no PSK — `psk_ke`/resumption is out of scope in v1).
pub struct KeySchedule {
    alg: HashAlg,
    stage: Stage,
    /// The "current" extract secret for the next stage transition.
    current: Secret,
    client_hs_traffic: Secret,
    server_hs_traffic: Secret,
    client_ap_traffic: Secret,
    server_ap_traffic: Secret,
}

impl core::fmt::Debug for KeySchedule {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "KeySchedule({:?}, {:?})", self.alg, self.stage)
    }
}

impl KeySchedule {
    /// Initialize with the Early Secret: `HKDF-Extract(0, 0)`
    /// (no PSK).
    pub fn new(alg: HashAlg) -> Self {
        let zeros = [0u8; MAX_HASH_LEN];
        let early = extract(alg, &[], &zeros[..alg.digest_len()]);
        Self {
            alg,
            stage: Stage::Early,
            current: early,
            client_hs_traffic: Secret::new(alg),
            server_hs_traffic: Secret::new(alg),
            client_ap_traffic: Secret::new(alg),
            server_ap_traffic: Secret::new(alg),
        }
    }

    /// The suite hash this schedule runs on.
    pub fn alg(&self) -> HashAlg {
        self.alg
    }

    /// Derive the handshake-traffic secrets (task 4.4).
    ///
    /// `ecdhe` is the raw key-exchange shared secret (32 bytes for
    /// x25519; 64 bytes — ML-KEM ss || X25519 ss — for the hybrid
    /// group). `transcript_hash` is Transcript-Hash(CH..SH).
    pub fn derive_handshake_secrets(&mut self, ecdhe: &[u8], transcript_hash: &[u8]) -> Result<()> {
        if self.stage != Stage::Early {
            return Err(TlsClientError::KeyExchange);
        }
        let empty_hash = TranscriptHash::new(self.alg).current();
        let derived = derive_secret(self.alg, self.current.as_bytes(), b"derived", &empty_hash)?;
        let handshake = extract(self.alg, derived.as_bytes(), ecdhe);
        self.client_hs_traffic = derive_secret(
            self.alg,
            handshake.as_bytes(),
            b"c hs traffic",
            transcript_hash,
        )?;
        self.server_hs_traffic = derive_secret(
            self.alg,
            handshake.as_bytes(),
            b"s hs traffic",
            transcript_hash,
        )?;
        self.current = handshake;
        self.stage = Stage::Handshake;
        Ok(())
    }

    /// Derive the application-traffic secrets (task 4.9).
    ///
    /// `transcript_hash` is Transcript-Hash(CH..server Finished).
    pub fn derive_application_secrets(&mut self, transcript_hash: &[u8]) -> Result<()> {
        if self.stage != Stage::Handshake {
            return Err(TlsClientError::KeyExchange);
        }
        let empty_hash = TranscriptHash::new(self.alg).current();
        let derived = derive_secret(self.alg, self.current.as_bytes(), b"derived", &empty_hash)?;
        let zeros = [0u8; MAX_HASH_LEN];
        let master = extract(
            self.alg,
            derived.as_bytes(),
            &zeros[..self.alg.digest_len()],
        );
        self.client_ap_traffic = derive_secret(
            self.alg,
            master.as_bytes(),
            b"c ap traffic",
            transcript_hash,
        )?;
        self.server_ap_traffic = derive_secret(
            self.alg,
            master.as_bytes(),
            b"s ap traffic",
            transcript_hash,
        )?;
        self.current = master;
        self.stage = Stage::Application;
        Ok(())
    }

    fn require(&self, stage: Stage) -> Result<()> {
        // Application stage retains handshake secrets — allow
        // reading hs secrets at either stage.
        let ok = match stage {
            Stage::Early => true,
            Stage::Handshake => self.stage != Stage::Early,
            Stage::Application => self.stage == Stage::Application,
        };
        if ok {
            Ok(())
        } else {
            Err(TlsClientError::KeyExchange)
        }
    }

    /// client_handshake_traffic_secret.
    pub fn client_hs_traffic_secret(&self) -> Result<&Secret> {
        self.require(Stage::Handshake)?;
        Ok(&self.client_hs_traffic)
    }

    /// server_handshake_traffic_secret.
    pub fn server_hs_traffic_secret(&self) -> Result<&Secret> {
        self.require(Stage::Handshake)?;
        Ok(&self.server_hs_traffic)
    }

    /// client_application_traffic_secret_0.
    pub fn client_ap_traffic_secret(&self) -> Result<&Secret> {
        self.require(Stage::Application)?;
        Ok(&self.client_ap_traffic)
    }

    /// server_application_traffic_secret_0.
    pub fn server_ap_traffic_secret(&self) -> Result<&Secret> {
        self.require(Stage::Application)?;
        Ok(&self.server_ap_traffic)
    }

    /// Per-direction AEAD keys from a traffic secret (§7.3).
    pub fn traffic_keys(&self, secret: &Secret) -> Result<TrafficKeys> {
        let mut key = [0u8; TRAFFIC_KEY_LEN];
        let mut iv = [0u8; NONCE_LEN];
        hkdf_expand_label(self.alg, secret.as_bytes(), b"key", &[], &mut key)?;
        hkdf_expand_label(self.alg, secret.as_bytes(), b"iv", &[], &mut iv)?;
        Ok(TrafficKeys { key, iv })
    }

    /// Finished verify_data for the given traffic secret (§4.4.4):
    ///
    /// ```text
    /// finished_key = HKDF-Expand-Label(BaseKey, "finished", "", Hash.length)
    /// verify_data = HMAC(finished_key, Transcript-Hash(...))
    /// ```
    pub fn finished_verify_data(&self, secret: &Secret, transcript_hash: &[u8]) -> Result<Vec<u8>> {
        let mut finished_key = [0u8; MAX_HASH_LEN];
        let len = self.alg.digest_len();
        hkdf_expand_label(
            self.alg,
            secret.as_bytes(),
            b"finished",
            &[],
            &mut finished_key[..len],
        )?;
        Ok(hmac(self.alg, &finished_key[..len], transcript_hash))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handshake::test_util::hex;

    fn h(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    // Synthetic-transcript key-schedule vectors. Every value was
    // generated twice — once with OpenSSL 3.0's `TLS13-KDF`
    // (EXTRACT_ONLY/EXPAND_ONLY modes) and once with a Python
    // stdlib HMAC/HKDF reference — and the two oracles agreed on
    // every intermediate secret.
    //
    // Transcript material (raw, unhashed):
    //   CH  = 01000200 || "synthetic-client-hello-body-for-kat"*3
    //   SH  = 02000100 || "synthetic-server-hello-body-for-kat"*2
    //   MID = "synthetic-ee||cert||certverify-bytes"*4
    //   FIN = "synthetic-server-finished-message"
    fn ch() -> Vec<u8> {
        let mut v = h("01000200");
        for _ in 0..3 {
            v.extend_from_slice(b"synthetic-client-hello-body-for-kat");
        }
        v
    }
    fn sh() -> Vec<u8> {
        let mut v = h("02000100");
        for _ in 0..2 {
            v.extend_from_slice(b"synthetic-server-hello-body-for-kat");
        }
        v
    }
    fn mid() -> Vec<u8> {
        let mut v = Vec::new();
        for _ in 0..4 {
            v.extend_from_slice(b"synthetic-ee||cert||certverify-bytes");
        }
        v
    }
    const FIN: &[u8] = b"synthetic-server-finished-message";

    struct Expected {
        alg: HashAlg,
        ecdhe: &'static str,
        t_ch_sh: &'static str,
        c_hs: &'static str,
        s_hs: &'static str,
        t_full: &'static str,
        c_ap: &'static str,
        s_ap: &'static str,
        s_hs_key: &'static str,
        s_hs_iv: &'static str,
        t_at_sfin_input: &'static str,
        s_verify_data: &'static str,
    }

    fn run_scenario(e: &Expected) {
        let mut ks = KeySchedule::new(e.alg);

        // Handshake secrets are gated until derived.
        assert!(ks.client_hs_traffic_secret().is_err());

        let mut transcript = TranscriptHash::new(e.alg);
        transcript.update(&ch());
        transcript.update(&sh());
        let t_ch_sh = transcript.current();
        assert_eq!(hex(&t_ch_sh), e.t_ch_sh);

        ks.derive_handshake_secrets(&h(e.ecdhe), &t_ch_sh).unwrap();
        assert_eq!(
            hex(ks.client_hs_traffic_secret().unwrap().as_bytes()),
            e.c_hs
        );
        assert_eq!(
            hex(ks.server_hs_traffic_secret().unwrap().as_bytes()),
            e.s_hs
        );

        // App secrets still gated.
        assert!(ks.client_ap_traffic_secret().is_err());

        // Server handshake traffic keys.
        let keys = ks
            .traffic_keys(ks.server_hs_traffic_secret().unwrap())
            .unwrap();
        assert_eq!(hex(&keys.key), e.s_hs_key);
        assert_eq!(hex(&keys.iv), e.s_hs_iv);

        // Server Finished verify_data over CH..CertificateVerify.
        transcript.update(&mid());
        let t_at_fin = transcript.current();
        assert_eq!(hex(&t_at_fin), e.t_at_sfin_input);
        let vd = ks
            .finished_verify_data(ks.server_hs_traffic_secret().unwrap(), &t_at_fin)
            .unwrap();
        assert_eq!(hex(&vd), e.s_verify_data);

        // App secrets over CH..server Finished.
        transcript.update(FIN);
        let t_full = transcript.current();
        assert_eq!(hex(&t_full), e.t_full);
        ks.derive_application_secrets(&t_full).unwrap();
        assert_eq!(
            hex(ks.client_ap_traffic_secret().unwrap().as_bytes()),
            e.c_ap
        );
        assert_eq!(
            hex(ks.server_ap_traffic_secret().unwrap().as_bytes()),
            e.s_ap
        );

        // Handshake secrets remain readable post-transition.
        assert_eq!(
            hex(ks.server_hs_traffic_secret().unwrap().as_bytes()),
            e.s_hs
        );

        // Double-transition refused.
        assert!(ks.derive_application_secrets(&t_full).is_err());
    }

    #[test]
    fn schedule_sha256_suite() {
        run_scenario(&Expected {
            alg: HashAlg::Sha256,
            ecdhe: "0a2643dee2a53e5ac789def7053c1a663e1a38ab87a65db0e93cdb666f08adf4",
            t_ch_sh: "db7d416283dbac908d50ee5965c3301f61018324927b2d76e2d59e5a48f6c0ca",
            c_hs: "f36435ad632b9bd79efc425723758ac8d8a9b7d0a414d0c1448bd4d5b6941b32",
            s_hs: "12b76515d84a0b861604020210fa6bdeca5dc318e5bd338691a0feefdf0261d4",
            t_full: "7b5d304b7647d6fba59c1aee9f78d03198650b42f3be455ff153ea65a5f13771",
            c_ap: "7947fd6113b86abdae06b622921c243134b6d75c79f5e858fa022c1c1a400c8d",
            s_ap: "52ea36e0ea0920079c53105ead6e5de899a1ce2170eb0aebe6b98eb871008b3b",
            s_hs_key: "d188e0e355f151b682b9732a74745fb0d4ff988eaf6b04716b0484d14923e8eb",
            s_hs_iv: "f4b6796a9f8a33ab1bf8610b",
            t_at_sfin_input: "0d925902be67d7ca179467bd7fb117551c970f7da6f21fcc62e817bb3d7523c4",
            s_verify_data: "7078cb671e7d749bb2061ee5616d0578e306debcbddc9155e2aa98b0562fc85b",
        });
    }

    #[test]
    fn schedule_sha384_suite() {
        run_scenario(&Expected {
            alg: HashAlg::Sha384,
            ecdhe: "0a2643dee2a53e5ac789def7053c1a663e1a38ab87a65db0e93cdb666f08adf4",
            t_ch_sh: "e803b50e86a7d46f9b2ec4f4c8340a92fcd23bd1a2111ab6c898167ee3c395dc\
72571378d25e72dd2a27b77e0f6fc7bf",
            c_hs: "55f8047598e2cea07b08133aba11964462e14415982f5b37b6989702b7e8588d\
db71929ea9adbd2be371be5640a12553",
            s_hs: "43ae32fb4edb01b3eaaad9e2c3576eb1a2a22eee2c6fa482dcdc50a7ea9ca4ea\
f46d0a5902698b2eb6c38f0c6b0caa25",
            t_full: "e53cfe4f5a7f0956890baa72d1e1f70fc52bb770dbb6322331a541ebcd0fe440\
dc32aa3f47fa04f4943976e3da706be8",
            c_ap: "3284fbd06e0fbf2b5c6bea3523723b5799091b5297e008120dbaeeb49e7fbae7\
414a3a04fa86b039f6d3edd4ce428f10",
            s_ap: "3c06bf11bb623e481ec4bd2d723fb9db8345ddcbe2bb7c72a3e1523d7f874f3f\
2153158cc24045228c579cfa81204fe0",
            s_hs_key: "9bbfd5b8e82b47cb83600f4029aba4e9e54de811889473c51b4124688474ce79",
            s_hs_iv: "694a44c03f1334da125d42c6",
            t_at_sfin_input: "883f481e52b466179a65f62cdacade01cf5f2eb79870a0593733247fb195b4b5\
edfa55339526265164790087b80d22f4",
            s_verify_data: "de435589fc3dd351b97d5714be5acb9bfb836991fa61b9eada848507b635208e\
e58af7cef835a568064a710bcb5b4717",
        });
    }

    #[test]
    fn schedule_sha256_hybrid_64_byte_ecdhe() {
        // The hybrid group's shared secret is 64 bytes
        // (ML-KEM-768 ss || X25519 ss) — exercise the wider IKM.
        run_scenario(&Expected {
            alg: HashAlg::Sha256,
            ecdhe: "9aa93c43c6959937ee1e467b892abf8d823d999ceda44a8e1f1a8f653ad04789\
f9a41999316e6976c96bf12e533324c925948512b0892b12d121825a216dd3d7",
            t_ch_sh: "db7d416283dbac908d50ee5965c3301f61018324927b2d76e2d59e5a48f6c0ca",
            c_hs: "cdd2c068573f9a404c38ef7779eebabfdac4058097c1a0c952ee5961935e72ec",
            s_hs: "21d9b6dbb2bb7b509eb7c07354e9fe320254cb21a12319581d6a0ae39d2cfae1",
            t_full: "7b5d304b7647d6fba59c1aee9f78d03198650b42f3be455ff153ea65a5f13771",
            c_ap: "61ed2cdf54749124aab58dad6b07b7ab8a0e2b512b0d1b489f28b5a7d27867a4",
            s_ap: "71cb2aeb7b692f1f90d999365c5d49ae8cb2a821b3a5478fd5eb27dabdecff1b",
            s_hs_key: "be510923febfcb36abaf7cb84e257278dfe58c18606a0fde28d84bf28f1c71ff",
            s_hs_iv: "1ded168b19c354965cbca710",
            t_at_sfin_input: "0d925902be67d7ca179467bd7fb117551c970f7da6f21fcc62e817bb3d7523c4",
            s_verify_data: "eb7c9e8fda3cc2c65e3f5442b89afcba014fbe38587c968a2881f198c622e268",
        });
    }

    #[test]
    fn early_secret_matches_known_value() {
        // HKDF-Extract(0, 0^32) over SHA-256 — the universal TLS 1.3
        // no-PSK early secret, also OpenSSL-confirmed.
        let ks = KeySchedule::new(HashAlg::Sha256);
        assert_eq!(
            hex(ks.current.as_bytes()),
            "33ad0a1c607ec03b09e6cd9893680ce210adf300aa1f2660e1b22e10f170f92a"
        );
    }

    #[test]
    fn suite_to_hash_mapping() {
        assert_eq!(
            HashAlg::for_suite(CipherSuite::Aes256GcmSha384),
            HashAlg::Sha384
        );
        assert_eq!(
            HashAlg::for_suite(CipherSuite::ChaCha20Poly1305Sha256),
            HashAlg::Sha256
        );
    }
}
