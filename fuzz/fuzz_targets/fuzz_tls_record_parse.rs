#![no_main]
use libfuzzer_sys::fuzz_target;
use smallaios_tls_client::record::{self, CipherSuite, RecordHeader};

// tls-tcp-client-v1 tasks 3.6 + 2.5: the inbound TLS record is
// attacker-controlled surface #1 on the audit-export path. Header
// parse, the 2^14+256 size caps, and AEAD open (tag check, padding
// strip, inner content type) must never panic, overflow, or
// over-allocate on arbitrary bytes.
fuzz_target!(|data: &[u8]| {
    let _ = RecordHeader::parse(data);

    let key = [0x42u8; 32];
    let iv = [0x24u8; 12];
    for suite in [
        CipherSuite::Aes256GcmSha384,
        CipherSuite::ChaCha20Poly1305Sha256,
    ] {
        let _ = record::open(suite, &key, &iv, 0, data);
    }

    // Direct AEAD open (task 2.5): arbitrary ciphertext + tag must
    // fail authentication cleanly, never crash.
    if data.len() >= 16 {
        let (ct, tag_bytes) = data.split_at(data.len() - 16);
        let mut buf = ct.to_vec();
        let mut tag = [0u8; 16];
        tag.copy_from_slice(tag_bytes);
        let _ = smallaios_security::crypto::chacha20_poly1305::open(&key, &iv, b"", &mut buf, &tag);
    }
});
