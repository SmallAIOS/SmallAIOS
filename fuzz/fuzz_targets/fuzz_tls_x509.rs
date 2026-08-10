#![no_main]
use libfuzzer_sys::fuzz_target;
use smallaios_tls_client::cert::{der, x509::Certificate};
use smallaios_tls_client::trust::TrustStore;

// tls-tcp-client-v1 task 5.9: the DER/X.509 decoder parses
// attacker-supplied certificate bytes mid-handshake — the largest
// parsing surface in the TLS client. The raw TLV reader, the full
// X.509v3 structure parser, and the PEM trust-store loader must
// reject garbage without panicking or over-allocating.
fuzz_target!(|data: &[u8]| {
    let mut reader = der::Reader::new(data);
    while !reader.is_empty() && reader.next_tlv().is_ok() {}

    let _ = Certificate::parse(data);

    if let Ok(pem) = core::str::from_utf8(data) {
        let _ = TrustStore::from_pem(pem);
    }
});
