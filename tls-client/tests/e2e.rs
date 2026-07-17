// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Real-endpoint end-to-end test (`tls-tcp-client-v1` task 7.6).
//!
//! `#[ignore]`-gated: never runs in CI's default pass. Run it by
//! hand against a live HTTPS endpoint:
//!
//! ```sh
//! TLS_E2E_URL=example.com:443 \
//! TLS_E2E_TRUST_PEM=/path/to/roots.pem \
//!   cargo test -p smallaios-tls-client --features std --test e2e -- --ignored
//! ```
//!
//! `TLS_E2E_URL` is `host[:port]` (port defaults to 443).
//! `TLS_E2E_TRUST_PEM` is a PEM bundle holding the CA root(s) that
//! anchor the endpoint's chain — SmallAIOS ships no baked-in bundle
//! (design.md D5), so the operator must supply one (e.g. exported
//! from the host's store). Both variables unset/empty → the test
//! reports how to enable itself and passes vacuously, so a plain
//! `--ignored` sweep doesn't fail on machines without network.
//!
//! What this proves that the corpus replay cannot: the full stack —
//! TCP, record layer, hybrid key exchange, handshake driver,
//! CertificateVerify, and a real WebPKI ECDSA/RSA chain through
//! `TrustStoreVerifier` — against a production TLS 1.3 server.

#![cfg(feature = "std")]

use smallaios_tls_client::cert::verify::TrustStoreVerifier;
use smallaios_tls_client::handshake::driver::ClientConfig;
use smallaios_tls_client::std_io::TcpTlsStream;
use smallaios_tls_client::trust::TrustStore;
use std::io::{Read, Write};
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
#[ignore = "needs TLS_E2E_URL + TLS_E2E_TRUST_PEM and network access"]
fn real_endpoint_handshake_and_http_get() {
    let url = std::env::var("TLS_E2E_URL").unwrap_or_default();
    let trust_pem = std::env::var("TLS_E2E_TRUST_PEM").unwrap_or_default();
    if url.is_empty() || trust_pem.is_empty() {
        eprintln!(
            "skipping: set TLS_E2E_URL=host[:port] and \
             TLS_E2E_TRUST_PEM=/path/to/roots.pem to run"
        );
        return;
    }

    let (host, port) = match url.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse::<u16>().expect("bad port")),
        None => (url, 443),
    };

    let pem = std::fs::read_to_string(&trust_pem).expect("read TLS_E2E_TRUST_PEM");
    let store = TrustStore::from_pem(&pem).expect("parse trust bundle");
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock before epoch")
        .as_secs() as i64;
    let verifier = TrustStoreVerifier::new(&store, now, true);

    let config = ClientConfig {
        server_name: Some(&host),
        require_pqc: false,
    };

    let mut stream =
        TcpTlsStream::connect(&host, port, config, &verifier).expect("TLS handshake failed");

    let req = format!("GET / HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n");
    stream.write_all(req.as_bytes()).expect("write request");
    stream.flush().expect("flush");

    let mut response = Vec::new();
    // Read until close_notify / EOF; tolerate servers that reset
    // after close_notify instead of a clean FIN.
    let _ = stream.read_to_end(&mut response);

    let head = String::from_utf8_lossy(&response[..response.len().min(64)]);
    assert!(
        head.starts_with("HTTP/1.1 ") || head.starts_with("HTTP/1.0 "),
        "expected an HTTP response head, got: {head:?}"
    );
    let _ = stream.close();
}
