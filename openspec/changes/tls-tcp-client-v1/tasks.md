## 1. Crate scaffolding

- [x] 1.1 Create `tls-client/` crate at workspace Layer 1; register in `Cargo.toml` (workspace 27 → 28); confirm `#![no_std]` + `extern crate alloc`
- [x] 1.2 Add `smallaios-security` and `smallaios-kernel` (for `kernel::clock()` access) as dependencies
- [ ] 1.3 Add `smallaios-tls-client = { path = "../tls-client", optional = true }` to `container/Cargo.toml`; extend the existing `audit-export` feature to enable it *(deferred to Phase 8 integration)*
- [x] 1.4 Update CLAUDE.md workspace architecture diagram (27 → 28 crates; `tls-client` under Layer 1)
- [x] 1.5 Add `tls-client` to `Justfile` `host_crates` list and DSM allow-list
- [x] 1.6 Cyclic-dep check passes; clippy `-D warnings` clean on empty crate

## 2. `security::crypto::chacha20_poly1305`

- [x] 2.1 Add `security/src/crypto/chacha20.rs` with the IETF ChaCha20 stream cipher (RFC 8439 § 2.4)
- [x] 2.2 Add `security/src/crypto/poly1305.rs` with the Poly1305 one-time MAC (RFC 8439 § 2.5)
- [x] 2.3 Compose into `security/src/crypto/chacha20_poly1305.rs` providing `seal` / `open` (matches the existing `aes_gcm` shape; constant-time tag compare via inline accumulator)
- [x] 2.4 NIST + RFC 8439 KAT tests (≥4 vectors) including the canonical RFC 8439 § 2.8 example *(11 tests pass: § 2.3.2, § 2.4.2, § 2.5.2, § 2.8.2 + tamper-rejection variants + empty round-trips)*
- [ ] 2.5 `cargo-fuzz` target on `Aead::open` against arbitrary bytes *(deferred — added with the rest of the tls-client fuzz targets in Phase 10)*

## 3. Record layer

- [ ] 3.1 `tls-client/src/record.rs` — `ContentType`, `TLSPlaintext`, `TLSCiphertext` encoders + decoders
- [ ] 3.2 Enforce `TLSPlaintext.length ≤ 2^14` and `TLSCiphertext.length ≤ 2^14 + 256` BEFORE allocation
- [ ] 3.3 Refuse `LegacyVersion` outside `{0x0303}` (post-handshake)
- [ ] 3.4 AEAD wrap/unwrap routing on cipher-suite negotiation
- [ ] 3.5 Unit tests: round-trip for each of the three suites; oversized rejection; legacy-version rejection
- [ ] 3.6 `cargo-fuzz` target on the inbound record parser

## 4. Handshake state machine

- [ ] 4.1 `tls-client/src/handshake.rs` — message types: ClientHello, ServerHello, EncryptedExtensions, Certificate, CertificateVerify, Finished, NewSessionTicket (parsed-and-ignored in v1)
- [ ] 4.2 Build ClientHello: legacy_version=0x0303, supported_versions=[0x0304], supported_groups=[x25519] (or hybrid first when `require_pqc`), key_share matching first supported_group, signature_algorithms allow-list (per `tls-client-handshake` spec), server_name when DNS endpoint
- [ ] 4.3 Parse ServerHello: pin `supported_versions = 0x0304`; reject `HelloRetryRequest` (refused in v1); confirm key_share group matches advertised
- [ ] 4.4 Derive handshake-traffic keys via `net::quic::tls::TlsKeySchedule::derive_handshake_secrets`
- [ ] 4.5 Parse EncryptedExtensions; check SNI ack (server MUST echo if we sent one and accepted)
- [ ] 4.6 Parse Certificate; pass each cert to the cert-chain verifier (capability `tls-client-cert-chain`)
- [ ] 4.7 Parse CertificateVerify; verify signature against the leaf's pubkey + transcript hash; enforce the signature-suite allow-list
- [ ] 4.8 Parse server Finished; verify HMAC over transcript
- [ ] 4.9 Send client Finished; derive application-traffic keys; transition to data phase
- [ ] 4.10 Unit tests: ServerHello selecting TLS 1.2 → `Version` error; PqcDowngrade when `require_pqc` + classical reply; SHA-1 CertificateVerify → `BadCertificate`

## 5. X.509v3 parser + chain verifier

- [ ] 5.1 `tls-client/src/cert.rs` — minimal DER decoder + the X.509 field subset listed in design.md D4
- [ ] 5.2 Refuse `version != v3`, SHA-1 signatures, missing SAN
- [ ] 5.3 Length-bound every OCTET STRING / SEQUENCE before allocation
- [ ] 5.4 Chain construction: leaf → intermediate(s) → anchor in trust store; each intermediate must have `BasicConstraints.CA = true` + `KeyUsage.keyCertSign`
- [ ] 5.5 Signature verification on every link (Ed25519, ECDSA-P256, RSA-PSS)
- [ ] 5.6 Validity-window check with `kernel::clock()`; unsynced-clock sentinel handling (`tls.require_synced_clock`); audit `audit_export_unsynced_clock` on bypass
- [ ] 5.7 RFC 6125 hostname matcher (DNS-ID + iPAddress SAN; wildcard rules per design.md D6)
- [ ] 5.8 Cross-vector tests in `tls-client/tests/corpus/`: ≥10 known-good cert chains + ≥6 known-bad (expired, wrong issuer, mismatched SAN, SHA-1, no SAN, malformed length)
- [ ] 5.9 `cargo-fuzz` target on the X.509 parser (the largest attacker-controlled surface)

## 6. Trust store

- [ ] 6.1 `tls-client/src/trust.rs` — PEM bundle loader (base64-decode each `BEGIN CERTIFICATE` / `END CERTIFICATE` block)
- [ ] 6.2 Reject empty bundles, non-CA certs, duplicate Subjects
- [ ] 6.3 Optional pin verification: when `tls.trust_store_pin` is set, refuse chains anchored elsewhere
- [ ] 6.4 Wire `Config::validate` in `audit-export::config` to enforce: `enabled = true && trust_store_path = ""` → `ConfigError::TrustStoreRequired`
- [ ] 6.5 Tests: empty bundle, non-CA cert, duplicate Subjects, valid bundle, pinned vs non-pinned chain accept/reject

## 7. `std`-IO adapter (`TcpTlsStream`)

- [ ] 7.1 `tls-client/src/std_io/mod.rs` — `TcpTlsStream` wrapping `std::net::TcpStream`
- [ ] 7.2 `TcpTlsStream::connect(host, port, config) -> Result<Self, TlsClientError>` — drives the handshake state machine over the TCP socket
- [ ] 7.3 Implement `Read`, `Write`, and the existing `container::audit_export_runtime::transport::TlsStreamLike` trait (with `close()` sending `close_notify`)
- [ ] 7.4 Map `TlsClientError` → `TransportError`: TcpConnect / Io / BadHandshake → `Retry`; everything else → `HardFail`
- [ ] 7.5 Unit tests with a `Vec<u8>`-backed mock socket exercising the handshake state machine end-to-end
- [ ] 7.6 `tls-client/tests/e2e.rs` — `#[ignore]`-gated test against `TLS_E2E_URL` (real https endpoint) when the env var is set

## 8. `audit-export` integration

- [ ] 8.1 In `container/src/audit_export_runtime/`, replace the `TlsStreamLike` placeholder docs with a pointer to `tls_client::std_io::TcpTlsStream`
- [ ] 8.2 Add a `connect_immudb(config: &Config) -> Result<TcpTlsStream, TransportError>` helper that bridges `Config::endpoint` + `Config::server_pubkey_fingerprint` + `Config::trust_store_path` into the TLS client
- [ ] 8.3 Update the `audit_export_runtime::transport` tests: at least one test that drives a full handshake against a `MockServer` exposing canned TLS records
- [ ] 8.4 Confirm the existing `audit-export-immudb-client` scenarios ("TLS 1.2 handshake rejected", "PQC hybrid offered first", "HTTP/2 server push refused") now pass end-to-end

## 9. Documentation

- [ ] 9.1 Add `docs/tls-client.md` — operator-facing setup guide: trust-store population, pinning, `require_pqc` flag, `require_synced_clock` flag, troubleshooting `TlsClientError::*` codes
- [ ] 9.2 Extend `docs/verifiable-audit-log.md` with a "TLS prerequisites" section that points operators at `docs/tls-client.md`
- [ ] 9.3 Update CLAUDE.md Crate Feature Flags section: `tls-client` (no feature flags in v1; transitive activation via `container::audit-export`)

## 10. CI plumbing

- [ ] 10.1 `tls-client` added to host-testable list (workspace-wide `cargo test` auto-picks)
- [ ] 10.2 Fuzz Testing job extended with `fuzz_tls_record_parse`, `fuzz_tls_x509`
- [ ] 10.3 Coverage gate: ≥85 % line coverage on `tls-client` first pass
- [ ] 10.4 Pin-check job ensures `tls.trust_store_path` permissions are 0640 (defense-in-depth — operator-readable, world-unreadable)

## 11. Verification

- [ ] 11.1 `cargo test -p smallaios-tls-client` — all unit + cross-vector tests pass
- [ ] 11.2 `cargo clippy -p smallaios-tls-client -- -D warnings` clean
- [ ] 11.3 `cargo test -p smallaios-container --features audit-export` — 30 existing audit_export_runtime tests still pass plus the new TLS-end-to-end test
- [ ] 11.4 `openspec validate tls-tcp-client-v1 --strict` passes
- [ ] 11.5 `cargo build -p smallaios-container` (no feature) still compiles cleanly — D10 Layer 1 zero-overhead-when-off invariant from `verifiable-audit-log-v1` preserved
