## Why

`verifiable-audit-log-v1` lands a `TlsGrpcTransport` adapter
keyed on the `TlsStreamLike: Read + Write + close()` trait,
but nothing in the workspace yet implements that trait over
TCP. Audit records cannot leave the box. The same TLS-over-
TCP gap blocks every other future change that wants to talk
to a vanilla HTTPS endpoint from a SmallAIOS deployment:
OTLP push to Grafana Cloud (`telemetry-otel-export-v1`), any
generic webhook surface, container-image pulls beyond
in-toto-anchored attestations.

The workspace already ships TLS 1.3's *cryptographic* core
in `net::quic::tls`: hybrid `X25519 + ML-KEM-768` key share,
key schedule, HKDF + SHA-3 derivations, AEAD record
protection. That code is wired for QUIC's CRYPTO-frame
transport, **not** TLS records over TCP. To stand up a
real TLS-over-TCP client we need to add the record framing
+ handshake state machine + certificate-chain parser, and
make the existing crypto reachable from a `TcpStream`-
backed `TlsStreamLike` implementation.

Doing this as its own change keeps the surface small,
testable, and reviewable: we deliver a single new TLS-over-
TCP client that drops cleanly into the trait already
shipped in `audit-export`, without entangling it with the
audit-export semantics.

## What Changes

### New crate: `tls-client/`

Layer 1 Rust crate, `#![no_std]` library with a thin `std`
adapter for the integration layer:

- **TLS record framing** (RFC 8446 § 5): `ContentType`,
  `TLSPlaintext`, `TLSCiphertext`, the 5-byte record header,
  16 KiB max plaintext / 16,640 B max ciphertext, the
  optional MAC-then-encrypt mode is **not** supported
  (TLS 1.2 — refused).
- **Handshake state machine** for the client side only
  (RFC 8446 § 4): ClientHello → ServerHello →
  EncryptedExtensions → Certificate →
  CertificateVerify → Finished → Client Finished. Refuses
  HelloRetryRequest in v1 (mid-handshake retry is rare for
  the immudb path; flag for v2 if needed).
- **Hybrid key exchange**: reuses `net::quic::tls`'s
  `HybridKeyShare` and `TlsKeySchedule` verbatim — we add
  the *plumbing* that wraps them in TLS records, not new
  crypto.
- **Cipher suites supported**: `TLS_AES_128_GCM_SHA256`,
  `TLS_AES_256_GCM_SHA384`, `TLS_CHACHA20_POLY1305_SHA256`.
  Each maps to the AEAD primitives already in
  `security::crypto::aes_gcm` (existing) and
  `security::crypto::chacha20_poly1305` (new — see
  modified-capability below).
- **Certificate chain verification**: parse minimum
  X.509v3 — issuer + subject + SAN + signature + validity
  window — verify the chain against a configured trust
  store, refuse chains that don't anchor in it. NameConstraints,
  CRL fetching, OCSP stapling all out of scope for v1.
- **Server name indication** (SNI) — required by
  reputable hosts and the dominant identity binding for
  HTTPS endpoints today.
- **`std` adapter**: `tls_client::std_io::TcpTlsStream`
  implementing the trait
  `container::audit_export_runtime::transport::TlsStreamLike`
  over `std::net::TcpStream`. Layered so the bare-metal
  (`#![no_std]`) callers can supply their own raw I/O.
- **No mTLS in v1**: client certificates are not
  presented. Operator-facing systems where the immudb
  bearer token is the auth primitive don't need it.
  Documented for follow-on `tls-mtls-v2`.

### What this does **not** ship

- TLS server-side code. Client only.
- 0-RTT (early data). Defer to v2.
- Session resumption / PSK. Defer to v2.
- DTLS / QUIC integration (already covered by
  `net::quic::tls`).
- Hardware TLS offload.

## Capabilities

### New Capabilities

- `tls-client-record-layer`: the TLS record header layout,
  size caps, AEAD framing, refusal-of-TLS-1.2-records rules,
  and the read/write loop that drives the post-handshake
  data path.
- `tls-client-handshake`: ClientHello / ServerHello /
  EncryptedExtensions / Certificate / CertificateVerify /
  Finished state machine, hybrid-key-share negotiation,
  SNI emission, version-pinning to TLS 1.3, and the
  Ed25519 / RSA-PSS / ECDSA-P256 + SHA-256 signature-suite
  matrix the client accepts on CertificateVerify.
- `tls-client-cert-chain`: minimum X.509v3 parser
  (issuer, subject, SAN, signature algorithm + value,
  validity window), trust-store-anchored chain
  verification, hostname-matching against SAN (DNS +
  IP), and rejection rules (expired, unknown issuer,
  mismatched SAN).
- `tls-client-trust-store`: the operator-facing
  configuration surface for trusted CAs and the
  invariants the loader enforces (PEM bundle path,
  per-CA fingerprint pinning option, refusal of an
  empty trust store).

### Modified Capabilities

- `audit-export-immudb-client`: the v1 spec required
  TLS 1.3 with PQC-hybrid available; the trait boundary
  (`TlsStreamLike`) was the bridge. This change wires the
  concrete `TcpTlsStream` impl, so the existing scenarios
  ("TLS 1.2 handshake rejected", "PQC hybrid offered first",
  "HTTP/2 server push refused") are now end-to-end
  testable. No requirement changes; the modification is
  just the binding of an existing trait to a concrete
  implementation.

### New crate-level addition: `chacha20_poly1305` in `security/`

`security::crypto::chacha20_poly1305` does not yet exist in
the workspace. TLS 1.3 mandates the cipher suite
`TLS_CHACHA20_POLY1305_SHA256` for environments without
AES hardware acceleration (low-power ARM, RISC-V boards
without the Cryptography Extension). This change adds the
implementation behind the existing `security/crypto/` module
boundary. Treated as a sub-capability of this proposal
rather than a separate change, since it has no consumer
outside TLS 1.3.

## Impact

- **Code**:
  - New `tls-client/` crate (~2,500 LOC):
    - `record.rs` — TLS record framing (~250).
    - `handshake.rs` — handshake state machine + message
      builders/parsers (~600).
    - `cipher.rs` — cipher-suite registry, AEAD adapters
      (~200).
    - `cert.rs` — X.509v3 parser + chain verification (~700).
    - `trust.rs` — trust-store loader + fingerprint pinning
      (~150).
    - `std_io/` — `TcpTlsStream` impl wrapping
      `std::net::TcpStream`, implementing `TlsStreamLike`
      from `container::audit_export_runtime::transport`
      (~300).
    - tests + fuzz harnesses (~300).
  - New `security::crypto::chacha20_poly1305` (~300 LOC +
    NIST KAT tests).

- **Tests**: ~120 new. Coverage:
  - Record-layer round-trips for the three cipher suites.
  - Handshake-message round-trips against vectors from the
    TLS 1.3 conformance corpus (`go.dev/x/crypto/tls`'s
    pre-recorded transcripts as ground truth).
  - X.509 parser against the standard
    `certificate-transparency` test corpus.
  - Chain verifier accepts a known-good chain; rejects
    expired, wrong-issuer, mismatched-SAN, and
    self-signed-without-trust-anchor variants.
  - Hostname matcher: literal DNS, wildcard `*.example.com`,
    rejection of mid-label wildcards (`x*.example.com`),
    IPv4 + IPv6 literal SAN.
  - PQC hybrid offered first when configured; pure-classical
    fallback when not.
  - TLS 1.2 ClientHello reply triggers handshake abort.
  - Fuzz target on the X.509 parser (the largest
    attacker-controlled surface).

- **Dependencies**:
  - `tls-client` depends on `security` (crypto primitives),
    `kernel` (timestamps for validity-window check).
  - `container` adds `tls-client` as an optional dep gated
    by the existing `audit-export` feature.
  - `audit-export` itself is unchanged — it already
    declares the `TlsStreamLike` trait it consumes.

- **Boot footprint** (when `audit-export` is enabled):
  ~400 KB code added for the TLS client, ~32 KB live (one
  TLS connection's key schedule + record buffers).

- **Threat model**:
  - X.509 parsers are notorious CVE surfaces — the fuzz
    target is non-optional and runs per-PR.
  - The handshake's signature-suite matrix is restrictive
    by design: Ed25519, RSA-PSS-3072+, ECDSA-P256, all with
    SHA-256 or stronger. SHA-1-signed certs refused.
  - Hostname matching against SAN follows RFC 6125
    verbatim, no permissive fallbacks.
  - Trust-store contents are operator-controlled; the
    `audit-export` operator guide gets a new section on
    how to populate it.

- **Out-of-band**: a planned follow-on `tls-mtls-v2` adds
  client-certificate presentation (the `audit-export`
  spec's `auth_mode = "mtls"` path).
