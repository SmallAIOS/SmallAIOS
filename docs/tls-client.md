# TLS Client — Operator Setup Guide

`tls-client` is SmallAIOS's clean-room TLS 1.3 client: `#![no_std]`
core, zero C dependencies, every primitive validated against
official vectors (see `docs/crypto-validation.md`). The audit
exporter uses it for the immudb connection; this guide covers the
operator-facing knobs. Design decisions live in
`openspec/changes/tls-tcp-client-v1/design.md`.

## What it speaks — and refuses

| Accepted | Refused |
|---|---|
| TLS 1.3 only | TLS 1.2 and below, anywhere in the handshake |
| `TLS_AES_256_GCM_SHA384`, `TLS_CHACHA20_POLY1305_SHA256` | `TLS_AES_128_GCM_SHA256` (design D2) |
| X25519, hybrid X25519+ML-KEM-768 key exchange | Other groups, HelloRetryRequest |
| Ed25519, ECDSA-P256, RSA-PSS certificate signatures | RSA PKCS#1 v1.5, SHA-1-signed certs |
| X.509 v3 with SAN | v1/v2 certs, missing SAN on the leaf |

No mTLS (v2, `auth_mode = "mtls"` is refused by the config
validator), no session resumption, no post-handshake KeyUpdate.

## Step 1 — Build the trust store

The client has **no baked-in CA bundle** (design D5): you provide a
PEM file of trust anchors, and an empty store refuses every chain.

```
# From the server operator, or extracted from a live endpoint:
openssl s_client -connect immudb.example.com:3322 -showcerts \
  </dev/null 2>/dev/null | awk '/BEGIN CERT/,/END CERT/' > chain.pem
# Keep only the anchor you intend to trust (usually the issuing CA).
```

Every block must be a CA certificate (`BasicConstraints CA:TRUE`)
with a unique Subject; the loader rejects non-CA certs, duplicate
Subjects, and empty bundles.

Install it where `tls.trust_store_path` points, with tight modes:

```
install -m 0640 anchors.pem /data/audit_export/anchors.pem
```

`connect_immudb` refuses a trust store whose permissions are laxer
than `0640` (or executable) — anchors are public material, but a
world-writable bundle is a CA-swap waiting to happen. Stricter
modes (`0600`, `0400`) pass.

**WebPKI note:** most public ECDSA chains are anchored at a P-384
root, which v1 cannot parse. Anchor at the SHA-256-signed
*intermediate* instead — the server-supplied chain still verifies.
Root-anchored WebPKI validation arrives with P-384 support
(`security-ecdsa-p384-v1`, proposed).

## Step 2 — Pin the anchor

`tls.server_pubkey_fingerprint` (mandatory when the exporter is
enabled) pins the chain to one exact trust anchor: the SHA-256 of
the anchor certificate's DER. Chains anchored at any other
certificate — including others in your own bundle — are refused.

```
openssl x509 -in anchors.pem -outform DER | sha256sum
```

The 64-hex-char digest goes in the TOML. If you rotate the CA, you
rotate this value with it.

## Step 3 — Choose the policy knobs

```toml
[tls]
require_pqc = false
server_pubkey_fingerprint = "<64 hex chars — SHA-256 of anchor DER>"
trust_store_path = "/data/audit_export/anchors.pem"
```

- `require_pqc = true` offers hybrid X25519+ML-KEM-768 **first**
  and hard-fails (`PqcDowngrade`) if the server picks a classical
  group. Leave it off unless the endpoint (or its TLS-terminating
  proxy) actually supports the hybrid draft — with it on, a
  classical-only server is unreachable by design.
- Certificate validity is checked against the system clock. A
  clock reading before 2026 is treated as *unsynced*: the
  verifier applies the design-D8 sentinel policy instead of
  hard-failing on `notBefore`. The stricter
  `require_synced_clock` switch exists at the API level
  (`TrustStoreVerifier::new`) but is not yet an `immudb.toml`
  key; the exporter currently runs with it off.
- Connection deadlines are fixed constants for now (10 s connect,
  30 s per read/write — `TLS_TIMEOUTS` in
  `container::audit_export_runtime::transport`); a timeout
  surfaces as a transient error and lands in the exporter's
  backoff loop.

## Troubleshooting

Failure classes below are what the exporter does with the error:
**Retry** = backoff and try again, **HardFail** = deterministic
policy failure, retrying cannot help — fix the config or the
server.

| Error | Class | Usual cause / fix |
|---|---|---|
| `TcpConnect` | Retry | Endpoint down, DNS, firewall. Check `endpoint`. |
| `Io` | Retry | Connection dropped mid-flight; read/write deadline hit. |
| `BadHandshake` | Retry | Malformed or out-of-order handshake message; HRR; CertificateRequest. Middlebox interference is the classic cause. |
| `BadRecord` | HardFail | Record framing violation (oversized, bad version byte). |
| `Version` | HardFail | Server negotiated TLS ≤ 1.2. Upgrade the server; the client will not downgrade. |
| `KeyExchange` | HardFail | Key-share math failed — usually a corrupt server share. |
| `BadCertificate` | HardFail | Leaf/intermediate failed to parse: v1/v2 cert, SHA-1 signature, missing SAN, unsupported P-384 anchor. |
| `ChainUntrusted` | HardFail | No path to a trust-store anchor, or pin mismatch. Re-check Step 1 + Step 2. |
| `Expired` | HardFail | Certificate outside its validity window — or the box's clock is wrong. |
| `NameMismatch` | HardFail | SAN doesn't cover the endpoint host (DNS-ID/iPAddress, RFC 6125 rules). |
| `PqcDowngrade` | HardFail | `require_pqc = true` but the server chose classical. |
| `Aead` | HardFail | Record decryption failed post-handshake — key desync or tampering. |

The lossy mapping onto the pipeline's `TransportError` collapses
most of these into `TlsHandshake`; the retry class above is what
`tls_retry_class` reports before that conversion.

## Testing hooks

- `tls-client/tests/e2e.rs` runs an `#[ignore]`-gated live
  handshake when `TLS_E2E_URL` and `TLS_E2E_TRUST_PEM` are set:
  ```
  TLS_E2E_URL=https://cloudflare.com:443 \
  TLS_E2E_TRUST_PEM=/path/to/anchor.pem \
  cargo test -p smallaios-tls-client --features std -- --ignored
  ```
- The `test-harness` feature exposes the mock TLS 1.3 server the
  unit tests drive the client against, so downstream crates can
  full-handshake-test their integration seams (`container`'s
  transport tests do). Dev-dependencies only — never enable it in
  a production build.
- Fuzz targets `fuzz_tls_record_parse` and `fuzz_tls_x509` cover
  the two attacker-controlled parsers (`fuzz/`).

## See also

- `docs/verifiable-audit-log.md` — the exporter this client ships
  with, including the TLS prerequisites checklist.
- `docs/crypto-validation.md` — how every primitive underneath is
  validated against official vectors.
