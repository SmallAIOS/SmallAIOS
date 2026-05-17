## Why

The `tls-client-handshake` spec's signature-scheme
allow-list includes `ecdsa_secp256r1_sha256` — the ECDSA
variant TLS 1.3 servers emit when their leaf cert is
signed with an NIST P-256 (secp256r1) key. That covers the
modern publicly-trusted ECDSA-signed half of the TLS
ecosystem (Let's Encrypt, Cloudflare-issued, AWS-issued
short-lived certs, …).

The workspace today has zero ECDSA support. Without it,
the Phase 5 cert verifier in `tls-tcp-client-v1` cannot
validate any chain anchored at a public CA whose
intermediate or leaf uses ECDSA-P256. Combined with the
parallel `security-rsa-pss-v1` change, this closes the
gap for the two dominant classical signature schemes in
real-world TLS deployments.

ECDSA on P-256 is also useful beyond TLS — Sigstore Rekor
inclusion proofs, COSE signed objects, JWT/JWS — so the
primitive earns its place as a Layer-0 building block.

## What Changes

### New module: `security::crypto::ecdsa_p256`

Layer 0 addition to `security/`, `#![no_std]`. **Verify-only**
in v1 (no key generation, no signing) — same scope
discipline as `security-rsa-pss-v1`.

- **Curve:** NIST P-256 (secp256r1) only. P-384 / P-521 /
  brainpool curves all out of scope — flag for follow-on
  if/when needed.
- **Hash:** SHA-256 paired with the curve, matching the
  `ecdsa_secp256r1_sha256` TLS signature scheme.
- **ECDSA verify:** classical algorithm per ANSI X9.62 /
  RFC 6979:
  1. Parse `(r, s)` from the DER ASN.1 SEQUENCE form
     `SEQUENCE { INTEGER r, INTEGER s }`.
  2. Reject `r` or `s` outside `[1, n-1]`.
  3. Compute `u1 = H(m) * s^-1 mod n`,
     `u2 = r * s^-1 mod n`.
  4. Compute `R = u1*G + u2*Q` where `Q` is the public
     key point and `G` is the curve generator.
  5. Reject the signature if `R` is the point at infinity.
  6. Accept iff `R.x mod n == r`.
- **DER pub-key parser:** extracts the public point from a
  `SubjectPublicKeyInfo` carrying an `id-ecPublicKey` +
  `secp256r1` parameter, uncompressed form (the standard
  TLS-emitted form).
- **Point arithmetic:** Jacobian coordinates for efficient
  point doubling/addition, plus a constant-time
  scalar-multiplication ladder.
- **Test vectors:** Wycheproof's `ecdsa_secp256r1_sha256_test.json`
  corpus (≥150 vectors covering known-good, malformed
  DER, edge-case r/s values, and the standard set of
  fault-injection scenarios).

### What this does **not** ship

- **ECDSA signing.** Verify-only. Signing requires a
  CSPRNG with the constant-time-nonce-derivation
  property and is its own correctness story.
- **Other curves.** P-384, P-521, secp256k1 — none are
  needed for the TLS path; out of scope.
- **EdDSA.** Already shipped as
  `security::crypto::ed25519`.

## Capabilities

### New Capabilities

- `security-ecdsa-p256-verify`: the
  `ecdsa_p256::verify` function contract, DER `(r, s)`
  parser refusal rules (range, length, malleability),
  the Wycheproof corpus replay contract.

### Modified Capabilities

- `tls-client-handshake` (from `tls-tcp-client-v1`): the
  CertificateVerify signature-suite allow-list now has a
  real implementation for the ECDSA variant. No
  requirement changes.

## Impact

- **Code:**
  - New `security/src/crypto/p256.rs` (~600 LOC: curve
    constants, field arithmetic, point operations).
  - New `security/src/crypto/ecdsa_p256.rs` (~200 LOC:
    DER parse + verify).
  - Test corpus (~3,000 lines of Wycheproof vectors).
- **Tests:** ~150 (Wycheproof) + ~10 (unit). Coverage
  target ≥95 % line coverage on the verify primitive.
- **Boot footprint:** ~25 KB code when tls-client feature
  is enabled.
- **Dependencies:** Uses existing `security/crypto/sha2`
  (SHA-256). No new external deps.
- **Threat model:**
  - Point-at-infinity / weak-key checks per Wycheproof
    fault corpus.
  - DER parse rejects negative `r` or `s` (high bit
    set), zero `r` or `s`, `r` or `s` ≥ n, oversized
    encodings.
  - Signature malleability: ECDSA is fundamentally
    malleable (s and n-s both verify). We do NOT enforce
    the "low-s only" rule in v1 because no public CA
    issues low-s-only signatures and rejecting valid
    high-s signatures would break the cert chain. Flag
    for v2 if a specific deployment requires it.
- **Out-of-band:** a follow-on `security-ecdsa-sign-v1`
  adds signing once a use case appears. Today, nothing on
  SmallAIOS produces outgoing ECDSA signatures.
