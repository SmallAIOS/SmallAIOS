## Why

`tls-tcp-client-v1` plans to verify CertificateVerify and
X.509 chain signatures. Its design.md and the
`tls-client-handshake` spec already pin the allowed
signature schemes to a list that includes
`rsa_pss_rsae_sha256`, `rsa_pss_rsae_sha384`, and
`rsa_pss_rsae_sha512` — the three RSA-PSS variants that
dominate the public-CA-signed certificate ecosystem.

The workspace today has zero RSA support. Without RSA-PSS
verification, the tls-client cannot validate any cert
chain anchored at a public CA that issues RSA leaves
(which is most of them). The Phase 5 cert verifier is
blocked on this primitive.

Doing RSA-PSS as its own change keeps the surface bounded:
modular exponentiation, MGF1, EMSA-PSS-VERIFY, and a clear
"verify-only, no signing" scope. The crypto belongs at
Layer 0 in `security/crypto/`; every other change consumes
it as a stable trait.

## What Changes

### New module: `security::crypto::rsa_pss`

Layer 0 addition to `security/`, `#![no_std]`. **Verify-only**
in v1 (no key generation, no signing) — clients verify
signatures; they never produce them.

- **Key length support:** RSA-2048, RSA-3072, RSA-4096.
  RSA-1024 explicitly refused (NIST SP 800-131A retires
  RSA < 2048).
- **Hash function support:** SHA-256, SHA-384, SHA-512 —
  matching the three sig_scheme codes from
  `tls-client-handshake`'s allow-list.
- **MGF1 mask generation:** parameterized by the
  underlying hash function.
- **EMSA-PSS-VERIFY** per RFC 8017 § 9.1.2: salt length =
  hash length (the standard configuration, what TLS 1.3
  servers emit).
- **Modular exponentiation:** clean-room big-integer
  arithmetic with constant-time exponentiation (Montgomery
  ladder); the verify operation does not need
  constant-time guarantees in the same way signing does,
  but writing it that way keeps the primitive reusable.
- **DER pub-key parser:** extracts `(n, e)` from a
  `SubjectPublicKeyInfo` carrying an RSA public key.
- **Test vectors:** NIST CAVP RSA-PSS verify vectors
  (≥30 vectors covering all 9 (key-size × hash)
  combinations, plus known-bad signatures that MUST
  reject).

### Crate-level addition: `security::crypto::big_int` (internal)

Big-integer arithmetic primitives needed by RSA. Scoped as
a private module — not re-exported — to keep the public
crypto surface tight. Operations needed:

- Variable-width unsigned bigint (backed by `Vec<u64>`).
- Add, subtract, multiply.
- Montgomery reduction.
- Modular exponentiation (Montgomery ladder).
- DER INTEGER parsing into bigint.

### What this does **not** ship

- **RSA-PKCS#1 v1.5 (PKCSv15) signing or verification.**
  Padding oracles + Bleichenbacher legacy. Refused.
- **RSA-PSS signing.** Verify-only.
- **RSA-OAEP encryption.** Different padding scheme;
  not a TLS 1.3 dependency.
- **Key generation.** Off-box concern.
- **Hardware acceleration** (AES-NI-style RSA cores).
  Pure-software pad now; AVX/NEON sub-add later if perf
  becomes a problem.

## Capabilities

### New Capabilities

- `security-rsa-pss-verify`: the `rsa_pss::verify`
  function contract, key-size + hash-size matrix, RFC 8017
  EMSA-PSS-VERIFY conformance, MGF1 parameterization rule,
  refusal of RSA < 2048 and SHA-1.

### Modified Capabilities

- `tls-client-handshake` (from `tls-tcp-client-v1`):
  the CertificateVerify signature-suite allow-list now
  has a real implementation. No requirement changes — this
  proposal lands the missing primitive the existing
  scenarios already reference.

## Impact

- **Code:**
  - New `security/src/crypto/big_int.rs` (~500 LOC).
  - New `security/src/crypto/mgf1.rs` (~50 LOC).
  - New `security/src/crypto/rsa_pss.rs` (~400 LOC + ~400
    LOC of NIST CAVP test vectors).
- **Tests:** ~50 new. NIST CAVP RSA-PSS verify vectors at
  each (key-size × hash) combination + known-bad vectors
  that must reject + Bleichenbacher-style malformed
  signatures.
- **Boot footprint:** ~30 KB code added when the tls-client
  feature is enabled (RSA verify isn't called when it's off).
- **Dependencies:** no new external deps. Uses the
  existing `security/crypto/sha2` (SHA-256) and would need
  SHA-384/SHA-512 — both small extensions of the existing
  FIPS 180-4 SHA family. Either added in this change or
  in a tiny sub-add `security-sha512-v1`.
- **Threat model:**
  - Constant-time modular exponentiation: deferred but
    flagged. RSA-PSS *verify* does not handle a secret, so
    timing leaks are less catastrophic than they are for
    signing. Documented.
  - Variable-time bigint operations on non-secret inputs
    are acceptable for verification.
  - The EMSA-PSS comparison MUST be constant-time
    (`constant_time_eq`), matching the workspace's
    existing convention.
- **Out-of-band:** a follow-on `security-rsa-sign-v1` adds
  signing for cases where SmallAIOS needs to produce
  outgoing RSA signatures (currently none — the only RSA
  consumer is the TLS-client X.509 verifier). Tracked, not
  blocking.
