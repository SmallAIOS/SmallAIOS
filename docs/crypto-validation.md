# Crypto Validation Strategy

**Status:** Active policy (decision record, 2026-07)
**Spec:** `openspec/specs/crypto-validation-policy/` (delta in
`openspec/changes/crypto-validation-strategy-v1/` until archived)

SmallAIOS implements all cryptography as clean-room `#![no_std]` Rust in
the Layer-0 `security` crate. This document records the policy that makes
that defensible, the decision **not** to adopt a validated C crypto
library (wolfSSL/wolfCrypt was the concrete candidate evaluated in
2026-07), and the conditions under which that decision gets revisited.

## Policy

1. **Official-corpus replay.** Every cryptographic primitive in
   `security/` replays an official public test corpus — Wycheproof/C2SP,
   NIST CAVP/ACVP, or the defining RFC/FIPS test vectors — inside the
   crate's standard `cargo test` run. The corpus is checked into the
   repository, every vector executes (no sampling), and a failing vector
   identifies itself (test-case id or index). A change proposal that adds
   a primitive must name its corpus before implementation starts.
2. **No C/C++ crypto libraries.** wolfSSL/wolfCrypt, OpenSSL, BoringSSL,
   mbedTLS, libsodium, and their binding crates are banned workspace-wide,
   at every layer. This is enforced mechanically: `deny.toml` carries
   `[bans]` entries for the known binding crates, so the Supply Chain
   Security CI gate fails any PR that introduces one. The ban list is a
   tripwire, not an exhaustive definition — the rule covers any C/C++
   cryptographic implementation, and review covers names the list misses.
3. **The FIPS question stays answered.** The trade study below is the
   standing answer to "should we just use a validated library?" — new
   discussion should start from it, not from scratch.

## Why not wolfSSL/wolfCrypt (or any validated C library)

Evaluated 2026-07 against wolfCrypt specifically (open source,
FIPS 140-3 certified, DO-178C DAL A kits sold). Four reasons, in
decreasing order of weight:

1. **Memory safety by construction.** SmallAIOS is a single-address-space
   unikernel whose core premise is a memory-safe kernel attack surface.
   Statically linking a C crypto library reintroduces exactly the defect
   class the architecture exists to exclude — in the most
   security-critical code — and inflates the `cargo-geiger` unsafe
   surface CI tracks.
2. **Licensing.** wolfSSL/wolfCrypt is dual-licensed GPL-or-commercial.
   Statically linked into the Apache-2.0 unikernel image, GPL terms
   extend to the shipped binary; the escape is a paid commercial license.
   The `cargo-deny` license gate rejects GPL in the dependency tree.
3. **FIPS validation does not transfer.** A FIPS 140-3 certificate
   attaches to the certified module build on its tested operational
   environments. Compiling wolfCrypt into a custom bare-metal unikernel
   is outside that boundary; the validation claim evaporates without a
   paid operational-environment addition or private-label validation.
   The main benefit of adopting C crypto would not actually be obtained.
4. **DO-178C evidence economics.** The project strategy is MC/DC
   coverage on our own Rust. For verify-only primitives with official
   vector oracles (e.g. 484 Wycheproof cases for ECDSA-P256), producing
   our own evidence is tractable — and cheaper than qualifying a
   third-party C library into a DAL A context.

The compensating control is Policy 1: our primitives are exercised by
the same public corpora the validated libraries are tested against.

## Corpus inventory

Audited 2026-07. Every primitive below except the four flagged gaps
replays its defining official corpus; the `*_test_vectors.rs` files are
`#[cfg(test)]` fixtures split out only to silence CodeQL's
hard-coded-crypto-value heuristic — they hold the same official vectors.

| Primitive | Source | Corpus | ~Vectors | Official? |
|---|---|---|---|---|
| SHA-256 / SHA-384 | `src/sha2.rs` | FIPS 180-4 §A KATs | 4 each | ✅ FIPS |
| SHA-1 | `src/sha1.rs` | FIPS 180-4 §A / RFC 3174 §7.3 | 4 | ✅ FIPS/RFC |
| SHA-3-256 | `src/crypto/sha3.rs` | FIPS 202 KATs | 2 | ✅ FIPS |
| SHAKE256 | `src/crypto/sha3.rs` | determinism self-tests only | 0 | ⚠️ **gap** |
| HMAC-SHA1 | `src/hmac_sha1.rs` | RFC 2202 §3 | 7 | ✅ RFC |
| HMAC-SHA256/384 | `src/hmac_sha2.rs` | RFC 4231 | 5 | ✅ RFC |
| Blake2b | `src/crypto/blake2b.rs` | RFC 7693 App. A | 1–2 | ✅ RFC |
| Argon2id | `src/argon2id.rs` | RFC 9106 §5.3 | 1 | ✅ RFC |
| ChaCha20 | `src/crypto/chacha20.rs` | RFC 8439 §2.3–2.4 | 2 | ✅ RFC |
| Poly1305 | `src/crypto/poly1305.rs` | RFC 8439 §2.5.2 | 1 | ✅ RFC |
| ChaCha20-Poly1305 | `src/crypto/chacha20_poly1305.rs` | RFC 8439 §2.8.2 | 1 | ✅ RFC |
| AES-256-GCM | `src/crypto/aes_gcm.rs` | FIPS 197 §C.3 + SP 800-38D | 3 | ✅ FIPS/NIST |
| Ed25519 | `src/crypto/ed25519.rs` | roundtrip + self-consistency only | 0 | ⚠️ **gap** |
| X25519 | `src/crypto/x25519.rs` | RFC 7748 | 2 | ✅ RFC |
| ML-KEM-768 | `src/crypto/ml_kem.rs` | roundtrip + algebraic checks only | 0 | ⚠️ **gap** |
| ML-DSA-65 | `src/crypto/ml_dsa.rs` | roundtrip + encode/decode only | 0 | ⚠️ **gap** |
| ECDSA-P256 | `src/crypto/ecdsa_p256.rs` | Wycheproof C2SP (113 key groups) | 484 | ✅ Wycheproof |

### Known gaps and remediation

Four primitives predate this policy and have no official known-answer
vectors pinned — only roundtrip/self-consistency tests, which catch
implementation regressions but not interop divergence. Each has a
well-defined upstream corpus and is closable independently:

- **SHAKE256** — pin NIST CAVSP XOF output vectors (SHA-3-256 already
  replays FIPS 202 KATs; only the XOF squeeze path is uncovered).
- **Ed25519** — pin RFC 8032 §7.1 signature KATs.
- **ML-KEM-768** — pin NIST FIPS 203 / ACVP known-answer vectors.
- **ML-DSA-65** — pin NIST FIPS 204 / ACVP known-answer vectors.

These are tracked as follow-on work; this policy applies in full to all
*new* primitives from the ECDSA-P256 change forward, and the four gaps
above must be closed rather than grandfathered. They do not block this
change (which establishes the policy and enforcement), but each should
become a small `security-*-kat-v1` change.

## If FIPS or Common Criteria becomes a hard requirement

Enumerated options, in preference order at time of writing:

1. **Contractual acceptance of corpus-tested clean-room crypto.** Many
   procurement contexts accept CAVP-style vector evidence plus process
   artifacts (this repo's coverage gates, audit trail, formal models)
   without a CMVP certificate. Try this first.
2. **Commercial wolfCrypt FIPS as a feature-gated, container-mode-only
   backend.** Confines the C dependency to the musl/std container build
   (never the bare-metal kernel), with the commercial license fee and an
   operational-environment listing for the container environment. The
   `deny.toml` ban would be relaxed via a scoped exception reviewed in
   the same PR.
3. **CMVP-validate SmallAIOS's own modules.** Strongest outcome, highest
   cost (lab engagement, multi-year timeline); only justified by a
   product commitment that demands it.

## Revisit triggers

Reopen this decision (via a new OpenSpec change referencing this
document) when any of the following occurs:

- A deployment contract requires FIPS 140-3 validated cryptography and
  option 1 above is rejected by the certifying authority.
- A certification authority rejects official-corpus vector evidence for
  a DAL A / Common Criteria target.
- A required primitive is judged too complex to clean-room safely
  (candidate example: a full TLS server stack with session tickets and
  0-RTT, or pairing-based cryptography).

## History

- 2026-07: Policy recorded; wolfSSL/wolfCrypt evaluated and declined
  (`crypto-validation-strategy-v1`). Prompted by the ECDSA-P256
  primitive decision in `security-ecdsa-p256-v1`.
