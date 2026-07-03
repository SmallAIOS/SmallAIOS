# Design — security-ecdsa-p256-v1

## Context

The TLS 1.3 client (`tls-tcp-client-v1`) advertises
`ecdsa_secp256r1_sha256` in its ClientHello signature-algorithm
allow-list, but `tls-client/src/cert/verify.rs` can only verify
Ed25519-signed chain links today — every ECDSA-P256-signed chain is
refused fail-closed with `ChainUntrusted`. Since Let's Encrypt,
Cloudflare, and AWS all issue ECDSA-P256 leaves/intermediates, no
publicly-trusted endpoint can be reached until this primitive exists.

The `security` crate already contains the house patterns this change
follows: `ed25519.rs` + `field25519.rs` split a signature scheme into
scheme logic vs. field/curve arithmetic; `security::sha2` provides
`sha256(&[u8]) -> [u8; 32]`; `crypto::constant_time` holds shared CT
helpers; typed per-module error enums (`Ed25519Error`) with
`Result<(), E>` verify entry points are the convention. A generated
Wycheproof corpus (`crypto/ecdsa_p256_test_vectors.rs`, 484 cases /
113 key groups, no `acceptable` results) is already checked into the
branch and unreferenced — this change wires it in.

Layer-0 constraint: `security` may not depend on `tls-client`
(Layer 1), so the DER/SPKI parsing this module needs is implemented
locally, not borrowed from `tls-client/src/cert/x509.rs`.

## Goals / Non-Goals

**Goals:**

- ECDSA-P256/SHA-256 signature **verification** per ANSI X9.62, as a
  `#![no_std]` Layer-0 primitive with no new external dependencies.
- Strict DER `(r, s)` parser and strict SPKI (`id-ecPublicKey` +
  `secp256r1`, uncompressed point) parser with fail-closed refusal
  rules — malformed input returns an error, never panics.
- Constant-time scalar-multiplication ladder (no scalar-bit-dependent
  branches or indexing).
- Full 484-vector Wycheproof corpus replay in `just test`, failure
  output naming the offending `tc_id`; ≥95 % line coverage on the
  verify path.

**Non-Goals:**

- Signing or key generation (follow-on `security-ecdsa-sign-v1`).
- P-384 / P-521 / secp256k1 / brainpool curves.
- Low-s malleability enforcement (public CAs emit high-s signatures;
  rejecting them breaks chains — revisit in v2 only if a deployment
  needs it).
- Wiring into `tls-client` — that is `tls-tcp-client-v1` task 5.5,
  kept out of this change so the spec deltas stay 1:1 with code
  (this change's only delta is `security-ecdsa-p256-verify`).

## Decisions

### D1. Module split mirrors ed25519/field25519

`security/src/crypto/p256.rs` holds field arithmetic, curve constants,
Jacobian point ops, and the CT ladder. `security/src/crypto/ecdsa_p256.rs`
holds DER `(r, s)` parsing, SPKI parsing, and `ecdsa_p256_verify`.
Alternative — one module — rejected: the 25519 precedent shows the
field/scheme split keeps the arithmetic reviewable in isolation, and
p256.rs is the piece a future `security-ecdsa-sign-v1` would reuse.

### D2. Two fixed-width 4×u64 field implementations (mod p and mod n)

ECDSA needs arithmetic in two moduli: the field prime `p` (point ops)
and the group order `n` (scalar inversion, `u1`/`u2`). Both are
implemented as 4×u64 little-endian limb arrays with Montgomery
multiplication and a shared macro/impl generating the two instances.

- Rejected: generic big-int — overkill for two fixed 256-bit moduli
  and harder to keep constant-time (that generality is
  `security-rsa-pss-v1`'s problem, which needs 2048–4096-bit values).
- Rejected: NIST fast reduction exploiting P-256's Solinas shape —
  faster, but a second reduction algorithm to review and it doesn't
  apply to `n` anyway. Montgomery everywhere is one algorithm, used
  twice. Handshake-time verification is not hot-path.

Inversions (`s⁻¹ mod n`, `Z⁻¹ mod p`) use Fermat exponentiation with a
fixed square-and-multiply schedule over the constant exponent —
constant-time by construction, no gcd branching.

### D3. Jacobian coordinates; explicit incomplete-case handling

Point double/add operate on Jacobian `(X, Y, Z)` with infinity encoded
as `Z = 0`. The standard Jacobian formulas are incomplete (add degrades
on P == Q and P == ±∞), so `point_add` detects the doubling and
infinity cases and selects the correct result via `constant_time`
selects rather than branches. Affine conversion happens exactly once,
when extracting `R.x` at the end of verify.

### D4. Two independent CT ladders, no Shamir's trick

`R = u1·G + u2·Q` is computed as two separate constant-time
double-and-add-always ladders followed by one point addition.
Rejected: Shamir/interleaved multi-scalar or wNAF — meaningful only
for throughput, and their table lookups/recodings are where CT bugs
live. Verification inputs are public, so CT is not strictly required
here — but the spec mandates a CT ladder so the routine is safe to
reuse for signing later, and double-and-add-always is the simplest
auditable shape.

### D5. Strict local DER parsing, no shared parser

`ecdsa_p256.rs` implements a ~50-line strict DER reader for exactly two
shapes: `SEQUENCE { INTEGER r, INTEGER s }` and the SPKI envelope.
Refusal rules per the spec: non-minimal lengths, negative INTEGERs,
zero / ≥ n scalars, redundant leading zeros, wrong OIDs, compressed or
non-65-byte points, trailing garbage. Rejected: reusing `tls-client`'s
x509 parser (layering violation) or the protobuf-style incremental
reader in `onnx-rt` (wrong grammar family). BER laxness is
deliberately not supported; Wycheproof's malformed-DER vectors are the
acceptance test for this decision.

### D6. Public API mirrors ed25519 conventions

```rust
pub struct EcdsaP256PublicKey { /* affine point, validated on-curve */ }
impl EcdsaP256PublicKey {
    pub fn from_spki_der(der: &[u8]) -> Result<Self, EcdsaP256Error>;
    pub fn from_uncompressed(bytes: &[u8; 65]) -> Result<Self, EcdsaP256Error>;
}
pub fn ecdsa_p256_verify(
    pk: &EcdsaP256PublicKey,
    message: &[u8],          // hashed internally with security::sha2::sha256
    der_signature: &[u8],
) -> Result<(), EcdsaP256Error>;
```

`from_uncompressed` exists because `tls-client`'s x509 parser already
extracts the raw point from certificates; `from_spki_der` exists
because the Wycheproof corpus keys (`KEYS_DER`) and non-TLS consumers
(COSE, Rekor) present SPKI. Both constructors validate the point
against the curve equation before it can reach the ladder. Errors are
a dedicated `EcdsaP256Error` enum (parse/range/off-curve/bad-signature
variants), matching `Ed25519Error` granularity.

### D7. Corpus wiring

`mod.rs` gains `pub mod ecdsa_p256;`, `pub mod p256;`, and
`#[cfg(test)] mod ecdsa_p256_test_vectors;`. The replay test iterates
every `WpCase`, decodes `msg`/`sig` hex, resolves `KEYS_DER[case.key]`
via `from_spki_der`, and asserts `verify().is_ok() == case.valid`,
panicking with the `tc_id` on mismatch. Vectors where the *key* itself
is invalid must fail in `from_spki_der` — the test accepts an error at
either stage as "rejected". Supplementary unit tests pin the arithmetic
identities from the spec: `n·G = ∞`, `(n-1)·G = -G`, plus field-op
round-trips.

## Risks / Trade-offs

- [Hand-rolled field arithmetic is the classic source of crypto CVEs]
  → 484 Wycheproof vectors specifically target carry bugs, edge-case
  r/s, and off-curve/infinity faults; arithmetic identity tests pin
  the group law; Montgomery-only keeps one multiplication algorithm
  under review. Fuzzing of the DER surface rides `tls-tcp-client-v1`
  Phase 10's fuzz targets.
- [Constant-time claims are asserted, not measured] → CT here is
  defense-in-depth (verify inputs are public). Code review enforces
  the structural rules (no secret-dependent branch/index); dudect-style
  measurement is explicitly out of scope until a signing change makes
  timing an actual attack surface.
- [Montgomery ladder is ~2-3× slower than optimized wNAF] → accepted;
  one verification per handshake, and `bench/` can quantify later.
- [Strict DER may reject exotic-but-real signatures] → CA-issued
  certificates are DER by profile (RFC 5280); Wycheproof's `valid` set
  is the canary — if all 484 pass, real chains parse.
- [~25 KB code-size increase] → within the <8 MB base budget; checked
  by the existing image-size CI job.

## Migration Plan

Pure addition — no existing API changes, no data migration. Lands as
one PR against `develop` (branch `change/security-ecdsa-p256-v1`).
Rollback = revert the PR; nothing depends on the module until
`tls-tcp-client-v1` task 5.5 wires it, in a separate PR. Note: this
branch is currently based on `feature/openspec-strict-validation`
(PR #226, carries this change's spec); rebase onto `develop` once #226
merges so the PR diff is implementation-only.

## Open Questions

- None blocking. Deferred by scope: prehashed-message variant
  (`verify_prehashed`) if a consumer ever needs to supply its own
  digest — TLS CertificateVerify and X.509 both hash the full message,
  so v1 hashes internally.
