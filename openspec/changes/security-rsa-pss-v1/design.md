# Design — security-rsa-pss-v1

## Context

`tls-client` advertises `rsa_pss_rsae_sha256/384/512` but cannot verify
any RSA-signed chain link — the workspace has zero RSA support. This
change lands verify-only RSASSA-PSS as a Layer-0 primitive, the RSA
half of what `tls-tcp-client-v1` task 5.5 needs (ECDSA-P256, the other
half, landed in `security-ecdsa-p256-v1` / PR #227).

House context this builds on: `security::crypto::ecdsa_p256` set the
pattern for a verify-only asymmetric primitive (strict local DER
parsing, official-corpus replay, `EcdsaP256Error`-style typed errors,
Montgomery arithmetic). `security::sha2` provides SHA-256 and SHA-384
today; SHA-384 is already a complete SHA-512 core (128-byte blocks,
`K512`, 80 rounds) that merely truncates output — so SHA-512 is a small
addition, not a new hash. `crypto::constant_time` provides the
`constant_time_eq` the PSS comparison requires.

## Goals / Non-Goals

**Goals:**

- RSASSA-PSS **verification** (RFC 8017 §8.1.2 / §9.1.2, `sLen == hLen`)
  for the 9-entry RSA-{2048,3072,4096} × SHA-{256,384,512} matrix.
- SHA-512 added to `security::sha2` (prerequisite).
- Clean-room `#![no_std]`, no new deps; NIST CAVP corpus replay;
  fail-closed parsing that never panics.

**Non-Goals:**

- RSA signing, key generation, OAEP, and **PKCS#1 v1.5 in either
  direction** (Bleichenbacher legacy — refused, and its absence is a
  spec requirement).
- `tls-client` wiring (that is `tls-tcp-client-v1` task 5.5).
- SIMD/hardware RSA. Pure software.

## Decisions

### D1. SHA-512 via a shared SHA-512 core, not a copy of SHA-384

Refactor `sha2.rs` so a single `Sha512Core` (state + 128-byte buffer +
the `K512`/80-round `compress`) is parameterized by its IV, and
`Sha384`/`Sha512` become thin wrappers differing only in initial hash
value and output width (6 vs 8 words). `Sha384`'s public API
(`new`/`update`/`finalize`, `sha384()`) is preserved byte-for-byte.
Rejected: adding a standalone `Sha512` struct beside `Sha384` — it would
duplicate ~40 lines of identical `update`/`finalize`/`compress`, which
is redundant and trips the SonarCloud new-code duplication gate (3%).
SHA-384's existing FIPS vectors guard the refactor against regression.

### D2. Hash dispatch by a small enum, not generics or trait objects

RSA-PSS and MGF1 must run with a hash chosen at runtime from the TLS
signature scheme. Model it as `enum PssHash { Sha256, Sha384, Sha512 }`
with `hlen()` and a `digest(&[u8]) -> Vec<u8>` / incremental helper.
Rejected: generic `<H: Digest>` — the three SHA types have different
output-array sizes and no shared trait in-crate, so a runtime enum is
simpler than inventing a `Digest` trait this change would be the only
user of. MGF1 is written once against `PssHash` (spec: no per-hash
duplicated body).

### D3. `big_int`: `Vec<u64>`-backed, Montgomery exponentiation, private

A variable-width unsigned bigint (little-endian `Vec<u64>` limbs) with
add/sub/mul, Montgomery reduction, and a Montgomery-ladder `mod_exp`.
Verification inputs (signature, `n`, `e`) are **public**, so
variable-time bigint ops are acceptable and documented — but `mod_exp`
is written constant-time in the exponent so the primitive is reusable
for a future signing change (mirrors the ECDSA ladder decision). The
module is private (`mod big_int;`, not `pub`), RSA modules its only
consumers. Rejected: fixed-width limb arrays like `p256.rs` — RSA spans
2048–4096 bits and must handle three sizes, so `Vec`-backed is the fit
(the fixed-width approach was right for P-256's single modulus; the
opposite is right here).

### D4. Strict local DER for the RSA SPKI and INTEGERs

Reuse the ECDSA change's strict-DER discipline (minimal lengths,
positive INTEGERs, no trailing bytes, error-not-panic) for
`SubjectPublicKeyInfo { rsaEncryption, RSAPublicKey { n, e } }`. Refuse
`n` shorter than 2048 bits **at parse time**, before any exponentiation.
No shared parser with `tls-client` (Layer-0 cannot depend on Layer-1).

### D5. EMSA-PSS-VERIFY details

Follow RFC 8017 §9.1.2 exactly: recover `EM = s^e mod n` (reject
`s >= n` and wrong-length signatures first), check the `0xBC` trailer,
the leftmost-`8*emLen - emBits` zero bits, the `0x01` DB separator after
the zero pad, recompute `H'` over `(0x00)*8 || mHash || salt`, and
compare `H == H'` with `constant_time_eq`. Every failure is an `Err`,
never a panic.

## Risks / Trade-offs

- [Hand-rolled bigint modular exponentiation is a classic CVE surface]
  → NIST CAVP corpus across all 9 (size × hash) combinations plus
  known-bad and Bleichenbacher-style malformed signatures; bigint
  `mod_exp` pinned by `(base, exp, mod, result)` fixtures at each size;
  `constant_time_eq` on the final compare.
- [SHA-384 refactor could regress a shipping hash] → public API
  unchanged; existing SHA-384 FIPS vectors + new SHA-512 CAVP vectors
  both run; the core is exercised by both wrappers.
- [Variable-time bigint leaks timing] → verification handles no secret;
  documented, and `mod_exp` is CT in the exponent regardless.
- [~30 KB code + a new corpus] → within the <8 MB budget; corpus is
  `#[cfg(test)]`, zero production cost.

## Migration Plan

Additive. SHA-512 extends `sha2`; RSA modules are new; `big_int` is
private. One PR against `develop`. Rollback = revert; nothing depends on
the modules until `tls-tcp-client-v1` task 5.5 wires them (separate PR).
Independent of PR #227 (ECDSA) — no shared files beyond `crypto/mod.rs`
module declarations.

## Open Questions

- None blocking. A future `security-rsa-sign-v1` (signing) and the
  `security-*-kat-v1` gap-closers (from `crypto-validation-strategy-v1`)
  are tracked separately.
