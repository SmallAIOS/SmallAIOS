# Tasks — security-ecdsa-p256-v1

## 1. Field Arithmetic (`security/src/crypto/p256.rs`)

- [x] 1.1 Define 4×u64 little-endian limb type and the P-256 constants: field prime `p`, group order `n`, curve coefficient `b`, generator `G` affine coordinates, Montgomery parameters (R², n′) for both moduli
- [x] 1.2 Implement constant-time limb primitives shared by both moduli: add-with-carry, sub-with-borrow, conditional select/swap (reusing `crypto::constant_time` helpers where they fit)
- [x] 1.3 Implement Montgomery multiplication/squaring and to/from-Montgomery conversion, instantiated for mod-`p` and mod-`n` via a shared macro or generic-over-modulus impl
- [x] 1.4 Implement modular add/sub/negate with constant-time final reduction for both moduli
- [x] 1.5 Implement Fermat inversion (fixed square-and-multiply schedule over the constant exponent) for mod-`p` and mod-`n`
- [x] 1.6 Unit-test field ops: round-trips through Montgomery form, `x * x⁻¹ == 1` for fixed and boundary values (1, p-1 / n-1), add/sub/mul against precomputed known-answer values

## 2. Curve and Point Operations (`security/src/crypto/p256.rs`)

- [x] 2.1 Define Jacobian point type with infinity encoded as `Z = 0`, plus affine⇄Jacobian conversion (affine extraction via mod-p inversion)
- [x] 2.2 Implement on-curve check `y² == x³ - 3x + b (mod p)` for affine points
- [x] 2.3 Implement Jacobian point doubling
- [x] 2.4 Implement Jacobian point addition with constant-time handling of the incomplete cases (P == Q → double, P or Q == ∞ → select other operand)
- [x] 2.5 Implement constant-time double-and-add-always scalar multiplication (no scalar-bit-dependent branches or memory indexing)
- [x] 2.6 Unit-test the group law: `n·G == ∞`, `(n-1)·G == -G` (same x, negated y), `2·G == G + G`, `G + ∞ == G`, small-multiple cross-check (`k·G` via ladder equals repeated addition for k ≤ 8)

## 3. DER and SPKI Parsing (`security/src/crypto/ecdsa_p256.rs`)

- [x] 3.1 Define `EcdsaP256Error` enum (malformed-DER, out-of-range scalar, wrong OID/format, off-curve point, bad signature variants) following the `Ed25519Error` convention
- [x] 3.2 Implement the strict DER `(r, s)` parser: exact `SEQUENCE { INTEGER r, INTEGER s }`, rejecting zero or ≥ n scalars, negative INTEGERs, non-minimal lengths, redundant leading zeros, wrong tags, truncation, and trailing garbage — error returns only, no panics, no curve arithmetic on the rejection path
- [x] 3.3 Implement the strict SPKI parser: `id-ecPublicKey` + `secp256r1` OIDs, BIT STRING carrying exactly `0x04 || X || Y` (65 bytes); reject compressed forms and any other shape
- [x] 3.4 Implement `EcdsaP256PublicKey::{from_spki_der, from_uncompressed}` with on-curve validation before the point is usable
- [x] 3.5 Unit-test parser refusal rules with hand-built negative cases for each rule (ahead of the full corpus replay)

## 4. Verification (`security/src/crypto/ecdsa_p256.rs`)

- [x] 4.1 Implement `ecdsa_p256_verify(pk, message, der_signature) -> Result<(), EcdsaP256Error>` per ANSI X9.62: SHA-256 via `security::sha2::sha256`, `u1 = H(m)·s⁻¹ mod n`, `u2 = r·s⁻¹ mod n`, `R = u1·G + u2·Q`, reject `R == ∞`, accept iff `R.x mod n == r`
- [x] 4.2 Unit-test the accept/reject seams: known-good vector verifies; same signature with a different message fails; high-s (`n/2 < s < n`) counterpart of a valid signature still verifies (no low-s rule)

## 5. Wycheproof Corpus Replay

- [x] 5.1 Wire the modules into `security/src/crypto/mod.rs`: `pub mod p256;`, `pub mod ecdsa_p256;`, `#[cfg(test)] mod ecdsa_p256_test_vectors;`, and update the module-header doc list
- [x] 5.2 Implement the corpus replay test: hex-decode every `WpCase`, resolve `KEYS_DER[case.key]` via `from_spki_der`, assert `verify().is_ok() == case.valid` (key-parse failure counts as rejection), failure message includes the `tc_id`
- [x] 5.3 All 484 vectors pass; assert the executed-case count equals the corpus length so no vector is silently skipped

## 6. Quality Gates

- [x] 6.1 `just fmt-check` and `just clippy` clean on the pinned nightly
- [x] 6.2 `cargo test -p smallaios-security` green; full `just test` green
- [x] 6.3 `#![no_std]` bare-metal builds stay green: `just build-kernel-x86`, `just build-kernel-arm` (and RISC-V target check) with the new modules compiled in
- [x] 6.4 Line coverage ≥95 % on `p256.rs` + `ecdsa_p256.rs` (`cargo llvm-cov -p smallaios-security`); no new external dependencies in `security/Cargo.toml` (diff is empty)

## 7. Land

- [x] 7.1 `openspec validate security-ecdsa-p256-v1 --type change --strict` passes
- [x] 7.2 Rebase `change/security-ecdsa-p256-v1` onto `develop` once PR #226 (spec baseline) merges, so the PR diff is implementation-only (done before #227 opened)
- [x] 7.3 PR against `develop` titled `feat(security): ECDSA-P256 signature verification (security-ecdsa-p256-v1)`, noting the tls-client wiring is deferred to `tls-tcp-client-v1` task 5.5 (landed as #227, merged 2026-07-03; tls wiring landed as #231)
