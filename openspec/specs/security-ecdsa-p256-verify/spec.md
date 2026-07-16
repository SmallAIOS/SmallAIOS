# security-ecdsa-p256-verify Specification

## Purpose

Verify-only ECDSA-P256 (ecdsa_secp256r1_sha256) signature verification primitive in the security crate — constant-time P-256 arithmetic, strict DER/SPKI parsing, Wycheproof corpus replay — serving TLS 1.3 certificate-chain and CertificateVerify validation.

## Requirements

### Requirement: ECDSA-P256 Verify Module Is a Verify-Only Layer-0 Primitive

The `security` crate SHALL provide a `security::crypto::ecdsa_p256` module implementing ECDSA signature **verification** on NIST P-256 (secp256r1) paired with SHA-256, matching the TLS 1.3 `ecdsa_secp256r1_sha256` signature scheme. The module SHALL be `#![no_std]`, SHALL live at Layer 0 (`security/src/crypto/p256.rs` for curve/field/point operations, `security/src/crypto/ecdsa_p256.rs` for DER parse + verify), and SHALL be verify-only in v1: no key generation and no signing. SHA-256 SHALL be provided by the existing `security::sha2` (`security/src/sha2.rs`) module; the change SHALL add no new external production dependencies.

#### Scenario: Public API is verify-only

- **WHEN** a reviewer reads the public API of `security::crypto::ecdsa_p256`
- **THEN** a signature-verification entry point SHALL be exposed
- **AND** no signing function SHALL be exposed
- **AND** no key-generation function SHALL be exposed

#### Scenario: P-256 is the only supported curve

- **WHEN** a reviewer reads the public API of `security::crypto::ecdsa_p256` and `security::crypto::p256`
- **THEN** only NIST P-256 (secp256r1) curve parameters SHALL be present
- **AND** no P-384, P-521, brainpool, or secp256k1 support SHALL exist

#### Scenario: No new external dependencies

- **WHEN** the `security` crate's `Cargo.toml` is diffed against its pre-change state
- **THEN** no new external production dependency SHALL appear
- **AND** message hashing inside `ecdsa_p256` SHALL call the existing `security::sha2` (`security/src/sha2.rs`) SHA-256 implementation
- **AND** the crate SHALL continue to build for `#![no_std]` bare-metal targets (`x86_64-unknown-none`, `aarch64-unknown-none`, `riscv64gc-unknown-none-elf`)

### Requirement: ECDSA Verification Follows ANSI X9.62

`ecdsa_p256::verify` SHALL implement classical ECDSA verification per ANSI X9.62 / RFC 6979: given a message, a DER-encoded signature, and a P-256 public key point `Q`, it SHALL (1) parse `(r, s)` from the DER form, (2) reject `r` or `s` outside `[1, n-1]`, (3) compute `u1 = H(m) * s^-1 mod n` and `u2 = r * s^-1 mod n` where `H` is SHA-256, (4) compute `R = u1*G + u2*Q`, (5) reject the signature if `R` is the point at infinity, and (6) accept if and only if `R.x mod n == r`.

#### Scenario: Known-good signature verifies

- **WHEN** `verify` is called with a Wycheproof `valid` vector's message, signature, and public key
- **THEN** verification SHALL succeed

#### Scenario: Signature over a different message is rejected

- **WHEN** `verify` is called with a signature that is valid for message `m1` but presented with a different message `m2`
- **THEN** `R.x mod n` SHALL NOT equal `r`
- **AND** verification SHALL fail without panicking

#### Scenario: Point-at-infinity result is rejected

- **WHEN** the computed `R = u1*G + u2*Q` is the point at infinity (as exercised by the Wycheproof fault-injection vectors)
- **THEN** verification SHALL fail
- **AND** the implementation SHALL NOT attempt to read an x-coordinate from the infinity point

#### Scenario: High-s signatures are accepted (no low-s rule in v1)

- **WHEN** `verify` is called with an otherwise-valid signature whose `s` component satisfies `n/2 < s < n`
- **THEN** verification SHALL succeed
- **AND** no low-s malleability check SHALL reject it, because public CAs emit high-s signatures and rejecting them would break TLS certificate chains

### Requirement: DER Signature Parser Refusal Rules

The `(r, s)` parser SHALL accept only a well-formed DER ASN.1 `SEQUENCE { INTEGER r, INTEGER s }`. It SHALL reject: zero `r` or `s`; `r` or `s` greater than or equal to the group order `n`; negative INTEGERs (leading content byte with the high bit set and no zero pad); oversized or non-minimal encodings; and structurally malformed input (truncated bytes, wrong tags, length mismatches, trailing garbage). Rejection SHALL be an error return, never a panic.

#### Scenario: Zero r or s is rejected

- **WHEN** the parser is fed a DER signature whose `r` INTEGER or `s` INTEGER decodes to zero
- **THEN** verification SHALL fail with an error
- **AND** no curve arithmetic SHALL be performed

#### Scenario: r or s at or above the group order is rejected

- **WHEN** the parser is fed a DER signature with `r == n`, `s == n`, or larger values
- **THEN** verification SHALL fail with an error before any scalar inversion is attempted

#### Scenario: Negative INTEGER encoding is rejected

- **WHEN** the parser is fed a DER INTEGER whose first content byte has the high bit set (a negative value under DER rules)
- **THEN** verification SHALL fail with an error
- **AND** the value SHALL NOT be reinterpreted as a large positive scalar

#### Scenario: Oversized or non-minimal encodings are rejected

- **WHEN** the parser is fed a signature whose INTEGER carries redundant leading zero bytes, uses a non-minimal length form, or whose SEQUENCE length exceeds what two in-range P-256 scalars can occupy
- **THEN** verification SHALL fail with an error

#### Scenario: Malformed DER never panics

- **WHEN** the parser is fed arbitrary malformed bytes (truncated SEQUENCE, wrong tag, trailing garbage after `s`) drawn from the Wycheproof malformed-DER vectors
- **THEN** every case SHALL return an error
- **AND** no case SHALL panic or read out of bounds

### Requirement: SubjectPublicKeyInfo Public-Key Parser

The module SHALL provide a DER parser that extracts a P-256 public key point from a `SubjectPublicKeyInfo` whose algorithm is `id-ecPublicKey` with `secp256r1` (prime256v1) as the named-curve parameter and whose subjectPublicKey BIT STRING carries the point in uncompressed form (`0x04 || X || Y`, 65 bytes) — the standard TLS-emitted form. Keys not matching this shape SHALL be rejected with an error, and the decoded point SHALL be validated against the P-256 curve equation before use.

#### Scenario: TLS-emitted SPKI parses to a usable public key

- **WHEN** the parser is fed one of the `KEYS_DER` SubjectPublicKeyInfo blobs from the Wycheproof corpus (`id-ecPublicKey` + `secp256r1`, uncompressed point)
- **THEN** parsing SHALL succeed
- **AND** the extracted point SHALL be usable as `Q` in `ecdsa_p256::verify`

#### Scenario: Wrong algorithm or curve OID is rejected

- **WHEN** the parser is fed an SPKI whose algorithm OID is not `id-ecPublicKey` or whose parameter OID is not `secp256r1`
- **THEN** parsing SHALL fail with an error
- **AND** no point SHALL be returned

#### Scenario: Non-uncompressed point form is rejected

- **WHEN** the parser is fed an SPKI whose BIT STRING starts with a compressed-form prefix (`0x02` or `0x03`) or has a length other than 65 bytes
- **THEN** parsing SHALL fail with an error

#### Scenario: Off-curve point is rejected

- **WHEN** the parser is fed an SPKI whose `(X, Y)` coordinates do not satisfy the P-256 curve equation (a weak-key / fault-injection case from the Wycheproof corpus)
- **THEN** parsing or verification SHALL fail with an error
- **AND** the point SHALL NOT participate in scalar multiplication

### Requirement: Jacobian Point Arithmetic with Constant-Time Scalar Multiplication

`security/src/crypto/p256.rs` SHALL implement P-256 point doubling and point addition in Jacobian coordinates, and SHALL provide scalar multiplication via a constant-time ladder whose sequence of field operations does not depend on the scalar's bit values.

#### Scenario: Point arithmetic uses Jacobian coordinates

- **WHEN** a reviewer reads `security/src/crypto/p256.rs`
- **THEN** point doubling and point addition SHALL operate on Jacobian `(X, Y, Z)` representations
- **AND** conversion to affine coordinates SHALL occur only where an affine result is required (e.g., extracting `R.x`)

#### Scenario: Scalar multiplication ladder is constant-time

- **WHEN** a reviewer reads the scalar-multiplication routine
- **THEN** it SHALL use a constant-time ladder
- **AND** it SHALL contain no scalar-bit-dependent branches or scalar-bit-dependent memory indexing

#### Scenario: Scalar multiplication is arithmetically correct

- **WHEN** the generator `G` is multiplied by the group order `n`
- **THEN** the result SHALL be the point at infinity
- **AND** multiplying `G` by `n - 1` SHALL yield the negation of `G` (same `x`, negated `y`)

### Requirement: Wycheproof Corpus Replay

The test suite SHALL replay the Wycheproof `ecdsa_secp256r1_sha256_test.json` corpus (C2SP testvectors_v1) against `ecdsa_p256::verify`, using the generated `security/src/crypto/ecdsa_p256_test_vectors.rs` module (484 cases across 113 key groups — satisfying the proposal's ≥150-vector floor, covering known-good signatures, malformed DER, edge-case `r`/`s` values, and the standard fault-injection scenarios). Every vector flagged `valid` SHALL verify; every vector flagged `invalid` SHALL be rejected; the generated corpus contains no `acceptable` results.

#### Scenario: Full corpus replays in the workspace test run

- **WHEN** `just test` runs the `security` crate's test suite
- **THEN** every `WpCase` in `ecdsa_p256_test_vectors.rs` SHALL be executed against `ecdsa_p256::verify`
- **AND** no vector SHALL be skipped

#### Scenario: Valid vectors accept and invalid vectors reject

- **WHEN** a corpus case with `valid == true` is replayed
- **THEN** verification SHALL succeed
- **AND** for every case with `valid == false`, verification SHALL fail with an error rather than a panic

#### Scenario: Corpus failure identifies the offending vector

- **WHEN** any corpus case produces the wrong accept/reject outcome
- **THEN** the test failure output SHALL include the Wycheproof `tc_id` of the offending case
