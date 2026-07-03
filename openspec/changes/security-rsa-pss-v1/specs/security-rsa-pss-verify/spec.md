## ADDED Requirements

### Requirement: RSA-PSS Verify Module Is a Verify-Only Layer-0 Primitive

The `security` crate SHALL provide a `security::crypto::rsa_pss` module implementing RSASSA-PSS signature **verification** matching the TLS 1.3 `rsa_pss_rsae_sha256`, `rsa_pss_rsae_sha384`, and `rsa_pss_rsae_sha512` signature schemes. The module SHALL be `#![no_std]`, SHALL live at Layer 0 (`security/src/crypto/big_int.rs` for big-integer arithmetic, `security/src/crypto/mgf1.rs` for mask generation, `security/src/crypto/rsa_pss.rs` for EMSA-PSS-VERIFY plus the verify entry point), and SHALL be verify-only in v1: no key generation, no signing, no RSA-OAEP encryption, and no RSA-PKCS#1 v1.5 support in either direction. Hashing SHALL come from the existing `security/crypto/sha2` FIPS 180-4 family; the change SHALL add no new external production dependencies.

#### Scenario: Public API is verify-only

- **WHEN** a reviewer reads the public API of `security::crypto::rsa_pss`
- **THEN** a signature-verification entry point SHALL be exposed
- **AND** no signing function SHALL be exposed
- **AND** no key-generation function SHALL be exposed
- **AND** no RSA-OAEP encryption or decryption function SHALL be exposed

#### Scenario: RSA-PKCS#1 v1.5 is refused entirely

- **WHEN** a reviewer searches the `security` crate for PKCS#1 v1.5 signature support
- **THEN** no EMSA-PKCS1-v1_5 encoding, signing, or verification path SHALL exist
- **AND** RSASSA-PSS verification SHALL be the only RSA signature scheme implemented

#### Scenario: Pure-software implementation with no new external dependencies

- **WHEN** the `security` crate's `Cargo.toml` is diffed against its pre-change state
- **THEN** no new external production dependency SHALL appear
- **AND** message hashing and MGF1 digests SHALL call the `security/crypto/sha2` FIPS 180-4 implementations (SHA-256, SHA-384, SHA-512), not a new crate
- **AND** no hardware-accelerated RSA path (AVX/NEON intrinsics or vendor RSA cores) SHALL be used
- **AND** the crate SHALL continue to build for `#![no_std]` bare-metal targets (`x86_64-unknown-none`, `aarch64-unknown-none`, `riscv64gc-unknown-none-elf`)

### Requirement: Key-Size and Hash-Function Support Matrix

`rsa_pss::verify` SHALL support RSA-2048, RSA-3072, and RSA-4096 moduli, each combinable with SHA-256, SHA-384, and SHA-512 — the full 9-entry (key-size × hash) matrix behind the `rsa_pss_rsae_sha256`, `rsa_pss_rsae_sha384`, and `rsa_pss_rsae_sha512` codes on the `tls-client-handshake` allow-list. Moduli shorter than 2048 bits SHALL be refused with an error per NIST SP 800-131A, and SHA-1 SHALL NOT be available as either the message hash or the MGF1 hash.

#### Scenario: All nine key-size × hash combinations verify

- **WHEN** `verify` is called with a known-good NIST CAVP vector at each of the nine (RSA-2048/3072/4096 × SHA-256/384/512) combinations
- **THEN** every combination SHALL verify successfully

#### Scenario: RSA-1024 keys are refused

- **WHEN** `verify` (or the SubjectPublicKeyInfo parser) is handed an RSA public key with a 1024-bit modulus
- **THEN** the operation SHALL fail with an error before any modular exponentiation is performed
- **AND** the refusal SHALL apply to every modulus shorter than 2048 bits

#### Scenario: SHA-1 is not offered

- **WHEN** a reviewer reads the hash-parameterization surface of `rsa_pss` and `mgf1`
- **THEN** only SHA-256, SHA-384, and SHA-512 SHALL be accepted as hash parameters
- **AND** no SHA-1 variant SHALL exist

### Requirement: EMSA-PSS-VERIFY Conforms to RFC 8017 § 9.1.2

`rsa_pss` SHALL implement EMSA-PSS-VERIFY per RFC 8017 § 9.1.2 with salt length equal to the hash length (`sLen == hLen`, the standard configuration TLS 1.3 servers emit). Verification SHALL fail with an error — never a panic — when the encoded message's trailer byte, leftmost-bits constraint, DB padding structure, or recomputed-digest comparison does not check out. The final `H == H'` digest comparison SHALL be constant-time using the workspace's existing `constant_time_eq` convention.

#### Scenario: Standard TLS-emitted PSS encoding verifies

- **WHEN** `verify` is called with a known-good CAVP vector whose salt length equals the hash output length
- **THEN** verification SHALL succeed

#### Scenario: Tampered message is rejected

- **WHEN** `verify` is called with a signature that is valid for message `m1` but presented with a different message `m2`
- **THEN** the recomputed `H'` SHALL NOT match `H`
- **AND** verification SHALL fail with an error without panicking

#### Scenario: Structurally invalid encoded message is rejected

- **WHEN** the recovered encoded message carries a wrong trailer byte (not `0xBC`), nonzero bits beyond `emBits`, or malformed DB padding (as exercised by the known-bad CAVP vectors)
- **THEN** verification SHALL fail with an error for every case
- **AND** no case SHALL panic

#### Scenario: Digest comparison is constant-time

- **WHEN** a reviewer reads the final `H == H'` comparison in EMSA-PSS-VERIFY
- **THEN** it SHALL use `constant_time_eq`
- **AND** it SHALL NOT early-exit on the first mismatching byte

### Requirement: MGF1 Mask Generation Parameterized by Hash

`security/src/crypto/mgf1.rs` SHALL implement the MGF1 mask generation function per RFC 8017 Appendix B.2.1, parameterized by the underlying hash function so that one implementation serves SHA-256, SHA-384, and SHA-512. Within a single PSS verification, MGF1 SHALL use the same hash function as the message digest.

#### Scenario: MGF1 is generic over the hash

- **WHEN** a reviewer reads `security/src/crypto/mgf1.rs`
- **THEN** the mask generation SHALL be written once, parameterized by the hash function
- **AND** no per-hash duplicated MGF1 body SHALL exist

#### Scenario: MGF1 hash matches the scheme hash

- **WHEN** `verify` runs for the `rsa_pss_rsae_sha384` scheme
- **THEN** MGF1 SHALL be instantiated with SHA-384
- **AND** the message digest SHALL also be SHA-384

#### Scenario: Multi-block masks follow the RFC 8017 counter construction

- **WHEN** MGF1 is asked for a mask longer than one hash output
- **THEN** the mask SHALL be the concatenation of `Hash(seed || C)` blocks where `C` is a 4-byte big-endian counter starting at 0
- **AND** the result SHALL be truncated to exactly the requested length

### Requirement: Constant-Time Modular Exponentiation

The RSA verify operation `s^e mod n` SHALL be implemented with clean-room big-integer arithmetic using Montgomery reduction and a constant-time Montgomery-ladder exponentiation whose sequence of operations does not depend on exponent bit values. Variable-time big-integer operations on non-secret inputs (signature, modulus, public exponent) are acceptable for verification, and this timing posture SHALL be documented in the module.

#### Scenario: Exponentiation ladder is constant-time

- **WHEN** a reviewer reads the modular-exponentiation routine
- **THEN** it SHALL use a Montgomery ladder
- **AND** it SHALL contain no exponent-bit-dependent branches or exponent-bit-dependent memory indexing

#### Scenario: Exponentiation is arithmetically correct

- **WHEN** the big-integer test suite exercises modular exponentiation with known `(base, exponent, modulus, result)` fixtures spanning 2048-, 3072-, and 4096-bit moduli
- **THEN** every fixture SHALL produce the expected result
- **AND** the encoded message recovered as `s^e mod n` SHALL match the EMSA-PSS encoding for every valid CAVP vector

#### Scenario: Timing posture is documented

- **WHEN** a reviewer reads the module documentation of `rsa_pss` and `big_int`
- **THEN** it SHALL state that verification handles no secret, so variable-time bigint operations on non-secret inputs are acceptable
- **AND** it SHALL state that the exponentiation is nevertheless written constant-time to keep the primitive reusable for a future signing change

### Requirement: Internal big_int Module Stays Private

`security/src/crypto/big_int.rs` SHALL provide the big-integer primitives RSA needs — a variable-width unsigned big integer backed by `Vec<u64>`, addition, subtraction, multiplication, Montgomery reduction, Montgomery-ladder modular exponentiation, and DER INTEGER parsing into a big integer — and SHALL be a private module, not re-exported from the `security` crate's public API.

#### Scenario: big_int is not part of the public surface

- **WHEN** a reviewer inspects the `security` crate's public API (crate-root re-exports or `cargo doc` output)
- **THEN** no `big_int` type or function SHALL be publicly visible
- **AND** the RSA modules SHALL be its only consumers

#### Scenario: Required operations are present

- **WHEN** a reviewer reads `security/src/crypto/big_int.rs`
- **THEN** it SHALL contain a `Vec<u64>`-backed unsigned big integer with add, subtract, and multiply operations
- **AND** Montgomery reduction and Montgomery-ladder modular exponentiation
- **AND** a DER INTEGER parser producing a big integer

#### Scenario: DER INTEGER parsing rejects malformed input

- **WHEN** the DER INTEGER parser is fed truncated bytes, a wrong tag, a negative INTEGER (high bit set with no zero pad), or a non-minimal length form
- **THEN** parsing SHALL fail with an error
- **AND** no case SHALL panic

### Requirement: SubjectPublicKeyInfo RSA Public-Key Parser

The module SHALL provide a DER parser that extracts the modulus `n` and public exponent `e` from a `SubjectPublicKeyInfo` carrying an RSA public key (algorithm `rsaEncryption`, subjectPublicKey BIT STRING wrapping `RSAPublicKey ::= SEQUENCE { INTEGER n, INTEGER e }`) — the form X.509 certificates carry and the form the `rsa_pss_rsae_*` schemes name. Inputs not matching this shape SHALL be rejected with an error, and moduli shorter than 2048 bits SHALL be refused at parse time.

#### Scenario: X.509-emitted SPKI parses to (n, e)

- **WHEN** the parser is fed a SubjectPublicKeyInfo from a public-CA-style RSA certificate (`rsaEncryption` algorithm OID, 2048-bit or larger modulus)
- **THEN** parsing SHALL succeed
- **AND** the extracted `(n, e)` SHALL be usable by `rsa_pss::verify`

#### Scenario: Non-RSA SPKI is rejected

- **WHEN** the parser is fed an SPKI whose algorithm OID is not `rsaEncryption` (e.g., `id-ecPublicKey`)
- **THEN** parsing SHALL fail with an error
- **AND** no key material SHALL be returned

#### Scenario: Malformed SPKI never panics

- **WHEN** the parser is fed truncated bytes, wrong tags, or trailing garbage in place of a well-formed SubjectPublicKeyInfo
- **THEN** every case SHALL return an error
- **AND** no case SHALL panic or read out of bounds

### Requirement: NIST CAVP RSA-PSS Verify Corpus Replay

The test suite SHALL replay NIST CAVP RSA-PSS signature-verification vectors against `rsa_pss::verify`: at least 30 vectors in total covering all nine (key-size × hash) combinations, plus known-bad signatures and Bleichenbacher-style malformed signatures that MUST reject. Every known-good vector SHALL verify; every known-bad or malformed vector SHALL be rejected with an error, never a panic.

#### Scenario: Corpus runs in the workspace test suite

- **WHEN** `just test` runs the `security` crate's tests
- **THEN** the CAVP RSA-PSS corpus SHALL execute against `rsa_pss::verify`
- **AND** the corpus SHALL contain at least 30 vectors spanning all nine (RSA-2048/3072/4096 × SHA-256/384/512) combinations
- **AND** no vector SHALL be skipped

#### Scenario: Good vectors accept and bad vectors reject

- **WHEN** a corpus vector marked valid is replayed
- **THEN** verification SHALL succeed
- **AND** for every known-bad vector, verification SHALL fail with an error rather than a panic

#### Scenario: Bleichenbacher-style malformed signatures never panic

- **WHEN** `verify` is fed Bleichenbacher-style malformed signatures (signature value greater than or equal to `n`, zero-length signature, wrong-length signature, garbage bytes)
- **THEN** every case SHALL return an error
- **AND** no case SHALL panic

#### Scenario: Corpus failure identifies the offending vector

- **WHEN** any corpus case produces the wrong accept/reject outcome
- **THEN** the test failure output SHALL identify the offending vector by its CAVP identifier or corpus index
