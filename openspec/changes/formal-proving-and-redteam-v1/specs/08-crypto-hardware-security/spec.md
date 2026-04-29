## ADDED Requirements

### Requirement: PQC NIST KAT verification
ML-KEM-768 and ML-DSA-65 SHALL pass the official NIST PQC Known Answer Test vectors from the FIPS 203 / 204 reference packages, vendored under `security/tests/kat-vectors/` and pinned by SHA-256 in `SHA256SUMS`. CI gate `pqc-kat-verify` is blocking.

#### Scenario: ML-KEM-768 KAT vector matches reference output
- **WHEN** the harness drives `mlkem768::keypair_from_seed(seed)` and `encapsulate_derand(pk, coins)` for every vector in `PQCkemKAT.rsp`
- **THEN** the produced `pk`, `sk`, `ct`, and `ss` are byte-exact equal to the reference

#### Scenario: ML-DSA-65 KAT vector signs deterministically
- **WHEN** the harness calls `mldsa65::sign_derand(sk, message, randomness)` for every vector in `PQCsignKAT.rsp`
- **THEN** the produced signature is byte-exact equal to the reference, and `verify(pk, message, sig)` returns `Ok(())`

#### Scenario: KAT vectors fail digest check
- **WHEN** any `.rsp` file under `security/tests/kat-vectors/` has a SHA-256 digest mismatching `SHA256SUMS`
- **THEN** the harness aborts setup before any comparisons and the CI job fails

### Requirement: PQC differential fuzz vs independent reference
ML-KEM-768 and ML-DSA-65 SHALL be fuzzed against `pqcrypto-kyber 0.8` and `pqcrypto-dilithium 0.5` for 60 s in PR CI and 1 h nightly. Bit-for-bit equality is required on deterministic API; functional equivalence (cross-decapsulate, cross-verify) is required on non-deterministic API.

#### Scenario: Seeded encapsulate produces byte-equal ciphertext
- **WHEN** the fuzzer drives `mlkem768::encapsulate_derand(pk, coins)` and `pqcrypto_kyber::kyber768::derand_encapsulate(pk, coins)` with identical inputs
- **THEN** both produce byte-equal `ct` and `ss`

#### Scenario: Cross-implementation decapsulate succeeds
- **WHEN** the fuzzer encapsulates with the own implementation and decapsulates with the reference (or vice versa)
- **THEN** both sides recover the same shared secret

### Requirement: PQC constant-time invariants documented
Every PQC code path in `security/src/crypto/` SHALL document its constant-time invariants in `docs/pqc-side-channel.md`. Non-CT operations SHALL be tagged `// NOT-CT:` with rationale. Equality on secret-derived bytes SHALL use `subtle::ConstantTimeEq`.

#### Scenario: Equality comparison on secret data uses constant-time
- **WHEN** code under `security/src/crypto/` compares byte slices that depend on secret material (decapsulated keys, signatures, MAC tags)
- **THEN** it uses `subtle::ConstantTimeEq::ct_eq`, never `==` or `slice::eq`
