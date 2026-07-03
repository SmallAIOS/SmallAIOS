# Tasks — security-rsa-pss-v1

## 1. SHA-512 prerequisite (`security/src/sha2.rs`)

- [x] 1.1 Refactor the SHA-512 core out of `Sha384` into a shared `Sha512Core` (state + 128-byte buffer + `K512` 80-round `compress`), parameterized by IV; keep `Sha384`'s public API (`new`/`update`/`finalize`, `sha384()`) byte-for-byte
- [x] 1.2 Add `Sha512` (H0_512 IV, 64-byte output), `SHA512_DIGEST_LEN`, and one-shot `sha512()` as a thin wrapper over `Sha512Core`
- [x] 1.3 Test SHA-512 against FIPS 180-4 / NIST CAVP known-answer vectors (empty, "abc", the two-block message, and a multi-block case); confirm SHA-384 vectors still pass after the refactor

## 2. Big-integer arithmetic (`security/src/crypto/big_int.rs`, private)

- [ ] 2.1 `Vec<u64>`-backed little-endian unsigned bigint with normalize, compare, bit-length, and byte (big-endian) import/export
- [ ] 2.2 Add, subtract, multiply (schoolbook), and division/remainder enough for Montgomery setup
- [ ] 2.3 Montgomery reduction (compute `n'`/`R^2 mod n`) and Montgomery-ladder `mod_exp` that is constant-time in the exponent
- [ ] 2.4 Strict DER INTEGER parser producing a bigint (reject truncation, wrong tag, negative/high-bit, non-minimal length; error, never panic)
- [ ] 2.5 Unit-test bigint: add/sub/mul known answers, `mod_exp` against `(base, exp, mod, result)` fixtures spanning 2048/3072/4096-bit moduli, DER INTEGER round-trips and refusals

## 3. MGF1 (`security/src/crypto/mgf1.rs`)

- [ ] 3.1 Define `PssHash { Sha256, Sha384, Sha512 }` with `hlen()` and digest helpers dispatching to `security::sha2`
- [ ] 3.2 Implement MGF1 (RFC 8017 App. B.2.1) once, parameterized by `PssHash`: concatenate `Hash(seed || C)` with a 4-byte BE counter from 0, truncate to the requested length
- [ ] 3.3 Unit-test MGF1 against RFC 8017 / CAVP mask vectors, including a multi-block (mask longer than one hash output) case

## 4. RSA public key + EMSA-PSS-VERIFY (`security/src/crypto/rsa_pss.rs`)

- [ ] 4.1 Define `RsaPssError` (malformed-DER, unsupported-key-size, wrong-algorithm, bad-signature, …) and `RsaPublicKey { n, e }`
- [ ] 4.2 Strict SPKI parser: `rsaEncryption` OID + BIT STRING wrapping `RSAPublicKey ::= SEQUENCE { INTEGER n, INTEGER e }`; refuse modulus < 2048 bits at parse time; reject non-RSA/malformed with an error
- [ ] 4.3 EMSA-PSS-VERIFY per RFC 8017 §9.1.2 (`sLen == hLen`): recover `EM = s^e mod n` (reject `s >= n`, wrong-length sig first), check `0xBC` trailer, leftmost zero bits, `0x01` DB separator, recompute `H'`, compare with `constant_time_eq`
- [ ] 4.4 `rsa_pss_verify(pk, hash, message, signature) -> Result<(), RsaPssError>` tying SPKI key + `PssHash` + mod_exp + EMSA-PSS-VERIFY together
- [ ] 4.5 Unit-test the seams: known-good vector verifies; tampered message rejects; `s >= n` / zero-length / wrong-length signatures reject without panic

## 5. NIST CAVP corpus replay

- [ ] 5.1 Generate/commit `security/src/crypto/rsa_pss_test_vectors.rs` from NIST CAVP RSA-PSS verify vectors: ≥30 vectors covering all 9 (RSA-2048/3072/4096 × SHA-256/384/512) combinations, plus known-bad and Bleichenbacher-style malformed signatures (follow the `*_test_vectors.rs` `#[cfg(test)]` pattern)
- [ ] 5.2 Wire `pub mod rsa_pss; mod big_int; pub mod mgf1;` (+ `#[cfg(test)] mod rsa_pss_test_vectors;`) into `crypto/mod.rs`; update the module-header doc list
- [ ] 5.3 Replay test: every good vector verifies, every bad/malformed rejects with an error (not panic), executed-count assertion so none is skipped, failure output names the offending vector

## 6. Quality gates

- [ ] 6.1 `just fmt-check` and `just clippy` clean; `big_int` is private (not in the public API / `cargo doc`)
- [ ] 6.2 `cargo test -p smallaios-security` green; full `just test` green
- [ ] 6.3 `#![no_std]` bare-metal builds green (`just build-kernel-x86`, `just build-kernel-arm`, RISC-V target check)
- [ ] 6.4 Coverage ≥95% on `rsa_pss.rs` + `mgf1.rs` + `big_int.rs`; no new external dependency in `security/Cargo.toml`

## 7. Land

- [ ] 7.1 `openspec validate security-rsa-pss-v1 --type change --strict` passes
- [ ] 7.2 PR against `develop` titled `feat(security): RSA-PSS signature verification + SHA-512 (security-rsa-pss-v1)`, noting the tls-client wiring is deferred to `tls-tcp-client-v1` task 5.5
