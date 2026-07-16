# Tasks — crypto-validation-strategy-v1

## 1. Decision Record

- [x] 1.1 Write `docs/crypto-validation.md`: corpus-replay policy, the four-point rationale against C crypto libraries (memory safety, GPL-or-commercial license, non-transferable FIPS operational-environment boundary, DO-178C evidence economics), the enumerated future FIPS options, and the explicit revisit triggers
- [x] 1.2 Audit the shipped `security/` primitives and record in the doc which official corpus each replays (SHA-2/SHA-3, ChaCha20-Poly1305, AES-256-GCM, Ed25519, X25519, ML-KEM-768, ML-DSA-65), noting any gaps found
- [x] 1.3 Add the one-line pointer to `docs/crypto-validation.md` under CLAUDE.md Key Design Decisions

## 2. Mechanical Enforcement

- [x] 2.1 Add `[bans]` entries to `deny.toml` for `openssl-sys`, `openssl`, `wolfssl-sys`, `wolfssl`, `boring-sys`, `boring`, `mbedtls-sys-auto`, `mbedtls`, `libsodium-sys`, `sodiumoxide`, each with a `reason` pointing at `docs/crypto-validation.md`
- [x] 2.2 Run `cargo deny check bans` on the unmodified workspace and confirm it passes (no existing transitive dependency trips the new bans); document any required exception in the same PR

## 3. Land

- [x] 3.1 `openspec validate crypto-validation-strategy-v1 --type change --strict` passes
- [x] 3.2 PR against `develop` titled `docs(security): crypto validation strategy — clean-room policy, corpus replay, C-crypto bans (crypto-validation-strategy-v1)` (landed as #228, merged 2026-07-03)
