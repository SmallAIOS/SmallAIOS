## Why

While planning `security-ecdsa-p256-v1`, the question came up: should
SmallAIOS adopt an established, validated crypto library — wolfSSL /
wolfCrypt being the canonical candidate (open source, FIPS 140-3
certified, DO-178C DAL A kits available) — instead of continuing to
hand-write crypto primitives in Rust?

The answer for now is **no**, but the reasoning deserves a durable,
in-tree record so the question is not re-litigated every time a new
primitive is proposed, and so the trigger conditions for revisiting
it are explicit. Four project-level decisions drive the answer:

1. **Memory safety by construction.** SmallAIOS is a clean-room
   `#![no_std]` pure-Rust unikernel with a single address space.
   Statically linking a C crypto library reintroduces the exact
   vulnerability class the architecture exists to exclude, and
   inflates the `cargo-geiger` unsafe surface CI tracks.
2. **License.** wolfSSL/wolfCrypt is dual-licensed GPL-or-commercial.
   Statically linked into the Apache-2.0 unikernel image, GPL terms
   would extend to the shipped binary; the `cargo-deny` license gate
   would reject it.
3. **FIPS validation does not transfer.** A FIPS 140-3 certificate
   attaches to the certified module build on its tested operational
   environments. Compiling wolfCrypt into a custom bare-metal
   unikernel is outside that boundary — the validation claim
   evaporates without a paid operational-environment addition.
4. **DO-178C economics.** The project's stated strategy is MC/DC
   coverage on its own Rust. For verify-only primitives with official
   vector oracles (484 Wycheproof vectors for ECDSA-P256, NIST CAVP
   for RSA-PSS), generating our own evidence is tractable.

The compensating control that makes clean-room crypto defensible is
**official-corpus validation**: every primitive replays the same test
corpora the validated libraries are tested against. That policy is
currently implicit convention — this change makes it a requirement
and enforces the no-C-crypto rule mechanically.

## What Changes

- **New policy capability** covering three requirements:
  1. Every crypto primitive in `security/` SHALL replay an official
     public test corpus (Wycheproof, NIST CAVP/ACVP, or the defining
     RFC's vectors) in the standard test run.
  2. Layer-0 crypto SHALL remain clean-room Rust: no C/C++ crypto
     libraries (wolfSSL, OpenSSL, BoringSSL, mbedTLS, libsodium)
     linked anywhere in the workspace, enforced via `cargo-deny`
     `[bans]` entries for their `-sys`/binding crates so the existing
     Supply Chain Security CI gate rejects them at PR time.
  3. The FIPS/Common Criteria path SHALL be documented: a
     `docs/crypto-validation.md` decision record capturing the trade
     study above and the revisit triggers (a contract requiring FIPS
     validation, a certification authority rejecting CAVP-vector
     evidence, or a primitive too complex to clean-room safely).
- **No code behavior changes.** The `security` crate is untouched;
  this change adds policy, docs, and supply-chain guardrails only.

## Capabilities

### New Capabilities

- `crypto-validation-policy`: official-corpus replay requirement for
  all `security/` primitives, the no-C-crypto-libraries rule and its
  cargo-deny enforcement, and the documented FIPS decision record
  with revisit triggers.

### Modified Capabilities

<!-- none — no existing spec's requirements change -->

## Impact

- **Code:** none (no `security/` changes).
- **Config:** `deny.toml` gains `[bans]` entries for C-crypto binding
  crates (e.g. `openssl-sys`, `wolfssl-sys`, `boring-sys`,
  `mbedtls-sys-auto`, `libsodium-sys`).
- **Docs:** new `docs/crypto-validation.md`; CLAUDE.md gains a
  one-line pointer under Key Design Decisions.
- **CI:** no new jobs — enforcement rides the existing
  Supply Chain Security (`cargo-deny`) gate.
- **Relationship to other changes:** records the context for
  `security-ecdsa-p256-v1` / `security-rsa-pss-v1` (both cite
  official corpora as their acceptance oracle); future
  confidential-AI-edge compliance changes inherit the documented
  FIPS options rather than restarting the debate.
