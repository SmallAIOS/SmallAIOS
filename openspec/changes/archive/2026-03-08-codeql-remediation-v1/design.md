## Context

GitHub CodeQL code scanning reports 27 open alerts against the SmallAIOS repository. These fall into three categories:

1. **Missing workflow permissions** (17 medium) — CI and release workflows lack explicit `permissions:` blocks, defaulting to broad `write-all` access.
2. **Cleartext logging of sensitive information** (3 high) — QUIC packet protection module logs key, IV, and HP key bytes during construction.
3. **Hard-coded cryptographic values** (7 critical) — False positives. CodeQL flags deterministic matrix/polynomial generation from public seeds in ML-DSA (FIPS 204) and ML-KEM (FIPS 203) as "hard-coded secrets." These are specification-mandated algorithm constants derived from public parameters (`rho` seed), not secret material. Test fixtures in `#[cfg(test)]` blocks are also flagged.

The project uses a custom `#![no_std]` SHAKE256-based CSPRNG for all runtime randomness. No `rand` crate ecosystem is used.

## Goals / Non-Goals

**Goals:**
- Resolve all 27 open CodeQL alerts to zero
- Establish least-privilege permissions for all GitHub Actions workflows
- Remove any logging of cryptographic key material
- Document why certain CodeQL alerts are false positives (spec-mandated algorithm constants)
- Enable CodeQL as a merge gate (zero open alerts baseline)

**Non-Goals:**
- Changing the CSPRNG implementation (SHAKE256-based is correct for `no_std` FIPS compliance)
- Modifying ML-DSA/ML-KEM algorithm logic (the flagged code is correct per FIPS 203/204)
- Adding `rand_chacha` — not applicable in `#![no_std]` environment; SHAKE256 provides equivalent 256-bit security

## Decisions

### 1. Use per-job `permissions:` blocks (not top-level)

**Decision:** Add `permissions:` to each job individually rather than a single top-level block.

**Rationale:** Per-job permissions provide finer-grained least-privilege. The `sonarcloud` job needs `pull-requests: read`, the `coverage` job needs `checks: write`, etc. A top-level block would need to be the union of all jobs' needs, which is overly broad.

**Alternative considered:** Top-level `permissions: read-all` with per-job overrides. Rejected because it's harder to audit and some jobs need no permissions at all.

### 2. Remove debug logging entirely (not redact)

**Decision:** Remove the three debug log lines in `protection.rs` that print key, IV, and HP key bytes. Do not replace with redacted versions.

**Rationale:** These were development-time debug prints. In a `no_std` OS kernel there is no stdout in production — these only fire in test builds. Removing them is cleaner than redacting.

**Alternative considered:** Replace with `[REDACTED]` or length-only logging. Rejected as unnecessary complexity for dead debug code.

### 3. Use `lgtm` comments for spec-mandated algorithm constants

**Decision:** Add `// lgtm[rust/hard-coded-cryptographic-value]` inline comments on the 7 flagged lines in ML-DSA and ML-KEM, with explanatory comments noting these are FIPS 203/204 specification-mandated deterministic operations on public seeds.

**Rationale:** These are genuine false positives. The "hard-coded" values are deterministic polynomial/matrix samples generated from the public seed `rho` as required by the FIPS specification. They are not secrets. CodeQL's Rust analysis doesn't understand that `sample_uniform(&rho, ...)` is a PRF expansion of a public parameter, not secret key material. The `lgtm` comment is CodeQL's standard suppression mechanism.

FIPS 204 Section 6.1 (KeyGen): "A ← ExpandA(ρ)" — this is exactly what the flagged `sample_uniform(&rho, ...)` calls implement. FIPS 203 Section 7.2 (Encrypt): same pattern with `sample_ntt(rho, ...)`.

Test fixture keys (`&[0xAA; 32]`, etc.) in `#[cfg(test)]` blocks are also false positives — they're deterministic test vectors, not production secrets.

**Alternative considered:** Custom CodeQL query configuration to exclude these paths. Rejected because `lgtm` comments are simpler, self-documenting, and don't require maintaining a separate config file.

### 4. No custom CodeQL configuration needed

**Decision:** Use inline suppressions only. Do not create `.github/codeql/` configuration.

**Rationale:** With only 7 false positives, inline comments are simpler and more transparent than a separate config. Each suppression is co-located with the code it applies to, making it obvious during code review.

## Risks / Trade-offs

- **[Risk] Future CodeQL rules may flag new code** → Mitigation: Once at zero alerts, enable CodeQL as a required status check so new alerts block PRs.
- **[Risk] `lgtm` comments may be misused to suppress real issues** → Mitigation: PR review process requires justification for any new `lgtm` comments. The FIPS spec references in the comments make the rationale clear.
- **[Risk] Removing debug logging reduces debuggability** → Mitigation: QUIC protection is well-tested (8 unit tests). If debugging is needed, temporary logging can be added locally.
