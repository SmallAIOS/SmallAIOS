## Why

GitHub CodeQL code scanning has 27 open alerts across 3 categories (7 critical, 3 high, 17 medium). These block security compliance and represent real issues (cleartext key logging) and false positives (algorithm constants flagged as hard-coded secrets). Fixing these now establishes a clean security baseline and enables CodeQL as a merge gate.

## What Changes

- Add explicit `permissions:` blocks to all CI and release workflow jobs (principle of least privilege)
- Remove cleartext logging of cryptographic key material in QUIC packet protection
- Suppress false-positive hard-coded cryptographic value alerts on ML-DSA/ML-KEM algorithm constants and test fixtures via CodeQL query filters or inline suppressions
- Add CodeQL configuration to prevent future regressions (alert threshold gate)

## Capabilities

### New Capabilities
- `workflow-permissions`: Explicit least-privilege permissions for all GitHub Actions workflow jobs — covers adding `permissions:` blocks to every job in `ci.yml` and `release.yml`, following GitHub's security hardening guidelines
- `crypto-logging-hygiene`: Remove cleartext logging of sensitive cryptographic material — covers replacing key/IV/nonce debug logging with redacted or length-only output in QUIC protection module
- `codeql-tuning`: CodeQL false-positive suppression and configuration — covers inline suppressions for spec-mandated algorithm constants (FIPS 203/204 matrix generation from public seeds), test fixture exclusions, and alert threshold configuration

### Modified Capabilities
None — this change fixes security alerts without changing existing behavior.

## Impact

- `.github/workflows/ci.yml` — add `permissions:` to all 15+ jobs
- `.github/workflows/release.yml` — add `permissions:` to all jobs
- `net/src/quic/protection.rs` — remove/redact debug logging of key material (lines 67-69)
- `security/src/crypto/ml_dsa.rs` — add CodeQL suppression comments on algorithm constants
- `security/src/crypto/ml_kem.rs` — add CodeQL suppression comment on algorithm constant
- `.github/codeql/` — potential custom query configuration for Rust crypto false positives
