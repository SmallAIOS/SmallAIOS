## ADDED Requirements

### Requirement: Suppress false-positive hard-coded crypto alerts on algorithm constants
Lines flagged by CodeQL rule `rust/hard-coded-cryptographic-value` that implement FIPS 203 (ML-KEM) or FIPS 204 (ML-DSA) specification-mandated deterministic operations on public seeds SHALL be suppressed with `lgtm` inline comments. Each suppression comment SHALL include a brief justification referencing the relevant FIPS specification section.

#### Scenario: ML-DSA matrix generation from public rho seed
- **WHEN** CodeQL flags `sample_uniform(&rho, ...)` in `ml_dsa.rs` as a hard-coded cryptographic value
- **THEN** the line SHALL have an `// lgtm[rust/hard-coded-cryptographic-value]` comment explaining this is FIPS 204 Section 6.1 ExpandA(ρ) on a public seed

#### Scenario: ML-KEM polynomial generation from public rho seed
- **WHEN** CodeQL flags `sample_ntt(rho, ...)` or `prf(random_coins, ...)` in `ml_kem.rs` as a hard-coded cryptographic value
- **THEN** the line SHALL have an `// lgtm[rust/hard-coded-cryptographic-value]` comment explaining this is FIPS 203 specification-mandated

#### Scenario: Test fixture keys are suppressed
- **WHEN** CodeQL flags test-only key material (e.g., `&[0xAA; 32]`) inside `#[cfg(test)]` blocks
- **THEN** the line SHALL have an `// lgtm[rust/hard-coded-cryptographic-value]` comment noting this is a deterministic test fixture

### Requirement: Zero open CodeQL alerts baseline
After all remediations are applied, the repository SHALL have zero open CodeQL code scanning alerts. CodeQL SHALL be enabled as a required status check on PRs targeting `main` and `develop`.

#### Scenario: PR with new CodeQL alert is blocked
- **WHEN** a PR introduces code that triggers a new CodeQL alert
- **THEN** the PR's CodeQL status check SHALL report failure and block merge

#### Scenario: Clean baseline after remediation
- **WHEN** all remediation changes are merged
- **THEN** `gh api repos/SmallAIOS/SmallAIOS/code-scanning/alerts --jq '[.[] | select(.state == "open")] | length'` SHALL return 0
