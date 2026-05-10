# codeql-suppression-policy Specification

## Purpose
TBD - created by archiving change codeql-quality-cleanup-v1. Update Purpose after archive.
## Requirements
### Requirement: Test-Vector Module Convention for Cryptographic KAT Data

Files containing cryptographic known-answer-test (KAT) vectors that trigger `rust/hard-coded-cryptographic-value` SHALL extract those vectors into a sibling submodule named `<base>_test_vectors.rs` (or `test_vectors/<topic>.rs` if multiple topics exist). Production code SHALL access the vectors via normal `use` paths. The CodeQL configuration (`.github/codeql/codeql-config.yml` or equivalent) SHALL exclude paths matching `**/*_test_vectors.rs` and `**/test_vectors/**` from the `rust/hard-coded-cryptographic-value` query.

#### Scenario: KAT vectors live in a dedicated module

- **WHEN** a reviewer opens `security/src/crypto/ml_kem.rs` (or any file flagged for `rust/hard-coded-cryptographic-value` because of KAT data)
- **THEN** the file SHALL contain no inline byte-array constants whose contents are KAT data
- **AND** any KAT data SHALL be in a sibling `<base>_test_vectors.rs` module (or `test_vectors/<topic>.rs`)
- **AND** the production code SHALL reference the vectors via `use super::test_vectors::*;` (or equivalent path)

#### Scenario: CodeQL config excludes test-vector modules

- **WHEN** CodeQL is run on the repository (default-setup or workflow-driven)
- **THEN** files matching `**/*_test_vectors.rs` and `**/test_vectors/**` SHALL NOT produce `rust/hard-coded-cryptographic-value` findings
- **AND** the exclusion SHALL be expressed in version-controlled configuration (not via UI dismissals)

#### Scenario: Test-vector files contain only data

- **WHEN** a reviewer opens any `*_test_vectors.rs` file
- **THEN** it SHALL contain only `pub(crate) const`/`pub(crate) static` byte-array declarations and brief documentation comments
- **AND** it SHALL NOT contain executable functions, trait implementations, or business logic

### Requirement: Inline Suppression Convention for One-Off False Positives

When a CodeQL false positive does not match the test-vector pattern (i.e., a one-off constant, not a recurring KAT-vector class), suppression SHALL use an inline `// lgtm[<rule-id>]` annotation on the offending line, accompanied by a comment explaining why the finding is not a real defect.

#### Scenario: Inline annotation has accompanying rationale

- **WHEN** a reviewer encounters a `// lgtm[rule-id]` annotation in source
- **THEN** the same line or the line immediately above SHALL contain a comment explaining the rationale
- **AND** the rationale SHALL identify the rule and why the flagged value is not a real defect (e.g., "domain-separation tag, not a secret")

#### Scenario: Inline annotation is not used for KAT-vector data

- **WHEN** a reviewer encounters multiple inline `// lgtm[rust/hard-coded-cryptographic-value]` annotations in the same file
- **THEN** the file SHALL be considered a candidate for migration to the test-vector module convention
- **AND** the migration SHALL happen in this change or a follow-up change

### Requirement: Suppression Policy Documentation

A document at `docs/code-quality.md` SHALL describe (a) the difference between GitHub Code Scanning alerts and the Code Quality view, (b) the suppression policy (test-vector module convention vs inline annotation), and (c) a "before you suppress, ask…" checklist for contributors. The repository's contributor-facing guide SHALL link to this document.

#### Scenario: Documentation exists and is linked

- **WHEN** a contributor reads the project's contributor-facing guide
- **THEN** they SHALL find a link to `docs/code-quality.md`
- **AND** `docs/code-quality.md` SHALL contain sections covering: Code Scanning vs Code Quality, suppression policy, KAT-vector module convention, when to fix vs suppress

#### Scenario: Policy enables consistent triage

- **WHEN** a contributor encounters a CodeQL false positive
- **THEN** they SHALL be able to follow `docs/code-quality.md` to decide between fixing, extracting to a test-vector module, applying an inline annotation, or escalating

### Requirement: Suppression Strategy Effectiveness Verification

After this change lands, a CodeQL run on the post-change `develop` (or the change branch) SHALL produce zero open `rust/hard-coded-cryptographic-value` findings on the previously-affected paths (`security/src/crypto/ml_kem.rs`, `security/src/crypto/ml_dsa.rs`, `net/src/quic/protection.rs`).

#### Scenario: Zero findings on affected paths post-change

- **WHEN** a CodeQL `language:rust` analysis is completed on the change branch (or post-merge `develop`)
- **THEN** the SARIF results SHALL contain zero entries with `ruleId == "rust/hard-coded-cryptographic-value"` whose location path is `security/src/crypto/ml_kem.rs`, `security/src/crypto/ml_dsa.rs`, or `net/src/quic/protection.rs`

#### Scenario: One week of green runs before archive

- **WHEN** at least 7 days have elapsed since this change merged to `develop`
- **AND** all daily CodeQL `language:rust` analyses on `develop` during that window have zero findings on the previously-affected paths
- **THEN** this change MAY be archived

#### Scenario: Recurrence triggers re-evaluation

- **WHEN** any CodeQL run after this change lands surfaces a new finding of `rust/hard-coded-cryptographic-value` on a previously-affected path
- **THEN** the policy SHALL NOT be considered effective for that case
- **AND** a follow-up change SHALL be opened to revise the suppression strategy

