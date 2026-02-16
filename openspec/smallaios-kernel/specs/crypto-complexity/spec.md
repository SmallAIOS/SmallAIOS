## ADDED Requirements

### Requirement: ml_dsa_65_sign cognitive complexity ≤ 15
The `ml_dsa_65_sign()` function in `security/src/crypto/ml_dsa.rs` SHALL have cognitive complexity ≤ 15. The function SHALL be decomposed into phase helpers: NTT preparation, challenge computation, norm validation, and signature packing. All existing ML-DSA signing tests and test vectors SHALL continue to pass.

#### Scenario: ml_dsa_65_sign refactored below threshold
- **WHEN** SonarCloud analyzes `ml_dsa_65_sign()` after refactoring
- **THEN** the cognitive complexity score SHALL be ≤ 15 (currently 58)

#### Scenario: ML-DSA signing correctness preserved
- **WHEN** existing ML-DSA-65 signing test vectors execute against the refactored code
- **THEN** all signatures SHALL match expected outputs exactly

### Requirement: ml_dsa_65_verify cognitive complexity ≤ 15
The `ml_dsa_65_verify()` function in `security/src/crypto/ml_dsa.rs` SHALL have cognitive complexity ≤ 15. Reconstruction and check logic SHALL be extracted into a helper. All existing ML-DSA verification tests SHALL continue to pass.

#### Scenario: ml_dsa_65_verify refactored below threshold
- **WHEN** SonarCloud analyzes `ml_dsa_65_verify()` after refactoring
- **THEN** the cognitive complexity score SHALL be ≤ 15 (currently 19)

### Requirement: KeccakState permute cognitive complexity ≤ 15
The `KeccakState::permute()` function in `security/src/crypto/sha3.rs` SHALL have cognitive complexity ≤ 15. Keccak-f round steps (theta, rho, pi, chi, iota) SHALL be extracted into per-step helpers. All existing SHA-3 hash tests SHALL continue to pass.

#### Scenario: KeccakState permute refactored below threshold
- **WHEN** SonarCloud analyzes `KeccakState::permute()` after refactoring
- **THEN** the cognitive complexity score SHALL be ≤ 15 (currently 20)

#### Scenario: SHA-3 hash correctness preserved
- **WHEN** existing SHA-3 test vectors (SHA3-256) execute against the refactored code
- **THEN** all hash outputs SHALL match expected values exactly

### Requirement: No public API changes in security crate
All extracted helper functions SHALL be private. No existing public types, traits, or function signatures in the `security` crate SHALL change.

#### Scenario: Public API unchanged
- **WHEN** downstream crates compile against the refactored security crate
- **THEN** compilation SHALL succeed without modification
