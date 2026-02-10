# Delta for Post-Quantum Cryptography

## ADDED Requirements

### Requirement: ML-KEM-768 Key Encapsulation
The system SHALL implement ML-KEM-768 (Kyber) per FIPS 203 for all key encapsulation operations.

#### Scenario: Generate keypair
- **WHEN** the crypto subsystem generates an ML-KEM-768 keypair
- **THEN** the public key MUST be 1184 bytes and the secret key MUST be 2400 bytes
- **AND** the keypair MUST pass NIST Known Answer Test (KAT) vectors

#### Scenario: Encapsulate and decapsulate
- **WHEN** a client encapsulates a shared secret using a valid ML-KEM-768 public key
- **THEN** the ciphertext MUST be 1088 bytes
- **AND** decapsulation with the corresponding secret key MUST recover the identical shared secret

### Requirement: ML-DSA-65 Digital Signatures
The system SHALL implement ML-DSA-65 (Dilithium) per FIPS 204 for all digital signature operations.

#### Scenario: Sign and verify model signature
- **WHEN** an ONNX model file is signed with an ML-DSA-65 private key
- **THEN** the signature MUST be 3309 bytes
- **AND** verification with the corresponding public key MUST succeed for unmodified models
- **AND** verification MUST fail for any single-bit modification of the model

### Requirement: Hybrid Cryptography Mode
The system SHALL support hybrid mode combining classical and post-quantum algorithms, with hybrid as the default.

#### Scenario: Hybrid key exchange in TLS
- **WHEN** a TLS 1.3 handshake uses hybrid mode
- **THEN** the key exchange MUST combine X25519 and ML-KEM-768
- **AND** the shared secret MUST be derived from both algorithms via HKDF
- **AND** an attacker MUST compromise both X25519 and ML-KEM-768 to recover the shared secret

#### Scenario: Hybrid signature verification
- **WHEN** model signature verification uses hybrid mode
- **THEN** both Ed25519 and ML-DSA-65 signatures MUST be present and valid
- **AND** verification MUST fail if either signature is invalid

### Requirement: AES-256-GCM with Hardware Acceleration
The system SHALL implement AES-256-GCM using hardware acceleration where available.

#### Scenario: x86-64 hardware acceleration
- **WHEN** the CPU supports AES-NI and PCLMULQDQ
- **THEN** AES-256-GCM MUST use hardware instructions for encryption and GCM multiplication
- **AND** throughput MUST exceed 1 GB/s on modern hardware

#### Scenario: ARM64 hardware acceleration
- **WHEN** the CPU supports ARMv8 Crypto Extensions
- **THEN** AES-256-GCM MUST use hardware AES and PMULL instructions

### Requirement: Constant-Time Implementation
All cryptographic operations SHALL execute in constant time to prevent timing side-channel attacks.

#### Scenario: No secret-dependent branches
- **WHEN** the crypto implementation is analyzed for timing side channels
- **THEN** there MUST be zero conditional branches dependent on secret data
- **AND** dudect-style statistical testing MUST confirm constant-time behavior with p > 0.05

### Requirement: CSPRNG from Hardware RNG
The system SHALL implement a CSPRNG seeded from hardware random number generators.

#### Scenario: Seed from x86-64 RDRAND
- **WHEN** running on x86-64 with RDRAND/RDSEED support
- **THEN** the CSPRNG MUST seed from RDSEED (preferred) or RDRAND
- **AND** MUST reseed every 1 MB of output or every 60 seconds

#### Scenario: Seed from ARM64 RNDR
- **WHEN** running on ARM64 with FEAT_RNG (RNDR instruction)
- **THEN** the CSPRNG MUST seed from RNDR
- **AND** MUST fall back to timer jitter entropy if RNDR is unavailable
