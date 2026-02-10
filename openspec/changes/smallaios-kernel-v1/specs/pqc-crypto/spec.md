# Delta for PQC Crypto

## ADDED Requirements

### Requirement: ML-KEM-768 Key Encapsulation
The cryptography subsystem SHALL implement ML-KEM-768 (Kyber) key encapsulation per FIPS 203.

#### Scenario: Generate ML-KEM keypair
- WHEN a component requests ML-KEM-768 key generation
- THEN the system MUST generate a valid keypair (public key and secret key)
- AND the keypair MUST be generated using output from the qualified CSPRNG

#### Scenario: Encapsulate and decapsulate shared secret
- WHEN a client encapsulates a shared secret using a valid ML-KEM-768 public key
- THEN decapsulation with the corresponding secret key MUST recover the identical shared secret
- AND the shared secret MUST be exactly 32 bytes

### Requirement: ML-DSA-65 Digital Signatures
The cryptography subsystem SHALL implement ML-DSA-65 (Dilithium) digital signatures per FIPS 204.

#### Scenario: Sign and verify a message
- WHEN a message is signed with an ML-DSA-65 private key
- THEN verification with the corresponding public key MUST succeed
- AND verification with any other public key MUST fail

#### Scenario: Reject tampered signature
- WHEN a valid ML-DSA-65 signature is modified by even one bit
- THEN verification MUST fail
- AND the system MUST return a SignatureInvalid error

### Requirement: Hybrid Cryptographic Mode
The cryptography subsystem SHALL support hybrid mode combining classical and post-quantum algorithms for interoperability.

#### Scenario: Hybrid KEM with X25519 and ML-KEM-768
- WHEN hybrid KEM mode is configured
- THEN the system MUST perform both X25519 and ML-KEM-768 key exchanges
- AND the final shared secret MUST be derived by combining both shared secrets via HKDF
- AND both algorithms MUST be broken to compromise the shared secret

#### Scenario: Hybrid signatures with Ed25519 and ML-DSA-65
- WHEN hybrid signature mode is configured
- THEN the system MUST produce both an Ed25519 and an ML-DSA-65 signature
- AND verification MUST require both signatures to be valid

### Requirement: AES-256-GCM Symmetric Encryption
The cryptography subsystem SHALL implement AES-256-GCM authenticated encryption with hardware acceleration.

#### Scenario: Encrypt and decrypt with AES-256-GCM
- WHEN data is encrypted with AES-256-GCM using a 256-bit key and 96-bit nonce
- THEN decryption with the same key and nonce MUST recover the original plaintext
- AND the 128-bit authentication tag MUST be verified before returning plaintext

#### Scenario: Hardware acceleration on x86-64
- WHEN the CPU supports AES-NI and PCLMULQDQ instructions
- THEN the implementation MUST use hardware-accelerated AES and GCM operations
- AND MUST fall back to software implementation only if hardware support is absent

#### Scenario: Hardware acceleration on ARM64
- WHEN the CPU supports ARMv8 Crypto Extensions
- THEN the implementation MUST use the hardware AES and polynomial multiply instructions

### Requirement: CSPRNG from Hardware RNG
The cryptography subsystem SHALL provide a cryptographically secure PRNG seeded from hardware random number generators.

#### Scenario: Seed from x86-64 RDRAND
- WHEN the system boots on x86-64 hardware with RDRAND/RDSEED support
- THEN the CSPRNG MUST be seeded with at least 256 bits from the hardware RNG
- AND the CSPRNG MUST use SHAKE256 in streaming mode as the output generator

#### Scenario: Seed from ARM64 RNDR
- WHEN the system boots on ARM64 hardware with RNDR instruction support
- THEN the CSPRNG MUST be seeded with at least 256 bits from the hardware RNG

#### Scenario: Periodic reseeding
- WHEN the CSPRNG has produced 1 MB of output or 60 seconds have elapsed
- THEN the CSPRNG MUST automatically reseed from the hardware RNG
- AND MUST reseed immediately on any VM snapshot/restore event

### Requirement: TLS 1.3 with PQC Key Exchange
The cryptography subsystem SHALL implement TLS 1.3 with post-quantum hybrid key exchange for IPC transport encryption.

#### Scenario: Establish TLS 1.3 session with hybrid KEM
- WHEN an external client connects with TLS 1.3 and offers x25519_mlkem768 key share
- THEN the server MUST complete the handshake using hybrid X25519+ML-KEM-768 key exchange
- AND MUST encrypt the session with AES-256-GCM

#### Scenario: Reject pre-TLS-1.3 connections
- WHEN a client attempts to negotiate TLS 1.2 or below
- THEN the server MUST reject the connection with a protocol_version alert
- AND MUST NOT fall back to an older TLS version

### Requirement: Model Signature Verification
The cryptography subsystem SHALL verify ONNX model signatures at load time using ML-DSA-65.

#### Scenario: Verify valid model signature
- WHEN an ONNX model is loaded with a valid smallaios-manifest.toml containing an ML-DSA-65 signature
- THEN the system MUST compute the SHA3-256 hash of the model file
- AND MUST verify the signature against the embedded public key
- AND MUST allow model execution only if verification succeeds

#### Scenario: Reject model with invalid signature
- WHEN an ONNX model's signature does not match its content hash
- THEN the system MUST reject the model with a SignatureVerificationFailed error
- AND MUST NOT load or execute the model

### Requirement: Constant-Time Implementation
All cryptographic operations SHALL be implemented in constant time to prevent timing side-channel attacks.

#### Scenario: No secret-dependent branches
- WHEN cryptographic code processes secret key material
- THEN the execution path MUST NOT contain branches conditioned on secret data
- AND MUST NOT contain memory access patterns dependent on secret data

#### Scenario: Verify constant-time behavior
- WHEN the constant-time property is tested
- THEN dudect-style statistical timing analysis MUST show no measurable timing variation based on input values
- AND assembly-level review of critical paths MUST confirm absence of variable-time instructions
