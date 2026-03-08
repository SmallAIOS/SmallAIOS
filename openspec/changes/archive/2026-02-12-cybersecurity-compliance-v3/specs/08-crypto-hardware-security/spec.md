# Delta for Quantum-Resilient Cryptography and Hardware Security

## ADDED Requirements

### Requirement: NIST SP 800-53 SC-12 Cryptographic Key Establishment and Management Mapping

The cryptographic subsystem SHALL implement key management practices conforming to NIST SP 800-53 Rev 5 SC-12 (Cryptographic Key Establishment and Management). The SC-12 mapping SHALL cover the complete key lifecycle: generation, distribution, storage, rotation, and destruction. Key generation MUST use approved random bit generators (hardware RNG seeded CSPRNG per SP 800-90A). Key distribution within the kernel MUST use capability-controlled internal transfer (no external key distribution protocol required for the unikernel architecture). Key storage MUST be in volatile memory only, with no persistent key store unless a TPM is available. Key rotation MUST occur at every reboot (new keys generated from fresh hardware entropy). Key destruction (zeroization) MUST occur at shutdown, watchdog reset, or task termination using memory overwrite that is not optimized away by the compiler.

#### Scenario: Key generation uses hardware RNG seeded CSPRNG

- WHEN the cryptographic subsystem generates a new key (signing key, KEM key pair, or symmetric key)
- THEN the key material MUST be derived from the SHAKE256-based CSPRNG
- AND the CSPRNG MUST have been seeded with at least 256 bits of entropy from the hardware RNG (RDRAND/RDSEED on x86-64, RNDR on ARM64)
- AND the key generation process MUST conform to the algorithm-specific key generation procedure defined in the applicable FIPS standard (FIPS 203 for ML-KEM, FIPS 204 for ML-DSA, FIPS 197 for AES)

#### Scenario: Key distribution via capability-controlled transfer

- WHEN a cryptographic key must be provided to a kernel subsystem (e.g., TLS session key to the network stack, signing key to the audit log subsystem)
- THEN the key transfer MUST be mediated by the capability system — the receiving subsystem MUST hold a capability granting READ access to the specific key resource
- AND the key material MUST be passed via a direct memory reference within the single address space (no serialization to IPC or external channel)
- AND the transfer MUST be logged as an audit event (key resource ID, source subsystem, destination subsystem) without including the key material itself

#### Scenario: Keys stored in volatile memory only (no TPM)

- WHEN SmallAIOS is running on a platform without a TPM
- THEN all cryptographic key material MUST reside exclusively in volatile memory (RAM)
- AND key material MUST NOT be written to any persistent storage (disk, flash, NVRAM)
- AND the memory pages containing key material MUST be excluded from any crash dump or memory snapshot facility

#### Scenario: Keys sealed to TPM when available

- WHEN SmallAIOS is running on a platform with a TPM 2.0 module
- THEN long-lived keys (TLS server key, model signing verification key) MAY be sealed to TPM PCR values using the TPM2_Create and TPM2_Load commands
- AND sealed keys MUST be unsealed only when PCR values match the expected boot configuration
- AND the TPM sealing policy MUST be documented in the key management section of the security model

#### Scenario: Key rotation on reboot

- WHEN SmallAIOS reboots (clean restart or watchdog-triggered reset)
- THEN all ephemeral cryptographic keys (TLS session keys, audit log signing keys, KEM key pairs) MUST be regenerated from fresh hardware entropy
- AND the previous key material MUST NOT be reused or recovered
- AND the new keys MUST be distributed to subsystems via the capability-controlled transfer mechanism before any subsystem begins processing

#### Scenario: Key zeroization on shutdown

- WHEN SmallAIOS initiates a shutdown sequence (clean shutdown, watchdog reset, or panic)
- THEN all cryptographic key material in memory MUST be overwritten with zeros using a volatile write operation that the compiler MUST NOT optimize away
- AND zeroization MUST be performed using a dedicated `zeroize()` function that uses `core::ptr::write_volatile` or inline assembly to guarantee the write is not elided
- AND zeroization MUST occur before any memory is released back to the allocator
- AND the zeroization operation MUST cover: all KEM secret keys, all signing private keys, all symmetric keys, all TLS session state, and all CSPRNG internal state

#### Scenario: SC-12 control mapping documented

- WHEN an auditor reviews the SC-12 control mapping
- THEN the documentation MUST provide a table mapping each SC-12 control enhancement (SC-12(1) through SC-12(3)) to the specific SmallAIOS mechanism: SC-12(1) (Availability) mapped to key regeneration on reboot; SC-12(2) (Symmetric Keys) mapped to AES-256 key generation from CSPRNG; and SC-12(3) (Asymmetric Keys) mapped to ML-KEM and ML-DSA key pair generation from CSPRNG
- AND each mapping MUST reference the implementing source module and test artifact

### Requirement: NIST SP 800-53 SC-13 Cryptographic Protection Mapping

The cryptographic subsystem SHALL document compliance with NIST SP 800-53 Rev 5 SC-13 (Cryptographic Protection) by providing: an algorithm selection rationale for each cryptographic algorithm used, a crypto module boundary definition identifying all cryptographic functions and their trust boundary, and a mapping from each SC-13 requirement to the SmallAIOS implementation. The algorithm selection rationale SHALL document why each algorithm was chosen over alternatives, the NIST standard or FIPS publication governing each algorithm, and the security level provided. The module boundary definition SHALL identify every cryptographic function (encryption, decryption, signing, verification, hashing, key generation, key encapsulation, random number generation) and its inputs, outputs, and security-relevant parameters.

#### Scenario: Algorithm selection rationale documented for ML-KEM-768

- WHEN the SC-13 mapping is reviewed for key encapsulation
- THEN the documentation MUST include a rationale entry for ML-KEM-768 stating: it is the NIST primary KEM standard (FIPS 203); it provides Level 3 security (AES-192 equivalent post-quantum); it was selected over NTRU and SIKE due to smaller key/ciphertext sizes and faster encapsulation/decapsulation; and it is used in hybrid mode with X25519 for transition-period interoperability

#### Scenario: Algorithm selection rationale documented for ML-DSA-65

- WHEN the SC-13 mapping is reviewed for digital signatures
- THEN the documentation MUST include a rationale entry for ML-DSA-65 stating: it is the NIST primary digital signature standard (FIPS 204); it provides Level 3 security; it was selected over SLH-DSA (SPHINCS+) as the primary signature algorithm due to smaller signature sizes; and SLH-DSA is retained as a conservative hash-based fallback

#### Scenario: Algorithm selection rationale documented for AES-256-GCM

- WHEN the SC-13 mapping is reviewed for symmetric encryption
- THEN the documentation MUST include a rationale entry for AES-256-GCM stating: it conforms to FIPS 197 (AES) and NIST SP 800-38D (GCM); AES-256 provides 128-bit post-quantum security against Grover's algorithm; GCM provides authenticated encryption; and hardware acceleration is available on both x86-64 (AES-NI) and ARM64 (ARMv8 Crypto Extensions)

#### Scenario: All cryptographic algorithms have selection rationale

- WHEN the complete SC-13 algorithm rationale section is reviewed
- THEN every cryptographic algorithm used by SmallAIOS (ML-KEM-768, ML-DSA-65, SLH-DSA, AES-256-GCM, SHA-3-256, SHAKE256, BLAKE3, HKDF-SHA3-256, X25519, Ed25519, ChaCha20-Poly1305) MUST have a documented selection rationale
- AND each rationale MUST reference the governing FIPS publication or standard, state the security level, and explain why the algorithm was chosen over alternatives

#### Scenario: SC-13 control mapping documented

- WHEN an auditor reviews the SC-13 control mapping
- THEN the documentation MUST map SC-13 to the SmallAIOS crypto subsystem and identify: the approved algorithms (FIPS 203, 204, 205, 197, 202), the implementation approach (clean-room from published specifications), the validation method (NIST test vectors, dudect timing analysis, MC/DC coverage), and the governing configuration parameters (`crypto.mode`, `crypto.kem`, `crypto.signature`)
- AND the mapping MUST confirm that all cryptographic operations use NIST-approved algorithms or document any exceptions with justification

### Requirement: Key Management Lifecycle

The cryptographic subsystem SHALL implement a complete key management lifecycle as follows: keys SHALL be generated at boot from the hardware RNG-seeded CSPRNG; keys SHALL be stored in volatile memory only (no persistent key store) unless a TPM 2.0 module is available for key sealing; keys SHALL be rotated on every reboot by generating fresh key material from new hardware entropy; and keys SHALL be zeroized on shutdown by overwriting all key material in memory with zeros using volatile writes. The lifecycle MUST apply to all key types: ML-KEM key pairs, ML-DSA signing key pairs, SLH-DSA signing key pairs, AES-256 symmetric keys, TLS 1.3 session keys, HKDF-derived keys, and CSPRNG internal state.

#### Scenario: Boot-time key generation sequence

- WHEN SmallAIOS completes hardware initialization and the CSPRNG is seeded
- THEN the key management subsystem MUST generate all required cryptographic keys in the following order: (1) audit log signing key pair (ML-DSA-65), (2) TLS server key pair (ML-DSA-65 + Ed25519 hybrid), (3) TLS KEM key pair (ML-KEM-768 + X25519 hybrid), and (4) any model verification keys loaded from the boot image
- AND all key generation MUST complete before the ONNX runtime or IPC router are started
- AND the completion of key generation MUST be logged as an audit event

#### Scenario: No persistent key storage without TPM

- WHEN SmallAIOS is running on a platform without a TPM 2.0 module
- THEN the key management subsystem MUST NOT write any key material to persistent storage
- AND if the system reboots or loses power, all key material MUST be irrecoverably lost
- AND this behavior MUST be documented as an accepted operational constraint in the key management policy

#### Scenario: Key rotation produces fresh independent keys

- WHEN SmallAIOS reboots and generates new keys
- THEN the new keys MUST be statistically independent from the previous boot's keys
- AND the CSPRNG MUST be reseeded from fresh hardware entropy before new key generation begins
- AND there MUST be no mechanism to derive the new keys from knowledge of the previous keys

#### Scenario: Zeroization covers all key material

- WHEN the zeroization procedure executes during shutdown
- THEN the procedure MUST enumerate all memory locations containing key material by consulting the key management registry
- AND each identified location MUST be overwritten with zeros using `core::ptr::write_volatile` or equivalent non-optimizable write
- AND after zeroization completes, a verification pass MUST confirm that all registered key locations contain only zeros
- AND the total zeroization time MUST NOT exceed 10 milliseconds to ensure completion before hardware power-down in watchdog reset scenarios

### Requirement: Crypto Module Boundary Definition per FIPS 140-3 Level 1

The cryptographic subsystem SHALL define a crypto module boundary conforming to FIPS 140-3 Level 1 requirements. The module boundary SHALL identify: all cryptographic algorithms and their implementations within the boundary; all data inputs to the module (plaintext, keys, configuration parameters); all data outputs from the module (ciphertext, signatures, hashes, random bytes); all control inputs (algorithm selection, mode selection, key length); all status outputs (success, error codes, verification results); and the trust boundary separating the crypto module from the rest of the kernel. The boundary SHALL encompass all code in the `security/src/crypto/` module tree and SHALL exclude non-cryptographic security code (capability system, audit logging, policy engine).

#### Scenario: Module boundary encompasses all cryptographic code

- WHEN the FIPS 140-3 module boundary diagram is reviewed
- THEN the boundary MUST encompass all source files in `security/src/crypto/`: `aes_gcm.rs`, `sha3.rs`, `ml_kem.rs`, `ml_dsa.rs`, `slh_dsa.rs`, `x25519.rs`, `ed25519.rs`, `hybrid.rs`, `csprng.rs`, `tls13.rs`, `verify.rs`, and `constant_time.rs`
- AND the boundary MUST NOT include `security/src/capability.rs`, `security/src/registry.rs`, `security/src/policy.rs`, or `security/src/audit.rs`
- AND the boundary diagram MUST be documented as a PlantUML component diagram in the Sphinx-needs documentation

#### Scenario: All cryptographic algorithms enumerated within boundary

- WHEN the FIPS 140-3 module boundary documentation is reviewed
- THEN it MUST enumerate every cryptographic algorithm implemented within the boundary: ML-KEM-768 (FIPS 203), ML-DSA-65 (FIPS 204), SLH-DSA (FIPS 205), AES-256-GCM (FIPS 197 + SP 800-38D), SHA-3-256 (FIPS 202), SHAKE256 (FIPS 202), X25519 (RFC 7748), Ed25519 (RFC 8032), HKDF-SHA3-256 (RFC 5869 variant), and ChaCha20-Poly1305 (RFC 8439)
- AND each algorithm entry MUST list the Rust module implementing it, the FIPS/RFC standard reference, and whether it is used in FIPS-approved mode or non-approved mode (BLAKE3 is non-approved and MUST be documented as used only for non-cryptographic purposes)

#### Scenario: Data inputs and outputs defined for each algorithm

- WHEN the module boundary documentation describes a specific algorithm (e.g., ML-DSA-65 signing)
- THEN the entry MUST list all data inputs: message to sign, private signing key, and optional context string
- AND MUST list all data outputs: signature bytes
- AND MUST list all control inputs: none (algorithm and parameters are fixed at compile time for ML-DSA-65)
- AND MUST list all status outputs: `Ok(signature)` or `Err(SigningError)` with enumerated error variants

#### Scenario: Trust boundary separation documented

- WHEN the FIPS 140-3 module boundary is reviewed for trust boundary clarity
- THEN the documentation MUST identify the interface between the crypto module and the rest of the kernel as the public API surface of the `security::crypto` Rust module
- AND the documentation MUST state that all key material entering or leaving the module boundary passes through defined API functions (no direct memory access from outside the module)
- AND the documentation MUST identify the hardware interface points: CPU cryptographic instruction extensions (AES-NI, ARMv8 Crypto, NEON/SVE for NTT), hardware RNG instructions (RDRAND/RDSEED, RNDR), and TPM 2.0 commands (when available)

#### Scenario: FIPS 140-3 Level 1 self-test requirements

- WHEN the crypto module initializes at boot
- THEN the module MUST execute power-on self-tests for each approved algorithm: a known-answer test (KAT) for ML-KEM-768 encapsulation/decapsulation, a KAT for ML-DSA-65 signing/verification, a KAT for AES-256-GCM encryption/decryption, a KAT for SHA-3-256, and an entropy source health test for the hardware RNG
- AND if any self-test fails, the crypto module MUST enter an error state and MUST NOT perform any cryptographic operations
- AND the self-test results (pass or fail per algorithm) MUST be logged as audit events
- AND the total self-test execution time MUST NOT exceed 500 milliseconds on the minimum supported hardware platform

#### Scenario: Module boundary does not include non-cryptographic code

- WHEN the FIPS 140-3 module boundary is audited for correctness
- THEN the boundary MUST exclude: the capability system (`capability.rs`, `registry.rs`, `policy.rs`), the audit logging system (`audit.rs`), the ONNX runtime, the IPC router, and all architecture-specific code outside the crypto hardware acceleration paths
- AND any code outside the boundary that calls into the crypto module MUST do so exclusively through the defined public API
- AND any attempt to access crypto module internals (private fields, internal functions) from outside the boundary MUST be prevented by Rust's module visibility rules and MUST be verified by a code audit
