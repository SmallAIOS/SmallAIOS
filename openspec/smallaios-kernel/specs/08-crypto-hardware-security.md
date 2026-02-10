# Spec 08: Quantum-Resilient Cryptography and Hardware Security

## Overview

SmallAIOS implements **post-quantum cryptography (PQC)** natively for all
cryptographic operations. As quantum computers advance, classical algorithms
(RSA, ECDSA, ECDH) become vulnerable to Shor's algorithm. SmallAIOS is designed
for long-lived deployments and must protect inference workloads against
harvest-now-decrypt-later attacks.

When running on bare metal or in a VM with hardware security features, SmallAIOS
leverages CPU and platform security primitives for defense in depth.

## Post-Quantum Cryptographic Algorithms

### Selected Algorithms (NIST PQC Standards)

| Use Case | Algorithm | Standard | Security Level |
|---|---|---|---|
| Key Encapsulation (KEM) | ML-KEM-768 (Kyber) | FIPS 203 | Level 3 (AES-192 equivalent) |
| Digital Signatures | ML-DSA-65 (Dilithium) | FIPS 204 | Level 3 |
| Hash-based Signatures | SLH-DSA (SPHINCS+) | FIPS 205 | Level 3 (stateless fallback) |
| Symmetric Encryption | AES-256-GCM | FIPS 197 | Quantum: Level 128 (Grover) |
| Hashing | SHA-3-256 / SHAKE256 | FIPS 202 | Quantum: Level 128 |
| Hashing (fast) | BLAKE3 | — | Non-cryptographic uses |
| KDF | HKDF-SHA3-256 | RFC 5869 variant | Key derivation |

### Hybrid Mode (Transition Period)

For interoperability with classical systems, SmallAIOS supports **hybrid** key
exchange and signatures that combine classical and post-quantum algorithms:

- **Hybrid KEM**: X25519 + ML-KEM-768 (both must be broken to compromise)
- **Hybrid Signatures**: Ed25519 + ML-DSA-65

Hybrid mode is the default. Pure PQC mode is available via configuration.

### Why These Choices

- **ML-KEM (Kyber)**: NIST's primary KEM standard. Small key/ciphertext sizes,
  fast encapsulation/decapsulation. Lattice-based — well-studied security assumptions.
- **ML-DSA (Dilithium)**: NIST's primary signature standard. Reasonable signature
  sizes for our use case (IPC, model signing, TLS).
- **SLH-DSA (SPHINCS+)**: Hash-based signatures as a conservative backup.
  Larger signatures but security relies only on hash function security.
- **AES-256**: Grover's algorithm halves symmetric key strength; AES-256 provides
  128-bit post-quantum security. Hardware-accelerated on both x86 (AES-NI) and
  ARM64 (ARMv8 Crypto Extensions).
- **SHA-3**: Keccak-based, independent of SHA-2 design. Provides domain separation
  via SHAKE XOF.

## Cryptographic Operations in SmallAIOS

### TLS 1.3 with PQC

IPC connections (external TCP) use TLS 1.3 with post-quantum key exchange:

```
Client                              SmallAIOS
  │                                    │
  ├─ ClientHello ──────────────────────►│
  │  supported_groups: x25519_mlkem768  │
  │  signature_algorithms: mldsa65      │
  │                                    │
  │◄────────────────── ServerHello ─────┤
  │  selected_group: x25519_mlkem768    │
  │  key_share: X25519 + ML-KEM-768    │
  │                                    │
  ├─ Finished ─────────────────────────►│
  │  (Hybrid KEM shared secret)         │
  │◄───────────────────── Finished ─────┤
  │                                    │
  │  ═══ AES-256-GCM encrypted ════════│
```

### ONNX Model Signing

Models can be cryptographically signed to prevent tampering:

```
Model Manifest (smallaios-manifest.toml):
┌────────────────────────────────────────────┐
│ [model]                                     │
│ name = "resnet50"                           │
│ version = "1.0"                             │
│ hash_algorithm = "sha3-256"                 │
│ hash = "a1b2c3d4..."                        │
│                                             │
│ [signature]                                 │
│ algorithm = "ml-dsa-65"                     │
│ public_key = "base64..."                    │
│ signature = "base64..."                     │
│                                             │
│ [signature.hybrid]  # Optional classical    │
│ algorithm = "ed25519"                       │
│ public_key = "base64..."                    │
│ signature = "base64..."                     │
└────────────────────────────────────────────┘
```

Verification at model load time:
1. Compute SHA3-256 hash of model file
2. Verify ML-DSA-65 signature against embedded public key
3. If hybrid: also verify Ed25519 signature
4. Reject model if any verification fails

### Secure Random Number Generation

```rust
pub fn sys_random(buf: &mut [u8]) -> Result<(), CryptoError> {
    // 1. Hardware RNG seed (RDRAND/RDSEED on x86, RNDR on ARM64)
    // 2. Mix with boot-time entropy (TSC, APIC timer jitter)
    // 3. Feed into SHAKE256-based CSPRNG
    // 4. Periodically reseed from hardware RNG
}
```

CSPRNG design:
- **Seed**: 256 bits from hardware RNG + boot entropy
- **Generator**: SHAKE256 in streaming mode (XOF)
- **Reseed**: Every 1MB of output or every 60 seconds
- **Fork safety**: Reseed on any state duplication event (VM snapshot)

## Hardware Security Features

### x86-64 Hardware Security

| Feature | Use | Configuration |
|---|---|---|
| **NX bit** (No-Execute) | Prevent code execution from data pages | Always enabled |
| **SMEP** (Supervisor Mode Execution Prevention) | Prevent kernel executing user pages | Enabled in VM mode |
| **SMAP** (Supervisor Mode Access Prevention) | Prevent kernel reading user pages | Enabled in VM mode |
| **PCID** (Process Context ID) | Efficient TLB management | Enabled if available |
| **IBRS/IBPB** (Indirect Branch Prediction) | Spectre v2 mitigation | Enabled |
| **STIBP** (Single Thread Indirect Branch Predictors) | Spectre mitigation for SMT | Enabled |
| **SSBD** (Speculative Store Bypass Disable) | Spectre v4 mitigation | Enabled |
| **CET** (Control-flow Enforcement Technology) | Shadow stack, IBT | Enabled if available |
| **AES-NI** | Hardware AES acceleration | Used by crypto layer |
| **PCLMULQDQ** | Hardware GCM acceleration | Used by crypto layer |
| **SHA Extensions** | Hardware SHA acceleration | Used if available |
| **PKS** (Protection Keys for Supervisor) | Fine-grained memory protection | Future use |
| **TDX** (Trust Domain Extensions) | Confidential computing | Future: run SmallAIOS as a TD |

### ARM64 Hardware Security

| Feature | Use | Configuration |
|---|---|---|
| **PAN** (Privileged Access Never) | Prevent kernel accessing user memory | Always enabled |
| **UAO** (User Access Override) | Controlled user memory access | Managed by POSIX layer |
| **BTI** (Branch Target Identification) | Forward-edge CFI | Enabled if available |
| **PAC** (Pointer Authentication) | Return address signing, pointer integrity | Enabled if available |
| **MTE** (Memory Tagging Extension) | Hardware memory safety (use-after-free, overflow) | Enabled if available |
| **ARMv8 Crypto** | AES, SHA hardware acceleration | Used by crypto layer |
| **RNG** (RNDR/RNDRRS instructions) | Hardware random numbers | Used for CSPRNG seeding |
| **TrustZone** | Secure world isolation | Future: secure key storage |
| **CCA** (Confidential Compute Architecture) | Realm-based isolation | Future: confidential inference |

### Secure Boot Chain (Bare Metal)

```
┌──────────────────────────────────────────────────────┐
│ 1. Platform firmware (UEFI Secure Boot)               │
│    Verifies: SmallAIOS bootloader signature            │
│    Algorithm: RSA-4096 or ML-DSA-65 (if firmware      │
│              supports PQC)                             │
├──────────────────────────────────────────────────────┤
│ 2. SmallAIOS bootloader                               │
│    Verifies: Kernel image hash + ML-DSA-65 signature  │
│    Measures: Kernel image into TPM PCR (if available)  │
├──────────────────────────────────────────────────────┤
│ 3. SmallAIOS kernel                                   │
│    Verifies: ONNX model signatures (ML-DSA-65)        │
│    Verifies: Configuration file integrity (SHA3-256)   │
├──────────────────────────────────────────────────────┤
│ 4. ONNX Runtime                                       │
│    All code paths verified, no dynamic loading         │
└──────────────────────────────────────────────────────┘
```

### TPM 2.0 Integration (Bare Metal)

When a TPM is present:
- Measure boot stages into PCR registers
- Seal secrets (TLS private keys) to PCR values
- Remote attestation of SmallAIOS configuration
- TPM-backed CSPRNG seeding

### NVIDIA GPU Security

- **GPU memory isolation**: Each inference session's GPU memory is fenced
- **Encrypted GPU memory**: Use NVIDIA's MIG (Multi-Instance GPU) if available
  for memory isolation between models
- **No GPU-to-system DMA without explicit mapping**: GPU cannot read arbitrary
  host memory
- **PTX verification**: Only signed/verified PTX kernels are loaded to GPU

## Cryptography Implementation

### Clean Room Approach

All cryptographic code is written from the published specifications:
- FIPS 203 (ML-KEM), FIPS 204 (ML-DSA), FIPS 205 (SLH-DSA)
- FIPS 197 (AES), FIPS 202 (SHA-3)
- RFC 7748 (X25519), RFC 8032 (Ed25519)

No code from existing crypto libraries (OpenSSL, libsodium, ring, RustCrypto)
is used. However, we reference their test vectors for validation.

### Constant-Time Implementation

All cryptographic operations are implemented in constant time to prevent
timing side-channel attacks:
- No secret-dependent branches
- No secret-dependent memory access patterns
- Verified with `dudect`-style statistical testing
- Assembly-level verification for critical paths

### Hardware Acceleration

| Algorithm | x86-64 | ARM64 |
|---|---|---|
| AES-256-GCM | AES-NI + PCLMULQDQ | ARMv8 Crypto Extensions |
| SHA-3 / SHAKE256 | Software (no HW accel) | SHA3 instructions (ARMv8.2-A) |
| SHA-256 | SHA Extensions (if present) | ARMv8 Crypto Extensions |
| ML-KEM (NTT) | AVX2 / AVX-512 | NEON / SVE |
| ML-DSA (NTT) | AVX2 / AVX-512 | NEON / SVE |

The Number Theoretic Transform (NTT) used in lattice-based PQC benefits
significantly from SIMD acceleration.

## Crate Structure

```
security/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── capability.rs     # Capability-based access control
    ├── registry.rs       # Capability registry
    ├── policy.rs         # Default capability policy
    ├── audit.rs          # Security audit logging
    └── crypto/
        ├── mod.rs
        ├── aes_gcm.rs    # AES-256-GCM (with HW accel)
        ├── sha3.rs       # SHA3-256, SHAKE256
        ├── ml_kem.rs     # ML-KEM-768 (Kyber) KEM
        ├── ml_dsa.rs     # ML-DSA-65 (Dilithium) signatures
        ├── slh_dsa.rs    # SLH-DSA (SPHINCS+) signatures
        ├── x25519.rs     # X25519 key exchange (hybrid)
        ├── ed25519.rs    # Ed25519 signatures (hybrid)
        ├── hybrid.rs     # Hybrid KEM and signature schemes
        ├── csprng.rs     # SHAKE256-based CSPRNG
        ├── tls13.rs      # Minimal TLS 1.3 with PQC
        ├── verify.rs     # Model/image signature verification
        └── constant_time.rs  # Constant-time utilities
```

## Configuration

```toml
[crypto]
# PQC mode: "hybrid" (default) or "pqc-only" or "classical-only"
mode = "hybrid"

# KEM algorithm for TLS
kem = "x25519-ml-kem-768"   # Hybrid default

# Signature algorithm for model verification
signature = "ml-dsa-65"

# Require model signatures
require_model_signatures = true

# TLS configuration
tls_min_version = "1.3"
tls_cipher_suites = ["TLS_AES_256_GCM_SHA384", "TLS_CHACHA20_POLY1305_SHA256"]

[hardware_security]
# Enable CPU security features (NX, SMEP, SMAP, CET, PAC, BTI, MTE)
enable_cpu_hardening = true

# Spectre/Meltdown mitigations
spectre_mitigation = true

# TPM integration (bare metal only)
tpm_enabled = false
tpm_pcr_extend = true
tpm_seal_keys = true
```
