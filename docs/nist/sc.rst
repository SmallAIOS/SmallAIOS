System and Communications Protection (SC)
==========================================

.. nist_control:: SC-8
   :title: Transmission Confidentiality and Integrity
   :status: implemented
   :implements: net/src/tls.rs, net/src/quic/, security/src/crypto/

   All network communication uses TLS 1.3 with post-quantum hybrid key
   exchange. QUIC provides authenticated encryption (AES-256-GCM) for all
   data in transit. No unencrypted transport protocols are available.

.. nist_control:: SC-12
   :title: Cryptographic Key Establishment and Management
   :status: implemented
   :implements: security/src/crypto/key_manager.rs, security/src/crypto/ml_kem.rs

   Key establishment uses ML-KEM-768 (FIPS 203) for post-quantum key
   encapsulation combined with X25519 for hybrid security. Keys are managed
   through a dedicated key manager with secure zeroization on drop.

.. nist_control:: SC-13
   :title: Cryptographic Protection
   :status: implemented
   :implements: security/src/crypto/

   Full post-quantum cryptographic stack:

   - **Hash**: SHA-3 (FIPS 202)
   - **AEAD**: AES-256-GCM
   - **KEM**: ML-KEM-768 (FIPS 203)
   - **Signatures**: ML-DSA-65 (FIPS 204) + Ed25519 hybrid
   - **Key exchange**: X25519
   - **CSPRNG**: ChaCha20-based, seeded from hardware RNG

.. nist_control:: SC-23
   :title: Session Authenticity
   :status: implemented
   :implements: net/src/quic/protection.rs, net/src/tls.rs

   QUIC connection IDs and TLS 1.3 session tickets provide session
   authenticity. Connection migration preserves session state with
   cryptographic binding.

.. nist_control:: SC-28
   :title: Protection of Information at Rest
   :status: partial
   :implements: security/src/crypto/aes_gcm.rs

   AES-256-GCM encryption is available for data at rest. Full disk/partition
   encryption is not yet implemented (bare-metal target has no filesystem).

.. nist_control:: SC-39
   :title: Process Isolation
   :status: implemented
   :implements: kernel/src/mem/, arch/*/src/paging.rs

   Single address space unikernel with hardware page table isolation.
   Capability-gated memory regions prevent unauthorized cross-process access.
