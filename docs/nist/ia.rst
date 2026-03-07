Identification and Authentication (IA)
=======================================

.. nist_control:: IA-5
   :title: Authenticator Management
   :status: implemented
   :implements: security/src/crypto/key_manager.rs, security/src/crypto/csprng.rs

   Cryptographic authenticators (keys, certificates) are managed through the
   key manager. Keys are generated using a CSPRNG seeded from hardware entropy
   (RDRAND on x86, TRNG on ARM). Keys are securely zeroized on drop.

.. nist_control:: IA-7
   :title: Cryptographic Module Authentication
   :status: implemented
   :implements: security/src/crypto/

   The crypto module provides FIPS-aligned algorithms:

   - ML-KEM-768 (FIPS 203) for key encapsulation
   - ML-DSA-65 (FIPS 204) for digital signatures
   - SHA-3 (FIPS 202) for hashing
   - AES-256-GCM for authenticated encryption

   All implementations are clean-room ``#![no_std]`` Rust with no external
   C dependencies.

.. nist_control:: IA-9
   :title: Service Identification and Authentication
   :status: implemented
   :implements: net/src/tls.rs, net/src/quic/

   TLS 1.3 mutual authentication via certificate exchange. Services connecting
   over QUIC must present valid certificates verified against the configured
   trust store.

.. nist_control:: IA-11
   :title: Re-authentication
   :status: partial
   :implements: net/src/quic/

   QUIC 0-RTT resumption includes anti-replay protections. Full session
   re-authentication is supported via TLS 1.3 KeyUpdate mechanism.
