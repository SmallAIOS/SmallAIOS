System and Information Integrity (SI)
======================================

.. nist_control:: SI-2
   :title: Flaw Remediation
   :status: implemented
   :implements: deny.toml, .github/workflows/ci.yml

   Supply chain security is enforced through cargo-deny (advisory database
   checks, license compliance, banned crate detection). Dependabot monitors
   for known vulnerabilities in dependencies.

.. nist_control:: SI-3
   :title: Malicious Code Protection
   :status: implemented
   :implements: security/src/crypto/, kernel/src/boot_integrity.rs

   The verified-boot feature validates kernel integrity at boot time using
   ML-DSA-65 signatures. ONNX model loading verifies model signatures before
   execution. No dynamic code loading or JIT is supported.

.. nist_control:: SI-6
   :title: Security and Privacy Function Verification
   :status: implemented
   :implements: formal/tla/, formal/spin/, formal/promela/

   Security functions are formally verified:

   - **TLA+**: 22 models verify safety invariants (memory allocator correctness,
     protocol state machines, arbitration fairness)
   - **SPIN**: 6 Promela models verify liveness properties (handshake completion,
     message delivery, scheduler fairness)
   - **Unit tests**: >4,100 tests with >93% line coverage

.. nist_control:: SI-7
   :title: Software, Firmware, and Information Integrity
   :status: implemented
   :implements: kernel/src/boot_integrity.rs, security/src/crypto/ml_dsa.rs

   Boot integrity verification measures and validates kernel components using
   cryptographic hashes (SHA-3) and post-quantum signatures (ML-DSA-65).
   Measurement log is maintained for attestation.

.. nist_control:: SI-10
   :title: Information Input Validation
   :status: implemented
   :implements: onnx-rt/src/parser/, net/src/quic/

   All external input is validated:

   - ONNX protobuf: parsed with bounds checking, no unsafe deserialization
   - Network packets: validated headers, checked lengths, no buffer overflows
   - QUIC frames: type/length validation before processing
   - IPC messages: capability-checked before delivery

.. nist_control:: SI-16
   :title: Memory Protection
   :status: implemented
   :implements: kernel/src/mem/, arch/*/src/paging.rs

   Hardware-enforced memory protection via page tables (NX, read-only kernel
   text, guard pages). Stack canaries and buddy allocator prevent heap
   corruption. No ``alloc`` in safety-critical paths when ``no-global-alloc``
   feature is enabled.
