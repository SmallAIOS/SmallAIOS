Access Control (AC)
===================

.. nist_control:: AC-1
   :title: Policy and Procedures
   :status: implemented
   :implements: security/src/capability.rs

   SmallAIOS enforces access control through a capability-based security model.
   All resource access requires valid capabilities checked at the syscall boundary.

.. nist_control:: AC-3
   :title: Access Enforcement
   :status: implemented
   :implements: security/src/capability.rs, kernel/src/syscall/capability.rs

   The kernel enforces mandatory access control via capabilities. Each process
   holds a capability set; operations are denied if the required capability is
   not present. Capabilities cannot be forged — they are managed by the kernel.

.. nist_control:: AC-4
   :title: Information Flow Enforcement
   :status: implemented
   :implements: security/src/formal_gate.rs, ipc/src/lib.rs

   IPC messages are filtered through the formal security gate when the
   ``formal-gate`` feature is enabled. Security labels on IPC channels enforce
   information flow policies (Bell-LaPadula style no-read-up, no-write-down).

.. nist_control:: AC-6
   :title: Least Privilege
   :status: implemented
   :implements: security/src/capability.rs

   The capability system implements least privilege by default. Processes start
   with no capabilities and must be explicitly granted the minimum set needed.
   Capabilities can be revoked but not escalated.

.. nist_control:: AC-17
   :title: Remote Access
   :status: implemented
   :implements: net/src/quic/, net/src/tls.rs

   Remote access is exclusively via QUIC/HTTP3 with TLS 1.3 and post-quantum
   hybrid key exchange (ML-KEM-768 + X25519). No plaintext remote access
   protocols are supported.

.. nist_control:: AC-25
   :title: Reference Monitor
   :status: implemented
   :implements: kernel/src/syscall/, security/src/capability.rs

   The syscall interface acts as a reference monitor — all resource access
   passes through the kernel's capability check. The reference monitor is:
   always invoked (all paths go through syscall), tamper-proof (kernel memory
   is isolated), and small enough to verify (46 syscalls).
