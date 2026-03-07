Audit and Accountability (AU)
==============================

.. nist_control:: AU-2
   :title: Event Logging
   :status: implemented
   :implements: security/src/audit.rs, kernel/src/boot_integrity.rs

   Security-relevant events are logged through the audit subsystem:

   - Capability grant/revoke operations
   - Authentication attempts (TLS handshake success/failure)
   - Boot integrity measurements (when verified-boot is enabled)
   - IPC security label violations (when formal-gate is enabled)

.. nist_control:: AU-3
   :title: Content of Audit Records
   :status: implemented
   :implements: security/src/audit.rs

   Audit records include: event type, timestamp (monotonic clock), source
   component, outcome (success/failure), and relevant identifiers (capability
   ID, connection ID, etc.).

.. nist_control:: AU-9
   :title: Protection of Audit Information
   :status: partial
   :implements: security/src/audit.rs

   Audit log is stored in kernel memory and protected by capability-based
   access control. External log forwarding (syslog, remote collection) is
   planned but not yet implemented.

.. nist_control:: AU-12
   :title: Audit Record Generation
   :status: implemented
   :implements: security/src/audit.rs, security/src/monitoring/

   Audit records are generated at the point of security decision (syscall
   boundary, capability check, TLS validation). The monitoring subsystem
   tracks latency statistics for anomaly detection.

.. nist_control:: AU-14
   :title: Session Audit
   :status: implemented
   :implements: net/src/quic/, security/src/audit.rs

   QUIC connection lifecycle events (handshake, migration, close) are
   auditable. Each connection is assigned a unique identifier for
   correlation.
