# Delta for Security Model

## ADDED Requirements

### Requirement: NIST SP 800-53 Control Cross-References for Security Mechanisms

Every security mechanism documented in the base security model (Spec 06) SHALL include a cross-reference to the applicable NIST SP 800-53 Rev 5 control(s). The cross-reference SHALL map: the capability-based access control system to AC-3 (Access Enforcement) and AC-6 (Least Privilege); audit and logging to AU-2 (Event Logging), AU-3 (Content of Audit Records), and AU-12 (Audit Record Generation); network security controls (TLS, firewall, input validation) to SC-7 (Boundary Protection), SC-8 (Transmission Confidentiality and Integrity), and SC-23 (Session Authenticity); memory safety enforcement to SI-16 (Memory Protection); resource limits and quotas to SC-5 (Denial-of-Service Protection); boot security to SI-7 (Software, Firmware, and Information Integrity); and the unsafe code policy to SA-11 (Developer Testing and Evaluation) and SI-2 (Flaw Remediation).

#### Scenario: Auditor requests control mapping for capability system

- WHEN an auditor examines the capability-based access control system
- THEN the security model documentation MUST include a cross-reference table entry mapping the capability system to NIST SP 800-53 Rev 5 controls AC-3 (Access Enforcement) and AC-6 (Least Privilege)
- AND the entry MUST identify the SmallAIOS source module (`security/src/capability.rs`, `security/src/registry.rs`, `security/src/policy.rs`) and the formal verification artifact (Lean 4 capability non-forgery proof) as supporting evidence

#### Scenario: Auditor requests control mapping for audit logging

- WHEN an auditor examines the security event logging subsystem
- THEN the security model documentation MUST include a cross-reference table entry mapping audit logging to NIST SP 800-53 Rev 5 controls AU-2 (Event Logging), AU-3 (Content of Audit Records), and AU-12 (Audit Record Generation)
- AND the entry MUST identify the SmallAIOS source module (`security/src/audit.rs`), the IPC key expression for log export (`smallaios/v1/logs`), and the audit event types (capability grants, capability revocations, inference requests, failed operations)

#### Scenario: Auditor requests control mapping for network security

- WHEN an auditor examines the network security controls
- THEN the security model documentation MUST include cross-reference entries mapping TLS 1.3 mutual authentication to SC-8 (Transmission Confidentiality and Integrity) and SC-23 (Session Authenticity), the minimal TCP stack and input validation to SC-7 (Boundary Protection), and connection/rate limits to SC-5 (Denial-of-Service Protection)
- AND each entry MUST reference the corresponding SmallAIOS source module and configuration parameters

#### Scenario: Auditor requests control mapping for memory safety

- WHEN an auditor examines memory safety mechanisms
- THEN the security model documentation MUST include a cross-reference entry mapping Rust language safety guarantees (ownership, borrowing, bounds checking) and the unsafe code policy to SI-16 (Memory Protection) and SA-11 (Developer Testing and Evaluation)
- AND the entry MUST reference the MISRA-Rust coding standard (Spec 12) and fuzz testing artifacts as supporting evidence

#### Scenario: Complete cross-reference table validation

- WHEN the security model cross-reference table is reviewed for completeness
- THEN every security mechanism described in the base Spec 06 (capability tokens, resource types, capability lifecycle, no ambient authority, memory safety, unsafe code policy, network security, TLS configuration, input validation, boot security, container mode security, resource limits, auditing and logging) MUST have at least one NIST SP 800-53 Rev 5 control cross-reference
- AND no security mechanism SHALL be left unmapped

### Requirement: Data Classification Policy

The security model SHALL define a data classification policy with the following classification levels: Public, Internal, and Restricted. Each data type handled by SmallAIOS SHALL be assigned exactly one classification level. The defined classifications SHALL be: model weights as Restricted, inference input/output data as Internal, audit logs as Internal, system configuration as Internal, cryptographic keys (signing keys, TLS private keys, CSPRNG state) as Restricted, and health metrics (CPU utilization, memory usage, inference latency statistics) as Public. The classification level SHALL determine the minimum protection requirements for data at rest and in transit.

#### Scenario: Model weights classified as Restricted

- WHEN an ONNX model is loaded into SmallAIOS
- THEN the model weight data MUST be classified as Restricted
- AND the system MUST enforce that model weight data is accessible only to tasks holding a capability with READ permission on the specific model resource
- AND model weight data MUST NOT be exported via IPC, logging, or health metrics endpoints

#### Scenario: Inference I/O classified as Internal

- WHEN inference input tensors are submitted or output tensors are produced
- THEN the inference I/O data MUST be classified as Internal
- AND the system MUST enforce that inference I/O is accessible only to the submitting task (input) and the designated output recipients (output) via capability-controlled IPC
- AND inference I/O tensor contents MUST NOT appear in audit logs or health metrics

#### Scenario: Audit logs classified as Internal

- WHEN audit log entries are generated by the security subsystem
- THEN the audit log data MUST be classified as Internal
- AND audit logs MUST be exported only via the designated IPC key expression (`smallaios/v1/audit`) to subscribers holding the appropriate capability
- AND audit log entries MUST NOT contain Restricted data (cryptographic keys, model weights)

#### Scenario: Cryptographic keys classified as Restricted

- WHEN cryptographic keys are generated, stored, or used by the crypto subsystem
- THEN all key material (signing keys, TLS private keys, KEM secret keys, CSPRNG internal state) MUST be classified as Restricted
- AND key material MUST NOT be accessible to any task other than the crypto subsystem itself
- AND key material MUST NOT be logged, exported via IPC, or included in health metrics under any circumstances

#### Scenario: Health metrics classified as Public

- WHEN the system publishes health metrics (CPU utilization, memory usage, inference latency statistics, task counts)
- THEN the health metrics data MUST be classified as Public
- AND health metrics MUST be publishable via IPC (`smallaios/v1/metrics`) without requiring elevated capability permissions beyond basic IPC subscribe
- AND health metrics MUST NOT include any Internal or Restricted data

#### Scenario: Classification policy documented in security model

- WHEN the security model documentation is reviewed
- THEN it MUST contain a data classification table listing every data type, its classification level, and the corresponding protection requirements
- AND the table MUST cover at minimum: model weights, inference I/O, audit logs, system configuration, cryptographic keys, and health metrics

### Requirement: Information Flow Enforcement via Capability System

The capability system SHALL enforce data flow isolation between task types. Each task type (inference, IPC router, system management, monitoring) SHALL have a defined set of accessible resource types, and cross-type access SHALL be denied by default. The information flow policy MUST ensure that: inference tasks can access only their assigned model (READ/EXECUTE), input tensors (READ), and output tensors (WRITE); the IPC router can access only network sockets (READ/WRITE) and IPC endpoints (READ/WRITE); system management tasks can access only system configuration (READ) and system control (EXECUTE); and monitoring tasks can access only health metrics (READ) and audit log export (READ). Any attempt by a task to access a resource type outside its defined set MUST be denied and logged as a security event.

#### Scenario: Inference task denied access to network socket

- WHEN an inference task attempts to acquire a capability for a network socket resource
- THEN the capability system MUST deny the request
- AND MUST log a security event with event type `CAPABILITY_DENIED`, the requesting task ID, the denied resource type (network socket), and a timestamp
- AND the inference task MUST receive an error indicating insufficient privileges

#### Scenario: IPC router denied access to model resource

- WHEN the IPC router task attempts to acquire a capability for an ONNX model resource with EXECUTE permission
- THEN the capability system MUST deny the request
- AND MUST log a security event with event type `CAPABILITY_DENIED`, the requesting task ID, the denied resource type (ONNX model), and a timestamp

#### Scenario: Inference task granted access to assigned model and tensors

- WHEN an inference task requests capabilities for its assigned ONNX model (READ/EXECUTE), its input tensor buffer (READ), and its output tensor buffer (WRITE)
- THEN the capability system MUST grant all three capabilities
- AND the granted capabilities MUST be scoped to exactly the specified resources and permissions (no additional resources or permissions)

#### Scenario: Cross-type access denied by default

- WHEN a new task type is registered with the capability system without an explicit resource access policy
- THEN the capability system MUST assign an empty capability set to the task
- AND all resource access attempts by the task MUST be denied until an explicit policy is configured
- AND each denial MUST be logged as a security event

#### Scenario: Information flow policy is documented and auditable

- WHEN the information flow enforcement policy is reviewed
- THEN the security model documentation MUST contain a task-type-to-resource-type access matrix listing every task type, every resource type, and the permitted operations (READ, WRITE, EXECUTE, GRANT, or DENY)
- AND the access matrix MUST be traceable to the capability policy implementation in `security/src/policy.rs`

### Requirement: Vulnerability Disclosure Policy

The security model SHALL include a documented vulnerability disclosure policy defining the process for reporting, triaging, and remediating security vulnerabilities in SmallAIOS. The policy MUST define: a reporting channel (security-specific contact and process), an acknowledgment SLA of 48 hours from initial report, severity classification criteria aligned with CVSS v4.0, triage SLAs per severity level, remediation SLAs per severity level, and a coordinated disclosure timeline. The severity-based SLAs SHALL be: Critical (CVSS >= 9.0) triaged within 24 hours and remediated within 7 days; High (CVSS 7.0-8.9) triaged within 48 hours and remediated within 30 days; Medium (CVSS 4.0-6.9) triaged within 5 business days and remediated within 90 days; Low (CVSS < 4.0) triaged within 10 business days and remediated within 180 days.

#### Scenario: Security vulnerability reported via disclosure channel

- WHEN a security researcher or user reports a vulnerability through the designated reporting channel
- THEN the project MUST acknowledge receipt of the report within 48 hours
- AND the acknowledgment MUST include a tracking identifier and the name or alias of the assigned triage lead

#### Scenario: Critical vulnerability triage and remediation

- WHEN a reported vulnerability is classified as Critical (CVSS >= 9.0)
- THEN the vulnerability MUST be triaged (root cause identified, affected components enumerated, exploitability assessed) within 24 hours of the report
- AND a remediation (patch, mitigation, or workaround) MUST be developed, tested, and released within 7 calendar days of the report
- AND the remediation MUST include regression tests that verify the vulnerability is no longer exploitable

#### Scenario: High severity vulnerability triage and remediation

- WHEN a reported vulnerability is classified as High (CVSS 7.0-8.9)
- THEN the vulnerability MUST be triaged within 48 hours of the report
- AND a remediation MUST be released within 30 calendar days of the report
- AND the remediation MUST include regression tests and updated threat model documentation if the vulnerability reveals a previously unidentified threat

#### Scenario: Medium severity vulnerability triage and remediation

- WHEN a reported vulnerability is classified as Medium (CVSS 4.0-6.9)
- THEN the vulnerability MUST be triaged within 5 business days of the report
- AND a remediation MUST be released within 90 calendar days of the report

#### Scenario: Low severity vulnerability triage and remediation

- WHEN a reported vulnerability is classified as Low (CVSS < 4.0)
- THEN the vulnerability MUST be triaged within 10 business days of the report
- AND a remediation MUST be released within 180 calendar days of the report

#### Scenario: Coordinated disclosure timeline

- WHEN a vulnerability has been reported and a remediation is available
- THEN the project MUST coordinate with the reporter on a public disclosure date
- AND the default coordinated disclosure timeline MUST be 90 days from initial report or 30 days after the remediation is released, whichever comes first
- AND the disclosure MUST include a CVE identifier (if applicable), affected versions, remediation instructions, and credit to the reporter (unless the reporter requests anonymity)

#### Scenario: Vulnerability disclosure policy is publicly documented

- WHEN a potential reporter seeks to report a security vulnerability
- THEN the project MUST maintain a publicly accessible vulnerability disclosure policy document (e.g., SECURITY.md or equivalent)
- AND the document MUST specify the reporting channel, expected response times per severity, the coordinated disclosure process, and the scope of covered components (all SmallAIOS crates and build infrastructure)
