## ADDED Requirements

### Requirement: NIST SP 800-53 control mapping
The documentation site SHALL include a traceability matrix mapping relevant NIST SP 800-53 Rev 5 security controls to SmallAIOS implementations, using sphinx-needs directives.

#### Scenario: Access Control (AC) family mapped
- **WHEN** the NIST traceability page is rendered
- **THEN** AC family controls relevant to an OS kernel (AC-3 Access Enforcement, AC-6 Least Privilege, AC-17 Remote Access) SHALL be mapped to implementing crates/modules with status indicators

#### Scenario: System and Communications Protection (SC) family mapped
- **WHEN** the SC traceability page is rendered
- **THEN** SC controls (SC-8 Transmission Confidentiality, SC-12 Cryptographic Key Management, SC-13 Cryptographic Protection, SC-28 Protection at Rest) SHALL be mapped to the security and net crates

#### Scenario: Traceability matrix is navigable
- **WHEN** a user views the NIST traceability section on the docs site
- **THEN** each control SHALL link to its implementing requirement and each requirement SHALL link back to its NIST control

### Requirement: sphinx-needs requirement IDs
All functional requirements in the documentation SHALL have unique sphinx-needs IDs that can be cross-referenced from design documents, test cases, and NIST control mappings.

#### Scenario: Requirement cross-reference works
- **WHEN** a sphinx-needs directive references a requirement ID
- **THEN** the rendered page SHALL show a clickable link to the requirement definition
