# Delta for NIST Control Mapping

## ADDED Requirements

### Requirement: NIST SP 800-53 Rev 5 Control Family Mapping
The system SHALL maintain a documented mapping of all implemented security controls to NIST SP 800-53 Rev 5 control families (AC, AU, CA, CM, CP, IA, IR, MA, PL, RA, SA, SC, SI, PM).

#### Scenario: Verify complete control family coverage
- WHEN an auditor requests the NIST SP 800-53 Rev 5 control mapping document
- THEN the system MUST provide a mapping table that covers all 14 control families: Access Control (AC), Audit and Accountability (AU), Security Assessment and Authorization (CA), Configuration Management (CM), Contingency Planning (CP), Identification and Authentication (IA), Incident Response (IR), Maintenance (MA), Planning (PL), Risk Assessment (RA), System and Services Acquisition (SA), System and Communications Protection (SC), System and Information Integrity (SI), and Program Management (PM)
- AND each control family entry MUST list every applicable control with its identifier (e.g., AC-1, AC-2), title, and the corresponding SmallAIOS mechanism or artifact that satisfies it

#### Scenario: Trace a specific control to implementation
- WHEN an assessor selects a specific control (e.g., SC-13 Cryptographic Protection)
- THEN the mapping MUST identify the SmallAIOS component (e.g., ML-KEM-768 + ML-DSA-65 hybrid in the security crate), the source file or module, and the relevant test or formal verification artifact that demonstrates the control is satisfied

#### Scenario: Flag unmapped controls
- WHEN a NIST SP 800-53 Rev 5 control has no corresponding SmallAIOS implementation
- THEN the mapping document MUST explicitly flag the control as Not Applicable with a documented justification, or as Planned with a target milestone

### Requirement: System Security Plan Skeleton
The system SHALL provide a System Security Plan (SSP) skeleton identifying each control's implementation status as one of: Implemented, Planned, Inherited, or Not Applicable.

#### Scenario: Generate SSP with implementation status for all controls
- WHEN the SSP skeleton is generated or updated
- THEN every NIST SP 800-53 Rev 5 control listed in the mapping MUST have exactly one implementation status assigned: Implemented, Planned, Inherited, or Not Applicable
- AND each status assignment MUST include a rationale or reference to supporting evidence

#### Scenario: Validate no control is left without status
- WHEN the SSP skeleton is reviewed for completeness
- THEN there MUST be zero controls with an undefined or blank implementation status
- AND automated validation tooling MUST reject an SSP that contains controls without a valid status value

#### Scenario: SSP includes system boundary description
- WHEN the SSP skeleton is populated for a specific deployment
- THEN it MUST include a system boundary description identifying all SmallAIOS components (kernel, arch crates, onnx-rt, ipc, net, security, container) and external interfaces (Zenoh IPC, network, GPU DMA, bus protocols)

### Requirement: Inherited Controls per Deployment Mode
The system SHALL identify inherited controls for each deployment mode: bare-metal, container, and Kubernetes (K8s/K3s).

#### Scenario: Bare-metal deployment inherited controls
- WHEN SmallAIOS is deployed in bare-metal mode (unikernel directly on hardware)
- THEN the SSP MUST document that no controls are inherited from an underlying operating system
- AND physical security controls (PE family) MUST be marked as inherited from the hosting facility with the responsible party identified

#### Scenario: Container deployment inherited controls
- WHEN SmallAIOS is deployed as a container image
- THEN the SSP MUST identify controls inherited from the container runtime (e.g., network isolation, resource limits) and the host operating system (e.g., audit logging, access control)
- AND each inherited control MUST reference the responsible external system and its expected compliance posture

#### Scenario: Kubernetes deployment inherited controls
- WHEN SmallAIOS is deployed on Kubernetes or K3s via Virtual Kubelet
- THEN the SSP MUST identify controls inherited from the Kubernetes control plane (e.g., RBAC, network policies, pod security standards) and the underlying infrastructure
- AND the boundary between SmallAIOS-provided and Kubernetes-provided controls MUST be explicitly documented

#### Scenario: Deployment mode comparison matrix
- WHEN an organization evaluates SmallAIOS for multiple deployment modes
- THEN the documentation MUST provide a comparison matrix showing which controls are Implemented, Inherited, or Not Applicable for each deployment mode (bare-metal, container, K8s)

### Requirement: POA&M Template
The system SHALL provide a Plan of Action and Milestones (POA&M) template for controls not yet fully implemented.

#### Scenario: POA&M entry for a Planned control
- WHEN a control is marked as Planned in the SSP
- THEN the POA&M MUST include an entry with the control identifier, weakness description, planned remediation actions, responsible party, estimated completion date, and risk level (High, Moderate, Low)

#### Scenario: POA&M tracks milestone progress
- WHEN a POA&M entry has defined milestones
- THEN each milestone MUST have a target date, current status (Not Started, In Progress, Completed, Delayed), and a completion percentage
- AND delayed milestones MUST include an updated estimated completion date and justification for the delay

#### Scenario: POA&M review cadence
- WHEN the POA&M is active with open items
- THEN the document MUST specify a review cadence of at least quarterly
- AND each review MUST be documented with the review date, reviewer, and disposition of each open item

### Requirement: NIST 800-53 to DO-178C Cross-Reference
The system SHALL cross-reference NIST SP 800-53 controls with DO-178C DAL A objectives where overlap exists.

#### Scenario: Identify overlapping objectives between NIST 800-53 and DO-178C
- WHEN both NIST SP 800-53 compliance and DO-178C DAL A certification are required
- THEN the system documentation MUST provide a cross-reference table mapping overlapping controls (e.g., NIST CM-3 Configuration Change Control to DO-178C objective A-2 Configuration Management, NIST CA-7 Continuous Monitoring to DO-178C objective A-7 Verification of Verification Process)
- AND each mapping MUST identify whether a single artifact satisfies both frameworks or separate artifacts are required

#### Scenario: Leverage DO-178C MC/DC coverage for NIST SI controls
- WHEN DO-178C DAL A requires MC/DC 100% coverage on safety-critical code paths
- THEN the cross-reference MUST demonstrate that NIST SI-7 (Software, Firmware, and Information Integrity) is partially or fully satisfied by the same MC/DC test suite
- AND any gaps MUST be documented with supplementary verification activities

#### Scenario: Unified audit trail for dual compliance
- WHEN both NIST AU (Audit) controls and DO-178C traceability objectives apply
- THEN the cross-reference MUST identify shared artifacts (e.g., audit logs, requirements traceability matrix) that satisfy both frameworks simultaneously
- AND MUST document any additional NIST-specific audit requirements not covered by DO-178C traceability
