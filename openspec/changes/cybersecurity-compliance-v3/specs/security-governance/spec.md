# Delta for Security Governance

## ADDED Requirements

### Requirement: Organizational Roles and Responsibilities
System documentation SHALL define organizational roles and responsibilities for security governance, including Security Lead, Change Control Board, and Incident Commander.

#### Scenario: Security Lead role definition
- WHEN the security governance documentation is reviewed
- THEN it MUST define the Security Lead role with responsibilities including security policy ownership, risk assessment coordination, compliance oversight, and authority to approve or reject security-relevant changes
- AND the Security Lead MUST be identified as the primary point of contact for all security governance activities

#### Scenario: Change Control Board role definition
- WHEN a security-relevant change is proposed
- THEN the documentation MUST define the Change Control Board (CCB) composition, including required members (Security Lead, system architect, test lead, domain expert) and quorum requirements
- AND the CCB MUST have documented authority to approve, reject, or defer changes to safety-critical and security-critical code

#### Scenario: Incident Commander role definition
- WHEN a security incident is declared
- THEN the documentation MUST define the Incident Commander role with responsibilities including incident classification, resource coordination, communication management, and post-incident review initiation
- AND the Incident Commander MUST have authority to invoke emergency procedures including capability revocation, task termination, and network isolation

#### Scenario: Role assignment and succession
- WHEN organizational roles are assigned
- THEN the documentation MUST specify at least one backup for each critical role (Security Lead, Incident Commander)
- AND succession procedures MUST be documented for cases where the primary role holder is unavailable

### Requirement: Risk Strategy Documentation
The system SHALL document risk strategy including risk appetite, risk tolerance thresholds, and residual risk acceptance criteria.

#### Scenario: Risk appetite statement
- WHEN stakeholders review the risk strategy
- THEN the documentation MUST include a risk appetite statement that defines the organization's willingness to accept risk across categories: security risk, safety risk, operational risk, and compliance risk
- AND the risk appetite MUST be expressed in qualitative terms (e.g., Low, Moderate, High) with supporting rationale

#### Scenario: Risk tolerance thresholds
- WHEN a risk assessment identifies a specific threat
- THEN the risk strategy MUST define quantitative or semi-quantitative tolerance thresholds including maximum acceptable CVSS score for unmitigated vulnerabilities (e.g., no unmitigated vulnerabilities with CVSS >= 9.0), maximum acceptable mean time to remediate by severity, and maximum acceptable residual risk score per component
- AND any risk exceeding tolerance thresholds MUST trigger escalation to the Security Lead

#### Scenario: Residual risk acceptance criteria
- WHEN a control implementation reduces but does not eliminate a risk
- THEN the residual risk acceptance criteria MUST require documented justification, risk owner sign-off, a review date, and compensating controls if the residual risk exceeds the defined tolerance threshold
- AND residual risk acceptance MUST be reviewed at least annually

### Requirement: Policy Lifecycle Management
System documentation SHALL define a policy lifecycle (draft, review, approve, publish, retire) for all security policies.

#### Scenario: New policy creation follows lifecycle stages
- WHEN a new security policy is created
- THEN it MUST progress through all lifecycle stages in order: Draft, Review, Approve, Publish
- AND each stage transition MUST be recorded with the date, responsible party, and any review comments or approval signatures

#### Scenario: Policy review and update
- WHEN a published security policy reaches its scheduled review date
- THEN the policy MUST re-enter the Review stage
- AND the review MUST assess the policy against current threats, regulatory changes, and lessons learned from incidents
- AND the reviewer MUST either approve the policy as-is, recommend updates (returning to Draft), or recommend retirement

#### Scenario: Policy retirement
- WHEN a security policy is no longer applicable
- THEN it MUST transition to the Retire stage with a documented justification, effective retirement date, and reference to any superseding policy
- AND retired policies MUST be archived and remain accessible for audit purposes for a minimum retention period defined in the records management policy

#### Scenario: Version control for policies
- WHEN a security policy is updated
- THEN the policy document MUST maintain a version history with version number, date, author, and summary of changes
- AND the current effective version MUST be clearly identified

### Requirement: Data Classification Policy
The system SHALL maintain a data classification policy with at least three levels (Public, Internal, Restricted) mapping to model data, inference I/O, audit logs, configuration, and cryptographic keys.

#### Scenario: Classify model data
- WHEN ONNX model files are stored or transmitted within SmallAIOS
- THEN the data classification policy MUST classify model data at the Restricted level by default
- AND the policy MUST specify handling requirements: encrypted at rest, encrypted in transit (TLS 1.3 or Zenoh encrypted sessions), access controlled via capabilities

#### Scenario: Classify inference I/O data
- WHEN inference input and output data flows through the system
- THEN the data classification policy MUST classify inference I/O at a minimum of Internal level
- AND the policy MUST define handling requirements based on the sensitivity of the inference domain (e.g., medical inference output classified as Restricted)

#### Scenario: Classify audit logs
- WHEN security audit log entries are generated
- THEN the data classification policy MUST classify audit logs at the Internal level at minimum
- AND the policy MUST require integrity protection (ML-DSA-65 signatures) and access restrictions preventing modification or deletion by non-administrative roles

#### Scenario: Classify cryptographic keys
- WHEN cryptographic keys (ML-KEM-768, ML-DSA-65) are generated or stored
- THEN the data classification policy MUST classify all cryptographic key material at the Restricted level
- AND the policy MUST mandate that keys are never stored in plaintext outside a hardware security boundary, never logged, and are subject to the key management lifecycle (generation, distribution, rotation, destruction)

#### Scenario: Classify configuration data
- WHEN system configuration files or runtime parameters are managed
- THEN the data classification policy MUST classify configuration data at the Internal level at minimum
- AND configuration data that controls security-relevant behavior (e.g., capability definitions, crypto parameters, network policies) MUST be classified as Restricted

### Requirement: Security Steering Committee Charter
The system SHALL document the security steering committee charter including meeting cadence, quorum requirements, and decision authority.

#### Scenario: Charter defines meeting cadence
- WHEN the security steering committee charter is established
- THEN it MUST specify a regular meeting cadence of at least monthly
- AND the charter MUST define procedures for calling emergency meetings in response to critical security events (e.g., active exploitation, zero-day vulnerability disclosure)

#### Scenario: Charter defines quorum requirements
- WHEN a security steering committee meeting is convened
- THEN the charter MUST define quorum as a minimum number or percentage of voting members (e.g., majority of designated members including the Security Lead or delegate)
- AND decisions made without quorum MUST be ratified at the next quorate meeting

#### Scenario: Charter defines decision authority
- WHEN the security steering committee makes a decision
- THEN the charter MUST define the committee's decision authority including: approval of security policies, approval of risk acceptance decisions above defined thresholds, prioritization of security remediation efforts, and allocation of security resources
- AND all decisions MUST be recorded in meeting minutes with the decision, rationale, dissenting opinions, and assigned action items

#### Scenario: Charter review and amendment
- WHEN the security steering committee charter is due for review
- THEN the charter MUST be reviewed at least annually
- AND amendments MUST require approval by the defined quorum and be documented with an effective date
