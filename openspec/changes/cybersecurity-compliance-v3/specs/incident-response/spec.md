# Delta for Incident Response

## ADDED Requirements

### Requirement: Incident Response Plan per NIST SP 800-61
System documentation SHALL define an incident response plan per NIST SP 800-61 covering: preparation, detection and analysis, containment/eradication/recovery, and post-incident activity.

#### Scenario: Preparation phase documentation
- WHEN the incident response plan is reviewed
- THEN the preparation section MUST define incident response team roles and responsibilities
- AND MUST define communication channels and contact lists
- AND MUST define the tools and resources available for incident handling
- AND MUST define training requirements for incident responders

#### Scenario: Detection and analysis phase documentation
- WHEN the incident response plan is reviewed
- THEN the detection and analysis section MUST define how security events are correlated into incidents
- AND MUST define the criteria for declaring an incident
- AND MUST define the analysis procedures for determining incident scope and impact
- AND MUST reference the continuous monitoring subsystem as the primary detection source

#### Scenario: Containment, eradication, and recovery phase documentation
- WHEN the incident response plan is reviewed
- THEN the containment section MUST define automated and manual containment actions
- AND the eradication section MUST define procedures for removing the root cause
- AND the recovery section MUST define procedures for restoring normal operations
- AND MUST define criteria for determining when each phase is complete

#### Scenario: Post-incident activity documentation
- WHEN the incident response plan is reviewed
- THEN the post-incident section MUST define the timeline for conducting a post-incident review
- AND MUST define the process for documenting lessons learned
- AND MUST define how corrective actions are tracked to completion

### Requirement: Incident Severity Classification
The system SHALL classify incidents by severity: Critical (system compromise, data breach), High (capability bypass, DoS), Medium (anomaly detection trigger, failed authentication), Low (configuration warning, threshold approach).

#### Scenario: Classify a system compromise as Critical
- WHEN a security event indicates unauthorized code execution or privilege escalation
- THEN the incident MUST be classified as Critical severity
- AND the incident record MUST include the classification rationale

#### Scenario: Classify a capability bypass as High
- WHEN a security event indicates a task accessed a resource without the required capability
- THEN the incident MUST be classified as High severity

#### Scenario: Classify an anomaly detection trigger as Medium
- WHEN the continuous monitoring subsystem generates an alert for a statistical anomaly
- THEN the incident MUST be classified as Medium severity unless operator analysis escalates it

#### Scenario: Classify a configuration warning as Low
- WHEN the system detects a non-optimal but non-exploitable configuration condition
- THEN the incident MUST be classified as Low severity
- AND the incident record MUST include the specific configuration parameter involved

#### Scenario: Severity escalation
- WHEN an incident initially classified at a lower severity reveals greater impact during analysis
- THEN the incident response team MUST escalate the severity classification
- AND the escalation MUST be recorded with justification and timestamp

### Requirement: Automated Containment Actions
The system SHALL support automated containment actions: capability revocation for compromised tasks, task termination, network connection reset, inference request rejection.

#### Scenario: Revoke capabilities for a compromised task
- WHEN a task is identified as compromised by the incident response procedure
- THEN the system MUST revoke all capabilities held by the compromised task
- AND MUST emit an audit log entry for each revoked capability
- AND the task MUST NOT be able to acquire new capabilities until explicitly re-authorized

#### Scenario: Terminate a compromised task
- WHEN automated containment determines a task must be terminated
- THEN the system MUST terminate the task and release all resources held by that task
- AND MUST emit an audit log entry recording the termination and the incident ID

#### Scenario: Reset network connections
- WHEN automated containment targets a network-based attack vector
- THEN the system MUST reset all network connections associated with the affected task or source address
- AND MUST emit an audit log entry for each connection reset

#### Scenario: Reject inference requests during containment
- WHEN the system is in an active containment state for a specific task
- THEN the system MUST reject any new inference requests from that task
- AND MUST return an error indicating the task is under containment
- AND MUST emit an audit log entry for each rejected request

### Requirement: Evidence Preservation on Incident
The system SHALL preserve evidence on incident: export audit log batches, capture current task state, export memory allocation statistics, export capability registry snapshot.

#### Scenario: Export audit log batches on incident declaration
- WHEN an incident is declared
- THEN the system MUST export all audit log batches from the retention store covering the time window relevant to the incident
- AND the exported batches MUST include their ML-DSA-65 signatures for integrity verification

#### Scenario: Capture task state on incident
- WHEN an incident is declared
- THEN the system MUST capture the current state of all tasks (running, blocked, terminated)
- AND MUST capture the capability set held by each task
- AND the captured state MUST be timestamped and included in the evidence package

#### Scenario: Export memory allocation statistics on incident
- WHEN an incident is declared
- THEN the system MUST export current memory allocation statistics including per-region usage (buddy, slab, tensor), fragmentation metrics, and recent allocation failure history

#### Scenario: Export capability registry snapshot on incident
- WHEN an incident is declared
- THEN the system MUST export a complete snapshot of the capability registry
- AND the snapshot MUST include all active capabilities, their owners, their permissions, and their creation timestamps

### Requirement: Communication Procedures
System documentation SHALL define communication procedures: who is notified at each severity level, escalation timelines, external reporting requirements.

#### Scenario: Critical severity notification
- WHEN an incident is classified as Critical
- THEN the communication procedure MUST require immediate notification (within 15 minutes) to the incident response team lead, system owner, and executive stakeholders
- AND MUST define the notification channel (e.g., out-of-band secure communication)

#### Scenario: High severity notification
- WHEN an incident is classified as High
- THEN the communication procedure MUST require notification within 1 hour to the incident response team and system owner

#### Scenario: Medium and Low severity notification
- WHEN an incident is classified as Medium or Low
- THEN the communication procedure MUST require notification within 24 hours to the incident response team
- AND Medium incidents MUST be included in the next scheduled security review

#### Scenario: Escalation timeline enforcement
- WHEN an incident remains unacknowledged beyond the defined notification timeline
- THEN the communication procedure MUST define automatic escalation to the next level of management
- AND the escalation MUST be recorded in the incident record

#### Scenario: External reporting requirements
- WHEN an incident involves a data breach or affects a safety-critical deployment
- THEN the communication procedure MUST define external reporting obligations (regulatory bodies, certification authorities, affected customers)
- AND MUST define the timeline for external notification in compliance with applicable regulations

### Requirement: Post-Incident Review Process
Post-incident review process SHALL be documented: root cause analysis template, corrective action tracking, lessons learned incorporation into security policies.

#### Scenario: Conduct root cause analysis
- WHEN an incident of Medium severity or higher is resolved
- THEN a post-incident review MUST be conducted within 5 business days
- AND the review MUST use the documented root cause analysis template
- AND the template MUST require identification of contributing factors, timeline reconstruction, and root cause determination

#### Scenario: Track corrective actions
- WHEN corrective actions are identified during post-incident review
- THEN each corrective action MUST be assigned an owner and a target completion date
- AND the status of each corrective action MUST be tracked until completion
- AND overdue corrective actions MUST be escalated per the defined escalation procedure

#### Scenario: Incorporate lessons learned
- WHEN a post-incident review is completed
- THEN the lessons learned MUST be reviewed for applicability to existing security policies
- AND applicable lessons MUST result in policy updates, configuration changes, or additional monitoring rules
- AND the incorporation MUST be documented and traceable to the originating incident

### Requirement: Automated Incident Alerting via Zenoh
The system SHALL support automated alerting: publish incident events on Zenoh key expression `smallaios/v1/incidents` with severity, description, timestamp, and affected resources.

#### Scenario: Publish incident event on declaration
- WHEN an incident is declared by the monitoring subsystem or by operator action
- THEN the system MUST publish an incident event on Zenoh key expression `smallaios/v1/incidents`
- AND the event MUST include the severity level, a human-readable description, a nanosecond-precision timestamp, and a list of affected resource identifiers

#### Scenario: Subscriber receives incident events
- WHEN an external system subscribes to Zenoh key expression `smallaios/v1/incidents`
- THEN the subscriber MUST receive all incident events in the order they were declared
- AND each event MUST be self-contained with all required fields present

#### Scenario: Incident event on severity escalation
- WHEN an incident severity is escalated
- THEN the system MUST publish an updated incident event on `smallaios/v1/incidents`
- AND the updated event MUST include the new severity, the previous severity, and the escalation reason

#### Scenario: Incident event on containment action
- WHEN an automated containment action is executed
- THEN the system MUST publish an incident event on `smallaios/v1/incidents` describing the containment action taken
- AND the event MUST include the action type, the affected task ID, and the incident ID that triggered the action
