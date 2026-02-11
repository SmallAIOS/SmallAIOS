# SmallAIOS Incident Response Plan

**Per NIST SP 800-61 Rev 2**

**Document Version:** 1.0
**Date:** 2026-02-10
**Classification:** Internal

---

## 1. Preparation

### 1.1 Team Structure
- **Incident Commander:** Activates and leads response; authorized to invoke emergency change procedures
- **Security Lead:** Technical analysis, containment decisions, forensic coordination
- **Verification Engineer:** Evidence preservation, audit log integrity verification
- **Communications Lead:** Internal/external notifications per severity matrix

### 1.2 Tools and Resources
- Audit log export tools: `smallaios/v1/audit` Zenoh subscription
- Evidence preservation: `security::incident::evidence` module (audit logs, task snapshots, memory stats, capability registry)
- Containment APIs: `security::incident::containment` (capability revocation, task termination, connection reset)
- Monitoring dashboards: Prometheus `/metrics` endpoint, Grafana integration

### 1.3 Communication Channels
- Primary: Secure internal messaging (Zenoh IPC for automated, out-of-band for human)
- Secondary: Email with PGP encryption for external parties
- Emergency: Phone tree for Critical severity incidents

### 1.4 Incident Classification

| Severity | Criteria | Examples | Response Time |
|----------|----------|----------|---------------|
| Critical | System compromise, data breach, safety-critical failure | Capability bypass allowing unauthorized access; crypto key exposure; safety system failure | Immediate (< 15 min) |
| High | Capability bypass, sustained DoS, integrity violation | Audit log tamper detected; persistent denial of service; unauthorized model loading | < 1 hour |
| Medium | Anomaly trigger, repeated auth failure, configuration drift | Monitoring threshold exceeded; multiple failed capability checks from single task | < 4 hours |
| Low | Configuration warning, threshold approach, minor anomaly | Latency p99 approaching bound; disk usage warning; non-critical test failure | < 24 hours |

---

## 2. Detection and Analysis

### 2.1 Detection Sources
- **Automated monitoring:** Capability denial rate tracker, memory allocation failure tracker, inference latency anomaly detector (3-sigma), SYN flood detector, watchdog timer
- **Audit log analysis:** Hash chain integrity verification, unexpected event patterns
- **CI/CD pipeline:** cargo-audit vulnerability alerts, test failures, coverage drops
- **External reports:** Vulnerability disclosure inbox per published policy

### 2.2 Analysis Procedures
1. **Triage:** Classify severity per Section 1.4 criteria
2. **Scope:** Identify affected components (kernel, security, ONNX-rt, network, IPC)
3. **Impact:** Assess data exposure, system availability, safety implications
4. **Root cause:** Correlate audit logs, monitoring metrics, and system state
5. **Attribution:** Determine whether incident is accidental, adversarial, or environmental

### 2.3 Indicators of Compromise
- Capability denial rate exceeding 10/sec from a single task
- Audit log hash chain discontinuity
- Unexpected task creation without corresponding capability grant
- Memory allocation pattern inconsistent with loaded model profile
- Network connections to unauthorized endpoints

---

## 3. Containment, Eradication, and Recovery

### 3.1 Containment Actions (Automated)

| Action | Trigger | Implementation |
|--------|---------|---------------|
| Capability revocation | Compromised task detected | `containment::revoke_task_capabilities(task_id)` |
| Task termination | Unrecoverable compromise | `containment::terminate_task(task_id)` |
| Network connection reset | Unauthorized connection | `containment::reset_connections(task_id)` |
| Inference rejection | Model integrity failure | `containment::reject_inference()` |

### 3.2 Containment Actions (Manual)

| Severity | Actions |
|----------|---------|
| Critical | Isolate affected system from network; preserve full system state; invoke emergency CCB |
| High | Revoke affected capabilities; restart affected services; increase monitoring granularity |
| Medium | Adjust monitoring thresholds; review recent configuration changes; schedule CCB review |
| Low | Log incident; adjust alerting if false positive; continue normal operations |

### 3.3 Eradication
1. Identify and remove root cause (malicious task, misconfiguration, vulnerable component)
2. Apply corrective patch through standard (or emergency) change process
3. Verify patch via test suite and formal verification (if applicable)
4. Update SBOM if dependencies changed

### 3.4 Recovery
1. Restore from known-good baseline (git tag + OCI image)
2. Verify system integrity: audit log chain, capability registry, model signature
3. Resume normal operations with increased monitoring (24-48 hours)
4. Confirm RTO met per deployment class

---

## 4. Post-Incident Activity

### 4.1 Post-Incident Review
- **Timeline:** Within 5 business days of incident closure
- **Participants:** Incident Commander, Security Lead, affected team members
- **Output:** Post-incident report using template

### 4.2 Post-Incident Report Template

```
## Incident Report: [ID]
- **Date/Time:** [Detection timestamp]
- **Duration:** [Time to resolution]
- **Severity:** [Critical/High/Medium/Low]
- **Affected Components:** [List]
- **Detection Method:** [How was it found?]

### Root Cause Analysis
[Five-whys or fishbone analysis]

### Impact Assessment
- Data exposure: [None/Limited/Significant]
- Availability: [None/Degraded/Total outage]
- Safety: [No impact/Degraded/Hazardous]

### Response Timeline
1. [Timestamp] - Detection
2. [Timestamp] - Triage and classification
3. [Timestamp] - Containment actions
4. [Timestamp] - Eradication
5. [Timestamp] - Recovery verified

### Corrective Actions
| Action | Responsible | Deadline | Status |
|--------|-------------|----------|--------|

### Lessons Learned
[What worked, what didn't, what to change]
```

### 4.3 Corrective Action Tracking
- All corrective actions entered into POA&M
- Tracked to completion by Security Lead
- Reviewed at next steering committee meeting

---

## 5. Communication Procedures

### Notification Matrix

| Severity | Internal Notification | External Notification | Timeline |
|----------|----------------------|----------------------|----------|
| Critical | All team + steering committee + organizational leadership | Affected customers, CERT (if applicable), law enforcement (if applicable) | Immediate |
| High | Security Lead + CCB + steering committee | Affected customers (if data impacted) | Within 4 hours |
| Medium | Security Lead + relevant team members | None unless escalated | Within 24 hours |
| Low | Security Lead (logged only) | None | Next business day |

### Escalation Timelines
- If no response from Incident Commander within 15 minutes: escalate to Security Lead
- If no containment within 1 hour (Critical): escalate to organizational leadership
- If root cause not identified within 24 hours (Critical/High): engage external security consultant

### External Reporting Requirements
- **Vulnerability disclosure:** Per published policy (Critical: 24h triage, High: 72h, Medium: 7d, Low: 30d)
- **Regulatory:** As required by deployment domain (aviation: CERT-AV, automotive: AUTO-ISAC, industrial: ICS-CERT)
- **Customer notification:** Within 72 hours if customer data potentially exposed (GDPR compliance)
