# SmallAIOS Security Governance

**Document Version:** 1.0
**Date:** 2026-02-10
**Classification:** Internal

---

## 1. Organizational Roles and Responsibilities

### Security Lead
- **Authority:** Final technical authority on security architecture decisions
- **Responsibilities:**
  - Review and approve all security-related changes (crypto, capability, audit)
  - Maintain vulnerability disclosure policy and triage SLAs
  - Conduct security impact assessments for proposed changes
  - Report security posture to steering committee quarterly
  - Own POA&M items and remediation tracking

### Change Control Board (CCB)
- **Composition:** Security Lead, Safety Engineer, Project Lead, designated reviewers
- **Authority:** Approve/reject all changes to safety-critical code paths
- **Responsibilities:**
  - Review impact assessments (security, safety, performance)
  - Approve PRs modifying: scheduler, memory management, syscall interface, capability system, cryptographic modules
  - Document approval decisions with rationale
  - Meet biweekly; emergency sessions within 4 hours of request

### Incident Commander
- **Authority:** Operational authority during active security incidents
- **Responsibilities:**
  - Coordinate containment and recovery actions
  - Activate communication procedures per severity matrix
  - Authorize emergency changes (bypassing standard CCB process)
  - Lead post-incident reviews
  - Report incidents to steering committee

### Verification Engineer
- **Authority:** Approve formal verification artifacts and test coverage reports
- **Responsibilities:**
  - Maintain TLA+, Lean 4, and SPIN formal models
  - Verify MC/DC coverage meets 100% on safety-critical paths
  - Review and approve test plans for new features
  - Validate requirements traceability matrix completeness

---

## 2. Risk Strategy

### Risk Appetite Statement
SmallAIOS accepts minimal residual risk for safety-critical deployments and moderate
residual risk for datacenter/edge deployments, commensurate with the threat landscape
and deployment context.

### Risk Tolerance Thresholds

| Domain | Risk Tolerance | Rationale |
|--------|---------------|-----------|
| Datacenter | Moderate | Standard enterprise risk; compensating controls available (network segmentation, monitoring) |
| Edge | Moderate-Low | Limited physical security; reduced monitoring capability; model theft risk |
| Safety-Critical (Avionics) | Very Low | DO-178C DAL A; loss of life potential; formal verification required |
| Safety-Critical (Automotive) | Very Low | ISO 26262 ASIL D; loss of life potential; WCET bounds required |
| Safety-Critical (Industrial) | Low | IEC 61508 SIL 3; equipment damage potential; fail-safe states required |

### Residual Risk Acceptance Criteria
- **Critical:** Not acceptable; must be remediated before deployment
- **High:** Acceptable only with documented compensating controls and CCB approval
- **Medium:** Acceptable with POA&M entry and quarterly review
- **Low:** Acceptable; tracked in security metrics

---

## 3. Data Classification Policy

### Classification Levels

| Level | Description | Handling Requirements |
|-------|-------------|----------------------|
| **Restricted** | Compromise would cause severe harm | Encryption at rest and in transit; access logged; key rotation; zeroization on decommission |
| **Internal** | Compromise would cause moderate harm | Encryption in transit; access controlled by capability system; audit logged |
| **Public** | No harm from disclosure | No encryption required; integrity protection recommended |

### Data Type Classification

| Data Type | Classification | Justification | Protection Mechanism |
|-----------|---------------|---------------|---------------------|
| Cryptographic keys | Restricted | Key compromise enables impersonation, decryption | Memory-only storage; volatile-write zeroization; CSPRNG generation |
| Model weights | Restricted | Intellectual property; model theft enables adversarial attacks | TLS in transit; signature verification; OCI registry access control |
| Inference I/O | Internal | May contain sensitive application data | TLS 1.3 in transit; capability-gated access; audit logged |
| Audit logs | Internal | Contain security-relevant events; integrity-critical | SHA-3 hash chain; ML-DSA-65 signatures; configurable retention |
| Configuration | Internal | Defines security posture; unauthorized change = policy bypass | Version-controlled; change-gated; audit logged |
| Health metrics | Public | Operational data; no sensitive content | Integrity-protected (Prometheus/Zenoh); no encryption required |

---

## 4. Policy Lifecycle Process

### Stages

1. **Draft:** Author creates policy document in `docs/security/`; assigns version 0.x
2. **Review:** Security Lead and at least one CCB member review; comments tracked in PR
3. **Approve:** CCB votes to approve (quorum required); approval recorded in PR
4. **Publish:** Merged to main branch; version incremented to x.0; changelog updated
5. **Retire:** When superseded, document is marked deprecated with pointer to replacement; retained in git history for audit

### Versioning
- Major version (x.0): Breaking changes, new requirements, or structural reorganization
- Minor version (x.y): Clarifications, additional guidance, non-breaking updates
- All versions tracked in git with commit history providing full audit trail

---

## 5. Security Steering Committee Charter

### Purpose
Provide strategic oversight of SmallAIOS security program, review risk posture, and approve security investment priorities.

### Membership
- Security Lead (permanent)
- Project Lead (permanent)
- Safety Engineer (permanent)
- Designated organizational stakeholder (rotating)

### Cadence
- Quarterly meetings (aligned with POA&M review cycle)
- Emergency sessions convened by Security Lead or Incident Commander

### Quorum
- Minimum 3 members, including Security Lead

### Decision Authority
- Approve risk tolerance changes
- Approve POA&M item closure
- Approve security investment priorities
- Escalation authority: organizational executive leadership

### Escalation Path
1. Security Lead escalates to steering committee
2. Steering committee escalates to organizational leadership
3. For safety-critical deployments: escalation to certification authority (DER/DAR)

---

## 6. NIST CSF 2.0 Function Coverage Matrix

| CSF Function | SmallAIOS Capabilities | Key Components |
|-------------|----------------------|----------------|
| **GOVERN** | Security governance documentation; roles and responsibilities; risk strategy; data classification; policy lifecycle | This document; `docs/security/` |
| **IDENTIFY** | Asset inventory (SBOM); trust boundary documentation; attack surface inventory; vulnerability scanning | `security/src/supply_chain/sbom.rs`; `security/src/boundary/` |
| **PROTECT** | Capability-based access control; PQC crypto; information flow enforcement; WCET bounds; fail-safe states | `security/src/capability.rs`; `security/src/crypto/`; `security/src/ot/` |
| **DETECT** | Continuous monitoring (denial rate, latency anomaly, SYN flood); OT anomaly detection; cargo-audit CI | `security/src/monitoring/`; `security/src/ot/anomaly.rs` |
| **RESPOND** | Automated containment (capability revocation, task termination); incident event publishing; evidence preservation | `security/src/incident/` |
| **RECOVER** | Watchdog reset; container restart; K8s pod rescheduling; model reload from OCI registry; configuration restore | `docs/security/contingency-plan.md` |
