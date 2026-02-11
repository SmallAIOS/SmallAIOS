# SmallAIOS System Security Plan (SSP)

## NIST SP 800-53 Rev 5 Control Mapping

**Document Version:** 1.0
**Date:** 2026-02-10
**Classification:** Internal
**System Name:** SmallAIOS — Minimal Secure OS for AI Inference

---

## 1. System Boundary Description

SmallAIOS is a unikernel operating system purpose-built for AI inference workloads.
It operates in a single address space with capability-based access control.

### Components

| Component | Crate | Purpose |
|-----------|-------|---------|
| Kernel | `smallaios-kernel` | Memory management, scheduler, syscall interface |
| Security | `smallaios-security` | Capability system, PQC crypto, audit, monitoring |
| ONNX Runtime | `smallaios-onnx-rt` | ONNX model parser and inference execution |
| IPC | `smallaios-ipc` | Zenoh-inspired pub/sub messaging |
| Network | `smallaios-net` | IPv4/IPv6, TCP/UDP native network stack |
| POSIX | `smallaios-posix` | Minimal POSIX compatibility layer |
| Container | `smallaios-container` | Entry point, configuration, health, metrics |
| Arch/x86_64 | `smallaios-arch-x86_64` | x86-64 HAL: boot, GDT, IDT, APIC, paging |
| Arch/aarch64 | `smallaios-arch-aarch64` | ARM64 HAL: boot, GICv3, paging, SVE, PSCI |
| Arch/nvidia | `smallaios-arch-nvidia` | GPU HAL: PCIe, GPU init, compute, DMA |

### External Interfaces

- **Zenoh IPC:** Internal pub/sub messaging (`smallaios/v1/*` key expressions)
- **Network:** TLS 1.3 with PQ hybrid key exchange (inference API, metrics, health)
- **GPU DMA:** PCIe BAR-mapped memory for tensor transfer
- **Bus Protocols:** CAN, ARINC 429, ARINC 664, MIL-STD-1553, SpaceWire, CCSDS

---

## 2. Control Family Mapping

### AC — Access Control

| Control | Title | Status | SmallAIOS Mechanism | Evidence |
|---------|-------|--------|---------------------|----------|
| AC-2 | Account Management | Implemented | Task-based identity; no user accounts; tasks receive capabilities at creation | `security/src/capability.rs` |
| AC-3 | Access Enforcement | Implemented | Capability-based access control; all resource access requires valid capability token | `security/src/capability.rs`, TLA+ `CapabilitySecurity.tla` |
| AC-4 | Information Flow Enforcement | Implemented | Per-task-type resource access matrix; ONNX runtime cannot access network | `security/src/boundary/data_flow_auth.rs`, Lean 4 `InformationFlow.lean` |
| AC-5 | Separation of Duties | Implemented | Task types (SYSTEM, IPC, INFERENCE) have disjoint privilege sets | `security/src/boundary/` |
| AC-6 | Least Privilege | Implemented | Capabilities grant minimal permissions; delegation cannot escalate | `security/src/capability.rs` |
| AC-7 | Unsuccessful Logon Attempts | Implemented | Capability denial rate tracking with configurable alert threshold | `security/src/monitoring/rate_tracker.rs` |
| AC-10 | Concurrent Session Control | Implemented | Maximum connection limits per trust boundary (Network: 256, K8s: 16) | `security/src/boundary/trust_boundaries.rs` |
| AC-12 | Session Termination | Implemented | Task timeout via watchdog; idle connection cleanup | `kernel/src/scheduler/` |
| AC-14 | Permitted Actions Without Identification | N/A | All actions require capability tokens; no anonymous access | — |

### AT — Awareness and Training

| Control | Title | Status | Mechanism |
|---------|-------|--------|-----------|
| AT-1 | Policy and Procedures | Planned | Security governance documentation (see Section 2.1) |
| AT-2 | Literacy Training and Awareness | Inherited | Organizational responsibility |

### AU — Audit and Accountability

| Control | Title | Status | SmallAIOS Mechanism | Evidence |
|---------|-------|--------|---------------------|----------|
| AU-2 | Event Logging | Implemented | Structured audit event taxonomy: capability, auth, resource, inference, system | `security/src/audit/taxonomy.rs` |
| AU-3 | Content of Audit Records | Implemented | Entries contain: timestamp_ns, event_type, task_id, resource_ref, operation, result, capability_id | `security/src/audit/entry.rs` |
| AU-6 | Audit Record Review, Analysis, and Reporting | Implemented | Zenoh IPC export on `smallaios/v1/audit`; Prometheus `/metrics` endpoint | `security/src/audit/ipc_export.rs`, `security/src/monitoring/prometheus.rs` |
| AU-8 | Time Stamps | Implemented | Nanosecond-precision monotonic clock timestamps | `security/src/audit/entry.rs` |
| AU-9 | Protection of Audit Information | Implemented | SHA-3-256 hash chain for tamper evidence; ML-DSA-65 batch signing | `security/src/audit/integrity.rs`, `security/src/audit/batch_signing.rs` |
| AU-10 | Non-repudiation | Implemented | ML-DSA-65 digital signatures on sealed audit batches | `security/src/audit/batch_signing.rs` |
| AU-12 | Audit Record Generation | Implemented | Batch accumulator (256 entries or 1s timeout); ring buffer for non-blocking collection | `security/src/audit/accumulator.rs` |

### CA — Security Assessment and Authorization

| Control | Title | Status | SmallAIOS Mechanism | Evidence |
|---------|-------|--------|---------------------|----------|
| CA-2 | Control Assessments | Implemented | Formal verification (TLA+, Lean 4, SPIN); 750+ unit tests in security crate | `formal/tla/*.tla`, `formal/lean4/*.lean` |
| CA-7 | Continuous Monitoring | Implemented | Capability denial rate, memory failure rate, inference latency anomaly detection | `security/src/monitoring/` |
| CA-8 | Penetration Testing | Planned | Security test categories defined: capability bypass, crypto validation, timing side-channel, resource exhaustion | `security/src/compliance/test_categories.rs` |

### CM — Configuration Management

| Control | Title | Status | SmallAIOS Mechanism | Evidence |
|---------|-------|--------|---------------------|----------|
| CM-2 | Baseline Configuration | Implemented | Git-tagged releases with SBOM; configuration management plan | `docs/security/change-control.md` |
| CM-3 | Configuration Change Control | Implemented | CCB process for safety-critical changes; PR-based workflow | `.github/workflows/ci.yml` |
| CM-5 | Access Restrictions for Change | Implemented | Branch protection; PR review required; CI gates must pass | `.github/workflows/ci.yml` |
| CM-6 | Configuration Settings | Implemented | Boot-time configurable thresholds; deployment class selection | `security/src/monitoring/alerts.rs` |
| CM-7 | Least Functionality | Implemented | ~46 syscalls (vs Linux ~450); no unnecessary services | `kernel/src/syscall/` |
| CM-8 | System Component Inventory | Implemented | CycloneDX SBOM generated for every release build | `security/src/supply_chain/sbom.rs` |
| CM-9 | Configuration Management Plan | Implemented | Documented in `docs/security/change-control.md` | — |

### CP — Contingency Planning

| Control | Title | Status | SmallAIOS Mechanism | Evidence |
|---------|-------|--------|---------------------|----------|
| CP-2 | Contingency Plan | Implemented | RTO/RPO targets per deployment class; automatic recovery procedures | `docs/security/contingency-plan.md` |
| CP-4 | Contingency Plan Testing | Planned | Quarterly recovery testing plan defined | `docs/security/contingency-plan.md` |
| CP-7 | Alternate Processing Site | Inherited | K8s pod rescheduling; container restart policies | — |
| CP-9 | System Backup | Implemented | Configuration persisted to durable storage; model state in OCI registry | — |
| CP-10 | System Recovery and Reconstitution | Implemented | Watchdog reset (bare-metal), container restart (Docker), pod rescheduling (K8s) | — |

### IA — Identification and Authentication

| Control | Title | Status | SmallAIOS Mechanism | Evidence |
|---------|-------|--------|---------------------|----------|
| IA-2 | Identification and Authentication | Implemented | Mutual TLS 1.3 with PQ hybrid key exchange for external connections | `security/src/crypto/` |
| IA-5 | Authenticator Management | Implemented | Key lifecycle: boot-time generation, memory-only storage, reboot rotation, zeroization | `security/src/crypto/key_manager.rs` |
| IA-7 | Cryptographic Module Authentication | Implemented | FIPS 140-3 Level 1 crypto module boundary defined; KATs at boot | `security/src/crypto/verify.rs` |

### IR — Incident Response

| Control | Title | Status | SmallAIOS Mechanism | Evidence |
|---------|-------|--------|---------------------|----------|
| IR-1 | Policy and Procedures | Implemented | Incident response plan per NIST SP 800-61 | `docs/security/incident-response-plan.md` |
| IR-4 | Incident Handling | Implemented | Automated containment: capability revocation, task termination, connection reset | `security/src/incident/containment.rs` |
| IR-5 | Incident Monitoring | Implemented | Incident event publishing on `smallaios/v1/incidents` | `security/src/incident/event.rs` |
| IR-6 | Incident Reporting | Implemented | Communication procedures with notification matrix per severity | `security/src/incident/communication.rs` |
| IR-8 | Incident Response Plan | Implemented | Full plan with preparation, detection, containment, recovery, post-incident phases | `docs/security/incident-response-plan.md` |

### MA — Maintenance

| Control | Title | Status | Mechanism |
|---------|-------|--------|-----------|
| MA-2 | Controlled Maintenance | Inherited | Handled at organizational/facility level |
| MA-4 | Nonlocal Maintenance | N/A | No remote maintenance interface; updates via OCI image push |

### MP — Media Protection

| Control | Title | Status | Mechanism |
|---------|-------|--------|-----------|
| MP-2 | Media Access | Inherited | Physical media access controlled at facility level |
| MP-6 | Media Sanitization | Implemented | Key zeroization via volatile writes; memory clearing at shutdown | `security/src/crypto/key_manager.rs` |

### PE — Physical and Environmental Protection

| Control | Title | Status | Mechanism |
|---------|-------|--------|-----------|
| PE-* | All PE controls | Inherited | Physical security is the responsibility of the hosting facility/organization |

### PL — Planning

| Control | Title | Status | Mechanism |
|---------|-------|--------|-----------|
| PL-1 | Policy and Procedures | Implemented | This SSP document and associated security documentation |
| PL-2 | System Security and Privacy Plans | Implemented | This document |

### PM — Program Management

| Control | Title | Status | Mechanism |
|---------|-------|--------|-----------|
| PM-1 | Information Security Program Plan | Implemented | Security governance documentation | `docs/security/security-governance.md` |
| PM-9 | Risk Management Strategy | Implemented | Risk strategy with tolerance thresholds per domain | `docs/security/security-governance.md` |

### PS — Personnel Security

| Control | Title | Status | Mechanism |
|---------|-------|--------|-----------|
| PS-* | All PS controls | Inherited | Personnel security is organizational responsibility |

### RA — Risk Assessment

| Control | Title | Status | SmallAIOS Mechanism | Evidence |
|---------|-------|--------|---------------------|----------|
| RA-3 | Risk Assessment | Implemented | Attack surface inventory per trust boundary | `security/src/boundary/attack_surface.rs` |
| RA-5 | Vulnerability Monitoring and Scanning | Implemented | cargo-audit in CI; continuous vulnerability assessment | `security/src/supply_chain/vulnerability.rs` |

### SA — System and Services Acquisition

| Control | Title | Status | SmallAIOS Mechanism | Evidence |
|---------|-------|--------|---------------------|----------|
| SA-4 | Acquisition Process | Implemented | Zero third-party code policy; clean-room development | `security/src/supply_chain/clean_room.rs` |
| SA-10 | Developer Configuration Management | Implemented | Git-based version control; CI/CD gates; SBOM | `.github/workflows/ci.yml` |
| SA-11 | Developer Testing and Evaluation | Implemented | 1,700+ tests; formal verification; MC/DC coverage on critical paths | `security/src/audit/mcdc_tests.rs` |

### SC — System and Communications Protection

| Control | Title | Status | SmallAIOS Mechanism | Evidence |
|---------|-------|--------|---------------------|----------|
| SC-7 | Boundary Protection | Implemented | Trust boundary definitions with auth mechanisms per boundary | `security/src/boundary/trust_boundaries.rs` |
| SC-8 | Transmission Confidentiality and Integrity | Implemented | TLS 1.3 with PQ hybrid (ML-KEM-768) for all network traffic | `security/src/crypto/` |
| SC-12 | Cryptographic Key Establishment and Management | Implemented | Key lifecycle: generation, storage, rotation, zeroization | `security/src/crypto/key_manager.rs` |
| SC-13 | Cryptographic Protection | Partial | SHA-3 production-quality; ML-KEM/ML-DSA/AES-GCM are stubs awaiting full implementation | `security/src/crypto/` |
| SC-23 | Session Authenticity | Implemented | Mutual TLS for all external sessions | — |
| SC-28 | Protection of Information at Rest | Planned | Model weights and configuration encryption at rest | — |

### SI — System and Information Integrity

| Control | Title | Status | SmallAIOS Mechanism | Evidence |
|---------|-------|--------|---------------------|----------|
| SI-2 | Flaw Remediation | Implemented | cargo-audit in CI; vulnerability disclosure policy with SLA | `security/src/compliance/vulnerability_disclosure.rs` |
| SI-4 | System Monitoring | Implemented | Continuous security monitoring: denial rate, memory failures, latency anomalies, SYN flood detection | `security/src/monitoring/` |
| SI-5 | Security Alerts, Advisories, and Directives | Implemented | Configurable alert threshold system; Prometheus + Zenoh export | `security/src/monitoring/alerts.rs` |
| SI-7 | Software, Firmware, and Information Integrity | Implemented | ONNX model signature verification; build attestation; SBOM validation | `security/src/supply_chain/attestation.rs` |
| SI-10 | Information Input Validation | Implemented | Capability token validation; protocol parser bounds checking | — |
| SI-16 | Memory Protection | Implemented | NX bit, SMEP, SMAP (x86); PAN, PXN (ARM64); guard pages | — |

### SR — Supply Chain Risk Management

| Control | Title | Status | SmallAIOS Mechanism | Evidence |
|---------|-------|--------|---------------------|----------|
| SR-2 | Supply Chain Risk Management Plan | Implemented | Clean-room policy; zero third-party code; vendor assessment | `security/src/supply_chain/` |
| SR-3 | Supply Chain Controls and Processes | Implemented | CycloneDX SBOM; reproducible builds; build attestation | `security/src/supply_chain/sbom.rs`, `security/src/supply_chain/attestation.rs` |
| SR-4 | Provenance | Implemented | Build attestation with ML-DSA-65 signature over binary/SBOM/toolchain hash | `security/src/supply_chain/attestation.rs` |
| SR-11 | Component Authenticity | Implemented | Vendor assessment checklist for hardware components | `security/src/supply_chain/vendor.rs` |

### PT — Personally Identifiable Information Processing and Transparency

| Control | Title | Status | Mechanism |
|---------|-------|--------|-----------|
| PT-* | All PT controls | N/A | SmallAIOS does not process PII; inference workloads are application-defined |

---

## 3. Inherited Controls per Deployment Mode

### Bare-Metal

| Control Area | Status | Justification |
|-------------|--------|---------------|
| PE (Physical) | Inherited from hosting facility | SmallAIOS runs directly on hardware; physical security is facility responsibility |
| PS (Personnel) | Inherited from organization | Personnel vetting is organizational responsibility |
| All other controls | Implemented by SmallAIOS | No underlying OS to inherit from |

### Container (Docker/Podman)

| Control Area | Status | Inherited From |
|-------------|--------|----------------|
| PE (Physical) | Inherited | Hosting facility |
| PS (Personnel) | Inherited | Organization |
| Network isolation | Inherited | Container runtime (Docker network, iptables) |
| Resource limits | Inherited | Container runtime (cgroups, ulimits) |
| Host audit logging | Inherited | Host OS auditd/journald |
| AC (host-level) | Inherited | Host OS user/group model |

### Kubernetes (K8s/K3s)

| Control Area | Status | Inherited From |
|-------------|--------|----------------|
| PE (Physical) | Inherited | Hosting facility / cloud provider |
| PS (Personnel) | Inherited | Organization |
| AC (cluster RBAC) | Inherited | Kubernetes RBAC, ServiceAccount tokens |
| Network isolation | Inherited | Kubernetes NetworkPolicy, CNI plugin |
| Pod security | Inherited | Kubernetes PodSecurityStandards (Restricted) |
| Audit logging (API) | Inherited | Kubernetes API server audit logging |
| Secrets management | Inherited | Kubernetes Secrets / external vault |
| Node security | Inherited | Node OS hardening (CIS benchmark) |

---

## 4. POA&M (Plan of Action and Milestones)

| ID | Control | Weakness | Remediation | Responsible | Target Date | Risk | Status |
|----|---------|----------|-------------|-------------|-------------|------|--------|
| 1 | SC-13 | Crypto stubs (AES-GCM, ML-KEM, ML-DSA) not production-ready | Complete clean-room implementations with NIST test vectors | Security Lead | Q3 2026 | High | In Progress |
| 2 | SC-28 | No encryption at rest for model weights | Implement AES-256-GCM encryption for stored model files | Security Lead | Q4 2026 | Medium | Not Started |
| 3 | CA-8 | No formal penetration testing completed | Execute pen test program per defined test categories | Security Lead | Q3 2026 | Medium | Not Started |
| 4 | CP-4 | Quarterly recovery testing not yet executed | Execute first quarterly test cycle | Ops Lead | Q2 2026 | Low | Not Started |

**Review cadence:** Quarterly (minimum), at each CCB meeting.

---

## 5. NIST 800-53 to DO-178C Cross-Reference

| NIST Control | DO-178C Objective | Shared Artifact | Gap |
|-------------|-------------------|-----------------|-----|
| CM-3 (Config Change Control) | A-2 (Configuration Management) | Git history, CCB records | NIST requires POA&M tracking; DO-178C requires Problem Reports |
| CA-2 (Control Assessments) | A-7 (Verification Process) | TLA+/Lean 4 formal models, test results | NIST CA-2 is broader than DO-178C A-7 |
| CA-7 (Continuous Monitoring) | A-7 (Verification of Verification) | Prometheus metrics, CI results | DO-178C requires structural coverage; NIST requires ongoing assessment |
| SI-7 (Integrity) | A-5 (Verification of Outputs) | MC/DC test suite, build attestation | NIST SI-7 includes software integrity; DO-178C A-5 focuses on output correctness |
| AU-2 (Event Logging) | A-4 (Traceability) | Audit logs, requirements traceability matrix | NIST AU requires real-time logging; DO-178C A-4 requires req-to-test tracing |
| AU-12 (Audit Generation) | A-4 (Traceability) | Audit log batches | Same artifacts serve both; NIST adds non-repudiation (AU-10) |
| CM-7 (Least Functionality) | A-3 (Safety Requirements) | Syscall interface (46 calls) | DO-178C requires safety analysis; NIST requires minimized attack surface |
| SA-11 (Developer Testing) | A-6 (Testing) | Unit tests (750+ in security), MC/DC coverage | DO-178C requires MC/DC 100% on DAL A paths; NIST requires thoroughness |
