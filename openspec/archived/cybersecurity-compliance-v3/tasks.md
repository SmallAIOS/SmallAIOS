## 1. NIST SP 800-53 Control Mapping

- [x] 1.1 Create SSP skeleton document with all 20 control families (AC, AT, AU, CA, CM, CP, IA, IR, MA, MP, PE, PL, PM, PS, RA, SA, SC, SI, SR, PT) and implementation status per control
- [x] 1.2 Map capability-based access control to AC family (AC-2, AC-3, AC-4, AC-5, AC-6, AC-7, AC-10, AC-12, AC-14)
- [x] 1.3 Map audit logging to AU family (AU-2, AU-3, AU-6, AU-8, AU-9, AU-10, AU-12)
- [x] 1.4 Map formal verification and testing to CA family (CA-2, CA-7, CA-8)
- [x] 1.5 Map configuration management to CM family (CM-2, CM-3, CM-5, CM-6, CM-7, CM-8, CM-9)
- [x] 1.6 Map cryptography to SC family (SC-7, SC-8, SC-12, SC-13, SC-23, SC-28)
- [x] 1.7 Map system integrity to SI family (SI-2, SI-4, SI-5, SI-7, SI-10, SI-16)
- [x] 1.8 Map supply chain to SR family (SR-2, SR-3, SR-4, SR-11)
- [x] 1.9 Map remaining families (IA, IR, PL, RA, SA, CP, PM) with implementation status
- [x] 1.10 Identify inherited controls per deployment mode (bare-metal, container, K8s) with inheritance justification
- [x] 1.11 Create POA&M template for controls not yet fully implemented
- [x] 1.12 Cross-reference NIST 800-53 controls with DO-178C DAL A objectives (table showing overlap)

## 2. Security Governance (NIST CSF 2.0 GOVERN)

- [x] 2.1 Document organizational roles and responsibilities (Security Lead, CCB, Incident Commander, Verification Engineer)
- [x] 2.2 Document risk strategy: risk appetite statement, risk tolerance thresholds per domain (datacenter, edge, safety-critical), residual risk acceptance criteria
- [x] 2.3 Create data classification policy: Public, Internal, Restricted levels with mapping to model weights, inference I/O, audit logs, configuration, crypto keys
- [x] 2.4 Document policy lifecycle process (draft → review → approve → publish → retire) with versioning
- [x] 2.5 Create security steering committee charter (cadence, quorum, decision authority, escalation path)
- [x] 2.6 Document CSF 2.0 function coverage matrix: GOVERN, MANAGE, PROTECT, DETECT, RESPOND, RECOVER mapped to SmallAIOS capabilities

## 3. Tamper-Evident Audit Logging

- [x] 3.1 Define structured audit event taxonomy: capability (grant/revoke/deny), authentication (TLS success/failure), resource (allocation/exhaustion), inference (request/completion/timeout), system (boot/shutdown/watchdog)
- [x] 3.2 Implement audit log entry struct: timestamp_ns, event_type, task_id, resource_ref, operation, result, capability_id
- [x] 3.3 Implement audit batch accumulator: collect up to 256 entries or 1-second timeout, whichever comes first
- [x] 3.4 Implement batch hash chain: SHA-3-256(previous_batch_hash || serialized_entries) per batch
- [x] 3.5 Implement ML-DSA-65 batch signing: async signing after batch seal, non-blocking to inference path
- [x] 3.6 Implement signed batch export via Zenoh IPC on key expression `smallaios/v1/audit`
- [x] 3.7 Implement audit log integrity verification function: verify batch signature + hash chain continuity
- [x] 3.8 Add configurable log retention policy (7 days edge, 90 days datacenter, 1 year safety-critical)
- [x] 3.9 Write unit tests: batch accumulation, hash chain correctness, signature verification, retention policy enforcement
- [x] 3.10 Achieve 100% MC/DC on audit subsystem critical paths

## 4. Supply Chain Security

- [x] 4.1 Integrate cargo-cyclonedx into build system: generate CycloneDX SBOM for every release build
- [x] 4.2 Integrate cargo-audit into CI pipeline: fail build on CVSS >= 7.0 vulnerabilities
- [x] 4.3 Implement reproducible build verification: script that builds twice with same toolchain and compares output hashes
- [x] 4.4 Implement build attestation signing: ML-DSA-65 signature over (binary_hash, sbom_hash, toolchain_version, build_timestamp)
- [x] 4.5 Create vendor assessment checklist template for hardware components (GPU, bus transceivers, FPGA, SoC)
- [x] 4.6 Document clean-room attestation process: zero third-party code policy, development methodology
- [x] 4.7 Add SBOM as OCI image label and standalone build artifact
- [x] 4.8 Write CI integration tests: SBOM generation validates, cargo-audit runs clean, attestation signature verifiable

## 5. Incident Response

- [x] 5.1 Create incident response plan document per NIST SP 800-61: preparation, detection/analysis, containment/eradication/recovery, post-incident activity
- [x] 5.2 Define incident severity classification: Critical (system compromise, data breach), High (capability bypass, DoS), Medium (anomaly trigger, auth failure), Low (config warning, threshold approach)
- [x] 5.3 Implement automated containment actions: capability revocation for compromised tasks, task termination, network connection reset, inference rejection
- [x] 5.4 Implement evidence preservation: export audit log batches, capture task state snapshot, export memory allocation stats, export capability registry snapshot
- [x] 5.5 Implement incident event publishing on Zenoh key expression `smallaios/v1/incidents` with severity, description, timestamp, affected resources
- [x] 5.6 Document communication procedures: notification matrix per severity level, escalation timelines, external reporting requirements
- [x] 5.7 Create post-incident review template: root cause analysis, corrective action tracking, lessons learned integration
- [x] 5.8 Write unit tests: containment action execution, evidence export format, incident event serialization

## 6. Continuous Security Monitoring

- [x] 6.1 Implement capability denial rate tracker: per-second per-task counter with configurable alert threshold (default 10/sec)
- [x] 6.2 Implement memory allocation failure rate tracker with configurable alert threshold
- [x] 6.3 Implement rolling inference latency statistics: p50, p99, p999 with sliding window, 3-sigma anomaly detection
- [x] 6.4 Implement watchdog time-remaining low-watermark tracker with alert at 50% of timeout
- [x] 6.5 Implement network connection rate tracker for SYN flood detection (default threshold 100/sec)
- [x] 6.6 Export security metrics via Prometheus endpoint (`GET /metrics`) in OpenMetrics format
- [x] 6.7 Export security metrics via Zenoh IPC (`smallaios/v1/metrics`) as structured messages
- [x] 6.8 Implement configurable alert threshold system: thresholds settable at boot time via system configuration
- [x] 6.9 Integrate cargo-audit results into continuous vulnerability assessment reporting
- [x] 6.10 Write unit tests: counter accuracy, anomaly detection trigger, threshold configuration, metrics export format

## 7. Security Boundary Documentation

- [x] 7.1 Document kernel boundary: capability-protected syscalls, entry points, data formats, validation mechanisms
- [x] 7.2 Document K8s boundary: Virtual Kubelet management API, mutual TLS, data flow direction, maximum data rate
- [x] 7.3 Document network boundary: TLS 1.3 termination, stateful firewall rules, connection limits, protocol restrictions
- [x] 7.4 Document bus protocol boundaries: CAN/ARINC 429/ARINC 664/MIL-STD-1553/SpaceWire/CCSDS transport isolation via capability system
- [x] 7.5 Document GPU boundary: DMA memory restrictions, command validation, BAR mapping controls
- [x] 7.6 Define information flow enforcement rules: ONNX runtime ↛ network, IPC router ↛ GPU, bus handlers ↛ ONNX
- [x] 7.7 Create PlantUML trust domain diagrams (one per boundary) traceable to Sphinx-needs requirements
- [x] 7.8 Create attack surface inventory per boundary: entry points, data formats accepted, validation mechanisms, known limitations
- [x] 7.9 Verify all cross-boundary data flows are authenticated and integrity-protected (capability, mutual TLS, or bus framing)

## 8. OT/ICS Security Hardening

- [x] 8.1 Implement WCET instrumentation hooks for critical kernel paths: syscall dispatch, capability check, memory allocation, task scheduling, interrupt handling
- [x] 8.2 Define static WCET bounds: document no-recursion policy, enumerate all bounded loops with maximum iteration counts
- [x] 8.3 Create WCET measurement framework: N>=10000 samples per path on target hardware, p99.9 reporting
- [x] 8.4 Define fail-safe states: inference timeout → error code (no partial results), memory exhaustion → reject new allocations, watchdog timeout → controlled reset with audit flush, capability violation → deny and log
- [x] 8.5 Document each fail-safe state: trigger condition, system behavior, recovery procedure, maximum time to safe state
- [x] 8.6 Implement OT anomaly detection: model output range validation (configurable bounds per tensor), inference timing bounds (configurable per model)
- [x] 8.7 Implement safe shutdown procedure: flush audit logs → revoke all capabilities → terminate all tasks → watchdog reset (within configurable bound, default 100ms)
- [x] 8.8 Create functional safety cross-reference table: map kernel components to IEC 61508 SIL levels, ISO 26262 ASIL levels, DO-178C DAL levels
- [x] 8.9 Write unit tests: WCET measurement collection, fail-safe state transitions, OT anomaly detection triggers, safe shutdown sequence

## 9. Change Control and Configuration Management

- [x] 9.1 Create configuration management plan: baseline identification, version control policies, configuration item naming conventions
- [x] 9.2 Define change control board (CCB) process: change request → impact assessment → CCB review → approve/reject → implement → verify
- [x] 9.3 Create impact assessment template: security impact (capability/crypto changes), safety impact (scheduler/memory/syscall), performance impact (latency budget)
- [x] 9.4 Document rollback procedures for each change type: code changes (git revert), model updates (OCI image rollback), configuration changes (config version restore), firmware updates (dual-bank fallback)
- [x] 9.5 Integrate change gates into CI: all tests pass, clippy clean, formal verification models pass, MC/DC coverage >= threshold
- [x] 9.6 Document approval workflow: who approves what (safety-critical = CCB + Safety Engineer, non-safety = CCB)

## 10. Contingency Planning

- [x] 10.1 Define RTO targets per deployment class: datacenter (30s), edge (5s), safety-critical (watchdog-bounded ~100ms)
- [x] 10.2 Define RPO targets: zero data loss for configuration, model state recoverable from OCI registry
- [x] 10.3 Document automatic recovery mechanisms: watchdog reset (bare-metal), container restart (Docker), pod rescheduling (K8s/K3s)
- [x] 10.4 Document failover procedures per deployment mode with step-by-step runbooks
- [x] 10.5 Document model and configuration backup/restore procedures
- [x] 10.6 Create recovery testing plan: quarterly testing with documented results per NIST CP-4

## 11. Security Model Extensions (Spec 06 Delta)

- [x] 11.1 Add NIST control cross-references to each security mechanism in Spec 06 (capability → AC-3/AC-6, audit → AU-2/AU-12, etc.)
- [x] 11.2 Add data classification policy to Spec 06: model weights (Restricted), inference I/O (Internal), audit logs (Internal), crypto keys (Restricted), health metrics (Public)
- [x] 11.3 Add information flow enforcement rules to capability system design: define per-task-type resource access matrix
- [x] 11.4 Add vulnerability disclosure policy: reporting process, triage SLA (Critical: 24h, High: 72h, Medium: 7d, Low: 30d), public disclosure timeline

## 12. Crypto/Hardware Security Extensions (Spec 08 Delta)

- [x] 12.1 Add SC-12 control mapping to crypto spec: key generation, distribution, storage, rotation, destruction lifecycle
- [x] 12.2 Add SC-13 control mapping: algorithm selection rationale per algorithm with FIPS references
- [x] 12.3 Implement key management lifecycle: boot-time generation, memory-only storage, reboot rotation, shutdown zeroization
- [x] 12.4 Implement key zeroization using volatile writes (`core::ptr::write_volatile`) with verification pass
- [x] 12.5 Implement crypto module self-test (KATs): AES-256-GCM, SHA-3-256, ML-KEM-768, ML-DSA-65 at boot
- [x] 12.6 Define FIPS 140-3 Level 1 crypto module boundary: PlantUML diagram, algorithm enumeration, I/O specification
- [x] 12.7 Implement key usage tracking with rotation warning at 2^32 operations per key
- [x] 12.8 Write unit tests: key zeroization verification, KAT execution, key usage counter, lifecycle events

## 13. Safety-Critical Extensions (Spec 12 Delta)

- [x] 13.1 Cross-reference DO-178C verification objectives with NIST 800-53 CA controls (CA-2, CA-7, CA-8)
- [x] 13.2 Add security-specific MC/DC coverage targets: 100% on crypto paths, 100% on capability check paths
- [x] 13.3 Create OT functional safety cross-reference table: IEC 61508 SIL ↔ ISO 26262 ASIL ↔ DO-178C DAL per kernel component
- [x] 13.4 Define security test categories: capability bypass, crypto validation (NIST vectors), timing side-channel (dudect), resource exhaustion

## 14. Formal Verification Extensions

- [x] 14.1 Write TLA+ model for audit log hash chain integrity: batch ordering, no gaps, no replay
- [x] 14.2 Write Lean 4 proof for information flow enforcement: task-type → resource-type access matrix correctness
- [x] 14.3 Add anomaly detection threshold correctness to existing scheduler TLA+ model
- [x] 14.4 Verify all new TLA+ models pass TLC model checker
- [x] 14.5 Verify all new Lean 4 proofs type-check

## 15. Documentation and Verification

- [x] 15.1 Create Sphinx-needs requirement set for all cybersecurity specs (REQ → SPEC → IMPL → TEST → VERIFY)
- [x] 15.2 Create PlantUML architecture diagrams: trust domain overview, data flow, crypto module boundary, incident response workflow
- [x] 15.3 Integrate all new requirements into bidirectional traceability matrix
- [x] 15.4 Validate OpenSpec change: `openspec validate --change cybersecurity-compliance-v3`
- [x] 15.5 Final review: all tasks complete, all tests pass, all formal models verified, traceability complete
