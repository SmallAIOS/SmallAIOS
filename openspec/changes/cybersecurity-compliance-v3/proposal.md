## Why

SmallAIOS has strong technical security controls (capability-based access, post-quantum cryptography, memory safety, formal verification) but lacks the governance, operational, and compliance layers that organizations need when mapping to frameworks like NIST CSF 2.0, NIST SP 800-53, and NIST SP 800-82. Many target deployment domains (federal, defense, aviation, automotive, critical infrastructure, space) require documented NIST control mappings, supply chain transparency, incident response procedures, tamper-evident audit logging, continuous vulnerability assessment, and fail-safe definitions for OT environments. Without these, SmallAIOS cannot be adopted in organizations that mandate NIST compliance.

## What Changes

- Add NIST SP 800-53 control mapping with System Security Plan (SSP) skeleton covering all 20 control families
- Add NIST CSF 2.0 governance framework documenting organizational roles, risk strategy, and security steering
- Add tamper-evident audit logging: cryptographically signed log entries (ML-DSA-65), structured security event taxonomy, remote syslog over TLS
- Add supply chain risk management: automated SBOM generation (cargo-sbom/syft in CI), dependency audit (cargo-audit), vendor assessment process, reproducible build attestation
- Add incident response framework: IR plan, communication procedures, evidence preservation, post-incident review, automated alerting on capability denials and resource exhaustion
- Add continuous security monitoring: runtime anomaly detection (latency outliers, resource usage spikes, capability denial patterns), automated vulnerability scanning in CI, security metrics dashboard via Prometheus/Zenoh
- Add security boundary documentation: formal trust domain diagrams for kernel, K8s/Virtual Kubelet, network, bus protocol, and GPU boundaries with data flow enforcement policies
- Add OT/ICS security hardening: worst-case execution time (WCET) analysis framework, fail-safe state definitions, OT-specific anomaly detection for inference timing and model output validation
- Add change control and configuration management: formal CM plan, baseline documentation, change tracking, rollback procedures
- Add contingency planning: RTO/RPO targets per deployment class, failover procedures, backup/restore for model and configuration state
- **BREAKING**: Extend Spec 06 (Security Model) with NIST control cross-references, data classification policy, and information flow enforcement

## Capabilities

### New Capabilities

- `nist-control-mapping`: NIST SP 800-53 Rev 5 control-by-control mapping to SmallAIOS design and implementation, SSP skeleton, POA&M template, control inheritance model for K8s/container/bare-metal deployment modes
- `security-governance`: NIST CSF 2.0 GOVERN function — organizational context, risk strategy, roles and responsibilities, security steering committee charter, policy lifecycle management
- `tamper-evident-audit`: Cryptographically signed audit log entries (ML-DSA-65 per log batch), structured security event taxonomy (capability grants/revocations, auth failures, resource exhaustion, inference anomalies), remote syslog over TLS, log integrity verification, retention policies
- `supply-chain-security`: Automated SBOM generation (CycloneDX format), cargo-audit integration in CI, dependency pinning policy, clean-room attestation, reproducible build verification, vendor assessment checklist for hardware components (GPU, bus transceivers, FPGA)
- `incident-response`: Incident response plan per NIST IR-1, incident classification (severity levels), communication procedures, evidence preservation (log export, memory snapshot), containment actions (capability revocation, task termination, network isolation), post-incident review process, automated alerting triggers
- `continuous-monitoring`: Runtime security monitoring — capability denial rate tracking, resource exhaustion detection, inference latency anomaly detection (statistical outlier flagging), automated vulnerability scanning (cargo-audit, CVE database), security metrics exported via Prometheus and Zenoh IPC, configurable alert thresholds
- `security-boundaries`: Formal trust domain documentation — kernel boundary (capability system), K8s boundary (Virtual Kubelet management API), network boundary (TLS termination, firewall), bus protocol boundaries (CAN/ARINC/1553/SpaceWire/CCSDS isolation), GPU boundary (DMA restrictions), data flow enforcement policies, attack surface inventory per boundary
- `ot-security-hardening`: OT/ICS-specific security per NIST SP 800-82 — WCET analysis framework for all kernel paths, fail-safe state definitions (inference timeout → safe error, watchdog → controlled reset), OT anomaly detection (model output range validation, inference timing bounds), safe shutdown procedures, functional safety integration (IEC 61508 / ISO 26262 cross-reference)
- `change-control`: Configuration management plan — change control board (CCB) process, baseline documentation, version control policies, rollback procedures, change impact assessment, approval workflow
- `contingency-planning`: Disaster recovery and business continuity — RTO/RPO targets per deployment class (datacenter: 30s RTO, edge: 5s RTO, safety-critical: watchdog-bounded), failover procedures (K8s pod restart, bare-metal watchdog reset), model and configuration backup/restore, recovery testing procedures

### Modified Capabilities

- `06-security-model`: Extend with NIST control cross-references on each security mechanism, add data classification policy (model data, inference I/O, audit logs, configuration), add information flow enforcement rules, add vulnerability disclosure policy
- `12-safety-critical`: Extend DO-178C process with NIST SP 800-53 CA (Security Assessment) cross-references, add security-specific MC/DC coverage requirements for crypto and capability paths, add OT functional safety cross-reference (IEC 61508, ISO 26262)
- `08-crypto-hardware-security`: Extend with NIST SP 800-53 SC-12/SC-13 control mapping, add key management lifecycle (generation, distribution, rotation, destruction), add crypto module boundary definition per FIPS 140-3

## Impact

- **Rust workspace**: Extend `security` crate with audit log signing, anomaly detection module, SBOM metadata; extend `kernel` crate with WCET instrumentation hooks
- **CI/CD pipeline**: Add cargo-audit, cargo-sbom, SBOM attestation, log signing key management
- **Documentation**: New Sphinx-needs document set for SSP, IR plan, CM plan, contingency plan, security boundary diagrams (PlantUML)
- **Build system**: Reproducible build verification, SBOM generation as build artifact
- **External tooling**: Prometheus metrics endpoint extension, remote syslog receiver configuration
- **Formal verification**: Extend TLA+ models with audit log integrity invariant; add Lean 4 proofs for information flow enforcement
- **Certification**: Cross-reference DO-178C DAL A artifacts with NIST 800-53 controls for dual compliance
