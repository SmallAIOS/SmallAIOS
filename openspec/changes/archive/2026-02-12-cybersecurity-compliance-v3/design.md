## Context

SmallAIOS has strong technical security controls — capability-based access control, post-quantum cryptography (ML-KEM-768 + ML-DSA-65), Rust memory safety, formal verification (TLA+, Lean 4, SPIN), and DO-178C DAL A process discipline. However, organizations deploying in federal, defense, aviation, automotive, and critical infrastructure domains require documented compliance with NIST frameworks (CSF 2.0, SP 800-53, SP 800-82) and operational security practices that go beyond technical controls.

The current security spec (06-security-model) defines the capability system and threat model. The crypto spec (08-crypto-hardware-security) covers post-quantum algorithms. The safety-critical spec (12-safety-critical) covers DO-178C process. What's missing is the governance, operational, and compliance layer that wraps these technical controls into auditable, mappable frameworks.

Target deployments span six domains with different compliance requirements:
- **Federal/DoD**: NIST SP 800-53 Rev 5, FedRAMP, CMMC
- **Aviation**: DO-178C DAL A (already adopted), NIST CSF for operational security
- **Automotive**: ISO 26262 ASIL D, NIST CSF for connected vehicle infrastructure
- **Critical Infrastructure**: NIST SP 800-82 (ICS/OT), IEC 62443
- **Space**: CCSDS security standards, NIST CSF
- **Edge/Datacenter**: NIST CSF 2.0, SOC 2 mapping

## Goals / Non-Goals

**Goals:**
- Map every SmallAIOS security mechanism to NIST SP 800-53 Rev 5 controls
- Document NIST CSF 2.0 governance structure (GOVERN, MANAGE, PROTECT, DETECT, RESPOND, RECOVER)
- Implement tamper-evident audit logging with cryptographic signatures
- Automate supply chain transparency (SBOM, dependency audit, build attestation)
- Define incident response procedures with automated alerting triggers
- Add runtime anomaly detection for capability denials, resource exhaustion, and inference timing
- Document all trust domain boundaries with formal data flow policies
- Add OT-specific security (WCET analysis, fail-safe states, functional safety cross-references)
- Establish change control and contingency planning processes

**Non-Goals:**
- Achieving full FedRAMP ATO (requires organizational evidence beyond software)
- Implementing a SIEM or SOC — SmallAIOS exports events; external systems aggregate
- Replacing DO-178C with NIST — the frameworks are complementary, not competing
- Runtime code signing verification (handled by secure boot chain already)
- Multi-tenant isolation (SmallAIOS is a single-purpose unikernel)
- Hardware security module (HSM) integration — use software crypto with hardware acceleration

## Decisions

### Decision 1: NIST 800-53 as primary control framework, CSF 2.0 for governance wrapper

**Choice**: Map individual security mechanisms to SP 800-53 Rev 5 controls, use CSF 2.0 functions as the organizational governance layer.

**Alternatives considered**:
- CSF 2.0 only — too high-level for control-by-control audit evidence
- ISO 27001 — less common in US federal/defense deployments
- CIS Controls — operational checklist but not a compliance framework

**Rationale**: 800-53 is the mandatory framework for federal systems (FISMA), and most other frameworks (FedRAMP, CMMC, NIST CSF) map directly to 800-53 controls. Organizations that need ISO 27001 or SOC 2 can derive mappings from our 800-53 baseline.

### Decision 2: Audit log signing at batch level with ML-DSA-65

**Choice**: Sign audit log batches (not individual entries) using ML-DSA-65 with a hash chain linking batches.

**Alternatives considered**:
- Individual entry signing — overhead too high at 1000+ events/sec; ML-DSA-65 signing is ~2ms
- HMAC-based log chains — faster but no non-repudiation; can't prove logs weren't modified by the signer
- External log shipping without signing — relies on transport security only

**Rationale**: Batch signing amortizes the ML-DSA-65 cost (~2ms per batch of up to 256 entries). Hash chaining provides forward integrity — tampering with any batch invalidates all subsequent batches. This satisfies NIST AU-9 (Protection of Audit Information) and AU-10 (Non-repudiation).

**Design**:
- Log entries accumulate in a kernel ring buffer (existing design)
- Every 256 entries or every 1 second (whichever comes first), a batch is sealed
- Batch = SHA-3-256(previous_batch_hash || entry_1 || ... || entry_N) → ML-DSA-65 signature
- Signed batches exported via Zenoh IPC (`smallaios/v1/audit`) for external collection
- Signing key generated at boot, optionally provisioned via secure boot / TPM

### Decision 3: SBOM as build artifact with CycloneDX format

**Choice**: Generate CycloneDX SBOM at build time using cargo metadata, include in OCI image labels and as standalone artifact.

**Alternatives considered**:
- SPDX format — equally valid but CycloneDX has better Rust tooling (cargo-cyclonedx)
- Runtime SBOM generation — unnecessary overhead; dependencies are static
- No SBOM, rely on cargo.lock — not machine-readable for compliance scanners

**Rationale**: CycloneDX is widely adopted by federal supply chain initiatives (EO 14028) and has mature Rust tooling. Build-time generation is correct because SmallAIOS is statically linked with no runtime dependency changes.

### Decision 4: Anomaly detection as statistical thresholds, not ML-based

**Choice**: Use configurable statistical thresholds (moving average, standard deviation, percentile) for anomaly detection. No machine learning models for security monitoring.

**Alternatives considered**:
- ML-based anomaly detection — ironic recursion (using ONNX to monitor ONNX), adds attack surface
- Fixed thresholds only — too brittle, requires manual tuning per deployment
- No runtime detection, rely on external monitoring — misses kernel-internal anomalies

**Rationale**: Statistical thresholds are deterministic, auditable, and formally verifiable. Moving averages with configurable sigma bounds detect anomalies without introducing ML complexity into the security monitoring path. Thresholds can be formally verified in TLA+ for correctness.

**Monitored signals**:
- Capability denial rate (per-second, per-task)
- Memory allocation failure rate
- Inference latency deviation from rolling p50 (>3 sigma triggers alert)
- Task scheduling latency deviation
- Watchdog time remaining (low-watermark tracking)
- Network connection rate (SYN flood detection)

### Decision 5: Security boundaries as Sphinx-needs traceability, not just diagrams

**Choice**: Document security boundaries as traceable Sphinx-needs requirements with PlantUML diagrams, not standalone architecture documents.

**Alternatives considered**:
- Standalone architecture document — not traceable to implementation/tests
- Code comments only — not auditable by compliance assessors
- External threat modeling tool (STRIDE/DREAD) — adds tooling dependency

**Rationale**: Sphinx-needs allows bidirectional traceability from boundary requirements to implementation code to test evidence, which satisfies DO-178C traceability AND NIST CA-2 (Security Assessments). PlantUML diagrams are generated from the same source, ensuring diagrams and requirements stay synchronized.

### Decision 6: WCET analysis via instrumentation + static bounds, not full static analysis

**Choice**: Combine compile-time static bounds (loop counts, recursion depth = 0) with runtime instrumentation (TSC/CNTPCT measurement of critical paths) to establish WCET.

**Alternatives considered**:
- Full static WCET analysis (aiT, OTAWA) — requires binary analysis tools not available for Rust/no_std
- Runtime measurement only — cannot guarantee worst case without proof
- Ignore WCET — unacceptable for OT/safety-critical deployments

**Rationale**: Rust's ownership model + no recursion rule + bounded loops gives us static composability. Runtime measurements on target hardware (N=10000+ runs) establish empirical WCET bounds. Combined with formal TLA+ scheduling proofs, this provides sufficient assurance for IEC 61508 SIL 3 / ISO 26262 ASIL D.

### Decision 7: Change control as documented process, not tooling

**Choice**: Define change control board (CCB) process, impact assessment checklist, and approval workflow as documentation. Use existing Git + CI/CD as the enforcement mechanism.

**Alternatives considered**:
- Custom change management tool — over-engineering for a kernel project
- JIRA/ServiceNow integration — adds heavyweight dependency
- No formal change control — unacceptable for DO-178C and NIST CM-3

**Rationale**: Git branch protection rules, CI gates (tests, clippy, formal verification), and code review requirements already enforce change control technically. The CCB documentation formalizes the human process (who approves what, impact assessment criteria, rollback triggers) needed for NIST CM-3 and DO-178C SDP compliance.

## Risks / Trade-offs

- **[Audit log signing overhead]** → ML-DSA-65 batch signing adds ~2ms per 256 entries; mitigated by batch amortization and async signing (non-blocking to inference path)
- **[SBOM maintenance burden]** → Automated in CI; cargo-cyclonedx generates from Cargo.lock with zero manual effort
- **[Anomaly detection false positives]** → Configurable thresholds per deployment; cold-start training period where alerts are suppressed; sigma bounds tunable (default 3-sigma)
- **[WCET measurement variability]** → Requires target-specific measurement campaigns; results are hardware-dependent; mitigated by mandating N=10000+ samples with p99.9 reporting
- **[Governance documentation scope creep]** → Scoped to SmallAIOS software only; organizational policies (HR, physical security) are out of scope and documented as "inherited controls"
- **[Dual compliance overhead (DO-178C + NIST)]** → Shared traceability infrastructure (Sphinx-needs) serves both; control mapping document identifies overlap to avoid duplicate evidence
