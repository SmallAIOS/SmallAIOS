# Delta for Change Control

## ADDED Requirements

### Requirement: Configuration Management Plan
The system SHALL document a configuration management plan covering baseline identification, change tracking, and version control.

#### Scenario: Baseline identification
- WHEN a new SmallAIOS release is designated
- THEN the configuration management plan MUST define a configuration baseline consisting of: all source code at the tagged git commit, the Rust toolchain version, the SBOM, the formal verification model versions (TLA+, Lean 4, SPIN), and all documentation artifacts
- AND the baseline MUST be assigned a unique identifier and recorded in the configuration management database

#### Scenario: Change tracking from baseline
- WHEN a change is made to any baselined artifact (source code, toolchain, models, documentation)
- THEN the configuration management plan MUST require the change to be recorded with: a unique change identifier, description, affected baseline items, originator, date, and approval status
- AND the change history MUST be traceable from the current state back to any previous baseline

#### Scenario: Version control policy
- WHEN source code, configuration files, or documentation are modified
- THEN the configuration management plan MUST require all changes to be committed to the version control system (git) with a descriptive commit message, author identity, and timestamp
- AND direct commits to the main branch MUST be prohibited; all changes MUST flow through pull requests

#### Scenario: Configuration audit
- WHEN a configuration audit is performed (at minimum before each release)
- THEN the audit MUST verify that all baselined items match their recorded versions, no unauthorized changes exist, and all changes since the last baseline have approved change records

### Requirement: CCB Approval for Safety-Critical Changes
All changes to safety-critical code SHALL require Change Control Board (CCB) approval before merge.

#### Scenario: Safety-critical code change requires CCB approval
- WHEN a pull request modifies code in safety-critical paths (scheduler, memory management, syscall interface, capability system, cryptographic modules)
- THEN the pull request MUST be labeled as safety-critical
- AND the pull request MUST NOT be merged until the CCB has reviewed and approved the change with a recorded vote

#### Scenario: CCB approval is recorded
- WHEN the CCB approves a safety-critical change
- THEN the approval MUST be recorded with: the change identifier, CCB meeting date, voting members present, vote outcome (approve/reject/defer), and any conditions attached to the approval
- AND the approval record MUST be linked to the corresponding pull request

#### Scenario: Emergency change bypass with post-hoc review
- WHEN an emergency change to safety-critical code is required to address an active security incident or safety hazard
- THEN the change MAY be merged with approval from the Security Lead and one CCB member
- AND the change MUST be reviewed by the full CCB at the next scheduled meeting with a documented post-hoc rationale

#### Scenario: CCB rejection requires rework
- WHEN the CCB rejects a safety-critical change
- THEN the rejection rationale MUST be documented
- AND the change originator MUST address the CCB's concerns and resubmit the change for review before it can be reconsidered

### Requirement: Impact Assessment Criteria
The system SHALL define impact assessment criteria covering security impact (capability changes, crypto modifications), safety impact (scheduler, memory, syscall changes), and performance impact (latency budget).

#### Scenario: Security impact assessment
- WHEN a change modifies capability definitions, cryptographic algorithms, key management, authentication mechanisms, or access control policies
- THEN the change MUST include a security impact assessment documenting: the specific security mechanism affected, the nature of the change (addition, modification, removal), potential attack surface changes, and whether the change requires updated formal verification models
- AND the security impact assessment MUST be reviewed by the Security Lead

#### Scenario: Safety impact assessment
- WHEN a change modifies the scheduler, memory management (buddy/slab/tensor/paging), syscall interface, or interrupt handling
- THEN the change MUST include a safety impact assessment documenting: the affected safety-critical path, worst-case execution time (WCET) impact, potential for deadlock or priority inversion, and whether MC/DC coverage is maintained at 100%
- AND the safety impact assessment MUST be reviewed by the CCB

#### Scenario: Performance impact assessment
- WHEN a change is expected to affect system latency, throughput, or resource utilization
- THEN the change MUST include a performance impact assessment documenting: the affected latency budget (SYSTEM, IPC, or INFERENCE priority class), benchmark results before and after the change, and whether the change respects the soft real-time timing constraints
- AND any change that degrades latency by more than 10% for any priority class MUST require explicit CCB approval

#### Scenario: Combined impact assessment
- WHEN a change affects multiple impact categories (e.g., a crypto change that also affects scheduler timing)
- THEN separate assessments MUST be completed for each applicable category
- AND the CCB MUST review the combined impact before approving the change

### Requirement: Rollback Procedures
The system SHALL provide rollback procedures for every deployed change including model updates, configuration changes, and firmware updates.

#### Scenario: Rollback a code change
- WHEN a deployed code change causes a regression or security issue
- THEN the rollback procedure MUST define how to revert to the previous known-good baseline using git revert or tag-based deployment
- AND the rollback MUST be achievable without data loss to configuration state or audit logs

#### Scenario: Rollback a model update
- WHEN a deployed ONNX model update causes inference errors or performance degradation
- THEN the rollback procedure MUST define how to restore the previous model version from the OCI registry
- AND the rollback MUST be executable within the defined RTO for the deployment class (datacenter: 30s, edge: 5s, safety-critical: watchdog-bounded)

#### Scenario: Rollback a configuration change
- WHEN a configuration change (e.g., capability policy, network policy, scheduler parameters) causes operational issues
- THEN the rollback procedure MUST define how to restore the previous configuration from version control
- AND the rollback MUST preserve the audit trail documenting both the original change and the rollback

#### Scenario: Rollback a firmware update
- WHEN a firmware update to a hardware component (GPU, bus transceiver, SoC) causes instability
- THEN the rollback procedure MUST document the firmware downgrade process for each supported hardware platform
- AND the procedure MUST include verification steps to confirm the previous firmware version is restored and functional

#### Scenario: Rollback verification
- WHEN any rollback is executed
- THEN the rollback procedure MUST include post-rollback verification steps: system health check, test suite execution, and confirmation that the issue is resolved
- AND the rollback event MUST be recorded in the change tracking system with the reason, executor, timestamp, and verification results

### Requirement: CI Change Gates
The CI pipeline SHALL enforce change gates: all tests pass, clippy clean, formal verification models pass, and MC/DC coverage maintained.

#### Scenario: All tests must pass
- WHEN a pull request is submitted
- THEN the CI pipeline MUST execute the full test suite (`cargo test` for all workspace crates)
- AND the pull request MUST NOT be mergeable if any test fails

#### Scenario: Clippy must be clean
- WHEN a pull request is submitted
- THEN the CI pipeline MUST run `cargo clippy` with warnings treated as errors
- AND the pull request MUST NOT be mergeable if any clippy warning or error is reported

#### Scenario: Formal verification models must pass
- WHEN a pull request modifies code covered by formal verification (TLA+ concurrency models, Lean 4 type proofs, SPIN protocol models)
- THEN the CI pipeline MUST execute the corresponding formal verification checks
- AND the pull request MUST NOT be mergeable if any formal verification model fails

#### Scenario: MC/DC coverage must be maintained
- WHEN a pull request modifies safety-critical code paths
- THEN the CI pipeline MUST verify that MC/DC coverage remains at 100% for the modified paths
- AND if coverage drops below 100%, the pull request MUST NOT be mergeable until additional tests restore full coverage

#### Scenario: All gates must pass simultaneously
- WHEN any single change gate fails
- THEN the CI pipeline MUST report the specific failing gate(s) in the pull request status
- AND the pull request MUST remain blocked until all gates pass on the same commit
