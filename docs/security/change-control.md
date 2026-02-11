# SmallAIOS Configuration Management and Change Control Plan

**Document Version:** 1.0
**Date:** 2026-02-10
**Classification:** Internal

---

## 1. Configuration Management Plan

### 1.1 Baseline Identification

A configuration baseline is established at each tagged release and consists of:

| Baseline Item | Identifier | Location |
|--------------|-----------|----------|
| Source code | Git tag (e.g., `v0.1.0`) + commit SHA | `SmallAIOS/SmallAIOS-Design` repository |
| Rust toolchain | Version in `rust-toolchain.toml` | `nightly-2026-02-01` |
| SBOM | CycloneDX JSON attached to release | OCI image label + build artifact |
| Formal verification models | Git-tracked in `formal/` | TLA+ (`.tla`), Lean 4 (`.lean`), SPIN (`.pml`) |
| Documentation | Git-tracked in `docs/` and `openspec/` | Sphinx RST, Markdown |
| Build configuration | `.cargo/config.toml`, `Makefile`, `Cargo.toml` | Repository root |

Each baseline receives a unique identifier: `BASELINE-<version>-<date>-<short-sha>`.

### 1.2 Version Control Policies

- All source code, configuration, and documentation changes MUST be committed to git
- All commits MUST include a descriptive message, author identity, and timestamp
- Direct commits to the `main` branch are prohibited
- All changes flow through pull requests with required reviews
- Squash merges are preferred for clean history; merge commits for large features

### 1.3 Configuration Item Naming Conventions

| Item Type | Convention | Example |
|-----------|-----------|---------|
| Crate | `smallaios-<name>` | `smallaios-security` |
| Branch | `change/<openspec-change-name>` | `change/cybersecurity-compliance-v3` |
| Tag | `v<major>.<minor>.<patch>` | `v0.1.0` |
| Baseline | `BASELINE-<version>-<YYYYMMDD>-<sha7>` | `BASELINE-v0.1.0-20260210-abc1234` |
| Change Request | `CR-<YYYY>-<NNN>` | `CR-2026-001` |

### 1.4 Configuration Audit

Configuration audits are performed:
- Before each release (mandatory)
- After any emergency change
- Quarterly (aligned with POA&M review)

Audit verifies:
1. All baselined items match their recorded versions (`git diff --stat <tag>`)
2. No unauthorized changes exist (all commits have associated PRs)
3. All changes since last baseline have approved change records
4. SBOM matches actual dependency tree (`cargo-cyclonedx` regeneration check)
5. Formal verification models are current with code changes

---

## 2. Change Control Board (CCB) Process

### 2.1 Process Flow

```
Change Request → Impact Assessment → CCB Review → Approve/Reject → Implement → Verify
```

1. **Change Request:** Author creates PR with description, impact assessment, and test plan
2. **Impact Assessment:** Author completes security/safety/performance assessment (see Section 3)
3. **CCB Review:** For safety-critical changes, CCB reviews at next scheduled meeting (or emergency session)
4. **Decision:** CCB votes to approve, reject, or defer with recorded rationale
5. **Implementation:** Author implements change; CI gates must pass
6. **Verification:** Reviewer confirms all tests pass, coverage maintained, formal models updated

### 2.2 What Requires CCB Approval

| Change Type | Approver |
|------------|---------|
| Safety-critical code (scheduler, memory, syscall, capability, crypto) | CCB + Safety Engineer |
| Security mechanism changes (audit, monitoring, incident response) | Security Lead + 1 CCB member |
| Non-safety code changes | Standard PR review (1 reviewer) |
| Documentation changes | Standard PR review (1 reviewer) |
| Emergency changes (active incident) | Security Lead + 1 CCB member; full CCB post-hoc review |

### 2.3 CCB Meeting Protocol
- **Frequency:** Biweekly (Tuesday 10:00 UTC)
- **Quorum:** Security Lead + Safety Engineer + 1 additional member
- **Agenda:** Open PRs requiring CCB review, POA&M status, pending impact assessments
- **Records:** Minutes with decisions, vote counts, conditions

---

## 3. Impact Assessment Templates

### 3.1 Security Impact Assessment

Required when modifying: capability definitions, cryptographic algorithms, key management,
authentication mechanisms, access control policies.

```
## Security Impact Assessment
- **Change:** [Description]
- **Affected Mechanism:** [capability/crypto/auth/access-control]
- **Nature:** [addition/modification/removal]
- **Attack Surface Change:** [increased/unchanged/decreased]
- **Formal Model Update Required:** [yes/no - which model?]
- **Reviewer:** Security Lead
```

### 3.2 Safety Impact Assessment

Required when modifying: scheduler, memory management (buddy/slab/tensor/paging),
syscall interface, interrupt handling.

```
## Safety Impact Assessment
- **Change:** [Description]
- **Affected Path:** [scheduler/memory/syscall/interrupt]
- **WCET Impact:** [increased/unchanged/decreased - by how much?]
- **Deadlock/Priority Inversion Risk:** [none/low/medium/high]
- **MC/DC Coverage Maintained:** [yes/no - current: X%]
- **Reviewer:** CCB
```

### 3.3 Performance Impact Assessment

Required when expected to affect system latency, throughput, or resource utilization.

```
## Performance Impact Assessment
- **Change:** [Description]
- **Affected Priority Class:** [SYSTEM/IPC/INFERENCE]
- **Benchmark Before:** [p50/p99 latency]
- **Benchmark After:** [p50/p99 latency]
- **Latency Change:** [+/- X%]
- **Requires CCB Approval (>10% degradation):** [yes/no]
```

---

## 4. Rollback Procedures

### 4.1 Code Changes

**Procedure:**
1. Identify the last known-good tag/commit
2. Create revert PR: `git revert <commit-range>`
3. Run full test suite on revert PR
4. Merge revert PR through standard (or emergency) process
5. Deploy reverted version via standard deployment pipeline

**Constraints:**
- No data loss to configuration state or audit logs
- Audit log records both the original change and the rollback
- POA&M entry created for the reverted issue

### 4.2 Model Updates

**Procedure:**
1. Identify previous model version in OCI registry (tag-based)
2. Update deployment configuration to reference previous model tag
3. Trigger model reload (container restart or API call)
4. Verify model signature and inference output correctness

**Constraints:**
- Rollback within RTO: datacenter 30s, edge 5s, safety-critical 100ms
- Model integrity verified (SHA-256 hash check) before loading

### 4.3 Configuration Changes

**Procedure:**
1. Identify previous configuration version in git or ConfigMap history
2. Restore previous configuration file/ConfigMap
3. Restart affected service to apply restored configuration
4. Verify system health and correct behavior

**Constraints:**
- Audit trail preserved: both change and rollback are logged
- Configuration version history maintained in git

### 4.4 Firmware Updates

**Procedure:**
1. Identify previous firmware version for affected hardware
2. Execute firmware downgrade per hardware vendor procedure:
   - GPU: NVIDIA firmware rollback via `nvflash` or management interface
   - SoC: Dual-bank fallback (A/B partitioning)
   - Bus transceiver: Vendor-specific update tool
3. Power cycle affected hardware
4. Verify firmware version and hardware functionality

**Constraints:**
- Dual-bank firmware (where supported) enables automatic fallback
- Single-bank firmware requires manual intervention and physical access

### 4.5 Post-Rollback Verification

For all rollback types:
1. Run system health check (all subsystems report ready)
2. Execute targeted test suite for affected components
3. Confirm monitoring metrics return to expected baseline
4. Record rollback event: reason, executor, timestamp, verification results
5. Create POA&M entry for the underlying issue that triggered rollback

---

## 5. CI Change Gates

All pull requests must pass the following gates before merge:

| Gate | Tool | Failure Action |
|------|------|----------------|
| All tests pass | `cargo test` (all workspace crates) | PR blocked |
| Clippy clean | `cargo clippy -- -D warnings` | PR blocked |
| Format check | `cargo fmt -- --check` | PR blocked |
| Formal verification | TLC model checker (TLA+ models) | PR blocked (safety-critical PRs) |
| MC/DC coverage | Coverage report on modified paths | PR blocked if < 100% on safety-critical paths |

Gate configuration: `.github/workflows/ci.yml`

All gates must pass simultaneously on the same commit.
If any gate fails, the specific failing gate is reported in PR status.

---

## 6. Approval Workflow Summary

| Change Type | Author | Reviewers | Approver | CI Gates |
|------------|--------|-----------|----------|----------|
| Safety-critical code | Any | Security Lead + Safety Engineer | CCB (vote) | All |
| Security mechanism | Any | Security Lead | Security Lead + 1 CCB | All |
| Non-safety code | Any | 1 reviewer | Reviewer | All |
| Documentation | Any | 1 reviewer | Reviewer | Format only |
| Emergency (incident) | Incident Commander | Security Lead | Security Lead + 1 CCB | All; post-hoc CCB review |
