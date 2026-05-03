## MODIFIED Requirements

### Requirement: Fuzz target coverage
The CI fuzz matrix was six fuzz targets (ONNX protobuf, ONNX tensor, IPC, TCP, UDP, USB) at 60 s each. It SHALL now include nine targets — the original six plus `fuzz_syscall_abi`, `fuzz_pqc_mlkem768`, `fuzz_pqc_mldsa65`. PQC differential targets SHALL upload `cargo-fuzz coverage` HTML as a nightly artifact.

#### Scenario: PR CI fuzz matrix executes all targets
- **WHEN** a PR opens against `develop` or `main`
- **THEN** the CI matrix runs each of the nine targets for 60 s and blocks merge on any panic, sanitizer violation, or differential mismatch

## ADDED Requirements

### Requirement: Syscall ABI adversarial coverage
The full syscall ABI surface (~65 entries across `sys_cap_*`, `sys_mem_*`, `sys_tensor_*`, `sys_ipc_*`, `sys_device_*`) SHALL have a fuzz target `fuzz/fuzz_targets/fuzz_syscall_abi.rs` exercising out-of-range numbers, malformed argument structs, forged capability handles, and resource exhaustion. Runs 60 s in PR CI, 1 h nightly. Any panic or invariant violation fails the build.

#### Scenario: Out-of-range syscall number rejected
- **WHEN** the fuzzer dispatches a syscall with `number > MAX_SYSCALL_NUMBER`
- **THEN** the dispatcher returns `Err(ENOSYS)` without panicking and without mutating any kernel state

#### Scenario: Forged capability handle rejected
- **WHEN** the fuzzer passes a capability handle with a generation not matching the entry in the capability table
- **THEN** the syscall returns `Err(EBADF)` and the capability table is unchanged

#### Scenario: Resource-exhaustion request rejected without OOM
- **WHEN** the fuzzer issues `sys_tensor_alloc(usize::MAX)` or pushes an IPC pipeline deeper than the per-process cap
- **THEN** the syscall returns `Err(ENOMEM)` or `Err(E2BIG)` and `MockKernel` invariants hold

### Requirement: PQC timing-leak advisory monitoring
PQC encapsulate and sign paths SHALL be exercised by a `dudect`-style detector on a weekly cron `pqc-timing-leak`. Advisory: t > 4.5 emits a warning, does not fail the build. Runs on `self-hosted-isolated`.

#### Scenario: Timing leak detected above threshold
- **WHEN** the weekly job observes Welch t > 4.5 on encapsulate or sign
- **THEN** the job emits a CI warning and posts to the tracking issue, but does NOT fail the build

### Requirement: Adversarial evidence in DAL A audit trail
DAL A audit evidence SHALL include the result of each adversarial test category from `crates/red-team-tests/`, archived per release.

#### Scenario: Release archives include red-team report
- **WHEN** a release is tagged from `main`
- **THEN** the release artifacts MUST include the `red-team-suite` CI job log and the playbook revision hash

### Requirement: CI gates red-team suite
The CI pipeline SHALL run `red-team-suite` on every PR targeting `develop` or `main`, advisory initially and blocking once stable per the gating policy in `docs/red-team-playbook.md`.

#### Scenario: Suite is advisory before promotion
- **WHEN** the suite has fewer than 5 consecutive green `develop` runs
- **THEN** the job is marked `continue-on-error: true`

#### Scenario: Suite is blocking after promotion
- **WHEN** the suite has 5+ consecutive green `develop` runs
- **THEN** the job is required for PR merge and the promotion is recorded in the playbook "Gating history" section
