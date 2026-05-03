## MODIFIED Requirements

### Requirement: Lean 4 proofs are machine-checked
The existing Lean 4 proofs in `formal/lean4/` SHALL be type-checked by CI on every pull request. Lean 4 SHALL no longer be described as "aspirational" in this specification.

#### Scenario: PR triggers `lean-verify`
- **WHEN** a developer opens a PR
- **THEN** the `lean-verify` job runs `lake build` and fails the PR if any proof fails to typecheck

#### Scenario: Cached vs cold build
- **WHEN** the CI cache is warm
- **THEN** `lean-verify` completes in under 2 minutes
- **AND** when cold, under 8 minutes

#### Scenario: Toolchain pin is honored
- **WHEN** a contributor changes `formal/lean4/lean-toolchain`
- **THEN** `elan` installs the pinned version and `lake build` re-runs against that version

## ADDED Requirements

### Requirement: Lean 4 toolchain is pinned and reproducible
The Lean 4 toolchain version SHALL be declared in `formal/lean4/lean-toolchain` and consumed by `elan`. The package SHALL be declared via `formal/lean4/lakefile.lean` using `lake` (not `leanpkg`).

#### Scenario: CI installs pinned elan and Lean version
- **WHEN** CI runs `lean-verify`
- **THEN** it installs `elan` from a pinned commit, reads the toolchain version from `lean-toolchain`, and uses `lake build` to compile every `.lean` file under `formal/lean4/`

#### Scenario: Toolchain bump goes through PR review
- **WHEN** a PR introduces a Lean version not yet vetted
- **THEN** the PR author updates `lean-toolchain` and re-runs the gate as part of review

### Requirement: Capability delegation monotonicity is model-checked
`formal/tla/CapabilitySecurity.tla` SHALL define and check a `DelegationMonotonicity` invariant ensuring every delegated capability's permission set is a subset of its parent's, and a `CapabilityIdStrictlyIncreasing` invariant ensuring the global capability ID counter is monotonic with no reuse.

#### Scenario: TLC reports zero counterexamples
- **WHEN** TLC runs `CapabilitySecurity.tla`
- **THEN** it explores the state space and reports zero counterexamples for both invariants

#### Scenario: Privilege-escalating change fails the gate
- **WHEN** a future change introduces a delegation operation that grants extra rights
- **THEN** TLC produces a counterexample trace and the `tla-verify` CI job fails

### Requirement: Scheduler state-machine fairness is model-checked
`formal/tla/SchedStateMachine.tla` SHALL model task transitions across `Ready`, `Running`, `Yielded`, `Blocked` states and check both safety (at most one task `Running` per core; no `Blocked → Running` shortcut) and LTL liveness (every `Ready` task is eventually `Running` under weak fairness on dispatch).

#### Scenario: TLC succeeds at bounded sizes
- **WHEN** TLC runs `SchedStateMachine.tla` at `N_CORES = 3` and `N_TASKS = 6`
- **THEN** it reports zero counterexamples for the safety invariants and the LTL liveness property

#### Scenario: Matrix wall-time stays under budget
- **WHEN** the model is added to the `tla-verify` matrix
- **THEN** the matrix grows from 22 to 23 models and total wall time stays under the 5-minute CI budget

### Requirement: SMT-based proofs of low-level invariants are gated on CI
The `smallaios-kernel` crate SHALL expose an opt-in `formal-smt` feature that pulls in the `z3` Cargo crate (with `bundled` Z3) and registers SMT-backed proofs as integration tests. CI SHALL run a `smt-verify` advisory job that builds the crate with `--features formal-smt` and executes those tests.

#### Scenario: Default build is unaffected
- **WHEN** `formal-smt` is disabled (the default)
- **THEN** the kernel crate builds with no Z3 dependency and `#![no_std]` targets are unaffected

#### Scenario: Opt-in build runs SMT proofs
- **WHEN** `formal-smt` is enabled
- **THEN** `cargo test -p smallaios-kernel --features formal-smt` runs the SMT proof suite, reports UNSAT for each proof obligation, and exits 0

### Requirement: Bump allocator pointer monotonicity is SMT-proven
The first SMT proof under `kernel/proofs/bump_allocator_smt.rs` SHALL encode `BumpAllocator` state (`base`, `current`, `end`) as 64-bit bitvectors and prove, by bounded model checking up to `N = 8` sequential `alloc(size)` calls, that `current` is monotonically non-decreasing and `current ≤ end` is invariant.

#### Scenario: Z3 returns UNSAT on the negation
- **WHEN** `smt-verify` runs the bump-allocator proof
- **THEN** Z3 returns UNSAT on the negation of the monotonicity assertion for every `N` from 1 to 8

#### Scenario: Mutated post-condition fails fast
- **WHEN** a developer mutates the post-condition to a known-false form
- **THEN** Z3 returns SAT and the test fails fast with a model witness logged
