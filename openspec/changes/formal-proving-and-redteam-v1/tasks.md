# Tasks — formal-proving-and-redteam-v1

Three phases. Each phase has explicit exit criteria. Sub-changes may split phases for separate landing.

---

## Phase 1: Verification stack expansion (~3 weeks)

### 1.1 Lean 4 CI integration

- [ ] 1.1.1 Add `formal/lean4/lean-toolchain` pinning `leanprover/lean4:v4.15.0`
- [ ] 1.1.2 Audit existing 6 proofs for `mathlib` imports and document findings in `formal/lean4/README.md`; decide vendor-vs-cache strategy
- [ ] 1.1.3 Add `formal/lean4/lakefile.lean` declaring the `SmallAIOSProofs` package and any required dependencies (mathlib gated on §1.1.2 outcome)
- [ ] 1.1.4 Verify `lake build` succeeds locally for `CapabilityNonForgery.lean`, `InformationFlow.lean`, `IntegrityLattice.lean`, `LabelComposition.lean`, `MessageTypeProperties.lean`, `TensorTypeInvariants.lean`
- [ ] 1.1.5 Add `lean-verify` job to `.github/workflows/ci.yml`: install `elan` (pinned commit), run `lake build` from `formal/lean4/`
- [ ] 1.1.6 Add GitHub Actions cache for `formal/lean4/.lake/build/` and `~/.elan/toolchains/` keyed on hash of `lean-toolchain` + `lakefile.lean`
- [ ] 1.1.7 Wire `lean-verify` into the `change-gates` meta-job as a **required** check (hard gate)
- [ ] 1.1.8 Update `CLAUDE.md` and `docs/architecture.md` to list `lean-verify` in the gate-jobs table
- [ ] 1.1.9 **Verification:** open a draft PR that introduces a deliberately broken `lean-toolchain` version and confirm `lean-verify` fails the PR; revert before merge

### 1.2 TLA+ gap-fill

- [ ] 1.2.1 Extend `formal/tla/CapabilitySecurity.tla` with `DelegationMonotonicity` invariant: `\A edge \in DelegationGraph: edge.child.rights \subseteq edge.parent.rights`
- [ ] 1.2.2 Extend `formal/tla/CapabilitySecurity.tla` with `CapabilityIdStrictlyIncreasing` invariant on the global ID counter, and prove fresh IDs are strictly greater than any prior live ID
- [ ] 1.2.3 Update `formal/tla/CapabilitySecurity.cfg` to declare both new invariants under `INVARIANTS`
- [ ] 1.2.4 Confirm `CapabilitySecurity` TLC run still finishes inside the 5-minute CI budget at existing bounds; reduce bounds if needed and document in the model header
- [ ] 1.2.5 Create `formal/tla/SchedStateMachine.tla` modeling task states (`Ready`, `Running`, `Yielded`, `Blocked`) and transitions per design §D1.2
- [ ] 1.2.6 Add safety invariants: `AtMostOneRunningPerCore`, `NoBlockedToRunningWithoutReady`
- [ ] 1.2.7 Add LTL liveness property: `[]<>(t \in ReadyTasks => <>(t \in RunningTasks))` under weak fairness on the dispatch action
- [ ] 1.2.8 Create `formal/tla/SchedStateMachine.cfg` bounded at `N_CORES = 3`, `N_TASKS = 6`
- [ ] 1.2.9 Wire `SchedStateMachine` into `.github/workflows/ci.yml` `tla-verify` matrix (the existing 22-model job — bumps to 23)
- [ ] 1.2.10 Update `openspec/smallaios-kernel/specs/13-formal-verification.md` to list both extensions and the new `SchedStateMachine` model
- [ ] 1.2.11 **Verification:** TLC reports zero counterexamples for all new invariants and the LTL property; total `tla-verify` job wall time stays under the 5-minute budget

### 1.3 SMT solver scaffolding

- [ ] 1.3.1 Add `formal-smt` feature flag to `kernel/Cargo.toml` with `optional = true` dep `z3 = { version = "0.12", features = ["bundled"] }`
- [ ] 1.3.2 Document the `formal-smt` feature in `CLAUDE.md` "Crate Feature Flags" — opt-in, pulls Z3, requires `std`
- [ ] 1.3.3 Create `kernel/proofs/mod.rs` (gated `#[cfg(feature = "formal-smt")]`) holding shared SMT helpers (Context, Solver, BV builder)
- [ ] 1.3.4 Create `kernel/proofs/bump_allocator_smt.rs`: encode `BumpAllocator` `(base, current, end)` state as 64-bit BVs
- [ ] 1.3.5 Encode the `alloc(size)` post-condition: `current' = current + size /\ current' <= end`
- [ ] 1.3.6 Prove pointer monotonicity over `N = 8` bounded sequential allocations: `current_after >= current_before` and `current_after <= end` for all reachable states; assert via `solver.check_assumptions(...) == Unsat` on the negation
- [ ] 1.3.7 Add a `smt-verify` advisory job to `.github/workflows/ci.yml` running `cargo test -p smallaios-kernel --features formal-smt --test bump_allocator_smt`
- [ ] 1.3.8 Cache the bundled-Z3 build artifact in CI (target/release/build/z3-sys-*) keyed on `z3-sys` version
- [ ] 1.3.9 Update `openspec/smallaios-kernel/specs/13-formal-verification.md` with a new "SMT (Z3)" section listing the feature flag, first proof, and CI job
- [ ] 1.3.10 **Verification:** `smt-verify` CI job succeeds, prints the SAT/UNSAT result for the monotonicity query, and fails fast on a deliberately mutated post-condition (smoke-test the negative path locally before merge)

### 1.4 Phase 1 exit criteria

- [ ] 1.4.1 All 6 existing Lean 4 proofs build in CI as a hard gate
- [ ] 1.4.2 `CapabilitySecurity.tla` checks `DelegationMonotonicity` and `CapabilityIdStrictlyIncreasing` in CI
- [ ] 1.4.3 `SchedStateMachine.tla` checks safety invariants and LTL liveness in CI (bringing the TLA+ corpus to 23 models)
- [ ] 1.4.4 Z3-backed bump-allocator monotonicity proof runs in the new `smt-verify` CI job
- [ ] 1.4.5 `docs/architecture.md`, `CLAUDE.md`, and spec 13 reflect the new gates and feature flag

---

## Phase 2: Fuzzing expansion + crypto KAT (~3 weeks)

### 2.1 Syscall ABI fuzzer

- [ ] 2.1.1 Add `cfg(fuzzing)` shim `kernel::syscall::dispatch_for_fuzz()` exporting dispatch with the `MockKernel` backend
- [ ] 2.1.2 Implement `MockKernel` in `fuzz/src/mock_kernel.rs` — capability table with monotone generations, stub allocator, stub IPC ring
- [ ] 2.1.3 Per-family `arbitrary::Arbitrary` impls in `fuzz/fuzz_targets/syscall_args.rs` (Cap, Mem, Tensor, Ipc, Device)
- [ ] 2.1.4 Create `fuzz/fuzz_targets/fuzz_syscall_abi.rs` driving `dispatch_for_fuzz()` on `SyscallCall { number: u32, args: Vec<u8> }`
- [ ] 2.1.5 Generate seed corpus from `kernel/tests/syscall_*.rs` via `cargo run -p smallaios-kernel --example dump_syscall_corpus`
- [ ] 2.1.6 Coverage: out-of-range numbers, malformed lengths, forged handles, expired generations, wrong-type tags, resource exhaustion
- [ ] 2.1.7 Assert no panics; assert `MockKernel` invariants after every call
- [ ] 2.1.8 Add to `.github/workflows/ci.yml` fuzz matrix: 60 s PR CI, 1 h nightly cron
- [ ] 2.1.9 Document harness + `MockKernel` limitations in `fuzz/README.md`

### 2.2 PQC differential fuzz

- [ ] 2.2.1 Add `pqcrypto-kyber = "0.8"` and `pqcrypto-dilithium = "0.5"` to `fuzz/Cargo.toml` `[dev-dependencies]`
- [ ] 2.2.2 Create `fuzz/fuzz_targets/fuzz_pqc_mlkem768.rs` — seeded keygen + encapsulate; assert bit-for-bit `ct` + `ss` equality vs `pqcrypto-kyber`
- [ ] 2.2.3 Round-trip path: own-impl encapsulate → ref decapsulate, and reverse; assert shared secrets match
- [ ] 2.2.4 Create `fuzz/fuzz_targets/fuzz_pqc_mldsa65.rs` — seeded keygen + sign; assert byte-equal signatures vs `pqcrypto-dilithium` derand path
- [ ] 2.2.5 Cross-verify: own-impl sign → ref verify, ref sign → own-impl verify
- [ ] 2.2.6 Surface failures with hex-dumped seeds, `pk`, `sk`, message, expected/actual output
- [ ] 2.2.7 Coverage target: every path in `security/src/crypto/{mlkem768,mldsa65}.rs` reached within 2^16 iterations; archive `cargo-fuzz coverage` HTML as a nightly CI artifact
- [ ] 2.2.8 Document differential model in `docs/pqc-differential-fuzz.md`

### 2.3 NIST KAT vectors

- [ ] 2.3.1 Add `pqc-kat` feature to `security/Cargo.toml` (test-only)
- [ ] 2.3.2 Vendor `.rsp` files from FIPS 203 / 204 reference packages under `security/tests/kat-vectors/{mlkem768,mldsa65}/`
- [ ] 2.3.3 Add `security/tests/kat-vectors/SHA256SUMS`; harness rejects mismatched digests at setup
- [ ] 2.3.4 Add `scripts/refresh-pqc-kats.sh` — downloads from canonical NIST URLs, updates `SHA256SUMS`, requires manual review before commit
- [ ] 2.3.5 Implement shared `security/tests/kat_parser.rs` — `KEY = HEX` parser yielding `Iterator<Item = KatVector>`
- [ ] 2.3.6 Implement `security/tests/kat_mlkem768.rs` — drive `keypair_from_seed`, `encapsulate_derand`, `decapsulate`; assert byte-exact `pk`, `sk`, `ct`, `ss`
- [ ] 2.3.7 Implement `security/tests/kat_mldsa65.rs` — drive `keypair_from_seed`, `sign_derand`, `verify`; assert byte-exact `pk`, `sk`, `sig`
- [ ] 2.3.8 Add CI job `pqc-kat-verify` (`cargo test -p smallaios-security --features pqc-kat`); **required check** on `develop` and `main`
- [ ] 2.3.9 Document harness, refresh procedure, version pinning in `docs/pqc-kat.md`

### 2.4 Side-channel guardrails (stretch — advisory)

- [ ] 2.4.1 Audit `security/src/crypto/{mlkem768,mldsa65}.rs` for data-dependent branches; tag leaky ops `// NOT-CT:` + rationale
- [ ] 2.4.2 Replace plain `==` ciphertext / tag comparisons with `subtle::ConstantTimeEq`
- [ ] 2.4.3 Write `docs/pqc-side-channel.md` documenting CT invariants per primitive (NTT, rejection sampling, comparison)
- [ ] 2.4.4 Add `dudect-rs` dev-dep + `security/tests/timing_mlkem768.rs` and `timing_mldsa65.rs`
- [ ] 2.4.5 Add weekly CI job `pqc-timing-leak` (advisory; t > 4.5 → warning) on `self-hosted-isolated` runner
- [ ] 2.4.6 [DEFERRED] `ct-verif` / Jasmin-based formal CT proofs — follow-up change; multi-week effort

### 2.5 CI integration & docs

- [ ] 2.5.1 Update `.github/workflows/ci.yml` fuzz matrix: add `fuzz_syscall_abi`, `fuzz_pqc_mlkem768`, `fuzz_pqc_mldsa65`
- [ ] 2.5.2 Wire `pqc-kat-verify` into the `change-gates` meta-job as a blocking gate
- [ ] 2.5.3 Add `pqc-timing-leak` as an advisory weekly job
- [ ] 2.5.4 Update `CLAUDE.md` "CI/CD" section with the three new jobs and the `pqc-kat` feature
- [ ] 2.5.5 Update `fuzz/README.md` with new target descriptions, corpus locations, reproduction commands
- [ ] 2.5.6 Cross-link `docs/pqc-side-channel.md`, `docs/pqc-kat.md`, `docs/pqc-differential-fuzz.md` from `docs/security.md`

### 2.6 Phase 2 exit criteria

- [ ] 2.6.1 Syscall ABI fuzz target runs 60 s in PR CI without panics or invariant violations
- [ ] 2.6.2 PQC differential fuzz targets pass bit-for-bit on deterministic API and round-trip on non-deterministic API
- [ ] 2.6.3 NIST KAT verification is a blocking CI gate for ML-KEM-768 and ML-DSA-65
- [ ] 2.6.4 PQC side-channel invariants documented; equality-on-secret uses `subtle::ConstantTimeEq` everywhere
- [ ] 2.6.5 `dudect`-style timing-leak job runs weekly as advisory

---

## Phase 3: Red-team / adversarial test suite (~3-4 weeks)

### 3.1 Adversarial corpus, playbook, and attack-surface docs

- [ ] 3.1.1 Create `docs/red-team-playbook.md` with sections: Scope, Scenario Catalog, Gating Policy, Gating History, Reporting
- [ ] 3.1.2 Populate Scenario Catalog with 5 categories (capability, ONNX, IPC, boot, network), each row mapping to threat-model line(s) in `openspec/smallaios-kernel/specs/06-security-model.md` and to a test module name in `crates/red-team-tests/`
- [ ] 3.1.3 Create `docs/attack-surface.md` per design §D3.4 — table of all external interfaces with adversarial coverage status
- [ ] 3.1.4 Cross-link `docs/red-team-playbook.md` from `docs/security/security-governance.md` (CCB review duty) and from `openspec/smallaios-kernel/specs/12-safety-critical.md` (DAL A audit evidence)
- [ ] 3.1.5 Add `just red-team` recipe stub to `Justfile` (runs the crate test once the scaffold lands in 3.2)

### 3.2 Adversarial test crate scaffold

- [ ] 3.2.1 Add `crates/red-team-tests/` to workspace `Cargo.toml` members list
- [ ] 3.2.2 Create `crates/red-team-tests/Cargo.toml` with `[features] red-team = []`, dev-dependencies on `proptest`, `arbitrary`
- [ ] 3.2.3 Create `crates/red-team-tests/src/lib.rs` with attack helper modules: `forge`, `corpus`, `assert_refused`, `assert_audit_logged`
- [ ] 3.2.4 Create `crates/red-team-tests/tests/` directory with one empty integration test per category (capability, onnx, ipc, boot, net) returning `Ok(())` — fills out in 3.3-3.7
- [ ] 3.2.5 Document corpus layout `crates/red-team-tests/corpus/{onnx,net,ipc}/seeds/`

### 3.3 Capability-violation suite (proptest + integration)

- [ ] 3.3.1 Implement `prop_delegation_subset_rights` (D3.3 P1)
- [ ] 3.3.2 Implement `prop_revocation_cascades` (D3.3 P2)
- [ ] 3.3.3 Implement `prop_cap_id_unique` (D3.3 P3)
- [ ] 3.3.4 Implement `prop_no_upgrade_via_alias` (D3.3 P4)
- [ ] 3.3.5 Implement `prop_quota_monotone` (D3.3 P5)
- [ ] 3.3.6 Implement `prop_audit_log_complete` (D3.3 P6)
- [ ] 3.3.7 Integration test: forge fake capability handle, attempt `ipc_send` on un-delegated topic — assert refused + audit-logged

### 3.4 ONNX graph injection suite

- [ ] 3.4.1 Add malformed protobuf seeds: cyclic graph, op-count overflow, attribute type confusion (3 seed files)
- [ ] 3.4.2 Resource-exhaustion graph (initializer >100 MB) — assert load fails before allocation
- [ ] 3.4.3 Deeply-nested subgraph (depth >64) — assert recursion bound rejection
- [ ] 3.4.4 Hook into existing fuzz target naming (`onnx_parser_fuzz`) to share corpus seeds

### 3.5 IPC amplification / DoS suite

- [ ] 3.5.1 Topic-flood test: spawn N publishers on one topic, assert back-pressure and no panic
- [ ] 3.5.2 Circular-wait deadlock test: 3-node ring with timeout assertion
- [ ] 3.5.3 Broken topic permission bypass: attempt subscribe without capability — assert refused

### 3.6 Boot integrity suite

- [ ] 3.6.1 Multiboot2 header bit-flip corpus + replay test asserts kernel halts with audit entry
- [ ] 3.6.2 Malformed DTB (truncated / cyclic phandle) — assert rejection at parse
- [ ] 3.6.3 Fake measurement test: replay log with mismatched hash — assert verified-boot feature gate refuses to continue
- [ ] 3.6.4 Document Phase-3 boot coverage gap (UEFI / TPM / TrustZone) in `docs/boot-security-matrix.md` § "Adversarial coverage"

### 3.7 Network protocol abuse suite

- [ ] 3.7.1 Malformed TCP option corpus beyond fuzzer seeds (TS-Echo, MD5 sig)
- [ ] 3.7.2 UDP amplification reflection scenario — assert rate limit
- [ ] 3.7.3 QUIC handshake manipulation: replay ClientHello with invalid PQC key share — assert TLS abort
- [ ] 3.7.4 DNS resolver attacks (if resolver active in build) — assert refused or marked N/A in playbook

### 3.8 CI integration

- [ ] 3.8.1 Add `red-team-suite` job to `.github/workflows/ci.yml`, runs `cargo test -p red-team-tests --features red-team`
- [ ] 3.8.2 Initially marked `continue-on-error: true` (advisory per D3.2 gating policy)
- [ ] 3.8.3 Upload corpus failures as workflow artifacts
- [ ] 3.8.4 After 5 green `develop` runs, flip to blocking and add entry to "Gating history" in `docs/red-team-playbook.md`
- [ ] 3.8.5 Cross-reference `red-team-suite` from NIST 800-53 SSP `docs/security/nist-800-53-ssp.md` controls CA-8 (Penetration Testing) and SI-3 (Malicious Code Protection)

### 3.9 Phase 3 exit criteria

- [ ] 3.9.1 `docs/red-team-playbook.md` covers all 5 categories with threat-model line refs
- [ ] 3.9.2 `crates/red-team-tests/` implements 6 property tests + 5 integration suites
- [ ] 3.9.3 `red-team-suite` is at least advisory-green on `develop` for 1 release cycle
- [ ] 3.9.4 `docs/attack-surface.md` enumerates every external interface with current coverage status
