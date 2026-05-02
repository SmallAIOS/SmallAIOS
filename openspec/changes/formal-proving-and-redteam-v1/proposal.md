# formal-proving-and-redteam-v1

## Summary

SmallAIOS targets DO-178C DAL A and ships meaningful formal verification scaffolding today — 25 TLA+ models (22 in CI), 6 SPIN/Promela models, 6 Lean 4 proofs, 6 fuzz targets, plus CodeQL / SonarCloud / cargo-deny / cargo-vet / cargo-careful / cargo-geiger / Miri / Kani in CI. But three structural gaps prevent that scaffolding from constituting verification *evidence*: (1) the 6 Lean 4 proofs are not type-checked in CI — they are aspirational text on disk; (2) the TLA+ corpus lacks two load-bearing invariant families (capability delegation monotonicity, scheduler state-machine fairness) and has zero SMT coverage of low-level invariants like allocator monotonicity; (3) the threat model in `06-security-model.md` is documented but **unexercised** — there is no syscall ABI fuzzer, no PQC differential fuzz vs reference, no NIST KAT vectors, no adversarial test corpus, and no attack-surface inventory. This change proposes a phased plan to close those gaps, in the order in which downstream evidence depends on them.

The change is structured as a roadmap: **three phases, ~9 weeks total**, each producing artifacts that gate the next. Implementation may proceed phase-by-phase as separate sub-changes once this umbrella is approved, or in one go if scope permits. The umbrella stays active in `openspec/changes/` until all phases are done.

## Phase 1 — Verification stack expansion (~3 weeks)

SmallAIOS already ships 25 TLA+ models (22 in CI), 6 SPIN/Promela models, and 6 Lean 4 proofs covering capability non-forgery, information flow, the integrity lattice, label composition, message-type properties, and tensor type invariants. The Lean proofs are not built by CI today — they are aspirational text on disk. Two structurally important TLA+ invariant families are missing: capability **delegation monotonicity** (the keystone of the non-forgery argument) and **scheduler state-machine fairness** (the keystone of the cooperative-async correctness argument that downstream WCET reasoning depends on). And the stack contains zero SMT coverage — no Z3, CVC5, or Bitwuzla bindings, no `.smt2` files, no Rust crate deps — even though several low-level kernel invariants (allocator monotonicity, ring-buffer indices, capability ID arithmetic) fit naturally in QF_BV. For a system targeting DO-178C DAL A traceability, each of these is a documented gap in the verification evidence trail.

Phase 1 closes those three gaps in roughly three weeks of focused work. We wire the existing Lean 4 proofs into a hard-gate CI job (`lean-verify`) via `elan` + pinned `lean-toolchain` + `lake build` with cached `.lake/build/`, so the type checker enforces the proofs on every PR. We extend `formal/tla/CapabilitySecurity.tla` with `DelegationMonotonicity` and `CapabilityIdStrictlyIncreasing` invariants, and add a new `formal/tla/SchedStateMachine.tla` modeling `Ready/Running/Yielded/Blocked` transitions with LTL liveness under weak fairness on dispatch. Finally we stand up the SMT scaffolding behind a new `formal-smt` Cargo feature (Z3 first, justified by mature Rust bindings and theory fit) and land a first proof — bump allocator pointer monotonicity by bounded model checking — to validate the toolchain end-to-end.

## Phase 2 — Fuzzing expansion + crypto KAT (~3 weeks)

SmallAIOS today has six structural fuzz targets covering parsers (ONNX protobuf, tensor, IPC, TCP, UDP, USB) but **zero adversarial coverage on the ~65-syscall ABI surface**. Forged capability handles, expired generations, out-of-range numbers, and `usize::MAX` allocation requests are caught only by hand-written property tests on the safe Rust wrappers — the raw `dispatch()` entry is never fuzzed. Phase 2 adds `fuzz/fuzz_targets/fuzz_syscall_abi.rs` driven by a `MockKernel`, fuzzed under libFuzzer + sanitizers for 60 s in PR CI and 1 h nightly.

The PQC stack (ML-KEM-768, ML-DSA-65) is also validated only by hand-written round-trip tests inside its own crate — no NIST KAT gate, no differential check. Phase 2 vendors NIST KAT vectors under `security/tests/kat-vectors/` (SHA-256-pinned) and runs them as a blocking `pqc-kat-verify` CI job; two new `cargo-fuzz` targets drive identical seeds through our impl and `pqcrypto-kyber 0.8` / `pqcrypto-dilithium 0.5` and assert bit-for-bit ciphertext and shared-secret equality. Differential fuzz catches own-impl regressions cheaply — divergence on any seed surfaces as a hex-dumped CI reproducer. A stretch side-channel track adds a `dudect`-based weekly advisory job and `docs/pqc-side-channel.md`; full Jasmin / `ct-verif` proofs are explicitly deferred.

## Phase 3 — Red-team / adversarial test suite (~3-4 weeks)

SmallAIOS has a documented threat model (`openspec/smallaios-kernel/specs/06-security-model.md`) and a NIST 800-53 control mapping (`docs/security/nist-800-53-ssp.md`), but the threat model is currently **unexercised**: no automated tests synthesize forged capability handles, malformed ONNX graphs, IPC floods, tampered Multiboot2 headers, or abusive network protocol sequences. DAL A certification under `openspec/smallaios-kernel/specs/12-safety-critical.md` requires adversarial evidence in the audit trail, and our boot security matrix (`docs/boot-security-matrix.md`) shows multiple "No" / "Partial" mechanisms that need explicit coverage statements rather than implicit gaps.

Phase 3 closes that gap by introducing `docs/red-team-playbook.md`, `docs/attack-surface.md`, and a new `crates/red-team-tests/` workspace crate that unifies fuzz seeds, `proptest` capability invariants, and integration scenarios under a single CI gate (`red-team-suite`, advisory then blocking). Six initial property tests pin down capability delegation, cascading revocation, ID non-reuse, alias safety, quota monotonicity, and audit completeness. Five scenario suites (capability, ONNX, IPC, boot, network) map every in-scope threat-model line to a concrete refusal or audit-logged outcome. The work sets up future Kani harnesses to mirror the same properties under bounded model checking.

## Out of scope

- **Lifting Lean 4 proofs to Rust via `aeneas` / `verus` / `creusot`** — meaningful, but a multi-month effort with its own toolchain integration; tracked as a follow-up change.
- **Bare-metal Miri** — non-trivial because Miri targets unsupported by `x86_64-unknown-none`, `aarch64-unknown-none`, `riscv64gc-unknown-none-elf`. Tracked as a separate investigation.
- **Full constant-time formal proofs** (Jasmin / `ct-verif`) for PQC primitives — Phase 2 adds documented invariants and a `dudect` advisory check; full proofs are a follow-up.
- **UEFI / TPM / TrustZone adversarial coverage** — the boot security matrix marks these "No" / "Partial"; Phase 3 covers Multiboot2 self-measurement only and explicitly defers the rest.
- **Capability-system Kani harnesses** — Phase 3 lays the property-test groundwork; mirroring into Kani is a Phase 4 follow-up.
- **SMT proofs of cryptographic primitives** — Bitwuzla is well-suited but the first SMT proof targets `BumpAllocator` to validate scaffolding before tackling crypto arithmetic.

## Sequencing

Phases run in declared order. Each phase is implementable independently once Phase 1 lands (Phase 2 + Phase 3 do not depend on each other and could parallelize). The recommended sequence is 1 → 2 → 3 because the verification scaffolding (Lean CI, SMT) makes it cheaper to land regressions caught later. Each phase has explicit exit criteria; archival of this umbrella waits until all three exit-criteria sets are satisfied, with deferred items marked inline.

## Effort estimate

| Phase | Scope | Estimate |
|-------|-------|----------|
| 1 | Lean 4 CI + TLA+ extensions + SMT scaffolding | ~3 weeks |
| 2 | Syscall ABI fuzzer + PQC differential + NIST KAT + side-channel guardrails | ~3 weeks |
| 3 | Adversarial playbook + red-team test crate + property tests + CI gate | ~3-4 weeks |
| **Total** | | **~9-10 weeks** |
