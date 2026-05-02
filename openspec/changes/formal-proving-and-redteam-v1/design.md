# Design — formal-proving-and-redteam-v1

This change closes three structural gaps in SmallAIOS verification + security evidence required for DO-178C DAL A traceability:
1. Lean 4 proofs not type-checked in CI ("aspirational" rot).
2. TLA+ corpus missing capability delegation monotonicity + scheduler state-machine fairness; zero SMT coverage.
3. Threat model documented but unexercised — no syscall ABI fuzzer, no PQC NIST KAT, no adversarial test corpus.

The phased structure follows: each phase produces evidence the next builds on. Phase 1 adds the verification scaffolding; Phase 2 adds adversarial coverage of crypto + ABI; Phase 3 unifies it into a red-team gate.

---

## Phase 1: Verification stack expansion — design decisions

### D1.1 Lean 4 CI integration approach

**Decision:** Add a new `lean-verify` job to `.github/workflows/ci.yml` as a **hard gate** alongside `clippy` and `unit-tests`. The job:

1. Installs `elan` (the Lean version manager) via the official `leanprover/elan` install script, pinned to a specific elan release commit for supply-chain stability.
2. Reads the toolchain version from a new `formal/lean4/lean-toolchain` file (single line, e.g. `leanprover/lean4:v4.15.0`). `elan` honors this file automatically. Pinning here keeps the Lean version under PR review.
3. Runs `lake build` from `formal/lean4/` to compile every existing proof (`CapabilityNonForgery`, `InformationFlow`, `IntegrityLattice`, `LabelComposition`, `MessageTypeProperties`, `TensorTypeInvariants`).
4. Caches `formal/lean4/.lake/build/` and `~/.elan/toolchains/` keyed on `lean-toolchain` + `lakefile.lean` hash to keep cold-build (~5–8 min) off the hot path.

**Build tool: `lake`, not `leanpkg`.** `leanpkg` is the deprecated Lean 3 tool; every existing proof in `formal/lean4/` already targets Lean 4, so `lake` is the only valid choice. We also add a `lakefile.lean` to declare the package and any `mathlib` dependency the proofs need.

**Failure mode: hard gate, not advisory.** Lean's type checker is deterministic and proofs that build today will keep building. Demoting to advisory (the current de-facto state) is what produced "aspirational" rot in the first place. The job runs in <2 min once cached, well under the existing CI budget.

**Toolchain version:** pin to `v4.15.0` (latest stable as of 2026-04). Lean 4 is post-1.0 and stable; the cost of a major upgrade later is small relative to the cost of unpinned drift. The pin lives in `formal/lean4/lean-toolchain` so `elan` picks it up implicitly.

### D1.2 TLA+ model additions

**Decision:** Extend `formal/tla/CapabilitySecurity.tla` with two new invariants and add one new model `formal/tla/SchedStateMachine.tla`. `IpcCapabilityTransfer.tla` is **deferred to Phase 2** (out of Phase 1 scope — see risks).

**`CapabilitySecurity.tla` extensions** (DAL A traceable to capability non-forgery requirement):
- `DelegationMonotonicity`: for every delegation edge `parent → child`, `child.rights ⊆ parent.rights`. This is the structural form of "no privilege escalation" and complements the existing non-forgery property by making the lattice ordering explicit on every state.
- `CapabilityIdStrictlyIncreasing`: the global capability ID counter is monotonically non-decreasing across every state transition, and freshly minted IDs are strictly greater than any ID present in the prior state. This rules out ID reuse / replay against revoked caps.

**`SchedStateMachine.tla`** (new model, DAL A traceable to scheduler liveness requirement):
- States: `Ready`, `Running`, `Yielded`, `Blocked`.
- Transitions: `Ready → Running` (dispatch), `Running → Yielded` (cooperative yield at op boundary), `Running → Blocked` (await), `Yielded → Ready` (requeue), `Blocked → Ready` (wake).
- Safety invariants: at most one task in `Running` per core; no `Blocked → Running` transition (must go via `Ready`).
- LTL fairness: `[]<>(task.state = Ready) ⇒ <>(task.state = Running)` under weak fairness on dispatch — every Ready task eventually runs. This is the exact property `docs/scheduling-model.md` claims but never machine-checks.

Bounded with N=3 cores, M=6 tasks. Stays inside the 5-minute TLC budget.

**Why these invariants and not others for DAL A:** capability monotonicity is the keystone of the security argument (a capability system without it cannot claim non-forgery); the scheduler liveness LTL is the keystone of the cooperative-async correctness argument that downstream worst-case execution time analysis depends on. Other potential additions (e.g., buddy-allocator merge correctness, NDP cache liveness) already have TLA+ coverage today.

**`IpcCapabilityTransfer.tla`: out of Phase 1.** Modeling capability passage across IPC rings between cores is genuinely useful but pulls in the multi-core IPC ring spec, which is a 1–2 week effort on its own. Deferred to Phase 2 to keep Phase 1 inside the 3-week budget.

### D1.3 SMT solver choice

**Decision: Z3 first**, behind a new `formal-smt` Cargo feature on the `smallaios-kernel` crate. Add `z3 = "0.12"` (which wraps `z3-sys` and the upstream Z3 4.x C API) as an optional dependency.

**Why Z3 over CVC5 / Bitwuzla:**
- **Maturity of Rust bindings.** `z3-sys` and `z3` (high-level wrapper) are the most-used Rust SMT bindings, regularly published, and have working examples for our likely first-proof patterns (allocator monotonicity, bounded BV reasoning).
- **Theory coverage matches our targets.** Our anticipated proof corpus (allocator pointer monotonicity, capability ID arithmetic, ring-buffer index monotonicity) lives squarely in QF_BV + QF_LIA — Z3's strongest fragments. CVC5's edge in non-linear arithmetic is irrelevant here. Bitwuzla's edge in pure bitvector / crypto-style proofs is real but narrow; we prefer one solver for now and can add Bitwuzla later for the PQC arithmetic phase.
- **Build hygiene.** `z3-sys` can either link a system Z3 (libz3-dev) or build from vendored sources via the `bundled` Cargo feature. We start with `bundled` for hermetic CI builds; revisit if compile time hurts.

**First proof target: `BumpAllocator::alloc` pointer monotonicity.** Encode the bump allocator state (`base`, `current`, `end`) as 64-bit bitvectors and prove: for any sequence of N successful `alloc(size_i)` calls, `current_after ≥ current_before` and `current_after ≤ end`. Bounded model at N=8. Lives in a new `kernel/proofs/bump_allocator_smt.rs` test module guarded by `#[cfg(feature = "formal-smt")]`. Runs in CI under a new `smt-verify` advisory job initially; promotes to hard gate once stable.

The `formal-smt` feature is opt-in and never enabled by default, so the `std`-pulling Z3 dependency stays out of `#![no_std]` kernel builds.

### D1.4 Risks and open questions

- **Lean toolchain drift.** Pinning to `v4.15.0` works today, but a future contributor adding a proof that needs a newer Lean might require a global bump. Mitigation: document the bump-Lean procedure in `formal/lean4/README.md` (Phase 1 task).
- **`mathlib` dependency weight.** If any existing proof imports `mathlib`, cold-build inflates to >30 min. We need to inspect lakefile imports as task 1.1.2 and decide whether to vendor minimal lemma subset vs. accept the cache-warm path.
- **Z3 `bundled` build time.** Bundled Z3 adds ~3–5 min to a clean `formal-smt`-enabled build. Acceptable for an opt-in feature; revisit if it shows up on developer machines.
- **Open: SMT proof framing.** Bump allocator state is small enough that a Lean proof might be cleaner than SMT. We pick SMT here specifically to exercise the scaffolding; the choice between SMT and Lean per-proof becomes a per-target decision in later phases.
- **Open: TLA+ → Lean refinement story.** Both tools cover overlapping ground for capabilities. A future phase will decide whether to express refinement (Lean implementation of the TLA+-modeled state machine) or keep them as independent evidence strands. Out of Phase 1.

---

## Phase 2: Fuzzing expansion + crypto KAT — design decisions

Phase 1 ships verification scaffolding. Two adversarial gaps remain: the **~65-syscall ABI surface** has no fuzzer, and PQC primitives (ML-KEM-768, ML-DSA-65) lack NIST KAT vectors and a differential check. Phase 2 closes both and adds a side-channel guardrail track. Effort: ~3 weeks.

### D2.1 Syscall ABI fuzzer

`fuzz/fuzz_targets/fuzz_syscall_abi.rs` driven by `libfuzzer-sys`. Each iteration owns a `MockKernel` (in-memory capability table with monotone generations, stub allocator, stub IPC ring). Inputs parse via `arbitrary::Arbitrary` into `SyscallCall { number: u32, args: Vec<u8> }` and dispatch through a new `cfg(fuzzing)` shim `kernel::syscall::dispatch_for_fuzz()`.

Per-syscall `Arbitrary` impls cover malformed length prefixes (zero, `usize::MAX`, off-by-one), alignment violations, forged capability handles (random `u64`, expired generation, wrong-type tag), and resource exhaustion (`sys_tensor_alloc(usize::MAX)`, IPC depth 1024). Pass criteria: no panics; `MockKernel` invariants hold; out-of-range numbers return `Err(ENOSYS)`. Lifetime: 60 s PR CI, 1 h nightly; corpus seeded from `kernel/tests/syscall_*.rs`.

### D2.2 PQC differential fuzz

`fuzz/fuzz_targets/fuzz_pqc_mlkem768.rs` and `fuzz_pqc_mldsa65.rs` against **`pqcrypto-kyber = "0.8"`** and **`pqcrypto-dilithium = "0.5"`** as `fuzz/` dev-deps. Rationale: pure-Rust bindings to the official PQClean C reference, already used by rustls-post-quantum and rage; avoids adding `bindgen` + C toolchain.

Equivalence: bit-for-bit on **deterministic** API — both algorithms expose `derand_keypair`, `derand_encapsulate`, `derand_sign`; same seeds → byte-equal outputs. Non-deterministic API uses **functional equivalence** (own encapsulate → ref decapsulate, and reverse). Divergence panics with hex-dumped seeds.

### D2.3 KAT vector ingestion

`security/tests/kat_mlkem768.rs` and `kat_mldsa65.rs`, gated by a new `pqc-kat` feature. Source: NIST `.rsp` files from FIPS 203 / 204 reference packages, vendored under `security/tests/kat-vectors/` (≤ 4 MiB) so CI is hermetic. Pinning: `SHA256SUMS` lockfile; `scripts/refresh-pqc-kats.sh` downloads from canonical NIST URLs and updates the lockfile; mismatched digests abort setup. Parser: ~100 LOC `kat_parser.rs` (line-oriented `KEY = HEX`). CI gate: new **blocking** `pqc-kat-verify` job.

### D2.4 Side-channel scope

In scope (advisory): `docs/pqc-side-channel.md` documenting CT invariants (NTT loops, rejection sampling, comparison via `subtle::ConstantTimeEq`); leaky ops tagged `// NOT-CT:` with rationale.

Stretch (advisory CI): `dudect-rs` tests, weekly `pqc-timing-leak` cron on `self-hosted-isolated`, report-only (t > 4.5 → warning). Blocking promotion deferred to Phase 3. Formal CT proofs (`ct-verif` / Jasmin) DEFERRED to a follow-up change.

### D2.5 Risks and open questions

| Risk | Mitigation |
|------|-----------|
| `pqcrypto-kyber` has a bug we duplicate | Cross-check vs NIST KAT — three independent oracles |
| KAT files exceed repo size | `zstd -19`; uncompressed ≈ 3.2 MiB |
| `MockKernel` drift masks real bugs | Phase 3 Kani on real `dispatch()`; Phase 2 fuzzer is coverage, not oracle |
| `dudect` flakes on shared CI | Dedicated `self-hosted-isolated` runner; weekly cadence absorbs noise |

Open questions: (1) vendor `.rsp` vs Git submodule? *Lean: vendor.* (2) Fuzz `verified-boot` syscalls? *Lean: no — measurement-log writes are idempotent.* (3) Use `arbitrary-derive`? *Lean: yes.*

---

## Phase 3: Red-team / adversarial — design decisions

Effort: ~3-4 weeks. Drives adversarial evidence for the threat model in `openspec/smallaios-kernel/specs/06-security-model.md` and the DAL A audit trail required by `openspec/smallaios-kernel/specs/12-safety-critical.md`.

### D3.1 Test crate structure

**Options:**
- (A) New workspace crate `crates/red-team-tests/` with its own `Cargo.toml`, depending on `kernel`, `security`, `onnx-rt`, `ipc`, `net`, `container`.
- (B) Co-located `tests/red-team/` directory inside each affected crate.

**Recommendation: (A) `crates/red-team-tests/`.** Adversarial tests cross crate boundaries (capability forgery touches `security` + `ipc`; ONNX injection touches `onnx-rt` + `kernel`). A dedicated crate gives one CI gate (`red-team-suite`), one corpus path (`crates/red-team-tests/corpus/`), and a single `lib.rs` of attack helpers. Trade-off: increases the layer-3 fan-in slightly; mitigated by gating compilation behind a `red-team` feature so production builds skip it.

### D3.2 Attack scenario taxonomy

Scenarios are grouped by threat-model line in `06-security-model.md`:

| Category | Threat-model ref | Test module |
|----------|------------------|-------------|
| Capability forgery / over-delegation | T-01, T-04 | `capability::forge` |
| ONNX graph injection | T-02 (malicious ONNX) | `onnx::inject` |
| IPC amplification / DoS | T-07 (DoS) | `ipc::flood` |
| Boot integrity tampering | T-03 (boot) | `boot::tamper` |
| Network protocol abuse | T-05 (network) | `net::abuse` |

**Gating policy.** `red-team-suite` runs as a **non-blocking advisory job** for the first 3 PRs that touch it (matching the cargo-careful pattern in `.github/workflows/ci.yml`). Promoted to **blocking** once the suite is green for 5 consecutive `develop` builds. Promotion is recorded in `docs/red-team-playbook.md` § "Gating history".

### D3.3 Property-test framework choice

**Options: `proptest` vs `quickcheck`.**

**Recommendation: `proptest`.** Reasons:
- Already implied by the existing fuzz corpus tooling; shrinking is more predictable than `quickcheck` for kernel state machines.
- `proptest-state-machine` models cascading capability revocation cleanly.
- `no_std`-compatible via `proptest = { default-features = false }` for use inside `kernel` test harness.

Properties (4-6 initial):
1. `prop_delegation_subset_rights` — child rights ⊆ parent rights.
2. `prop_revocation_cascades` — revoking parent invalidates all transitive children within one tick.
3. `prop_cap_id_unique` — capability IDs never reused after revocation.
4. `prop_no_upgrade_via_alias` — aliasing a capability cannot grant rights the original lacked.
5. `prop_quota_monotone` — delegated quota never exceeds parent quota.
6. `prop_audit_log_complete` — every cap operation appears in the audit log exactly once.

### D3.4 Attack-surface tracking doc

`docs/attack-surface.md` is a single Markdown table with columns:

| Interface | Source file | Threat-model line | Adversarial coverage | Owner |

Rows enumerate: ~46 syscalls (one row each, grouped), TCP/UDP listen ports, IPC topics (per-topic ACL), USB endpoint classes, CAN routes (see `docs/can-inference.md`), GPU device ABI (CUDA stub today). Each row links to a test in `crates/red-team-tests/` or marks coverage as `NONE` / `FUZZ-ONLY` / `PROPERTY` / `INTEGRATION`. The doc is regenerated quarterly and reviewed by Security Lead per `docs/security/security-governance.md`.

### D3.5 Risks and open questions

- **R1 — flaky DoS tests.** Topic-flood timing is sensitive to CI runner load. Mitigation: tests assert behavior (refuse / drop / audit), not throughput.
- **R2 — corpus size growth.** Malformed protobuf / TCP corpora can bloat the repo. Mitigation: store seeds only; expand at runtime.
- **R3 — boot integrity coverage gap.** Per `docs/boot-security-matrix.md` most boot mechanisms are "No" or "Partial". Phase 3 covers Multiboot2 self-measurement only; UEFI / TPM tampering deferred until those mechanisms move past "Partial".
- **OQ1** — should `red-team-suite` run on `main` only, or also on `develop`? Recommend both, advisory on PRs.
- **OQ2** — do we mirror property tests into Kani harnesses? Defer to a follow-up phase.
