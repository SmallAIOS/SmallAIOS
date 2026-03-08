## Context

SmallAIOS is a `#![no_std]` Rust OS kernel targeting safety-critical AI inference (DO-178C DAL A, IEC 61508 SIL 3/4, ISO 26262 ASIL D). Current tooling: nightly Rust compiler, TLA+ formal models (19 models, all verified), >93% line coverage via cargo-llvm-cov, SonarCloud SAST, Codecov, and CodeQL. The 2025 safety-critical Rust ecosystem has matured significantly — Ferrocene (TÜV SÜD-qualified compiler), Kani (AWS model checker), and the Safety-Critical Rust Consortium provide tools that did not exist 2 years ago.

Current gaps:
1. No certified compiler (using nightly — not acceptable for certification submission)
2. No formal verification of `unsafe` code (TLA+ covers protocols, not Rust memory safety)
3. No supply chain auditing beyond Dependabot (no license compliance, no banned crate checks)
4. No mutation testing (line coverage alone is insufficient safety evidence)
5. Architecture documentation is ad-hoc (no recognized framework, no NIST traceability)
6. No SPIN/Promela models (only TLA+ — SPIN adds LTL model checking for different properties)
7. No cyclic dependency enforcement across the 18-crate workspace

The user has DoDAF experience but wants a lighter framework for this project. SysML is familiar but potentially heavy. PlantUML is the preferred diagramming tool.

## Goals / Non-Goals

**Goals:**
- Evaluate Ferrocene as the production compiler and document the migration path
- Integrate Kani proofs for all `unsafe` blocks in the codebase
- Add Miri to nightly CI for dynamic UB detection
- Add cargo-deny for supply chain security (advisories, licenses, bans)
- Add cargo-geiger for unsafe surface area tracking
- Add cargo-mutants for mutation testing on critical modules
- Adopt a lightweight architecture framework with PlantUML diagrams
- Add NIST SP 800-53 traceability via sphinx-needs on GitHub Pages
- Add SPIN model checker support alongside TLA+
- Enforce acyclic crate dependency graph
- Integrate with the existing sphinx-needs + GitHub Pages documentation pipeline (from github-pages-v1)

**Non-Goals:**
- Full SysML or DoDAF adoption (too heavy for current team size)
- Replacing TLA+ with SPIN (complementary, not competing)
- Implementing Ferrocene migration now (evaluation and path documentation only)
- Prusti/Creusot adoption (research-grade, evaluate later)
- crates.io publishing (all crates have `publish = false`)

## Decisions

### Decision 1: Architecture framework — 4+1 View Model with PlantUML

**Choice**: Adopt the Kruchten 4+1 architectural view model, rendered in PlantUML, stored in `docs/architecture/`.

**Views**:
- **Logical View** — crate/module decomposition, trait interfaces (component diagrams)
- **Process View** — cooperative scheduler, async task flow, interrupt handling (sequence/activity diagrams)
- **Physical View** — deployment targets: x86-64/AArch64/RISC-V bare metal, container, Jetson (deployment diagrams)
- **Development View** — workspace structure, CI pipeline, build targets (package diagrams)
- **+1 Scenarios** — key use cases: boot → inference, QUIC key exchange, tensor alloc/free (use case diagrams)

**Rationale**: 4+1 is ISO/IEC/IEEE 42010 compatible, well-understood in automotive (AUTOSAR uses similar views), and maps naturally to PlantUML diagram types. Lighter than SysML/DoDAF while providing the architectural completeness needed for safety cases. Each view becomes a sphinx-needs traceable artifact.

**Alternative considered**: AUTOSAR architecture style. Too automotive-specific for a general-purpose safety OS. C4 model. Too focused on web services, doesn't capture hardware/deployment well.

### Decision 2: Ferrocene evaluation before migration

**Choice**: Evaluate Ferrocene compatibility in a separate branch without committing to migration. Document findings in a compatibility report.

**Evaluation steps**:
1. Check Ferrocene target support (x86_64-unknown-none, aarch64-unknown-none, riscv64gc-unknown-none-elf)
2. Test build of all 18 crates with Ferrocene nightly
3. Identify nightly features used that Ferrocene may not support (`naked_functions`, `asm`, `build-std`)
4. Document qualification artifact requirements and cost
5. Produce go/no-go recommendation

**Rationale**: Ferrocene source is MIT/Apache-2.0 on GitHub (they contribute upstream), but prebuilt binaries and certification artifacts require a commercial license. We need to understand compatibility before committing budget.

### Decision 3: Kani for unsafe code, Miri for test suite

**Choice**: Use Kani for proving properties on critical `unsafe` blocks (memory safety, absence of panics). Use Miri for running the full test suite under strict UB detection.

**Kani targets** (initial):
- `kernel/src/mem/` — buddy allocator, slab allocator, tensor pool
- `kernel/src/state.rs` — UnsafeCell-based kernel state
- `arch/*/src/` — boot assembly, page table manipulation
- `security/src/crypto/` — constant-time operations

**Miri integration**: Weekly nightly CI job (Miri requires nightly, runs slower than normal tests).

**Rationale**: Kani provides mathematical proof of safety properties on functions with `unsafe`. Miri catches UB dynamically across the entire test suite. Together they provide defense-in-depth for unsafe code.

### Decision 4: cargo-deny for supply chain, cargo-geiger for unsafe tracking

**Choice**: Add `deny.toml` configuration and integrate both tools into CI.

**cargo-deny configuration**:
- Advisories: deny all known vulnerabilities
- Licenses: allow-list Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause, ISC, Zlib
- Bans: deny GPL/LGPL/AGPL in dependencies, deny known-problematic crates
- Sources: crates.io only (no git dependencies in production)

**cargo-geiger**: Run in CI, produce unsafe usage report. Track unsafe count over time as a metric.

### Decision 5: SPIN/Promela alongside TLA+

**Choice**: Add SPIN models for properties better expressed in LTL (liveness, fairness). Keep TLA+ for safety/invariant properties.

**Division of responsibility**:
- TLA+: Safety properties (invariants, deadlock freedom, bounded state) — existing 19 models
- SPIN: Liveness properties (LTL), protocol conformance, message passing correctness

**Initial SPIN targets**: QUIC handshake protocol, IPC pub/sub delivery guarantees, scheduler fairness.

**Rationale**: TLA+ excels at safety properties but LTL liveness checking is limited (TLC only does bounded checking). SPIN natively supports LTL and can prove liveness properties like "every request eventually gets a response."

### Decision 6: NIST SP 800-53 traceability via sphinx-needs

**Choice**: Map NIST SP 800-53 Rev 5 controls to SmallAIOS implementations using sphinx-needs requirement directives. Render on the GitHub Pages documentation site.

**Approach**:
- Create `docs/nist/` with one RST file per control family (AC, AU, CM, IA, SC, SI, etc.)
- Each control becomes a `.. spec::` or `.. req::` directive with status (implemented/partial/planned)
- Link to implementing code via `.. impl::` directives pointing to crate/module
- sphinx-needs renders a traceability matrix automatically

**Rationale**: NIST SP 800-53 is the de facto framework for federal/defense systems. The existing `cybersecurity-compliance-v3` OpenSpec (complete, 110/110 tasks) already implemented many controls — this adds traceability documentation.

### Decision 7: Cyclic dependency detection

**Choice**: Add a CI check that verifies the crate dependency graph is a DAG (no cycles). Use `cargo metadata` to extract the graph and a simple script to check.

**Rationale**: Cyclic dependencies between workspace crates would indicate architectural coupling. A DAG structure ensures each crate can be tested, built, and reasoned about independently — essential for modular safety certification.

## Risks / Trade-offs

- **[Risk] Ferrocene may not support all nightly features we use** → Mitigation: evaluation-first approach; document feature gaps before committing
- **[Risk] Kani proof maintenance overhead** → Mitigation: start with critical unsafe blocks only; proofs are checked in CI and fail if code changes break them
- **[Risk] Miri is slow (10-100x slower than normal tests)** → Mitigation: run weekly or on-demand, not on every PR
- **[Risk] cargo-mutants is very slow on large codebases** → Mitigation: target critical modules only (crypto, memory, scheduler), not the entire workspace
- **[Risk] SPIN/Promela learning curve** → Mitigation: user has SPIN experience; start with one model and expand
- **[Risk] NIST traceability is extensive (800+ controls)** → Mitigation: focus on control families relevant to an OS kernel (AC, AU, IA, SC, SI), not all 20 families
