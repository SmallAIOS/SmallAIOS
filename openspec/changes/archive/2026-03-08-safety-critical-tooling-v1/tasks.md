## 1. Ferrocene Evaluation

- [x] 1.1 Audit all nightly features used across the workspace (`naked_functions`, `asm`, `build-std`, etc.) and document in `docs/ferrocene-eval.md`
- [x] 1.2 Check Ferrocene target support for x86_64-unknown-none, aarch64-unknown-none, riscv64gc-unknown-none-elf
- [x] 1.3 Attempt workspace build with Ferrocene toolchain (if available) and document results
- [x] 1.4 Document qualification artifact requirements, commercial license costs, and go/no-go recommendation

## 2. Kani Model Checking

- [x] 2.1 Add Kani as a dev dependency and create initial proof harness structure in `kani/` or inline `#[kani::proof]`
- [x] 2.2 Write Kani proofs for buddy allocator (`kernel/src/mem/buddy.rs`) — no panics, no OOB
- [x] 2.3 Write Kani proofs for slab allocator (`kernel/src/mem/slab.rs`) — no panics, no OOB
- [x] 2.4 Write Kani proofs for tensor pool (`kernel/src/mem/tensor.rs`) — handle validity, no double-free
- [x] 2.5 Write Kani proofs for constant-time operations (`security/src/crypto/constant_time.rs`)
- [x] 2.6 Add Kani verification job to `.github/workflows/ci.yml`

## 3. Miri UB Detection

- [x] 3.1 Add weekly Miri CI job to `.github/workflows/ci.yml` (schedule: weekly, nightly toolchain)
- [x] 3.2 Run Miri locally on all host-testable crates and fix any UB findings
- [x] 3.3 Document Miri-incompatible tests (if any) and add `#[cfg_attr(miri, ignore)]` annotations

## 4. Supply Chain Security (cargo-deny)

- [x] 4.1 Create `deny.toml` with license allow-list (Apache-2.0, MIT, BSD-2/3-Clause, ISC, Zlib)
- [x] 4.2 Configure advisory checks (deny all RustSec advisories)
- [x] 4.3 Configure ban rules (no GPL/LGPL/AGPL transitive deps)
- [x] 4.4 Add cargo-deny CI job to `.github/workflows/ci.yml`
- [x] 4.5 Add cargo-geiger CI job that produces an unsafe usage report artifact

## 5. Mutation Testing

- [x] 5.1 Run cargo-mutants on `security/src/crypto/` and document baseline mutation score
- [x] 5.2 Run cargo-mutants on `kernel/src/mem/` and document baseline mutation score
- [x] 5.3 Add on-demand mutation testing CI job (manual trigger)
- [x] 5.4 Document surviving mutants and create follow-up test tasks

## 6. Architecture Documentation (4+1 Views)

- [x] 6.1 Create `docs/architecture/` directory structure with README explaining the 4+1 view model
- [x] 6.2 Create logical view PlantUML diagram (`logical-view.puml`) showing all 18 crates and dependency relationships
- [x] 6.3 Create process view PlantUML diagram (`process-view.puml`) showing boot → scheduler → inference flow
- [x] 6.4 Create physical view PlantUML diagram (`physical-view.puml`) showing all deployment targets
- [x] 6.5 Create development view PlantUML diagram (`development-view.puml`) showing CI pipeline and build matrix
- [x] 6.6 Create scenario diagrams for key use cases (boot-to-inference, QUIC handshake, tensor lifecycle)

## 7. Cyclic Dependency Detection

- [x] 7.1 Create `scripts/check-cycles.sh` that uses `cargo metadata` to extract workspace dependency graph and verify it is a DAG
- [x] 7.2 Add cyclic dependency check to CI pipeline
- [x] 7.3 Verify current workspace has no cycles

## 8. NIST SP 800-53 Traceability

- [x] 8.1 Create `docs/nist/` directory with index.rst listing applicable control families
- [x] 8.2 Create Access Control (AC) family traceability RST with sphinx-needs directives
- [x] 8.3 Create System & Communications Protection (SC) family traceability RST
- [x] 8.4 Create Identification & Authentication (IA) family traceability RST
- [x] 8.5 Create Audit & Accountability (AU) family traceability RST
- [x] 8.6 Create System & Information Integrity (SI) family traceability RST
- [x] 8.7 Add traceability matrix rendering to sphinx conf.py (needs integration with github-pages-v1)

## 9. SPIN Model Checker Integration

- [x] 9.1 Create `formal/promela/` directory structure with README and Makefile
- [x] 9.2 Write QUIC handshake Promela model with LTL liveness property
- [x] 9.3 Write IPC pub/sub delivery Promela model with LTL eventual delivery property
- [x] 9.4 Write scheduler fairness Promela model with LTL no-starvation property
- [x] 9.5 Add SPIN verification job to CI (install spin, compile + verify all models)
- [x] 9.6 Document SPIN vs TLA+ division of responsibility in `formal/README.md`
