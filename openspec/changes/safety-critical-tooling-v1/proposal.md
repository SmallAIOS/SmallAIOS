## Why

SmallAIOS targets safety-critical deployment (DO-178C DAL A, IEC 61508 SIL 3/4, ISO 26262 ASIL D). The codebase has TLA+ formal models and >93% line coverage, but lacks integration of Rust-specific safety tooling that has matured significantly in 2025: Kani model checking, Miri UB detection, cargo-geiger unsafe tracking, cargo-deny supply chain auditing, and mutation testing. Additionally, Ferrocene (the first TÜV SÜD-qualified Rust compiler) should be evaluated as the production compiler. The architecture documentation needs to follow a recognized framework with PlantUML diagrams and NIST SP 800-53 traceability via sphinx-needs on the GitHub Pages site.

## What Changes

- Evaluate and integrate Ferrocene as the certified compiler toolchain (or document compatibility path)
- Add Kani model checking for critical `unsafe` code paths (memory safety proofs, absence of panics)
- Add Miri to CI for dynamic UB detection on the test suite
- Integrate cargo-geiger for unsafe surface area tracking
- Add cargo-deny for license compliance, dependency bans, and advisory checks
- Add cargo-mutants for mutation testing evidence
- Evaluate cargo-semver-checks for internal API stability
- Adopt a safety-oriented architecture framework (4+1 / AUTOSAR-style / SysML-lite) with PlantUML diagrams
- Add NIST SP 800-53 traceability via sphinx-needs directives on the documentation site
- Add SPIN model checker support alongside existing TLA+ models
- Verify no cyclic dependencies across crates (enforce DAG structure)
- Evaluate Prusti/Creusot for deductive verification on high-assurance crypto modules

## Capabilities

### New Capabilities
- `ferrocene-evaluation`: Evaluate Ferrocene compiler for safety-critical certification — covers compatibility testing, qualification document review, and migration path from nightly to Ferrocene
- `formal-verification-ci`: Kani + Miri integration in CI pipeline — covers Kani proofs on critical unsafe code, Miri nightly test runs, and proof maintenance strategy
- `supply-chain-ci`: cargo-deny + cargo-audit + cargo-geiger in CI — covers dependency advisory checks, license compliance, banned crate lists, and unsafe surface area metrics
- `mutation-testing`: cargo-mutants integration for test quality evidence — covers mutation score tracking, critical module targeting, and safety case evidence generation
- `architecture-framework`: Safety-critical architecture documentation with PlantUML — covers adoption of a view-based architecture framework, component/deployment/sequence diagrams, and cyclic dependency detection
- `nist-traceability`: NIST SP 800-53 control traceability via sphinx-needs — covers mapping security controls to implementation, requirement IDs, and rendered traceability matrix on GitHub Pages
- `spin-integration`: SPIN model checker support for concurrent protocol verification — covers Promela model creation, integration alongside TLA+, and CI verification

### Modified Capabilities
None — this change adds tooling and documentation without modifying existing specs.

## Impact

- `.github/workflows/ci.yml` — add Kani, Miri, cargo-deny, cargo-geiger, cargo-mutants jobs
- `Cargo.toml` — potential toolchain changes for Ferrocene evaluation
- `rust-toolchain.toml` — document Ferrocene compatibility path
- `deny.toml` — cargo-deny configuration (licenses, advisories, bans)
- `docs/` — architecture diagrams (PlantUML), NIST traceability (sphinx-needs)
- `formal/promela/` — new SPIN/Promela models alongside existing TLA+
- `scripts/` — cyclic dependency checker, unsafe surface area reporter
- OpenSpec artifacts and documentation across all changes
