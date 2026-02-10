# Delta for Supply Chain Security

## ADDED Requirements

### Requirement: CycloneDX SBOM Generation
The build system SHALL generate a CycloneDX SBOM for every release build including all Rust crate dependencies with versions and license information.

#### Scenario: SBOM generated on release build
- WHEN a release build is triggered in the CI pipeline
- THEN the build system MUST produce a CycloneDX-format SBOM file as a build artifact
- AND the SBOM MUST list every Rust crate dependency (direct and transitive) with its name, version, and SPDX license identifier

#### Scenario: SBOM includes SmallAIOS workspace crates
- WHEN the SBOM is generated
- THEN it MUST include all 10 SmallAIOS workspace crates (kernel, arch/x86_64, arch/aarch64, arch/nvidia, onnx-rt, ipc, net, posix, security, container) with their current versions
- AND each crate entry MUST list its direct dependencies

#### Scenario: SBOM completeness validation
- WHEN the SBOM is generated
- THEN automated validation MUST verify that the number of SBOM entries matches the total dependency count from `cargo metadata`
- AND any discrepancy MUST cause the build to fail with a descriptive error message

#### Scenario: SBOM includes build environment metadata
- WHEN the SBOM is generated
- THEN it MUST include build environment metadata: Rust toolchain version, target triple, build timestamp (UTC), and git commit SHA
- AND the SBOM format version MUST be CycloneDX 1.5 or later

### Requirement: Dependency Vulnerability Audit
The CI pipeline SHALL run cargo-audit on every commit and fail the build if known vulnerabilities with CVSS >= 7.0 are found in dependencies.

#### Scenario: Build fails on high-severity vulnerability
- WHEN cargo-audit detects a dependency with a known vulnerability having CVSS score >= 7.0
- THEN the CI pipeline MUST fail the build
- AND the build output MUST report the vulnerable crate name, version, CVE identifier, CVSS score, and advisory URL

#### Scenario: Build succeeds with low-severity vulnerabilities
- WHEN cargo-audit detects only vulnerabilities with CVSS score < 7.0
- THEN the CI pipeline MUST allow the build to succeed
- AND the build output MUST emit a warning listing each low-severity vulnerability for tracking

#### Scenario: Audit runs against current advisory database
- WHEN cargo-audit executes in CI
- THEN it MUST use the latest RustSec advisory database (updated within the last 24 hours)
- AND the audit report MUST include the advisory database revision used

#### Scenario: No dependencies present (clean-room verification)
- WHEN the dependency tree contains only SmallAIOS workspace crates and Rust core/alloc
- THEN cargo-audit MUST still execute successfully and report zero third-party advisories
- AND this result MUST be logged as confirmation of the clean-room policy

### Requirement: Reproducible Builds
The build system SHALL support reproducible builds: same source + same toolchain + same target SHALL produce bit-identical output.

#### Scenario: Verify bit-identical output across builds
- WHEN the same source tree is built twice with the same Rust nightly toolchain version and the same target triple
- THEN the resulting binary artifacts MUST be bit-identical (identical SHA-256 hash)
- AND the CI pipeline MUST include a reproducibility verification step that performs this comparison

#### Scenario: Identify reproducibility-breaking changes
- WHEN a code change introduces non-determinism into the build (e.g., embedded timestamps, random values, non-deterministic link order)
- THEN the reproducibility verification step MUST detect the discrepancy and fail the build
- AND the failure message MUST identify the differing binary sections to aid debugging

#### Scenario: Document reproducibility prerequisites
- WHEN an external party attempts to reproduce a SmallAIOS build
- THEN the build documentation MUST specify the exact toolchain version (rustup toolchain), target triple, environment variables, and build command required to reproduce the output
- AND the documentation MUST be included alongside every release

### Requirement: Hardware Vendor Assessment Checklist
The system SHALL document a vendor assessment checklist for hardware components (GPU, bus transceivers, FPGA, SoC) covering supply chain origin, firmware update process, and known vulnerability history.

#### Scenario: Assess a new GPU vendor
- WHEN a new GPU model (e.g., NVIDIA Maxwell through Blackwell) is evaluated for SmallAIOS support
- THEN the vendor assessment checklist MUST be completed covering: manufacturer and country of origin, firmware update mechanism and signing process, known CVE history for the component, end-of-life and support timeline, and availability of public datasheets or programming references

#### Scenario: Assess a bus transceiver vendor
- WHEN a bus transceiver (CAN, ARINC, MIL-STD-1553, SpaceWire) is evaluated
- THEN the vendor assessment checklist MUST document the hardware supply chain (fabrication facility, assembly location), available firmware update process (if applicable), and any known hardware errata or vulnerabilities

#### Scenario: Periodic reassessment of approved vendors
- WHEN a previously assessed vendor's component is included in a new SmallAIOS release
- THEN the vendor assessment MUST be reviewed at least annually or upon disclosure of a new vulnerability affecting the component
- AND the review MUST be documented with the review date, reviewer, and disposition

#### Scenario: Vendor assessment blocks unapproved hardware
- WHEN a hardware component does not have a completed vendor assessment checklist
- THEN the component MUST NOT be added to the SmallAIOS supported hardware list (Tier 1 or Tier 2)
- AND any pull request adding support for an unassessed component MUST be rejected by the CCB

### Requirement: Clean-Room Policy for Third-Party Code
All third-party code SHALL be zero: SmallAIOS maintains a clean-room policy with no vendored or copy-pasted external code.

#### Scenario: Verify zero vendored code in repository
- WHEN the CI pipeline runs on each commit
- THEN an automated check MUST verify that no vendored or copy-pasted third-party source files exist in the repository
- AND the check MUST scan for common vendoring patterns (vendor/ directories, license headers from external projects, copy-paste attribution comments)

#### Scenario: Reject PR introducing third-party code
- WHEN a pull request introduces source code originating from an external project
- THEN the CI pipeline MUST detect the external origin (via license header scanning, code provenance analysis, or maintainer attestation)
- AND the PR MUST be rejected with a message referencing the clean-room policy

#### Scenario: Clean-room attestation per release
- WHEN a release build is produced
- THEN the release artifacts MUST include a signed clean-room attestation document confirming that all code was developed from public specifications, standards documents, and original engineering without reference to proprietary implementations
- AND the attestation MUST list all reference documents used (e.g., ONNX specification, ISO 11898, ARINC 429 standard, ARM Architecture Reference Manual)

### Requirement: Signed Build Attestation
Build artifacts SHALL include a signed build attestation (ML-DSA-65) linking the SBOM to the specific binary hash.

#### Scenario: Generate signed attestation for release build
- WHEN a release build completes successfully
- THEN the build system MUST generate a build attestation document containing: the SHA-256 hash of each binary artifact, a reference to the CycloneDX SBOM (by hash), the git commit SHA, the build timestamp (UTC), and the builder identity
- AND the attestation MUST be signed using ML-DSA-65 with the project's release signing key

#### Scenario: Verify attestation signature
- WHEN a consumer receives SmallAIOS build artifacts
- THEN the consumer MUST be able to verify the ML-DSA-65 signature on the build attestation using the project's published public key
- AND signature verification failure MUST cause the consumer's deployment tooling to reject the artifacts

#### Scenario: Attestation links SBOM to binary
- WHEN the signed build attestation is inspected
- THEN it MUST contain the SHA-256 hash of the CycloneDX SBOM and the SHA-256 hash of each binary artifact
- AND a verifier MUST be able to confirm that the SBOM referenced in the attestation matches the SBOM distributed with the build

#### Scenario: Attestation covers all target architectures
- WHEN a release is built for multiple targets (x86_64, AArch64, NVIDIA)
- THEN a separate signed attestation MUST be generated for each target-specific binary
- AND each attestation MUST reference the same SBOM (since dependencies are shared) but a distinct binary hash
