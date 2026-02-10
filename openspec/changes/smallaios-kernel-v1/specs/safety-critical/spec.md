# Delta for Safety-Critical

## ADDED Requirements

### Requirement: MISRA-Rust Coding Standards
All safety-critical code SHALL comply with MISRA-Rust coding standards adapted from MISRA-C:2023 for the Rust language.

#### Scenario: No unrestricted unsafe usage
- WHEN a developer writes an unsafe block in safety-critical code
- THEN the block MUST include a documented `// SAFETY:` justification comment
- AND the unsafe code MUST be encapsulated within a safe abstraction boundary
- AND CI MUST fail if any unsafe block lacks a SAFETY comment

#### Scenario: No recursion in safety-critical paths
- WHEN code is designated as safety-critical (DAL A)
- THEN it MUST NOT use unbounded recursion
- AND all call depths MUST be statically analyzable

#### Scenario: No unwrap on fallible operations
- WHEN kernel code calls a function returning Result or Option
- THEN the code MUST use explicit error handling (match, ?, if let) instead of .unwrap() or .expect()
- AND clippy lint clippy::unwrap_used MUST be set to deny for all kernel crates

#### Scenario: Enforce formatting and lint standards
- WHEN code is submitted for review
- THEN cargo fmt --check MUST pass with zero differences
- AND cargo clippy with the project deny-list MUST produce zero warnings

### Requirement: DO-178C DAL A Process Compliance
The development process SHALL comply with DO-178C Design Assurance Level A objectives for all safety-critical kernel components.

#### Scenario: Verify all DAL A objectives
- WHEN the project undergoes certification review
- THEN all DO-178C Table A-1 through A-10 objectives applicable to DAL A MUST be satisfied
- AND evidence of compliance MUST be recorded in the PSAC

#### Scenario: Software Development Plan completeness
- WHEN the DO-178C Software Development Plan is produced
- THEN it MUST define the software lifecycle, development environment, coding standards, review process, and configuration management
- AND it MUST reference the MISRA-Rust coding standard

#### Scenario: Independent verification of safety-critical outputs
- WHEN a safety-critical software component is modified
- THEN verification MUST be performed by a person independent from the developer
- AND verification results MUST be traceable to the requirement being verified
- AND the verification record MUST document verifier identity and date

### Requirement: MC/DC 100% Structural Code Coverage
All safety-critical code paths SHALL achieve 100% Modified Condition/Decision Coverage as required by DO-178C DAL A.

#### Scenario: Achieve MC/DC for conditional logic
- WHEN a safety-critical function contains a boolean decision with multiple conditions
- THEN test cases MUST demonstrate that each condition independently affects the decision outcome
- AND coverage results MUST be recorded and traceable to the function under test

#### Scenario: Coverage gap remediation
- WHEN MC/DC analysis reveals uncovered conditions in safety-critical code
- THEN the project MUST either add test cases to cover the gap or provide a justified deactivated-code rationale
- AND the rationale MUST be approved and documented

#### Scenario: Coverage tool qualification
- WHEN MC/DC coverage is measured
- THEN the coverage tool MUST be qualified per DO-330 TQL-5 for verification tools
- AND tool qualification records MUST be maintained

### Requirement: Requirements Traceability
The project SHALL maintain bidirectional traceability from specification through code implementation to test cases and verification results.

#### Scenario: Trace requirement to implementation and test
- WHEN a new requirement is added to the specification
- THEN a corresponding implementation artifact MUST be created and linked
- AND a corresponding test artifact MUST be created and linked
- AND a verification result MUST be recorded upon test execution
- AND the traceability matrix MUST have zero orphans

#### Scenario: Detect orphan code
- WHEN code exists that is not traced to any requirement
- THEN the traceability analysis MUST flag it as orphan code
- AND it MUST be either traced to a requirement or justified as defensive coding

### Requirement: Static Analysis
All code SHALL pass static analysis with zero findings of undefined behavior and no unjustified unsafe usage.

#### Scenario: No undefined behavior
- WHEN the codebase is analyzed with Miri and clippy
- THEN zero instances of undefined behavior MUST be detected
- AND all integer arithmetic in safety-critical paths MUST use checked or saturating operations

#### Scenario: Unsafe audit compliance
- WHEN the codebase is scanned for unsafe blocks
- THEN every unsafe block MUST have a corresponding SAFETY comment
- AND the total count of unsafe blocks MUST be tracked across releases
- AND unsafe blocks in non-HAL code MUST be zero

### Requirement: Hazard Analysis and Risk Assessment
The project SHALL perform hazard analysis per ARP4761 for all identified system hazards.

#### Scenario: Identify and classify hazards
- WHEN the system design is reviewed for safety
- THEN all hazards related to memory corruption, scheduler deadlock, and incorrect inference MUST be identified
- AND each hazard MUST be classified by severity (catastrophic, hazardous, major, minor, no effect)

#### Scenario: Mitigate catastrophic hazards
- WHEN a hazard is classified as catastrophic or hazardous
- THEN a mitigation strategy MUST be documented in the safety assessment
- AND the mitigation MUST be verified through testing or formal analysis
- AND residual risk MUST be shown to be acceptable per ARP4761 criteria
