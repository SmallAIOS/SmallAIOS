## ADDED Requirements

### Requirement: Code Quality Triage Guide

A document at `docs/code-quality.md` SHALL exist and SHALL be linked from the contributor-facing guide. The document SHALL explain the difference between GitHub Code Scanning alerts and the GitHub Code Quality view, the project's CodeQL suppression policy, the test-vector module convention, and a triage checklist for contributors who encounter a CodeQL finding.

#### Scenario: Document is present and linked

- **WHEN** a contributor opens the project's contributor-facing guide (CONTRIBUTING.md, CLAUDE.md, or equivalent)
- **THEN** they SHALL find a link to `docs/code-quality.md`
- **AND** clicking the link SHALL navigate to the document

#### Scenario: Document covers required sections

- **WHEN** a reviewer reads `docs/code-quality.md`
- **THEN** the document SHALL contain at least the following named sections:
  - "Code Scanning Alerts vs Code Quality View" — explains the two surfaces and how to access each
  - "Suppression Policy" — explains the choice between fixing, extracting to a test-vector module, applying an inline annotation, and escalating
  - "Test-Vector Module Convention" — explains `*_test_vectors.rs` naming and the corresponding CodeQL config exclusion
  - "Triage Checklist" — a step-by-step "before you suppress, ask…" decision tree

#### Scenario: Document references the suppression-policy capability

- **WHEN** a reviewer reads `docs/code-quality.md`
- **THEN** the document SHALL reference (by name or via cross-link) the `codeql-suppression-policy` capability spec as the source of truth for the normative rules
