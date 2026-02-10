# Spec 12: Safety-Critical Development Standards

## Overview

SmallAIOS follows safety-critical software development practices adapted from
aerospace (DO-178C) and automotive (MISRA) standards. While SmallAIOS is not
certified avionics software, these standards provide the most rigorous framework
for developing high-assurance AI inference systems where reliability matters:
autonomous vehicles, medical devices, industrial control, and critical infrastructure.

## MISRA-Rust Coding Standard

### Adaptation from MISRA-C:2023

MISRA-C:2023 defines ~175 rules for C code. Many are irrelevant to Rust because
the Rust compiler already prevents the corresponding classes of bugs. We adapt
the remaining rules for Rust idioms:

### Rules Enforced by the Rust Compiler (No Action Needed)

| MISRA Rule | Rust Equivalent |
|---|---|
| No null pointer dereference | `Option<T>`, no null pointers |
| No buffer overflow | Bounds-checked indexing |
| No use-after-free | Ownership system |
| No data races | `Send`/`Sync` traits |
| No uninitialized reads | `MaybeUninit` explicit handling |
| No implicit type conversions | `as` requires explicit cast |
| No dangling pointers | Lifetime system |

### Rules Requiring Additional Enforcement

| Rule ID | Description | Enforcement |
|---|---|---|
| MR-001 | No `.unwrap()` or `.expect()` in kernel code | Clippy lint: `clippy::unwrap_used` = deny |
| MR-002 | All `unsafe` blocks require `// SAFETY:` comment | Custom lint / CI check |
| MR-003 | `unsafe` code must be wrapped in safe API | Code review policy |
| MR-004 | No `as` casts that may truncate | Clippy lint: `clippy::cast_possible_truncation` = deny |
| MR-005 | No wildcard imports (`use foo::*`) | Clippy lint: `clippy::wildcard_imports` = deny |
| MR-006 | Integer arithmetic in safety paths must be checked/saturating | Custom lint |
| MR-007 | No recursion in kernel code (stack depth must be bounded) | Code review + static analysis |
| MR-008 | All match arms must be explicit (no wildcard `_` on enums) | Clippy lint: `clippy::wildcard_enum_match_arm` = deny |
| MR-009 | Public functions must have doc comments | `#![warn(missing_docs)]` |
| MR-010 | No `todo!()`, `unimplemented!()` in release builds | Custom lint |

### Clippy Configuration

```toml
# .clippy.toml (or Cargo.toml [lints])
[lints.clippy]
unwrap_used = "deny"
expect_used = "deny"
cast_possible_truncation = "deny"
wildcard_imports = "deny"
wildcard_enum_match_arm = "deny"
panic = "deny"              # No panic! in kernel code
todo = "deny"
unimplemented = "deny"
unreachable = "deny"
indexing_slicing = "warn"   # Prefer .get() for safety-critical paths
```

## DO-178C Process

### Design Assurance Level A

DAL A is the most stringent level, required when software failure could cause
catastrophic consequences. SmallAIOS targets DAL A for:
- All kernel core code (memory, scheduler, syscalls)
- All safety-critical ONNX runtime paths (session management, operator dispatch)
- Capability-based security system
- Cryptographic operations

DAL A requires:
1. **MC/DC structural coverage** (not just statement or branch coverage)
2. **Independence of verification** (developer ≠ verifier)
3. **Requirements-based testing** (tests trace to requirements)
4. **Structural coverage analysis** (tests exercise all code paths)
5. **Formal methods** (where applicable, to supplement testing)

### Required Documents (DO-178C Table A-1)

| Document | Abbreviation | Description |
|---|---|---|
| Plan for Software Aspects of Certification | PSAC | Top-level plan, system overview, lifecycle |
| Software Development Plan | SDP | Dev process, standards, tools, environment |
| Software Verification Plan | SVP | Test strategy, coverage criteria, tools |
| Software Configuration Management Plan | SCMP | Version control, baselines, change management |
| Software Quality Assurance Plan | SQAP | QA process, audits, compliance monitoring |
| Software Requirements Standards | SRS | How requirements are written and managed |
| Software Design Standards | SDS | Architecture and design documentation standards |
| Software Code Standards | SCS | Coding standards (MISRA-Rust) |
| Software Requirements Data | SRD | High-level and low-level requirements |
| Software Design Description | SDD | Architecture and detailed design |
| Source Code | — | The actual Rust source code |
| Software Verification Cases and Procedures | SVCP | Test cases and procedures |
| Software Verification Results | SVR | Test execution results, coverage reports |
| Software Configuration Index | SCI | Configuration of delivered software |
| Software Accomplishment Summary | SAS | Summary of compliance evidence |

### MC/DC Coverage

**Modified Condition/Decision Coverage** requires that:
1. Every entry and exit point is invoked (statement coverage)
2. Every decision takes every possible outcome (decision coverage)
3. Every condition in a decision takes every possible outcome (condition coverage)
4. Every condition independently affects the decision's outcome (MC/DC)

Example:
```rust
if a && (b || c) {
    // ...
}
```

MC/DC requires test cases showing:
- `a` independently affects the outcome (toggle `a` while `b || c` is true)
- `b` independently affects the outcome (toggle `b` while `a` is true and `c` is false)
- `c` independently affects the outcome (toggle `c` while `a` is true and `b` is false)

### Coverage Tooling

- **cargo-llvm-cov**: LLVM-based code coverage with MC/DC support (via `-Cinstrument-coverage` and LLVM 18+ MC/DC instrumentation)
- **grcov**: Alternative coverage aggregator
- **Tool qualification**: The coverage tool must be qualified per DO-330 (Software Tool Qualification) at TQL-5 (verification tool that could fail to detect an error)

## Hazard Analysis (ARP4761)

### Methodology

Follow ARP4761 Functional Hazard Assessment (FHA) for each system function:

1. Identify functions (inference, IPC, memory management, etc.)
2. Identify failure conditions per function
3. Classify severity (Catastrophic, Hazardous, Major, Minor, No Effect)
4. Assign DAL based on severity
5. Define safety requirements to mitigate each hazard

### Key Hazards

| Hazard | Severity | Mitigation | DAL |
|---|---|---|---|
| Memory corruption leads to wrong inference result | Catastrophic | Rust memory safety, bounds checking, MC/DC | A |
| Scheduler deadlock prevents inference | Hazardous | TLA+ verified deadlock freedom, watchdog | A |
| Capability bypass allows unauthorized model execution | Hazardous | Lean 4 proven capability non-forgery | A |
| ONNX parser crash on malicious input | Major | Fuzz testing, input validation, POSIX layer isolation | B |
| GPU failure during inference | Major | CPU fallback, error handling, retry | B |
| Network stack crash from malformed packet | Major | Fuzz testing, firewall, input bounds checking | B |
| Log buffer overflow loses diagnostic data | Minor | Ring buffer, size limits | C |
| Configuration parse error | Minor | Default values, validation at boot | C |
