# SmallAIOS MISRA-Rust Coding Standard

SPDX-License-Identifier: Apache-2.0

## Overview

SmallAIOS follows a MISRA-Rust inspired coding standard targeting
DO-178C DAL A certification with MC/DC 100% structural coverage.

## Mandatory Rules

### M1: No Unsafe Without Justification
All `unsafe` blocks must have a `// SAFETY:` comment explaining why the
invariants hold. Unjustified unsafe is a build failure.

### M2: No Panicking in Kernel Code
Functions callable from interrupt context or the scheduler must not panic.
Use `Result<T, E>` for fallible operations.

### M3: No Dynamic Allocation After Init
After the boot phase completes, no new heap allocations are permitted
in the hot path. Pre-allocate all buffers during initialization.

### M4: No Recursion
Recursive functions are prohibited. Use iterative algorithms with explicit
stacks to maintain bounded stack usage.

### M5: Bounded Loops
All loops must have a provable upper bound. Use `for` loops with ranges
rather than `while` with manual conditions where possible.

### M6: No Floating Point in Kernel
The kernel proper must not use floating-point operations. SIMD/FP is
permitted only in the ONNX runtime execution providers.

### M7: Error Handling
All errors must be explicitly handled. Unwrap/expect are prohibited
outside of test code. Use `?` operator or explicit match.

### M8: Function Complexity
Functions must not exceed cognitive complexity of 25 (enforced by clippy).
Functions must not exceed 100 lines.

### M9: No Wildcard Imports
`use module::*` is prohibited. All imports must be explicit.

### M10: Documentation
All public items must have documentation comments. Safety invariants
for unsafe code must be documented.

## Clippy Enforcement

The following clippy lint groups are enforced as errors:

```toml
# In workspace Cargo.toml or via RUSTFLAGS
-D clippy::all
-D clippy::pedantic
-D clippy::nursery
-D clippy::cargo

# Specific critical lints
-D clippy::unwrap_used
-D clippy::expect_used
-D clippy::panic
-D clippy::indexing_slicing
-D clippy::arithmetic_side_effects
-D clippy::float_arithmetic       # M6
-D clippy::wildcard_imports        # M9
-D clippy::missing_docs_in_private_items  # M10
-D clippy::undocumented_unsafe_blocks     # M1
```

## MC/DC Coverage Requirements

- All decision outcomes must be tested (decision coverage)
- All condition outcomes must independently affect the decision (MC/DC)
- Target: 100% MC/DC for all kernel code paths
- Tool: cargo-llvm-cov with MC/DC instrumentation

## Traceability

Every requirement in the OpenSpec specs must trace to:
1. Design decision in design docs
2. Implementation in source code
3. Test case(s) that verify the requirement
4. MC/DC coverage evidence

This traceability matrix is maintained via Sphinx-needs directives.
