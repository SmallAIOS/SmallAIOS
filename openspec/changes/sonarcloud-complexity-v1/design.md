## Context

SonarCloud reports 21 functions exceeding cognitive complexity 15 (rule `rust:S3776`). These functions are spread across 8 crates but share common complexity patterns: deeply nested loops, large match-based state machines with repeated conditional branches, and procedural algorithms with multi-level coordinate/index tracking.

All 21 functions are in the existing codebase on `develop` — none are new regressions. The project passes SonarCloud's quality gate today because these are pre-existing, but they represent the entirety of SmallAIOS's code smell inventory.

### Complexity patterns observed

| Pattern | Functions | Example |
|---------|-----------|---------|
| Nested loops (4–7 levels) | op_conv, op_add, op_softmax, ml_dsa_65_sign | op_conv has 6 nested for-loops for batch/channel/spatial/kernel |
| Large match + repeated conditionals | on_segment, ModeCodeProcessor::process, LongHeader::decode | TCP state machine re-checks RST/ACK/FIN flags in each match arm |
| State machine with inline handling | parse_dtb, KernelAllocator::alloc, Scheduler::poll | DTB parser mixes token dispatch with property parsing and depth tracking |
| Coordinate iteration with broadcast | op_add, op_softmax, op_reshape, plan_memory | Carry-based coordinate increment + dual broadcast stride computation |

## Goals / Non-Goals

**Goals:**

- Reduce all 21 functions to cognitive complexity ≤ 15
- Zero SonarCloud code smells after completion
- No public API changes — extracted helpers are `fn` or `pub(crate)`, never `pub`
- All existing tests pass without modification
- Extracted helpers get their own unit tests where behavior is independently testable

**Non-Goals:**

- Performance optimization (refactoring must be performance-neutral; no algorithmic changes)
- Adding new features or changing behavior
- Refactoring functions that are already ≤ 15 complexity
- Reducing cyclomatic complexity (only cognitive complexity is targeted)
- Changing module structure or file layout (helpers stay in the same file)

## Decisions

### D1: Extract helpers into the same file, not new modules

**Choice:** New helper functions are placed as private `fn` in the same file as the parent function.

**Rationale:** Keeps the refactoring minimal and avoids churn in `mod.rs` declarations. Each helper is tightly coupled to its parent and not reusable elsewhere. The only exception is ONNX broadcast iteration (see D3).

**Alternative considered:** Creating `helpers.rs` submodules — rejected because it adds module boilerplate for functions used in exactly one place.

### D2: Decompose state machines into per-state handler methods

**Choice:** For `TcpConnection::on_segment()`, `ModeCodeProcessor::process()`, and `LongHeader::decode()`, extract each match arm into a dedicated method (e.g., `handle_listen()`, `handle_syn_sent()`).

**Rationale:** Each state's logic is independent. Extracting to methods reduces the match body to single-line dispatches while making each state's logic testable in isolation. This is a well-established pattern for TCP implementations.

**Alternative considered:** Using a dispatch table (array of function pointers) — rejected because Rust's enum-based match with exhaustiveness checking is safer and more idiomatic.

### D3: Shared broadcast coordinate iterator for ONNX operators

**Choice:** Extract a `BroadcastIter` utility struct (in `onnx-rt/src/operators.rs`) that encapsulates the carry-based coordinate increment + dual-stride broadcast index computation used by op_add, op_softmax, op_reshape, and plan_memory.

**Rationale:** Four functions share the same ~20-line coordinate iteration pattern with minor variations. A shared iterator eliminates the primary complexity driver from all four functions simultaneously.

**Alternative considered:** Inline helper per function — would reduce complexity individually but misses the shared pattern and leads to 4 near-identical helpers.

### D4: Extract inner kernel computation for nested-loop operators

**Choice:** For `op_conv()`, extract the innermost loop body (input_channels × kernel_h × kernel_w summation) into a `convolve_at()` helper that takes batch, output channel, and output coordinates.

**Rationale:** The 6-level nesting is the sole complexity driver. Splitting at the natural computation boundary (per-output-pixel) reduces the main function to 3 loops while keeping the inner computation as a pure function.

### D5: Extract ML-DSA rejection sampling phases

**Choice:** For `ml_dsa_65_sign()` (complexity 58), decompose into:
- `prepare_ntt_vectors()` — NTT forward transforms on s1, s2, t0
- `compute_challenge()` — matrix multiplication + challenge hash
- `check_signature_norms()` — z norm, low_bits, ct0 checks
- `pack_ml_dsa_signature()` — final signature encoding

For `ml_dsa_65_verify()` (complexity 19), extract `reconstruct_and_check()`.

**Rationale:** The signing function performs 4 distinct phases inside a rejection loop. Each phase is independently testable and the phase boundaries are natural (transforms, computation, validation, encoding).

### D6: Extract DTB token handlers

**Choice:** For `parse_dtb()`, extract `handle_begin_node()`, `handle_prop()`, and `handle_end_node()` as closures or helper functions, plus `is_memory_node()` and `parse_reg_property()` predicates.

**Rationale:** The parser loop mixes token dispatch with property interpretation and depth tracking. Separating concerns flattens the nesting from 5 levels to 2.

### D7: Flatten allocator fallback chain

**Choice:** For `KernelAllocator::alloc()`, extract `try_slab_alloc()` and `try_buddy_alloc()` helpers that return `Option<NonNull<u8>>`, then chain with `or_else`.

**Rationale:** The current 4-level nesting (try slab → grow slab from buddy → try slab again → try buddy directly) becomes a flat `try_slab().or_else(|| grow_and_retry()).or_else(|| try_buddy())` chain.

## Risks / Trade-offs

**[Risk] Refactoring changes observable behavior** → Mitigation: All existing tests must pass. No algorithmic changes — only structural extraction. PR CI gate includes full test suite + clippy.

**[Risk] Extracted helpers increase total line count** → Mitigation: Acceptable trade-off. Cognitive complexity measures human readability, not code size. Expected ~10-15% line count increase per refactored function.

**[Risk] SonarCloud complexity calculation differs from manual estimate** → Mitigation: After each crate's refactoring, run SonarCloud analysis on a draft PR to verify the complexity scores actually dropped. Iterate if needed.

**[Risk] Crypto function refactoring introduces subtle bugs** → Mitigation: ML-DSA and SHA-3 have extensive test vectors. The refactoring extracts phases without changing computation order. Run the full crypto test suite after each change.

**[Trade-off] BroadcastIter adds an abstraction** → The shared iterator is justified by 4 consumers. It replaces duplicated complexity with a single, well-tested utility. If a future operator needs different iteration, it can use its own helper without touching BroadcastIter.
