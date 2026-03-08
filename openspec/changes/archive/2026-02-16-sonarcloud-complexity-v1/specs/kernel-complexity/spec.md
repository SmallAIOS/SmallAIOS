## ADDED Requirements

### Requirement: parse_dtb cognitive complexity ≤ 15
The `parse_dtb()` function in `kernel/src/mem/phys.rs` SHALL have cognitive complexity ≤ 15. DTB token handling SHALL be extracted into per-token helper functions (`handle_begin_node()`, `handle_prop()`, `handle_end_node()`). Memory node detection and register property parsing SHALL be extracted into predicate helpers. All existing DTB parsing tests SHALL continue to pass.

#### Scenario: parse_dtb refactored below threshold
- **WHEN** SonarCloud analyzes `parse_dtb()` after refactoring
- **THEN** the cognitive complexity score SHALL be ≤ 15 (currently 32)

#### Scenario: DTB parsing behavior preserved
- **WHEN** existing DTB parsing tests execute against the refactored code
- **THEN** all memory region detection results SHALL be identical

### Requirement: KernelAllocator alloc cognitive complexity ≤ 15
The `KernelAllocator::alloc()` (GlobalAlloc impl) function in `kernel/src/mem/global.rs` SHALL have cognitive complexity ≤ 15. The slab/buddy fallback chain SHALL be extracted into `try_slab_alloc()` and `try_buddy_alloc()` helpers composed with `or_else`. All existing allocator tests SHALL continue to pass.

#### Scenario: KernelAllocator alloc refactored below threshold
- **WHEN** SonarCloud analyzes `KernelAllocator::alloc()` after refactoring
- **THEN** the cognitive complexity score SHALL be ≤ 15 (currently 29)

#### Scenario: Allocation fallback behavior preserved
- **WHEN** existing allocator tests (including OOM and fallback scenarios) execute
- **THEN** all tests SHALL pass with the same allocation outcomes

### Requirement: trace_status cognitive complexity ≤ 15
The `trace_status()` function in `kernel/src/safety/traceability.rs` SHALL have cognitive complexity ≤ 15. All existing traceability tests SHALL continue to pass.

#### Scenario: trace_status refactored below threshold
- **WHEN** SonarCloud analyzes `trace_status()` after refactoring
- **THEN** the cognitive complexity score SHALL be ≤ 15 (currently 16)

### Requirement: No public API changes in kernel crate
All extracted helper functions SHALL be private. No existing public types, traits, or function signatures in the `kernel` crate SHALL change.

#### Scenario: Public API unchanged
- **WHEN** downstream crates compile against the refactored kernel crate
- **THEN** compilation SHALL succeed without modification
