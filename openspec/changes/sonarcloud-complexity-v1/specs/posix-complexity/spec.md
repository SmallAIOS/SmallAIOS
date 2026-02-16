## ADDED Requirements

### Requirement: VfsTree lookup cognitive complexity ≤ 15
The `VfsTree::lookup()` function in `posix/src/vfs.rs` SHALL have cognitive complexity ≤ 15. Path traversal logic SHALL be simplified by extracting component resolution into a helper. All existing VFS tests SHALL continue to pass.

#### Scenario: VfsTree lookup refactored below threshold
- **WHEN** SonarCloud analyzes `VfsTree::lookup()` after refactoring
- **THEN** the cognitive complexity score SHALL be ≤ 15 (currently 23)

#### Scenario: Path resolution behavior preserved
- **WHEN** existing VFS lookup tests (absolute paths, relative paths, symlinks, not-found) execute
- **THEN** all tests SHALL pass with identical resolution results

### Requirement: No public API changes in posix crate
All extracted helper functions SHALL be private. No existing public types, traits, or function signatures in the `posix` crate SHALL change.

#### Scenario: Public API unchanged
- **WHEN** downstream crates compile against the refactored posix crate
- **THEN** compilation SHALL succeed without modification
