# fs-delta-update Specification

## Purpose
TBD - created by archiving change embedded-filesystem-v1. Update Purpose after archive.
## Requirements
### Requirement: bsdiff delta applier
The `fs` crate SHALL provide a clean-room `#![no_std]` Rust applier for the bsdiff binary-diff format. The applier SHALL accept a reference blob (the active squashfs) and a bsdiff patch payload, SHALL produce the new blob via the bsdiff streaming algorithm, and SHALL write the result to the inactive squashfs partition. No new external Rust crate dependencies SHALL be added in the production dep graph.

#### Scenario: Apply produces expected blob
- **WHEN** a known-good `(reference, patch, expected_output)` triple is processed
- **THEN** the resulting bytes written to the inactive partition SHALL `cmp`-equal `expected_output` byte-for-byte

#### Scenario: Truncated patch rejected
- **WHEN** a bsdiff patch is truncated mid-stream
- **THEN** the applier SHALL detect the truncation via length checks
- **AND** SHALL return `Err: bsdiff patch truncated`
- **AND** SHALL leave the inactive partition in an `unbootable` state (record marked `valid=0`)

#### Scenario: Out-of-bounds offset rejected
- **WHEN** a bsdiff control command references an out-of-range offset in the reference blob
- **THEN** the applier SHALL return `Err: bsdiff offset out of range`
- **AND** SHALL NOT proceed with downstream commands

### Requirement: Pre-apply integrity checks
Before any byte is written to the inactive partition, the delta-update flow SHALL verify:
1. The ML-DSA-65 signature on the delta payload (signed by the SmallAIOS update-signing key) is valid.
2. The reference-blob hash declared in the delta payload matches the SHA-3-256 of the active squashfs partition currently on disk.

If either check fails, no inactive-partition writes SHALL occur and the flow SHALL return `Err: delta verification failed` with details of which check failed.

#### Scenario: Forged delta rejected
- **WHEN** a delta payload arrives with an invalid ML-DSA-65 signature
- **THEN** verification SHALL fail before any disk write
- **AND** SHALL append an audit record `delta_signature_invalid`

#### Scenario: Wrong-version reference rejected
- **WHEN** a delta payload's `reference_hash` does not match the active squashfs's actual hash
- **THEN** verification SHALL fail with `Err: delta reference mismatch (expected <a>, found <b>)`
- **AND** SHALL append an audit record naming both hashes

### Requirement: Post-apply integrity checks
After the bsdiff applier finishes writing to the inactive partition, the flow SHALL:
1. Verify the SHA-3-256 manifest in the new partition's footer against every block in the partition.
2. Verify the ML-DSA-65 signature in the footer over the manifest's hash array.

If either check fails, the inactive-slot record SHALL be marked `valid=0` and the flow SHALL return `Err: post-apply verification failed`. The active slot SHALL remain in service.

#### Scenario: Post-apply manifest mismatch fails closed
- **WHEN** the post-apply per-block verify finds a single mismatched block
- **THEN** the flow SHALL return `Err: manifest hash mismatch at block <n>`
- **AND** the inactive-slot boot config record SHALL remain `valid=0`
- **AND** the active slot SHALL continue serving inference normally

#### Scenario: Post-apply signature verify
- **WHEN** the manifest array hashes correctly but the ML-DSA-65 signature in the footer does not verify
- **THEN** the flow SHALL return `Err: footer signature invalid`
- **AND** the inactive-slot record SHALL remain `valid=0`

### Requirement: Hand-off to remote-update-v1
On successful pre + apply + post checks, the delta-update flow SHALL: (1) write the new boot config record (per `fs-ab-boot`) to the previously-inactive slot's record area with `active_slot` flipped, `tentative=1`, `boot_success=0`, and `generation = current_max + 1`; (2) emit an `update_staged` event for `remote-update-v1` to consume; (3) return `Ok(new_active_slot)`. The actual reboot SHALL be initiated by `remote-update-v1`, not the FS layer.

#### Scenario: Successful staging hands off
- **WHEN** all pre/apply/post checks pass
- **THEN** the flow SHALL update the boot config to mark the new slot active+tentative
- **AND** SHALL emit `update_staged{ new_slot, generation }`
- **AND** SHALL NOT initiate the reboot itself

#### Scenario: Audit record on staging
- **WHEN** staging completes
- **THEN** an audit record `update_staged` SHALL be appended with the new slot, generation, delta size, and the manifest hash

