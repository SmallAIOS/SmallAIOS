## ADDED Requirements

### Requirement: Mount-time manifest verification
At squashfs mount time, the kernel SHALL: (1) read the manifest footer (per `fs-squashfs-readonly`), (2) verify the ML-DSA-65 signature in the footer over the manifest hash array using the SmallAIOS update-signing public key, (3) refuse the mount with `Err: squashfs signature invalid` if verification fails. No data block SHALL be served from the mount before the signature passes.

#### Scenario: Valid signature mounts
- **WHEN** the footer signature verifies against the embedded public key
- **THEN** mount SHALL succeed and reads SHALL be served

#### Scenario: Invalid signature rejected
- **WHEN** the footer signature fails ML-DSA-65 verification
- **THEN** mount SHALL fail with `Err: squashfs signature invalid`
- **AND** an audit record `mount_signature_invalid` SHALL be appended

### Requirement: Per-read SHA-3-256 block verification
On every `read_block` from a squashfs mount, the FS layer SHALL compute the SHA-3-256 of the bytes read and SHALL compare against the manifest's recorded hash for that block. If the hashes do not match, the read SHALL return `Err(BlockError::BadCrc)` and SHALL NOT pass any byte of the corrupt block to the caller. The corruption event SHALL be appended to the audit log with the block number, expected hash, and observed hash.

#### Scenario: Correct block passes through
- **WHEN** a read returns bytes whose SHA-3-256 matches the manifest
- **THEN** the bytes SHALL be returned to the caller
- **AND** no audit record SHALL be appended

#### Scenario: Corrupted block fails closed
- **WHEN** a read returns bytes whose SHA-3-256 does NOT match the manifest entry
- **THEN** the call SHALL return `Err(BlockError::BadCrc)`
- **AND** zero bytes of the corrupt block SHALL be visible to the caller
- **AND** an audit record `block_hash_mismatch` SHALL be appended

#### Scenario: SHA-3-256 reused from existing security crate
- **WHEN** the integrity layer hashes a block
- **THEN** it SHALL call into the existing `security` crate's SHA-3-256 implementation
- **AND** SHALL NOT introduce a separate hash implementation

### Requirement: Both-slots-bad halt
If both squashfs slots A and B fail their mount-time signature verification, the kernel SHALL refuse to mount `/models/`, SHALL print a recovery hint on the console, SHALL append a `both_slots_invalid` audit record, and SHALL continue boot far enough that login, audit, and `/data/` remain available so an operator can investigate. Inference SHALL be hard-disabled until at least one valid slot is restored.

#### Scenario: Both slots invalid leaves system reachable
- **WHEN** slot A and slot B both fail signature verification
- **THEN** boot SHALL continue past mount with no `/models/` mounted
- **AND** `auth_login` SHALL work normally
- **AND** the operator SHALL be able to read the audit log and stage a recovery image

#### Scenario: Inference disabled on both-bad
- **WHEN** `/models/` is unmounted due to both-slots-bad
- **THEN** any attempt to call `model_load` SHALL fail with `-EROFS` and a `system not in service` message
- **AND** an audit record SHALL be appended for each attempt

### Requirement: F2FS metadata CRC verification
F2FS native CRC32C on the superblock, checkpoint, NAT, and SIT structures SHALL be checked on every read of these metadata blocks. CRC failure SHALL return `Err(BlockError::BadCrc)` and SHALL trigger fallback to the alternate copy of the metadata where F2FS provides one (e.g., the secondary superblock at fixed offset 1024 sectors).

#### Scenario: Primary superblock corrupt falls through to secondary
- **WHEN** the primary F2FS superblock CRC fails and the secondary CRC passes
- **THEN** the mount SHALL proceed using the secondary
- **AND** SHALL log a one-time warning

#### Scenario: Both superblocks corrupt rejected
- **WHEN** both F2FS superblocks fail CRC
- **THEN** mount SHALL fail with `Err: F2FS superblocks corrupt`
- **AND** the kernel SHALL halt with a recovery hint

### Requirement: Application-layer checksums on /data/
Files under `/data/` whose schema is owned by SmallAIOS SHALL carry application-layer integrity that does NOT rely on the filesystem alone. Specifically: the `/data/auth/shadow` file's PHC strings carry their own integrity (the Argon2id tag is implicit), the `/data/audit/log.jsonl` file is a SHA-3-256 hash chain (per `mgmt-audit-log`), and `/data/mgmt/policy.toml` SHALL include a SHA-3-256 fingerprint header that the loader checks before parsing.

#### Scenario: policy.toml fingerprint valid
- **WHEN** the loader reads `/data/mgmt/policy.toml`
- **THEN** the loader SHALL recompute the SHA-3-256 over the body and compare against the header fingerprint
- **AND** SHALL refuse the file with `Err: policy.toml fingerprint mismatch` if they differ

#### Scenario: Audit chain verifies
- **WHEN** the audit log is read and the chain head is validated
- **THEN** every record's `prev_hash` SHALL match the previous record's `hash`
- **AND** any mismatch SHALL be reported via the existing `mgmt-audit-log` chain-check mechanism
