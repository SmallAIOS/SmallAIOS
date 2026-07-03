## ADDED Requirements

### Requirement: Update Session Keyspace And Lifecycle

The Zenoh update transport SHALL expose four endpoints under the `management-login-v1` admin tree: `smallaios/admin/update/begin` (request carries the image manifest metadata; response carries an opaque session id and the chunk size), `smallaios/admin/update/chunk/<session>/<index>` (payload bytes), `smallaios/admin/update/commit/<session>` (finalizes the session: signature verify, slot write, boot-pointer update), and `smallaios/admin/update/abort/<session>` (drops the staged bytes). The Zenoh handler SHALL live in `container/src/mgmt_update.rs`.

#### Scenario: begin opens a session

- **WHEN** a client sends the image manifest metadata to `smallaios/admin/update/begin`
- **THEN** the response SHALL carry an opaque session id and the chunk size the receiver expects
- **AND** subsequent chunks for that session SHALL be addressed as `smallaios/admin/update/chunk/<session>/<index>`

#### Scenario: commit runs the full verification pipeline

- **WHEN** all chunks of a session have been received and the client sends `smallaios/admin/update/commit/<session>`
- **THEN** the reassembled image SHALL pass through ML-DSA-65 signature verification, then the slot writer, then the boot-pointer update
- **AND** a verification failure SHALL fail the commit with staged bytes dropped and the boot pointer untouched

#### Scenario: abort drops staged bytes

- **WHEN** a client sends `smallaios/admin/update/abort/<session>` mid-upload
- **THEN** all staged bytes for that session SHALL be discarded
- **AND** the boot pointer SHALL be untouched

### Requirement: Per-Chunk CRC-32 Early Failure

Every chunk SHALL carry a CRC-32 so corruption is detected at chunk granularity rather than after the full 8 MB image has been staged. A chunk whose CRC-32 does not match SHALL be rejected at receipt and SHALL NOT be staged.

#### Scenario: Corrupted chunk fails early

- **WHEN** a chunk arrives at `smallaios/admin/update/chunk/<session>/<index>` whose CRC-32 does not match its payload
- **THEN** the receiver SHALL reject that chunk with an error response
- **AND** the chunk SHALL NOT be added to the staged image
- **AND** the failure SHALL be reported before the transfer completes, not at commit time

### Requirement: PQC Transport Reuse, Chunked Sizing, And Progress Metrics

The Zenoh update path SHALL re-use the existing PQC-backed Zenoh transport — no new transport plumbing. Uploads SHALL be chunked (not monolithic) so the ~8 MB image fits over links whose usable MTU may be ~1 KB, and upload progress SHALL be observable in `smallaios/metrics/update`.

#### Scenario: Existing PQC transport carries the update

- **WHEN** an update session runs over Zenoh
- **THEN** it SHALL use the same PQC-backed Zenoh transport as the existing admin keyspace
- **AND** no update-specific transport stack SHALL be introduced

#### Scenario: Progress observable during upload

- **WHEN** a chunked upload is in flight
- **THEN** `smallaios/metrics/update` SHALL reflect the session's progress
- **AND** an operator subscribed to that key SHALL observe progress advance as chunks land
