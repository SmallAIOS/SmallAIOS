## ADDED Requirements

### Requirement: Self-Describing `smallaios-img v1` Manifest

Update images SHALL use the self-describing `smallaios-img v1` format: a human-readable manifest followed by an opaque payload. The manifest SHALL open with the literal line `smallaios-img v1`, be delimited by `---` lines, and carry the fields `arch` (a Rust target triple such as `aarch64-unknown-none` or `x86_64-unknown-none`), `version`, `build_date` (RFC 3339), `payload_len` (bytes), `payload_hash` (32-byte SHA3-256 of the payload), and `signature` (ML-DSA-65 over manifest+payload, base64). The payload bytes follow the closing `---` delimiter. The `update` crate SHALL provide a manifest parser for this format.

#### Scenario: Well-formed manifest round-trips through the parser

- **WHEN** the manifest parser is fed a valid `smallaios-img v1` image whose manifest declares `arch: aarch64-unknown-none`, a `payload_len`, a `payload_hash`, and a `signature`
- **THEN** parsing SHALL succeed
- **AND** the parsed structure SHALL expose `arch`, `version`, `build_date`, `payload_len`, `payload_hash`, and `signature` matching the manifest text
- **AND** the payload SHALL be treated as opaque bytes (no interpretation beyond length and hash)

#### Scenario: Manifest is debuggable as plain text

- **WHEN** an operator inspects the first bytes of a `smallaios-img v1` file on a flaky link
- **THEN** the version line, field names, and field values SHALL be readable as plain text without any tooling
- **AND** only the payload after the closing `---` SHALL be non-human-readable

#### Scenario: Malformed manifest rejected

- **WHEN** the parser is fed bytes that lack the `smallaios-img v1` header line, are missing a required field, or are truncated before `payload_len` payload bytes are present
- **THEN** parsing SHALL fail with an error identifying the failure mode
- **AND** no slot write SHALL be attempted
- **AND** the boot pointer SHALL be untouched

### Requirement: Payload Hash And Length Validation

Before signature verification succeeds an image, the received payload SHALL be validated against the manifest: the payload byte count SHALL equal `payload_len` and the SHA3-256 digest of the payload SHALL equal `payload_hash`. A mismatch in either SHALL abort the update without touching the boot pointer.

#### Scenario: Tampered payload fails the hash check

- **WHEN** an image arrives whose payload bytes differ from those the manifest's `payload_hash` was computed over
- **THEN** the SHA3-256 digest comparison SHALL fail
- **AND** the update SHALL abort without writing the slot or the boot pointer

#### Scenario: Length mismatch aborts

- **WHEN** the received payload byte count differs from the manifest's `payload_len`
- **THEN** the update SHALL abort cleanly
- **AND** the boot pointer SHALL be untouched

### Requirement: ML-DSA-65 Signature Over Manifest Plus Payload

Every update image SHALL carry an ML-DSA-65 signature computed over the manifest and the payload together. The signature verifier SHALL run after transfer completes and before any slot write; an image whose signature does not verify SHALL never be written as bootable. The signature SHALL use the same ML-DSA-65 key and algorithm as the existing `verified-boot` boot-time chain.

#### Scenario: Correctly signed image passes verification

- **WHEN** an image signed with the `verified-boot` ML-DSA-65 key is received over any transport
- **THEN** signature verification over manifest+payload SHALL succeed
- **AND** the image SHALL proceed to the slot writer

#### Scenario: Unsigned or badly signed image refused

- **WHEN** an image arrives with a missing, corrupted, or wrong-key `signature` field
- **THEN** verification SHALL fail
- **AND** the inactive slot SHALL NOT be marked bootable
- **AND** the boot pointer SHALL be untouched

### Requirement: Architecture Compatibility Check

The update pipeline SHALL compare the manifest `arch` field against the running platform's target triple and SHALL abort the update on mismatch, before any slot write.

#### Scenario: Wrong-arch image rejected

- **WHEN** an image whose manifest declares `arch: x86_64-unknown-none` is submitted to a device running `aarch64-unknown-none`
- **THEN** the update SHALL abort with a wrong-arch failure
- **AND** no slot bytes SHALL be written
- **AND** the boot pointer SHALL be untouched

#### Scenario: Matching arch accepted

- **WHEN** an image whose manifest declares `arch: aarch64-unknown-none` is submitted to a device running `aarch64-unknown-none`
- **THEN** the arch check SHALL pass and the pipeline SHALL continue to hash and signature validation
