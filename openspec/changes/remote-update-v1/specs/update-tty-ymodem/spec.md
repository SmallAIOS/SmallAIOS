## ADDED Requirements

### Requirement: YMODEM-1K Receive Mode On The Console TTY

When the `update` command is issued from a root-authenticated serial-console session, the kernel SHALL enter YMODEM-1K receive mode on the same TTY: 1024-byte data blocks, CRC-16 error detection, and the `<NAK>`/`<C>`/`<ACK>`/`<EOT>` state machine, using the `<C>`-mode (CRC) handshake to initiate. The receiver SHALL interoperate with standard terminal senders (`minicom` `Ctrl-A S → ymodem`, `picocom -t '!! sx -k'`, `tio --send`) with no custom host-side tooling.

#### Scenario: Receiver initiates with the C handshake

- **WHEN** the kernel enters YMODEM receive mode
- **THEN** it SHALL transmit `<C>` to request CRC-16 mode
- **AND** SHALL accept 1024-byte (STX-framed) data blocks from the sender

#### Scenario: Standard sx sender completes a transfer

- **WHEN** an operator sends a valid image with `sx -k` (YMODEM-1K) from `picocom`
- **THEN** every correctly received block SHALL be `<ACK>`ed
- **AND** on `<EOT>` the receiver SHALL complete the session and exit receive mode

### Requirement: Clean-Room `no_std` YMODEM Implementation

The YMODEM-1K receiver SHALL be clean-room `#![no_std]` Rust living in the `update` crate (`update/src/ymodem.rs`). It SHALL NOT depend on any third-party `xmodem`/`ymodem` crate.

#### Scenario: No third-party YMODEM crates in the dependency tree

- **WHEN** the production dependency tree of the `update` crate is inspected
- **THEN** no external `xmodem` or `ymodem` crate SHALL appear
- **AND** the receiver SHALL compile for `no_std` bare-metal targets (`aarch64-unknown-none`, `x86_64-unknown-none`)

### Requirement: Block-Level Error Recovery

The receiver SHALL validate each block's number byte, its complement, and its CRC-16. A corrupted block SHALL be answered with `<NAK>` so the sender retransmits — the protocol SHALL make progress on a physically noisy serial line at the cost of speed, never at the cost of accepting corrupt data.

#### Scenario: CRC mismatch triggers retransmission

- **WHEN** a data block arrives whose CRC-16 does not match its contents
- **THEN** the receiver SHALL respond `<NAK>`
- **AND** SHALL accept the sender's retransmission of the same block
- **AND** the assembled image SHALL contain the corrected block exactly once

#### Scenario: Inconsistent block numbering rejected

- **WHEN** a block arrives whose block-number byte and its ones-complement byte disagree
- **THEN** the receiver SHALL NOT accept the block's payload
- **AND** SHALL `<NAK>` so the sender retransmits

### Requirement: Post-Transfer Handoff And Clean Abort

After `<EOT>`, the receiver SHALL hand the assembled bytes to the common pipeline: manifest parser → ML-DSA-65 signature verifier → slot writer. Any failure — unrecoverable CRC errors, manifest parse failure, signature failure, wrong arch, insufficient slot space — SHALL abort cleanly: staged bytes discarded, boot pointer untouched, and the TTY returned to the console session.

#### Scenario: Successful transfer flows into the verification pipeline

- **WHEN** a YMODEM transfer completes with `<EOT>`
- **THEN** the received bytes SHALL be parsed as a `smallaios-img v1` manifest
- **AND** on successful verification SHALL be written to the inactive slot via the slot writer

#### Scenario: Signature failure after a clean transfer aborts without side effects

- **WHEN** a transfer completes but the image's ML-DSA-65 signature does not verify
- **THEN** the staged bytes SHALL be discarded
- **AND** the boot pointer SHALL be untouched
- **AND** the operator SHALL be returned to the console prompt with an error message
