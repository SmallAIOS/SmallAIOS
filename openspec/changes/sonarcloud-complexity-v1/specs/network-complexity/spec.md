## ADDED Requirements

### Requirement: TcpConnection on_segment cognitive complexity ≤ 15
The `TcpConnection::on_segment()` function in `net/src/tcp.rs` SHALL have cognitive complexity ≤ 15. Each TCP state's segment handling SHALL be extracted into a per-state method (e.g., `handle_listen()`, `handle_syn_sent()`, `handle_established()`). All existing TCP tests SHALL continue to pass.

#### Scenario: on_segment refactored below threshold
- **WHEN** SonarCloud analyzes `TcpConnection::on_segment()` after refactoring
- **THEN** the cognitive complexity score SHALL be ≤ 15 (currently 33)

#### Scenario: TCP state machine behavior preserved
- **WHEN** existing TCP connection tests (3-way handshake, data transfer, close sequences) execute
- **THEN** all tests SHALL pass with identical state transitions

### Requirement: LongHeader decode cognitive complexity ≤ 15
The `LongHeader::decode()` function in `net/src/quic/packet.rs` SHALL have cognitive complexity ≤ 15. Per-packet-type decoding (Initial, Handshake, 0-RTT, Retry) SHALL be extracted into dedicated helper functions. All existing QUIC packet tests SHALL continue to pass.

#### Scenario: LongHeader decode refactored below threshold
- **WHEN** SonarCloud analyzes `LongHeader::decode()` after refactoring
- **THEN** the cognitive complexity score SHALL be ≤ 15 (currently 29)

### Requirement: parse_ndp_options cognitive complexity ≤ 15
The `parse_ndp_options()` function in `net/src/ndp.rs` SHALL have cognitive complexity ≤ 15. Option type parsing SHALL be extracted into per-type helpers. All existing NDP tests SHALL continue to pass.

#### Scenario: parse_ndp_options refactored below threshold
- **WHEN** SonarCloud analyzes `parse_ndp_options()` after refactoring
- **THEN** the cognitive complexity score SHALL be ≤ 15 (currently 19)

### Requirement: No public API changes in net crate
All extracted helper functions SHALL be private. No existing public types, traits, or function signatures in the `net` crate SHALL change.

#### Scenario: Public API unchanged
- **WHEN** downstream crates compile against the refactored net crate
- **THEN** compilation SHALL succeed without modification
