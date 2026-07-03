## ADDED Requirements

### Requirement: Optional SecOC-Equivalent MAC Layer

The `automotive` crate SHALL provide an optional SecOC-equivalent authentication layer that protects every UDS request and response payload using the existing AES-256-GCM primitive truncated to a 32-bit MAC, plus a 16-bit freshness counter, in framing compatible with AUTOSAR SecOC. The layer SHALL add exactly 6 bytes to every protected payload. The layer SHALL be active only when a SecOC key file is configured; wire-level compatibility with a specific vendor's AUTOSAR SecOC profile and hardware-backed key storage are explicit non-goals for v1.

#### Scenario: Protected payloads carry exactly six extra bytes

- **WHEN** the SecOC-equivalent layer is enabled via a configured key file and a UDS request is transmitted
- **THEN** the payload SHALL grow by exactly 6 bytes (4-byte truncated MAC plus 2-byte freshness counter)
- **AND** the responder SHALL apply the same 6-byte protection to its response

#### Scenario: Tampered MAC is rejected

- **WHEN** a received payload's truncated MAC does not verify under AES-256-GCM with the configured key
- **THEN** the payload SHALL be discarded without dispatching any UDS handler

#### Scenario: Layer is inert without a configured key

- **WHEN** no SecOC key file path is present in the configuration
- **THEN** the SecOC-equivalent layer SHALL NOT be applied to any payload
- **AND** the bus-isolation policy SHALL govern whether the listener may bind at all

### Requirement: Freshness Counter Replay Protection

The 16-bit freshness counter SHALL increase with each protected message, and the receiver SHALL reject payloads whose counter is stale or repeated, preventing replay of previously captured frames.

#### Scenario: Replayed frame is rejected

- **WHEN** an attacker replays a previously captured, correctly-MAC'd payload
- **THEN** the receiver SHALL reject it because its freshness counter is not newer than the last accepted counter
- **AND** no UDS handler SHALL be invoked for the replayed payload

#### Scenario: Fresh counter is accepted

- **WHEN** a payload arrives with a valid MAC and a freshness counter newer than the last accepted value
- **THEN** the payload SHALL be accepted and dispatched
