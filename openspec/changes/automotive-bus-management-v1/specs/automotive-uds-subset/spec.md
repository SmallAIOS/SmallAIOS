## ADDED Requirements

### Requirement: Gating Research And Design Deliverables

The change SHALL produce `docs/automotive-bus-research.md` and `docs/automotive-bus-design.md` before any `automotive/` crate code is written. The research document SHALL synthesize ISO 15765-2 framing, the ISO 14229 service catalog split into implemented and explicitly-not-implemented subsets, the security options tradeoff, and a survey of existing Rust CAN crates. The design document SHALL give the concrete Rust trait and module layout, map every `management-login-v1` verb to a UDS service ID, and record the SecOC/CAN-XL/isolated-bus decision.

#### Scenario: Research document covers the required topics

- **WHEN** a reviewer reads `docs/automotive-bus-research.md`
- **THEN** it SHALL cover ISO 15765-2 framing (single-frame, first-frame, consecutive-frame, flow-control, padding rules, classical-CAN vs CAN-FD differences)
- **AND** it SHALL list the ISO 14229 UDS services that will be implemented and the services that will explicitly not be implemented, with rationale
- **AND** it SHALL present a tradeoff table for AUTOSAR SecOC, CAN XL with TLS, and physically-isolated diagnostic bus, with a recommended choice
- **AND** it SHALL survey `socketcan`, `embedded-can`, `isotp-rs`, and `automotive-rs`, noting which are `no_std` and which are replaced by clean-room code

#### Scenario: Design document maps verbs and enumerates flow-control states

- **WHEN** a reviewer reads `docs/automotive-bus-design.md`
- **THEN** it SHALL present the concrete Rust trait / module layout for the `automotive/` crate
- **AND** it SHALL map every `management-login-v1` verb to a UDS service ID
- **AND** it SHALL record the SecOC/CAN-XL/isolated-bus decision
- **AND** it SHALL enumerate every ISO-TP flow-control state transition so the test plan can replay vectors from the standard

#### Scenario: Documents gate implementation

- **WHEN** the change's tasks are sequenced
- **THEN** both documents SHALL be completed before any `automotive/` implementation task begins

### Requirement: UDS Service Subset Boundary

`automotive/src/uds.rs` SHALL implement exactly the following ISO 14229 services: `0x11 ECU Reset`, `0x22 Read Data By Identifier`, `0x27 Security Access` (level 1 only), `0x34 Request Download`, `0x36 Transfer Data`, `0x37 Request Transfer Exit`, and `0x3E Tester Present`. Any other service ID SHALL receive a negative response. CANopen, J1939, DoIP, programming sessions beyond download, security level 2+, dynamic DID definition, routine control, and ReadDTC SHALL NOT be implemented in v1.

#### Scenario: Supported service is dispatched

- **WHEN** a request with service ID `0x22` arrives over ISO-TP
- **THEN** the Read Data By Identifier handler SHALL be invoked

#### Scenario: Unsupported service receives a negative response

- **WHEN** a request with a service ID outside the v1 subset (e.g., `0x19` ReadDTCInformation) arrives
- **THEN** the handler SHALL return a UDS negative response
- **AND** no partial handling of the service SHALL occur

#### Scenario: No second protocol stack is linked

- **WHEN** a reviewer inspects the `automotive` crate
- **THEN** it SHALL contain no CANopen (NMT/SDO/PDO/EDS) or J1939 implementation

### Requirement: ECU Reset Service

The `0x11 ECU Reset` handler SHALL wrap `system_power(REBOOT)` from `system-power-control-v1`, so a UDS-triggered reset takes the same reboot path as the existing management surfaces.

#### Scenario: ECU Reset request reboots via system_power

- **WHEN** a valid `0x11 ECU Reset` request is received
- **THEN** the handler SHALL invoke `system_power(REBOOT)`
- **AND** the unit SHALL reboot through the same `system-power-control-v1` path used by the other management surfaces

### Requirement: Read Data By Identifier Service

The `0x22 Read Data By Identifier` handler SHALL serve the same telemetry values that Zenoh exposes on `smallaios/metrics/**`. DID assignments SHALL come from an operator-defined DID table in the configuration file; no automotive-OEM DID assignment SHALL be assumed.

#### Scenario: Configured DID returns the matching telemetry value

- **WHEN** the DID table maps a DID to a metric and a `0x22` request for that DID arrives
- **THEN** the response SHALL carry the same value that the corresponding `smallaios/metrics/**` key would report

#### Scenario: Unconfigured DID receives a negative response

- **WHEN** a `0x22` request arrives for a DID absent from the operator's DID table
- **THEN** the handler SHALL return a UDS negative response
- **AND** no telemetry value SHALL be leaked

### Requirement: Security Access Seed/Key Bridge

The `0x27 Security Access` handler (level 1) SHALL implement a seed/key challenge whose key derivation function uses SHA-3 over the challenge seed combined with a per-unit pre-shared secret stored alongside the shadow file. A successful key exchange SHALL bridge to the `auth_login` syscall under `management-login-v1`, so UDS sessions carry the same authenticated identity as the other management surfaces.

#### Scenario: Seed request returns a challenge

- **WHEN** a `0x27` request-seed message for level 1 arrives
- **THEN** the handler SHALL return a challenge seed

#### Scenario: Correct key grants security access via auth_login

- **WHEN** the tester returns the key derived with SHA-3 from the seed and the per-unit pre-shared secret
- **THEN** security access level 1 SHALL be granted
- **AND** the handler SHALL authenticate the session through the `auth_login` syscall

#### Scenario: Incorrect key is rejected

- **WHEN** the tester returns a key that does not match the SHA-3 derivation
- **THEN** the handler SHALL return a UDS negative response
- **AND** no `auth_login` session SHALL be established

### Requirement: Download Service Flow

The `0x34 Request Download` / `0x36 Transfer Data` / `0x37 Request Transfer Exit` handlers SHALL implement the firmware transfer flow that feeds the `update::Transport` trait from `remote-update-v1`, so the same A/B-slot machinery handles a UDS-driven update.

#### Scenario: Full download sequence drives an A/B update over loopback CAN

- **WHEN** a tester issues `0x34`, a sequence of `0x36` transfers, and `0x37` over loopback CAN
- **THEN** the transferred image SHALL flow through the `update::Transport` trait into the existing A/B-slot machinery
- **AND** the end-to-end "drive an A/B update over loopback CAN" test SHALL pass

#### Scenario: Transfer exit hands off to the existing update machinery

- **WHEN** `0x37 Request Transfer Exit` completes a transfer
- **THEN** the subsequent A/B-slot handling SHALL be identical to an update delivered over any other `remote-update-v1` transport

### Requirement: Tester Present Session Keep-Alive

The `0x3E Tester Present` handler SHALL keep the diagnostic session alive.

#### Scenario: Periodic Tester Present keeps the session active

- **WHEN** `0x3E Tester Present` requests arrive within the session timeout
- **THEN** the diagnostic session, including any granted security access, SHALL remain active

#### Scenario: Session expires without Tester Present

- **WHEN** no request arrives within the session timeout
- **THEN** the diagnostic session SHALL expire
- **AND** a new `0x27 Security Access` exchange SHALL be required before secured operations

### Requirement: Automotive Configuration Schema

The `automotive` TOML configuration (`automotive/uds.toml` under the `mgmt` Config model's `/data/` layout) SHALL define: the CAN interface using the same selectors the inference bridge already uses (`loopback`, `mcp2515:<path>`, `axi:<addr>`); the diagnostic request and response CAN IDs; the isolated-bus assertion (`isolated = true|false`); an optional SecOC key file path; and the DID table for `0x22 Read Data By Identifier`.

#### Scenario: Complete configuration parses

- **WHEN** an `automotive/uds.toml` supplying an interface selector, request/response CAN IDs, `isolated`, an optional SecOC key path, and a DID table is loaded
- **THEN** parsing SHALL succeed and each field SHALL be available to the UDS listener

#### Scenario: Interface selectors match the inference bridge grammar

- **WHEN** the operator specifies `mcp2515:/dev/spidev0.0` or `axi:0xa0010000` as the CAN interface
- **THEN** the selector SHALL be interpreted with the same grammar the CAN inference bridge uses for `SMALLAIOS_CAN_DEVICE`
