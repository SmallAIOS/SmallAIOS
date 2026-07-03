## ADDED Requirements

### Requirement: Receive-Only LLDP TLV Parsing

The `net` crate SHALL provide a listen-only IEEE 802.1AB LLDP receiver (`net/src/lldp.rs`) on every interface, parsing the `Chassis ID`, `Port ID`, `System Name`, `System Capabilities`, and `Management Address` TLVs. The receiver SHALL NOT transmit LLDP frames in v1.

#### Scenario: Neighbor LLDPDU parsed and logged

- **WHEN** an LLDPDU arrives from the attached switch carrying Chassis ID, Port ID, System Name, System Capabilities, and Management Address TLVs
- **THEN** all five TLVs SHALL be parsed
- **AND** the parsed values SHALL be logged

#### Scenario: No LLDP transmission

- **WHEN** traffic on any interface is captured over an extended period
- **THEN** no LLDP frame originated by SmallAIOS SHALL appear

#### Scenario: Malformed TLV does not crash the parser

- **WHEN** an LLDPDU contains a truncated or over-length TLV
- **THEN** the parser SHALL reject or skip the malformed TLV without panicking
- **AND** subsequent well-formed LLDPDUs SHALL still be processed

#### Scenario: TLV round-trip test coverage

- **WHEN** the LLDP unit tests run the TLV round-trip suite
- **THEN** encoding a parsed TLV set SHALL reproduce the original bytes

### Requirement: Neighbor Table for Diagnostics

Parsed LLDP neighbors SHALL be recorded in an on-disk neighbor table, keyed by receiving interface, for operator diagnostics ("what switch is this plugged into?"). LLDP data SHALL NOT drive automatic bond-mode selection in v1.

#### Scenario: Neighbor recorded per interface

- **WHEN** LLDPDUs are heard on `eth0` and `eth1` from different switches
- **THEN** the neighbor table SHALL contain one entry per interface with the parsed TLV values of the switch heard on that interface

#### Scenario: No auto-mode selection from LLDP

- **WHEN** LLDP data indicates the attached switch's capabilities
- **THEN** no bond mode or interface configuration SHALL change automatically as a result
