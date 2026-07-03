## ADDED Requirements

### Requirement: IEEE 802.1AX LACP State Machines

The `net` crate SHALL provide a clean-room `802.3ad` LACP bond mode (Linux mode 4) implementing the full IEEE 802.1AX LACPDU state machines: Mux, Selection, Receive, Periodic, and Churn detection.

#### Scenario: Aggregation forms with a cooperating partner

- **WHEN** both slaves of an `802.3ad` bond exchange LACPDUs with a managed switch running LACP
- **THEN** each slave SHALL progress through the Receive and Mux machines to the collecting/distributing state
- **AND** the bond SHALL carry traffic across both aggregated slaves

#### Scenario: Silent partner expires out of the aggregate

- **WHEN** the partner stops sending LACPDUs on one slave and the Receive machine's timeout expires
- **THEN** that slave SHALL leave the aggregate
- **AND** the bond SHALL continue on the remaining aggregated slaves

#### Scenario: Churn detection flags instability

- **WHEN** a slave oscillates in and out of sync with its partner
- **THEN** the Churn detection machine SHALL detect the churn
- **AND** the churn event SHALL be logged

#### Scenario: State machine fuzzed against the IEEE reference

- **WHEN** the LACP state machine is fuzzed against the IEEE 802.1AX reference state diagram
- **THEN** no state transition SHALL diverge from the reference diagram

### Requirement: LACPDU Transmission and Slow-Protocols Handling

LACP in v1 SHALL be fully bidirectional: the bond SHALL transmit LACPDUs, addressed to the slow-protocols MAC `01:80:c2:00:00:02`, and SHALL honor short/long timeout negotiation, adjusting its periodic transmission rate to the partner's requested timeout.

#### Scenario: LACPDUs use the slow-protocols MAC

- **WHEN** any LACPDU transmitted by the bond is captured
- **THEN** its destination MAC SHALL be `01:80:c2:00:00:02`

#### Scenario: Partner-requested short timeout speeds up transmission

- **WHEN** the partner's LACPDUs request the short timeout
- **THEN** the Periodic machine SHALL transmit at the fast periodic rate
- **AND** when the partner requests the long timeout, the Periodic machine SHALL fall back to the slow periodic rate

### Requirement: Partner Key Matching in Selection

The Selection logic SHALL aggregate a slave only when its partner system and operational key match those of the existing aggregate. A slave with a mismatched partner key SHALL be excluded from the aggregate, and the exclusion SHALL be logged rather than dropping the slave silently.

#### Scenario: Mismatched partner key excluded and logged

- **WHEN** slave `eth0` sees partner key 17 and slave `eth1` sees partner key 42 (e.g., cabled to different switches)
- **THEN** only one consistent set of slaves SHALL be aggregated
- **AND** the excluded slave SHALL be reported in the log with its observed partner system and key
