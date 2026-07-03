## ADDED Requirements

### Requirement: Clean-Room ISO-TP Transport Crate

A new `automotive/` crate SHALL be added at Layer 1 of the workspace, depending on the `peripheral` crate — and no other crate — for the CAN HAL. Its remaining workspace production dependencies SHALL be limited to the same-or-lower-layer crates required by this change's other requirements: the `update` crate (the `update::Transport` trait from `remote-update-v1`) and the `security` crate (SHA-3 for `0x27 Security Access` key derivation). The crate SHALL be `#![no_std]` and SHALL contain a clean-room ISO 15765-2 (ISO-TP) implementation in `automotive/src/isotp.rs`, written in the parser style used elsewhere in the workspace, with no third-party ISO-TP crate dependency. When the runtime is configured with `bus_backend != can`, the ISO-TP management listener SHALL NOT start and the automotive management path SHALL contribute zero boot footprint.

#### Scenario: Crate sits at Layer 1 with minimal workspace dependencies

- **WHEN** the crate-level dependency graph is regenerated (`just depgraph` / `just arch-check`)
- **THEN** `automotive` SHALL appear as a Layer 1 crate
- **AND** its workspace production dependencies SHALL be limited to `peripheral` (CAN HAL only), `update`, and `security`
- **AND** no crate-level dependency cycle SHALL be introduced

#### Scenario: no_std bare-metal build succeeds

- **WHEN** the `automotive` crate is built for a bare-metal target (e.g., `aarch64-unknown-none`)
- **THEN** the build SHALL succeed without the standard library
- **AND** the workspace default build SHALL remain green

#### Scenario: Zero footprint when the CAN backend is not selected

- **WHEN** SmallAIOS runs with `bus_backend != can`
- **THEN** the ISO-TP management listener SHALL NOT be started
- **AND** the automotive management path SHALL contribute zero boot footprint

### Requirement: Single-Frame Transmission

The ISO-TP implementation SHALL encode payloads of at most 7 bytes as a classical-CAN single frame (SF) and SHALL decode received single frames back to the original payload. Payloads larger than 7 bytes SHALL NOT be sent as a single frame.

#### Scenario: Seven-byte payload round-trips as one frame

- **WHEN** a 7-byte payload is transmitted over the ISO-TP layer on classical CAN
- **THEN** exactly one single frame SHALL be emitted
- **AND** the receiving side SHALL reassemble a byte-identical payload

#### Scenario: Eight-byte payload is not a single frame

- **WHEN** an 8-byte payload is transmitted
- **THEN** the implementation SHALL NOT emit a single frame
- **AND** it SHALL use the first-frame + consecutive-frame path instead

### Requirement: Multi-Frame Segmentation And Reassembly

The ISO-TP implementation SHALL segment payloads larger than 7 bytes into a first frame (FF) followed by consecutive frames (CF), supporting payloads up to the 4095-byte classical ISO-TP limit. v1 SHALL cap the maximum payload at 4 KiB; payloads exceeding the cap SHALL be rejected with an error before any frame is emitted.

#### Scenario: 4095-byte payload segments and reassembles byte-identically

- **WHEN** a 4095-byte payload is transmitted
- **THEN** the sender SHALL emit a first frame carrying the total length and the initial payload bytes
- **AND** SHALL emit consecutive frames carrying the remainder
- **AND** the receiver SHALL reassemble a byte-identical 4095-byte payload

#### Scenario: Payload above the v1 cap is rejected

- **WHEN** a payload larger than the v1 4 KiB cap is submitted for transmission
- **THEN** the transmit call SHALL return an error
- **AND** no frames SHALL be emitted on the bus

### Requirement: Flow-Control Negotiation

The ISO-TP implementation SHALL implement the flow-control (FC) frame exchange: after receiving a first frame, the receiver SHALL emit an FC frame carrying its block size (BS) and minimum separation time (STmin), and the sender SHALL honor both values for the remainder of the transfer. A transfer whose expected flow-control frame never arrives SHALL be aborted with an error and no partial payload SHALL be delivered to the caller.

#### Scenario: Sender honors the negotiated block size

- **WHEN** the receiver's FC frame specifies block size `BS = N`
- **THEN** the sender SHALL transmit at most `N` consecutive frames
- **AND** SHALL wait for the next FC frame before continuing

#### Scenario: Sender honors STmin spacing

- **WHEN** the receiver's FC frame specifies a non-zero STmin
- **THEN** the sender SHALL separate successive consecutive frames by at least STmin

#### Scenario: Missing flow control aborts the transfer

- **WHEN** the sender emits a first frame and no FC frame arrives
- **THEN** the transfer SHALL be aborted with an error
- **AND** no partial payload SHALL be surfaced to the receiving application

### Requirement: Pad-To-Eight-Bytes Option

The ISO-TP implementation SHALL provide a configurable pad-to-8-bytes option, as required by some CAN controllers. When enabled, every emitted frame SHALL be padded to an 8-byte data length; when disabled, frames SHALL carry only the bytes the protocol requires.

#### Scenario: Padding enabled pads every frame

- **WHEN** the pad-to-8-bytes option is enabled and a 3-byte payload is sent as a single frame
- **THEN** the emitted CAN frame SHALL carry 8 data bytes
- **AND** the receiver SHALL still recover exactly the 3-byte payload

#### Scenario: Padding disabled emits minimal frames

- **WHEN** the pad-to-8-bytes option is disabled and a 3-byte payload is sent as a single frame
- **THEN** the emitted CAN frame SHALL carry only the protocol-required bytes, not 8

### Requirement: Golden Frame Vector Validation

The change SHALL include golden ISO-TP frame vectors covering single-frame, first-frame/consecutive-frame, and flow-control exchanges. The vectors SHALL be cross-checked against SocketCAN's `isotp-tools` on the developer workstation, and unit tests SHALL replay them against the clean-room implementation.

#### Scenario: Unit tests replay the golden vectors

- **WHEN** the `automotive` crate's test suite runs
- **THEN** each golden vector SHALL be replayed through the encoder and decoder
- **AND** any byte-level divergence from the vector SHALL fail the test

#### Scenario: Vector provenance is recorded

- **WHEN** a reviewer inspects the golden vector fixtures
- **THEN** each vector SHALL record how it was produced with `isotp-tools` so it can be independently re-derived
