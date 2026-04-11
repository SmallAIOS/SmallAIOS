## ADDED Requirements

### Requirement: RTL8169/8168 NIC driver
The system SHALL include a driver for the Realtek RTL8169/8168/8111 family of Gigabit Ethernet controllers, supporting TX and RX of Ethernet frames.

#### Scenario: NIC detected on PCIe bus
- **WHEN** PCIe enumeration discovers a device with vendor ID `0x10EC` and device ID in the RTL8169 family (`0x8168`, `0x8169`, `0x8136`)
- **THEN** the driver SHALL claim the device and begin initialization

#### Scenario: Link up
- **WHEN** the NIC is initialized and an Ethernet cable is connected
- **THEN** the driver SHALL detect link-up status via the PHY status register and report the negotiated speed (10/100/1000 Mbps)

### Requirement: Frame transmission
The NIC driver SHALL transmit Ethernet frames via a DMA descriptor ring.

#### Scenario: Send an Ethernet frame
- **WHEN** the network stack calls `nic.send(frame_data)`
- **THEN** the driver SHALL write the frame to a TX descriptor, set the OWN bit, and signal the NIC to transmit by writing to the TxPoll register

#### Scenario: TX completion
- **WHEN** the NIC completes frame transmission
- **THEN** the driver SHALL reclaim the TX descriptor for reuse

### Requirement: Frame reception
The NIC driver SHALL receive Ethernet frames via a DMA descriptor ring and pass them to the network stack.

#### Scenario: Receive an Ethernet frame
- **WHEN** the NIC receives a frame and writes it to an RX descriptor buffer
- **THEN** the driver SHALL read the frame data from the RX buffer and pass it to the network stack's Ethernet handler

#### Scenario: RX buffer replenishment
- **WHEN** an RX descriptor is consumed
- **THEN** the driver SHALL replenish the descriptor with a fresh buffer and return it to the NIC

### Requirement: MAC address
The driver SHALL read the MAC address from the NIC's hardware registers and make it available to the network stack.

#### Scenario: MAC address read
- **WHEN** the NIC is initialized
- **THEN** the driver SHALL read the 6-byte MAC address from IDR0-IDR5 (registers at offset `0x00-0x05`) and configure the network stack's Ethernet layer with it

### Requirement: DMA buffer management
TX and RX DMA descriptor rings and their associated data buffers SHALL be allocated from physically contiguous memory.

#### Scenario: Descriptor ring allocation
- **WHEN** the driver initializes
- **THEN** it SHALL allocate TX and RX descriptor rings (minimum 64 entries each) from physically contiguous pages and program their physical addresses into TNPDS (TX) and RDSAR (RX) registers

### Requirement: Integration with net crate
The RTL8169 driver SHALL implement a `NetworkDevice` trait that integrates with the existing `net` crate Ethernet/ARP/IPv4/TCP stack.

#### Scenario: End-to-end packet flow
- **WHEN** the net stack sends an IPv4 packet
- **THEN** the packet SHALL flow through: IPv4 → Ethernet framing (with ARP for MAC resolution) → RTL8169 TX DMA → wire
