## ADDED Requirements

### Requirement: Tegra X1 PCIe root complex initialization
The system SHALL initialize the Tegra X1 PCIe root complex (AFI at `0x01003000`, pads at `0x01003200`, RPx config at `0x01000000`) to enable PCIe device enumeration.

#### Scenario: PCIe controller init
- **WHEN** `tegra_pcie::init()` is called
- **THEN** the PCIe controller SHALL be enabled with clocks ungated, PHY trained, and link established on at least one root port

#### Scenario: RTL8168 discoverable
- **WHEN** PCIe enumeration scans the bus after init
- **THEN** the Realtek RTL8168 NIC SHALL be detected at its assigned bus/device/function with vendor ID `0x10EC`

### Requirement: PCIe configuration space access
The driver SHALL provide functions to read and write PCIe configuration space registers (Type 0/Type 1) for any device on the bus.

#### Scenario: Read config register
- **WHEN** `tegra_pcie::config_read(bus, dev, func, offset)` is called
- **THEN** the function SHALL return the 32-bit value from the device's configuration space at the given offset

#### Scenario: Write config register
- **WHEN** `tegra_pcie::config_write(bus, dev, func, offset, value)` is called
- **THEN** the value SHALL be written to the device's configuration space

### Requirement: BAR mapping
The driver SHALL allocate and assign Base Address Registers (BARs) for discovered PCIe devices, mapping their MMIO regions into the CPU's physical address space.

#### Scenario: NIC BAR assigned
- **WHEN** the RTL8168 is enumerated
- **THEN** its BAR0 SHALL be assigned a physical address in the Tegra PCIe MMIO window and the device SHALL be accessible via MMIO reads/writes to that address

### Requirement: Bus mastering
The driver SHALL enable bus mastering on PCIe devices that require DMA (like the RTL8168).

#### Scenario: Enable bus master
- **WHEN** a PCIe device is initialized for DMA
- **THEN** the Bus Master Enable bit (bit 2 of the PCI Command register) SHALL be set

### Requirement: Clock and reset management
The PCIe controller initialization SHALL enable the required clocks and deassert resets via the Tegra CAR (Clock and Reset controller at `0x60006000`).

#### Scenario: PCIe clocks enabled
- **WHEN** `tegra_pcie::init()` begins
- **THEN** the AFI, PCIe, and CML clocks SHALL be enabled and the PCIe module SHALL be taken out of reset before any PCIe register access
