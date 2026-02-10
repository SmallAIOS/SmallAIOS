# Delta for SoC FPGA Platform

## ADDED Requirements

### Requirement: AXI4-Lite Register Access Driver
The kernel SHALL implement an AXI4-Lite memory-mapped register read/write driver for accessing FPGA soft-IP control and status registers.

#### Scenario: Read a 32-bit register
- WHEN a driver reads a register from an AXI4-Lite peripheral at a given MMIO offset
- THEN the driver MUST perform a single 32-bit volatile read at the base address plus offset
- AND the read MUST use the appropriate memory ordering barriers for MMIO access

#### Scenario: Write a 32-bit register
- WHEN a driver writes a value to an AXI4-Lite peripheral register
- THEN the driver MUST perform a single 32-bit volatile write at the base address plus offset
- AND the write MUST be followed by a memory barrier to ensure the write reaches the peripheral

#### Scenario: Register access with invalid offset
- WHEN a register access is attempted at an offset beyond the peripheral's address range
- THEN the driver MUST return an error rather than performing an out-of-bounds MMIO access
- AND the error MUST include the attempted offset and the peripheral's valid range

### Requirement: AXI4 Full Burst Transfer Driver
The kernel SHALL implement an AXI4 full memory-mapped burst transfer driver for high-throughput data movement between CPU memory and FPGA peripherals.

#### Scenario: Initiate a burst read
- WHEN the kernel requests a burst read from an AXI4 peripheral
- THEN the driver MUST configure the burst length (1-256 beats), burst size (matching bus width), and burst type (INCR)
- AND the driver MUST wait for all beats to complete before signaling transfer completion

#### Scenario: Initiate a burst write
- WHEN the kernel requests a burst write to an AXI4 peripheral
- THEN the driver MUST send the address and all data beats with correct write strobes
- AND the driver MUST wait for the write response before signaling completion

#### Scenario: Burst transfer error handling
- WHEN an AXI4 burst transfer receives a SLVERR or DECERR response
- THEN the driver MUST abort the remaining beats if any
- AND the driver MUST return an error indicating the AXI response code and the beat at which the error occurred

### Requirement: AXI DMA Controller Driver
The kernel SHALL implement an AXI DMA controller driver supporting both simple DMA mode and scatter-gather mode for efficient data transfers between system memory and FPGA peripherals.

#### Scenario: Simple DMA transfer
- WHEN a subsystem requests a DMA transfer with a contiguous source and destination buffer
- THEN the DMA driver MUST program the source address, destination address, and transfer length registers
- AND the driver MUST start the transfer and report completion via interrupt or polling
- AND the source and destination buffers MUST be cache-coherent or explicitly flushed/invalidated

#### Scenario: Scatter-gather DMA transfer
- WHEN a subsystem requests a DMA transfer with non-contiguous memory regions
- THEN the DMA driver MUST build a scatter-gather descriptor chain in DMA-accessible memory
- AND each descriptor MUST specify source address, destination address, length, and next-descriptor pointer
- AND the driver MUST start the chain and report completion after the last descriptor completes

#### Scenario: DMA transfer error
- WHEN a DMA transfer encounters a bus error or timeout
- THEN the driver MUST halt the DMA channel
- AND the driver MUST report the error with the faulting address and transfer state
- AND any partially completed scatter-gather descriptors MUST be identifiable for recovery

### Requirement: DTB-Based Peripheral Discovery
The kernel SHALL discover FPGA soft-IP peripherals at boot time by parsing the flattened device tree blob (DTB) passed by the bootloader or firmware.

#### Scenario: Enumerate FPGA peripherals from DTB
- WHEN the kernel parses the DTB during early boot
- THEN it MUST identify all nodes with compatible strings matching known FPGA soft-IP drivers
- AND for each matched node, the kernel MUST extract the reg property (MMIO base and size) and interrupts property

#### Scenario: Handle unknown FPGA peripheral
- WHEN the DTB contains a node with an unrecognized compatible string
- THEN the kernel MUST log a warning with the compatible string and node path
- AND the kernel MUST skip the node without affecting boot of other peripherals

#### Scenario: Validate peripheral address ranges
- WHEN the kernel discovers a peripheral from the DTB
- THEN it MUST verify the MMIO address range does not overlap with kernel memory or other peripherals
- AND if an overlap is detected, the kernel MUST skip the peripheral and log an error

### Requirement: FPGA Fabric Interrupt Routing
The kernel SHALL support routing interrupts from FPGA fabric peripherals to the CPU interrupt controller (PLIC on RISC-V, GIC on ARM).

#### Scenario: Register FPGA interrupt on PLIC-based system
- WHEN an FPGA peripheral driver registers its interrupt on a RISC-V system with PLIC
- THEN the interrupt routing layer MUST map the FPGA interrupt line to the correct PLIC source ID
- AND the PLIC driver MUST enable and set the priority for that source

#### Scenario: Register FPGA interrupt on GIC-based system
- WHEN an FPGA peripheral driver registers its interrupt on an ARM system with GIC
- THEN the interrupt routing layer MUST map the FPGA interrupt line to the correct GIC SPI number
- AND the GIC driver MUST enable and configure the trigger type (level or edge) for that SPI

#### Scenario: Shared interrupt line
- WHEN multiple FPGA peripherals share a single interrupt line
- THEN the interrupt routing layer MUST dispatch to all registered handlers for that line
- AND each handler MUST check its peripheral's status register to determine if it is the interrupt source

### Requirement: Xilinx/AMD Zynq UltraScale+ Platform Support Package
The kernel SHALL provide a platform support package for the Xilinx/AMD Zynq UltraScale+ MPSoC, covering the PS-PL interface and clock/reset management for FPGA fabric peripherals.

#### Scenario: PS-PL AXI interface initialization
- WHEN the kernel boots on a Zynq UltraScale+ platform
- THEN it MUST configure the PS-PL AXI master and slave interfaces (HPM, HPC, LPD)
- AND each interface MUST be enabled only if the corresponding FPGA peripheral is present in the DTB

#### Scenario: FPGA fabric clock management
- WHEN a Zynq UltraScale+ FPGA peripheral requires a specific clock frequency
- THEN the platform support package MUST configure the PL fabric clock (FCLK) via the Clock Wizard or CRL_APB registers
- AND the configured frequency MUST be within 1% of the requested value

#### Scenario: FPGA fabric reset management
- WHEN the kernel initializes or reinitializes an FPGA peripheral
- THEN the platform support package MUST assert and deassert the corresponding PL reset line via the CRL_APB FPGA_RST_CTRL register
- AND the reset pulse MUST be held for at least the duration specified by the peripheral's datasheet

### Requirement: Microchip PolarFire SoC Platform Support Package
The kernel SHALL provide a platform support package for the Microchip PolarFire SoC, which combines RISC-V harts with FPGA fabric.

#### Scenario: PolarFire SoC boot with FPGA fabric
- WHEN the kernel boots on a PolarFire SoC
- THEN it MUST initialize the RISC-V harts via the platform's HSS (Hart Software Services) boot flow
- AND the kernel MUST detect FPGA fabric peripherals via the DTB provided by HSS

#### Scenario: PolarFire SoC fabric interface controller (FIC) initialization
- WHEN the kernel accesses FPGA peripherals on PolarFire SoC
- THEN it MUST configure the Fabric Interface Controllers (FIC0-FIC3) for AXI4 access to the fabric
- AND each FIC MUST be initialized with the correct address mapping before any peripheral access

#### Scenario: PolarFire SoC MSS-fabric interrupt routing
- WHEN FPGA fabric peripherals generate interrupts on PolarFire SoC
- THEN the platform support package MUST route them through the MSS (Microprocessor SubSystem) interrupt controller to the RISC-V PLIC
- AND the interrupt mapping MUST match the fabric design's interrupt assignment

### Requirement: FPGA Peripherals as MMIO Devices
FPGA peripherals SHALL be treated as standard memory-mapped I/O devices. The kernel SHALL NOT perform bitstream programming, partial reconfiguration, or any FPGA configuration operations.

#### Scenario: Access FPGA peripheral via standard MMIO
- WHEN a driver accesses an FPGA-implemented peripheral
- THEN the access MUST use the same MMIO read/write interfaces as any platform device
- AND no FPGA-specific configuration or programming interface SHALL be invoked

#### Scenario: Reject bitstream programming requests
- WHEN any subsystem attempts to invoke FPGA bitstream programming or partial reconfiguration
- THEN the kernel MUST return an unsupported-operation error
- AND the kernel MUST NOT expose any API for FPGA configuration management
