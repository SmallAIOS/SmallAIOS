# Delta for Device and Hardware Abstraction Layer

## ADDED Requirements

### Requirement: Bus Peripheral HAL Trait

The HAL SHALL define a `BusController` trait providing a uniform interface for bus peripheral controllers across CAN, ARINC 429, ARINC 664, MIL-STD-1553, SpaceWire, and CCSDS hardware. The trait SHALL include methods for controller initialization, frame/word transmit, frame/word receive, interrupt handling, and error status reporting. Each bus protocol implementation SHALL provide a concrete type implementing `BusController` with protocol-specific configuration (e.g., CAN baud rate, ARINC 429 speed selection, MIL-STD-1553 bus controller vs. remote terminal mode).

#### Scenario: Initialize CAN controller via HAL trait

- WHEN the kernel initializes a CAN bus controller discovered via DTB
- THEN the HAL MUST call `BusController::init()` with a `BusConfig::Can` configuration specifying baud rate (125 Kbps to 1 Mbps for CAN 2.0, up to 8 Mbps data phase for CAN FD) and acceptance filter masks
- AND `init()` MUST return `Ok(())` when the controller enters operational state
- AND `init()` MUST return `Err(BusError::InitFailed)` if the controller does not respond within 100 ms

#### Scenario: Transmit a CAN frame via HAL trait

- WHEN a component calls `BusController::tx_frame()` with a CAN frame (ID, DLC, data)
- THEN the HAL MUST submit the frame to the controller's transmit FIFO
- AND MUST return `Ok(())` when the frame is accepted for transmission
- AND MUST return `Err(BusError::TxFifoFull)` if the transmit FIFO has no available slots

#### Scenario: Receive a frame via HAL trait with interrupt notification

- WHEN a bus controller receives a frame and raises an interrupt
- THEN the HAL top-half interrupt handler MUST acknowledge the interrupt and enqueue a work item
- AND the bottom-half handler MUST call `BusController::rx_frame()` to retrieve the received frame
- AND the frame MUST be delivered to the registered transport adapter callback

#### Scenario: Report bus error status

- WHEN a CAN controller enters bus-off state due to excessive transmit errors
- THEN `BusController::error_status()` MUST return `BusError::BusOff`
- AND the HAL MUST initiate automatic bus-off recovery per ISO 11898-1
- AND MUST log the bus-off event to the kernel syslog

#### Scenario: Initialize ARINC 429 transceiver via HAL trait

- WHEN the kernel initializes an ARINC 429 transceiver
- THEN `BusController::init()` MUST configure the channel speed (low speed 12.5 Kbps or high speed 100 Kbps)
- AND MUST configure label filtering for receive channels
- AND the transceiver MUST be ready for transmit and receive operations within 50 ms

#### Scenario: Initialize MIL-STD-1553 controller in bus controller mode

- WHEN the kernel initializes a MIL-STD-1553 controller with `BusConfig::Mil1553 { mode: BusControllerMode }`
- THEN the HAL MUST configure the controller for bus controller operation on the specified bus (A, B, or both for redundancy)
- AND MUST set up the message schedule table for cyclic command transmission

### Requirement: FPGA Fabric Interface HAL

The HAL SHALL define an `FpgaFabric` trait providing methods for AXI/AXI-Lite memory-mapped register read and write operations and AXI DMA transfer initiation and completion. The trait SHALL abstract the FPGA fabric bus so that soft-IP peripherals instantiated in the FPGA appear as standard memory-mapped devices to higher layers. All register accesses MUST use volatile reads/writes with appropriate memory barriers.

#### Scenario: Read an AXI-Lite register from FPGA peripheral

- WHEN a driver calls `FpgaFabric::read_reg(base_addr, offset)` to read a 32-bit register from an FPGA soft-IP peripheral
- THEN the HAL MUST perform a volatile MMIO read at the computed physical address (base_addr + offset)
- AND MUST issue a read memory barrier after the access
- AND MUST return the 32-bit register value

#### Scenario: Write an AXI-Lite register to FPGA peripheral

- WHEN a driver calls `FpgaFabric::write_reg(base_addr, offset, value)` to write a 32-bit register
- THEN the HAL MUST issue a write memory barrier before the access
- AND MUST perform a volatile MMIO write at the computed physical address
- AND the write MUST be visible to the FPGA peripheral within one AXI clock cycle

#### Scenario: Initiate an AXI DMA transfer from FPGA to system memory

- WHEN a driver calls `FpgaFabric::dma_start(channel, src_addr, dst_addr, length, direction)` with direction `FpgaToMemory`
- THEN the HAL MUST program the AXI DMA controller registers with the source, destination, and transfer length
- AND MUST start the DMA transfer
- AND MUST return a `DmaToken` that can be used to poll or await completion

#### Scenario: DMA transfer completes with interrupt

- WHEN an AXI DMA transfer completes and the DMA controller raises a completion interrupt
- THEN the HAL MUST acknowledge the interrupt
- AND MUST mark the corresponding `DmaToken` as complete
- AND any task awaiting the `DmaToken` MUST be woken by the scheduler

#### Scenario: DMA transfer error handling

- WHEN an AXI DMA transfer encounters a bus error (e.g., decode error, slave error)
- THEN the HAL MUST mark the `DmaToken` as failed with `DmaError::BusError`
- AND MUST reset the DMA channel to a known-good state
- AND MUST log the error including the faulting address and error type

### Requirement: RISC-V HAL Implementation

The HAL SHALL provide a RISC-V (RV64GC) architecture implementation in a `smallaios-arch-riscv64` crate, following the same pattern as the existing x86-64 and AArch64 HAL crates. The implementation SHALL support SV48 four-level page tables, PLIC (Platform-Level Interrupt Controller) for external interrupt routing, CLINT (Core Local Interruptor) for timer and inter-processor interrupts, and SBI (Supervisor Binary Interface) calls for firmware services. SmallAIOS SHALL run in S-mode (supervisor mode) with OpenSBI providing M-mode firmware.

#### Scenario: RISC-V boot via OpenSBI

- WHEN SmallAIOS boots on a RISC-V platform with OpenSBI firmware
- THEN the entry point MUST receive the hart ID in register `a0` and the DTB pointer in register `a1`
- AND the boot hart MUST set up the SV48 page tables, configure `satp` with mode=9 (SV48), and enable the MMU
- AND the boot hart MUST initialize the PLIC and CLINT before starting secondary harts

#### Scenario: SV48 page table setup

- WHEN the RISC-V HAL initializes virtual memory
- THEN it MUST create a four-level page table hierarchy (root table, level-2, level-1, level-0) with 4 KiB pages
- AND MUST support 2 MiB mega-pages (level-1 leaf entries) and 1 GiB giga-pages (level-2 leaf entries) for large mappings
- AND MUST set the PTE permission bits (R, W, X) and status bits (V, U, G, A, D) correctly for each mapping
- AND MUST flush the TLB via `sfence.vma` after page table modifications

#### Scenario: PLIC interrupt handling

- WHEN an external interrupt is pending in the PLIC
- THEN the HAL MUST read the PLIC claim register for the current hart to determine the interrupt source
- AND MUST dispatch to the registered interrupt handler for that source
- AND MUST write the interrupt ID to the PLIC complete register after the handler returns
- AND interrupt priority thresholds MUST be configurable per-hart

#### Scenario: CLINT timer interrupt

- WHEN the CLINT timer fires (mtime >= mtimecmp for the current hart)
- THEN the HAL MUST handle the supervisor timer interrupt (cause = 5)
- AND MUST set the next timer deadline by calling SBI `sbi_set_timer()` with the next tick value
- AND the timer interrupt MUST be used for scheduler tick processing

#### Scenario: SMP boot via SBI HSM extension

- WHEN the boot hart needs to start secondary harts for SMP operation
- THEN the HAL MUST call `sbi_hart_start(hartid, start_addr, opaque)` from the SBI HSM extension for each secondary hart
- AND each secondary hart MUST begin execution at `start_addr` with its hart ID in `a0`
- AND each secondary hart MUST initialize its own PLIC context and CLINT timer before entering the scheduler

#### Scenario: SBI console output for early boot

- WHEN the kernel needs to output text during early boot before the full IPC/logging stack is ready
- THEN the HAL MUST use `sbi_debug_console_write()` (SBI DBCN extension) to write bytes to the firmware console
- AND MUST fall back to legacy `sbi_console_putchar()` if the DBCN extension is not available

#### Scenario: RISC-V CPU feature detection

- WHEN the RISC-V HAL initializes on a hart
- THEN it MUST read the `misa` CSR (via SBI or DTB) to determine the supported ISA extensions (I, M, A, F, D, C)
- AND MUST parse the DTB `riscv,isa` string for extensions not reflected in `misa`
- AND MUST populate a `RiscvFeatures` struct with detected capabilities including base ISA, multiply/divide, atomics, single/double float, and compressed instructions
