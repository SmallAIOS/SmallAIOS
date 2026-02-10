# Delta for RISC-V Architecture

## ADDED Requirements

### Requirement: OpenSBI Boot
The kernel SHALL boot on RISC-V RV64GC platforms via OpenSBI M-mode firmware, with SmallAIOS executing in S-mode. The target triple SHALL be riscv64gc-unknown-none-elf.

#### Scenario: S-mode entry from OpenSBI
- WHEN OpenSBI completes M-mode initialization on the RISC-V platform
- THEN SmallAIOS MUST receive control in S-mode at the designated entry point
- AND the hart ID and device tree pointer MUST be available in registers a0 and a1

#### Scenario: QEMU virt machine boot
- WHEN SmallAIOS is loaded on the QEMU virt machine for riscv64
- THEN the kernel MUST boot successfully through OpenSBI to the kernel_main entry point
- AND QEMU virt SHALL be the primary test and development platform

### Requirement: Assembly Entry Point
The kernel SHALL provide a RISC-V assembly entry point that sets up the initial execution environment before calling kernel_main.

#### Scenario: Stack initialization
- WHEN the assembly entry point receives control from OpenSBI
- THEN it MUST set up the boot stack pointer to a pre-allocated, 16-byte-aligned stack region
- AND the stack MUST be sized to at least 64 KiB for the boot hart

#### Scenario: BSS clearing
- WHEN the assembly entry point executes before kernel_main
- THEN it MUST zero the entire BSS section from __bss_start to __bss_end
- AND BSS clearing MUST complete before any Rust code executes

#### Scenario: Trap vector configuration
- WHEN the assembly entry point initializes the execution environment
- THEN it MUST write the trap handler base address to the stvec CSR
- AND the trap vector MUST be configured before enabling any interrupts

#### Scenario: Transfer to kernel_main
- WHEN stack, BSS, and trap vector initialization are complete
- THEN the entry point MUST call kernel_main with the hart ID and DTB pointer as arguments
- AND kernel_main MUST NOT return (the entry point SHALL include a terminal spin loop as a safety net)

### Requirement: SV48 Page Table Management
The kernel SHALL implement SV48 4-level page table management for RISC-V, providing virtual-to-physical address translation with 48-bit virtual addresses.

#### Scenario: Map a virtual page
- WHEN the kernel requests mapping a virtual address to a physical address
- THEN the SV48 page table walker MUST create or update entries across all four levels (PGD, PUD, PMD, PTE)
- AND the mapping MUST respect the requested permission flags (read, write, execute)

#### Scenario: Unmap a virtual page
- WHEN the kernel requests unmapping a virtual address
- THEN the page table manager MUST clear the corresponding PTE
- AND the manager MUST issue an sfence.vma instruction to invalidate the TLB entry for that address

#### Scenario: Protect a mapped page
- WHEN the kernel requests a permission change on an existing mapping
- THEN the page table manager MUST update the R/W/X bits in the PTE without changing the physical address
- AND the manager MUST issue an sfence.vma instruction to flush the stale TLB entry

#### Scenario: TLB flush via sfence.vma
- WHEN any page table modification occurs (map, unmap, or protect)
- THEN the kernel MUST execute sfence.vma with the appropriate address and ASID arguments
- AND on SMP systems the flush MUST be broadcast to all harts that may have cached the translation

### Requirement: PLIC Interrupt Controller Driver
The kernel SHALL implement a driver for the Platform-Level Interrupt Controller (PLIC) to manage external interrupt sources.

#### Scenario: Set interrupt priority
- WHEN a device driver registers an interrupt source with the PLIC
- THEN the PLIC driver MUST write the requested priority value (1-7) to the source's priority register
- AND priority 0 MUST effectively disable the interrupt source

#### Scenario: Enable interrupt for a hart context
- WHEN the kernel enables an interrupt source for a specific hart S-mode context
- THEN the PLIC driver MUST set the corresponding bit in the enable register for that context
- AND the interrupt MUST be deliverable to the hart when its priority exceeds the context threshold

#### Scenario: Claim and complete an interrupt
- WHEN a hart receives an external interrupt (scause = supervisor external interrupt)
- THEN the interrupt handler MUST read the PLIC claim register to obtain the source ID
- AND after servicing the interrupt, the handler MUST write the source ID to the complete register
- AND claiming with no pending interrupt MUST return source ID 0

### Requirement: CLINT Timer Driver
The kernel SHALL implement a driver for the Core Local Interruptor (CLINT) to provide periodic timer ticks using the mtime and mtimecmp registers.

#### Scenario: Configure periodic timer tick
- WHEN the scheduler requires a periodic timer interrupt
- THEN the CLINT driver MUST program mtimecmp for the target hart to mtime + tick_interval
- AND each tick handler MUST reprogram mtimecmp for the next tick

#### Scenario: Read current time
- WHEN the kernel queries the current time
- THEN the CLINT driver MUST read the mtime register and return a monotonic timestamp
- AND the timestamp MUST be consistent across all harts (mtime is shared)

### Requirement: SBI HSM Extension for SMP Boot
The kernel SHALL use the SBI Hart State Management (HSM) extension to boot secondary harts for symmetric multiprocessing.

#### Scenario: Start a secondary hart
- WHEN the boot hart has completed kernel initialization and needs to bring up secondary harts
- THEN it MUST call sbi_hart_start with the target hart ID, start address, and opaque value
- AND the secondary hart MUST begin execution at the specified start address in S-mode

#### Scenario: Stop a hart
- WHEN the kernel decides to take a hart offline (power management or error isolation)
- THEN it MUST call sbi_hart_stop on the target hart
- AND the hart MUST enter the STOPPED state and cease executing instructions

#### Scenario: Query hart status
- WHEN the kernel needs to determine whether a specific hart is running
- THEN it MUST call sbi_hart_get_status with the target hart ID
- AND the call MUST return one of: STARTED, STOPPED, START_PENDING, or STOP_PENDING

### Requirement: SBI IPI Extension
The kernel SHALL use the SBI IPI extension to send inter-processor interrupts between harts.

#### Scenario: Send IPI to a set of harts
- WHEN a hart needs to notify one or more remote harts (e.g., for TLB shootdown or scheduler wakeup)
- THEN it MUST call sbi_send_ipi with the target hart mask
- AND each targeted hart MUST receive a supervisor software interrupt

#### Scenario: Handle received IPI
- WHEN a hart receives a supervisor software interrupt caused by an IPI
- THEN the handler MUST clear the SSIP bit in the sip CSR
- AND the handler MUST dispatch to the appropriate IPI reason handler (TLB shootdown, reschedule, etc.)

### Requirement: UART Driver
The kernel SHALL implement a UART driver compatible with the NS16550A interface for serial console I/O on the QEMU virt machine.

#### Scenario: Transmit a character
- WHEN the kernel writes a byte to the serial console
- THEN the UART driver MUST poll the Line Status Register until the Transmitter Holding Register is empty
- AND then write the byte to the Transmitter Holding Register

#### Scenario: Receive a character
- WHEN the UART signals data available via interrupt or polling
- THEN the driver MUST read the Receiver Buffer Register
- AND return the received byte to the kernel console subsystem

#### Scenario: Initialize UART on QEMU virt
- WHEN the kernel boots on the QEMU virt machine
- THEN the UART driver MUST configure the NS16550A at the platform-defined MMIO base address (0x10000000)
- AND the driver MUST set baud rate divisor, enable FIFOs, and configure interrupt enables

### Requirement: Feature Detection
The kernel SHALL detect RISC-V ISA extensions at runtime to verify the required RV64GC feature set and optional extensions.

#### Scenario: Detect atomic extension (A)
- WHEN the kernel initializes on a RISC-V hart
- THEN it MUST verify the presence of the Atomic (A) extension via the misa CSR or device tree
- AND if the A extension is absent, the kernel MUST panic with a descriptive error

#### Scenario: Detect compressed extension (C)
- WHEN the kernel initializes on a RISC-V hart
- THEN it MUST verify the presence of the Compressed (C) extension
- AND if absent, the kernel MUST log a warning but MAY continue if the binary does not use compressed instructions

#### Scenario: Detect floating-point extensions (D/F)
- WHEN the kernel initializes on a RISC-V hart
- THEN it MUST detect the presence of single-precision (F) and double-precision (D) floating-point extensions
- AND the kernel MUST enable the FPU by setting the FS field in sstatus before any floating-point operation
- AND if D/F extensions are absent, the kernel MUST disable hardware FPU usage and report the limitation
