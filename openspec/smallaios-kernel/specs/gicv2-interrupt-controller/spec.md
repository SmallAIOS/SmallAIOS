## ADDED Requirements

### Requirement: GICv2 Distributor initialization
The GICv2 driver SHALL initialize the distributor at the platform-configured GICD base address, enabling interrupt forwarding to the CPU interface.

#### Scenario: Distributor init on Tegra X1
- **WHEN** `gicv2::init_gicd()` is called with GICD at `0x50041000`
- **THEN** the distributor SHALL be enabled (GICD_CTLR bit 0 set) and the physical timer PPI (IRQ 30) SHALL be enabled in GICD_ISENABLER

### Requirement: GICv2 CPU Interface initialization
The GICv2 driver SHALL initialize the CPU interface at the platform-configured GICC base address, setting the priority mask to accept all interrupts.

#### Scenario: CPU interface init
- **WHEN** `gicv2::init_cpu_interface()` is called with GICC at `0x50042000`
- **THEN** GICC_CTLR SHALL be set to enable interrupt signaling and GICC_PMR SHALL be set to `0xFF` (accept all priorities)

### Requirement: Interrupt acknowledge and end-of-interrupt
The GICv2 driver SHALL provide functions to acknowledge interrupts (read GICC_IAR) and signal end-of-interrupt (write GICC_EOIR) via MMIO.

#### Scenario: Acknowledge timer interrupt
- **WHEN** a timer interrupt fires and `gicv2::iar()` is called
- **THEN** the function SHALL return the interrupt ID (30 for physical timer) read from GICC_IAR at GICC + `0x0C`

#### Scenario: End of interrupt
- **WHEN** `gicv2::eoir(irq_id)` is called after handling an interrupt
- **THEN** the function SHALL write `irq_id` to GICC_EOIR at GICC + `0x10`

### Requirement: SPI enable/disable
The GICv2 driver SHALL allow enabling and disabling individual Shared Peripheral Interrupts (SPI, IRQ 32+).

#### Scenario: Enable an SPI
- **WHEN** `gicv2::enable_irq(irq_id)` is called with `irq_id >= 32`
- **THEN** the corresponding bit in GICD_ISENABLER[n] SHALL be set

#### Scenario: Disable an SPI
- **WHEN** `gicv2::disable_irq(irq_id)` is called with `irq_id >= 32`
- **THEN** the corresponding bit in GICD_ICENABLER[n] SHALL be set

### Requirement: Same public API as GICv3
The GICv2 module SHALL export the same function signatures as the GICv3 module for interrupt acknowledge, EOI, and timer init, so that platform-independent kernel code can use either without conditional compilation.

#### Scenario: API compatibility
- **WHEN** kernel code calls `interrupts::iar()`, `interrupts::eoir(id)`, `interrupts::init_timer(ticks)`
- **THEN** these functions SHALL resolve to GICv2 or GICv3 implementations based on the platform feature flag, with identical signatures and semantics
