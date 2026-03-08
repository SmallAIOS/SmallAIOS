## ADDED Requirements

### Requirement: Tegra X1 platform feature flag
The `arch/aarch64` crate SHALL provide a `tegra-x1` Cargo feature flag that selects Tegra X1 SoC-specific constants (UART base, GIC base, memory map). The `qemu-virt` feature SHALL remain the default.

#### Scenario: Build with tegra-x1 feature
- **WHEN** the crate is compiled with `--features tegra-x1`
- **THEN** all platform constants SHALL resolve to Tegra X1 addresses (UART at `0x70006000`, GICD at `0x50041000`, GICC at `0x50042000`, DRAM base at `0x80000000`)

#### Scenario: Build with default features
- **WHEN** the crate is compiled without explicit feature flags
- **THEN** all platform constants SHALL resolve to QEMU virt addresses (PL011 at `0x09000000`, GICD at `0x08000000`, GICR at `0x080A0000`)

#### Scenario: Mutually exclusive features
- **WHEN** both `tegra-x1` and `qemu-virt` features are enabled simultaneously
- **THEN** the build SHALL fail with a compile error indicating the features are mutually exclusive

### Requirement: Tegra X1 memory map constants
The platform module SHALL expose named constants for all Tegra X1 peripheral base addresses used by SmallAIOS.

#### Scenario: Memory map constants available
- **WHEN** the `tegra-x1` feature is enabled
- **THEN** the module SHALL export constants for: UART-A base (`0x70006000`), GIC Distributor (`0x50041000`), GIC CPU Interface (`0x50042000`), DRAM base (`0x80000000`), PCIe root complex (`0x01003000`), and CAR (`0x60006000`)

### Requirement: Tegra-specific linker script
The build system SHALL provide a linker script `arch/aarch64/linker-tegra.ld` with base address `0x80080000` for Tegra X1 kernel builds.

#### Scenario: Tegra kernel linked at correct address
- **WHEN** the kernel is built with the `tegra-x1` feature
- **THEN** the `.text.boot` section SHALL begin at physical address `0x80080000`

#### Scenario: QEMU kernel address unchanged
- **WHEN** the kernel is built with default features (qemu-virt)
- **THEN** the `.text.boot` section SHALL begin at physical address `0x40080000`

### Requirement: Early boot UART for Tegra X1
The `uart` module SHALL provide a polled TX-only NS16550A writer at `0x70006000` with register shift 2 when the `tegra-x1` feature is enabled. This writer SHALL require no initialization (U-Boot pre-configures UART-A at 115200 8N1).

#### Scenario: Print to UART immediately after boot
- **WHEN** `_start` transfers control to `kernel_main` on Tegra X1 hardware
- **THEN** calling `uart::puts("Hello")` SHALL produce output on UART-A (J44 debug header) at 115200 baud with no prior init call

#### Scenario: UART character output
- **WHEN** `uart::putc(byte)` is called
- **THEN** the function SHALL spin-wait on the LSR THRE bit at `0x70006014` and write the byte to THR at `0x70006000`

### Requirement: EL2 to EL1 transition
The boot assembly SHALL detect the current exception level and drop from EL2 to EL1 if the firmware hands off at EL2.

#### Scenario: Boot at EL2
- **WHEN** U-Boot/ATF transfers control at EL2
- **THEN** `_start` SHALL configure EL2 registers (HCR_EL2, SCTLR_EL1, SPSR_EL2) and perform an `eret` to enter EL1 before calling `kernel_main`

#### Scenario: Boot at EL1
- **WHEN** firmware transfers control at EL1
- **THEN** `_start` SHALL proceed directly to BSS clear and stack setup without any EL2 register access
