## ADDED Requirements

### Requirement: AArch64 platform power contract

The `arch/aarch64` HAL SHALL provide `platform_reset()` and `platform_off()` implemented via the existing PSCI wrappers — `psci::system_reset()` and `psci::system_off()` — invoked through `HVC #0`. Both paths SHALL be verified on QEMU virt and on Jetson Orin under KVM.

#### Scenario: platform_reset issues PSCI SYSTEM_RESET

- **WHEN** the kernel invokes `platform_reset()` on AArch64
- **THEN** the HAL SHALL call `psci::system_reset()` via `HVC #0`
- **AND** the machine SHALL reset (observed on QEMU virt and on Jetson Orin under KVM)

#### Scenario: platform_off issues PSCI SYSTEM_OFF

- **WHEN** the kernel invokes `platform_off()` on AArch64
- **THEN** the HAL SHALL call `psci::system_off()` via `HVC #0`
- **AND** the machine SHALL halt without resuming
