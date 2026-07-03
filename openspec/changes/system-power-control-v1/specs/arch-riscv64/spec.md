## ADDED Requirements

### Requirement: RISC-V platform power contract

The `arch/riscv64` HAL SHALL provide `platform_reset()` via the SBI SRST extension and `platform_off()` via the SBI HSM extension (`SBI_HSM_HART_STOP`). For v1 these paths SHALL be validated under QEMU only (no hardware available), and that limitation SHALL be documented.

#### Scenario: platform_reset uses SBI SRST under QEMU

- **WHEN** the kernel invokes `platform_reset()` on RISC-V under QEMU
- **THEN** the HAL SHALL issue an SBI SRST reset call
- **AND** the QEMU guest SHALL reset

#### Scenario: platform_off stops the hart via SBI HSM

- **WHEN** the kernel invokes `platform_off()` on RISC-V under QEMU
- **THEN** the HAL SHALL issue `SBI_HSM_HART_STOP`
- **AND** the QEMU guest SHALL halt without resuming

#### Scenario: QEMU-only validation is documented

- **WHEN** a reviewer reads the change's documentation for the RISC-V power paths
- **THEN** it SHALL state that the SBI SRST/HSM paths are QEMU-validated only for v1
