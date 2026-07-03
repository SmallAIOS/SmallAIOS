## ADDED Requirements

### Requirement: x86-64 platform power contract

The `arch/x86_64` HAL SHALL provide `platform_off()` via ACPI `\_S5` and `platform_reset()` via the 8042 keyboard controller (writing `0xFE` to port `0x64`), with a triple-fault fallback when the 8042 reset does not take effect.

#### Scenario: platform_reset uses the 8042 controller

- **WHEN** the kernel invokes `platform_reset()` on x86-64
- **THEN** the HAL SHALL write `0xFE` to port `0x64`
- **AND** the machine SHALL reset

#### Scenario: Triple-fault fallback when 8042 reset fails

- **WHEN** the 8042 reset write does not take effect
- **THEN** the HAL SHALL fall back to forcing a triple fault
- **AND** the machine SHALL still reset

#### Scenario: platform_off enters ACPI S5

- **WHEN** the kernel invokes `platform_off()` on x86-64
- **THEN** the HAL SHALL enter ACPI `\_S5`
- **AND** the machine SHALL power off without resuming
