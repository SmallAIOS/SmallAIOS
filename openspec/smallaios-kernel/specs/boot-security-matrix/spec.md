## ADDED Requirements

### Requirement: Platform boot security comparison matrix
The project SHALL maintain a cross-platform comparison of boot security capabilities documenting the hardware-assisted and firmware-level protections available on each supported architecture: x86-64 (UEFI Secure Boot, TPM, Intel Boot Guard), AArch64 (ARM Trusted Firmware, TrustZone, Secure Boot), and RISC-V (OpenSBI, PMP, vendor-specific secure boot). The matrix SHALL identify what SmallAIOS can leverage on each platform and what remains as accepted risk.

#### Scenario: Matrix covers all three architectures
- **WHEN** the boot security matrix document is reviewed
- **THEN** it contains entries for x86-64, AArch64, and RISC-V with at least: firmware trust chain, hardware root of trust, measured boot capability, secure boot mechanism, and runtime integrity protection

#### Scenario: Matrix identifies SmallAIOS integration points
- **WHEN** the boot security matrix is reviewed for a given architecture
- **THEN** each capability row identifies whether SmallAIOS currently integrates with it (Yes/No/Partial), what would be needed to integrate, and whether integration requires hardware access

### Requirement: Boot trust boundary documentation
The project SHALL document the trust boundaries at each stage of the boot process for each architecture. The documentation SHALL clearly distinguish between: firmware-verified stages (outside SmallAIOS control), SmallAIOS-verified stages (kernel self-integrity, model signatures), and unverified stages (accepted risks).

#### Scenario: Trust boundaries are explicit per architecture
- **WHEN** the trust boundary documentation is reviewed for a given architecture
- **THEN** each boot stage is classified as firmware-verified, SmallAIOS-verified, or unverified with a rationale for the classification

#### Scenario: Accepted risks are documented with rationale
- **WHEN** a boot stage is classified as unverified
- **THEN** the documentation includes the specific threat it leaves open and why the risk is accepted (e.g., requires hardware not available, diminishing returns for threat model)

### Requirement: Platform-specific boot security recommendations
The project SHALL provide actionable recommendations for deploying SmallAIOS securely on each platform, covering both the software-level protections implemented in this change and the hardware-level protections available per platform.

#### Scenario: x86-64 deployment recommendations
- **WHEN** a user reviews the x86-64 recommendations
- **THEN** the document covers UEFI Secure Boot key enrollment, kernel signing for GRUB/shim, and how SmallAIOS `verified-boot` feature complements UEFI Secure Boot

#### Scenario: AArch64 deployment recommendations
- **WHEN** a user reviews the AArch64 recommendations
- **THEN** the document covers ARM Trusted Firmware configuration, U-Boot verified boot, and how SmallAIOS `verified-boot` feature complements ATF

#### Scenario: RISC-V deployment recommendations
- **WHEN** a user reviews the RISC-V recommendations
- **THEN** the document covers OpenSBI PMP configuration, platform-specific secure boot options, and how SmallAIOS `verified-boot` feature complements them
