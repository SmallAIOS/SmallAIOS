## ADDED Requirements

### Requirement: Boot measurement log records component hashes
The kernel SHALL maintain a boot measurement log that records the SHA-3-256 hash of every component loaded during boot. Each measurement entry SHALL contain: a component identifier (up to 64 bytes), the SHA-3-256 digest, a monotonic timestamp, and a verification status (Verified, Unverified, Failed). The log SHALL have a fixed capacity of 32 entries and SHALL be immutable after boot completes (no entries can be added or removed post-boot).

#### Scenario: Kernel text section is measured at boot
- **WHEN** the kernel boots with the `verified-boot` feature enabled
- **THEN** the boot measurement log contains an entry with component identifier `kernel:text+rodata` and a valid SHA-3-256 hash of the kernel's `.text` and `.rodata` sections

#### Scenario: DTB is measured at boot on AArch64 and RISC-V
- **WHEN** the kernel boots on AArch64 or RISC-V with `verified-boot` enabled and a DTB pointer is provided by firmware
- **THEN** the boot measurement log contains an entry with component identifier `firmware:dtb` and the SHA-3-256 hash of the DTB blob

#### Scenario: Measurement log is immutable after boot
- **WHEN** boot initialization completes and the kernel transitions to normal operation
- **THEN** any attempt to add, modify, or remove entries from the boot measurement log SHALL return an error

#### Scenario: Measurement log is queryable via IPC
- **WHEN** a component with the appropriate capability queries the boot measurement log
- **THEN** the system returns all measurement entries with their component identifiers, hashes, timestamps, and verification statuses

### Requirement: Kernel self-integrity verification
The kernel SHALL verify its own integrity at early boot by comparing a SHA-3-256 hash of its `.text` and `.rodata` sections against an Ed25519-signed hash embedded in the `.boot_sig` section. If verification fails, the kernel SHALL log the failure to the boot measurement log and either halt (in Enforce mode) or continue with a warning (in WarnOnly mode).

#### Scenario: Self-integrity check passes
- **WHEN** the kernel boots with `verified-boot` enabled and the computed SHA-3-256 hash matches the embedded signed hash
- **THEN** the boot measurement log entry for `kernel:text+rodata` has verification status Verified and boot proceeds normally

#### Scenario: Self-integrity check fails in Enforce mode
- **WHEN** the kernel boots with `verified-boot` enabled, verification policy is Enforce, and the computed hash does not match the embedded signed hash
- **THEN** the kernel logs a boot measurement entry with status Failed and halts execution

#### Scenario: Self-integrity check fails in WarnOnly mode
- **WHEN** the kernel boots with `verified-boot` enabled, verification policy is WarnOnly, and the computed hash does not match the embedded signed hash
- **THEN** the kernel logs a boot measurement entry with status Failed and continues boot with a warning logged to the kernel ring buffer

#### Scenario: No embedded signature present
- **WHEN** the kernel boots with `verified-boot` enabled but the `.boot_sig` section is empty or absent
- **THEN** the boot measurement log entry has status Unverified and the kernel follows the verification policy (halt if Enforce, continue if WarnOnly)

### Requirement: ONNX model signature verification at load time
The ONNX runtime SHALL verify the signature of every model before execution using the existing `security::crypto::verify` module. Verification SHALL support Ed25519, ML-DSA-65, and hybrid Ed25519+ML-DSA-65 signatures. The verification policy (Enforce, WarnOnly, Disabled) SHALL be configurable at build time via feature flags and at runtime via kernel configuration.

#### Scenario: Signed model passes verification
- **WHEN** an ONNX model with a valid signature is loaded for inference with verification policy Enforce
- **THEN** the model is loaded, a boot measurement entry is recorded with status Verified, and inference proceeds

#### Scenario: Unsigned model rejected in Enforce mode
- **WHEN** an ONNX model without a signature block is loaded with verification policy Enforce
- **THEN** the load fails with an error, a boot measurement entry is recorded with status Failed, and no inference executes

#### Scenario: Unsigned model allowed in WarnOnly mode
- **WHEN** an ONNX model without a signature block is loaded with verification policy WarnOnly
- **THEN** the model is loaded, a boot measurement entry is recorded with status Unverified, a warning is logged, and inference proceeds

#### Scenario: Tampered model detected
- **WHEN** an ONNX model with a signature block is loaded but the SHA-3-256 hash of the model bytes does not match the hash in the signature
- **THEN** the load fails regardless of verification policy, a boot measurement entry is recorded with status Failed, and the error includes the expected and actual hashes

### Requirement: Verification policy configuration
The system SHALL support three verification policies: Enforce (reject failures), WarnOnly (log failures but continue), and Disabled (skip verification). The policy SHALL be configurable independently for kernel self-integrity and model verification. The default policy when `verified-boot` is enabled SHALL be WarnOnly.

#### Scenario: Policy defaults to WarnOnly
- **WHEN** the kernel is built with `verified-boot` feature and no explicit policy configuration
- **THEN** both kernel self-integrity and model verification use WarnOnly policy

#### Scenario: Policy can be set to Enforce
- **WHEN** the kernel configuration sets verification policy to Enforce
- **THEN** any integrity or signature failure causes the corresponding operation to halt or reject

#### Scenario: Disabled policy skips all verification
- **WHEN** the verification policy is set to Disabled
- **THEN** no hash computation or signature verification occurs, and no boot measurement entries are created for verification checks
