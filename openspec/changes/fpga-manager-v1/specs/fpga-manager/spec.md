## ADDED Requirements

### Requirement: FpgaManager Full Bitstream Loading

The ZynqMP platform support SHALL provide an `FpgaManager` API exposing `load_bitstream(&[u8]) -> Result<()>`, which performs a full PL reconfiguration by handing validated raw configuration data to the PMU over the IPI channel for programming via the PCAP / CSU DMA path. Loading SHALL NOT require a reboot or power cycle.

#### Scenario: Runtime overlay swap without reboot

- **WHEN** one bitstream overlay is already configured in the PL (e.g., a DPU overlay)
- **AND** `FpgaManager::load_bitstream` is called with a different validated bitstream (e.g., a debug overlay)
- **THEN** the PL SHALL be fully reconfigured with the new bitstream
- **AND** the call SHALL return `Ok(())` once the PMU reports successful configuration
- **AND** no reboot SHALL be required

#### Scenario: Recovery from a corrupted PL load without power cycle

- **WHEN** a prior PL configuration attempt failed and left the PL unconfigured
- **AND** `FpgaManager::load_bitstream` is called again with a valid bitstream
- **THEN** the reload SHALL succeed without a power cycle

### Requirement: Bitstream Image Format Validation

`FpgaManager` SHALL parse Xilinx `.bit` image headers, exposing the header metadata (target device, design name, build date), and SHALL validate the header's target device against the running SOM before any data is handed to the PMU. Raw `.bin` images (configuration data with no header) SHALL be accepted only when accompanied by an out-of-band SmallAIOS manifest carrying target-device and signature metadata. After validation, only the raw configuration data SHALL be handed off to the PMU.

#### Scenario: Valid `.bit` for the running SOM is accepted

- **WHEN** `load_bitstream` is called with a Xilinx `.bit` image whose header target device matches the running SOM
- **THEN** header parsing SHALL yield the target device, design name, and build date
- **AND** the `.bit` header SHALL be stripped so that only raw configuration data is handed to the PMU

#### Scenario: `.bit` targeting a different device is rejected before PMU handoff

- **WHEN** `load_bitstream` is called with a `.bit` image whose header target device does not match the running SOM
- **THEN** the call SHALL return an error identifying the target-device mismatch
- **AND** no data SHALL be handed to the PMU

#### Scenario: Raw `.bin` without a manifest is rejected

- **WHEN** `load_bitstream` is called with a headerless raw `.bin` image and no accompanying SmallAIOS manifest
- **THEN** the call SHALL return an error
- **AND** no data SHALL be handed to the PMU

#### Scenario: Raw `.bin` with a valid SmallAIOS manifest is accepted

- **WHEN** `load_bitstream` is called with a raw `.bin` image accompanied by a SmallAIOS manifest whose target-device metadata matches the running SOM
- **THEN** validation SHALL use the manifest's target-device and signature metadata
- **AND** the raw configuration data SHALL be handed to the PMU after validation succeeds

### Requirement: IRQ-Driven Load Completion and Configuration Error Surfacing

`FpgaManager` load operations SHALL use IRQ-driven completion: after submitting a configuration request, the caller SHALL wait for the PMU response rather than busy-polling, resuming when the PMU's IPI response interrupt fires. Configuration errors reported by the PMU SHALL be surfaced to the caller as distinct error variants covering at least: CRC mismatch, unsupported format, and partial-reconfig-not-allowed.

#### Scenario: Completion is signaled by the PMU response IRQ

- **WHEN** `load_bitstream` has handed configuration data to the PMU
- **THEN** the calling task SHALL wait for the PMU response
- **AND** SHALL resume when the IPI response interrupt for the PMU channel fires
- **AND** SHALL NOT busy-poll the IPI registers while waiting

#### Scenario: CRC mismatch is surfaced

- **WHEN** the PMU response reports a CRC mismatch during PL configuration
- **THEN** the load call SHALL return an error variant identifying the CRC mismatch

#### Scenario: Unsupported format is surfaced

- **WHEN** the PMU response reports that the configuration data format is unsupported
- **THEN** the load call SHALL return an error variant identifying the unsupported format

#### Scenario: Partial-reconfig-not-allowed is surfaced

- **WHEN** the PMU response reports that partial reconfiguration is not allowed
- **THEN** the load call SHALL return an error variant identifying partial-reconfig-not-allowed

### Requirement: Partial Reconfiguration Behind a Dedicated Feature

`FpgaManager` SHALL expose `load_partial(&[u8], region) -> Result<()>` for reconfiguring only a region of the PL, gated behind its own non-default Cargo feature. Default builds SHALL NOT enable this feature and SHALL NOT contain the partial-reconfiguration code path.

#### Scenario: Partial-reconfig feature off in default builds

- **WHEN** the workspace is built with default features
- **THEN** the partial-reconfiguration feature SHALL be off
- **AND** `load_partial` SHALL NOT be present in the compiled binary
- **AND** the default build SHALL succeed

#### Scenario: load_partial reconfigures only the named region

- **WHEN** the partial-reconfiguration feature is enabled
- **AND** `load_partial` is called with a validated partial bitstream and a target PL region
- **THEN** only the specified region of the PL SHALL be reconfigured via the PMU
- **AND** the call SHALL return `Ok(())` once the PMU reports successful configuration

### Requirement: Verified-Boot Bitstream Signature Hook

When the existing `verified-boot` feature is enabled, `FpgaManager` SHALL verify an ML-DSA-65 signature over the bitstream, using the project's PQC stack in the `security` crate, before any configuration data is handed off to the PMU. When the feature is disabled, no signature-verification code SHALL be linked into the load path.

#### Scenario: Valid ML-DSA-65 signature allows the load

- **WHEN** `verified-boot` is enabled
- **AND** `load_bitstream` is called with a bitstream carrying a valid ML-DSA-65 signature
- **THEN** signature verification SHALL succeed
- **AND** the load SHALL proceed to the PMU handoff

#### Scenario: Invalid or missing signature blocks the PMU handoff

- **WHEN** `verified-boot` is enabled
- **AND** `load_bitstream` is called with a bitstream whose ML-DSA-65 signature is invalid or absent
- **THEN** the call SHALL return an error
- **AND** no configuration data SHALL be handed to the PMU

#### Scenario: verified-boot off links no signature check

- **WHEN** the workspace is built without the `verified-boot` feature
- **THEN** the bitstream signature-verification hook SHALL NOT be compiled into the load path
- **AND** the default build SHALL succeed

### Requirement: Capability-Gated Privileged Loading

Runtime PL reconfiguration SHALL be a privileged operation gated behind the existing capability system. Arbitrary user processes SHALL NOT be able to load bitstreams.

#### Scenario: Caller without the required capability is denied

- **WHEN** a process lacking the required capability attempts an `FpgaManager` load operation
- **THEN** the operation SHALL be denied with an error
- **AND** no PMU IPI request SHALL be issued

#### Scenario: Caller holding the required capability may load

- **WHEN** a caller holding the required capability invokes `load_bitstream` with a valid bitstream
- **THEN** the load SHALL proceed through validation and PMU handoff

### Requirement: Static-vs-Dynamic Loading Documentation

The change SHALL deliver documentation describing when to use static bitstream loading (preloaded by FSBL, baked into `BOOT.BIN`) versus dynamic loading via `FpgaManager`, and the security implications of runtime PL reconfiguration.

#### Scenario: Documentation covers loading modes and security

- **WHEN** a reviewer reads the documentation delivered by this change
- **THEN** it SHALL describe when to use static (FSBL) loading versus dynamic (`FpgaManager`) loading
- **AND** it SHALL describe the security implications of runtime PL reconfiguration, including the capability gating and the `verified-boot` signature hook

### Requirement: Runtime Bitstream Swap Demo Recipes

The change SHALL add `just` recipes demonstrating a runtime bitstream swap via `FpgaManager`.

#### Scenario: Demo recipe swaps bitstreams at runtime

- **WHEN** the runtime bitstream swap demo recipe is run via `just`
- **THEN** it SHALL load one bitstream and then swap to another using `FpgaManager::load_bitstream`
- **AND** the recipe SHALL appear in `just --list`
