## ADDED Requirements

### Requirement: Cognitive Complexity Threshold
All public and private functions in the workspace SHALL have a cognitive complexity score below 15.

#### Scenario: Refactored dispatch_node passes threshold
- **WHEN** SonarCloud analyzes `onnx-rt/src/executor.rs::dispatch_node()`
- **THEN** the cognitive complexity MUST be below 15
- **AND** behavior MUST be unchanged from the previous implementation

#### Scenario: Refactored op_cast passes threshold
- **WHEN** SonarCloud analyzes `onnx-rt/src/operators.rs::op_cast()`
- **THEN** the cognitive complexity MUST be below 15

### Requirement: Tensor Byte I/O Helpers
The ONNX runtime SHALL provide reusable helpers for converting between bytes and numeric types.

#### Scenario: Read f32 from byte buffer
- **WHEN** code needs to extract an f32 value from a tensor's raw_data
- **THEN** it MUST use the shared `read_f32(data, idx)` helper
- **AND** MUST NOT inline `f32::from_le_bytes([data[i*4], ...])`

#### Scenario: Write f32 to byte buffer
- **WHEN** code needs to write an f32 value into a tensor's raw_data
- **THEN** it MUST use the shared `write_f32(data, idx, val)` helper

#### Scenario: Allocate tensor data buffer
- **WHEN** code needs to allocate a tensor data buffer
- **THEN** it MUST use `allocate_tensor_data(elements, dtype)` instead of `vec![0u8; elements * 4]`

### Requirement: Parameter Count Threshold
Functions in the workspace SHALL accept no more than 7 parameters; functions exceeding this MUST use a parameter struct.

#### Scenario: conv_compute uses ConvParams struct
- **WHEN** `conv_compute()` is called
- **THEN** it MUST accept a `ConvParams` struct rather than 9 individual parameters

### Requirement: Named Constants for Magic Values
Numeric literals used in mathematical computations SHALL be defined as named constants.

#### Scenario: Polynomial coefficients are named
- **WHEN** `expf_approx()` uses polynomial approximation
- **THEN** the coefficients MUST be defined as named constants
- **AND** the clamp limits MUST be named constants

### Requirement: No Duplicated Parsing Logic
Functions that parse the same data format SHALL share a common parsing primitive.

#### Scenario: DHCP option parsing
- **WHEN** `parse_options()` and `get_option_value()` parse the DHCP options field
- **THEN** they MUST share a common iterator helper rather than duplicating the parse loop
