## ADDED Requirements

### Requirement: BF16 tensor data type support
The runtime SHALL support BF16 (bfloat16) as a first-class tensor data type with raw byte storage and conversion helpers.

#### Scenario: BF16 raw storage
- **WHEN** a tensor has `data_type == DataType::BFloat16`
- **THEN** its `raw_data` SHALL store 2 bytes per element in little-endian BF16 format
- **AND** `byte_size()` SHALL return `total_elements * 2`

#### Scenario: BF16 to f32 conversion helper
- **WHEN** `bf16_to_f32(bytes: &[u8])` is called with BF16 byte data
- **THEN** it SHALL produce a `Vec<f32>` with each element converted by zero-extending the BF16 mantissa to 23 bits
- **AND** the conversion SHALL be lossless from BF16's representable range

#### Scenario: f32 to BF16 conversion helper
- **WHEN** `f32_to_bf16(values: &[f32])` is called with f32 data
- **THEN** it SHALL produce a `Vec<u8>` of 2N bytes
- **AND** rounding SHALL use round-to-nearest-even on the truncated mantissa bits

### Requirement: BF16 in CPU operators used by Gemma
CPU implementations of operators required for Gemma inference SHALL accept BF16 input tensors and produce BF16 output tensors.

#### Scenario: RMSNorm with BF16 input
- **WHEN** `RMSNormalization` is dispatched with a BF16 input tensor
- **THEN** the operator SHALL convert to f32 internally for the variance computation
- **AND** produce a BF16 output tensor (convert back on write)

#### Scenario: Element-wise add/mul with BF16 inputs
- **WHEN** `Add` or `Mul` is dispatched with BF16 input tensors
- **THEN** the operator SHALL produce a BF16 output tensor

#### Scenario: f32-only operators reject BF16
- **WHEN** an operator without BF16 support receives a BF16 tensor
- **THEN** it SHALL return an error identifying the operator and the unsupported dtype
- **AND** SHALL NOT silently coerce or produce incorrect output
