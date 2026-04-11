## ADDED Requirements

### Requirement: QuantizeLinear Operator
The ONNX runtime SHALL implement the QuantizeLinear operator that converts float32 tensors to int8/uint8.

#### Scenario: Quantize float to int8
- **WHEN** `op_quantize_linear` is called with input float tensor, scale=0.1, zero_point=0
- **THEN** it MUST return a tensor of i8 values where each element is `clip(round(input/scale) + zero_point, -128, 127)`

#### Scenario: Per-axis quantization
- **WHEN** the scale and zero_point are 1D tensors (per-channel quantization)
- **THEN** the operator MUST broadcast them along the appropriate axis

### Requirement: DequantizeLinear Operator
The ONNX runtime SHALL implement the DequantizeLinear operator that converts int8/uint8 back to float32.

#### Scenario: Dequantize int8 to float
- **WHEN** `op_dequantize_linear` is called with int8 input, scale=0.1, zero_point=0
- **THEN** it MUST return float32 values where each element is `(input - zero_point) * scale`

#### Scenario: Round-trip preservation
- **WHEN** values are quantized then dequantized
- **THEN** the result MUST be within `scale` of the original (within rounding precision)

### Requirement: QLinearMatMul Operator
The ONNX runtime SHALL implement quantized matrix multiplication.

#### Scenario: Quantized matmul produces correct output
- **WHEN** `op_qlinear_matmul` is called with quantized inputs A, B, their scales, zero_points, and output scale/zero_point
- **THEN** it MUST produce a quantized result that, when dequantized, matches the f32 matmul result within 1% tolerance

### Requirement: QLinearConv Operator
The ONNX runtime SHALL implement quantized convolution.

#### Scenario: Quantized convolution produces correct output
- **WHEN** `op_qlinear_conv` is called with quantized inputs and weights
- **THEN** it MUST produce a quantized result equivalent (after dequantization) to f32 convolution within 1% tolerance
