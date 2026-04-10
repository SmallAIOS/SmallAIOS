## MODIFIED Requirements

### Requirement: Tier 1 CPU Operator Completeness
The ONNX runtime SHALL implement all 29 Tier 1 operators for CPU execution with f32 tensors, with parallel variants for compute-heavy operators.

#### Scenario: Element-wise binary operators (Sub, Mul, Div)
- **WHEN** two f32 tensors are provided as inputs to Sub, Mul, or Div
- **THEN** the operator MUST compute the element-wise result with NumPy-style broadcasting
- **AND** the output shape MUST match the broadcast output shape
- **AND** if num_elements exceeds the parallel threshold, computation MUST be distributed across available cores

#### Scenario: Gemm operator wraps GEMM micro-kernel
- **WHEN** a Gemm node is dispatched with matrices A, B, and optional bias C
- **THEN** the operator MUST compute `alpha * A @ B + beta * C` using the existing `gemm_f32` micro-kernel
- **AND** MUST support `transA` and `transB` attributes
- **AND** if M × K × N exceeds the parallel threshold, tile rows MUST be distributed across available cores

#### Scenario: Activation operators (Sigmoid, Tanh)
- **WHEN** an f32 tensor is provided to Sigmoid or Tanh
- **THEN** the operator MUST compute the element-wise activation function
- **AND** if num_elements exceeds the parallel threshold, computation MUST be distributed across available cores
