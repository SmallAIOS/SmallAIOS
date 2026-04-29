## ADDED Requirements

### Requirement: Einsum Operator
The ONNX runtime SHALL implement the Einsum operator with support for the equation patterns commonly used in transformer models.

#### Scenario: Batched matmul via einsum
- **WHEN** `op_einsum` is called with equation `"bij,bjk->bik"` and two 3D input tensors
- **THEN** it MUST compute the batched matrix multiply

#### Scenario: Attention QK^T via einsum
- **WHEN** `op_einsum` is called with `"bhij,bhkj->bhik"` for attention scores
- **THEN** it MUST compute the QK^T product across batch and head dimensions

### Requirement: Split Operator
The ONNX runtime SHALL implement the Split operator that divides a tensor along a specified axis.

#### Scenario: Equal split
- **WHEN** `op_split` is called with axis=1 and num_outputs=3 on a tensor with shape [2, 6]
- **THEN** it MUST return three tensors of shape [2, 2]

#### Scenario: Custom split sizes
- **WHEN** `op_split` is called with `split=[2, 3, 1]` along axis=0
- **THEN** it MUST return three tensors of the specified sizes

### Requirement: Expand Operator
The ONNX runtime SHALL implement the Expand operator that broadcasts a tensor to a target shape.

#### Scenario: Broadcast scalar to tensor
- **WHEN** `op_expand` is called with a scalar input and target shape [2, 3]
- **THEN** it MUST return a [2, 3] tensor with the scalar value

### Requirement: Tile Operator
The ONNX runtime SHALL implement the Tile operator that repeats a tensor.

#### Scenario: Tile along axes
- **WHEN** `op_tile` is called with input shape [2, 3] and repeats [2, 1]
- **THEN** it MUST return a tensor with shape [4, 3] containing two copies of the input

### Requirement: OneHot Operator
The ONNX runtime SHALL implement the OneHot operator for classification.

#### Scenario: One-hot encoding
- **WHEN** `op_one_hot` is called with indices [0, 1, 2], depth=3, on_value=1.0, off_value=0.0
- **THEN** it MUST return a 3x3 identity-like matrix
