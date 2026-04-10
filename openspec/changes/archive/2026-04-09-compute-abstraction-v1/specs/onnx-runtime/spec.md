## MODIFIED Requirements

### Requirement: CUDA Execution Provider
The runtime SHALL support GPU inference via a CUDA execution provider that launches custom PTX kernels on NVIDIA GPUs.

#### Scenario: GPU kernel launch for MatMul
- **WHEN** an inference session is configured with the CUDA execution provider
- **AND** the graph contains a MatMul operator
- **THEN** the runtime MUST launch a GPU kernel using tensor core HMMA instructions for fp16/bf16
- **AND** MUST use the GPU memory pool for intermediate tensor allocation

#### Scenario: Async DMA transfers
- **WHEN** input tensors reside in host memory and the session uses GPU execution
- **THEN** the runtime MUST transfer inputs to GPU via async DMA
- **AND** MUST overlap DMA transfers with computation where possible

#### Scenario: GPU operator dispatch with CPU fallback
- **WHEN** the executor encounters an operator during graph traversal
- **THEN** it MUST query the active `ComputeProvider` via `supports_op()`
- **AND** if the GPU backend supports the operator, MUST dispatch to GPU
- **AND** if the GPU backend does not support the operator, MUST fall back to CPU execution
- **AND** MUST handle host↔device data transfers at CPU/GPU transition boundaries
