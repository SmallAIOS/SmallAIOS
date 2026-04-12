## MODIFIED Requirements

### Requirement: Operator Dispatch Path
The executor's `dispatch_node` function SHALL check for an available GPU backend before falling through to CPU execution. When a GPU backend is present and the operator is in its supported set, execution SHALL occur on the GPU. When no GPU backend is present or the operator is not supported, execution SHALL fall through to the existing CPU implementation with zero behavioral change.

#### Scenario: Mixed GPU/CPU graph execution
- **WHEN** a model graph contains both GPU-supported and unsupported operators
- **THEN** the executor MUST interleave GPU and CPU dispatch within the same graph traversal
- **AND** tensor values MUST be transferred between host and device as needed
- **AND** the final graph outputs MUST be identical (within floating-point tolerance) to a pure-CPU execution of the same graph
