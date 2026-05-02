## ADDED Requirements

### Requirement: Tier 2 Operator Coverage
The ONNX runtime SHALL implement the second tier of operators required for transformer and quantized model execution.

#### Scenario: Math primitives
- **WHEN** the executor encounters Pow, Sqrt, Exp, Log, Erf, Neg, Abs, Floor, Ceil, or Round operators
- **THEN** the operator MUST execute correctly with element-wise semantics

#### Scenario: Comparison and selection
- **WHEN** the executor encounters Equal, Less, Greater, LessOrEqual, GreaterOrEqual, NotEqual, Where, Min, Max, or Not
- **THEN** the operator MUST produce the correct boolean or selected output with broadcasting

#### Scenario: Composite activations
- **WHEN** the executor encounters Gelu, LeakyRelu, Elu, or Swish
- **THEN** the operator MUST compute the activation per its mathematical definition

### Requirement: Operator Registry Update
The OperatorRegistry SHALL include all new Tier 2 operator names.

#### Scenario: Tier 2 operators are registered
- **WHEN** a model contains an LSTM, GRU, Gelu, QuantizeLinear, or any other Tier 2 op
- **THEN** the registry MUST recognize it as supported
- **AND** the validator MUST allow the model to load
- **AND** the executor MUST dispatch to the correct implementation
