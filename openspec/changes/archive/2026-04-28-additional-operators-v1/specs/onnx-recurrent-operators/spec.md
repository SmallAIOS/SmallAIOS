## ADDED Requirements

### Requirement: LSTM Operator
The ONNX runtime SHALL implement the LSTM (Long Short-Term Memory) operator per ONNX opset 14+.

#### Scenario: LSTM forward pass
- **WHEN** `op_lstm` is called with input X (sequence_length × batch × input_size), weights W and R, bias B, and initial states (h0, c0)
- **THEN** it MUST return the output sequence Y, final hidden state Y_h, and final cell state Y_c
- **AND** the forward direction MUST compute `(i, f, g, o) = sigmoid(...), tanh(...), sigmoid(...)` per timestep with peephole connections optional

#### Scenario: LSTM bidirectional
- **WHEN** `op_lstm` is called with `direction="bidirectional"`
- **THEN** it MUST run a forward and reverse pass and concatenate the outputs along the direction axis

### Requirement: GRU Operator
The ONNX runtime SHALL implement the GRU (Gated Recurrent Unit) operator.

#### Scenario: GRU forward pass
- **WHEN** `op_gru` is called with the standard W/R/B weights and initial hidden state
- **THEN** it MUST return the output sequence Y and final hidden state Y_h
- **AND** the gate computation MUST follow `r_t = sigmoid(W_r·x_t + R_r·h_{t-1} + b_r)`, etc.

### Requirement: RNN Operator
The ONNX runtime SHALL implement the basic RNN operator with tanh activation by default.

#### Scenario: RNN forward pass
- **WHEN** `op_rnn` is called with X, W, R, B, and initial hidden state
- **THEN** the per-timestep computation MUST be `h_t = tanh(W·x_t + R·h_{t-1} + b)`
