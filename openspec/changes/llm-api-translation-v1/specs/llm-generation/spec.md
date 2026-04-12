## ADDED Requirements

### Requirement: Autoregressive Token Generation
The container SHALL implement an autoregressive generation loop that repeatedly invokes the ONNX model to produce tokens until a stop condition is met.

#### Scenario: Generation with max_tokens limit
- **WHEN** generation is invoked with `max_tokens: 64`
- **THEN** the loop MUST produce at most 64 new tokens
- **AND** finish_reason MUST be `MaxTokens` if the limit is reached

#### Scenario: Generation stops on EOS token
- **WHEN** the model outputs the EOS token ID
- **THEN** the loop MUST stop immediately
- **AND** finish_reason MUST be `Stop`

#### Scenario: Generation stops on stop sequence
- **WHEN** a stop sequence (e.g., `"\n\nHuman:"`) appears in the decoded output
- **THEN** the loop MUST stop and truncate the output before the stop sequence
- **AND** finish_reason MUST be `StopSequence`

### Requirement: Sampling Strategies
The generation loop SHALL support temperature, top-k, and top-p sampling.

#### Scenario: Temperature scaling
- **WHEN** `temperature: 0.0` is specified
- **THEN** the sampler MUST select the argmax token (greedy decoding)

#### Scenario: Top-p (nucleus) sampling
- **WHEN** `top_p: 0.9` is specified
- **THEN** the sampler MUST restrict the candidate set to the smallest set of tokens whose cumulative probability exceeds 0.9

#### Scenario: Top-k sampling
- **WHEN** `top_k: 50` is specified
- **THEN** the sampler MUST restrict the candidate set to the top 50 tokens by probability

### Requirement: SSE Streaming
The generation loop SHALL support emitting each token as an SSE event during generation, enabling real-time streaming responses.

#### Scenario: Token emitted during generation
- **WHEN** streaming is enabled and a new token is sampled
- **THEN** the token MUST be immediately emitted as an SSE data line
- **AND** the response MUST use chunked transfer encoding
