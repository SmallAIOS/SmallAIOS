## ADDED Requirements

### Requirement: OpenAI Chat Completions Endpoint
The container SHALL expose a `POST /v1/chat/completions` endpoint that accepts the OpenAI Chat Completions request schema and returns the OpenAI response schema.

#### Scenario: Non-streaming chat completion
- **WHEN** a POST request is sent to `/v1/chat/completions` with `stream: false`
- **AND** the request contains `model`, `messages[]`, and `max_tokens`
- **THEN** the response MUST have status 200 with `Content-Type: application/json`
- **AND** the body MUST contain `choices[0].message.content` with the generated text
- **AND** the body MUST contain `choices[0].finish_reason` as `"stop"` or `"length"`
- **AND** the body MUST contain `usage` with `prompt_tokens`, `completion_tokens`, `total_tokens`

#### Scenario: Streaming chat completion
- **WHEN** a POST request is sent with `stream: true`
- **THEN** the response MUST use `Content-Type: text/event-stream`
- **AND** each token MUST be emitted as an SSE `data:` line with a delta object
- **AND** the stream MUST end with `data: [DONE]`

#### Scenario: Model not found
- **WHEN** the `model` field does not match any loaded model
- **THEN** the response MUST have status 404 with an error message

#### Scenario: Sampling parameters applied
- **WHEN** `temperature`, `top_p`, or `stop` parameters are provided
- **THEN** the generation MUST apply the specified sampling strategy
