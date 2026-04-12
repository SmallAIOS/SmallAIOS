## ADDED Requirements

### Requirement: Anthropic Messages Endpoint
The container SHALL expose a `POST /v1/messages` endpoint that accepts the Anthropic Messages request schema and returns the Anthropic response schema.

#### Scenario: Non-streaming message
- **WHEN** a POST request is sent to `/v1/messages` with `stream: false`
- **AND** the request contains `model`, `messages[]`, `max_tokens`, and optionally `system`
- **THEN** the response MUST have status 200 with `Content-Type: application/json`
- **AND** the body MUST contain `content[0].type` as `"text"` and `content[0].text` with the generated text
- **AND** the body MUST contain `stop_reason` as `"end_turn"` or `"max_tokens"`
- **AND** the body MUST contain `usage` with `input_tokens` and `output_tokens`

#### Scenario: Streaming message
- **WHEN** a POST request is sent with `stream: true`
- **THEN** the response MUST use `Content-Type: text/event-stream`
- **AND** text deltas MUST be emitted as `event: content_block_delta` SSE events
- **AND** the stream MUST end with `event: message_stop`

#### Scenario: System prompt from top-level field
- **WHEN** the request includes a `system` field (string)
- **THEN** the system prompt MUST be applied to the generation context
- **AND** the system prompt MUST NOT appear in the `messages` array in the response

#### Scenario: Anthropic-to-OpenAI field mapping
- **WHEN** the Anthropic adapter receives a request
- **THEN** `stop_sequences` MUST map to internal `stop_sequences`
- **AND** `max_tokens` MUST map to internal `max_tokens`
- **AND** finish reason `"end_turn"` MUST map from internal `FinishReason::Stop`
