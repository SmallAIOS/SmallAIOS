## ADDED Requirements

### Requirement: Parse HuggingFace config.json
The runtime SHALL parse the HuggingFace `config.json` file format and extract transformer architecture metadata into a typed Rust struct.

#### Scenario: Parse Gemma config
- **WHEN** a model directory contains `config.json` with `"model_type": "gemma3"` (or `gemma4`)
- **THEN** the loader SHALL parse the JSON and produce a `GemmaConfig` struct
- **AND** extract: `num_hidden_layers`, `hidden_size`, `intermediate_size`, `num_attention_heads`, `num_key_value_heads`, `vocab_size`, `max_position_embeddings`, `rope_theta`, `sliding_window`, `rms_norm_eps`

#### Scenario: Parse generation config
- **WHEN** a model directory contains `generation_config.json`
- **THEN** the loader SHALL parse default sampling parameters
- **AND** extract: `temperature`, `top_p`, `top_k`, `bos_token_id`, `eos_token_id`, `pad_token_id`

### Requirement: Architecture detection
The loader SHALL identify which model architecture is in a directory based on `config.json` metadata.

#### Scenario: Gemma detection
- **WHEN** `config.json` has `"architectures": ["Gemma3ForCausalLM"]` or `"Gemma4ForCausalLM"`
- **THEN** the loader SHALL recognize this as the Gemma family
- **AND** route loading to the Gemma graph builder

#### Scenario: Unsupported architecture error
- **WHEN** `config.json` has an architecture not yet supported (e.g. `LlamaForCausalLM` before Llama support is added)
- **THEN** the loader SHALL return an error identifying the architecture name
- **AND** list which architectures ARE supported

### Requirement: Config validation
The loader SHALL validate that required fields are present and have sensible values.

#### Scenario: Missing required field
- **WHEN** `config.json` is missing a required field for the detected architecture (e.g. Gemma without `num_hidden_layers`)
- **THEN** the loader SHALL return an error naming the missing field

#### Scenario: Invalid field value
- **WHEN** a config field has an invalid value (e.g. `num_hidden_layers: 0` or `vocab_size: -1`)
- **THEN** the loader SHALL return a validation error
