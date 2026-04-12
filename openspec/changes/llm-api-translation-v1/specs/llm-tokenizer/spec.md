## ADDED Requirements

### Requirement: BPE Tokenizer
The container SHALL implement a Byte-Pair Encoding tokenizer that loads HuggingFace `tokenizer.json` files for text-to-token and token-to-text conversion.

#### Scenario: Encode text to tokens
- **WHEN** `tokenizer.encode("Hello world")` is called
- **THEN** it MUST return a Vec of token IDs matching the HuggingFace reference tokenizer output for the same vocabulary

#### Scenario: Decode tokens to text
- **WHEN** `tokenizer.decode(&[token_ids])` is called
- **THEN** it MUST return the UTF-8 text string corresponding to those token IDs

#### Scenario: Special tokens
- **WHEN** the tokenizer encounters special tokens (BOS, EOS, PAD, model-specific tokens)
- **THEN** they MUST be handled according to the `added_tokens` section of `tokenizer.json`

#### Scenario: Load from tokenizer.json
- **WHEN** `Tokenizer::from_file("tokenizer.json")` is called
- **THEN** it MUST parse the vocab, merges, and added_tokens from the HuggingFace format
- **AND** return a ready-to-use tokenizer instance

### Requirement: Prompt Templates
The container SHALL apply model-specific prompt templates to convert chat message arrays into token sequences.

#### Scenario: Llama 3 prompt format
- **WHEN** a chat request targets a Llama model
- **THEN** the prompt MUST use Llama 3's `<|begin_of_text|>...<|eot_id|>` format

#### Scenario: Gemma prompt format
- **WHEN** a chat request targets a Gemma model
- **THEN** the prompt MUST use Gemma's `<start_of_turn>...<end_of_turn>` format

#### Scenario: Custom prompt template
- **WHEN** a `prompt_template.json` file exists alongside the model
- **THEN** it MUST override the built-in template for that model
