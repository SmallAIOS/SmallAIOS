# gemma-architecture Specification

## Purpose
TBD - created by archiving change safetensors-model-loader-v1. Update Purpose after archive.
## Requirements
### Requirement: Gemma transformer layer template
The runtime SHALL provide a function that constructs the Gemma transformer layer pattern as an `ExecutionGraph` segment from a `GemmaConfig` and weight bindings.

#### Scenario: Build a single Gemma layer
- **WHEN** the graph builder is invoked with a layer index and Gemma config
- **THEN** it SHALL emit operators in the order: input RMSNorm → Q/K/V projection (3 separate MatMul) → RoPE on Q and K → attention → output projection → residual add → post-attention RMSNorm → MLP gate/up projections → SwiGLU activation → MLP down projection → final residual add
- **AND** all weight tensor names SHALL match the HuggingFace `model.layers.{idx}.{module}.{tensor}` naming convention

#### Scenario: Build full Gemma model graph
- **WHEN** the loader is invoked with a Gemma model directory
- **THEN** it SHALL build `num_hidden_layers` Gemma layers
- **AND** wire them in sequence
- **AND** prepend an embedding lookup (`model.embed_tokens.weight`)
- **AND** append a final RMSNorm and a language-modeling head (`lm_head.weight` MatMul to vocab logits)

### Requirement: Sliding-window attention pattern
The Gemma layer template SHALL alternate between sliding-window local attention and global attention according to the Gemma 4 sliding window pattern.

#### Scenario: Local attention layer
- **WHEN** a layer index is NOT a global-attention layer per the architecture pattern
- **THEN** the attention node SHALL use a sliding-window mask of size `config.sliding_window` (1024 for Gemma 4)
- **AND** only attend to keys within the window of the current query position

#### Scenario: Global attention layer
- **WHEN** a layer index IS a global-attention layer (final layer + every Nth layer per pattern)
- **THEN** the attention node SHALL use a full causal mask
- **AND** attend to all keys up to the current query position

### Requirement: Proportional RoPE (p-RoPE)
The Gemma layer template SHALL apply the proportional RoPE variant when configured for it.

#### Scenario: p-RoPE on query and key
- **WHEN** Gemma 4 architecture is detected
- **THEN** the `RotaryEmbedding` node SHALL receive a `p_rope: true` attribute
- **AND** the rope theta value from the config

### Requirement: Grouped-query attention (GQA)
The Gemma layer template SHALL configure GQA when the config indicates fewer key/value heads than query heads.

#### Scenario: GQA configuration
- **WHEN** `config.num_key_value_heads < config.num_attention_heads`
- **THEN** the attention node SHALL be configured with the correct query-to-kv head ratio
- **AND** the K and V projection weights SHALL have shape `[num_kv_heads * head_dim, hidden_size]` not `[num_heads * head_dim, hidden_size]`

### Requirement: SwiGLU MLP
The Gemma layer template SHALL implement the gated SwiGLU MLP pattern.

#### Scenario: SwiGLU forward pass
- **WHEN** the MLP block is constructed
- **THEN** it SHALL emit: gate = MatMul(input, gate_proj.weight); up = MatMul(input, up_proj.weight); intermediate = SiLU(gate) * up; output = MatMul(intermediate, down_proj.weight)
- **AND** the SiLU and element-wise multiply MAY be fused into a single operator if a `SiLUMul` op is available

### Requirement: RMSNorm with Gemma-specific epsilon and weight format
The Gemma layer template SHALL use RMSNorm with the model-specific epsilon and Gemma's `1 + weight` convention if applicable.

#### Scenario: RMSNorm epsilon from config
- **WHEN** an RMSNorm node is constructed
- **THEN** its epsilon attribute SHALL come from `config.rms_norm_eps`

#### Scenario: Gemma weight offset (1 + w)
- **WHEN** the model is Gemma (not Llama)
- **THEN** the RMSNorm op SHALL apply the Gemma convention of `output = x * rsqrt(mean(x^2) + eps) * (1 + weight)` where the `+ 1` is part of the operator
- **AND** weights stored in safetensors are the raw values without the +1 offset

