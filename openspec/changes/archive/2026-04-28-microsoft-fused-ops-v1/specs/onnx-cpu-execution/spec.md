## ADDED Requirements

### Requirement: Domain-Aware OpKind
The ONNX runtime SHALL distinguish operator provenance by domain, so that operators from the `com.microsoft` contrib op set are tracked separately from standard ONNX operators and dispatched through a separate handler table.

#### Scenario: Standard and Microsoft ops with the same name resolve differently
- **WHEN** the graph builder encounters a node with `op_type = "MultiHeadAttention"` and `domain = "com.microsoft"`
- **THEN** the registry MUST resolve the node to the `OpKind::MultiHeadAttention` variant tagged `Domain::MicrosoftFused`
- **AND** the dispatcher MUST route the node to `ops::microsoft::op_multi_head_attention`
- **AND** a future node with the same `op_type` but `domain = ""` or `domain = "ai.onnx"` MUST resolve to a different `OpKind` value (or fail with `UnsupportedOperator` if no standard-domain op of that name exists)

#### Scenario: Operator registry reports domain per entry
- **WHEN** a contributor queries `OperatorRegistry` for the status of `"GroupQueryAttention"` in the `com.microsoft` domain
- **THEN** the registry MUST return `OperatorStatus::Implemented` with `Domain::MicrosoftFused`
- **AND** the same query in the standard domain MUST return `UnsupportedOperator`

### Requirement: SimplifiedLayerNormalization Operator
The ONNX runtime SHALL implement the `com.microsoft.SimplifiedLayerNormalization` operator, mathematically equivalent to RMS normalization over the specified axis, with optional bias and configurable epsilon.

#### Scenario: Matches RMSNorm on a last-axis input
- **WHEN** `op_simplified_layer_normalization` is called with an input of shape `(2, 4, 8)`, a scale vector of length 8, `axis = -1`, and `epsilon = 1e-5`
- **THEN** the output MUST be element-wise equal (within 1e-6 absolute error) to the result of the existing `op_rms_normalization` with the same arguments

#### Scenario: Optional bias is added after normalization
- **WHEN** the operator is called with three inputs (`x`, `scale`, `bias`)
- **THEN** the output MUST equal `(x / rms(x)) * scale + bias`
- **AND** the bias MUST be broadcast along the non-normalized axes

### Requirement: SkipSimplifiedLayerNormalization Operator
The ONNX runtime SHALL implement the `com.microsoft.SkipSimplifiedLayerNormalization` operator, which fuses a residual-stream element-wise add with RMS normalization, producing both the normalized output and the pre-normalization sum.

#### Scenario: Fused residual add and normalization
- **WHEN** the operator is called with `input`, `skip`, `scale`, and `bias` tensors
- **THEN** the first output MUST equal `rms_normalize(input + skip + bias) * scale`
- **AND** the second output MUST equal `input + skip + bias` (the pre-normalization sum)

#### Scenario: Bias is optional
- **WHEN** the operator is called with only `input`, `skip`, and `scale` (no bias)
- **THEN** the sum MUST be `input + skip`
- **AND** the first output MUST equal `rms_normalize(input + skip) * scale`

### Requirement: GroupQueryAttention Operator
The ONNX runtime SHALL implement the `com.microsoft.GroupQueryAttention` operator, which fuses rotary positional embedding, past-KV-cache concatenation, grouped-query attention dispatch, scaled dot-product attention, and causal masking into a single op.

#### Scenario: Matches Python ORT reference on a small hand-crafted model
- **WHEN** `op_group_query_attention` is called with `num_heads = 4`, `kv_num_heads = 2`, `head_dim = 8`, a batch-1 query of shape `(1, 3, 32)`, empty past-KV tensors of shape `(1, 2, 0, 8)`, and cos/sin caches sized `(16, 4)`
- **THEN** the output MUST match a Python `onnxruntime.InferenceSession` reference within 1e-5 absolute error
- **AND** the returned `present_key` and `present_value` MUST have shape `(1, 2, 3, 8)` and contain the post-RoPE key and value projections

#### Scenario: KV-cache is mutated in place across two calls
- **WHEN** the operator is called twice against the same `past_key` and `past_value` buffers, first with `Sq = 3` and then with `Sq = 1`
- **THEN** the second call MUST see the first call's key/value writes in the past cache
- **AND** the second call's output attention MUST attend over all four total key positions (3 from call 1 + 1 from call 2)

#### Scenario: do_rotary = 0 skips the internal RoPE step
- **WHEN** the operator is called with attribute `do_rotary = 0` and a query tensor that is already rotary-embedded
- **THEN** the internal `apply_rope_in_place` MUST NOT be called
- **AND** the query passed into the SDPA helper MUST be bit-identical to the input query

#### Scenario: Causal mask is applied on the fly
- **WHEN** the operator is called with `Sq = 4` and `past_Sk = 0`
- **THEN** position `(q=0, k=1)`, `(q=0, k=2)`, `(q=0, k=3)`, `(q=1, k=2)`, `(q=1, k=3)`, `(q=2, k=3)` MUST all receive `-inf` masking in the attention scores
- **AND** the softmax output at those positions MUST be exactly zero

### Requirement: MultiHeadAttention Operator
The ONNX runtime SHALL implement the `com.microsoft.MultiHeadAttention` operator as a non-grouped attention fusion that reuses the same scaled-dot-product-attention helper as `GroupQueryAttention`.

#### Scenario: Unpacked Q, K, V inputs with past KV-cache
- **WHEN** `op_multi_head_attention` is called with three separate Q, K, V inputs of shape `(1, 4, 64)` with `num_heads = 8`, plus `past_key` and `past_value` tensors of shape `(1, 8, 2, 8)`
- **THEN** the output MUST match a Python `onnxruntime.InferenceSession` reference within 1e-5 absolute error
- **AND** the `present_key` / `present_value` MUST have shape `(1, 8, 6, 8)` (2 past + 4 new)

#### Scenario: No internal RoPE (DeepSeek style)
- **WHEN** the operator is called on inputs that were already rotary-embedded upstream by a standalone `RotaryEmbedding` node
- **THEN** the operator MUST NOT apply RoPE internally (it has no cos/sin cache inputs)
- **AND** the output MUST equal the SDPA helper's result on the unrotated call's rotated inputs

### Requirement: RotaryEmbedding Operator
The ONNX runtime SHALL implement the `com.microsoft.RotaryEmbedding` operator in both interleaved and non-interleaved variants, sharing the `apply_rope_in_place` helper with `GroupQueryAttention`.

#### Scenario: Non-interleaved rotation matches HuggingFace layout
- **WHEN** `op_rotary_embedding` is called with `interleaved = 0` on an input of shape `(1, 2, 64)` (one head, two positions, head_dim=64), with `position_ids = [0, 1]` and matching cos/sin caches
- **THEN** the output at position 0 MUST equal the input (rotation by angle 0)
- **AND** the output at position 1 MUST equal the non-interleaved rotation of the input (elements `[0..32]` paired with `[32..64]`)

#### Scenario: Interleaved rotation matches DeepSeek layout
- **WHEN** the same operator is called with `interleaved = 1`
- **THEN** the output at position 1 MUST equal the interleaved rotation (adjacent element pairs `[0,1]`, `[2,3]`, ... each rotated)
- **AND** the output MUST NOT equal the non-interleaved result (confirming the two variants are distinct)

### Requirement: Canonical LLM Model Loading
The ONNX runtime SHALL load the canonical HuggingFace ONNX exports of `meta-llama/Llama-3.2-1B`, `google/gemma-3-1b-it`, and `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` end-to-end via `Session::new_from_file()` and produce numerically-correct output for at least one generation step.

#### Scenario: Llama-3.2-1B single-token generation matches Python ORT
- **WHEN** the canonical HF export of `Llama-3.2-1B` (ONNX format, int4-weight / int8-activation quantization) is loaded via `Session::new_from_file()` and a single-token generation is run
- **THEN** the session MUST load without `UnsupportedOperator` errors
- **AND** the top-1 output token MUST match a Python `onnxruntime` reference on the same input
- **AND** the top-5 output logits MUST match the reference within 1e-2 absolute error

#### Scenario: Gemma 3 1b single-token generation matches Python ORT
- **WHEN** the canonical HF export of `google/gemma-3-1b-it` is loaded and a single-token generation is run
- **THEN** the session MUST load without errors
- **AND** the top-1 output token MUST match a Python `onnxruntime` reference
- **AND** the top-5 output logits MUST match the reference within 1e-2 absolute error

#### Scenario: DeepSeek-R1-Distill-Qwen-1.5B forward pass matches Python ORT
- **WHEN** the canonical HF export of `DeepSeek-R1-Distill-Qwen-1.5B` (ONNX, f32) is loaded and a single forward pass is run
- **THEN** the session MUST load without errors
- **AND** the output hidden states MUST match a Python `onnxruntime` reference within 1e-3 absolute error
