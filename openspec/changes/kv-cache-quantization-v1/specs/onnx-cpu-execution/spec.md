## ADDED Requirements

### Requirement: Compressed KV Cache
The ONNX runtime SHALL provide a `CompressedKVCache` struct that owns a compressed representation of the per-head K and V caches used by autoregressive attention operators, exposes a stable block-wise read/write API for use by `GroupQueryAttention`, `MultiHeadAttention`, and `RotaryEmbedding` kernels, and hides the details of rotation, quantization, and low-rank factorization from its callers.

#### Scenario: GroupQueryAttention reads and writes through CompressedKVCache
- **WHEN** a `GroupQueryAttention` op is dispatched inside a `Loop` body and receives a `CompressedKVCache` handle from the outer value_map
- **THEN** the op MUST call `write_block(kv_head, start_row, k_new, v_new)` once for each new token row appended on this iteration
- **AND** the op MUST compute attention logits via `attention_logits(kv_head, query, num_rows)` or, at the caller's choice, via `read_block(...)` followed by an f16 inner product
- **AND** the op MUST compute the weighted value sum via `weighted_value_sum(kv_head, weights, num_rows)` or via `read_block(...)` followed by an f16 matmul
- **AND** the op MUST NOT access the cache's internal quantized, rotated, or low-rank representation directly

#### Scenario: Cache handle survives across Loop iterations
- **WHEN** a `CompressedKVCache` handle is placed in the outer value_map before a `Loop` node and referenced by name from inside the Loop body
- **THEN** the sub-graph executor MUST make the cache handle visible inside the body's value_map per iteration (via the existing outer-ref copy mechanism)
- **AND** writes performed by the body MUST persist on the cache between iterations
- **AND** writes performed by the body MUST remain visible in the outer scope after the Loop completes

### Requirement: PolarQuant and QJL Residual Pipeline
The runtime SHALL implement the TurboQuant two-stage encoder and decoder, consisting of a deterministic random rotation (Walsh–Hadamard transform with random sign flips), a per-channel scalar quantizer at a configurable bit width (3, 4, or 8 bits), and an optional 1-bit Quantized Johnson–Lindenstrauss residual correction, with bit-exact reproducibility across machines running the same model.

#### Scenario: Round-trip dequantization within the grid step
- **WHEN** a block of 64 f16 vectors is encoded with `kv_quant_bits = 4` and the QJL residual disabled
- **THEN** decoding each vector MUST produce a reconstruction whose per-element error is bounded by one half of the per-channel grid step after the inverse rotation
- **AND** the reconstructed vector MUST have the same shape as the original

#### Scenario: Deterministic rotation across machines
- **WHEN** the same model bytes and the same input vectors are encoded on two different machines with the same `kv_quant_bits` setting
- **THEN** the stored quantized payloads MUST be bit-identical on both machines
- **AND** the reconstructed vectors MUST be bit-identical on both machines
- **AND** the rotation seed MUST be derivable from `BLAKE3(model_bytes) XOR SMALLAIOS_KV_ROTATION_SALT`

### Requirement: Low-Rank K Factorization (Optional)
The runtime SHALL implement an optional incremental singular value decomposition of the K cache that retains the top-*k* singular values (default `k = min(head_dim / 2, 64)`), stores only the factored form, re-orthogonalizes every 512 writes to bound accumulated drift, and is enabled via the session configuration knob `kv_lowrank_k = Some(k)`.

#### Scenario: Low-rank reconstruction error bounded by the (k+1)-th singular value
- **WHEN** a K cache of 1024 rows is built with `kv_lowrank_k = Some(64)` and then reconstructed back to full K
- **THEN** the Frobenius norm of `K_reconstructed - K_original` MUST be bounded above by `sqrt(sum of squares of singular values s_{k+1..})` of the original K with a slack factor of 2 for incremental-SVD accumulated error
- **AND** re-orthogonalization MUST be invoked at iteration counts 512 and 1024

#### Scenario: Low-rank K disabled by default
- **WHEN** a session is created with default `KVCacheConfig`
- **THEN** `kv_lowrank_k` MUST be `None`
- **AND** the K cache MUST be stored in the same quantized-rotated format as V
- **AND** no SVD MUST be computed
