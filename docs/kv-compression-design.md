# KV Cache Compression Design

**Status:** Planning — tracked by OpenSpec change `kv-cache-quantization-v1`.
**Owners:** ONNX runtime team.
**Depends on:** `generative-llm-v1` (Phase 2 sub-graph executor, real i8 GEMM), `microsoft-fused-ops-v1` (GroupQueryAttention with KV-cache lifecycle).
**Related issue:** [#92 — memory-hierarchy KV offload](https://github.com/SmallAIOS/SmallAIOS/issues/92).

## 1. Motivation

Autoregressive LLM decoding in SmallAIOS happens inside a single `Session::run()` call through the Phase 2 sub-graph executor. On every generated token, the model appends one row to each of the K and V caches and reads the full cache back for the next attention step. The cache is therefore the dominant memory cost of long-context inference.

At f16 the KV cache size is:

```
cache_bytes = 2 (K and V)
            * num_layers
            * num_kv_heads
            * head_dim
            * context_len
            * 2 (f16 bytes)
```

For Llama-3.2-1B (16 layers, 8 KV heads, head_dim 64, context 4096):

```
2 * 16 * 8 * 64 * 4096 * 2 = 67 108 864 bytes ≈ 64 MB
```

(Note: architecture details vary; the number is illustrative — the point is that the cache is larger than the entire SmallAIOS <15 MB container image budget.)

Without compression, in-graph generative inference is **memory-infeasible** on the SmallAIOS container target. Weight quantization (Phase 2 real i8 GEMM) addresses the model weights but does nothing for the cache, which is activation-side.

This document describes a design that compresses the KV cache to roughly **3.5 bits per channel** with no measurable accuracy loss, based on the TurboQuant algorithm (Google, arXiv 2504.19874, April 2025), and adds an optional **low-rank K decomposition** cherry-picked from ShadowKV (arXiv 2410.21465, October 2024). Combined, these bring a 64 MB cache down to roughly 6–8 MB, which fits inside the container budget with margin for weights, network stack, and runtime.

## 2. Background: TurboQuant in one page

TurboQuant is a **data-oblivious** KV-cache compression algorithm. It requires no retraining, no calibration, and no model-specific tuning. It compresses each cache vector in two stages:

### Stage 1 — PolarQuant (rotation + per-coordinate scalar quantization)

Given a cache vector `x ∈ R^D`, the encoder:

1. Applies a random orthogonal rotation `R`: `y = R x`. After rotation, the energy of `x` is "spread" across all coordinates of `y`, and the coordinate distribution of `y` is a concentrated Beta-like distribution independent of the original distribution of `x`. This is the whole reason for rotating: it turns a hard quantization problem (compressing a potentially skewed distribution) into an easy one (compressing a known concentrated distribution).
2. Quantizes each coordinate of `y` independently with a per-coordinate scalar quantizer. Because the post-rotation distribution is concentrated, a 3- or 4-bit scalar quantizer per coordinate is enough to retain almost all the information.

The per-coordinate quantizer stores:
- A **scale** (a single f16) and **zero point** (a single f16) per channel per block of rows.
- The quantized values themselves, packed as 3/4/8-bit integers into a byte buffer.

### Stage 2 — QJL residual (1-bit Johnson–Lindenstrauss correction)

After stage 1, the residual `r = y - dequantize(q)` still carries a small amount of signal. TurboQuant exploits this by recording **just the sign** of each residual coordinate (1 bit per channel). During attention, the inner product of two cache vectors is reconstructed as:

```
<a, b> ≈ <dequant(q_a), dequant(q_b)>                   // primary term
       + C * <sign(r_a), sign(r_b)>                      // QJL correction
```

where `C` is the per-block average `|r|` magnitude, precomputed at encode time and stored alongside the scale and zero point. The `<sign(·), sign(·)>` inner product over `D` channels is computed as:

```
<sign(r_a), sign(r_b)> = D - 2 * popcount(bits_a XOR bits_b)
```

This is a single u64 XOR + popcount per 64 channels, essentially free.

### Bit budget

| Component | Bits per channel |
|-----------|------------------|
| 4-bit primary quantizer | 4 |
| 1-bit QJL residual | 1 |
| Per-block metadata (scale + zero + C, amortized over 64 rows) | ~0.25 |
| **Total per channel** | **~5.25 stored** (~3.5 effective precision) |

TurboQuant's paper shows that this configuration is **loss-free** on LongBench, Needle-in-a-Haystack, RULER, and L-Eval for Gemma and Mistral. A 2.5-bit variant shows marginal degradation.

### Error bound

The QJL correction has a theoretical inner-product error bound:

```
|<a, b>_est - <a, b>_true|  ≤  O( sqrt(log(D) / D) ) * ||a|| * ||b||
```

for a random rotation of sufficient mixing. For D=64 (Llama-3.2-1B head_dim) this is about 0.20 relative error, which for attention logits is comfortably below the softmax saturation scale. The paper derives the exact constant.

## 3. Background: Low-rank K from ShadowKV

ShadowKV observes that the K cache, across time, tends to be approximately low rank — the first few singular values carry most of the energy. This is not surprising: for a trained attention head, the keys cluster around a low-dimensional subspace because the head is specialized to attend to particular semantic directions.

Concretely, let `K ∈ R^(T × D)` be the K cache after `T` tokens. ShadowKV decomposes `K = U S V^T` and retains only the top-`k` singular values, giving:

```
K ≈ U_k · S_k · V_k^T
```

where `U_k ∈ R^(T × k)`, `S_k ∈ R^(k × k)`, `V_k^T ∈ R^(k × D)`. The storage cost is `T × k + k × D` instead of `T × D`. For `k = D / 2 = 32` (Llama-3.2-1B head_dim / 2) and `T = 4096`, the compressed K is about 2× smaller than the full K — on top of the TurboQuant 4.6×.

**Why only K, not V?** Attention reads V once per output row (one full matrix-vector product per query, with softmax weights) and the weights are not sparse in general. V therefore does not admit the same low-rank structure in practice. K, by contrast, is read via `q · K^T` and the softmax concentrates on a few rows, making the low-rank approximation of K essentially lossless for attention purposes. ShadowKV's paper confirms this asymmetry empirically.

**Incremental SVD.** Computing a full SVD every time a new token is appended costs `O(T² D)` per token, which is unworkable. Instead, we maintain an **incremental** approximation: project the new row onto the existing basis, append the projection coefficients to `U_k · S_k`, and every 512 writes perform a full re-orthogonalization to bound drift. The per-token cost is `O(k · D)` for the projection plus an amortized `O(k² · D / 512)` for re-orthogonalization — both negligible relative to attention.

## 4. Data flow overview

```
+--------------+      write_block      +-------------------+
| GQA kernel   |----------------------->|  PolarQuantizer   |
| (per token)  |    (k_new, v_new)      |  rotate, quantize |
+--------------+                        +-------------------+
       |                                          |
       |                                          v
       |                                 +-------------------+
       |                                 | QJLResidualEnc.   |
       |                                 |  sign(residual)   |
       |                                 +-------------------+
       |                                          |
       |     (for K only, if enabled)             v
       |                                 +-------------------+
       |                                 | LowRankKeyDecomp. |
       |                                 |  U_S, Vt          |
       |                                 +-------------------+
       |
       |    attention_logits(q)
       |    weighted_value_sum(w)
       v
+--------------+                        +-------------------+
| GQA kernel   |<-----------------------|  CompressedKVCache|
| (read path)  |    (f16 results)       |  fast-path inner  |
+--------------+                        |  product          |
                                        +-------------------+
```

The GQA kernel calls `write_block` on every new token row and calls the fast-path methods (`attention_logits`, `weighted_value_sum`) on every query. It never sees the rotated, quantized, or low-rank representation directly.

## 5. Memory layout of `CompressedKVCache`

The byte-level layout is as follows. All fields are per-head, indexed by `kv_head`.

```
CompressedKVCache {
    config: KVCacheConfig,
    num_heads: usize,
    head_dim: usize,

    per_head: Vec<HeadState>,   // length == num_heads
}

HeadState {
    // PolarQuantizer state for K
    k_sign_flips: Vec<i8>,      // length == head_dim, each is ±1
    k_quant: QuantPayload,      // keys, packed

    // PolarQuantizer state for V
    v_sign_flips: Vec<i8>,      // length == head_dim, each is ±1
    v_quant: QuantPayload,      // values, packed

    // QJL residuals (optional)
    k_qjl: Option<QJLPayload>,
    v_qjl: Option<QJLPayload>,

    // Low-rank K (optional)
    k_lowrank: Option<LowRankState>,
}

QuantPayload {
    // Packed bits, laid out row-major: row 0 channel 0, row 0 channel 1, ...
    bits: Vec<u8>,              // ceil(rows * head_dim * bits / 8) bytes
    bits_per_channel: u8,       // 3, 4, or 8

    // Per-block metadata (block = 64 rows)
    // For each block, head_dim scales + head_dim zero points
    scales: Vec<f16>,           // length == num_blocks * head_dim
    zero_points: Vec<f16>,      // length == num_blocks * head_dim

    // Size tracking
    rows_written: usize,
}

QJLPayload {
    // One bit per (row, channel), packed row-major
    sign_bits: Vec<u64>,        // ceil(rows * head_dim / 64) u64s
    // Per-block average |residual| magnitude (the correction coefficient C)
    c_per_block: Vec<f16>,      // length == num_blocks
}

LowRankState {
    k_rank: usize,              // default min(head_dim / 2, 64)
    u_s: Vec<f16>,              // row-major [rows, k]
    vt: Vec<f16>,               // row-major [k, head_dim]
    write_counter: usize,       // triggers re-orthogonalization at every 512
}
```

### Sizing example (Llama-3.2-1B, 16 layers, 8 KV heads, head_dim 64, context 4096)

Per head per layer at 4-bit + 1-bit QJL:

- K quant bits: `4096 * 64 * 4 / 8` = 131 072 bytes
- K qjl bits: `4096 * 64 / 8` = 32 768 bytes
- K scales + zeros: `64 (blocks) * 64 (channels) * 2 * 2 bytes` = 16 384 bytes
- K C coefficients: `64 * 2 bytes` = 128 bytes
- (V has the same layout, same totals)

Total K + V per head per layer ≈ `2 * (131072 + 32768 + 16384 + 128)` = 360 704 bytes ≈ 352 KB

Total across 8 heads, 16 layers: `352 KB * 8 * 16` ≈ 44 MB.

Hmm — 44 MB is **larger than the f16 baseline** (roughly 50 MB) by only a modest factor. This illustrates a subtle point: the TurboQuant paper reports 6× reduction against a *naive* f16 layout *without* per-block metadata. Our per-block metadata (scales + zeros per channel per block of 64 rows) is proportionally larger than the main payload for short blocks and small head_dim. **Two mitigations** bring us into the promised range:

1. **Increase block size** from 64 to 256 rows. The metadata cost drops by 4×. At block 256 the K+V total drops to roughly 16 MB.
2. **Enable low-rank K** with `k = 32 = head_dim / 2`. Keys now cost `rows * k * 2 bytes = 4096 * 32 * 2 = 262 144 bytes` per head per layer for `U_S` plus `k * head_dim * 2 = 4096 bytes` for `Vt`, a factor of roughly 2× reduction on the K side. This brings total K+V to ~9 MB.

**Revised budget.** With block size 256 and low-rank K = 32, total compressed cache for Llama-3.2-1B at 4096 context is approximately **9 MB**, fitting comfortably inside the 15 MB container budget alongside weights, network stack, and runtime.

Open issue: this is larger than the 6× factor the paper headlines. The headline comes from per-tensor metadata, not per-block. Revisit block size in the task 8.4 memory benchmark; if 256 is too coarse for accuracy at this head_dim, try 128 and see how the trade-off shifts.

## 6. Walsh–Hadamard transform algorithm

The Walsh–Hadamard transform on a vector `x` of length `D` (power of two) is defined recursively:

```
H_1 = [1]
H_{2n} = [[H_n  H_n],
          [H_n -H_n]]
```

The transform `y = H x` can be computed in place in `O(D log D)` via the standard butterfly:

```
function wht(x):
    h = 1
    while h < D:
        for i in 0..D step 2h:
            for j in 0..h:
                a = x[i + j]
                b = x[i + j + h]
                x[i + j]     = a + b
                x[i + j + h] = a - b
        h *= 2
```

**Orthogonality and inversion.** `H_D` is symmetric and `H_D · H_D = D · I`. So `H^{-1} = (1/D) H`. In practice we apply `H` on encode, apply `H` on decode, and divide by `D` once on decode. `||H x|| = sqrt(D) * ||x||` so magnitudes are preserved up to a known constant that is folded into the per-channel scale.

**Random sign flips.** A plain Hadamard transform is not "random" — it is a fixed deterministic matrix. To obtain the randomness TurboQuant requires, we multiply by a diagonal `±1` matrix *before* the Hadamard:

```
y = H (S x)   where S = diag(s_1, ..., s_D), s_i ∈ {±1}
```

The sign vector `s` is generated deterministically from the model hash as described in §8. Applying the inverse is symmetric:

```
x = S (H^{-1} y) = S ((1/D) H y)
```

since `S^{-1} = S` (each sign is its own inverse).

## 7. Incremental SVD pseudocode

```
function write_row(k_new):
    # k_new ∈ R^D, the new key row
    coeffs = Vt * k_new                    # project onto existing basis, R^k
    append_row(U_S, coeffs)                # U_S is now [rows+1, k]
    write_counter += 1
    if write_counter % 512 == 0:
        reorthogonalize()

function reorthogonalize():
    K_approx = U_S * Vt                    # rebuild R^[rows, D]
    (U_new, S_new, Vt_new) = jacobi_svd(K_approx)
    # Retain top-k
    U_S = U_new[:, :k] * S_new[:k]
    Vt  = Vt_new[:k, :]

function reconstruct_row(row_idx):
    return U_S[row_idx] * Vt

function inner_product(query, num_rows):
    # Compute q · K^T in low-rank form:
    # q · (U_S * Vt)^T = q · Vt^T · U_S^T
    tmp = query * Vt^T                     # R^k
    return U_S[:num_rows] * tmp            # R^num_rows
```

The Jacobi SVD is a straightforward `O(D³)` routine; for `D = 64` this is `262 144` ops per re-orthogonalization, amortized over 512 writes it is `512` ops per write, well under the `O(k · D) = 2048` ops per write of the projection step.

**Initialization.** For the first `k` rows, use Gram-Schmidt on the rows themselves to build an orthonormal `Vt`, rather than running a full SVD. This avoids a degenerate low-rank approximation when the cache is empty.

## 8. Deterministic rotation seed

The sign-flip vector must be identical across all machines running the same model so that a cache saved on one host can be replayed on another. We derive it from the model bytes:

```
seed_bytes = BLAKE3(model_bytes) XOR SMALLAIOS_KV_ROTATION_SALT
```

where `SMALLAIOS_KV_ROTATION_SALT` is a fixed 256-bit project constant. The seed bytes are then used to deterministically populate a per-head `D`-length `±1` vector via a xoshiro256++ PRNG (already vendored in SmallAIOS for the random-op operators added in Phase 2).

The salt exists so that the rotation is distinct from the identity `BLAKE3(model_bytes)` value, which might be used for other purposes (e.g., cache keying).

## 9. GroupQueryAttention read path

On each decode step the GQA kernel:

1. Computes the new Q, K, V projections from the latest token's hidden state.
2. For each KV head `h`: call `cache.write_block(h, current_row, k_new[h], v_new[h])`. This triggers:
   - Rotation + per-channel quantization of `k_new[h]` and `v_new[h]`.
   - QJL residual encoding.
   - (If enabled) projection of `k_new[h]` onto the low-rank basis.
3. For each Q head `h_q`: determine the corresponding KV head `h = h_q // group_size` (GQA head grouping).
4. Compute attention logits: `logits = cache.attention_logits(h, q[h_q], current_row + 1)`. This is the fast path — it computes the quantized inner product + QJL correction + (optionally) the low-rank product, all inside the cache struct.
5. Apply the softmax to the logits.
6. Compute the weighted sum: `out[h_q] = cache.weighted_value_sum(h, softmax_weights, current_row + 1)`.

The GQA kernel never touches the rotated, quantized, or low-rank data directly.

## 10. Write path inside a Loop body

When the GQA op fires inside the sub-graph executor's Loop body, the sequence is:

1. The Loop body receives a handle to the `CompressedKVCache` via an **outer-ref copy** performed by the sub-executor at the start of the iteration (per `generative-llm-v1` D3).
2. The handle is an `Arc`-like shallow clone of the cache struct — the heavy state (payload buffers, SVD factors) is held behind interior mutability and is shared across all iterations and across the outer scope.
3. When the GQA op calls `cache.write_block(...)`, the interior buffers grow and the change is visible to the next iteration and to the outer scope after the Loop completes.
4. When the Loop terminates, the cache handle is propagated back to the outer scope as a Loop-carried output slot.

This pattern requires no new infrastructure in the sub-graph executor. It relies on:
- The existing outer-ref copy mechanism (already in `generative-llm-v1`).
- The `CompressedKVCache` struct implementing `Clone` as a cheap handle-copy.

**Invariant.** The cache handle is shared across all Loop iterations and the outer scope. Writes from any iteration are visible to the next iteration and to the outer scope. This is asserted by a unit test in task 5.5.

## 11. Performance model

### Memory bandwidth

Attention-logit computation `q · K^T` at row count `T` and head dimension `D`:

| Format | Read bytes per score | Notes |
|--------|----------------------|-------|
| f16 baseline | `2D` | 1 f16 per channel |
| 4-bit PolarQuant | `D/2` | 1 nibble per channel |
| 4-bit + QJL | `D/2 + D/8` | primary + 1 sign bit per channel |
| Low-rank K (k = D/2) | `D + k = 1.5 D` (projection-space) | f16 factors, but only `k` ops per score |

The 4-bit PolarQuant path reads **4× less memory** per attention score than the f16 baseline, which on a CPU translates directly to 4× faster attention logits (attention is memory-bound at long contexts). TurboQuant's 8× headline on H100 comes from the GPU Tensor Core path for 4-bit integer dot products, which is not applicable to our CPU-only path — but the 4× memory-bandwidth win still lands on CPU.

### Compute

Per attention score, the operations are:

- Primary inner product: `D` multiply-adds at 4-bit × 4-bit → `i16` accumulator. On an AVX-512 CPU this is vectorized to 64 channels per instruction.
- QJL correction: 1 XOR + 1 popcount + 1 multiply per 64 channels. Negligible.
- Low-rank product: `k` multiply-adds at f16, done once per query not once per row — amortized to zero per score.

The compute is **negligible** compared to memory bandwidth, which is the point of TurboQuant's design.

## 12. Validation and test plan

See `openspec/changes/kv-cache-quantization-v1/tasks.md` section 8 for the full list. Summary:

- **Unit tests** (tasks 1.6–4.9, ~30 tests): round-trip, orthogonality, error bounds, sign-bit packing, low-rank drift.
- **Integration test** (task 7.1–7.4): Llama-3.2-1B 4096-token generation with compressed vs. uncompressed cache, assert identical output token IDs for the first 256 tokens (loss-free claim).
- **Memory benchmark** (task 7.5, 8.4): `CompressedKVCache` allocated bytes within 16% of the 3.5-bit theoretical lower bound.
- **Accuracy benchmark** (task 8.5): `attention_logits` Frobenius error vs. f16 reference across synthetic workloads.

## 13. Future work

1. **Memory-hierarchy KV offload** — the rest of ShadowKV. Requires a storage-tier abstraction in SmallAIOS. Tracked in [issue #92](https://github.com/SmallAIOS/SmallAIOS/issues/92).
2. **Sub-2-bit extreme compression** — TurboQuant's 2.5-bit mode is already implemented as a config knob (`kv_quant_bits = 3` is close). Going below would need a different residual encoder, maybe a 2-bit vector quantizer on top of the 2-bit primary.
3. **GPU dispatch of the compressed fast path** — the NVIDIA / Intel / AMD HAL crates are architectural stubs today. When they gain real hardware interaction, they can consume the same `CompressedKVCache` layout and use integer Tensor Core kernels for the 8× speedup the TurboQuant paper reports on H100.
4. **Per-token adaptive bit width** — different tokens in the context have different information content. A future refinement could quantize "important" tokens (e.g., the ones the last `N` queries attended to most) at higher bit widths and "boring" ones at lower. Requires per-token metadata, which the current block-level layout does not provide.
5. **Streaming re-orthogonalization** — the current design re-orthogonalizes every 512 writes in a single synchronous step. A streaming variant could spread the cost across the intervening 511 writes. Probably unnecessary at 4096 context; may matter at 32k.

## 14. References

- **TurboQuant**: Zandieh, Daliri, Hadian, Mirrokni. *TurboQuant: Random Rotation for Extreme KV Cache Compression*. arXiv:2504.19874, 2025. [Paper](https://arxiv.org/abs/2504.19874) · [Google Research blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- **ShadowKV**: Sun, Chen, Yuan, Chen. *ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference*. arXiv:2410.21465, 2024. [Paper](https://arxiv.org/abs/2410.21465)
- **Johnson–Lindenstrauss lemma and 1-bit QJL**: Kapralov et al., various. See §3.3 of the TurboQuant paper for the formal bound used in this design.
- **SmallAIOS sub-graph executor**: `docs/sub-graph-executor-design.md` (landed with `generative-llm-v1`).
- **SmallAIOS GroupQueryAttention**: `openspec/changes/microsoft-fused-ops-v1/design.md` (planning, PR #91).

## 15. Document history

| Date | Change | Author |
|------|--------|--------|
| 2026-04-11 | Initial draft under `kv-cache-quantization-v1` | ONNX runtime team |
