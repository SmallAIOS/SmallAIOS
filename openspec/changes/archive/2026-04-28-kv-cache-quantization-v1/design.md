## Context

The SmallAIOS ONNX runtime gains in-graph autoregressive generation via `generative-llm-v1` (Phase 2 sub-graph executor, control-flow operators, real i8 GEMM) and `microsoft-fused-ops-v1` (fused ops including `GroupQueryAttention` with KV-cache lifecycle). Together these make it *possible* to run Llama-3.2-1B end-to-end inside a single `Session::run()` call. They do not, however, make it *memory-feasible* in a <15 MB container: the f16 KV cache for a 4096-token context is already larger than the entire container image budget.

The bottleneck is not compute — the sub-graph executor and real i8 kernels are fine. The bottleneck is the raw bytes of the KV cache, and no amount of operator cleverness fixes that. The only path forward is compressing the cache itself.

Two lines of research converge on a solution:

1. **TurboQuant** (Google, arXiv 2504.19874, April 2025) — random rotation + per-coordinate scalar quantization + 1-bit QJL residual. 3.5 bits/channel with **no accuracy loss** on long-context benchmarks. 6× memory reduction. Data-oblivious; no retraining; runtime overhead characterized as "negligible".
2. **ShadowKV** (CMU + ByteDance, arXiv 2410.21465, October 2024) — low-rank K decomposition plus tiered offload. The low-rank K piece is independent of — and orthogonal to — quantization; the offload piece requires infrastructure SmallAIOS does not have.

This change adopts TurboQuant wholesale and cherry-picks only the low-rank K decomposition from ShadowKV. The rest of ShadowKV is [issue #92](https://github.com/SmallAIOS/SmallAIOS/issues/92).

## Goals / Non-Goals

**Goals:**
- Compress the KV cache to ~3.5 bits/channel with no measurable accuracy loss on Llama-3.2-1B at 4096 context.
- Add optional low-rank K decomposition that composes with quantization for a further 2–3× reduction on the K side.
- Integrate transparently with `GroupQueryAttention` from `microsoft-fused-ops-v1` via a stable `CompressedKVCache` API — the kernel asks for K/V blocks and does not need to know about rotation, quantization, or SVD.
- Keep the implementation `#![no_std]`, `alloc`-only, zero new dependencies.
- Fit a 1B-param LLM at 4096 context inside the <15 MB container budget.

**Non-Goals:**
- Memory-hierarchy KV offload (issue #92).
- Sub-2-bit extreme compression beyond the QJL 1-bit residual.
- Weight quantization (unchanged — Phase 2 i8 GEMM already handles that).
- GPU dispatch of the compressed fast path (CPU only in this change).
- Changes to the Phase 2 sub-graph executor, the WCET budget machinery, or any other crate.

## Decisions

### D1: Pick TurboQuant over GPTQ / AWQ / SmoothQuant

**Decision.** Adopt TurboQuant (rotation + per-coordinate scalar quantizer + QJL residual) as the sole KV-cache compression algorithm.

**Rationale.** GPTQ, AWQ, and SmoothQuant are all **weight** quantization techniques. They require offline calibration, they operate on model weights, and they ship as a one-shot preprocessing step before inference. They do not compete in the same niche as KV cache compression at all — the KV cache consists of activations computed at *inference time*, not weights loaded from disk. No offline calibration is possible because the activations do not exist until the user sends a prompt.

TurboQuant is the opposite: it is data-oblivious, applies at runtime, and is specifically designed for the activation-side KV cache. The paper tests it on Gemma and Mistral; both show loss-free operation at 3.5 bits/channel. There is no realistic alternative in the same category.

**Alternatives considered.** KIVI (2-bit KV cache) — promising but reports some accuracy loss at long contexts and requires per-channel *and* per-token calibration. Rejected because TurboQuant dominates it at the accuracy/speed trade-off and is simpler (no per-token calibration). Plain int8 quantization — works but gives only 2× reduction, insufficient for the <15 MB target.

### D2: Random rotation via Walsh–Hadamard transform with random sign flips

**Decision.** Implement the random rotation as a Walsh–Hadamard transform composed with a diagonal matrix of random `±1` sign flips. The sign-flip vector is stored; the Hadamard matrix itself is not — it is applied in-place via the standard butterfly algorithm in O(D log D) time per vector, where D is `head_dim`.

The sign-flip seed is derived from the model hash:
```
seed = BLAKE3(model_bytes) XOR SMALLAIOS_KV_ROTATION_SALT
```
where `SMALLAIOS_KV_ROTATION_SALT` is a fixed 256-bit project constant. The seed deterministically populates a per-head D-length `±1` vector. The same model on any machine produces the same rotation, which is required for save/restore of compressed caches.

**Rationale.** A full D×D random rotation matrix costs D² = 16384 bytes per head at D=128 (Llama-3.2-1B) and would defeat the whole point of compression — the rotation matrix itself would exceed the compressed cache for short contexts. The Walsh–Hadamard transform is a structured orthogonal matrix that requires no explicit storage and runs in O(D log D) time (compared to O(D²) for a dense matrix-vector product). The random sign flips break the symmetry of the pure Hadamard matrix and recover the theoretical concentration properties that TurboQuant depends on. TurboQuant's paper uses this construction (or an equivalent SRHT — subsampled randomized Hadamard transform) precisely because dense random rotations are not feasible at scale.

**Alternative considered.** Store a full random rotation matrix per head. Rejected: the storage cost is D² per head × num_heads, which for a 32-head 128-dim model is 512 KB before the cache even starts. Defeats the compression budget.

### D3: Per-channel scalar quantizer at 3.5 bits (encoded as 4 bits + 1-bit QJL residual)

**Decision.** Apply a per-channel scalar quantizer after rotation. Each rotated coordinate is quantized independently with its own scale and zero-point, computed per block-of-rows (not per-token) to amortize metadata cost. The payload is stored as **4 bits per channel**, plus the QJL 1-bit residual from D4. The effective precision is ~3.5 bits per TurboQuant's analysis.

The scale and zero point are f16 per channel per block. Block size is **64 rows** (one cache line of rotated coordinates). This gives a storage overhead of `2 × 2 bytes × (D / 64)` per 64 rows, which at D=128 is 8 bytes per 64 rows per head = 0.125 bytes/row overhead, or under 1 bit per channel amortized.

**Implementation detail.** Pack the 4-bit values as two-per-byte using low-nibble and high-nibble. Read-out unpacks into `i8` in a small temporary buffer before the inner-product accumulator.

**Rationale.** Per-channel quantization (vs. per-tensor) is important because after rotation the dynamic range varies across channels — channels that concentrate near zero get tight quantization grids, channels with heavy tails get wider grids. Per-tensor quantization would waste precision on the sparse-tail channels and lose precision on the concentrated channels.

4-bit + 1-bit QJL as encoding of "3.5-bit precision" is the TurboQuant recommended configuration. The paper explicitly shows that the two-stage encoding is strictly better than a flat 4-bit uniform quantizer at the same storage cost.

**Alternative considered.** Uniform 4-bit with no residual. Simpler, but the paper shows a measurable accuracy delta on long-context benchmarks. Rejected because long-context is exactly the regime SmallAIOS targets.

### D4: QJL 1-bit residual encoder

**Decision.** After the per-channel scalar quantizer writes its payload, compute the residual `r = x_rotated - dequantize(q)` and store `sign(r)` as a single bit per channel. During attention, the inner product `<a, b>` is reconstructed as:

```
<a, b> ≈ <dequant(q_a), dequant(q_b)> + C * <sign(r_a), sign(r_b)>
```

where `C` is the per-channel-average residual magnitude precomputed at quantization time (and stored once per block alongside the scale and zero-point). The `<sign, sign>` inner product over D channels is computed as `D - 2 * popcount(bits_a XOR bits_b)`, a single 64-bit XOR + popcount instruction per 64 channels — essentially free on any modern CPU.

**Math.** Per the QJL lemma (Kapralov et al., see the TurboQuant paper for the exact bound), the expected estimation error of this two-term inner product is bounded by `O(sqrt(log(D) / D))` relative to `||a|| * ||b||`, for random rotations of sufficient mixing. This bound is what TurboQuant's "loss-free at 3.5 bits" claim rests on.

**Rationale.** The 1-bit residual is the cheapest possible correction that still bounds the bias of the primary quantizer. It adds 1 bit per channel to the cache (total storage per channel: 4 + 1 = 5 bits, which amortizes to ~3.5 bits of effective precision after accounting for the correlation between the primary and residual terms). The runtime cost is one extra XOR+popcount per attention score, which is negligible compared to the primary quantized dot product.

**Alternative considered.** No residual (4-bit only). Rejected — as discussed in D3, the accuracy delta matters at long context and the 1-bit overhead is trivial.

### D5: Low-rank K decomposition (cherry-picked from ShadowKV)

**Decision.** Implement an **optional** incremental SVD-based low-rank decomposition for the K cache only. When enabled, at most `k` singular values are retained, default `k = min(head_dim / 2, 64)`. The stored form is `U_k · S_k` (dimensions `[rows, k]`) plus `V_k^T` (dimensions `[k, head_dim]`). A query-key inner product `q · K^T` is computed as `q · V_k · (U_k · S_k)^T`, which costs `O(head_dim · k + rows · k)` instead of the naive `O(head_dim · rows)`.

The decomposition is **incremental**: as each new key row is written during decode, it is projected onto the existing `V_k` basis and appended to `U_k · S_k`. A full re-orthogonalization is triggered every **512 writes** to bound accumulated drift. The incremental-update cost is `O(k · head_dim)` per token, which is much cheaper than the attention computation itself (`O(rows · head_dim)` for small `k`).

This only applies to **K**, not V. V is dense-read (every value participates in every attention output) and does not admit the same low-rank structure as K in practice. ShadowKV's paper confirms this asymmetry empirically.

**Rationale.** K in transformer decoders is read once per query token (in the `q · K^T` step) and the resulting attention scores are typically sparse (softmax concentrates on a few rows). Low-rank K captures this read-heavy locality for free. V, by contrast, is read by every softmax-weighted sum and is not low-rank in practice; compressing V via quantization (as TurboQuant already does) is the right move and low-rank V is not.

The configuration defaults `k = min(head_dim / 2, 64)`; for Llama-3.2-1B `head_dim = 128` giving `k = 64`, a 2× reduction on the K side on top of the 4.6× reduction from TurboQuant. Net K compression: roughly 9×.

**Alternative considered.** Full-SVD-per-token recompute. Cost: `O(rows² · head_dim)` per token, unworkable at long contexts. Rejected outright.

### D6: `CompressedKVCache` as a runtime-wrapped struct

**Decision.** The `CompressedKVCache` struct owns the rotation sign-flip state, the quantized payloads, the QJL residual bitmaps, the low-rank K factors (when enabled), and the per-block scale/zero-point metadata. It exposes the following public API:

```rust
pub struct CompressedKVCache { /* opaque */ }

impl CompressedKVCache {
    pub fn new(config: KVCacheConfig, num_heads: usize, head_dim: usize) -> Self;

    pub fn write_block(&mut self, kv_head: usize, start_row: usize, k: &[f16], v: &[f16]);

    pub fn read_block(&self, kv_head: usize, start_row: usize, len: usize) -> (Vec<f16>, Vec<f16>);

    pub fn attention_logits(
        &self,
        kv_head: usize,
        query: &[f16],
        num_rows: usize,
    ) -> Vec<f16>;  // fast path: compute q · K^T directly on the compressed representation

    pub fn weighted_value_sum(
        &self,
        kv_head: usize,
        weights: &[f16],
        num_rows: usize,
    ) -> Vec<f16>;  // fast path: compute softmax(weights) · V directly on the compressed representation
}
```

The GroupQueryAttention kernel calls `write_block` once per token (to append the new K and V) and calls `attention_logits` + `weighted_value_sum` once per query-row to compute the attention output. The kernel does not access the raw rotated or quantized data; it treats the cache as opaque.

**Rationale.** A stable external API insulates the rest of the runtime from the details of rotation, quantization, and SVD. Future algorithm swaps (e.g., sub-2-bit extreme compression, memory-hierarchy offload, GPU fast paths) can happen behind this boundary without touching the GQA kernel or any op code. The fast-path methods (`attention_logits`, `weighted_value_sum`) let the cache implement the inner product directly on the quantized representation, which is where the 8× speedup from TurboQuant comes from — a naive "dequantize then f16 matmul" path would give the memory savings but lose the speedup.

**Alternative considered.** Inline the decompression into the GQA kernel. Rejected: tight coupling, hard to evolve, cannot swap algorithms, and the GQA kernel grows another ~200 LOC of quantization logic that does not belong there.

### D7: Session configuration surface

**Decision.** Three knobs, set at session creation and immutable for the session lifetime:

```rust
pub struct KVCacheConfig {
    /// Scalar quantizer bit width. Default 4. Valid: 3, 4, 8.
    pub kv_quant_bits: u8,
    /// Enable the QJL 1-bit residual correction. Default true.
    pub kv_qjl_residual: bool,
    /// Rank for low-rank K decomposition. None disables it. Some(k) enables with rank k.
    /// Default: None (disabled; opt-in feature).
    pub kv_lowrank_k: Option<usize>,
}
```

`KVCacheConfig` is threaded through `Session::new_with_config` and defaulted inside the existing `Session::new`. All three knobs are model-independent but session-scoped.

**Rationale.** Session-scoped immutability means the compressed cache layout is fixed for the lifetime of the session, which eliminates a whole class of mid-flight reconfiguration bugs. Defaulting low-rank K to disabled keeps the change opt-in and lets users verify the quantization path alone before engaging SVD.

**Alternative considered.** Per-model-or-per-call configuration. Rejected — mid-flight changes to the cache layout would require re-rotating and re-quantizing the existing cache, which defeats the point.

### D8: Phase 2 sub-graph executor interaction

**Decision.** The `CompressedKVCache` struct lives in the **outer** `value_map` (the top-level session value map) per `microsoft-fused-ops-v1` D8, which already places KV cache tensors in outer scope as Loop-carried values. The Phase 2 sub-graph executor (`generative-llm-v1` D3) already copies outer-ref values into the inner Loop body's `value_map` via a shallow clone at the start of each iteration. The `CompressedKVCache` struct exposes `Clone` as a cheap handle-clone (the quantized payloads are held behind an `Arc<Mutex<...>>`-like interior, so cloning the handle does not copy the compressed bytes).

Writes from inside the Loop body propagate back to the outer scope because the Loop-carried-output slot for the cache handle is just the same handle — any `write_block` call mutates the shared payload. The inner graph never sees the concrete `CompressedKVCache` type; it only sees an opaque tensor handle that the GQA op interprets.

**No new sub-executor work is needed.** The contract is that the cache struct is opaque to the inner graph and only the GQA op touches it. This invariant must be stated in `docs/kv-compression-design.md` and asserted by a unit test that confirms a cache written from inside a `Loop` body survives to the next iteration and then to the outer scope.

**Rationale.** Reusing the existing outer-value-map plumbing avoids any new work in the sub-graph executor, which is a large piece of infrastructure that was scoped carefully in `generative-llm-v1`. The only new assumption is that `CompressedKVCache` can be handle-cloned cheaply, which is an implementation detail of the cache struct.

**Alternative considered.** Add a new "persistent value" slot to the sub-graph executor, specifically for KV caches. Rejected — duplicates the existing outer-ref mechanism and requires changes to a module this change has committed not to touch.

### D9: Validation strategy

**Decision.** Three layers of validation:

1. **Unit tests** (~30 total) covering:
   - Per-channel quantize/dequantize round-trip at 3, 4, 8 bits with bounded reconstruction error.
   - QJL residual: verify `<sign(r_a), sign(r_b)>` correlates with `<r_a, r_b>` within the paper's theoretical bound.
   - Walsh–Hadamard transform: orthogonality preservation (`||H x|| == ||x||` to machine precision) and deterministic round-trip.
   - Low-rank K reconstruction error vs full K, bounded by the `(k+1)`-th singular value.
   - Incremental SVD drift after 512 writes bounded by a measured tolerance, re-orthogonalization restores it.
   - CompressedKVCache `write_block` + `read_block` round-trip within the quantization grid.
   - CompressedKVCache `attention_logits` matches a naive `q · K_f16^T` within ±1 quantized step.
   - CompressedKVCache `weighted_value_sum` matches a naive `weights · V_f16` within ±1 quantized step.

2. **Integration test:** end-to-end Llama-3.2-1B 4096-token generation with `kv_quant_bits = 4`, `kv_qjl_residual = true`, `kv_lowrank_k = Some(64)` versus the uncompressed f16 baseline. Assertion: the **output token IDs are identical** for the first 256 generated tokens. TurboQuant claims loss-free at 3.5 bits, so any token-level divergence is a bug.

3. **Memory test:** a micro-benchmark that measures the actual allocated bytes of a `CompressedKVCache` instance for Llama-3.2-1B at 4096 context and asserts it is within **16%** of the theoretical 3.5-bit lower bound. The 16% slack accounts for unavoidable metadata overhead (scales, zero points, QJL block anchors, low-rank basis, re-orthogonalization scratch).

**Rationale.** The memory test is the only one that directly validates the core motivation ("fit inside <15 MB container"). The accuracy test is the one that directly validates the paper's loss-free claim. The unit tests cover the building blocks so that failures at the integration level are debuggable at the block level.

## Alternatives Considered

### A1: Plain int8 KV cache, no rotation, no residual, no low-rank

A 2× memory reduction from f16. Simple to implement. Used by the reference ONNX Runtime's KV cache quantization path.

**Rejected** because 2× is not enough. Llama-3.2-1B at 4096 context goes from 50 MB to 25 MB — still larger than the 15 MB container budget. Needs to go to ~6–8 MB to fit with margin for weights and runtime, which requires both quantization *and* low-rank.

### A2: Sub-2-bit extreme compression (e.g., KIVI or AQLM-style)

2-bit KV cache with calibration. Gives 8× reduction from f16. Reported accuracy loss at long contexts.

**Rejected** for this change because the target is *loss-free* at long context, and TurboQuant at 3.5 bits meets that target without any calibration. Sub-2-bit extreme compression is future work if a use case emerges where 3.5-bit is still too much.

### A3: Memory-hierarchy offload (full ShadowKV)

ShadowKV's full algorithm: low-rank K in fast memory, quantized V offloaded to slower memory with sparse on-the-fly reconstruction.

**Rejected for this change, deferred to [issue #92](https://github.com/SmallAIOS/SmallAIOS/issues/92).** SmallAIOS does not have a storage-tier abstraction — there is no "slower memory" to offload to in a unikernel running in a 15 MB container. Adding one is a significant architectural change outside the scope of a compression change.

### A4: GPU-only fast path with CPU fallback

Implement the compressed inner product only on the GPU (via the NVIDIA/Intel/AMD HAL crates).

**Rejected** because the GPU HAL crates are architectural stubs today with no hardware interaction. CPU is the only execution target that actually works, and CPU is where the container-budget constraint bites hardest.

## Open Questions

### Q1: Should the Walsh–Hadamard transform be applied per-head or across all heads?

TurboQuant's paper operates on full attention-head vectors, so per-head (D = head_dim = 128 for Llama-3.2-1B) is the default. But a cross-head rotation (D = num_heads × head_dim = 4096 for Llama-3.2-1B) would give stronger concentration and potentially let us drop below 3.5 bits.

**Options:**
- (a) Per-head only — matches the paper, simplest.
- (b) Per-head-group only for Grouped Query Attention — aligns rotation with the KV-head grouping.
- (c) Full cross-head — strongest mixing but breaks the GQA broadcast pattern and requires undoing the rotation before every attention score.

**Leaning.** (a). The paper works at per-head, the implementation is simplest, and the 3.5-bit accuracy is already loss-free. Revisit if and when sub-2-bit becomes a goal.

### Q2: How often to re-orthogonalize the incremental SVD?

Every 512 writes is a heuristic. A 4096-context generation performs 4096 writes per K head; that is 8 re-orthogonalizations, each costing `O(k² · head_dim)`. At `k=64`, `head_dim=128`, that is 524288 ops per re-orthogonalization — negligible relative to the rest of the attention.

**Options:**
- (a) Fixed every 512 writes.
- (b) Drift-driven: re-orthogonalize when the measured Frobenius drift exceeds a threshold.
- (c) Never — accept the drift; the SVD is only used for the K side, which is already low precision.

**Leaning.** (a) for the first implementation. Switch to (b) if (a) causes measurable accuracy regression on long contexts.

### Q3: Should `kv_lowrank_k` default to enabled?

Current default: `None` (disabled). Opt-in.

**Arguments for enabling by default:** it is the only way to hit the 6–8 MB target for a 4096-context 1B model. Users who do not enable it will exceed the container budget.

**Arguments against:** SVD failures (near-degenerate K matrices) could produce silent accuracy regressions, and the feature is newer / less battle-tested than TurboQuant alone.

**Leaning.** Ship this change with low-rank K defaulted to disabled and call it out prominently in the session documentation. Flip the default to `Some(64)` in a follow-up change after the integration test suite has accumulated a few hundred model-hours of runtime data.
