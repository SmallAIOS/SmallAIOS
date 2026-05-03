## Why

KV cache is the dominant memory cost at long contexts during autoregressive LLM decoding. For a 1B-parameter LLaMA-class model at a 4096-token context, the f16 KV cache is on the order of 50 MB — **larger than the entire SmallAIOS <15 MB container budget** before a single inference has run. Without aggressive cache compression, in-graph generative inference (unlocked by Phase 2's sub-graph executor, `generative-llm-v1`) is memory-bound well before it is compute-bound, and long-context models simply cannot run.

Two recent results change the trade-off:

1. **TurboQuant** (Zandieh, Daliri, Hadian, Mirrokni, Google, arXiv 2504.19874) shows that a random rotation followed by per-coordinate scalar quantization, plus a 1-bit Quantized Johnson-Lindenstrauss (QJL) residual correction, reaches **3.5 bits/channel with no measurable accuracy loss** on LongBench, Needle-in-a-Haystack, RULER, and L-Eval, and **2.5 bits/channel** with marginal degradation. This is a **6× KV memory reduction** over f16, data-oblivious (no retraining, no calibration), with "negligible runtime overhead". The paper reports an **8× attention-logit speedup at 4 bits on H100** because the rotated-quantized representation enables an SIMD-friendly inner-product accumulator.
2. **ShadowKV** (Sun et al., CMU + ByteDance, arXiv 2410.21465) shows that the K-cache is low-rank in practice and that an SVD-based factorization of K gives another 2–3× compression *on top of* quantization, with no accuracy cost, because K is accessed once per query token and tolerates reconstruction latency that V cannot.

This change adopts **TurboQuant wholesale** for both K and V, and **cherry-picks only the low-rank K decomposition** from ShadowKV. The rest of ShadowKV — memory-hierarchy offload, sparse on-the-fly reconstruction, storage-tier placement — is explicitly deferred to [issue #92](https://github.com/SmallAIOS/SmallAIOS/issues/92) until SmallAIOS grows a storage-tier abstraction.

Combined, compressed 3.5-bit K (further factorized low-rank) + 3.5-bit V cuts a 50 MB KV cache to roughly 6–8 MB, fitting comfortably inside the <15 MB container with margin for model weights, network, and runtime.

This change layers on top of `microsoft-fused-ops-v1` (PR #91), which introduces `GroupQueryAttention` with KV-cache lifecycle management. The GQA kernel there is the natural — and only — consumer of the compressed cache.

## What Changes

A new runtime module `onnx-rt/src/kv_compression.rs` that implements a drop-in compressed KV cache layer. Concretely:

1. **`PolarQuantizer`** — applies a deterministic random rotation matrix (a Walsh–Hadamard transform with random sign flips, seed derived from the model hash) and then a per-coordinate scalar quantizer at **3, 4, or 8 bits**. Stores the quantized payload plus per-channel scale and per-channel zero point. The rotation is O(D log D) per vector and requires no stored D×D matrix.
2. **`QJLResidualEncoder`** — computes a 1-bit sign quantization of the residual left over after `PolarQuantizer`, packed one bit per channel. Used as a correction term during attention inner-product reconstruction. Adds a fraction of a bit of effective precision to the dominant inner-product estimate (the main "3.5-bit" claim in the TurboQuant paper comes from combining a 4-bit scalar quantizer with this residual).
3. **`LowRankKeyDecomposer`** — performs an **incremental** SVD on the K-cache as new key rows are written during decode. Retains the top-*k* singular values, default `k = min(head_dim / 2, 64)`. Stores `U·S` (the low-rank factored form) rather than the full K. **This is the cherry-picked piece from ShadowKV.**
4. **`CompressedKVCache`** — a struct that owns the rotation matrix state, the quantized payloads (keys and values), the QJL residual bitmaps, and the low-rank K factors. Exposes a stable **`read_block(head, start, len) -> (K_slice, V_slice)`** / **`write_block(head, start, data)`** API so that the `GroupQueryAttention` kernel from `microsoft-fused-ops-v1` can use it transparently. The GQA kernel does not need to know about rotation, quantization, or SVD — it asks for K and V blocks and gets uncompressed slices on demand (or, in the fast path, asks for the quantized inner-product result directly).

The change also extends `ops/microsoft.rs`'s GQA kernel to read/write through `CompressedKVCache` instead of raw tensors, and adds three session-level configuration knobs: `kv_quant_bits`, `kv_qjl_residual`, and `kv_lowrank_k`.

## Capabilities

### Modified Capabilities
- `onnx-cpu-execution`: Add requirements for `CompressedKVCache`, the PolarQuant + QJL encode/decode pipeline, and the optional low-rank K factorization.

## Impact

- **Code:**
  - `onnx-rt/src/kv_compression.rs` — **new file**, approximately 1200 LOC, including the PolarQuantizer, QJLResidualEncoder, LowRankKeyDecomposer, and CompressedKVCache struct and their unit tests.
  - `onnx-rt/src/ops/microsoft.rs` — **modified**. The `GroupQueryAttention` kernel (landing in `microsoft-fused-ops-v1`) reads and writes through `CompressedKVCache` instead of raw K/V tensors. No changes to public op attributes or ONNX conformance.
  - `onnx-rt/src/session.rs` — **modified**. Session configuration gains three new knobs (`kv_quant_bits`, `kv_qjl_residual`, `kv_lowrank_k`), set at session creation, immutable for the session lifetime.
  - `onnx-rt/src/lib.rs` — **modified**. Re-export `CompressedKVCache` and the three config knobs.
  - **No changes** to the sub-graph executor, the WCET budget machinery, or any other crate.

- **Memory:** Approximately **6× reduction** in KV cache footprint at 3.5 bits/channel versus f16, plus an additional 2–3× from low-rank K on top of quantization. A 1B-param LLM at 4096 context drops from ~50 MB uncompressed to ~6–8 MB compressed — fitting inside the <15 MB container target with margin.

- **Speed:** At 4 bits the attention-logit inner product is 8× faster than f32 (TurboQuant paper claim for H100; on CPU the speedup is smaller but still positive because the rotated quantized representation is an SIMD-friendly `i8` dot product with a small scalar correction term). No slowdown expected in any configuration; at 8 bits the pipeline is pass-through-equivalent to the f16 baseline.

- **Accuracy:** TurboQuant claims **loss-free** at 3.5 bits on long-context benchmarks. SmallAIOS adds an empirical regression test: Llama-3.2-1B 4096-token generation with compressed cache versus uncompressed, asserting **identical output token IDs** and inner-product reconstruction within **±1 quantized step** of an f16 reference.

- **APIs:** No breaking changes. `CompressedKVCache` is a new type, and the three config knobs default to `kv_quant_bits = 4`, `kv_qjl_residual = true`, `kv_lowrank_k = None` (low-rank disabled by default; opt-in).

- **Dependencies:** None new. The math is elementary linear algebra (Hadamard transform, SVD via Jacobi rotations, 1-bit sign), all implementable in `#![no_std]` with `alloc`.

- **Testing:** ~40 new unit tests, plus one end-to-end accuracy test against Llama-3.2-1B at 4096 context.

## Out of Scope

Deliberately excluded from this change and tracked elsewhere:

- **Memory-hierarchy KV offload** — the rest of ShadowKV (offloading V to slower tiers with on-the-fly sparse reconstruction). Tracked in [issue #92](https://github.com/SmallAIOS/SmallAIOS/issues/92). Deferred until SmallAIOS grows a storage-tier abstraction; without one there is nowhere to offload *to*.
- **Disk-backed KV cache** — same reason, same issue.
- **Sub-2-bit extreme compression** — beyond the QJL 1-bit residual. Not on the critical path until 3.5-bit accuracy proves insufficient on some target model.
- **Weight quantization** — this change is KV cache only. Weight matmuls continue to use the Phase 2 real i8 GEMM kernel from `generative-llm-v1`.
- **GPU dispatch of the compressed-cache fast path** — CPU only. The GPU HAL stays untouched. A future change can teach the NVIDIA / Intel / AMD crates to consume the same `CompressedKVCache` layout.
- **JIT of the rotated-quantized inner product** — the implementation is a plain Rust SIMD-friendly loop. Machine-code generation is out of scope.
- **Changes to the Phase 2 sub-graph executor** — the executor is treated as a black box per `docs/sub-graph-executor-design.md`. The `CompressedKVCache` lives in the outer value_map and is passed into Loop bodies as an outer reference, which the existing sub-executor already supports.

## Risks

- **Random-rotation reproducibility.** The rotation must produce the same outputs on every machine running the same model, otherwise a cache saved on one host cannot be replayed on another. *Mitigation:* seed the Walsh–Hadamard sign flips from `BLAKE3(model_bytes) XOR SMALLAIOS_KV_ROTATION_SALT` where the salt is a fixed project constant. Document the seed derivation in `docs/kv-compression-design.md` so third-party toolchains can match it.
- **KV cache lifecycle inside Loop bodies.** When `GroupQueryAttention` inside a `Loop` body writes to the cache, the compressed representation must persist across iterations. The `CompressedKVCache` struct lives in the **outer** value_map (per `microsoft-fused-ops-v1` D8, which puts KV-cache tensors in outer scope as Loop-carried values). The Phase 2 sub-graph executor (per `generative-llm-v1` D3) already copies outer-ref values into the inner scope per iteration; that copy is a shallow clone of the struct handle, not a deep copy of the compressed payload. The invariant "cache handle is shared across iterations" must be documented and asserted in a unit test.
- **QJL dot-product correction cost.** The 1-bit residual adds a second `popcount`-based inner product to every attention score computation. On a modern CPU the cost is under 5% of the primary quantized inner product, but it is non-zero and must be measured.
- **Incremental SVD stability.** Incremental SVD accumulates rounding error per step. For K caches that grow to 4096 rows, the accumulated drift must be bounded. *Mitigation:* re-orthogonalize every 512 writes; document the budget and validate with a round-trip reconstruction test.
- **Parameter interaction with GQA head grouping.** GQA broadcasts one K/V head across multiple Q heads. The compressed cache is indexed by KV-head (not Q-head), and the GQA kernel must respect this indexing. The API on `CompressedKVCache` takes `kv_head: usize` explicitly to make the indexing unambiguous.
