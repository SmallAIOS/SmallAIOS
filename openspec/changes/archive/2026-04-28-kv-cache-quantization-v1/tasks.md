## 1. PolarQuantizer Module

- [x] 1.1 Create `onnx-rt/src/kv_compression.rs` module skeleton with the `PolarQuantizer` struct, field definitions for the per-head sign-flip vector, per-block f16 scale, per-block f16 zero point, and the quantized payload byte buffer
- [x] 1.2 Implement the Walsh–Hadamard transform in-place butterfly over an `&mut [f16]` of power-of-two length; return an error if length is not a power of two
- [x] 1.3 Implement the sign-flip application (`x[i] *= sign[i]`) and the seed-to-sign-vector derivation via a deterministic PRNG seeded from `BLAKE3(model_bytes) XOR SMALLAIOS_KV_ROTATION_SALT`
- [x] 1.4 Implement `PolarQuantizer::encode_block` that rotates a block of 64 vectors, computes per-channel min/max across the block, derives per-channel f16 scale and zero point, quantizes each rotated coordinate to the configured bit width (3, 4, or 8), and packs the quantized values into the payload buffer
- [x] 1.5 Implement `PolarQuantizer::decode_block` that unpacks the quantized payload, dequantizes using the stored scale/zero point, and applies the inverse rotation (sign-flip then Walsh–Hadamard transform, which is self-inverse up to a normalization factor)
- [x] 1.6 Unit test: round-trip at 4 bits with the QJL residual disabled stays within ½ grid step per channel
- [x] 1.7 Unit test: round-trip at 8 bits matches f16 within 1 ULP
- [x] 1.8 Unit test: Walsh–Hadamard transform applied twice (with appropriate scaling) returns the original vector to machine precision
- [x] 1.9 Unit test: deterministic rotation — same model seed on two `PolarQuantizer` instances yields byte-identical payloads

## 2. QJLResidualEncoder Module

- [x] 2.1 Add the `QJLResidualEncoder` struct that owns a packed-bit residual buffer (1 bit per channel, packed 64 bits per u64)
- [x] 2.2 Implement `QJLResidualEncoder::encode` that computes `residual = x_rotated - dequantize(q)` and writes the sign bit (0 for non-negative, 1 for negative) into the packed buffer; also precomputes and stores the per-block average `|residual|` magnitude for use as the correction coefficient `C`
- [x] 2.3 Implement `QJLResidualEncoder::inner_product_correction(kv_head, row_a, row_b) -> f16` that computes `C * (D - 2 * popcount(bits_a XOR bits_b))` via a u64 XOR + popcount loop
- [x] 2.4 Unit test: the sum of the primary quantized inner product plus the QJL correction matches the true f16 inner product within the theoretical QJL error bound `O(sqrt(log(D) / D)) * ||a|| * ||b||`
- [x] 2.5 Unit test: sign-bit packing and unpacking round-trip
- [x] 2.6 Unit test: popcount-based correction computation matches a naive per-bit reference loop

## 3. LowRankKeyDecomposer (Incremental SVD)

- [x] 3.1 Add the `LowRankKeyDecomposer` struct owning `U_S: Vec<Vec<f16>>` (row-major `[rows, k]`), `Vt: Vec<Vec<f16>>` (`[k, head_dim]`), a write counter, and a re-orthogonalization cadence constant
- [x] 3.2 Implement `LowRankKeyDecomposer::new(head_dim, k)` that initializes `Vt` to an orthonormal basis derived from the first `k` key rows via Gram-Schmidt
- [x] 3.3 Implement `LowRankKeyDecomposer::write_row(k_new: &[f16])` that projects `k_new` onto the existing `Vt`, appends the projection coefficients as a new row of `U_S`, and increments the write counter
- [x] 3.4 Implement `LowRankKeyDecomposer::maybe_reorthogonalize()` that, when the write counter is a multiple of 512, performs a full SVD on `U_S * Vt` and replaces both factors with the top-`k` components from a fresh SVD via Jacobi rotations
- [x] 3.5 Implement `LowRankKeyDecomposer::reconstruct_row(row_idx) -> Vec<f16>` that returns `U_S[row_idx] * Vt`
- [x] 3.6 Implement `LowRankKeyDecomposer::inner_product(query: &[f16], num_rows: usize) -> Vec<f16>` that computes `q * K^T = q * Vt^T * (U_S)^T` in `O(head_dim * k + num_rows * k)`
- [x] 3.7 Unit test: reconstruction error after 1024 writes bounded by 2x the theoretical (k+1)-th singular value
- [x] 3.8 Unit test: re-orthogonalization triggers at writes 512 and 1024, drift is reset to near zero
- [x] 3.9 Unit test: `inner_product` matches naive `q * K_full^T` within the reconstruction error bound
- [x] 3.10 Unit test: initialization with fewer than `k` rows falls back to using `num_rows` as the effective rank and grows to `k` as more rows arrive

## 4. CompressedKVCache Wrapper

- [x] 4.1 Add the `CompressedKVCache` struct owning a `KVCacheConfig`, per-head `PolarQuantizer` instances for K and V, per-head `QJLResidualEncoder` instances for K and V (when enabled), and per-head optional `LowRankKeyDecomposer` instances for K (when `kv_lowrank_k` is Some)
- [x] 4.2 Implement `CompressedKVCache::new(config, num_heads, head_dim)` that allocates empty per-head state
- [x] 4.3 Implement `CompressedKVCache::write_block(kv_head, start_row, k, v)` that quantizes both K and V via `PolarQuantizer::encode_block`, encodes the QJL residual when enabled, and additionally feeds K rows through `LowRankKeyDecomposer::write_row` when enabled
- [x] 4.4 Implement `CompressedKVCache::read_block(kv_head, start_row, len) -> (Vec<f16>, Vec<f16>)` that dequantizes via `PolarQuantizer::decode_block` and applies the QJL correction on reconstruction where applicable; when low-rank K is enabled, K is reconstructed via `LowRankKeyDecomposer::reconstruct_row` per row
- [x] 4.5 Implement `CompressedKVCache::attention_logits(kv_head, query, num_rows) -> Vec<f16>` fast path: compute `q * K^T` directly on the compressed representation (quantized inner product + QJL correction + low-rank factor product when enabled)
- [x] 4.6 Implement `CompressedKVCache::weighted_value_sum(kv_head, weights, num_rows) -> Vec<f16>` fast path: compute `weights * V` directly on the compressed V representation
- [x] 4.7 Unit test: `write_block` followed by `read_block` round-trip within ±1 grid step per channel at 4 bits
- [x] 4.8 Unit test: `attention_logits` matches `q * K_f16^T` within ±1 quantized step
- [x] 4.9 Unit test: `weighted_value_sum` matches `weights * V_f16` within ±1 quantized step

## 5. GroupQueryAttention Integration

- [ ] 5.1 In `onnx-rt/src/ops/microsoft.rs`, modify the `GroupQueryAttention` kernel (landing in `microsoft-fused-ops-v1`) to accept a `CompressedKVCache` handle from the value_map instead of raw K/V tensors — **DEFERRED**: GQA wiring depends on follow-up integration; tracked for future change.
- [ ] 5.2 Replace the current K/V append paths with `CompressedKVCache::write_block` calls, one per new token row — **DEFERRED**: GQA wiring depends on follow-up integration; tracked for future change.
- [ ] 5.3 Replace the current `q * K^T` computation with `CompressedKVCache::attention_logits` — **DEFERRED**: GQA wiring depends on follow-up integration; tracked for future change.
- [ ] 5.4 Replace the current softmax-weighted `V` sum with `CompressedKVCache::weighted_value_sum` — **DEFERRED**: GQA wiring depends on follow-up integration; tracked for future change.
- [ ] 5.5 Unit test: GQA with a compressed cache produces outputs matching GQA with an uncompressed cache within ±1 quantized step on a hand-built 8-row input — **DEFERRED**: GQA wiring depends on follow-up integration; tracked for future change.

## 6. Session Configuration Surface

- [x] 6.1 Add the `KVCacheConfig { kv_quant_bits, kv_qjl_residual, kv_lowrank_k }` struct and a `Default` impl (`4`, `true`, `None`) in `onnx-rt/src/session.rs`
- [x] 6.2 Extend `Session::new_with_config` (or add a new constructor) that accepts a `KVCacheConfig` and threads it into any `CompressedKVCache` allocated inside the session's value_map for GQA/MHA initializer tensors
- [x] 6.3 Re-export `KVCacheConfig` and `CompressedKVCache` from `onnx-rt/src/lib.rs`

## 7. End-to-End Model Validation

- [ ] 7.1 Add an integration test fixture that loads Llama-3.2-1B from a local ONNX export (or a tiny mock standing in for Llama when the real model is not available in CI) — **DEFERRED**: Llama-3.2-1B end-to-end test depends on §5 (GQA wiring) and full Llama E2E test infrastructure; tracked separately.
- [ ] 7.2 Run 4096-token generation from a fixed prompt with `kv_quant_bits = 4`, `kv_qjl_residual = true`, `kv_lowrank_k = Some(64)` — **DEFERRED**: Llama-3.2-1B end-to-end test depends on §5 (GQA wiring) and full Llama E2E test infrastructure; tracked separately.
- [ ] 7.3 Run the same 4096-token generation with the uncompressed f16 baseline and capture its output token IDs — **DEFERRED**: Llama-3.2-1B end-to-end test depends on §5 (GQA wiring) and full Llama E2E test infrastructure; tracked separately.
- [ ] 7.4 Assert the first 256 generated token IDs are **identical** between compressed and uncompressed runs (TurboQuant loss-free claim) — **DEFERRED**: Llama-3.2-1B end-to-end test depends on §5 (GQA wiring) and full Llama E2E test infrastructure; tracked separately.
- [ ] 7.5 Add a memory-footprint assertion: the `CompressedKVCache` allocated bytes are within 16% of the theoretical `ceil(3.5 bits/channel * num_heads * head_dim * 4096 * 2)` target — **DEFERRED**: Llama-3.2-1B end-to-end test depends on §5 (GQA wiring) and full Llama E2E test infrastructure; tracked separately.

## 8. Validation, Benchmarks, and Documentation

- [x] 8.1 `cargo fmt --check` passes
- [x] 8.2 `cargo clippy -- -D warnings` passes for `onnx-rt` with the new module
- [x] 8.3 `cargo test --workspace` passes including all new unit tests for the compression module (the end-to-end Llama-3.2-1B accuracy test is part of the deferred §7 scope and is not exercised here)
- [ ] 8.4 Add a memory micro-benchmark to `bench/` measuring `CompressedKVCache` allocated bytes at 512, 1024, 2048, and 4096 context lengths — **DEFERRED**: bench/docs follow-on; tracked separately.
- [ ] 8.5 Add an accuracy micro-benchmark to `bench/` measuring the Frobenius error of `CompressedKVCache::attention_logits` versus the f16 reference across a synthetic random-query workload — **DEFERRED**: bench/docs follow-on; tracked separately.
- [ ] 8.6 Update `docs/kv-compression-design.md` with any implementation-driven refinements discovered during the task list (for example, the exact re-orthogonalization cadence if 512 turns out to be wrong) — **DEFERRED**: bench/docs follow-on; tracked separately.
- [ ] 8.7 Add a section to `docs/inference-bus.md` (or the GQA op doc if one exists by then) noting that GQA requires a `CompressedKVCache` in the outer value map when `KVCacheConfig` is non-default — **DEFERRED**: bench/docs follow-on; tracked separately.
