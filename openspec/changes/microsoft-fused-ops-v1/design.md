# Design — Microsoft Fused Ops for Llama / Gemma / DeepSeek

## Context

The empirical coverage probe (`tools/coverage-probe/REPORT.md`) walked
the canonical HuggingFace ONNX exports of three modern decoder-only
LLM families — `meta-llama/Llama-3.2-1B`, `google/gemma-3-1b-it`, and
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` — counting node frequencies
by op name and domain. Every standard-domain ONNX op in all three
models is already implemented by Phase 1 + Phase 2. The only unknown
ops all live in the `com.microsoft` domain, and together they form
exactly five distinct op names:

| Op | Llama-3.2-1B | Gemma 3 1b | DeepSeek-R1-Distill-Qwen-1.5B |
|---|:---:|:---:|:---:|
| `SimplifiedLayerNormalization` | ✅ | ✅ | ✅ (1 use) |
| `SkipSimplifiedLayerNormalization` | ✅ | ❌ | ✅ |
| `GroupQueryAttention` | ✅ | ✅ | ❌ |
| `MultiHeadAttention` | ❌ | ❌ | ✅ |
| `RotaryEmbedding` | ❌ (inside GQA) | ❌ (inside GQA) | ✅ |

The union is irreducible: no op can be removed without breaking at
least one family. Gemma uses a plain `Add` residual where Llama uses
`SkipSimplifiedLayerNormalization`, and DeepSeek replaces
`GroupQueryAttention` + fused RoPE with the older
`MultiHeadAttention` + a standalone `RotaryEmbedding` node.

The original `onnx-full-coverage-roadmap-v1` change (now archived)
classified every `com.microsoft` op as *Skipped-vendor* on the
assumption that they were ORT-specific optimizations. The coverage
probe shows that assumption was wrong: these ops are not optional
optimizations, they are the output of the HuggingFace
`optimum-cli export onnx` pipeline, which runs the ORT transformer
optimizer by default. Without these five ops, SmallAIOS cannot load
any of the three mainstream 1B-class open-weights decoder LLMs off
the shelf.

This change reverses the Skipped-vendor decision for these five
specific ops while leaving the rest of the `com.microsoft` op tail
Skipped. If a future model target requires another one, a future
change will add it.

## Goals / Non-Goals

**Goals:**
- Load `Llama-3.2-1B`, `Gemma 3 1b`, and
  `DeepSeek-R1-Distill-Qwen-1.5B` from their canonical HF ONNX exports
  via `Session::new_from_file()` and complete a 1-token generation.
- Add the 5 operators as first-class, domain-namespaced entries in
  `OperatorRegistry` so provenance is preserved (Standard ONNX vs
  Microsoft fusion).
- Maintain `#![no_std]` compatibility. No new external dependencies.
- Reuse the Phase 2 RMSNorm kernel (`op_rms_normalization`) and the
  Phase 2 sub-graph executor with *no* changes to either. All KV-cache
  interactions are handled by how the outer graph passes in values,
  not by changes to the sub-graph executor itself.

**Non-Goals:**
- GPU dispatch. CPU only.
- Other `com.microsoft` ops (`QLinear*` fused activations,
  `QAttention`, `EmbedLayerNormalization` variants, `MatMulNBits`,
  `GatherBlockQuantized`, etc.). Stay Skipped-vendor.
- FP16 / BF16 specialized kernels. Accept these dtypes via existing
  type-promotion shims, nothing more.
- Rewriting or extending the sub-graph executor. D8 documents the
  constraint KV-cache tensors impose; it does not change executor code.
- Implementing flash-attention or paged-attention fast paths. The
  first implementation uses a straightforward tiled masked SDPA. If
  performance becomes a gate, a follow-up change adds those.

## Decisions

### D1: Treat `com.microsoft` as a first-class but namespaced domain

**Decision.** Add a `Domain` enum to `onnx-rt/src/operators.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Domain {
    StandardOnnx,
    MicrosoftFused,
}
```

Every `OpKind` variant records its domain (for existing standard ops
this is always `Domain::StandardOnnx`). `OperatorRegistry` gets a
`fn lookup_by_domain_and_name(domain: &str, name: &str) -> Option<OpKind>`
helper that routes `"com.microsoft"` to the MicrosoftFused table and
`""` (the default domain) or `"ai.onnx"` to the StandardOnnx table.

The node dispatcher in `executor.rs` currently ignores `NodeProto.domain`.
That field is now read and forwarded to `lookup_by_domain_and_name`.
Two nodes named `"MultiHeadAttention"` in different domains will resolve
to different `OpKind` values (and dispatch to different code paths,
even if only one is implemented today).

**Rationale.** We want provenance. Treating Microsoft ops as if they
were standard ONNX ops (by aliasing them into the StandardOnnx
namespace) would collapse the distinction and prevent future
cross-vendor support — imagine an `ai.amd` or `ai.nvidia` contrib op
with the same name but different semantics. Collapsing domains would
also confuse tooling: a coverage probe should be able to say "we
implement 131 standard-domain ops and 5 microsoft-domain ops", not
"we implement 136 ops of uncertain origin".

The runtime cost of the enum check is one branch on a cold path
(graph compilation time, not per-op dispatch); it does not affect
steady-state inference performance.

**Alternative considered.** Alias each Microsoft op into the
StandardOnnx namespace under its natural name (e.g.,
`SimplifiedLayerNormalization` becomes an alias for
`RMSNormalization`). Rejected. It loses provenance, it breaks the
principle that our `OpKind` enum reflects what the model says it
needs, and it creates surprising-by-distance behavior when a future
standard ONNX opset adds a genuinely-different operator under one of
these names.

### D2: `SimplifiedLayerNormalization` is a thin alias for `RMSNormalization`

**Decision.** `op_simplified_layer_normalization` is ~20 LOC; it
delegates to the Phase 2 `op_rms_normalization` kernel.

**Math.** `SimplifiedLayerNormalization` computes
`y = x / sqrt(mean(x^2) + epsilon) * scale` with an optional bias
add. Critically, there is **no mean subtraction**. This is identical
to the math of RMSNorm as defined by Zhang & Sennrich (2019) and
identical to the Phase 2 `op_rms_normalization` implementation.

**Attributes (per ORT contrib op spec):**
- `epsilon` (float, default 1e-5) — numerical stability term inside
  the square root. Same semantics as RMSNorm `epsilon`.
- `axis` (int, default -1) — axis to normalize over. Always the last
  axis for the HF LLM exports we care about. Phase 2 RMSNorm already
  supports arbitrary negative axes.
- `stash_type` (int, default 1 = f32) — intermediate accumulator
  dtype. We always use f32 regardless of input dtype.

**Outputs.** `SimplifiedLayerNormalization` has an optional second
output (`inv_std_dev`) used by the backward pass. We only produce the
first output; if the second is requested by a model, we error out
cleanly (no legitimate inference graph asks for it).

**Implementation shape:**

```rust
pub fn op_simplified_layer_normalization(
    inputs: &[Tensor],
    attrs: &[AttributeProto],
) -> Result<Vec<Tensor>, OperatorError> {
    let epsilon = attr_float(attrs, "epsilon").unwrap_or(1e-5);
    let axis = attr_int(attrs, "axis").unwrap_or(-1);
    // Delegate to the Phase 2 RMSNorm kernel, which already handles
    // axis normalization and the optional bias.
    super::generative::op_rms_normalization_raw(
        &inputs[0], &inputs[1], inputs.get(2), epsilon, axis,
    ).map(|t| alloc::vec![t])
}
```

**Rationale.** The ops are mathematically identical. The only reason
this is not literally a direct reuse is the namespacing from D1 — the
dispatcher needs a handle to call for the Microsoft-domain name. A
thin wrapper preserves both the mathematical identity and the
registry provenance.

### D3: `SkipSimplifiedLayerNormalization` fuses RMSNorm + residual Add

**Decision.** `op_skip_simplified_layer_normalization` is ~50 LOC.
It reads three inputs (`x`, `skip`, `scale`) plus an optional bias,
and computes:

```
pre_layernorm = x + skip + bias  // element-wise
y             = pre_layernorm / sqrt(mean(pre_layernorm^2) + epsilon) * scale
```

It produces two outputs: the normalized `y` and the pre-layernorm sum
`pre_layernorm` (which is re-used as the input to the next layer's
skip connection in the transformer block, so ORT emits it to avoid a
redundant add in the next block).

**Why ORT fuses this.** In the transformer residual stream, every
layer reads `hidden + residual`, normalizes, computes attention/FFN,
and writes a new `hidden + residual`. The sum is both the output of
layer N and the skip input of layer N+1. A non-fused export would
read the residual stream twice (once for the normalize, once for the
next add), doubling the memory bandwidth on the residual path. The
fused op reads once and produces both outputs.

**Gemma does not use this op.** The coverage probe confirms Gemma 3
emits plain `Add` + `SimplifiedLayerNormalization` nodes in sequence
where Llama emits a single `SkipSimplifiedLayerNormalization`. This
is a legitimate exporter choice (Gemma's export was produced with
ORT transformer optimizations partially disabled), and it means
`op_skip_simplified_layer_normalization` is required *only* by Llama
and DeepSeek. We still implement it unconditionally — there is no
conditional-compile reason not to, and future Gemma exports may
enable the fusion.

**Implementation.** Sum the three (or two) input tensors into a
contiguous buffer, call `op_rms_normalization_raw` on that buffer
with the provided scale, and return both tensors. The sum buffer is
the `pre_layernorm` output; the normalized tensor is `y`. ~50 LOC
total.

### D4: `GroupQueryAttention` is the substantial piece

**Decision.** `op_group_query_attention` is the largest single
operator in this tier — approximately 600-800 LOC. It fuses RoPE,
KV-cache concatenation, grouped-query dispatch, scaled dot-product
attention, and causal masking into one op.

**Algorithm.**

1. **Parse inputs.** The ORT contrib op spec defines the following
   inputs (with exact indices):
   - `query` (B, Sq, hidden) — where `hidden = num_heads * head_dim`
   - `key` (B, Sq, kv_hidden) — where `kv_hidden = num_kv_heads * head_dim`
   - `value` (B, Sq, kv_hidden)
   - `past_key` (B, num_kv_heads, past_Sk, head_dim) — the KV-cache
     from previous Loop iterations
   - `past_value` (B, num_kv_heads, past_Sk, head_dim)
   - `seqlens_k` (B,) — int32 vector giving the *current* key-sequence
     length (per batch) including all past tokens
   - `total_sequence_length` (scalar int32) — max key-sequence length
     across the batch, used to size the working tensors
   - `cos_cache` (max_seq_len, head_dim / 2)
   - `sin_cache` (max_seq_len, head_dim / 2)

2. **Attributes.**
   - `num_heads` (int, required)
   - `kv_num_heads` (int, required, `num_heads % kv_num_heads == 0`)
   - `scale` (float, optional; defaults to `1/sqrt(head_dim)`)
   - `local_window_size` (int, default -1, meaning no sliding window;
     we support `-1` only in this tier — sliding-window attention is
     not required by Llama or Gemma, and if a future model needs it
     we add it as a new change)
   - `do_rotary` (int, default 1 — meaning RoPE is applied inside GQA;
     if 0, RoPE was already applied in a separate `RotaryEmbedding`
     node upstream and we skip step 4)
   - `rotary_interleaved` (int, default 0)

3. **Reshape Q, K, V** from `(B, Sq, num_heads * head_dim)` into
   `(B, num_heads, Sq, head_dim)` for Q and
   `(B, kv_num_heads, Sq, head_dim)` for K and V.

4. **Apply RoPE in-place to Q and K** (skipped if `do_rotary == 0`)
   using the shared `apply_rope_in_place` helper (see D6). Position
   IDs for the RoPE are computed on-the-fly as
   `position = past_Sk + i` for `i in 0..Sq`, per batch.

5. **Concatenate K, V with the past KV-cache** along the sequence
   axis to produce full keys `K_full` of shape
   `(B, kv_num_heads, past_Sk + Sq, head_dim)` and similarly `V_full`.
   The concatenation is an in-place append into the `past_key` /
   `past_value` tensors — see D8 on why those tensors must live in
   the outer scope. The *output* `present_key` / `present_value` are
   views into the updated past tensors.

6. **Grouped attention dispatch.** Since `num_heads >= kv_num_heads`,
   each KV head is shared across `num_heads / kv_num_heads` query
   heads. Rather than materialize a `num_heads`-sized K tensor by
   tiling, we compute the attention per KV group: for each of the
   `kv_num_heads` groups, run the shared SDPA helper over the
   `num_heads / kv_num_heads` query heads with that group's K and V.

7. **Call the shared SDPA helper** (see below and D5):
   ```text
   attn_out = scaled_dot_product_attention(
       q_group,      // (B, group_size, Sq, head_dim)
       k_group,      // (B, 1,          Sk, head_dim)
       v_group,      // (B, 1,          Sk, head_dim)
       causal_mask,  // on-the-fly, see step 8
       scale,        // attribute or default
   )
   ```
   The helper broadcasts `k_group` / `v_group` across the `group_size`
   query heads internally.

8. **Causal mask handling.** ORT exposes two mask modes: model-supplied
   (where an earlier node in the graph pre-computes a
   `(B, 1, Sq, Sk)` additive mask) and on-the-fly. SmallAIOS implements
   on-the-fly only: the SDPA helper walks the output tile and, for
   each `(q_idx, k_idx)` pair where `k_idx > q_idx + past_Sk`, writes
   `-inf` into the attention-score accumulator before the softmax.
   This costs one branch per element of the attention score tile,
   which is free next to the matmul itself.

9. **Reshape output** from `(B, num_heads, Sq, head_dim)` back to
   `(B, Sq, hidden)` and return it alongside the updated
   `present_key` / `present_value` (which are the same tensors as the
   input `past_key` / `past_value`, mutated in place).

**Internal helpers.**
- `scaled_dot_product_attention(q, k, v, scale, causal_mask_fn)` —
  shared with `MultiHeadAttention` (D5). Computes
  `softmax(Q @ K^T * scale + mask) @ V` per head with online-softmax
  numerics (max-stabilized to avoid fp32 overflow on long sequences).
  Takes a *function* for the mask rather than a tensor, so the caller
  can compute the mask on the fly without materializing it.
- `apply_rope_in_place(tensor, cos_cache, sin_cache, positions,
  interleaved)` — shared with standalone `RotaryEmbedding` (D6).
  Mutates the input tensor; returns nothing. The `positions`
  parameter is a `&[i64]` slice of length `Sq`, one entry per query
  position.

**KV-cache memory ownership.** The `past_key` / `past_value` tensors
are mutated in place. This requires two runtime guarantees:

1. The outer-graph executor must NOT alias the `past_key` /
   `past_value` tensor names — that is, no two live nodes in the
   current iteration may hold references to the same underlying
   buffer. This is already true in the Phase 2 `execute_graph`
   because tensors are looked up by name out of a `BTreeMap`, and
   GQA is the only writer to its own KV-cache names.
2. The tensors must be declared in the *outer* value map (the
   `Session`'s main scope), not in a sub-graph's inner value map.
   See D8 for why.

The mutation is safe because the Rust borrow checker sees the
in-place update as `&mut Tensor`, and `dispatch_node` in the executor
already holds an `&mut BTreeMap<String, Tensor>` for the duration of
a single node's execution.

### D5: `MultiHeadAttention` is the older, simpler fusion

**Decision.** `op_multi_head_attention` is approximately 250 LOC. It
reuses the `scaled_dot_product_attention` helper from D4.

**Differences from GroupQueryAttention:**
- No grouping. `num_heads == kv_num_heads` (implicit). Every query
  head has its own KV head.
- RoPE is *not* applied inside. DeepSeek-R1 factors RoPE out as a
  separate `RotaryEmbedding` node upstream of MHA, so the MHA op
  receives already-rotated Q and K.
- KV-cache handling is still supported (`past_key` / `past_value`
  inputs, same in-place concat semantics as D4).
- Supports both "packed QKV" (single input of shape
  `(B, Sq, 3*hidden)` that gets split) and "unpacked" (three separate
  inputs). DeepSeek uses unpacked.

**Inputs (per ORT contrib op spec).** See the ORT docs for the exact
list; the two important variants are the 3-input unpacked form (Q, K, V)
and the 1-input packed form. Both are implemented by the same function
with a branch at the top reading input count.

**Algorithm.** Identical to GroupQueryAttention steps 6-9 with
`group_size = 1`. The only difference is that the dispatcher does not
need the grouping loop — each query head has its own KV head, and
the shared SDPA helper is called once per head (or once with the
`num_heads` axis left intact and handled internally).

**Rationale for reuse.** The SDPA helper is the hot loop. Factoring
it out means GQA and MHA share the same optimized implementation,
the same cache-blocking strategy, and the same numerical behavior.
If we later add a flash-attention-style fast path, it lives in the
helper and both ops get it for free.

### D6: `RotaryEmbedding` standalone op

**Decision.** `op_rotary_embedding` is approximately 150 LOC. It
wraps the shared `apply_rope_in_place` helper with input/output
marshalling.

**Inputs (per ORT contrib op spec):**
- `input` — shape `(B, Sq, hidden)` or
  `(B, num_heads, Sq, head_dim)`. Both are valid; the op infers the
  layout from rank.
- `position_ids` — shape `(B, Sq)` or `(1, Sq)` (broadcast). Integer
  positions into the `cos_cache` / `sin_cache`.
- `cos_cache` — shape `(max_seq_len, head_dim/2)`
- `sin_cache` — shape `(max_seq_len, head_dim/2)`

**Attributes:**
- `interleaved` (int, default 0)
- `rotary_embedding_dim` (int, default 0 = full `head_dim`)
- `num_heads` (int, used only when input is rank-3 to reshape into
  rank-4 internally)
- `scale` (float, default 1.0 — for some exporter variants that
  pre-scale cos/sin)

**Interleaved vs non-interleaved.** The critical attribute. HuggingFace
and Llama use **non-interleaved** rotation: given a head of dim D,
the first D/2 elements pair with the second D/2 elements, and the
rotation is `(x[i], x[i+D/2]) → (cos*x[i] - sin*x[i+D/2], sin*x[i] + cos*x[i+D/2])`.
DeepSeek uses **interleaved** rotation: the pairing is
`(x[0], x[1]), (x[2], x[3]), ...` and the rotation applies to each
adjacent pair with the same cos/sin but a different indexing pattern:
`(x[2i], x[2i+1]) → (cos*x[2i] - sin*x[2i+1], sin*x[2i] + cos*x[2i+1])`.

Both must be supported. The shared `apply_rope_in_place` helper
takes the `interleaved` flag and branches on it in the hot loop.

**Output.** Same shape as input, with RoPE applied.

**Rationale.** DeepSeek's export pipeline produces a standalone
`RotaryEmbedding` node ahead of `MultiHeadAttention`. Llama and Gemma
fuse RoPE inside `GroupQueryAttention` (controlled by the `do_rotary`
attribute). Both styles must work. Factoring `apply_rope_in_place`
out as a shared helper from D4 means this op is almost trivial —
reshape, call helper, reshape back.

### D7: Op grouping and file layout

**Decision.** All five operators plus the two shared helpers go into
a single new file: `onnx-rt/src/ops/microsoft.rs`.

```text
onnx-rt/src/ops/
├── mod.rs                          # registers microsoft module
├── activations.rs
├── generative.rs                   # Phase 2 RMSNorm etc. (unchanged)
├── microsoft.rs                    # NEW — this change
│   ├── op_simplified_layer_normalization
│   ├── op_skip_simplified_layer_normalization
│   ├── op_group_query_attention
│   ├── op_multi_head_attention
│   ├── op_rotary_embedding
│   ├── scaled_dot_product_attention      (internal helper)
│   └── apply_rope_in_place               (internal helper)
└── ...
```

The dispatcher gets a new `dispatch_microsoft_fused(op_kind, inputs,
attrs)` helper in `onnx-rt/src/executor.rs` that matches on the
`MicrosoftFused` subset of `OpKind` and calls the corresponding
function from `ops::microsoft`.

**Rationale.** All five ops share the SDPA and RoPE helpers. Splitting
them across multiple files would require duplicating the helpers or
exposing them publicly across modules. A single `microsoft.rs` keeps
the helpers `pub(super)` and avoids indirection. The file will be
large (~1500 LOC with tests co-located) but it is the natural unit.

If `microsoft.rs` grows unwieldy in a future tier, it can be split
into a sub-directory `ops/microsoft/{mod.rs, attention.rs, norm.rs,
rotary.rs}` without breaking the public API.

### D8: KV-cache lifecycle interaction with the sub-graph executor

**Background.** Phase 2's sub-graph executor (`onnx-rt/src/sub_executor.rs`)
evaluates `Loop` bodies by allocating a fresh inner `BTreeMap<String,
Tensor>` value map each time the `Loop` is invoked at graph level. Per
the Phase 2 design document Q1 decision, the inner map is *reused and
cleared* between iterations rather than reallocated, but all tensor
entries living in that inner map are invalidated between iterations.
The only tensors that survive iteration-to-iteration are:

1. Loop-carried values (declared as `v_initial`/`v_final` by the ONNX
   Loop node)
2. Outer-referenced values (the `outer_refs` list, passed in by name
   from the outer scope at the start of each iteration)

**The problem.** A GroupQueryAttention node inside a `Loop` body
wants to mutate its `past_key` / `past_value` in place so that the
next iteration sees the extended KV-cache. If the cache tensors live
in the inner value map, they are invalidated at the end of each
iteration and the mutation is lost. The next iteration sees an empty
past cache and the model produces garbage.

**Decision.** KV-cache tensors MUST live in the outer-graph scope,
not in the inner sub-graph scope. They are passed into the `Loop`
body via the `outer_refs` mechanism (by name reference, not by
value-copy), and GQA mutates them in place through that outer
reference. The inner value map never owns a KV-cache tensor — it
only borrows one from the outer scope.

Concretely, this means the ONNX model graph structure looks like:

```text
outer_graph:
  Constant → past_key_init                 (zero tensor, (B, num_kv_heads, 0, head_dim))
  Constant → past_value_init
  Loop(M, cond, past_key_init, past_value_init, ...) ──┐
    └── body (inner_graph):                            │
          inputs:                                      │
            iter_num       (loop var)                  │
            cond_in        (loop var)                  │
            past_key       (loop-carried from outer)   │
            past_value     (loop-carried from outer)   │
          nodes:                                       │
            GroupQueryAttention(...,                   │
              past_key, past_value,                    │  ← mutates in-place
              ...)                                     │
          outputs:                                     │
            cond_out                                   │
            present_key  = past_key     (same buffer)  │
            present_value = past_value  (same buffer)  │
  outputs of Loop: final_past_key, final_past_value ───┘
```

The `past_key` / `past_value` names are **loop-carried values**, not
outer refs — meaning the sub-graph executor explicitly rotates them
iteration-to-iteration (iteration N's `present_key` becomes iteration
N+1's `past_key`). Loop-carried values are the one class of inner
tensor that is *not* cleared between iterations (see Phase 2 design
D3 on scope and value passing).

**Constraint on exporters.** This is already how ORT-exported Llama
and Gemma write their graphs: KV-cache tensors are always loop-carried,
never inner-local. SmallAIOS relies on this. If a future exporter
produced a graph where the KV-cache tensor lived in a pure inner
scope, the model would produce incorrect output on iteration 2+.
We do not detect this case at load time; we treat it as the
exporter's responsibility (consistent with how ORT treats it).

**Non-change to the sub-graph executor.** This decision requires zero
code changes to `onnx-rt/src/sub_executor.rs`. The loop-carried
value mechanism already works. We are documenting a contract, not
extending the executor.

**Implication for tests.** The end-to-end model-loading tests in
tasks section 10 validate this path by loading the canonical HF
exports (which use loop-carried KV-caches) and checking that iteration
2 sees the mutation from iteration 1. The unit tests for GQA in
tasks section 8 test the in-place mutation directly by calling
`op_group_query_attention` twice against the same `past_key` /
`past_value` buffers and asserting the second call sees the first
call's writes.

### D9: Validation — numerical match against Python ORT reference

**Decision.** The end-to-end tests in tasks section 10 load each of
the three model families via `Session::new_from_file()`, run a
single-token generation (or for DeepSeek f32, a single forward pass),
and compare the output against a Python `onnxruntime` reference
captured as golden data.

**Bounds.**
- **Llama-3.2-1B** — the canonical HF export is int4-weight-quantized
  but with int8 activations and f32 accumulators. The output logits
  are f32. Bound: top-1 token must match; max absolute error on the
  top-5 logits ≤ 1e-2.
- **Gemma 3 1b** — same as Llama. Bound: same as Llama.
- **DeepSeek-R1-Distill-Qwen-1.5B** — canonical HF export is f32
  end-to-end. Bound: max absolute error on the output hidden states
  ≤ 1e-3.

**Golden data.** The Python reference outputs are captured once,
committed to `onnx-rt/tests/fixtures/microsoft_fused_ops/` as raw
binary files (one per model), and loaded from disk in the test. The
capture script (`tools/capture-ort-reference.py`) is not part of this
change; it is checked in as an unversioned tool with a README in the
fixtures directory explaining how to regenerate the files against a
specific HF mirror checksum.

**Rationale.** Numerical matching against a reference implementation
is the only way to validate a vendor contrib op whose specification
is "the C++ source code of onnxruntime". Matching top-1 token on a
1-token generation exercises the full attention + norm + RoPE path
and is sensitive to any subtle bug in any of the five ops.

**Test performance.** Loading a 1B-parameter model from disk, even
for a 1-token generation, is expensive (seconds of wall clock). The
end-to-end tests are gated behind the `--ignored` test flag by
default and run in a separate CI job (`real-model-tests`) that is
advisory-only initially and promoted to a gate once it is stable.
The per-op unit tests in sections 3-8 stay in the default test suite
and cover correctness on small hand-crafted tensors.

## Alternatives Considered

### A1: Implement only GQA + the two norm ops, skip MHA and RotaryEmbedding

Skip DeepSeek. Ship support for Llama and Gemma only. MHA and
standalone RoPE are not needed.

**Rejected.** DeepSeek-R1-Distill-Qwen-1.5B is the single most
popular open-weight reasoning model in the 1.5B class and a reference
target for agent work. Leaving it out means the first LLM tier
shipping in SmallAIOS cannot load the one model that users would most
likely want to benchmark against. The marginal cost of MHA (~250
LOC reusing the SDPA helper) and standalone RoPE (~150 LOC reusing
`apply_rope_in_place`) is small because both reuse helpers that GQA
needs anyway.

### A2: Alias Microsoft ops into the StandardOnnx namespace

Rather than add a `Domain` enum, pretend these ops live in the
standard domain. `SimplifiedLayerNormalization` becomes just another
name for RMSNorm, etc.

**Rejected.** Loses provenance (see D1). Confuses tooling (the
coverage probe cannot distinguish "we support the standard ONNX op
set fully" from "we support an ORT-optimized flavor"). Fails on the
day a future standard ONNX opset adds a genuine
`SimplifiedLayerNormalization` with slightly different semantics.

### A3: Skip RoPE and rely on pre-rotated tensors being passed in

Some ONNX exports produce RoPE as a chain of standard ops
(`Cos`/`Sin`/`Concat`/`Mul`/`Add`). Force the HF exporter to emit
that path instead of the fused `RotaryEmbedding` op.

**Rejected.** This is a user-facing support burden (every user has
to learn and configure a non-default export flag), and the HF
`optimum-cli export onnx` default uses the fused path for a reason —
it is an order of magnitude fewer nodes in the graph and avoids
materializing the intermediate cos/sin broadcasts. Meeting users
where they are means supporting the fused op.

### A4: Implement a model-surgery pass at load time that rewrites Microsoft ops into standard ops

At `Session::new_from_file()` time, walk the graph and replace each
Microsoft op with an equivalent sub-graph of standard ops (e.g.,
`GroupQueryAttention` → `Transpose`+`MatMul`+`Softmax`+`MatMul`+
`Reshape`+...).

**Rejected.** This rewrites the graph structure invisibly, making
the profile output lie (the user sees "MatMul" rows for operators
that were actually GQA nodes in the source graph). It also adds a
new failure mode (the rewriter may produce an incorrect decomposition
for edge-case attribute combinations). The straightforward
implementation — a real fused op with real math — is clearer and
easier to debug.

## Open Questions

### Q1: Do we need `local_window_size` for sliding-window attention?

Gemma 3 uses sliding-window attention (window=4096) in some layers.
The coverage probe confirmed the 1b model works without SWA (either
the 1b variant uses dense attention throughout, or the ORT exporter
did not set `local_window_size`). If a larger Gemma variant is later
added, we need to support `local_window_size > -1` in GQA.

**Leaning.** Implement in a follow-up change if required. Document
in the GQA code that `local_window_size != -1` currently returns
`UnsupportedAttribute`. The coverage probe should be re-run against
Gemma 3 4b before claiming support for that model.

### Q2: Does int4 weight-quantization work through the existing int8 MatMul kernels?

Llama-3.2-1B and Gemma 3 1b use int4 weight quantization in practice
(the HF export is `awq`-style with int4 weights and int8
activations). The standard ONNX ops the model contains are the int8
variants we already support, so loading *should* work, but runtime
performance on the int4 weight tensors is untested.

**Leaning.** The end-to-end tests (task 10.1 and 10.2) answer this
empirically. If they pass, the int4 → int8 path works. If they fail,
we open a separate change for int4 kernel support and mark these
two models as deferred until that change lands.

### Q3: How big should the attention score working buffer be?

SDPA computes `Q @ K^T` into a `(B, num_heads, Sq, Sk)` buffer
before applying softmax. For `B=1, num_heads=32, Sq=1, Sk=4096` (a
typical decoding step on Llama-3.2-1B), this is 128 KB of f32 per
step. That is fine. For `B=1, num_heads=32, Sq=1024, Sk=4096`
(prefill), it is 128 MB. That is not fine for a 15 MB container.

**Leaning.** Tile the SDPA helper in the `Sq` dimension at 64 rows
per tile. This keeps the working buffer at <10 MB at all times. The
tile size is a compile-time constant initially; tunable in a
follow-up if profiling shows a hot spot.
