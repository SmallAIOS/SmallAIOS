## Why

SmallAIOS v0.2.1 has 29 working ONNX operators (Tier 1 — arithmetic, activations, shape, normalization, pooling, reduction). This covers MLPs and basic CNNs but excludes:

- **Recurrent networks** — LSTM, GRU (used in audio, time-series, classic NLP)
- **Transformer ops** — `MatMul + Softmax + scaled dot-product attention`, layer norm with proper dims, embedding gather + positional encoding (the basic blocks for any transformer-derived model)
- **Quantized inference** — INT8 ops (smaller models, faster inference, edge deployment)
- **More activations** — `Erf` (for GELU), `Pow`, `Sqrt`, `Exp`, `Log`, `Where`, `Equal`, `Greater`, `Less` (logical/comparison ops needed by attention masking)

This change adds the second tier of operators needed to run real transformer models (BERT, GPT-2-small, Whisper-tiny) and quantized variants of common CNNs.

## What Changes

### Tier 2A: Math primitives (needed by other ops)
- `op_pow` — element-wise power
- `op_sqrt` — element-wise square root
- `op_exp` — element-wise exponential
- `op_log` — element-wise natural log
- `op_erf` — error function (for GELU)
- `op_neg` — unary negation
- `op_abs` — absolute value
- `op_floor`, `op_ceil`, `op_round`

### Tier 2B: Comparison and selection
- `op_equal`, `op_not_equal`, `op_less`, `op_less_or_equal`, `op_greater`, `op_greater_or_equal`
- `op_where` — element-wise select based on condition
- `op_min`, `op_max` — element-wise min/max with broadcasting
- `op_not` — logical not

### Tier 2C: Composite activations
- `op_gelu` — Gaussian Error Linear Unit (uses Erf)
- `op_leaky_relu`
- `op_elu` — Exponential Linear Unit
- `op_swish` / `op_silu` — Sigmoid-weighted Linear Unit

### Tier 2D: Recurrent
- `op_lstm` — Long Short-Term Memory
- `op_gru` — Gated Recurrent Unit
- `op_rnn` — Vanilla RNN

### Tier 2E: Transformer building blocks
- `op_einsum` — for general tensor contractions (attention uses this)
- `op_scaled_dot_product_attention` — fused attention op (newer ONNX opset)
- `op_split` — split tensor along axis (used in QKV projection)
- `op_expand` — broadcast a tensor to a target shape
- `op_tile` — repeat a tensor
- `op_one_hot` — one-hot encoding for classification

### Tier 2F: Quantized (INT8)
- `op_quantize_linear` — float32 → int8 with scale and zero point
- `op_dequantize_linear` — int8 → float32
- `op_qlinear_matmul` — INT8 matrix multiply (for quantized models)
- `op_qlinear_conv` — INT8 convolution

## Capabilities

### New Capabilities
- `onnx-recurrent-operators`: LSTM, GRU, RNN with bidirectional support
- `onnx-transformer-operators`: Attention building blocks (einsum, split, expand, tile, scaled dot-product)
- `onnx-quantized-operators`: INT8 quantize/dequantize and quantized matmul/conv

### Modified Capabilities
- `onnx-cpu-execution`: Add ~30 new operators to the dispatch table
- `onnx-runtime`: Update OpKind enum and registry

## Impact

- **Code:** ~2,000 lines of new operator implementations in `onnx-rt/src/operators.rs`
- **OpKind:** ~30 new variants
- **Dispatch:** ~30 new arms in executor `dispatch_*` helpers
- **Tests:** Each operator gets unit tests (target: 60+ new tests)
- **Models unlocked:** BERT-base, GPT-2-small, MobileNet-v3 (quantized), Whisper-tiny, ResNet-50 quantized
- **No new dependencies:** All in pure Rust no_std + alloc
