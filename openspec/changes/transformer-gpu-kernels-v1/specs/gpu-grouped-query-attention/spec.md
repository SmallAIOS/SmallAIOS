## ADDED Requirements

### Requirement: GPU grouped-query attention via decomposed cuBLAS
The runtime SHALL provide a GPU implementation of `GroupQueryAttention` that composes `cublasGemmStridedBatchedEx` calls with custom softmax and head-expansion kernels, producing attention output in device memory.

#### Scenario: Standard MHA (num_kv_heads == num_attention_heads)
- **WHEN** `GroupQueryAttention` is dispatched with `num_kv_heads` equal to `num_attention_heads`
- **THEN** the KV expansion step SHALL be skipped
- **AND** `cublasGemmStridedBatchedEx` SHALL be invoked with `batchCount = num_attention_heads` for both QK^T and softmax·V
- **AND** the result SHALL match a CPU reference within the dtype's tolerance band (1e-3 F32 / 1e-2 BF16)

#### Scenario: GQA with num_kv_heads less than num_attention_heads
- **WHEN** `GroupQueryAttention` is dispatched with `num_kv_heads < num_attention_heads` (e.g. Gemma 4: 16 kv heads vs 32 attention heads)
- **THEN** the `gqa_kv_expand` kernel SHALL replicate each KV head `num_attention_heads / num_kv_heads` times along the head axis
- **AND** subsequent GEMM calls SHALL use the expanded K and V tensors with `batchCount = num_attention_heads`
- **AND** the numerical result SHALL match the CPU GQA reference

#### Scenario: Causal masking
- **WHEN** `GroupQueryAttention` is dispatched with a global (non-sliding-window) layer and `seq_len_q > 1`
- **THEN** the masked-softmax kernel SHALL set attention scores at positions `j > i` to `-inf` before the softmax reduction
- **AND** the resulting softmax row SHALL sum to 1.0 within 1e-5

#### Scenario: Sliding window masking
- **WHEN** `GroupQueryAttention` is dispatched with a sliding-window layer and a configured window size
- **THEN** the masked-softmax kernel SHALL additionally mask positions `j < i - window` to `-inf`
- **AND** only positions in `[max(0, i - window), i]` SHALL contribute to the softmax output
- **AND** the softmax row SHALL still sum to 1.0 within 1e-5

#### Scenario: KV cache append per invocation
- **WHEN** `GroupQueryAttention` is dispatched with `Some(&mut GpuKvCache)` for layer `L`
- **THEN** the wrapper SHALL first call `GpuKvCache::append(L, new_k, new_v, current_position)` to write the current step's K and V into the per-layer cache
- **AND** the append SHALL occur before any GEMM call

#### Scenario: KV cache view per invocation
- **WHEN** `GroupQueryAttention` is dispatched with a cache reference after the append step
- **THEN** the wrapper SHALL call `GpuKvCache::view(L, current_position + 1)` to obtain a `KvView` spanning the full cached history
- **AND** the `KvView` pointers SHALL be used as the K and V operands of the Q·K^T and softmax·V GEMM calls
- **AND** the `seq_len_kv` dimension SHALL equal `current_position + 1` for global layers or `min(current_position + 1, window)` for sliding-window layers

#### Scenario: Zero-length KV cache on first token
- **WHEN** `GroupQueryAttention` is dispatched for the first token of a generation session (`current_position == 0`)
- **THEN** the append step SHALL write `(K_0, V_0)` into the cache
- **AND** the view SHALL return a `KvView` with `seq_len_kv = 1`
- **AND** the attention output SHALL equal a direct single-token self-attention result

#### Scenario: Numerical correctness vs CPU reference
- **WHEN** `gpu_gqa` is run on a small synthetic input (e.g. `batch=1`, `seq_len_q=4`, `num_heads=2`, `head_dim=16`)
- **THEN** the output SHALL match the CPU `GroupQueryAttention` implementation in `ops/microsoft.rs` within 1e-2 absolute tolerance for BF16 and 1e-3 for F32

### Requirement: Attention intermediate workspace cap
The runtime SHALL cap the attention score scratch buffer size at session creation time and fail fast when a request exceeds the cap.

#### Scenario: Scratch buffer allocation at session init
- **WHEN** a GPU-resident session is created from a safetensors model
- **THEN** the runtime SHALL allocate an attention scratch `DeviceBuffer` sized by `num_attention_heads * max_attention_len * max_attention_len * sizeof(f32)` where `max_attention_len` is `sliding_window` for local-only layers or a configurable prefill limit for global layers
- **AND** subsequent `gpu_gqa` invocations SHALL reuse this buffer without reallocation

#### Scenario: Scratch exceeded
- **WHEN** a request's `seq_len_kv` at a global attention layer exceeds the configured prefill limit
- **THEN** `gpu_gqa` SHALL return a `CudaError` indicating the scratch cap was exceeded
- **AND** SHALL NOT launch any GEMM
- **AND** the error message SHALL name the layer and the limit

### Requirement: GroupQueryAttention fails fast on unsupported configurations
The GQA kernel SHALL reject inputs it cannot handle.

#### Scenario: Unsupported dtype
- **WHEN** `GroupQueryAttention` is dispatched with a Q, K, or V tensor whose dtype is not BF16 or F32
- **THEN** the dispatcher SHALL return a `CudaError` naming the rejected dtype
- **AND** SHALL NOT launch any kernel

#### Scenario: Non-divisible head grouping
- **WHEN** `GroupQueryAttention` is dispatched with `num_attention_heads % num_kv_heads != 0`
- **THEN** the dispatcher SHALL return a `CudaError` naming the invalid head configuration
- **AND** SHALL NOT launch the KV expansion kernel
