## MODIFIED Requirements

### Requirement: GPU-only operator dispatch
The GPU-resident executor SHALL dispatch every operator in a forward pass to a GPU implementation, with no silent host fallback, and SHALL support the full Gemma / Llama / Qwen transformer operator set.

#### Scenario: Supported operator dispatch
- **WHEN** an operator is encountered during a GPU-resident forward pass and the operator kind is one of `MatMul`, `Gemm`, `MatMulInteger`, `Conv`, `Gather`, `Add`, `Mul`, `Silu`, `RMSNormalization`, `RotaryEmbedding`, or `GroupQueryAttention`
- **THEN** `dispatch_gpu_node` SHALL route the node to the corresponding GPU implementation
- **AND** SHALL produce a `DeviceTensor` output that is fed to the next operator without host transfer

#### Scenario: Unsupported operator fail-fast
- **WHEN** an operator is encountered during a GPU-resident forward pass and no GPU implementation exists
- **THEN** the dispatcher SHALL return a `CudaError` naming the operator kind and the node name
- **AND** SHALL NOT call `cudaMemcpy(DeviceToHost)` on the operator's inputs
- **AND** SHALL NOT invoke a CPU implementation as a silent fallback

#### Scenario: Forward pass completes through the dispatcher
- **WHEN** a Gemma graph is executed via `execute_graph_gpu_with_weights` end-to-end
- **THEN** every node SHALL be dispatched via one of the supported GPU paths above
- **AND** the final output tensor SHALL contain logits of shape `[batch, seq_len, vocab_size]` as a `DeviceTensor`
- **AND** no intermediate tensor SHALL have been copied to host memory

### Requirement: KV cache threading through the GPU dispatcher
The GPU-resident executor SHALL thread an optional mutable reference to the session's `GpuKvCache` through to operators that require it.

#### Scenario: Attention dispatch with KV cache
- **WHEN** a `GroupQueryAttention` node is dispatched and the executor was called with `Some(&mut GpuKvCache)`
- **THEN** the dispatcher SHALL pass the mutable cache reference into the GQA wrapper
- **AND** the GQA wrapper SHALL append the current K/V and then read the full history via `KvView`
- **AND** the cache state SHALL be visible to the next `Session::run()` call

#### Scenario: Non-attention dispatch with KV cache present
- **WHEN** a non-attention node (e.g. `MatMul`, `Add`, `RMSNormalization`) is dispatched and the executor was called with `Some(&mut GpuKvCache)`
- **THEN** the dispatcher SHALL thread the mutable borrow through without touching the cache
- **AND** the cache SHALL remain unchanged after the node executes

#### Scenario: Executor called without a KV cache
- **WHEN** `execute_graph_gpu` or `execute_graph_gpu_with_weights` is called with `None` for the cache parameter and the graph contains no `GroupQueryAttention` nodes
- **THEN** the forward pass SHALL complete successfully
- **AND** the executor SHALL NOT require a cache to be present

#### Scenario: Attention without a cache on a test-only path
- **WHEN** `GroupQueryAttention` is dispatched and the executor was called with `None` for the cache parameter
- **THEN** the GQA wrapper SHALL compute attention using only the current step's K and V with `seq_len_kv = seq_len_q`
- **AND** this path SHALL be used only in unit tests, not in production `Session::run()` flows

### Requirement: Session-level KV cache integration
`Session::run_safetensors` SHALL lock its `Arc<Mutex<GpuKvCache>>` and thread the guard into the executor as a mutable reference for the duration of the forward pass.

#### Scenario: Session run acquires the cache mutex
- **WHEN** `Session::run_safetensors` is invoked
- **THEN** it SHALL lock the `Arc<Mutex<GpuKvCache>>` field before calling `execute_graph_gpu_with_weights`
- **AND** it SHALL pass `Some(&mut *guard)` as the cache parameter
- **AND** SHALL release the mutex when the forward pass returns

#### Scenario: Concurrent Session run is serialized
- **WHEN** two tasks attempt to call `Session::run_safetensors` on the same session concurrently
- **THEN** the second task SHALL block on the mutex until the first completes
- **AND** KV cache state SHALL remain consistent across both invocations
