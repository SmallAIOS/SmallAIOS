## ADDED Requirements

### Requirement: GPU-resident tensor execution
The runtime SHALL provide an execution path where intermediate tensors stay in GPU memory between operators for the entire forward pass, with no per-operator host↔device transfer.

#### Scenario: Forward pass with GPU residency
- **WHEN** `Session::run()` is invoked on a GPU-resident session with input tensors
- **THEN** the executor SHALL transfer inputs host→device once at the start
- **AND** all intermediate tensors SHALL be `DeviceBuffer`s that remain in GPU VRAM
- **AND** outputs SHALL be transferred device→host once at the end (or kept on device for the next call)
- **AND** SHALL NOT call `cudaMemcpy(DeviceToHost)` or `cudaMemcpy(HostToDevice)` between operators within a single forward pass

#### Scenario: GPU-only operator dispatch
- **WHEN** an operator is encountered during a GPU-resident forward pass
- **THEN** the executor SHALL dispatch to a GPU implementation of that operator
- **AND** SHALL fail with a clear error if no GPU implementation exists for a required operator (no silent CPU fallback during the forward pass)

### Requirement: GPU-resident KV cache
Sessions backed by GPU-resident execution SHALL maintain a per-layer KV cache in GPU memory that persists across `Session::run()` calls.

#### Scenario: KV cache initialization
- **WHEN** a GPU-resident session is created
- **THEN** the cache SHALL be allocated lazily on the first `Session::run()` call
- **AND** sized according to `max_position_embeddings`, `num_key_value_heads`, and `head_dim` from the model config

#### Scenario: KV cache append on each token
- **WHEN** `Session::run()` is called with a new query token
- **THEN** the new key and value tensors for that token SHALL be appended to the per-layer cache (in-place on GPU)
- **AND** the attention operator SHALL read from the full cached K and V history

#### Scenario: KV cache reset
- **WHEN** the caller invokes `Session::reset_kv_cache()` (or starts a new generation session)
- **THEN** the cache SHALL be cleared without freeing the underlying `DeviceBuffer` allocation
- **AND** subsequent calls SHALL begin appending from position 0

### Requirement: GPU-required session validation
Sessions that require GPU execution SHALL fail fast at creation if no GPU is available.

#### Scenario: GPU-required session without GPU
- **WHEN** a GPU-required session is created and no `CudaRuntime` is available
- **THEN** the creation SHALL return an error indicating the model requires a GPU
- **AND** SHALL NOT silently fall back to CPU execution
- **AND** the error message SHALL state which model required the GPU

#### Scenario: GPU-required session at boot
- **WHEN** the container loads a safetensors model directory at boot and no `CudaRuntime` is available
- **THEN** the container SHALL log an error identifying the model
- **AND** SHALL NOT add the model to the available sessions
- **AND** SHALL continue serving any other models that don't require GPU
