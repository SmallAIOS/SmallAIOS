## ADDED Requirements

### Requirement: Batched Inference API
The runtime SHALL expose a `Session::run_batched(inputs: &[InferenceInput], batch_size: usize) -> Result<Vec<InferenceOutput>, SessionError>` method that accepts `N` inputs per named input port and produces `N` outputs per named output port. The method SHALL stack inputs along axis 0 (the batch dimension) before dispatching the existing executor exactly once, and SHALL unstack outputs back into per-image tensors before returning.

#### Scenario: Static batch size returns one output per input
- **WHEN** a session is configured with `BatchPolicy::Static(4)` and `run_batched` is called with exactly 4 inputs (per named input port) of identical shape and dtype
- **THEN** the runtime MUST stack the 4 inputs along axis 0 to produce a single tensor of shape `[4, ...]`
- **AND** MUST dispatch the executor once on the stacked tensor
- **AND** MUST unstack the resulting `[4, ...]` output along axis 0 into 4 separate `InferenceOutput`s
- **AND** the returned `Vec<InferenceOutput>` MUST have length `4 * num_output_names`, ordered by image then by name

#### Scenario: Single-input run is a degenerate batched call
- **WHEN** `Session::run` is called with a single `InferenceInput`
- **THEN** the runtime MUST execute through the same code path as `run_batched(inputs, 1)`
- **AND** the resulting output MUST be byte-for-byte identical to a pre-`dynamic-batching-v1` `Session::run` call

#### Scenario: Output tensors are unstacked correctly
- **WHEN** the underlying graph produces a single batched output of shape `[N, C]`
- **THEN** the unstacker MUST return `N` `InferenceOutput`s, each with tensor shape `[C]`
- **AND** the byte content of output `i` MUST equal the slice `[i*C..(i+1)*C]` of the batched tensor
- **AND** the dtype of each unstacked output MUST equal the dtype of the batched output

### Requirement: Batch Policy Enforcement
The runtime SHALL enforce a `BatchPolicy` (set on `SessionConfig::batch_policy`) when `run_batched` is invoked. The policy SHALL determine which batch sizes are accepted and whether undersized batches are padded.

#### Scenario: Disabled policy rejects run_batched
- **WHEN** `SessionConfig::batch_policy` is `BatchPolicy::Disabled` (the default) and `run_batched` is called with any input count
- **THEN** the runtime MUST return `SessionError::BatchPolicyViolation` with a message indicating that batching is disabled for this Session
- **AND** MUST NOT execute the underlying graph

#### Scenario: Static policy enforces exact batch size
- **WHEN** `BatchPolicy::Static(N)` is set and `run_batched` is called with `M` inputs where `M != N`
- **THEN** the runtime MUST return `SessionError::BatchPolicyViolation` naming `N` (expected) and `M` (received)

#### Scenario: Dynamic policy without padding accepts variable batch
- **WHEN** `BatchPolicy::Dynamic { max: N, pad: false }` is set and `run_batched` is called with `K` inputs where `1 <= K <= N`
- **THEN** the runtime MUST execute the underlying graph at the actual batch size `K`
- **AND** MUST return exactly `K * num_output_names` outputs

#### Scenario: Dynamic policy with padding repeats the last input to reach max
- **WHEN** `BatchPolicy::Dynamic { max: N, pad: true }` is set and `run_batched` is called with `K` inputs where `1 <= K < N`
- **THEN** the runtime MUST internally append `N - K` copies of the last input to reach `N`
- **AND** MUST execute the underlying graph at batch size `N`
- **AND** MUST return only the first `K * num_output_names` outputs to the caller, discarding the padded outputs

#### Scenario: Exceeding max batch size errors
- **WHEN** `BatchPolicy::Dynamic { max: N, pad: _ }` is set and `run_batched` is called with `M > N` inputs
- **THEN** the runtime MUST return `SessionError::BatchPolicyViolation` with a message naming `N` (max) and `M` (received)

### Requirement: Batched Input Validation
The runtime SHALL validate that all inputs in a single `run_batched` call share the same per-image shape and dtype, and that all input names in the batch are present.

#### Scenario: Inputs with different shapes are rejected
- **WHEN** `run_batched` is called with two inputs that have different shapes (apart from the batch axis being absent or differing)
- **THEN** the runtime MUST return `SessionError::BatchShapeMismatch` naming the offending input index and the mismatched shape pair
- **AND** MUST NOT execute the underlying graph

#### Scenario: Inputs with different dtypes are rejected
- **WHEN** `run_batched` is called with two inputs of the same name but different dtypes
- **THEN** the runtime MUST return `SessionError::BatchShapeMismatch` indicating dtype mismatch

#### Scenario: Empty batch is rejected
- **WHEN** `run_batched` is called with zero inputs
- **THEN** the runtime MUST return `SessionError::BatchEmpty`

#### Scenario: Missing input name is rejected
- **WHEN** the underlying model expects input named `x` and `run_batched` is called without any input bound to that name
- **THEN** the runtime MUST return `SessionError::InvalidInput("x")`

### Requirement: Throughput Targets at Batch Size
On DGX Spark with `GpuResidency::Hybrid`, throughput in images-per-second SHALL scale near-linearly with batch size up to the GPU's compute saturation point. Specifically:

#### Scenario: Throughput improves at B=4 vs B=1
- **WHEN** the `bench_resnet50_throughput_b4` benchmark is run with `BatchPolicy::Static(4)` and hybrid mode active
- **THEN** the measured images-per-second MUST be at least 3.5× the B=1 baseline (`bench_resnet50_throughput_b1`)

#### Scenario: Throughput improves at B=16 vs B=1
- **WHEN** the `bench_resnet50_throughput_b16` benchmark is run with `BatchPolicy::Static(16)`
- **THEN** the measured images-per-second MUST be at least 10× the B=1 baseline

#### Scenario: Throughput improves at B=64 vs B=1
- **WHEN** the `bench_resnet50_throughput_b64` benchmark is run with `BatchPolicy::Static(64)`
- **THEN** the measured images-per-second MUST be at least 20× the B=1 baseline

#### Scenario: Single-request latency does not regress
- **WHEN** `Session::run` (the B=1 entry point) is called on a session with `BatchPolicy::Disabled`
- **THEN** the measured latency MUST NOT exceed the pre-`dynamic-batching-v1` `bench_resnet50_cpu_vs_gpu_hybrid` baseline by more than 5%
