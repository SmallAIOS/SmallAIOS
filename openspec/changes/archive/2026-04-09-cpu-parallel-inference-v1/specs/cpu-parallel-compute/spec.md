## ADDED Requirements

### Requirement: CorePool Thread Pool Abstraction
The ONNX runtime SHALL provide a `CorePool` abstraction that distributes parallel work across CPU cores in both kernel mode and container mode.

#### Scenario: Container mode uses scoped threads
- **WHEN** the runtime is compiled for container mode (with `std`)
- **THEN** `CorePool` MUST use `std::thread::scope` to spawn parallel workers
- **AND** all workers MUST complete before `parallel_for` returns
- **AND** no heap allocation for thread handles SHALL be required

#### Scenario: Kernel mode uses scheduler run queues
- **WHEN** the runtime is compiled for kernel mode (`no_std`)
- **THEN** `CorePool` MUST post sub-tasks to per-core `RunQueue` entries as INFERENCE-class tasks
- **AND** MUST wait for all sub-tasks to complete via atomic completion counter
- **AND** MUST yield cooperatively while waiting

#### Scenario: Configurable core count
- **WHEN** a `Session` is created with `max_threads` parameter
- **THEN** the `CorePool` MUST limit parallelism to the specified number of threads
- **AND** if `max_threads` is 1, operators MUST execute sequentially with no fork/join overhead

### Requirement: Parallel GEMM
The ONNX runtime SHALL parallelize GEMM (General Matrix Multiply) by distributing tile row bands across cores.

#### Scenario: Parallel GEMM above threshold
- **WHEN** a MatMul or Gemm operator is dispatched with M × K × N > 65,536
- **THEN** the runtime MUST split the row dimension across available cores
- **AND** each core MUST compute its assigned tile rows using the existing `gemm_tile` + `micro_kernel_8x8`
- **AND** the output MUST be numerically identical to sequential GEMM

#### Scenario: Sequential GEMM below threshold
- **WHEN** a MatMul or Gemm operator is dispatched with M × K × N <= 65,536
- **THEN** the runtime MUST execute GEMM sequentially on a single core
- **AND** no thread pool overhead SHALL be incurred

### Requirement: Parallel Convolution
The ONNX runtime SHALL parallelize convolution by distributing output channels across cores.

#### Scenario: Parallel Conv above threshold
- **WHEN** a Conv operator is dispatched with output_channels × H × W > 16,384
- **THEN** the runtime MUST split output channels across available cores
- **AND** each core MUST compute its assigned output feature maps independently
- **AND** the output MUST be numerically identical to sequential Conv

### Requirement: Parallel Element-wise Operations
The ONNX runtime SHALL parallelize element-wise operators by splitting the data range across cores.

#### Scenario: Parallel element-wise above threshold
- **WHEN** an element-wise operator (Add, Sub, Mul, Div, Relu, Sigmoid, Tanh, Clip) is dispatched with num_elements > 32,768
- **THEN** the runtime MUST split the flat data array into equal chunks across cores
- **AND** each core MUST compute its chunk independently

### Requirement: Parallel Reduction Operations
The ONNX runtime SHALL parallelize reduction operators using a two-phase parallel reduction.

#### Scenario: Parallel reduction above threshold
- **WHEN** a reduction operator (ReduceMean, ReduceSum) is dispatched with num_elements > 65,536
- **THEN** phase 1 MUST compute partial results on each core over its chunk
- **AND** phase 2 MUST merge partial results on the main thread
- **AND** the output MUST be numerically equivalent to sequential reduction (within f32 accumulation tolerance)

#### Scenario: Parallel Softmax
- **WHEN** a Softmax operator is dispatched with num_elements > 65,536
- **THEN** the runtime MUST compute parallel max, parallel exp+sum, and parallel normalize
- **AND** numerical stability (subtract max before exp) MUST be preserved

### Requirement: Auto-Tuning Parallelism Threshold
The ONNX runtime SHALL automatically decide whether to parallelize each operator based on input size.

#### Scenario: Threshold check before parallelization
- **WHEN** an operator is dispatched
- **THEN** the runtime MUST check the input tensor size against the operator's parallelism threshold
- **AND** MUST execute sequentially if below threshold

#### Scenario: Configurable thresholds
- **WHEN** a `Session` is created with custom `parallel_thresholds` configuration
- **THEN** the runtime MUST use the specified thresholds instead of defaults
- **AND** default thresholds MUST be: GEMM 65536, Conv 16384, element-wise 32768, reduction 65536
