# Delta for coverage-benchmarks

## ADDED Requirements

### Requirement: ONNX Operator Benchmarks
The bench crate SHALL include benchmarks for all implemented ONNX operators measuring throughput.

#### Scenario: MatMul operator benchmark
- GIVEN the MatMul operator from the onnx-rt crate
- WHEN benchmark runs with matrices of size 64x64, 256x256, and 1024x1024
- THEN the benchmark MUST report operations per second for each size
- AND results MUST be stored in a machine-readable format (JSON or CSV)

#### Scenario: All operator benchmarks
- GIVEN each implemented ONNX operator (MatMul, Conv, Relu, Sigmoid, Softmax, Gemm)
- WHEN the benchmark suite runs
- THEN each operator MUST have at least one benchmark with a representative input size
- AND each benchmark MUST run for enough iterations to produce stable results (coefficient of variation < 5%)

### Requirement: Cryptography Benchmarks
The bench crate SHALL include benchmarks for all cryptographic operations.

#### Scenario: Hash throughput benchmark
- GIVEN the SHA-3-256 implementation from the security crate
- WHEN hashing data blocks of 64B, 1KB, 64KB, and 1MB
- THEN the benchmark MUST report throughput in bytes per second for each block size

#### Scenario: Symmetric encryption benchmark
- GIVEN the AES-256-GCM implementation from the security crate
- WHEN encrypting and decrypting data blocks of 64B, 1KB, 64KB, and 1MB
- THEN the benchmark MUST report throughput in bytes per second for each block size

#### Scenario: Post-quantum key exchange benchmark
- GIVEN the ML-KEM-768 implementation from the security crate
- WHEN running key generation, encapsulation, and decapsulation
- THEN the benchmark MUST report operations per second for each operation

#### Scenario: Digital signature benchmarks
- GIVEN ML-DSA-65 and Ed25519 implementations from the security crate
- WHEN running key generation, signing, and verification
- THEN the benchmark MUST report operations per second for each operation and algorithm

### Requirement: Network Stack Benchmarks
The bench crate SHALL include benchmarks for packet processing throughput.

#### Scenario: TCP packet parse throughput
- GIVEN the TCP packet parser from the net crate
- WHEN parsing pre-constructed valid TCP packets in a tight loop
- THEN the benchmark MUST report packets parsed per second

#### Scenario: Ethernet frame processing
- GIVEN the Ethernet frame parser from the net crate
- WHEN processing pre-constructed valid Ethernet frames
- THEN the benchmark MUST report frames processed per second

### Requirement: IPC Message Benchmarks
The bench crate SHALL include benchmarks for IPC pub/sub message throughput and latency.

#### Scenario: IPC publish/subscribe throughput
- GIVEN a configured IPC pub/sub channel
- WHEN messages of 64B, 1KB, and 64KB are published and received
- THEN the benchmark MUST report messages per second for each message size
- AND the benchmark MUST report end-to-end latency in nanoseconds

### Requirement: Memory Allocator Benchmarks
The bench crate SHALL include benchmarks for kernel memory allocation performance.

#### Scenario: Allocation throughput
- GIVEN the kernel memory allocator
- WHEN allocating and freeing blocks of 64B, 4KB, 64KB, and 1MB in sequence
- THEN the benchmark MUST report allocations per second for each block size

#### Scenario: Fragmentation measurement
- GIVEN the kernel memory allocator
- WHEN performing a mixed workload of random-sized allocations and frees
- THEN the benchmark MUST report the fragmentation ratio (total free / largest contiguous free)

### Requirement: Benchmark Regression Detection
The benchmark suite SHALL detect performance regressions by comparing against stored baselines.

#### Scenario: Regression detection threshold
- GIVEN a stored baseline result for each benchmark
- WHEN the current benchmark run completes
- THEN any benchmark that is more than 10% slower than its baseline MUST be flagged as a regression
- AND the regression report MUST identify the benchmark name, baseline value, current value, and percentage change

#### Scenario: Baseline update
- GIVEN a new benchmark run that should become the new baseline
- WHEN the baseline update command is executed (make bench-update-baseline)
- THEN the baseline file MUST be overwritten with the current results
- AND the baseline file MUST be in a committed, version-controlled format

### Requirement: Benchmark Framework Compatibility
The bench crate SHALL use criterion for host-targeted benchmarks.

#### Scenario: Criterion integration
- GIVEN benchmark source files in the bench crate
- WHEN cargo bench is executed on the host
- THEN criterion MUST produce HTML reports in target/criterion/
- AND JSON results MUST be written for machine consumption
- AND the bench crate MUST compile with std on the host target
