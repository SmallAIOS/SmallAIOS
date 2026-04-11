# Delta for Benchmark Infrastructure

## ADDED Requirements

### Requirement: Cold Start Measurement
The benchmark infrastructure SHALL measure cold start latency defined as the time from power-on or reset to the first inference result returned.

#### Scenario: Measure SmallAIOS cold start
- WHEN a SmallAIOS instance is booted from a power-off or reset state with a pre-configured model
- THEN the benchmark MUST record the elapsed time from reset to the first inference result
- AND the measurement MUST include boot, model loading, and first inference execution

#### Scenario: Measure Linux baseline cold start
- WHEN a Linux baseline (bare metal, Docker, or K8s/K3s) is started with the same model
- THEN the benchmark MUST record the elapsed time from process start (or container start, or pod scheduled) to the first inference result
- AND the measurement methodology MUST be documented to ensure fair comparison

#### Scenario: Cold start statistical validity
- WHEN cold start measurements are collected
- THEN each configuration MUST be measured at least 10 times
- AND the report MUST include mean, median, minimum, maximum, and standard deviation

### Requirement: Warm Inference Latency
The benchmark infrastructure SHALL measure warm inference latency at p50, p99, and p999 percentiles over a minimum of N=1000 runs.

#### Scenario: Collect latency samples
- WHEN the benchmark runs warm inference latency tests
- THEN it MUST execute at least 1000 inference requests after a warmup period of at least 100 requests
- AND each request's end-to-end latency MUST be recorded individually with microsecond precision

#### Scenario: Report percentile latencies
- WHEN latency samples have been collected for a given configuration
- THEN the report MUST include p50, p99, and p999 percentile values
- AND the report MUST include the total number of samples and the warmup count

#### Scenario: Compare SmallAIOS against Linux baselines
- WHEN latency results are available for both SmallAIOS and Linux baselines on the same hardware and model
- THEN the report MUST present side-by-side percentile comparisons
- AND the report MUST calculate the relative difference (percentage) for each percentile

### Requirement: Throughput Measurement
The benchmark infrastructure SHALL measure maximum inference throughput in inferences per second at batch sizes 1, 4, 16, and 64.

#### Scenario: Measure throughput at each batch size
- WHEN the benchmark runs throughput tests for a given model and hardware target
- THEN it MUST measure sustained inferences per second at batch sizes 1, 4, 16, and 64
- AND each throughput measurement MUST run for at least 60 seconds to reach steady state

#### Scenario: Report throughput scaling
- WHEN throughput results are available across batch sizes
- THEN the report MUST include a table showing inferences/sec for each batch size
- AND the report MUST note whether throughput scales linearly, sub-linearly, or reaches a plateau

### Requirement: Jitter Measurement
The benchmark infrastructure SHALL measure inference latency jitter including standard deviation and maximum deviation from the mean.

#### Scenario: Calculate jitter statistics
- WHEN warm inference latency samples have been collected (N >= 1000)
- THEN the benchmark MUST compute the standard deviation of inference latency
- AND the benchmark MUST compute the maximum absolute deviation from the mean latency

#### Scenario: Compare jitter between SmallAIOS and Linux
- WHEN jitter results are available for both SmallAIOS and Linux baselines
- THEN the report MUST present side-by-side standard deviation and max deviation values
- AND lower jitter MUST be highlighted as an indicator of better real-time determinism

### Requirement: Memory Footprint Measurement
The benchmark infrastructure SHALL measure peak memory usage in two configurations: OS + runtime (no model loaded) and OS + runtime + model.

#### Scenario: Measure baseline memory footprint
- WHEN the benchmark measures memory footprint with no model loaded
- THEN it MUST report peak RSS (or equivalent) for SmallAIOS and each Linux baseline
- AND for SmallAIOS the measurement MUST include kernel, allocator metadata, and runtime overhead

#### Scenario: Measure loaded model memory footprint
- WHEN the benchmark measures memory footprint with a model loaded and ready for inference
- THEN it MUST report peak RSS (or equivalent) for SmallAIOS and each Linux baseline
- AND the report MUST include the delta between no-model and with-model footprints to isolate model memory

#### Scenario: Memory footprint comparison
- WHEN memory footprint results are available for all configurations
- THEN the report MUST present a table comparing SmallAIOS versus each Linux baseline
- AND the report MUST express the SmallAIOS footprint as a percentage of each Linux baseline

### Requirement: Benchmark Models
The benchmark infrastructure SHALL use three standard ONNX models covering vision, text, and audio/signal inference modalities.

#### Scenario: MobileNetV2 vision benchmark
- WHEN the vision benchmark is executed
- THEN it MUST use the MobileNetV2 ONNX model (approximately 14 MB)
- AND the input MUST be a standard 224x224 RGB image tensor
- AND the output MUST be a 1000-class classification result

#### Scenario: DistilBERT text benchmark
- WHEN the text benchmark is executed
- THEN it MUST use the DistilBERT ONNX model (approximately 250 MB)
- AND the input MUST be tokenized text with standard BERT tokenization
- AND the output MUST be the model's embedding or classification result

#### Scenario: Whisper-tiny audio/signal benchmark
- WHEN the audio/signal benchmark is executed
- THEN it MUST use the Whisper-tiny ONNX model (approximately 150 MB)
- AND the input MUST be a standard 30-second mel spectrogram
- AND the output MUST be the transcribed text or token sequence

### Requirement: Linux Baselines
The benchmark infrastructure SHALL compare SmallAIOS against three Linux-based deployment configurations using ONNX Runtime.

#### Scenario: Bare metal ONNX Runtime baseline
- WHEN the bare metal Linux baseline is benchmarked
- THEN it MUST run ONNX Runtime directly on the Linux host without containerization
- AND the Linux kernel version, ONNX Runtime version, and all relevant library versions MUST be documented

#### Scenario: Docker + ONNX Runtime baseline
- WHEN the Docker Linux baseline is benchmarked
- THEN it MUST run ONNX Runtime inside a Docker container with the official ONNX Runtime image
- AND the Docker version, container runtime, and any resource limits MUST be documented

#### Scenario: K8s/K3s + ONNX Runtime baseline
- WHEN the Kubernetes Linux baseline is benchmarked
- THEN it MUST run ONNX Runtime as a Kubernetes pod (K8s for datacenter, K3s for edge)
- AND the Kubernetes version, pod resource requests/limits, and scheduling configuration MUST be documented

### Requirement: Hardware Targets
The benchmark infrastructure SHALL run all benchmarks on four hardware targets spanning datacenter and edge deployments.

#### Scenario: DGX Spark datacenter GPU target
- WHEN benchmarks are executed on the DGX Spark
- THEN all three models MUST be benchmarked with GPU acceleration enabled
- AND the GPU model, driver version, and CUDA version MUST be documented

#### Scenario: Intel Xeon datacenter CPU target
- WHEN benchmarks are executed on the Intel Xeon platform
- THEN all three models MUST be benchmarked using CPU inference
- AND the CPU model, core count, and memory configuration MUST be documented

#### Scenario: Jetson Orin Nano edge GPU target
- WHEN benchmarks are executed on the Jetson Orin Nano
- THEN all three models MUST be benchmarked with GPU acceleration enabled
- AND the JetPack version, power mode, and thermal configuration MUST be documented

#### Scenario: Raspberry Pi 4/5 edge CPU target
- WHEN benchmarks are executed on the Raspberry Pi 4 or Pi 5
- THEN all three models MUST be benchmarked using CPU inference
- AND the Pi model, OS version, and memory configuration MUST be documented
- AND if a model exceeds available memory, the benchmark MUST report OOM rather than skip silently

### Requirement: Reproducibility
The benchmark infrastructure SHALL document all hardware and software configuration necessary to reproduce benchmark results.

#### Scenario: Document BIOS and firmware settings
- WHEN benchmark results are published
- THEN the report MUST include relevant BIOS settings (virtualization, power management, turbo boost)
- AND any non-default firmware configuration MUST be explicitly listed

#### Scenario: CPU frequency pinning
- WHEN benchmarks are executed
- THEN CPU frequency MUST be pinned to a fixed value (disabling dynamic frequency scaling)
- AND the pinned frequency MUST be documented in the benchmark report
- AND if frequency pinning is not possible on a platform, the report MUST note this limitation

#### Scenario: Thermal state documentation
- WHEN benchmarks are executed
- THEN the ambient temperature and cooling configuration MUST be documented
- AND if thermal throttling is detected during a benchmark run, the affected results MUST be flagged
- AND the benchmark MUST include a thermal stabilization period before measurement begins

### Requirement: Report Generation
The benchmark infrastructure SHALL generate comparison reports with markdown tables and charts.

#### Scenario: Generate markdown comparison tables
- WHEN all benchmark runs for a hardware target are complete
- THEN the report generator MUST produce markdown tables comparing SmallAIOS against each Linux baseline
- AND the tables MUST include cold start, latency percentiles, throughput, jitter, and memory footprint

#### Scenario: Generate comparison charts
- WHEN benchmark data is available for visualization
- THEN the report generator MUST produce charts (bar charts for latency/throughput, line charts for batch scaling)
- AND charts MUST be saved as PNG or SVG files suitable for inclusion in documentation

#### Scenario: Report includes hardware and configuration metadata
- WHEN a benchmark report is generated
- THEN it MUST include a metadata section listing hardware target, software versions, date, and git commit hash
- AND the metadata MUST be sufficient to reproduce the exact benchmark configuration

### Requirement: Benchmark Storage
All benchmark scripts, configurations, and analysis tools SHALL be stored in the bench/ directory at the repository root.

#### Scenario: Benchmark script organization
- WHEN a developer looks for benchmark tooling
- THEN all benchmark runner scripts MUST be located under bench/
- AND the directory MUST include a README or usage documentation explaining how to run benchmarks

#### Scenario: Baseline configuration storage
- WHEN Linux baseline configurations (Dockerfiles, K8s manifests, bare metal setup scripts) are needed
- THEN they MUST be stored under bench/ in clearly named subdirectories
- AND each configuration MUST be version-controlled alongside the benchmark scripts
