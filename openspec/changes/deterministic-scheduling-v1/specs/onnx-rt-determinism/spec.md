# Capability: onnx-rt-determinism

## ADDED Requirements

### Requirement: SessionConfig deterministic flag

`SessionConfig` SHALL expose a `deterministic: bool` field (default `false`) and a `deterministic_seed: u64` field (default `0`), and the runtime SHALL honor them by collapsing all sources of host-timing-derived nondeterminism into a single reproducible execution path.

#### Scenario: Deterministic flag plumbs to the session

- **GIVEN** a developer constructing a `Session` with `SessionConfig { deterministic: true, deterministic_seed: 42, ..Default::default() }`
- **WHEN** the session is constructed
- **THEN** the session SHALL initialize all internal RNG state from `deterministic_seed`
- **AND** the session SHALL force `StreamConfig::SingleStream` regardless of any explicit `stream_config` value
- **AND** the session SHALL bypass CUDA Graphs capture and replay paths
- **AND** the resulting session SHALL be marked deterministic for the purpose of downstream queries (e.g. `Session::is_deterministic()`)

#### Scenario: Conflicting stream config emits a warning, not an error

- **GIVEN** a developer constructing a `Session` with both `deterministic: true` and `stream_config: StreamConfig::Overlap { transfer_streams: 2 }`
- **WHEN** the session is constructed
- **THEN** the session SHALL succeed
- **AND** the session SHALL emit a one-line warning to syslog explaining that determinism overrides the overlap configuration
- **AND** the resulting session SHALL behave as if `stream_config = SingleStream`

#### Scenario: Default config is unchanged

- **GIVEN** a developer constructing a `Session` with `SessionConfig::default()`
- **WHEN** the session is constructed
- **THEN** the `deterministic` field SHALL be `false`
- **AND** the session SHALL behave bit-for-bit identically to the pre-`deterministic-scheduling-v1` default
- **AND** existing throughput benchmarks SHALL be unchanged

### Requirement: Deterministic CUDA stream collapse

When `Session::is_deterministic()` is `true`, the CUDA executor SHALL run all GPU work on a single stream and SHALL issue a `cudaStreamSynchronize` at every operator boundary.

#### Scenario: Single stream is used in deterministic mode

- **GIVEN** a deterministic session executing a GPU-accelerated ONNX model
- **WHEN** the executor processes the first operator
- **THEN** the operator's GPU work SHALL be issued on the session's single compute stream
- **AND** no transfer streams SHALL be allocated
- **AND** the `StreamPool::is_single_stream()` invariant SHALL hold for the session lifetime

#### Scenario: Op-boundary synchronize is issued

- **GIVEN** a deterministic session executing a GPU operator
- **WHEN** the operator's GPU work completes
- **THEN** the executor SHALL issue `cudaStreamSynchronize` on the compute stream before returning control to the cooperative scheduler
- **AND** the synchronize SHALL block until all preceding GPU work on the stream has completed
- **AND** a failed synchronize SHALL surface as a fatal session error using the existing error path

#### Scenario: CUDA Graphs capture is bypassed

- **GIVEN** a deterministic session executing an ONNX model that the default-mode runtime would normally capture via CUDA Graphs
- **WHEN** the session runs an inference
- **THEN** the runtime SHALL NOT call `cudaGraphCapture` / `cudaGraphInstantiate` / `cudaGraphLaunch`
- **AND** the operators SHALL execute via the uncaptured op-by-op path
- **AND** the documented throughput cost (5-10% on the Jetson Orin path) SHALL be acknowledged in `docs/determinism.md`

### Requirement: Per-session deterministic RNG

The runtime SHALL provide a `DeterministicRng` type keyed on `(session_seed, op_index, draw_counter)` and SHALL route every RNG-consuming operator through it when the session is in deterministic mode.

#### Scenario: Same seed produces same draws

- **GIVEN** two deterministic sessions constructed with `deterministic_seed = S` for the same model
- **WHEN** each session runs an inference with the same input
- **THEN** every RNG draw on every operator SHALL produce the same value across the two sessions
- **AND** the output tensors SHALL be byte-identical

#### Scenario: Different seed produces different draws

- **GIVEN** two deterministic sessions constructed with `deterministic_seed = S1` and `deterministic_seed = S2` where `S1 ≠ S2`, for the same model
- **WHEN** each session runs an inference with the same input
- **THEN** the per-operator RNG draws SHALL differ between the two sessions (with overwhelmingly high probability for any reasonable hash mixing)
- **AND** the output tensors SHALL differ for any operator whose output depends on RNG draws

#### Scenario: No wall-clock fallback

- **GIVEN** a deterministic session running an operator that previously used host wall-clock time as a "random" seed (e.g. `Multinomial` or `Bernoulli` with `seed = 0`)
- **WHEN** the operator draws a sample
- **THEN** the draw SHALL come from `DeterministicRng` keyed on the session seed and operator index
- **AND** the runtime SHALL NOT call `std::time::Instant::now`, `clock_gettime`, or `cudaDeviceGetTimerValue` on the inference hot path

#### Scenario: RNG draws are stable across session restarts

- **GIVEN** a deterministic session constructed with `deterministic_seed = S` for model `M` with input `I`
- **AND** the session being destroyed and reconstructed with the same `(S, M, I)`
- **WHEN** the second session runs an inference
- **THEN** the output tensors SHALL be byte-identical to the first session's output

### Requirement: Deterministic-safe operator registry

The runtime SHALL maintain a registry of which ONNX operators are deterministic-safe and SHALL reject session construction with `deterministic = true` if the loaded model contains an operator that is not deterministic-safe.

#### Scenario: All current operators are deterministic-safe

- **GIVEN** the v1 implementation of `deterministic-scheduling-v1`
- **WHEN** the operator registry is initialized
- **THEN** every operator currently implemented in `onnx-rt/src/ops/` SHALL be marked deterministic-safe (this is a v1 invariant — no operator currently introduces nondeterminism that we cannot remove)
- **AND** a future operator that legitimately requires nondeterminism SHALL set its registry entry to `false` and document the reason

#### Scenario: Unsafe operator rejects deterministic session

- **GIVEN** a future ONNX operator marked deterministic-unsafe in the registry
- **WHEN** a developer attempts to construct a session with `deterministic = true` on a model containing that operator
- **THEN** session construction SHALL return a typed error naming the offending operator
- **AND** the error message SHALL include a pointer to `docs/determinism.md` for further reading

### Requirement: Reproducibility test surface

The repository SHALL provide a `just test-determinism` recipe and a `determinism-reproducibility` advisory CI job that together verify the bit-identical reproducibility contract on at least four representative models.

#### Scenario: `just test-determinism` runs locally

- **GIVEN** a developer with the workspace built and the CUDA toolchain available (or skipping CUDA fixtures)
- **WHEN** they run `just test-determinism`
- **THEN** the recipe SHALL invoke `scripts/test-determinism.sh`, which runs each fixture model twice in deterministic mode and byte-diffs the outputs
- **AND** the recipe SHALL exit 0 if all fixture outputs match, 30 if a model failed to run, 40 if any output diverged (with a hex dump of the first divergent bytes printed to stderr)

#### Scenario: CI advisory job runs on every PR

- **GIVEN** a PR that touches `onnx-rt/`, `kernel/src/sched/`, or `onnx-rt/src/cuda/`
- **WHEN** the PR pipeline runs
- **THEN** the `determinism-reproducibility` advisory CI job SHALL execute `just test-determinism` against the CPU-only and RNG-op fixture models
- **AND** the job SHALL be marked `continue-on-error: true` until promoted to a gate
- **AND** a comment in the workflow YAML SHALL document the promote-to-gate criterion

#### Scenario: Throughput regression budget enforced

- **GIVEN** the `bench/llm-inference-bench` benchmark extended with a deterministic-mode comparison
- **WHEN** the benchmark runs on the Jetson Orin Industrial reference platform or x86-64 CUDA
- **THEN** deterministic-mode throughput SHALL be at least 65% of default-mode throughput (i.e. at most 35% regression)
- **AND** a regression below 65% SHALL fail the benchmark and require proposal revision before re-merge
