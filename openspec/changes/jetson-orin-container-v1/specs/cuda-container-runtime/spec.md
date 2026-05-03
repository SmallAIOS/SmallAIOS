## MODIFIED Requirements

### Requirement: Session is thread-safe under the cuda feature

When the `cuda` feature is enabled, `smallaios_onnx_rt::session::Session` SHALL be `Send + Sync` so it can be embedded in multi-threaded HTTP request handlers and `std::thread::spawn` closures.

#### Scenario: Session held across HTTP worker threads

- **GIVEN** the smallaios container compiled with `--features cuda,nvidia_gpu` is started with at least one model loaded
- **WHEN** the `HttpServer::route_fn` thread pool dispatches an inference request to a worker thread different from the one that constructed the `Session`
- **THEN** the worker SHALL be able to call `session.run(...)` without a Rust compile-time `Send`/`Sync` error
- **AND** the worker SHALL not race with concurrent requests on the same `Session` (writes to the GPU graph cache, stream pool, and device weight cache are serialized via `Mutex`)

#### Scenario: Static thread-safety assertion

- **GIVEN** any future change touches `Session` or any field reachable from it under the `cuda` feature
- **THEN** the workspace SHALL contain a `const _: fn() = || { fn assert_send_sync<T: Send + Sync>() {} assert_send_sync::<Session>(); };` (or equivalent static check)
- **AND** the assertion SHALL be unconditionally compiled when the `cuda` feature is on, so any regression is caught at `cargo check` time before reaching review

#### Scenario: Raw CUDA handles wrapped behind Send/Sync newtypes

- **GIVEN** any cached CUDA handle (`cudaGraphExec_t`, `cudaGraph_t`, `cudaStream_t`, `cublasHandle_t`, `cudnnHandle_t`) stored on `Session` or a type owned by `Session`
- **THEN** the handle SHALL be wrapped in a newtype with `unsafe impl Send + Sync` and a `// SAFETY:` comment naming the CUDA contract that justifies the impl
- **AND** the newtype SHALL be the only place those bounds are asserted unsafely (no scattered `unsafe impl Send` for `*mut c_void` etc.)

### Requirement: cargo check --features cuda is gated in CI

The repository SHALL enforce a CI gate that runs `cargo check --workspace --features cuda,nvidia_gpu` on every PR.

#### Scenario: cuda-only regression caught in CI

- **GIVEN** a PR that breaks `cargo check --features cuda` without breaking the default-feature build
- **WHEN** the PR pipeline runs
- **THEN** the `cuda-check` job SHALL fail
- **AND** the change-gates meta-job SHALL block merge until it is fixed
