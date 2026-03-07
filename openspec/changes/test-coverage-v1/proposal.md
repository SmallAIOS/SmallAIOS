## Why

Overall line coverage is 93.84% but 18 files are below 90%, with the peripheral crate hardware drivers averaging ~55% coverage. For a safety-critical OS targeting DO-178C DAL A compliance, near-100% coverage with MC/DC on critical paths is essential. Improving coverage now catches latent bugs and establishes the baseline before hardware bring-up.

## What Changes

- Add unit tests to all peripheral driver files currently under 70% coverage (UART, GPIO, SPI, I2C, camera — 15 files)
- Add tests for peripheral driver files between 70-90% coverage
- Add integration tests for cross-crate workflows (boot → inference, network → ONNX)
- Add end-to-end tests for container and bare-metal boot paths
- Expand fuzzing coverage beyond crypto to ONNX parser, protobuf, network packet handling
- Add performance benchmarks with regression detection (bench crate currently at 0%)
- Exclude container/src/main.rs from line coverage targets (binary entry point)
- Set up per-crate coverage thresholds in CI (fail if any crate drops below 90%)
- Configure Codecov with `codecov.yml` for path exclusions and project/patch targets
- Target: near-100% overall line coverage

## Capabilities

### New Capabilities
- `coverage-unit-tests`: Unit test coverage for undertested code — covers adding mock-based unit tests for peripheral drivers (UART, GPIO, SPI, I2C, camera — 15 files), and closing gaps in any crate below 90%
- `coverage-integration-tests`: Integration and end-to-end tests — covers cross-crate integration tests (kernel↔security, net↔onnx-rt, container boot sequences), QEMU-based e2e tests, and Docker container validation
- `coverage-fuzzing`: Expanded fuzz testing — covers fuzz targets for ONNX protobuf parser, network packet parsing (TCP/UDP/QUIC), IPC message handling, and USB descriptor parsing, building on existing crypto fuzzing
- `coverage-benchmarks`: Performance test coverage — covers benchmark harness for ONNX operator throughput, crypto operations, memory allocator, scheduler latency, and network stack throughput with regression detection
- `coverage-ci-gates`: CI coverage enforcement — covers per-crate minimum thresholds (90%+), Codecov configuration, coverage regression detection, and exclusion rules for untestable code

### Modified Capabilities
None — this adds tests without changing existing behavior.

## Impact

- `peripheral/src/` — 15+ files get additional unit tests
- `bench/src/` — benchmark harness with performance regression detection
- `kernel/tests/`, `security/tests/` — new integration test files
- `fuzz/` — new fuzz targets (or inline fuzz modules)
- `.github/workflows/ci.yml` — per-crate coverage threshold enforcement
- `codecov.yml` — coverage targets, path exclusions, PR comment configuration
- Target coverage delta: 93.84% → near-100%
