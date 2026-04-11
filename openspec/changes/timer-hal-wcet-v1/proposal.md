## Why

The `OperatorBudget` struct in `kernel/src/sched/executor.rs` was built as part of `smallaios-kernel-v1` with well-defined per-operator time budgets (1ms elementwise, 10ms reduction, 100ms GEMM, 500ms attention, 1s GPU kernel) and a 4-level `BudgetResult` enum (Ok/Warning/SoftLimit/HardLimit). The `sched/timer.rs` module has `Timestamp::now()` and `elapsed_us_since()` primitives. But nothing wires them together:

- `sys_time()` returns 0 (stub)
- The ONNX executor's `yield_fn` callback fires between operators but doesn't measure execution time
- `OperatorBudget::check()` is never called in the hot path
- The `SessionConfig.enable_profiling` flag does nothing

This change closes the loop: real time measurement + budget enforcement + per-operator profiling for WCET analysis (DO-178C DAL A).

## What Changes

- Implement `sys_time()` in `kernel/src/syscall/system.rs` using the existing `Timestamp::now()` primitive
- Add a container-mode time source that uses `std::time::Instant` (the kernel-mode path already uses architecture timers)
- Wire the ONNX executor to measure wall-clock time per operator when `enable_profiling = true`
- Classify each operator into an `OperatorClass` (Elementwise/Reduction/Gemm/Attention/GpuKernel)
- Call `OperatorBudget::check()` with measured time, act on the result:
  - `Ok`: continue silently
  - `Warning`: log via eprintln (container) / syslog (kernel)
  - `SoftLimit`: log + increment metric counter
  - `HardLimit`: abort inference with `SessionError::ExecutionFailed("operator exceeded hard time limit: <op> <ms>")`
- Add profiling report: after `Session::run()`, expose per-operator timing via a new `get_profile()` method on Session
- Add tests that verify budget enforcement triggers correctly

## Capabilities

### New Capabilities
- `operator-profiling`: Per-operator wall-clock measurement, classification, budget enforcement, and profile reporting

### Modified Capabilities
- `onnx-runtime`: executor measures timing when profiling enabled, aborts on hard-limit violations
- `kernel-core`: `sys_time()` returns real monotonic time instead of 0

## Impact

- **Code:** New `onnx-rt/src/profile.rs` (~200 lines), modifications to `executor.rs` (~80 lines), `session.rs` (profile field), `kernel/src/syscall/system.rs` (wire `Timestamp::now()`)
- **Behavior:** When `SessionConfig::enable_profiling = true`, slow operators now log warnings and hard-limit violations abort. Default config (profiling off) has zero overhead.
- **Tests:** Add tests that construct a fake slow operator and verify all four `BudgetResult` paths fire correctly
- **DO-178C:** Enables the WCET analysis framework we've designed but not used
