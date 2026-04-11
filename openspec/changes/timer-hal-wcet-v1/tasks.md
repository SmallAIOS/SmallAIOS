## 1. Profile Module (onnx-rt)

- [x] 1.1 Create `onnx-rt/src/profile.rs` with `TimeSource` trait
- [x] 1.2 Implement `NullTimeSource` (zero-overhead default, returns 0)
- [x] 1.3 Implement `StdTimeSource` behind `#[cfg(feature = "std")]` using `std::time::Instant` + `OnceLock` for the epoch
- [x] 1.4 Define local `OperatorClass` enum (Elementwise/Reduction/Gemm/Attention/GpuKernel) and `BudgetResult` enum (Ok/Warning/SoftLimit/HardLimit)
- [x] 1.5 Define `OperatorBudget` struct mirroring the kernel-side version (same defaults)
- [x] 1.6 Define `InferenceProfile` and `OperatorMeasurement` structs
- [x] 1.7 Implement `classify_op(op_type: &str) -> OperatorClass` per the design
- [x] 1.8 Unit tests: classification coverage, NullTimeSource, StdTimeSource monotonicity, budget check for all 4 results

## 2. Executor Integration

- [x] 2.1 Update `execute_graph()` signature to accept `Option<&mut InferenceProfile>`, `&OperatorBudget`, `&dyn TimeSource`
- [x] 2.2 Wrap each `dispatch_node()` call in time measurement when profile is Some
- [x] 2.3 Classify op, call `budget.check()`, handle all four BudgetResult variants
- [x] 2.4 Hard-limit abort: return `SessionError::ExecutionFailed` with op name + measured time
- [x] 2.5 Populate profile: add OperatorMeasurement, increment counters, accumulate total_us
- [x] 2.6 Update existing callers of `execute_graph()` to pass None / default budget / NullTimeSource

## 3. Session API

- [x] 3.1 Add `Session::run_with_profile()` method returning `(Vec<InferenceOutput>, InferenceProfile)`
- [x] 3.2 Implement by delegating to `execute_graph()` with profile Some and StdTimeSource (in container mode)
- [x] 3.3 Existing `Session::run()` passes profile None and NullTimeSource — zero overhead path
- [x] 3.4 Unit test: run_with_profile returns populated profile for a multi-op graph
- [x] 3.5 Unit test: run() has identical output to run_with_profile() ignoring the profile

## 4. Budget Enforcement Tests

- [x] 4.1 Create a mock slow operator via `std::thread::sleep` in a test-only op wrapper
- [x] 4.2 Test: op under budget → BudgetResult::Ok, no log
- [x] 4.3 Test: op at 1.5x budget → BudgetResult::Warning, warnings_count increments
- [x] 4.4 Test: op at 3x budget → BudgetResult::SoftLimit, soft_limit_count increments
- [x] 4.5 Test: op at 15x budget → BudgetResult::HardLimit, execution aborts with error
- [x] 4.6 Test: after hard-limit abort, Session is not poisoned (can still initialize another session)

## 5. Kernel sys_time Fix

- [x] 5.1 Update `kernel/src/syscall/system.rs::sys_time()` to call `crate::sched::timer::Timestamp::now()` and return the value
- [x] 5.2 Handle uncalibrated timer case (return raw ticks, document in comment)
- [x] 5.3 Unit test: sys_time monotonicity across two calls

## 6. Library Registration and Docs

- [x] 6.1 Register `profile` module in `onnx-rt/src/lib.rs`
- [x] 6.2 Update `onnx-rt/Cargo.toml` if new features needed (std already exists from cpu-parallel-inference-v1)
- [x] 6.3 Update `docs/scheduling-model.md` section 2 (Per-operator budgets) to note that enforcement is now live
- [x] 6.4 Add example to `docs/inference-bus.md` showing how to enable profiling via `run_with_profile()`

## 7. Validation

- [x] 7.1 `just fmt` clean
- [x] 7.2 `just clippy --all-targets` clean
- [x] 7.3 `just test` all passing
- [x] 7.4 New test count: at least 15 profile-related tests in onnx-rt
- [x] 7.5 Verify zero-overhead path: run() unchanged performance on benchmarks (no regression)
