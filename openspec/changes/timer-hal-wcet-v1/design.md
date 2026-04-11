## Context

The `kernel/src/sched/timer.rs` module has architecture timer primitives:
- `Timestamp::now()` → wraps `read_cntpct()` (ARM64) / `rdtsc()` (x86_64) / CLINT mtime (RISC-V)
- `Timestamp::elapsed_us_since(earlier)` → uses calibrated `TIMER_FREQ_HZ`
- `ticks_to_us()` / `us_to_ticks()` helpers

The `kernel/src/sched/executor.rs` module has `OperatorBudget` with `check()` returning `BudgetResult`.

The ONNX executor (`onnx-rt/src/executor.rs::execute_graph`) calls the `yield_fn` callback between operators but doesn't measure execution time. The `Session` config has `enable_profiling: bool` that does nothing.

The gap: ONNX is a `#![no_std]` crate (Layer 1) and can't depend on `kernel` (Layer 0) — but it needs a time source. The cleanest fix is a time-source trait in the ONNX crate that both kernel and container implement.

## Goals / Non-Goals

**Goals:**
- Zero-overhead time measurement when `enable_profiling = false`
- Wall-clock measurement per operator when profiling enabled
- Classify operators to match OperatorBudget categories
- Actionable BudgetResult handling: log warnings, abort on hard limits
- Per-session profile report: total time, per-operator breakdown
- Works in both kernel mode (bare metal timer) and container mode (std::time::Instant)
- Tests verify all 4 BudgetResult outcomes

**Non-Goals:**
- Multi-threaded profiling (single-threaded cooperative model)
- Nanosecond precision — microseconds are sufficient for operator budgets
- Real WCET calibration (just measurement; calibration is future work)
- Integrating with external tracing systems (perf, ftrace) — future

## Decisions

### D1: TimeSource Trait in onnx-rt (Layer 1)

Define a minimal trait in `onnx-rt/src/profile.rs`:

```rust
pub trait TimeSource: Send + Sync {
    /// Return the current time in microseconds since some fixed epoch.
    /// Epoch can be arbitrary — only differences matter.
    fn now_us(&self) -> u64;
}
```

Two implementations:
1. **`StdTimeSource`** (container mode, `#[cfg(feature = "std")]`) — uses `std::time::Instant` stored in an `OnceLock` as the epoch
2. **`NullTimeSource`** (default, always returns 0) — used when profiling is disabled

The kernel crate provides its own `KernelTimeSource` in a follow-up change — for now, container mode is the priority since that's where we actually run inference.

### D2: Profile Report Structure

```rust
#[derive(Debug, Clone, Default)]
pub struct InferenceProfile {
    pub total_us: u64,
    pub operators: Vec<OperatorMeasurement>,
    pub soft_limit_count: u32,
    pub warnings_count: u32,
    pub hard_limit_aborted: bool,
}

#[derive(Debug, Clone)]
pub struct OperatorMeasurement {
    pub op_type: String,
    pub class: OperatorClass,
    pub actual_us: u64,
    pub budget_result: BudgetResult,
}
```

The `OperatorClass` and `BudgetResult` types are duplicated from kernel-side (no_std friendly). For now they're local to onnx-rt; a future refactor can share them via a `sched-types` crate.

### D3: Classification by Operator Name

Map ONNX op_type strings to OperatorClass:

```rust
pub fn classify_op(op_type: &str) -> OperatorClass {
    match op_type {
        "Add" | "Sub" | "Mul" | "Div" | "Relu" | "Sigmoid" | "Tanh" 
        | "Clip" | "Cast" | "Reshape" | "Flatten" | "Squeeze" | "Unsqueeze" 
        | "Transpose" | "Concat" | "Slice" | "Pad" | "Gather" => OperatorClass::Elementwise,
        
        "Softmax" | "LayerNormalization" | "BatchNormalization" | "MaxPool" 
        | "AveragePool" | "GlobalAveragePool" | "ReduceMean" | "ReduceSum" => OperatorClass::Reduction,
        
        "MatMul" | "Gemm" | "Conv" => OperatorClass::Gemm,
        
        _ => OperatorClass::Elementwise, // default for unknown ops
    }
}
```

### D4: Executor Integration

Modify `execute_graph()` signature to optionally return a profile:

```rust
pub fn execute_graph(
    graph: &ExecutionGraph,
    inputs: &[(String, Tensor)],
    initializers: &[TensorProto],
    yield_fn: Option<fn()>,
    profile: Option<&mut InferenceProfile>,  // new
    budget: &OperatorBudget,                  // new (use default if caller doesn't care)
    time_source: &dyn TimeSource,             // new
) -> Result<Vec<InferenceOutput>, SessionError>
```

Inside the node loop:
```rust
let start = time_source.now_us();
let outputs = dispatch_node(...)?;
let elapsed = time_source.now_us() - start;

if let Some(profile) = profile.as_mut() {
    let class = classify_op(&node.op_type);
    let result = budget.check(class, elapsed);
    profile.operators.push(OperatorMeasurement { ... });
    profile.total_us += elapsed;
    
    match result {
        BudgetResult::Ok => {}
        BudgetResult::Warning => profile.warnings_count += 1,
        BudgetResult::SoftLimit => profile.soft_limit_count += 1,
        BudgetResult::HardLimit => {
            profile.hard_limit_aborted = true;
            return Err(SessionError::ExecutionFailed(format!(
                "operator '{}' exceeded hard time limit: {} us", 
                node.op_type, elapsed
            )));
        }
    }
}
```

When `profile` is `None` and `time_source` is `NullTimeSource`, the compiler should inline the measurement to zero cost.

### D5: Session API

Add to `Session`:
```rust
impl Session {
    pub fn run_with_profile(
        &self, 
        inputs: &[InferenceInput]
    ) -> Result<(Vec<InferenceOutput>, InferenceProfile), SessionError> {
        let mut profile = InferenceProfile::default();
        let outputs = self.run_internal(inputs, Some(&mut profile))?;
        Ok((outputs, profile))
    }
}
```

The existing `run()` still works — it just passes `None` for the profile.

### D6: sys_time() Fix

In `kernel/src/syscall/system.rs::sys_time()`:
```rust
pub fn sys_time(_args: &SyscallArgs) -> SyscallResult {
    // Return nanoseconds since boot via the Timestamp primitive
    let now = crate::sched::timer::Timestamp::now();
    now.as_u64() as SyscallResult  // clamped to i64 positive range
}
```

Nanosecond conversion is handled by the `ticks_to_us()` helper — we multiply by 1000 to get nanoseconds (or preserve as raw ticks if freq isn't calibrated yet).

## Risks / Trade-offs

**[Risk] Profiling overhead when disabled** — If the measurement branches aren't properly gated, even the "disabled" path costs cycles. Mitigation: `time_source` is a trait object, so `NullTimeSource::now_us()` is a vtable call returning 0. Modern CPUs predict this perfectly (~1 cycle). Alternatively, wrap the whole profiling block in `if profile.is_some()` so the time source isn't even called.

**[Risk] `std::time::Instant` resolution on macOS** — 1μs resolution is typical but not guaranteed. Mitigation: Our budgets are at ms scale; μs-resolution jitter is fine.

**[Risk] Hard-limit abort in middle of inference leaves state inconsistent** — The tensor value map may have partial results. Mitigation: The error returns before the map is read for outputs, so the caller sees the error and discards the session run. No state corruption.

**[Trade-off] Duplicated enums between kernel and onnx-rt** — `OperatorClass` and `BudgetResult` exist in both places. A future `sched-types` crate can unify them. For now, duplication is acceptable since the types are tiny and rarely change.

## Open Questions

- **Q1:** Should the profile be attached to the Session or returned from `run()` explicitly? *Leaning toward: returned explicitly via `run_with_profile()` so the standard `run()` path stays simple.*
- **Q2:** Do we want per-class default budgets or allow custom budgets per session? *Leaning toward: default OperatorBudget::DEFAULT for now; custom budgets are a future feature.*
