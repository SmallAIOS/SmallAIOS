## Context

After PR #69 (`timer-hal-wcet-v1`), three types are duplicated:

| Type | Kernel location | onnx-rt location |
|------|-----------------|------------------|
| `OperatorClass` | `kernel/src/sched/executor.rs:174-182` | `onnx-rt/src/profile.rs:~95` |
| `BudgetResult` | `kernel/src/sched/executor.rs:184-195` | `onnx-rt/src/profile.rs:~110` |
| `OperatorBudget` | `kernel/src/sched/executor.rs:143-218` | `onnx-rt/src/profile.rs:~120` |

The duplication exists because of the 4-layer architecture rule: `onnx-rt` is at Layer 1 (Core Services) and cannot depend on `kernel` at Layer 0 (Foundation). The standard fix in this layering pattern is to extract shared types into a Layer 0 crate that both layers can depend on.

## Goals / Non-Goals

**Goals:**
- Single source of truth for `OperatorClass`, `BudgetResult`, `OperatorBudget`
- Both kernel and onnx-rt depend on the new crate
- No behavior changes — pure refactoring
- All existing tests pass unchanged
- Remove the `sonar.cpd.exclusions` workaround

**Non-Goals:**
- Move other types out of kernel — only the operator budget types
- Change the budget enforcement semantics
- Add new operator categories or budget tunables
- Refactor the kernel scheduler itself

## Decisions

### D1: New `sched-types/` Crate at Layer 0

Create `sched-types/` crate alongside `kernel/`, `security/`, `compute/`:

```toml
[package]
name = "smallaios-sched-types"
version.workspace = true
edition.workspace = true
license.workspace = true
description = "Shared scheduler primitive types for SmallAIOS (operator classes, budgets, results)"

[dependencies]
# None — pure type definitions, no_std + alloc-free
```

The crate is `#![no_std]` and **alloc-free** (the types are all `Copy`/`PartialEq` enums and structs). This makes it usable from any crate at any layer.

### D2: Type Definitions

```rust
// sched-types/src/lib.rs
#![no_std]
#![forbid(unsafe_code)]

/// Operator category for budget lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum OperatorClass {
    Elementwise = 0,
    Reduction = 1,
    Gemm = 2,
    Attention = 3,
    GpuKernel = 4,
}

/// Result of checking an operator's execution time against its budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetResult {
    Ok,
    Warning,
    SoftLimit,
    HardLimit,
}

/// Per-operator execution time budgets in microseconds.
#[derive(Debug, Clone, Copy)]
pub struct OperatorBudget {
    pub elementwise_us: u64,
    pub reduction_us: u64,
    pub gemm_us: u64,
    pub attention_us: u64,
    pub gpu_kernel_us: u64,
    pub soft_multiplier: u64,
    pub hard_multiplier: u64,
}

impl OperatorBudget {
    pub const DEFAULT: Self = Self {
        elementwise_us: 1_000,
        reduction_us: 10_000,
        gemm_us: 100_000,
        attention_us: 500_000,
        gpu_kernel_us: 1_000_000,
        soft_multiplier: 2,
        hard_multiplier: 10,
    };

    pub const fn check(&self, class: OperatorClass, actual_us: u64) -> BudgetResult {
        let budget = match class {
            OperatorClass::Elementwise => self.elementwise_us,
            OperatorClass::Reduction => self.reduction_us,
            OperatorClass::Gemm => self.gemm_us,
            OperatorClass::Attention => self.attention_us,
            OperatorClass::GpuKernel => self.gpu_kernel_us,
        };
        if actual_us > budget * self.hard_multiplier {
            BudgetResult::HardLimit
        } else if actual_us > budget * self.soft_multiplier {
            BudgetResult::SoftLimit
        } else if actual_us > budget {
            BudgetResult::Warning
        } else {
            BudgetResult::Ok
        }
    }
}

impl Default for OperatorBudget {
    fn default() -> Self { Self::DEFAULT }
}
```

### D3: Backwards-Compatible Re-exports in Kernel

The kernel currently exports these from `sched::executor`. To avoid breaking existing call sites:

```rust
// kernel/src/sched/executor.rs
pub use smallaios_sched_types::{OperatorClass, BudgetResult, OperatorBudget};

// Remove the local definitions
```

Existing kernel code that does `use crate::sched::executor::OperatorClass;` continues to work because of the re-export.

### D4: onnx-rt Uses sched-types Directly

```rust
// onnx-rt/src/profile.rs
pub use smallaios_sched_types::{OperatorClass, BudgetResult, OperatorBudget};

// Remove the duplicated definitions
```

Add to `onnx-rt/Cargo.toml`:
```toml
smallaios-sched-types = { path = "../sched-types" }
```

### D5: Remove sonar.cpd.exclusions

After the refactor, `profile.rs` no longer duplicates kernel code. Remove the exclusion line from `sonar-project.properties`.

## Risks / Trade-offs

**[Risk] Breaking the kernel's existing public API surface** — If anything outside the kernel imports these types from `kernel::sched::executor::*`, the re-export must be exhaustive. Mitigation: search for `kernel::sched::executor::OperatorClass` (and friends) usages workspace-wide and update if needed. Re-exports preserve the path so most users don't notice.

**[Risk] Kernel-side tests reference the moved types** — They use `OperatorBudget::DEFAULT.check(...)`. Mitigation: re-export preserves the API exactly. Tests should pass unchanged.

**[Trade-off] Workspace gets one more crate (20 → 21)** — A small cost for the architectural cleanliness benefit. The Layer 0 set grows from `kernel + security + compute` to `kernel + security + compute + sched-types`.
