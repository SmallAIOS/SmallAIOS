## Context

Static analysis (SonarCloud + manual code review) identified 18 code quality issues across the codebase. None are bugs — they're cognitive complexity, code duplication, parameter count, and magic numbers. The highest-impact areas:

1. `onnx-rt/src/executor.rs` — `dispatch_node()` is 260+ lines with deeply nested attribute extraction
2. `onnx-rt/src/operators.rs` — `op_cast()` and tensor I/O patterns repeated 50+ times
3. `net/src/dhcp.rs` — duplicated option parsing in `parse_options()` and `get_option_value()`

## Goals / Non-Goals

**Goals:**
- Reduce cognitive complexity of `dispatch_node()` from ~45 to <15 (SonarCloud threshold)
- Eliminate `f32::from_le_bytes` boilerplate by extracting helpers
- Replace magic constants (`4` for f32 size, polynomial coefficients in `expf_approx`) with named constants
- Maintain 100% behavior parity — no functional changes
- All existing tests must continue to pass

**Non-Goals:**
- Adding new features
- Changing public APIs (refactoring is internal-only)
- Comprehensive style cleanup beyond the 18 issues identified
- Performance optimization (though some refactors may incidentally help)

## Decisions

### D1: Extract Tensor Byte Helpers to a New Module

Create `onnx-rt/src/tensor/bytes.rs` (or `onnx-rt/src/byte_io.rs`):

```rust
pub const F32_SIZE: usize = 4;
pub const I32_SIZE: usize = 4;
pub const I64_SIZE: usize = 8;
pub const F64_SIZE: usize = 8;

#[inline]
pub fn read_f32(data: &[u8], idx: usize) -> f32 {
    let off = idx * F32_SIZE;
    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

#[inline]
pub fn write_f32(data: &mut [u8], idx: usize, val: f32) {
    let off = idx * F32_SIZE;
    let bytes = val.to_le_bytes();
    data[off..off + F32_SIZE].copy_from_slice(&bytes);
}

#[inline]
pub fn read_i32(data: &[u8], idx: usize) -> i32 { ... }
pub fn read_i64(data: &[u8], idx: usize) -> i64 { ... }
pub fn read_f64(data: &[u8], idx: usize) -> f64 { ... }

pub fn allocate_tensor_data(elements: usize, dtype: DataType) -> Vec<u8> {
    vec![0u8; elements * dtype.element_size()]
}
```

`read_f32`/`write_f32` already exist as private functions in `operators.rs` — promote them to this module and make them `pub`. Update all 50+ call sites.

### D2: Split dispatch_node() by Operator Category

Current: one giant match with 30+ arms in `executor.rs::dispatch_node()`.

New: extract per-category helpers:

```rust
fn dispatch_arithmetic(kind: OpKind, inputs: &[...], attrs: &[...]) -> Result<Tensor, OpError>
fn dispatch_activation(kind: OpKind, ...) -> Result<Tensor, OpError>
fn dispatch_shape(kind: OpKind, ...) -> Result<Tensor, OpError>
fn dispatch_normalization(kind: OpKind, ...) -> Result<Tensor, OpError>
fn dispatch_pooling(kind: OpKind, ...) -> Result<Tensor, OpError>
fn dispatch_reduction(kind: OpKind, ...) -> Result<Tensor, OpError>
```

The top-level `dispatch_node` becomes a 6-arm match that delegates. Each helper has cognitive complexity <15.

### D3: Split op_cast() into Per-Conversion Functions

Current: `op_cast()` has 4 nested if-else branches.

New: `cast_f32_to_i32`, `cast_i32_to_f32`, `cast_f32_to_i64`, `cast_i64_to_f32` as private functions. Top-level `op_cast` is a small match dispatching by `(input.data_type, target)`.

### D4: ConvParams Struct

```rust
pub struct ConvParams<'a> {
    pub n: usize,
    pub c_out: usize,
    pub oh: usize,
    pub ow: usize,
    pub raw_data: &'a mut [u8],
}
```

`conv_compute(input, weight, bias, params)` — 4 args instead of 9.

### D5: Extract Common Helpers in executor.rs

Add to the existing private helper section:
```rust
fn read_first_f32(tensor: Option<&Tensor>) -> Option<f32> {
    tensor.and_then(|t| {
        if t.raw_data.len() >= 4 {
            Some(f32::from_le_bytes([t.raw_data[0], t.raw_data[1], t.raw_data[2], t.raw_data[3]]))
        } else { None }
    })
}
```

Replaces the duplicated min_val/max_val/constant_value extraction patterns.

### D6: Named Constants in expf_approx

```rust
const EXP_CLAMP_MAX: f32 = 88.7;
const EXP_CLAMP_MIN: f32 = -88.7;
const EXP_POLY_C2: f32 = 0.5;
const EXP_POLY_C3: f32 = 1.0 / 6.0;
const EXP_POLY_C4: f32 = 1.0 / 24.0;
const F32_EXPONENT_BIAS: i32 = 127;
const F32_MANTISSA_BITS: u32 = 23;
```

### D7: Extract DHCP Option Iterator

Replace duplicated parsing loops in `parse_options()` and `get_option_value()` with a shared iterator function that yields `(option_code, option_value_slice)`.

## Risks / Trade-offs

**[Risk] Behavior changes during refactor** — Extracting helpers can subtly change byte-handling. Mitigation: All 1300+ existing tests must pass; add specific tests for any new helper functions.

**[Risk] Performance regression** — Function calls (vs inlined code) can hurt hot paths. Mitigation: Mark helpers `#[inline]`. Compiler should inline them at -O2/-O3.

**[Trade-off] Increased file count** — Adding `byte_io.rs` increases module count. Worth it for the deduplication benefit.
