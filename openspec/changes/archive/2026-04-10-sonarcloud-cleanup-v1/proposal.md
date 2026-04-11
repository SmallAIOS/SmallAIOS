## Why

SonarCloud reports 16 code complexity and duplication issues on the develop branch. While none are bugs, they reduce maintainability and increase the risk of future defects. Most are concentrated in the ONNX runtime files (`executor.rs`, `operators.rs`) where the pace of recent feature work prioritized correctness over refactoring. With the inference pipeline now stable (PRs #55, #57, #58, #60, #61, #62, #63 all merged), it's a good time to clean these up.

## What Changes

- Extract repeated `f32::from_le_bytes([data[i*4], ...])` patterns into helper functions (appears 6+ times in executor.rs, 50+ times in operators.rs)
- Reduce cognitive complexity of `dispatch_node()` (~45 → <15) by extracting per-operator dispatch helpers
- Reduce complexity of `op_cast()` by splitting into per-conversion functions
- Introduce typed helper for byte ↔ numeric encoding (`encode_numeric_to_bytes`, `read_i32`, `read_i64`, etc.)
- Replace magic numbers in `expf_approx()` with named constants
- Extract repeated patterns in `op_concat()`, `op_transpose()`, `op_squeeze()`
- Bundle `op_conv` parameters into a `ConvParams` struct (reduce 9 args → 4)
- Extract DHCP option parsing duplication in `net/src/dhcp.rs`
- Add `F32_SIZE` constant and `allocate_tensor_data()` helper to eliminate the `total * 4` magic number

## Capabilities

### Modified Capabilities
- `onnx-runtime`: No requirement changes — refactoring only, behavior identical
- `networking`: Same — DHCP behavior unchanged

## Impact

- **Code:** Refactoring only across `onnx-rt/src/executor.rs`, `onnx-rt/src/operators.rs`, `kernel/src/syscall/mod.rs`, `net/src/dhcp.rs`
- **Tests:** All 246 onnx-rt tests + 1128 bus tests must continue passing — refactoring is behavior-preserving
- **Coverage:** Should improve since smaller functions are easier to test
- **No new dependencies:** Pure refactoring
