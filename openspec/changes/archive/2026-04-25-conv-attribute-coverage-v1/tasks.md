## 1. Shared `ConvAttrs` parser

- [x] 1.1 Define `pub(crate) struct ConvAttrs { pads: [i32; 4], strides: [i32; 2], dilations: [i32; 2], group: i32 }` in `onnx-rt/src/operators.rs`
- [x] 1.2 Implement `impl Default for ConvAttrs` with ONNX defaults (`group = 1`, `strides = [1, 1]`, `pads = [0, 0, 0, 0]`, `dilations = [1, 1]`)
- [x] 1.3 Implement `ConvAttrs::from_attributes(attrs: &[AttributeProto]) -> Result<ConvAttrs, OpError>`
- [x] 1.4 Reject unsupported attributes (e.g. unknown `auto_pad` values) with `OpError::InvalidAttribute`
- [x] 1.5 Ignore benign unrelated attributes (`kernel_shape`) without error
- [x] 1.6 Unit test: `from_attributes` with no attrs returns `ConvAttrs::default()`
- [x] 1.7 Unit test: `from_attributes` populates each field from its corresponding `AttributeProto`
- [x] 1.8 Unit test: `from_attributes` returns `OpError::InvalidAttribute` for an unsupported `auto_pad` value

## 2. CPU validation + kernel rewrite

- [x] 2.1 Change `op_conv` signature to `op_conv(input, weight, bias, attrs: &ConvAttrs)` and update all in-tree callers
- [x] 2.2 Update `validate_conv_inputs` to check `input.dims[1] == weight.dims[1] * group` and `C_out % group == 0`
- [x] 2.3 Return `OpError::ShapeMismatch` with a message naming `C_in`, `weight.dims[1]`, and `group` on violation
- [x] 2.4 Unit test: validator accepts depthwise shapes (`[1, 32, ·, ·]` + `[32, 1, ·, ·]` + `group = 32`)
- [x] 2.5 Unit test: validator rejects `C_out % group != 0`
- [x] 2.6 Rewrite `conv_compute` to compute output spatial dims via the ONNX Conv-11 formula
- [x] 2.7 Rewrite the input-coordinate calculation to apply `pads` (top/left/bottom/right) and `dilations`
- [x] 2.8 Rewrite the channel loop to partition `C_out` into `group` blocks, restricting input-channel range per block
- [x] 2.9 Preserve the existing fast path for `stride=1, pad=0, dilation=1, group=1` — no branches in the hot loop for default case
- [x] 2.10 Unit test: `stride = [2, 2]` produces the expected half-spatial output shape
- [x] 2.11 Unit test: asymmetric `pads = [1, 1, 2, 2]` produces correct output dims
- [x] 2.12 Unit test: `dilations = [2, 2]` produces correct output dims and numerical values
- [x] 2.13 Unit test: `group = 2` splits `C_out` into two independent halves (manually compute the reference)
- [x] 2.14 Unit test: depthwise (`group = C_in`) matches a reference CPU loop written inline in the test
- [x] 2.15 Regression: existing 1×1 and 3×3 default Conv tests pass unchanged

## 3. CPU dispatch wiring

- [x] 3.1 In `executor.rs::dispatch_convolution`, replace `_attrs` with a real attrs-consuming parameter
- [x] 3.2 Parse attrs via `ConvAttrs::from_attributes` and return any parse error verbatim
- [x] 3.3 Pass the parsed `&ConvAttrs` to `op_conv`
- [x] 3.4 Update any in-tree tests that invoke `op_conv` directly (not through dispatch) to pass `&ConvAttrs::default()`

## 4. GPU dispatch wiring

- [x] 4.1 In `executor.rs::try_cuda_dispatch` Conv branch, replace the inline attr parsing with a `ConvAttrs::from_attributes` call
- [x] 4.2 Extend the signature of `cuda::conv::gpu_conv2d` to accept `group: i32` (or a full `&ConvAttrs`)
- [x] 4.3 If `cudnnSetConvolutionGroupCount` is not yet bound in `onnx-rt/src/cuda/ffi.rs`, add the extern declaration
- [x] 4.4 In `gpu_conv2d`, call `cudnnSetConvolutionGroupCount(conv_desc, group)` immediately after `cudnnCreateConvolutionDescriptor` and before the algorithm / workspace query
- [x] 4.5 Guard the call behind `group > 1` so the default path stays byte-identical for `group == 1`
- [x] 4.6 Ensure workspace allocation uses `cudnnGetConvolutionForwardWorkspaceSize` after the group count is set (workspace may grow) — existing `gpu_conv2d` already queries workspace size after descriptor setup, so group-count propagation is automatic.
- [x] 4.7 Release or reuse the workspace buffer cleanly across inferences; verify no leak on repeated calls — existing `DeviceBuffer::alloc` + RAII drop already handles this; the forward-bench `bench_squeezenet_cpu_vs_gpu` exercises repeated inferences and VRAM readings remained stable.

## 5. GPU tests (`onnx-rt/tests/test_cuda.rs`, all `#[ignore]`-gated)

- [x] 5.1 `test_gpu_conv_group2_matches_cpu`: group=2 Conv, compare GPU vs CPU output within `1e-3`
- [x] 5.2 `test_gpu_conv_depthwise_matches_cpu`: `group = C_in`, input `[1, 32, 14, 14]` kernel `[32, 1, 3, 3]`
- [x] 5.3 `test_gpu_conv_stride2_matches_cpu`: `strides = [2, 2]`, verify output shape halves spatially
- [x] 5.4 `test_gpu_conv_pad1_matches_cpu`: explicit `pads = [1, 1, 1, 1]` on a 3×3 kernel keeps spatial dims
- [x] 5.5 `test_gpu_conv_dilation2_matches_cpu`: `dilations = [2, 2]`
- [x] 5.6 Regression: re-run the existing `test_gpu_conv2d_*` default-attribute tests, assert they still pass

## 6. End-to-end benchmark validation

- [x] 6.1 Update `bench_mobilenet_v2_cpu_vs_gpu` to hard-assert output shape `[1, 1000]` once both paths succeed — added via `run_cpu_vs_gpu_with(expected_output_dims = Some(&[1, 1000]))`. MobileNetV2 currently still fails at a downstream `Gather` op, so the assertion is gated on both paths succeeding.
- [x] 6.2 Update `bench_resnet50_cpu_vs_gpu` to hard-assert output shape `[1, 1000]` once both paths succeed — same mechanism; ResNet-50 now runs end-to-end and the assertion fires, both paths return `[1, 1000]`.
- [x] 6.3 Run both benches on DGX Spark; record the CPU-vs-GPU output diff and latency in `docs/benchmarks/arm64-gpu-cpu-vs-gpu.md`
- [x] 6.4 Update `docs/benchmarks/arm64-gpu-cpu-vs-gpu.md` to move MobileNetV2 and ResNet-50 from "FAILED" into the passing table — ResNet-50 moved into passing row with 1.17× speedup. MobileNetV2 documented as still blocked (on `Gather` now, not Conv); has its own spawned follow-up task.
- [x] 6.5 Verify `bench_squeezenet_cpu_vs_gpu` still passes (default-attr Conv path must not regress) — passes; as a bonus the change fixed the previously documented `[1, 4000]` vs `[1, 1000]` CPU shape bug (root cause was a stride-ignoring Conv).
- [x] 6.6 Verify `bench_mlp_cpu_vs_gpu` still passes (Gemm-only model unaffected, but smoke check) — passes with 3.66× speedup.

## 7. Final verification

- [x] 7.1 `cargo fmt -p smallaios-onnx-rt`
- [x] 7.2 `cargo clippy -p smallaios-onnx-rt --no-default-features --features cpu -- -D warnings` clean
- [x] 7.3 `cargo clippy -p smallaios-onnx-rt --no-default-features --features cuda -- -D warnings` clean
- [x] 7.4 `cargo test -p smallaios-onnx-rt --no-default-features --features cpu` — 874 passed (14 new Conv tests).
- [x] 7.5 `cargo test -p smallaios-onnx-rt --lib --no-default-features --features cuda` — 878 passed.
- [x] 7.6 `cargo test -p smallaios-onnx-rt --release --test bench_vision_models --features cuda -- --ignored --nocapture` — MLP, SqueezeNet, ResNet-50 succeed; MobileNetV2 now hits a downstream `Gather` op gap (logged as its own follow-up task).
- [x] 7.7 `openspec validate conv-attribute-coverage-v1 --strict` passes.
