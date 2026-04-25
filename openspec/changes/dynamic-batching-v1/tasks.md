## 1. Stack / unstack helpers

- [ ] 1.1 Create `onnx-rt/src/batch.rs` module
- [ ] 1.2 Implement `pub fn stack_along_batch_axis(tensors: &[&Tensor]) -> Result<Tensor, OpError>` — validates ranks/shapes/dtypes, allocates a single buffer of size `N * per_image_bytes`, copies each input via `copy_from_slice`, returns a tensor of shape `[N, D1, D2, ...]`
- [ ] 1.3 Implement `pub fn unstack_along_batch_axis(tensor: &Tensor) -> Result<Vec<Tensor>, OpError>` — splits a `[N, ...]` tensor into N `[...]` tensors via `copy_from_slice`
- [ ] 1.4 Wire `pub mod batch;` in `onnx-rt/src/lib.rs`
- [ ] 1.5 Unit tests in `batch.rs::tests`: stack of 4 f32 `[3, 4]` tensors → `[4, 3, 4]`; unstack `[4, 3, 4]` → 4 × `[3, 4]`; round-trip identity; rank mismatch errors; dtype mismatch errors; empty-input error

## 2. SessionConfig + BatchPolicy

- [ ] 2.1 Add `pub enum BatchPolicy { Disabled, Static(usize), Dynamic { max: usize, pad: bool } }` with `#[derive(Default)] #[default] Disabled` in `onnx-rt/src/session.rs`
- [ ] 2.2 Add `pub batch_policy: BatchPolicy` field to `SessionConfig`
- [ ] 2.3 Update `Default for SessionConfig` to include `batch_policy: BatchPolicy::default()`
- [ ] 2.4 Add new `SessionError` variants: `BatchPolicyViolation(String)`, `BatchShapeMismatch(String)`, `BatchEmpty`
- [ ] 2.5 Update SessionConfig literals in `tests/integration_inference.rs` etc. to include `batch_policy`

## 3. `Session::run_batched` API

- [ ] 3.1 Add `pub fn run_batched(&self, inputs: &[InferenceInput], batch_size: usize) -> Result<Vec<InferenceOutput>, SessionError>`
- [ ] 3.2 Validate against `self.config.batch_policy` (return `BatchPolicyViolation` on mismatch)
- [ ] 3.3 Group inputs by name; verify all groups have the same length and all images per group share shape+dtype (return `BatchShapeMismatch` otherwise)
- [ ] 3.4 If `Dynamic { max, pad: true }` and the input count `K < max`, append `max - K` repetitions of the last image to each group
- [ ] 3.5 Stack each name's group via `batch::stack_along_batch_axis` to produce a batched `Tensor`
- [ ] 3.6 Build a single-element `[(name, stacked_tensor)]` slice and dispatch through the existing executor (`execute_graph` or `execute_graph_hybrid` as configured)
- [ ] 3.7 Unstack each output via `batch::unstack_along_batch_axis` and return the first `K` per name (discarding padded outputs when present)

## 4. `Session::run` becomes a shim

- [ ] 4.1 Rewrite `Session::run(inputs)` as `self.run_batched_internal(inputs, 1, /* skip_policy_check */ true)` — single-image path always allowed regardless of `BatchPolicy`
- [ ] 4.2 Verify existing tests (`integration_inference.rs`, `test_real_model.rs`) still pass without modification

## 5. Throughput benchmarks

- [ ] 5.1 Add `BenchMode::HybridBatched(usize)` variant to `bench_vision_models.rs`
- [ ] 5.2 Add `bench_resnet50_throughput_b1` (uses `run`, baseline)
- [ ] 5.3 Add `bench_resnet50_throughput_b4` / `_b16` / `_b64` using `BatchPolicy::Static(N)` and `run_batched`
- [ ] 5.4 Each bench runs a fixed iteration count (e.g. 100 batches) and reports images-per-second
- [ ] 5.5 Add a similar set for SqueezeNet and MobileNetV2 (B=1 / B=16 / B=64)
- [ ] 5.6 Run all on DGX Spark; record results in `docs/benchmarks/arm64-gpu-cpu-vs-gpu.md`
- [ ] 5.7 Verify throughput targets: B=4 ≥ 3.5×B=1, B=16 ≥ 10×B=1, B=64 ≥ 20×B=1 on ResNet-50

## 6. Per-op batch sanity tests

- [ ] 6.1 In `onnx-rt/src/operators.rs::tests` — add `test_op_matmul_batch4_matches_n1`: run `op_matmul` on 4 separate `B=1` cases vs a single `B=4` stacked call; assert results match
- [ ] 6.2 Same for `op_conv` with `B=4`
- [ ] 6.3 Same for `op_relu`, `op_batch_normalization`, `op_averagepool` (each with `B=4`)
- [ ] 6.4 In `onnx-rt/tests/test_cuda.rs` (cuda feature) — add `test_gpu_conv2d_batched_matches_n1` and `test_gpu_batchnorm_batched_matches_n1`

## 7. Hybrid path batch validation

- [ ] 7.1 Add an integration test `test_hybrid_batched_mlp_equivalent_to_n1` in `bench_vision_models.rs`: run the MLP fixture once with B=1 and once with B=4 (4 distinct inputs), compare the unstacked B=4 outputs to the 4 individual B=1 outputs within `1e-4`
- [ ] 7.2 Run with `BatchPolicy::Static(4)` + `GpuResidency::Hybrid` (without graph capture) — assert correctness
- [ ] 7.3 Run with `BatchPolicy::Static(4)` + `GpuResidency::Hybrid` + `CudaGraphMode::Capture` (interaction with `cuda-graphs-v1` if it lands first) — assert correctness and that the capture cache holds exactly one `B=4` graph

## 8. Documentation

- [ ] 8.1 Add a "Batched Inference" section to `docs/architecture.md` explaining `BatchPolicy` semantics and the stacking model
- [ ] 8.2 Update `docs/benchmarks/arm64-gpu-cpu-vs-gpu.md` with throughput results
- [ ] 8.3 Document the `Session::run_batched` API in inline doc comments
- [ ] 8.4 Migration note for `SessionConfig::max_batch_size`: explain the deprecation in favor of `batch_policy`

## 9. Error path tests

- [ ] 9.1 `test_run_batched_disabled_policy_errors`: default config rejects `run_batched`
- [ ] 9.2 `test_run_batched_static_count_mismatch_errors`: `Static(4)` rejects 3 or 5 inputs
- [ ] 9.3 `test_run_batched_dynamic_max_exceeded_errors`: `Dynamic { max: 8 }` rejects 16 inputs
- [ ] 9.4 `test_run_batched_shape_mismatch_errors`: differing shapes cause `BatchShapeMismatch`
- [ ] 9.5 `test_run_batched_empty_errors`: zero-input call produces `BatchEmpty`
- [ ] 9.6 `test_run_batched_dynamic_pad_repeats_last_input`: `Dynamic { max: 4, pad: true }` with 2 inputs produces 2 outputs that match the no-padding 2-input result

## 10. Final verification

- [ ] 10.1 `cargo fmt -p smallaios-onnx-rt`
- [ ] 10.2 `cargo clippy -p smallaios-onnx-rt --no-default-features --features cpu -- -D warnings`
- [ ] 10.3 `cargo clippy -p smallaios-onnx-rt --no-default-features --features cuda -- -D warnings`
- [ ] 10.4 `cargo test -p smallaios-onnx-rt --no-default-features --features cpu` — full suite green
- [ ] 10.5 `cargo test -p smallaios-onnx-rt --lib --no-default-features --features cuda` — full suite green
- [ ] 10.6 `cargo test -p smallaios-onnx-rt --release --test bench_vision_models --features cuda -- --ignored --nocapture` — all throughput targets met
- [ ] 10.7 `openspec validate dynamic-batching-v1 --strict` passes
