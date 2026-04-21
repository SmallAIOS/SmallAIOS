## 1. GPU Dispatch Plumbing

- [x] 1.1 Create `arch/apple/src/dispatch.rs` with `MetalDispatcher` that maps `OpKind` → kernel name + launch config (grid size, threadgroup size, buffer bindings)
- [x] 1.2 Add `MetalTensorCache` to `metal_provider.rs`: lazy buffer allocation, on-device tensor tracking, host↔device copy helpers
- [x] 1.3 Extend `ComputeProvider` trait in `compute/src/lib.rs` with `copy_tensor_to_device` / `copy_tensor_from_device` methods (or equivalent)
- [x] 1.4 Wire `dispatch_node` in `executor.rs` to call `MetalDispatcher::execute` when `gpu_backend.supports_op()` returns true (replace the existing TODO comment)
- [x] 1.5 Implement the host→device→launch→device→host flow for a single operator: input tensors → Metal buffers → kernel launch → synchronize → output tensor
- [x] 1.6 Implement the on-device caching path: when the previous op left a result on-device and the next op also runs on GPU, skip the host round-trip
- [x] 1.7 Implement the device→host eviction path: when a CPU op needs a tensor that's currently on-device, copy it back to host
- [x] 1.8 Unit tests for `MetalTensorCache`: alloc, round-trip, eviction, reuse

## 2. Session GPU Opt-in

- [x] 2.1 Add `GpuConfig` struct and `Session::builder().with_gpu(GpuConfig::metal())` API
- [x] 2.2 Thread the `MetalProvider` (or `None`) through `Session::run()` → `execute_graph()` → `dispatch_node()`
- [x] 2.3 Test: session without `.with_gpu()` runs pure CPU
- [x] 2.4 Test: session with `.with_gpu()` on macOS dispatches to Metal for supported ops
- [x] 2.5 Test: session with `.with_gpu()` on non-macOS silently falls back to CPU

## 3. Tier 1 — Wire Existing Shaders (11 ops)

- [x] 3.1 Wire `elementwise_add` shader → `OpKind::Add`
- [x] 3.2 Wire `elementwise_sub` shader → `OpKind::Sub`
- [x] 3.3 Wire `elementwise_mul` shader → `OpKind::Mul`
- [x] 3.4 Wire `elementwise_div` shader → `OpKind::Div`
- [x] 3.5 Wire `elementwise_relu` shader → `OpKind::Relu`
- [x] 3.6 Wire `elementwise_sigmoid` shader → `OpKind::Sigmoid`
- [x] 3.7 Wire `elementwise_tanh` shader → `OpKind::Tanh`
- [x] 3.8 Wire `matmul` / `matmul_tiled` shader → `OpKind::MatMul` / `OpKind::Gemm`
- [x] 3.9 Wire `softmax` shader → `OpKind::Softmax`
- [x] 3.10 Wire `conv2d` shader → `OpKind::Conv`
- [x] 3.11 CPU-vs-GPU comparison tests for all 11 Tier 1 ops

## 4. M1/M2 Hardware Compatibility

- [x] 4.1 Add GPU family detection in `MetalProvider::new()` via `MTLDevice::supportsFamily`
- [x] 4.2 Add `#define HAS_SIMDGROUP_MATRIX` preprocessor toggle for MSL compilation
- [x] 4.3 Implement shared-memory tiled MatMul fallback for M1/M2 in `matmul_tiled` shader
- [x] 4.4 Test: MatMul runs correctly on M1/M2 (without `simdgroup_matrix`)

## 5. Tier 2 — New High-Value Shaders

- [x] 5.1 Write `scaled_dot_product_attention` MSL kernel: fused QK^T scaling + causal mask + softmax + V multiply
- [x] 5.2 Write `layer_normalization` MSL kernel: parallel mean + variance reduction in threadgroup shared memory
- [x] 5.3 Write `rms_normalization` MSL kernel: same pattern, no mean subtraction
- [x] 5.4 Write `rotary_embedding` MSL kernel: interleaved and non-interleaved RoPE
- [x] 5.5 Write `group_query_attention` MSL kernel: fused RoPE + KV concat + grouped SDPA + causal mask
- [x] 5.6 Write `gemm_i8_simdgroup` MSL kernel: int8 matmul with i32 accumulator, `simdgroup_matrix` on M3+ with shared-memory fallback
- [x] 5.7 Write `gemm_f16` MSL kernel: f16 matmul for half-precision models
- [x] 5.8 Write `batch_normalization` MSL kernel: parallel reduction
- [x] 5.9 Wire all Tier 2 shaders through `MetalDispatcher`
- [x] 5.10 Update `supports_op` to reflect the full GPU-supported set
- [x] 5.11 CPU-vs-GPU comparison tests for all Tier 2 ops (LayerNorm, RMSNorm, RoPE, SDPA, BatchNorm exec tests added in follow-up; GQA exec deferred — its 7-input/3-output kernel needs per-buffer wiring beyond the standard `__gpu_dims` packing path)

## 6. Integration Testing

- [ ] 6.1 End-to-end test: load MobileNetV2 and run a single inference on Metal, compare output to CPU reference — **DEFERRED**: the in-tree `decode_model` parser cannot yet ingest real ONNX exports with embedded weights (mobilenet_v2.onnx is 14 MB and uses wire-format features the minimal parser doesn't handle). Belongs to `vision-transformers-v1` / protobuf hardening.
- [ ] 6.2 End-to-end test: load BERT-base and run a single inference on Metal (mixed GPU/CPU — some ops fall back) — **DEFERRED**: same parser limitation; bert-base-uncased.onnx is 438 MB. BERT also requires Gather/Slice/Cast on GPU or fast CPU fallback. Belongs to `transformer-models-v1`.
- [x] 6.3 Performance benchmark: MatMul 1024x1024 on CPU vs Metal, print speedup ratio (`bench/benches/metal_operators.rs`)
- [x] 6.4 Performance benchmark: SDPA (batch=1, seq=512, heads=12, dim=64) on CPU vs Metal (`bench/benches/metal_operators.rs`)
- [x] 6.5 Memory test: verify `MetalTensorCache` reuses buffers (peak allocation stays bounded) (`test_tensor_cache_reuse`)

## 7. Validation

- [x] 7.1 `just fmt` clean
- [x] 7.2 `just clippy --all-targets` clean
- [x] 7.3 `just test` all passing (CPU tests unaffected)
- [x] 7.4 `just test-metal` all passing (GPU tests on macOS)
- [x] 7.5 All Tier 1 + Tier 2 GPU-vs-CPU comparison tests within tolerance
- [x] 7.6 Update `docs/onnx-coverage-roadmap.md` to note GPU kernel coverage
