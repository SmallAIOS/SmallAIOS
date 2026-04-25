## Context

The SmallAIOS ONNX runtime has a working Conv implementation for the
trivial case (1×1 / 3×3 kernels, `stride=1`, `pad=0`, `group=1`,
`dilation=1`), but it silently ignores any Conv attribute that deviates
from those defaults. Two observations from the
`arm64-gpu-container-v1` benchmark harness (`onnx-rt/tests/bench_vision_models.rs`):

1. **MobileNetV2-12** fails at Conv_2: depthwise-separable convolutions
   use `group = input_channels` and weight shape `[C_in, 1, KH, KW]`.
   Our `validate_conv_inputs` check `input.dims[1] == weight.dims[1]`
   reads `C_in == 1` and hard-fails before the kernel is ever invoked.
2. **ResNet-50 v2** fails at `resnetv24_stage1__plus0`: the stem uses
   `strides = [2, 2]`. Because strides aren't honored, the CPU produces
   a 2× too large feature map; the downstream residual `Add` then hits
   mismatched spatial dimensions and reports "incompatible dimensions
   for broadcasting".

Both failures share the same root cause: the CPU path has no mechanism
to receive attributes from the graph. The dispatcher
(`onnx-rt/src/executor.rs:1671`, `dispatch_convolution`) prefixes its
attrs parameter with `_`, and the operator
(`onnx-rt/src/operators.rs:1862`, `op_conv`) has no attribute parameter
at all.

The CUDA path (`try_cuda_dispatch` at `onnx-rt/src/executor.rs:481`)
parses `pads`, `strides`, and `dilations` but not `group` and never
forwards grouping information to `cuda::conv::gpu_conv2d`
(`onnx-rt/src/cuda/conv.rs`). cuDNN supports grouped convolution
natively via `cudnnSetConvolutionGroupCount`; we just aren't calling
it.

The existing `onnx-cpu-execution` spec already requires Conv attribute
pass-through (Requirement: "Conv operator receives padding and stride
attributes"), so part of this change is correcting a conformance gap —
not inventing new behavior.

## Goals / Non-Goals

**Goals:**

- End-to-end MobileNetV2-12 inference completes on both CPU and GPU
  (hybrid) paths, producing an output tensor of shape `[1, 1000]`.
- End-to-end ResNet-50 v2 inference completes on both CPU and GPU
  paths, producing an output tensor of shape `[1, 1000]`.
- CPU and GPU outputs for the above agree to within
  `max_abs_diff < 1e-2` on the deterministic benchmark input (consistent
  with TF32 rounding on Blackwell).
- Existing Conv unit tests (1×1, 3×3, strided, dilated in the
  `test_cuda.rs` suite) and all operator tests continue to pass.
- A single `ConvAttrs::from_attributes` helper is the one-and-only
  place attribute parsing happens, shared by CPU and GPU dispatch.
- `op_conv` and `gpu_conv2d` correctly implement `group` by partitioning
  the `C_out` and `C_in` axes, matching the ONNX Conv-11 reference
  semantics.

**Non-Goals:**

- `ConvTranspose` attribute coverage. Different op; would duplicate
  most of the plumbing but touches a different kernel. Separate change.
- `QLinearConv` attribute coverage. Same plumbing concepts, but
  involves quantization scale/zero-point handling. Separate change.
- 3D (`NCDHW`) convolutions. Our current CPU kernel and GPU descriptor
  setup are 4D-only; extending to 3D changes the loop nests.
- New auto-pad modes beyond what the GPU path already accepts. If the
  model requires an unsupported auto-pad mode we surface a clear
  unsupported-attribute error rather than silently picking `VALID`.
- Performance tuning of the grouped inner loop. Correctness first. A
  follow-up can introduce im2col + GEMM fallback, AVX/NEON intrinsics,
  or dispatch grouped conv to cuBLAS batched GEMM.
- Fixing the separately-tracked SqueezeNet `[1, 4000]` CPU output shape
  bug (that's in Reshape/AveragePool/Dropout, not Conv).

## Decisions

### 1. Introduce a shared `ConvAttrs` struct, parsed once

**Decision:** Add `ConvAttrs { pads, strides, dilations, group }` in
`onnx-rt/src/operators.rs`. Provide
`ConvAttrs::from_attributes(&[AttributeProto]) -> ConvAttrs` that
parses attributes with ONNX defaults:

- `group = 1`
- `strides = [1, 1]`
- `pads = [0, 0, 0, 0]`  (top, left, bottom, right)
- `dilations = [1, 1]`
- `kernel_shape`: ignored at attribute-parse time; always inferred
  from `weight.shape`.

**Rationale:** The CPU dispatch (`dispatch_convolution`) and the
CUDA dispatch (`try_cuda_dispatch`) both currently parse attributes
independently; putting the parsing in one place eliminates drift and
makes defaults obvious. The struct fits in a cache line and can be
passed by value through the op boundary.

**Alternatives considered:**

- Keep parsing inline in both dispatch sites. **Rejected** — the CPU
  site has zero parsing today, so we'd copy-paste the GPU parser and
  immediately grow two sources of truth.
- Put the parser in a new module. **Rejected** — we only have one type
  with ~20 lines of construction. Not worth another module.

### 2. Extend `op_conv` signature with a `ConvAttrs` parameter

**Decision:** Change signature to

```rust
pub fn op_conv(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    attrs: &ConvAttrs,
) -> Result<Tensor, OpError>
```

Add a `ConvAttrs::default()` that matches ONNX defaults; existing
callers with no attrs can pass `&ConvAttrs::default()` or we can keep a
wrapper `op_conv_default(...)` for the in-tree tests to limit churn.

**Rationale:** Minimal signature change, makes attribute handling
explicit at the boundary. The builder-pattern alternative (e.g.
`ConvCall::new().with_strides(...)`) buys us nothing — Conv has a
bounded, well-understood attribute set.

### 3. Rewrite `conv_compute` to handle strides, pads, dilations, and groups

**Decision:** `conv_compute` changes in three ways:

1. Compute output spatial dimensions as
   `oh = (h + pad_t + pad_b - (kh - 1) * dilation_h - 1) / stride_h + 1`
   (mirroring the cuDNN formula) rather than assuming `oh = h - kh + 1`.
2. Partition `C_out` into `group` contiguous blocks. Output channel
   `c_out` in group `g` reads from input channels
   `[g * C_in/group, (g+1) * C_in/group)` only.
3. Apply dilation to the kernel offset; apply pad offsets to the input
   coordinate; skip reads that fall outside the input (zero padding).

The outer loop nest remains
`for n → for c_out → for oh → for ow → for kh → for kw → for c_in_in_group`
so we don't change the overall structure — just the index math.

**Rationale:** Direct implementation of the ONNX Conv-11 reference.
Keeps the implementation inspection-friendly; no novel algorithm.

**Alternatives considered:**

- im2col + `op_matmul`. **Rejected** — more efficient but adds a
  memory-reshaping pass and requires handling bias separately; worth
  doing as a performance follow-up, not in a correctness-first change.
- Special-case depthwise (`group == C_in`) with a separate kernel.
  **Rejected** — the general grouped formula handles depthwise
  correctly; a separate kernel would be a pure performance win and
  lives in a tuning change.

### 4. Fix `validate_conv_inputs` to match ONNX Conv-11

**Decision:** Replace `input.dims[1] == weight.dims[1]` with
`input.dims[1] == weight.dims[1] * group`. Also ensure
`C_out % group == 0`.

**Rationale:** Current check is wrong for any `group > 1`. The corrected
check is the ONNX spec requirement verbatim.

### 5. GPU path: forward `group` to cuDNN

**Decision:** In `try_cuda_dispatch` (Conv branch at
`executor.rs:481`), parse `group` via `ConvAttrs::from_attributes` and
pass it to `cuda::conv::gpu_conv2d`. Inside `gpu_conv2d`, after
`cudnnCreateConvolutionDescriptor`, call
`cudnnSetConvolutionGroupCount(conv_desc, group)` — this must happen
before the algorithm selection / workspace query, per cuDNN's
documented API order. If `group == 1` (the default), we can skip the
call to preserve today's behavior byte-for-byte.

**Rationale:** Matches cuDNN's documented contract. Calling with
`group == 1` is a no-op in cuDNN, but we prefer to guard the call
behind `group > 1` so the default path is unchanged.

**Implementation detail:** If `cudnnSetConvolutionGroupCount` is not
yet declared in `onnx-rt/src/cuda/ffi.rs`, add it to the `cudnn` extern
block.

### 6. Attribute-parse error handling

**Decision:** If an unknown / unsupported attribute appears on a Conv
node (e.g. `auto_pad = SAME_UPPER` when our decoder doesn't handle it
yet), return
`OpError::InvalidAttribute("Conv: unsupported auto_pad value ...")`
rather than silently using `VALID`. This forces a loud failure rather
than a silent wrong answer.

**Rationale:** We've already been bitten once by silent attribute
drop; a loud failure is more debuggable.

### 7. Test strategy: unit tests + CPU-vs-GPU diff + end-to-end

**Decision:** Three layers of testing.

1. **Unit tests** in `onnx-rt/src/operators.rs`: run `op_conv` with
   each attribute combination against a reference computed inline in
   Rust. Cover depthwise (`group = C_in`), group-of-2, stride=2,
   asymmetric pads, dilation=2.
2. **CPU-vs-GPU numerical tests** in `onnx-rt/tests/test_cuda.rs`:
   same input to `op_conv` and `gpu_conv2d`, compare outputs with
   tolerance `max_abs < 1e-3` for f32 (TF32 default precision).
3. **End-to-end**: the already-ignored-gated
   `bench_mobilenet_v2_cpu_vs_gpu` and `bench_resnet50_cpu_vs_gpu`
   tests in `onnx-rt/tests/bench_vision_models.rs` must run through
   and produce `[1, 1000]` outputs; update their pass criteria
   accordingly.

**Rationale:** Unit tests catch the algorithm bugs, CPU-vs-GPU tests
catch dispatch / attribute-forwarding bugs, end-to-end tests catch
integration bugs. Each layer is narrow and fast to localize.

## Risks / Trade-offs

- [**Risk**: silent regression on existing 1×1 / 3×3 conv tests because
  the loop math changed] → Mitigation: the new formula reduces to the
  old one exactly when `stride=1, pad=0, dilation=1, group=1`. Keep
  all existing tests in place and run them before any attribute-aware
  tests are added.
- [**Risk**: performance regression for the default case because of
  extra arithmetic in the inner loop] → Mitigation: Keep the inner-loop
  hot path branchless where the defaults apply; profile before landing
  with the `bench_squeezenet_cpu_vs_gpu` benchmark (which uses
  default-attr Conv extensively).
- [**Risk**: cuDNN algorithm selection produces a different algorithm
  when `group > 1` and the workspace size exceeds the current hardcoded
  budget] → Mitigation: query the workspace size dynamically for
  grouped conv and allocate accordingly; reuse the workspace buffer
  across inferences. If allocation fails, return
  `OpError::InternalError` with a clear message.
- [**Risk**: `cudnnSetConvolutionGroupCount` API order is subtle — it
  must be called after descriptor creation but before algorithm
  selection, and changing `group` after algorithm selection is
  undefined behavior] → Mitigation: wrap the descriptor lifecycle in a
  helper that enforces the order at compile time by building the
  descriptor, setting the group count, then immediately selecting the
  algorithm inside a single function.
- [**Risk**: ResNet-50 v2 might fail on a different operator after
  Conv is fixed — e.g. BatchNormalization has its own shape quirks]
  → Mitigation: the benchmark harness already reports per-operator
  failures; if another op emerges as blocker, it becomes a follow-up
  change and we still close MobileNetV2.
- [**Trade-off**: we chose direct naive implementation for grouped
  conv instead of im2col + GEMM] → Rationale: correctness ships first.
  A perf-tuned grouped kernel (or cuBLAS batched GEMM fallback) is a
  clean follow-up change with a clear diff baseline.

## Migration Plan

No migration — this is a purely additive correctness fix. All existing
callers pass the default attrs (matching today's behavior) and get
identical results. Grouped / strided / padded / dilated Conv works for
the first time.

If we ever need to roll back, the change is contained to `op_conv`,
`dispatch_convolution`, `try_cuda_dispatch` (Conv branch),
`gpu_conv2d`, and `ConvAttrs` — reverting the commit restores the
prior behavior.

## Open Questions

- Do we want to introduce a `ConvAttrs::from_attributes` variant that
  also carries `kernel_shape` for validation (e.g. confirm it matches
  `weight.dims[2..]`)? Skipping for now — weight shape is authoritative.
- Should we expose `ConvAttrs` publicly (`pub use`) so external callers
  can build one manually? Proposing to keep it `pub(crate)` until we
  have a second caller outside the ops/dispatch layer.
- How many of the existing ignored benchmark tests should flip to
  always-on after this change? Proposing to keep them `#[ignore]`
  because they still require DGX Spark hardware for the GPU leg; the
  CPU leg could run in CI but it's >5s per inference which is too slow
  for a default test run.
