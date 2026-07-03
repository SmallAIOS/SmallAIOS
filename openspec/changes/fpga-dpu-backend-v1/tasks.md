## 1. Phase 1 — `.xmodel` parser and offline corpus

- [ ] 1.1 Survey the XIR `.xmodel` protobuf schema; pin a Vitis AI version (e.g., 3.5) and document the schema subset we need (subgraph, attribute, op-def, tensor-def) in a parser comment block.
- [ ] 1.2 Extend the existing `onnx-rt` hand-rolled `#![no_std]` protobuf decoder with the XIR-specific message types. Do NOT introduce a new third-party protobuf crate.
- [ ] 1.3 Implement `XmodelParser` that walks the protobuf and emits a structured `XmodelGraph { subgraphs, residual_ops }` with: DPU subgraph boundaries, instruction blob byte slice, per-tensor descriptors (shape, quant scale, zero-point), weight/bias blob references.
- [ ] 1.4 Define `XmodelError` variants: `Truncated`, `BadMagic`, `MissingRequiredField(&'static str)`, `UnsupportedDpuVariant`, `IncompatibleSchemaVersion`. Each variant SHALL carry enough context to reference `docs/zynqmp-dpu.md`.
- [ ] 1.5 Implement protobuf forward-compat: skip unknown field tags per wire-format rules; emit a single warning line listing observed unknowns so newer minor Vitis AI versions degrade gracefully.
- [ ] 1.6 Synthesize `OpDescriptor::DpuSubgraph { input_tensors, output_tensors, instruction_blob_id, dpu_variant }` from each parsed subgraph; re-emit residual ops as the existing `OpDescriptor` variants the runtime already understands.
- [ ] 1.7 Commit a fixtures corpus under `tests/fixtures/dpu/`: at minimum a tiny MatMul-only `.xmodel` and a MatMul + LayerNorm + Softmax `.xmodel` (LayerNorm/Softmax are residual). These are produced offline; document the production recipe inline.
- [ ] 1.8 Add host-side unit tests covering each spec scenario in the `.xmodel` parser requirement: round-trip, mixed DPU+host, unknown tags, truncation.
- [ ] 1.9 Verify `just clippy` and `just fmt-check` pass for the new module; verify `just arch-check` reports no new layer violations.

## 2. Phase 2 — DPU register protocol and `arch/aarch64-zynqmp` driver

- [ ] 2.1 Read AMD PG338 *DPU IP Product Guide* for the K26 stock DPU (DPUCZDX8G B4096); enumerate the control registers we need (DPU_VER, DPU_CTRL, DPU_INSTR_BASE, DPU_INSTR_LEN, DPU_INTR_STATUS, DPU_INTR_CLEAR) and document offsets in `arch/aarch64-zynqmp::dpu::regs`.
- [ ] 2.2 Implement `DpuPeripheral` wrapping the AXI-mapped DPU control region using the typed register-access primitives from the AXI framework.
- [ ] 2.3 Implement `DpuPeripheral::probe()` that reads `DPU_VER`, validates the variant byte against the supported list, returns `Err(BackendUnavailable)` cleanly on mismatch / unreadable.
- [ ] 2.4 Implement instruction-stream submission: program `DPU_INSTR_BASE` and `DPU_INSTR_LEN`, set start bit in `DPU_CTRL`, ensure required AArch64 barriers around the doorbell write.
- [ ] 2.5 Implement IRQ handler registration via the GIC-400 driver: the DPU SPI line (board constant) wakes a per-dispatch completion `Waker`. Use `core::task::Waker`-based async pattern consistent with the rest of the kernel scheduler.
- [ ] 2.6 Implement fault detection: read `DPU_INTR_STATUS` in the IRQ handler, distinguish "completion" vs "instruction-fault" vs "timeout"; return a structured `DpuCompletion` enum the backend can map to `Ok(())` or `ExecError::FallbackToCpu`.
- [ ] 2.7 Define a `DpuHandle` struct exported from `arch/aarch64-zynqmp::dpu` carrying the bound peripheral + IRQ subscription + DMA channel handles. This is what the runtime consumes.
- [ ] 2.8 Add a polling-only diagnostic path behind a non-default `dpu-polling-debug` feature; ensure the feature is not enabled in any default workspace build, in CI matrix, or in any `just` recipe other than an explicit `dpu-bringup-poll` recipe.
- [ ] 2.9 Add unit tests using the AXI framework's debug-mode peripheral mock: probe-success, probe-failure, dispatch-completion, dispatch-fault, dispatch-timeout. No real hardware required.
- [ ] 2.10 Add a QEMU smoke test that boots the kernel with the DPU feature on and confirms `DpuBackend::probe` returns `BackendUnavailable` cleanly (since QEMU's ZynqMP machine model does not emulate the DPU). DEFERRED until Phase 3 wires `DpuBackend`.

## 3. Phase 3 — `DpuBackend` wiring, fallback, instrumentation

- [ ] 3.1 Create `onnx-rt::backend::dpu` module behind a non-default `dpu` Cargo feature.
- [ ] 3.2 Implement `DpuBackend::new(handle: DpuHandle)`. No AXI addresses or vendor symbols leak out of this module.
- [ ] 3.3 Implement `ExecutionBackend` for `DpuBackend`: `name() -> "dpu"`, `probe()` delegating to `DpuPeripheral::probe`, `can_run` returning true only for `OpDescriptor::DpuSubgraph` with a matching variant, `dispatch` orchestrating DMA + instruction submit + IRQ wait.
- [ ] 3.4 Implement `estimated_ns(&self, op)` returning a static placeholder (e.g., 1_000_000 ns sentinel). Refine once Phase 5 collects perf data.
- [ ] 3.5 Wire activation tensors to `DmaBuffer<HpcPort>`; weights and instruction blobs to `DmaBuffer<HpPort>` with explicit `clean_for_device()` at load time. Verify the type system rejects misuse with a compile-fail test.
- [ ] 3.6 Implement session-build extension: when the input model is ONNX accompanied by a `.xmodel` sidecar and the `dpu` feature is on, run the parser, splice synthetic `DpuSubgraph` ops + residual ops into the dispatch graph. When `.xmodel` is missing, fall through to pure-ONNX/CPU dispatch.
- [ ] 3.7 Implement `dpu-profile` feature: per-dispatch counters (latency ns, bytes in, bytes out, IRQ wait ns), per-op-type aggregation, summary at `DpuBackend::drop` to stderr. Verify zero overhead in default builds via a code-size check.
- [ ] 3.8 Add the analysis script at `tools/dpu-profile/parse.py` that reads the stderr summary into a structured CSV/JSON for `fpga-custom-npu-v1` to consume.
- [ ] 3.9 Add an end-to-end test (host-side, using mocked `DpuHandle` + the tiny MatMul fixture): load the `.xmodel`, build the session with `[DpuBackend, CpuBackend]`, simulate completion via the mock, verify the dispatch table places `DpuSubgraph` on `DpuBackend` and residuals on `CpuBackend`, verify outputs match the CPU reference within quant tolerance.
- [ ] 3.10 Add a session-build test for the missing-CpuBackend case: residual op + `[DpuBackend]` only ⇒ `OnnxError::NoBackendForOp`.
- [ ] 3.11 Verify `just clippy`, `just fmt-check`, `just test`, `just arch-check` all pass with the `dpu` feature enabled.

## 4. Phase 4 — Documentation and offline pipeline

- [ ] 4.1 Write `docs/zynqmp-dpu.md`: offline pipeline (ONNX → Brevitas-quantized ONNX → Vitis AI compile → `.xmodel`), version pins (Vitis AI, Brevitas, AMD K26 BSP), the K26 stock bitstream provenance, BOOT.BIN packaging notes, the QEMU caveat, supported-DPU-variant list, troubleshooting matrix.
- [ ] 4.2 Document the `dpu-profile` summary format and link to `tools/dpu-profile/parse.py`.
- [ ] 4.3 Document fallback semantics from a user's perspective: which ONNX ops typically fall to CPU after Vitis AI compile, what perf characteristics to expect, when to suspect a residual-op blow-up.
- [ ] 4.4 Document the `dpu-polling-debug` feature, its bring-up-only status, and that it MUST NOT ship in production.
- [ ] 4.5 Update `docs/accelerator-hal.md` with a "real backend example" section pointing at `DpuBackend` as the worked example of a production `ExecutionBackend` impl.
- [ ] 4.6 Update `CLAUDE.md` "Crate Feature Flags" section with `dpu`, `dpu-profile`, and `dpu-polling-debug` entries.

## 5. Phase 5 — `just` recipes and CI integration

- [ ] 5.1 Add `just build-kernel-arm-zynqmp-dpu` recipe that builds the kernel with the `dpu` feature on.
- [ ] 5.2 Add `just run-arm-zynqmp-dpu` recipe that boots the kernel in QEMU with a stock-DPU-bearing bitstream artifact in the boot image. The recipe SHALL print a banner stating the QEMU caveat before launch.
- [ ] 5.3 Add a CI matrix entry (advisory at first) building with `--features dpu`. Verify clippy, fmt, and the host-side unit tests pass.
- [ ] 5.4 Add a CI advisory job running the `.xmodel` parser unit tests against the `tests/fixtures/dpu/` corpus.
- [ ] 5.5 Add a CI gate ensuring no symbols from the `dpu` module leak into a default (no-`dpu`-feature) build (e.g., grep the produced binary for `dpu::` strings).

## 6. Phase 6 — Real-hardware bring-up plan (deferred until KV260/KR260 available)

- [ ] 6.1 DEFERRED Acquire a KV260 (or KR260) board with the K26 stock bitstream loaded.
- [ ] 6.2 DEFERRED Bring up the FSBL+ATF+SmallAIOS BOOT.BIN per `docs/zynqmp-boot.md` with the DPU bitstream included.
- [ ] 6.3 DEFERRED Confirm `DpuBackend::probe()` returns `Ok(())` against the real DPU.
- [ ] 6.4 DEFERRED Run the tiny MatMul `.xmodel` end-to-end on hardware; verify completion IRQ fires; verify output matches CPU reference within quant tolerance.
- [ ] 6.5 DEFERRED Run a representative target model (small CNN; small transformer if it compiles); collect `dpu-profile` measurements; commit the report to `docs/perf/dpu-baseline.md`.
- [ ] 6.6 DEFERRED Validate the cache-coherency port choices on real silicon: under sustained load, no stale-data faults observed across 10⁵ inferences.
- [ ] 6.7 DEFERRED Hand off the perf report as the trigger for `fpga-custom-npu-v1` to begin its design phase.

## 7. Phase 7 — Verification and archive prep

- [ ] 7.1 Run `/opsx:verify` against this change before requesting archive.
- [ ] 7.2 Confirm every spec requirement has a corresponding test (host-side or deferred-real-hardware).
- [ ] 7.3 Confirm `just dsm-analyze` shows no new coupling clusters or layer violations introduced by the `dpu` module.
- [ ] 7.4 Update the change's Followers section: link the perf report from 6.5 as the input to `fpga-custom-npu-v1`.
- [ ] 7.5 Open the next change shells if not already open: confirm `fpga-custom-npu-v1` and `fpga-manager-v1` are still in roadmap state and reference this change as predecessor.
