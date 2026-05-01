## 1. Phase 1 — ExecutionBackend trait + CPU refactor

- [ ] 1.1 Define `ExecutionBackend` trait in `onnx-rt::backend` (object-safe, `#![no_std]`, vendor-neutral). Include `can_run`, `dispatch`, `probe`, `name`, and an `estimated_cost` hook for future cost-based selection.
- [ ] 1.2 Define supporting types: `OpDescriptor`, `TensorEnv`, `ExecError` (with `FallbackToCpu`, `BackendUnavailable`, `Internal` variants), `BackendError`.
- [ ] 1.3 Refactor existing host-CPU dispatch (NEON/SVE/AVX/AVX-512 kernels) into a `CpuBackend` struct implementing `ExecutionBackend`. No behavior change; outputs must be byte-identical.
- [ ] 1.4 Update `SessionConfig` with a `backends: Vec<Box<dyn ExecutionBackend>>` field (heapless equivalent for `no_std`). Default = `[CpuBackend::new()]`.
- [ ] 1.5 Implement session-build dispatch table: walk the graph, call `can_run` per op, bind to first matching backend, fail with `OnnxError::NoBackendForOp` if none cover.
- [ ] 1.6 Refactor inference loop to dispatch ops via the precomputed table; remove the legacy CPU-specific dispatcher path.
- [ ] 1.7 Implement per-op fallback semantics: on `Err(ExecError::FallbackToCpu)`, retry with the next backend in priority order; `OnnxError::DispatchExhausted` when none succeed; log the fallback.
- [ ] 1.8 Add unit tests proving byte-identical CPU output before vs after refactor on a representative model set.
- [ ] 1.9 Add unit tests for fallback ordering and `NoBackendForOp` detection at session build.
- [ ] 1.10 Verify `just clippy` and `just fmt-check` pass; verify `just arch-check` reports no new layer violations.

## 2. Phase 2 — QEMU stub backend

- [ ] 2.1 Specify the stub MMIO device register layout (control, status, descriptor base, IRQ-clear) in a doc comment + table in `docs/qemu-stub-device.md`.
- [ ] 2.2 Decide implementation vehicle: QEMU `-device` C patch vs out-of-tree QEMU plugin. Document in `docs/qemu-stub-device.md`. Default: QEMU device patch maintained in `tools/qemu-stub/`.
- [ ] 2.3 Implement the stub QEMU device with deterministic op semantics for at minimum INT8/FP32 MatMul; a fallback "checksum" path for unsupported ops; a configurable fixed latency.
- [ ] 2.4 Implement `QemuStubBackend` in `onnx-rt::backend::qemu_stub` behind a `qemu-stub` Cargo feature.
- [ ] 2.5 `QemuStubBackend::probe` detects the stub device via known MMIO signature; returns `Err(BackendUnavailable)` cleanly when absent.
- [ ] 2.6 Wire `QemuStubBackend` to the AXI/DMA framework from §3 for input/output tensor transfer.
- [ ] 2.7 Add a `just run-arm-zynqmp-stub` recipe that boots SmallAIOS in QEMU with the stub device attached.
- [ ] 2.8 Add an end-to-end test: load a tiny MatMul-only ONNX model, confirm dispatch hits `QemuStubBackend`, output matches `CpuBackend` within f32 epsilon.
- [ ] 2.9 Add a CI job (advisory at first, gate later) that runs the QEMU stub end-to-end test.
- [ ] 2.10 Confirm the `qemu-stub` feature is off by default and no QEMU symbols leak into the default build.

## 3. Phase 3 — AXI/AXI-DMA framework

- [ ] 3.1 Decide framework location (start in `arch/aarch64-zynqmp::axi`; plan extraction to `drivers/axi` if >500 LOC).
- [ ] 3.2 Implement typed register-access primitives with width-checked compile-time errors and AArch64 `DMB`/`DSB` barriers.
- [ ] 3.3 Implement `AxiPeripheral` abstraction for register-mapped IPs.
- [ ] 3.4 Implement scatter-gather AXI-DMA driver: descriptor-ring management, MM2S and S2MM channels, IRQ-driven async completion.
- [ ] 3.5 Implement `DmaBuffer<P: AxiPort>` with phantom-typed port discrimination. Provide `HpcPort` (coherent, no maintenance API) and `HpPort` (non-coherent, with `clean_for_device`/`invalidate_for_cpu`).
- [ ] 3.6 Implement cache maintenance helpers (`DC CIVAC`, `DC IVAC`, `DC CVAC` as appropriate) for `HpPort` buffers; ensure they are unavailable on `HpcPort` (compile error).
- [ ] 3.7 Implement IRQ wiring through GIC-400; support multiple concurrent DMA channels without lost wakeups.
- [ ] 3.8 Add a debug-build cache-maintenance tracker for `HpPort` buffers (records expected vs actual flush/invalidate calls) and a unit test that detects a missing `clean_for_device` call.
- [ ] 3.9 Add unit tests: MM2S 64 KiB transfer completes; S2MM 64 KiB transfer completes; cancelled future cleans up; two concurrent channels both receive completion.
- [ ] 3.10 Document port choice rules in `docs/axi-dma.md` (when to pick HPC vs HP vs ACP).

## 4. Phase 4 — arch/aarch64-zynqmp board crate

- [ ] 4.1 Create `arch/aarch64-zynqmp` crate skeleton (Cargo.toml, lib.rs, `#![no_std]`, edition 2021); add to workspace; verify `cargo build --target aarch64-unknown-none` succeeds.
- [ ] 4.2 Implement EL1 entry point matching ATF handoff (DTB pointer in `x0`); minimal early init (BSS clear, stack setup).
- [ ] 4.3 Implement Cadence UART driver (NOT PL011); wire as kernel console.
- [ ] 4.4 Implement GIC-400 driver: Distributor + CPU Interface init, SPI/PPI/SGI handling, priority masking, EOI signaling.
- [ ] 4.5 Implement ARMv8 generic timer driver: read frequency from `CNTFRQ_EL0`, support one-shot deadlines.
- [ ] 4.6 Define DDR memory map constants (PS DDR base + size, OCM range, ATF/PMU reserved regions); wire to kernel allocator init.
- [ ] 4.7 Confirm R5F cores remain in reset and Mali-400 register space is untouched (review + grep tests).
- [ ] 4.8 Add `just build-kernel-arm-zynqmp` recipe.
- [ ] 4.9 Add `just run-arm-zynqmp` recipe targeting QEMU `xlnx-zcu102` (or closest available ZynqMP machine model).
- [ ] 4.10 Boot test in QEMU: confirm boot banner appears on emulated UART0; confirm timer tick fires; confirm at least one GIC SPI is handled.

## 5. Phase 5 — BOOT.BIN packaging documentation

- [ ] 5.1 Document the FSBL+ATF chain in `docs/zynqmp-boot.md`: which AMD-supplied components are used, version pinning policy, where they come from.
- [ ] 5.2 Document the `bootgen` invocation that produces a working `BOOT.BIN` from FSBL + ATF + the SmallAIOS ELF.
- [ ] 5.3 Pin a specific Vitis version known to produce a working image; document soft-fail behavior on mismatch.
- [ ] 5.4 (Optional) Provide a `just package-bootbin` recipe that wraps `bootgen` if Vitis is available locally; otherwise prints actionable instructions.
- [ ] 5.5 Document what is NOT packaged here (verified-boot signing, partial reconfig, R5F payloads) and reference the future changes that will cover them.

## 6. Phase 6 — Documentation, CI, and verification

- [ ] 6.1 Write `docs/accelerator-hal.md`: trait surface, how to write a backend, fallback rules, ownership model.
- [ ] 6.2 Update `docs/architecture.md` to reflect the new `arch/aarch64-zynqmp` crate, the AXI/DMA framework, and the HAL layer in `onnx-rt`.
- [ ] 6.3 Update CI: add `arch-aarch64-zynqmp` build matrix entry; add advisory job for the QEMU stub end-to-end test.
- [ ] 6.4 Run full `just dsm-analyze` and verify no new layer violations or coupling clusters introduced.
- [ ] 6.5 Run `just clippy` (clean), `just fmt-check` (clean), `just test` (all passing); verify coverage ratchet not regressed.
- [ ] 6.6 Update `CLAUDE.md` "Workspace Architecture" and "Build Configuration" sections with the new crate and recipes.
- [ ] 6.7 `/opsx:verify` the change before marking it done.
- [ ] 6.8 Open follow-up change shells: `fpga-dpu-backend-v1` (DPU `.xmodel` runtime as a Backend), `fpga-custom-npu-v1` (HLS NPU informed by perf measurements), `fpga-manager-v1` (dynamic bitstream reconfig). Link from this change's PR description.
