# deterministic-scheduling-v1

## Summary

SmallAIOS already ships a cooperative scheduler that yields at ONNX operator boundaries (`kernel/src/sched/executor.rs`, design captured in `docs/scheduling-model.md`) and a hybrid CUDA executor that overlaps host/device transfers across multiple streams (`onnx-rt/src/cuda/streams.rs`, archived as `2026-05-03-async-multistream-v1`). The cooperative model gives us *bounded* execution windows — every operator runs to completion before the next preempt opportunity — but the actual *order* in which sibling tasks run, the operator-to-stream binding chosen by the hybrid executor, and the values produced by RNG-driven operators (`Multinomial`, `Bernoulli`, `Dropout`, `RandomUniform`, `RandomNormal` — see `onnx-rt/src/ops/generative.rs`) all depend on host wall-clock timing today. That means **the same `(model, input)` pair can produce different outputs on the same hardware on different runs.** For sampling-style generative inference this is sometimes intentional; for DO-178C DAL A certification it is fatal — input-output traceability is a hard objective and cannot be claimed without bit-identical reproducibility, and automotive Diagnostic Coverage credit under ISO 26262 likewise requires a deterministic baseline against which redundant-channel votes can be compared.

This change adds a single new runtime mode — `--deterministic` (kernel) and `SessionConfig::deterministic = true` (host/container) — that forces the cooperative scheduler to dispatch sibling tasks in a stable, hash-defined order, collapses the multi-stream CUDA executor back onto a single synchronous stream at op boundaries, and replaces every wall-clock-dependent RNG source with a seeded counter-based DRBG that is driven by a deterministic per-session sequence number rather than host time. Same input → same output, on the same hardware, regardless of how many other processes the host is running or how many cores are warm. The default mode (multi-stream, free-running ordering, current behavior) is preserved bit-for-bit; deterministic mode is opt-in and pays a measurable throughput cost (we estimate 20-35% on hybrid-executor workloads where multi-stream overlap is the dominant speedup) in exchange for the certification claim.

Three deliveries: (1) extend the existing `kernel-core` cooperative-scheduler spec with a `Deterministic` ordering mode; (2) add a new `onnx-rt-determinism` capability spec covering the CUDA-stream collapse and the seeded-DRBG contract; (3) wire enough config plumbing through `SessionConfig`, the `Justfile`, and the container env-var layer that a developer can flip the mode with a single flag and verify reproducibility via a new `just test-determinism` recipe that runs an inference twice and diffs the outputs byte-for-byte.

## Why

- **DO-178C DAL A input-output traceability is unclaimable without deterministic mode.** Table A-3 objective 1 ("software requirements comply with system requirements") and Table A-7 objective 8 ("verification of verification activities") both require that every verification run against a given input produce the same observable output as the version that was reviewed and certified. Today's multi-stream executor breaks this: stream scheduling order is decided by the CUDA driver, which is not part of our certifiable boundary. The `formal-proving-and-redteam-v1` change (also in flight) needs this same property to make claims about formal proofs covering executable behavior — a formal proof over a non-deterministic implementation only covers the relation, not the function. Deterministic mode is the precondition for both.
- **Automotive Diagnostic Coverage credit needs a reproducible reference channel.** When `watchdog-lockstep-v1` (the sibling change in this batch) wires up dual-core lockstep on Cortex-A78AE, the lockstep voter compares two replicas of the same inference. If the replicas are running in non-deterministic mode they can legitimately produce different outputs — same model, same input, same hardware, but different timer-derived RNG draws and different multi-stream scheduling. The voter has no signal. Deterministic mode is what makes lockstep voting meaningful: a divergence between replicas in deterministic mode is *always* a soft fault (the whole point of the redundancy), never a benign timing artifact.
- **The RNG path already wants to be deterministic — the design just isn't enforced.** `onnx-rt/src/ops/generative.rs` lines 16-20 already document: *"Random operators use a deterministic xorshift32 PRNG seeded from the `seed` attribute (float) and `shape` attribute (ints) so repeated calls with the same seed produce reproducible results. This is critical for DO-178C DAL A certification — 'random' is a misnomer; every draw must be deterministically replayable from (seed, call-site)."* The `XorShift32` type in that file is wired correctly per-operator, but the *whole-session* state (which operator runs first when two are eligible, when the multi-stream executor decides to flush, how token-sampling temperature interacts with the per-op seed) is not yet under that same discipline. This change finishes the work the operator path started.
- **Cost is bounded and measurable.** Multi-stream gives ~1.3-1.5× on serving workloads (per the `2026-05-03-async-multistream-v1` archive). Deterministic mode forfeits that overlap by definition — but we already documented and shipped the single-stream path as the default, so the deterministic mode is implementable by *reusing the existing `StreamConfig::SingleStream` plumbing*, not by writing new code. The op-boundary `cudaStreamSynchronize` is a one-line addition.

## What changes

### Kernel scheduling — extend `kernel-core`

- Add a `SchedulerMode` enum to `kernel/src/sched/executor.rs`: `Default` (current behavior — multi-stream-friendly, work-stealing on the Inference queue) and `Deterministic` (stable dispatch order, work-stealing disabled, all multi-core work serialized through a deterministic round-robin over the configured `CpuAffinity` set). The mode is set once at scheduler init and is immutable for the lifetime of the boot.
- Replace the LIFO-on-tie ordering currently used in `RunQueue::dequeue` with a deterministic tiebreaker keyed on `(priority, task_id, deterministic_sequence_number)`. The sequence number is a per-scheduler monotonic counter incremented at task spawn — it produces a stable order across runs without requiring host time.
- Disable Inference-queue work-stealing (`steal_task()` in `kernel/src/sched/executor.rs`) when in `Deterministic` mode. Work-stealing makes per-core run order depend on which core happened to be idle first, which is the definition of nondeterminism we are trying to remove. Single-core inference is the cost; for AMP topologies the Inference queue collapses to its assigned core only.

### CUDA execution — collapse multi-stream overlap

- Honor `SessionConfig::deterministic = true` by forcing `StreamConfig::SingleStream` regardless of any explicit `Overlap { transfer_streams }` setting (with a runtime warning if the user passed both). The plumbing already exists; this change is one branch in `onnx-rt/src/session.rs::Session::ensure_stream_pool`.
- Add an op-boundary `cudaStreamSynchronize` to the hybrid executor's run loop (`onnx-rt/src/cuda/gpu_executor.rs`) when `deterministic = true`. Under `SingleStream` mode the synchronize is technically redundant for correctness (everything is already on one stream), but it converts the implicit per-stream FIFO ordering into a checkpoint that the lockstep voter (and the formal-verification surface) can hang assertions on. Cheap insurance.
- Bypass the CUDA Graphs capture path entirely in deterministic mode for v1. Graph capture amortizes launch overhead by recording a stream's sequence of work, but the capture itself is a function of *when* operators were issued, so reusing a captured graph across runs that were captured in non-deterministic mode would leak nondeterminism into the deterministic run. A future revision can teach graph capture to also key on the deterministic sequence number; out of scope for v1.

### RNG — single per-session deterministic DRBG

- Add a `DeterministicRng` type to `onnx-rt/src/profile.rs` (or a new `onnx-rt/src/determinism.rs`) keyed on `(session_seed: u64, op_index: u32, draw_index: u32)`. The existing `XorShift32` in `onnx-rt/src/ops/generative.rs` is the right starting kernel; the new type wraps it so that every RNG-consuming operator threads the same `(session_seed, op_index)` pair and gets a fresh `draw_index` per call. Same operator, same position in the graph, same model, same input → same draws.
- Replace any call to `cudaDeviceGetTimerValue` / wall-clock-derived sampling temperatures (currently used in `multinomial` and `Bernoulli` paths) with reads from the per-session `DeterministicRng`. Document the contract: in deterministic mode, "wall clock" is the per-session counter, not host time.
- Reject `SessionConfig::deterministic = true` if the loaded model contains an operator that demonstrably cannot be made deterministic (none today; this is a forward-compatibility check). Return a typed error at session creation, not at first inference.

### Config + verification surfaces

- New `SessionConfig::deterministic: bool` (default `false`). Feeds the four behaviors above.
- New container env-var `SMALLAIOS_DETERMINISTIC=1` that sets the same flag on the container path.
- New `just test-determinism` recipe that builds, runs an inference twice with `deterministic=true`, byte-diffs the outputs, fails on diff. Wired into CI as an advisory check initially, gate after one release of stability.
- New `docs/determinism.md` documenting the deterministic-mode guarantees, the throughput cost, and the certification claims it unlocks.

## Relation to prior work

This change **extends** the cooperative scheduling design captured in `docs/scheduling-model.md` and the operator-level RNG contract documented in `onnx-rt/src/ops/generative.rs` (which itself was reviewed as part of `2026-04-21-generative-llm-v1`, archived). It does not contradict any archived design. It **conflicts with** `2026-05-03-async-multistream-v1` only in the narrow sense that it disables that path's overlap mode under one specific flag — the default mode is preserved bit-for-bit, so no existing deployment regresses.

The hard precondition is the `timer-hal-wcet-v1` work (archived `2026-04-10-timer-hal-wcet-v1`) which already replaced wall-clock dependency in `sys_time()` with the architecture timer; deterministic mode builds on that by also gating the per-operator measurement away from any flow that influences scheduling decisions. (Measurement still happens — it just doesn't feed back into ordering.)

## Out of scope

- **GPU kernel determinism within a single op.** cuDNN's GEMM and convolution kernels are not bit-deterministic across CUDA driver versions (the reduction-tree shape depends on tile counts and warp-scheduling). v1 of deterministic mode is "deterministic across runs *on the same driver + hardware*". A future change can layer cuBLAS-deterministic-mode (`CUBLAS_PEDANTIC_MATH`) and cuDNN-deterministic-algorithm-selection to extend the guarantee across driver versions, with a measurable additional throughput cost.
- **Multi-process determinism.** This change covers a single SmallAIOS instance. If two instances run inference on shared hardware, the host kernel's GPU scheduling between them is not under our control.
- **Distributed determinism (across nodes).** Out of scope by deliberate choice — would require a deterministic network transport, which is a separate (very large) change.
- **Bit-identical floating-point across architectures.** AArch64 and x86-64 NEON/AVX paths in some CPU operators use slightly different reduction orders. Determinism is defined per-architecture in v1.

## Sequencing

This change should land **before** `watchdog-lockstep-v1` so the lockstep voter has a deterministic reference channel to compare against. It is **independent** of the Tegra234 BSP from `unikernel-orin-bringup-v1` — deterministic mode is a host-architecture-agnostic property of the kernel + ONNX runtime, and can be developed and tested entirely on x86-64 hardware. The Orin path picks it up "for free" once the BSP merges.

## Effort estimate

| Sub-area | Scope | Estimate |
|----------|-------|----------|
| Scheduler mode + deterministic tiebreaker | `kernel/src/sched/executor.rs` + `RunQueue` changes | ~1 week |
| CUDA stream collapse + op-boundary sync | `onnx-rt/src/cuda/{gpu_executor,streams}.rs` + `SessionConfig` plumb | ~1 week |
| `DeterministicRng` + operator wiring | `onnx-rt/src/ops/generative.rs` + new `determinism.rs` | ~1 week |
| `just test-determinism` recipe + CI advisory job + docs | `Justfile`, `.github/workflows/ci.yml`, `docs/determinism.md` | ~1 week |
| Reproducibility golden tests across 3-4 representative models | New `tests/` modules | ~1 week |
| **Total** | | **~4-5 weeks** |
