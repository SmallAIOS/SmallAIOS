# Tasks — deterministic-scheduling-v1

## 0. Pre-flight — pin the determinism baseline

- [ ] 0.1 Document the current bit-level determinism status of each representative model: capture three runs of `bench/llm-inference-bench` and the four `tests/fixtures/` model paths and record where outputs diverge today (CPU vs GPU vs hybrid). The list is the design's "before" state; we ratchet against it.
- [ ] 0.2 Capture the baseline throughput of `bench/llm-inference-bench` in default mode on the Jetson Orin Industrial reference platform and on x86-64 CUDA. These numbers anchor the "max 35% regression" budget in deterministic mode.
- [ ] 0.3 Identify (search `onnx-rt/src/` for `Instant::now`, `clock_gettime`, `time(0)`, `cudaDeviceGetTimerValue`) every wall-clock call on the inference hot path. Each one must either: (a) be removed in deterministic mode, (b) be replaced with a `DeterministicRng` draw, or (c) be documented as deterministic-safe (e.g. profiling-only, no feedback into scheduling). Produce the list as `notes/0.3-wall-clock-call-sites.md`.

## 1. Scheduler — extend `kernel-core`

### 1a. `SchedulerMode` enum + init

- [ ] 1.1 Add `SchedulerMode` enum (`Default`, `Deterministic`) to `kernel/src/sched/executor.rs`. Default to `Default`. Setter is a one-shot init function callable only before the first task spawn; subsequent calls return an error.
- [ ] 1.2 Plumb the mode through `kernel::init`: read from boot args (`--deterministic` flag) or from a static set at compile time in container mode (driven by `SMALLAIOS_DETERMINISTIC` env var). Document the boot-arg parse in `docs/scheduling-model.md`.
- [ ] 1.3 Expose `Scheduler::mode()` getter so other subsystems (the CUDA executor in particular) can read it without taking a runtime branch on every call.

### 1b. Deterministic tiebreaker

- [ ] 1.4 Add `spawn_sequence_number: u64` to `Task` in `kernel/src/sched/task.rs`. Assigned atomically at task spawn from a per-scheduler `AtomicU64` counter.
- [ ] 1.5 Modify `RunQueue::dequeue` in `kernel/src/sched/executor.rs`: in `Deterministic` mode, sort the dequeue candidates on `(priority_class, task_id, spawn_sequence_number)` and return the smallest. In `Default` mode, current FIFO/LIFO behavior is preserved bit-for-bit (use a feature branch or runtime branch — measure cost; if branch cost is non-trivial, gate on a `cfg(feature = "deterministic")` or use a function pointer set at scheduler init).
- [ ] 1.6 Disable `RunQueue::steal_task` in `Deterministic` mode: return `None` immediately. Document in `docs/scheduling-model.md` that this collapses Inference-queue parallelism to one core per Inference task in deterministic mode.

### 1c. Tests + verification

- [ ] 1.7 Unit tests in `kernel/src/sched/executor.rs`: spawn N tasks with deliberately-shuffled `task_id`s, verify deterministic-mode dequeue order is sorted regardless of spawn-call ordering on the test thread.
- [ ] 1.8 Unit tests: verify `Default` mode dequeue behavior is bit-identical to the pre-change implementation (regression-pinning the current contract).
- [ ] 1.9 Update `docs/scheduling-model.md` with a new "Deterministic mode" section documenting the ordering rule, the work-stealing disable, and the cost.

## 2. CUDA — collapse multi-stream

### 2a. Stream-config override

- [ ] 2.1 Add `deterministic: bool` field to `SessionConfig` in `onnx-rt/src/session.rs`, default `false`. Add `deterministic_seed: u64`, default `0`. Document the field meaning in the struct doc-comment.
- [ ] 2.2 Modify `Session::ensure_stream_pool` to compute the *effective* stream config: `(deterministic, stream_config) -> effective_stream_config` where `deterministic = true` collapses any `Overlap { ... }` to `SingleStream`. Emit a one-line warning to syslog if the user passed both.
- [ ] 2.3 Add a runtime branch in `onnx-rt/src/cuda/gpu_executor.rs` that calls `cudaStreamSynchronize` at the end of each operator's GPU work when the session is in deterministic mode. Plumb the deterministic flag from `Session` into the executor.

### 2b. Bypass graph capture in deterministic mode

- [ ] 2.4 In `onnx-rt/src/cuda/graph.rs` / `onnx-rt/src/cuda/graph_cache.rs`, add a guard: if the session is deterministic, skip capture and skip replay — fall through to the uncaptured op-by-op path. Document the cost (5-10% throughput regression on the Orin path) in `docs/determinism.md`.
- [ ] 2.5 Add a unit test that runs the same model twice with `deterministic = true` and asserts identical output tensors. Use a small CUDA model from `tests/fixtures/`.

## 3. RNG — `DeterministicRng`

### 3a. The type

- [ ] 3.1 Create `onnx-rt/src/determinism.rs` (or sub-module of `profile.rs`) housing `DeterministicRng`. Public API: `new(session_seed, op_index)`, `draw_u32(&self) -> u32`, `draw_f32(&self) -> f32`, `draw_into(&self, buf: &mut [u32])`.
- [ ] 3.2 Internal: use `XorShift32` from `onnx-rt/src/ops/generative.rs` as the kernel; pre-mix the per-draw state via `hash_combine(session_seed, op_index, draw_counter)` so each draw produces an independent state. Match the existing operator-level seeding contract documented in lines 16-20 of `generative.rs`.

### 3b. Operator integration

- [ ] 3.3 Modify `RandomNormal`, `RandomNormalLike`, `RandomUniform`, `RandomUniformLike`, `Bernoulli`, `Dropout`, `Multinomial`, `EyeLike` in `onnx-rt/src/ops/generative.rs` to draw from `DeterministicRng` when the session is in deterministic mode. Keep the existing per-op `seed` attribute as one of the hash inputs.
- [ ] 3.4 Identify and remove any wall-clock fallback when `seed = 0` (currently used to derive a "random" seed). In deterministic mode, `seed = 0` is a valid seed (it folds with `session_seed` and `op_index` so the result is non-trivial).
- [ ] 3.5 Add a registry of "is this op deterministic-safe?" defaults — populate with all current ops marked safe. Reject session construction if a future op marks itself unsafe and `deterministic = true`.

### 3c. RNG tests

- [ ] 3.6 Unit tests: same `(session_seed, op_index, draw_count)` produces same output across two runs.
- [ ] 3.7 Unit tests: different `session_seed` produces different output (sanity check on the hash mixing).
- [ ] 3.8 Unit tests: `XorShift32` invariants preserved (no zero collapse, period ≥ 2^32 - 1).

## 4. Config + verification surfaces

### 4a. `just test-determinism` recipe

- [ ] 4.1 Add `just test-determinism [MODEL]` recipe that runs the named model twice in deterministic mode, byte-diffs the outputs, exits 0 on match. Default model is a small CPU-only fixture so the recipe runs without CUDA. Two failure exit codes: 30 = inference failed, 40 = outputs diverged (with hex dump of the first divergent bytes).
- [ ] 4.2 Add `scripts/test-determinism.sh` as the underlying wrapper (so CI can use the same code path).

### 4b. CI advisory job

- [ ] 4.3 Add a `determinism-reproducibility` advisory CI job to `.github/workflows/ci.yml` that runs `just test-determinism` for each of three fixture models: one CPU-only, one with explicit RNG ops, one with CUDA-CPU hybrid (kept advisory until we have a self-hosted runner for the CUDA case).
- [ ] 4.4 Document the promote-to-gate criterion in the workflow YAML.

### 4c. Docs

- [ ] 4.5 Create `docs/determinism.md` covering: the contract (same `(model, input, seed, hardware, driver)` → same output), the throughput cost, the certification claims (DO-178C input-output traceability, ISO 26262 Diagnostic Coverage as the channel for lockstep voting), the deterministic-mode CLI/env-var flags, the user-side responsibility (don't reuse seed if you want different draws), and the operator-level deterministic-safe registry.
- [ ] 4.6 Update `docs/scheduling-model.md` "POSIX Scheduling Alignment" section with a deterministic-mode subsection (linking to `docs/determinism.md`).
- [ ] 4.7 Update `CLAUDE.md` "Crate Feature Flags" section to list the new `SessionConfig::deterministic` field (no new feature flag — runtime config).
- [ ] 4.8 Update `README.md` deployment matrix or feature table to surface that deterministic mode is available; one line + link to `docs/determinism.md`.

## 5. Benchmarks

- [ ] 5.1 Extend `bench/llm-inference-bench` with a deterministic-mode comparison. Print throughput (default mode) and throughput (deterministic mode) side-by-side. Assert deterministic-mode throughput ≥ 65% of default-mode (the 35% regression budget).
- [ ] 5.2 Capture benchmark numbers in `notes/5.1-deterministic-throughput-baseline.md` for the next reviewer.
- [ ] 5.3 If deterministic-mode throughput falls below 65%, escalate before merging — the design's cost model is wrong and the proposal needs revision.

## 6. Reproducibility golden tests

- [ ] 6.1 Add a `tests/determinism/` directory with fixture inputs and expected outputs (byte-identical) for at least four models: one CPU-only small classifier, one CUDA-pure ResNet-ish path, one CUDA-CPU hybrid (e.g. ONNX with int8 quantize on CPU + GEMM on GPU), one with explicit RNG-op coverage (Dropout + Multinomial).
- [ ] 6.2 Wire `tests/determinism/` into `cargo test` so it runs in PR CI. Skip the CUDA fixtures when CUDA is not available (existing `#[cfg(feature = "cuda")]` pattern).
- [ ] 6.3 Document in `docs/determinism.md` how to regenerate the golden outputs when the underlying model intentionally changes.

## 7. Verify + close-out

- [ ] 7.1 Run `openspec validate deterministic-scheduling-v1 --strict`.
- [ ] 7.2 PR title: `feat(kernel,onnx-rt): deterministic-scheduling-v1 — opt-in reproducibility mode`. Target `develop`.
- [ ] 7.3 PR description includes: (a) before/after benchmarks from 5.2, (b) reproducibility evidence (output of `just test-determinism` against each fixture), (c) the certification claim text (deterministic-mode unlocks DO-178C input-output traceability and ISO 26262 Diagnostic Coverage on the lockstep voter).
- [ ] 7.4 Reviewer sign-off + squash-merge.

## 8. Archive (after `watchdog-lockstep-v1` lands)

- [ ] 8.1 Run `openspec validate deterministic-scheduling-v1 --strict` post-merge.
- [ ] 8.2 Move to `openspec/changes/archive/YYYY-MM-DD-deterministic-scheduling-v1` and sync the spec deltas to `openspec/specs/kernel-core/` and `openspec/specs/onnx-rt-determinism/`.
