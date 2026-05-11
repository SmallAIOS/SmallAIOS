# Design — deterministic-scheduling-v1

## Goal

A single opt-in mode (`SessionConfig::deterministic = true`; container env-var `SMALLAIOS_DETERMINISTIC=1`; kernel CLI `--deterministic`) that makes `(model, input)` → `output` a *function* in the mathematical sense, on a given hardware + driver combination, regardless of host scheduling, multi-stream timing, or wall-clock-derived state.

The deliverable is the contract, not the magic: every nondeterminism source is either removed, replaced with a deterministic counterpart, or explicitly documented as out-of-scope for v1.

Success = `just test-determinism` runs an inference twice and produces byte-identical outputs across at least four representative models (one CPU-only, one CUDA-CPU-hybrid, one CUDA-pure, one with explicit RNG ops like a small generative LLM head).

## Nondeterminism sources we are removing

| Source | Where it lives today | Deterministic-mode replacement |
|--------|----------------------|--------------------------------|
| Sibling-task dispatch order in `RunQueue` (timer-derived FIFO/LIFO tiebreaker on the per-core queue) | `kernel/src/sched/executor.rs` | Strict `(priority, task_id, spawn_seq)` ordering — no host-time input |
| Inference work-stealing across cores | `kernel/src/sched/executor.rs::steal_task` | Work-stealing disabled in deterministic mode; Inference queue pinned to its AMP-assigned core |
| Multi-stream CUDA execution order | `onnx-rt/src/cuda/streams.rs`, `onnx-rt/src/cuda/gpu_executor.rs`, configured via `SessionConfig::stream_config` | Force `StreamConfig::SingleStream` + op-boundary `cudaStreamSynchronize` |
| CUDA Graphs replay (capture order depends on initial issue order) | `onnx-rt/src/cuda/graph.rs`, `onnx-rt/src/cuda/graph_cache.rs` | Bypass graph capture entirely in deterministic mode (v1); honor capture in v2 once it is keyed on the deterministic sequence number |
| Per-op xorshift32 PRNG seeded from `seed` attribute but called by the operator without a session-level salt | `onnx-rt/src/ops/generative.rs` | Per-session `DeterministicRng` keyed on `(session_seed, op_index, draw_index)`; the operator-level seed becomes a hash input, not the whole seed |
| Wall-clock-derived sampling temperatures and Multinomial seed paths | `onnx-rt/src/ops/generative.rs` Multinomial / Bernoulli implementations | Read from `DeterministicRng`; "wall clock" in deterministic mode means the per-session monotonic counter, not host time |
| `enable_profiling` measurement feedback (measured op time → budget enforcement decision → potential abort) | `onnx-rt/src/profile.rs`, executor budget check | Decision left enabled (we still want the abort behavior), but the *decision branch* must be enforced after the op's outputs are computed, not before — so a Heisenbug from "this op was fast on this run" can't change subsequent op behavior. Already mostly true; we audit and lock in. |

## Alternatives considered

### Alternative A — Make the default mode deterministic, deprecate multi-stream

**Rejected.** Multi-stream delivers 1.3-1.5× on serving workloads and is the right default for the production container path (Jetson Orin AI server, GPU-accelerated inference at scale). Forcing every deployment to pay the determinism cost when most deployments don't need certification claims would be a major performance regression. The opt-in flag is the right user contract.

### Alternative B — Two scheduler binaries (build-time switch)

Build-time selection of `Default` vs `Deterministic` schedulers (analogous to `kernel` vs `container` Cargo features). **Rejected** because: (a) we want a single CI artifact that can verify both modes, (b) the runtime flip is cheap (one branch in `RunQueue::dequeue`, one branch in `ensure_stream_pool`), (c) DO-178C does *not* require build-time exclusion — it requires verifiable mode setting, which a runtime flag with a one-way init satisfies, and (d) supporting both modes from one binary keeps the existing test matrix from doubling.

### Alternative C — Per-task determinism (some tasks deterministic, some not)

**Rejected.** Determinism is a session-level property; mixing modes inside a session means the deterministic ops can be perturbed by the non-deterministic ones (cache contention, queue jostling). The DO-178C surface is "this inference session is deterministic" or "this inference session is not"; we cleanly model the same.

### Alternative D — DRBG instead of xorshift32 for the per-session RNG

**Deferred to a follow-up.** xorshift32 is the existing operator-level kernel and it has known cryptographic weaknesses, but cryptographic strength is not a determinism requirement — we just need a long-period reproducible stream. If a future change wants ChaCha20-DRBG (NIST SP 800-90A) for FIPS alignment, it can replace `XorShift32` under the existing `DeterministicRng` facade without breaking the determinism contract. The seam is intentional.

### Alternative E — Hardware-RNG fallback for non-RNG operators

Some hardware-accelerated kernels (e.g. cuRAND) bypass our PRNG entirely. **Rejected for v1** — we do not currently use cuRAND on the hybrid executor path (the int8 quantize/dequantize ops don't need RNG, and our `Multinomial`/`Bernoulli` implementations run on the CPU side of the hybrid split). When a future change moves these ops to GPU we will need a deterministic-mode branch that either disables the GPU implementation or routes through cuRAND-pseudo-host mode; tracked but not in scope here.

## Scheduler ordering — the deterministic tiebreaker

Current `RunQueue::dequeue` in `kernel/src/sched/executor.rs` returns the highest-priority task from the per-core queue. Within a priority class, the existing implementation uses queue insertion order (FIFO), which is deterministic *within* a single execution but depends on which core scheduled the task spawn first — a function of host timing on multi-core systems.

New deterministic ordering:

```text
sort_key(task) = (priority_class, task_id, spawn_sequence_number)
```

- `priority_class` — existing (`System < Ipc < Inference`).
- `task_id` — existing; assigned at task creation. Stable across runs because task creation order is itself determined by deterministic scheduling once at-rest.
- `spawn_sequence_number` — new; a per-scheduler atomic counter incremented at task spawn. Stable across runs because deterministic mode makes spawn order itself deterministic (chicken-and-egg resolved by initializing the counter at zero before the boot path starts).

The key insight: in default mode, two tasks of equal priority can run in either order depending on which core polled the queue first. In deterministic mode, the queue itself is sorted on `(priority, task_id, spawn_sequence_number)` so the ordering is a *function* of the queue contents, not of the polling pattern. Work-stealing is disabled in deterministic mode to maintain this invariant.

## CUDA — single stream + op boundary sync

`SessionConfig::deterministic = true` forces `StreamConfig::SingleStream` regardless of any explicit `stream_config` value. Explicitly: in `onnx-rt/src/session.rs::Session::ensure_stream_pool`, the existing dispatch on `stream_config` gains a wrapper that resolves `(deterministic, stream_config) -> effective_stream_config` as:

```text
(true, _)                  -> SingleStream  // determinism overrides
(false, x)                 -> x             // default plumbing unchanged
```

If a user sets both `deterministic = true` and `stream_config = Overlap { ... }`, the session emits a one-line warning to syslog at construction time and proceeds in single-stream mode. The warning is suppressible via a `SessionConfig::deterministic_silent: bool` if it becomes noisy in container deployments (probably not needed; we will see).

The op-boundary `cudaStreamSynchronize` is added to the hybrid executor's run loop (`onnx-rt/src/cuda/gpu_executor.rs`) as the last action of each operator's GPU-side work, before the executor returns control to the cooperative scheduler. Cost on the single-stream path is at most one CUDA driver round-trip per operator — typically sub-millisecond on Orin Ampere — and it serves as the lockstep voter's checkpoint.

CUDA Graphs (`onnx-rt/src/cuda/graph.rs` capture/replay path) is bypassed in deterministic mode. The reasoning: a captured graph's behavior is a function of the host-side issue order at capture time; replaying it later in deterministic mode would leak the capture-time nondeterminism into the deterministic run. Bypassing capture costs roughly 5-10% of throughput on the Jetson Orin path (per the `cuda-graphs-v1` archived measurements); this is documented as part of the deterministic-mode throughput cost.

## RNG — the `DeterministicRng` contract

A new type lives in `onnx-rt/src/determinism.rs` (or as a sub-module of `profile.rs`):

```text
struct DeterministicRng {
    session_seed: u64,         // From SessionConfig::deterministic_seed, default 0
    op_index: u32,             // Operator's position in the graph, assigned at load time
    draw_counter: AtomicU32,   // Bumped on each draw, reset per-session
}
```

Per draw:

```text
state = hash_combine(session_seed, op_index, draw_counter.fetch_add(1))
xorshift32 = XorShift32::new(state as u32)
return xorshift32.next()
```

Two draws on the same op produce different values; two runs of the same `(session_seed, model)` produce the same draws. `hash_combine` is a public deterministic mixing function (e.g. SplitMix or FxHash-derived) — pick the simplest one that produces low-bias 32-bit outputs.

The existing `XorShift32` in `onnx-rt/src/ops/generative.rs` stays as the kernel. Operators that currently call `XorShift32::new(op_attribute_seed)` change to call `DeterministicRng::draw(op_attribute_seed)` which folds the attribute seed in as an additional hash input.

For Multinomial and Bernoulli (the two ops that draw a *sample* per run, not just a fixed PRNG state), the wall-clock-derived sampling currently uses host time as a fallback when `seed = 0`. Deterministic mode rewrites the fallback to use `(session_seed, op_index, draw_counter)`, never host time.

## Failure modes and abort behavior

- **A new operator that requires nondeterminism is added** (e.g. a future op that reads `cuda_get_device_count` at runtime). Deterministic-mode session construction must reject the model with a typed error. Implementation: maintain a registry of "is this op deterministic-safe?" — defaults to `true`, ops that opt out (currently none, this is forward-compat) set `false`.
- **The user sets `deterministic_seed = 0` and runs twice expecting different outputs.** Document clearly: same seed → same output is the contract. If the user wants different outputs across runs in deterministic mode they must vary the seed themselves (e.g. via a session-level counter persisted to disk).
- **`cudaStreamSynchronize` fails.** Existing error path applies (fatal session error). Deterministic mode does not change error semantics.

## CI verification surface

- New `just test-determinism` recipe runs an inference twice in deterministic mode, diffs the outputs.
- New CI job `determinism-reproducibility` (advisory initially, gate after a release of stability) runs the same recipe in PR CI against a small set of fixture models stored under `tests/fixtures/determinism/`.
- Existing tests keep their existing assertions; deterministic mode is additive.
- Coverage gate (`cargo-llvm-cov --fail-under-lines 80`) still applies; new code paths must meet the existing 93% coverage ratchet.

## Throughput cost budget

We accept up to 35% throughput regression in deterministic mode vs. default mode on the hybrid-executor workload, measured on `bench/llm-inference-bench` on the Jetson Orin Industrial reference platform (cc 8.7, JetPack 6 base). Above 35%, the design is wrong and we should reopen the proposal. Below 35%, ship it.

Measurement methodology, baseline targets, and acceptance numbers are captured in `tasks.md` section 5 (Benchmarks).

## What this change explicitly does NOT do

- Does not change the default mode. Every existing deployment keeps the multi-stream / work-stealing behavior bit-for-bit.
- Does not change the `OperatorBudget` enforcement semantics. Budget checks still run; budget violations still abort.
- Does not change `onnx-rt`'s public API surface in a breaking way. `SessionConfig::deterministic` is a new field with a backward-compatible default.
- Does not add a new syscall. The kernel-mode flag is plumbed through existing init paths.
- Does not introduce SMP scheduling. AMP topology is preserved; deterministic mode only affects ordering *within* the existing AMP partitioning.
