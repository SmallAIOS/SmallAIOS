# SmallAIOS Architecture

SmallAIOS is a 20-crate `#![no_std]` Rust workspace organized into a strict 4-layer dependency model. Higher layers may depend on same-layer or lower-layer crates only. This document covers the layer model, dependency structure, design rationale, and acyclicity guarantees.

## 4-Layer Model

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3 — Integration                                              │
│  ┌──────────────────┐  ┌──────────────────┐                         │
│  │  container        │  │  bench (dev-only) │                        │
│  │  Entry point,     │  │  Benchmarks       │                        │
│  │  config, health,  │  │                   │                        │
│  │  metrics          │  │                   │                        │
│  └──────────────────┘  └──────────────────┘                         │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 2 — HAL / Drivers                                            │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                       │
│  │ arch/x86_64│ │arch/aarch64│ │arch/riscv64│  CPU HALs             │
│  └────────────┘ └────────────┘ └────────────┘                       │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                       │
│  │ arch/nvidia│ │  arch/amd  │ │arch/intel  │  GPU HALs (stubs)     │
│  └────────────┘ └────────────┘ └────────────┘                       │
│  ┌────────────┐                                                     │
│  │ arch/apple │  Apple Metal HAL (macOS)                             │
│  └────────────┘                                                     │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                       │
│  │ peripheral │ │    bus     │ │    sdr     │  Device drivers        │
│  └────────────┘ └────────────┘ └────────────┘                       │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 1 — Core Services                                            │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐            │
│  │  net   │ │  ipc   │ │ posix  │ │onnx-rt │ │  usb   │            │
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘            │
├─────────────────────────────────────────────────────────────────────┤
│  LAYER 0 — Foundation                                               │
│  ┌───────────────────────────┐  ┌───────────────────────────────┐   │
│  │  kernel                   │  │  security                     │   │
│  │  Memory (buddy allocator, │  │  Capability-based access,     │   │
│  │  tensor pool), cooperative│  │  PQC crypto (SHA-3, AES-GCM,  │   │
│  │  scheduler, ~46 syscalls  │  │  ML-KEM, ML-DSA, Ed25519,     │   │
│  │                           │  │  X25519), formal gate          │   │
│  └───────────────────────────┘  └───────────────────────────────┘   │
│  ┌───────────────────────────┐  ┌───────────────────────────────┐   │
│  │  compute                  │  │  sched-types                  │   │
│  │  CPU/GPU/NPU backends,    │  │  Shared scheduler primitives  │   │
│  │  tensor buffer management │  │  (OperatorClass, Budget)      │   │
│  └───────────────────────────┘  └───────────────────────────────┘   │
│           kernel ──depends-on──▶ security                           │
│           kernel ──depends-on──▶ sched-types                        │
└─────────────────────────────────────────────────────────────────────┘

  Dependency direction: Layer 3 → Layer 2 → Layer 1 → Layer 0
  (higher layers depend on lower layers only)
```

### Crate-to-Layer Assignment

| Layer | Crate | Role |
|-------|-------|------|
| 0 | `smallaios-kernel` | Memory management, cooperative scheduler, syscall interface |
| 0 | `smallaios-security` | Capability system, PQC crypto stack, formal verification gate |
| 0 | `smallaios-compute` | Unified compute abstraction: device registry, kernel dispatch, tensor buffers |
| 0 | `smallaios-sched-types` | Shared scheduler primitive types (`OperatorClass`, `OperatorBudget`, `BudgetResult`) |
| 1 | `smallaios-net` | IPv4/IPv6, TCP/UDP, ARP/NDP, QUIC/HTTP3, TLS 1.3 |
| 1 | `smallaios-ipc` | Zenoh-inspired pub/sub messaging |
| 1 | `smallaios-posix` | Minimal POSIX compatibility layer |
| 1 | `smallaios-onnx-rt` | Clean-room ONNX runtime (protobuf parser, optimizer, 6 operators) |
| 1 | `smallaios-usb` | USB core stack, xHCI host controller |
| 2 | `smallaios-arch-x86_64` | x86-64 HAL: boot, GDT, IDT, APIC, paging, syscall |
| 2 | `smallaios-arch-aarch64` | ARM64 HAL: boot, GICv3, paging, SVE, PSCI |
| 2 | `smallaios-arch-riscv64` | RISC-V HAL: boot, SBI, trap handling, paging |
| 2 | `smallaios-arch-nvidia` | NVIDIA GPU HAL stub: PCIe, GPU init, compute, DMA |
| 2 | `smallaios-arch-amd` | AMD RDNA/CDNA GPU HAL stub |
| 2 | `smallaios-arch-intel-gpu` | Intel Xe GPU HAL stub |
| 2 | `smallaios-arch-apple` | Apple Metal GPU HAL (macOS only) |
| 2 | `smallaios-peripheral` | I2C, SPI, GPIO, UART, CSI camera, I2S audio |
| 2 | `smallaios-bus` | CAN, ARINC 429/664, MIL-STD-1553, SpaceWire |
| 2 | `smallaios-sdr` | Software-defined radio: HackRF One, ADALM-Pluto |
| 3 | `smallaios-container` | Entry point, config, health checks, metrics |
| 3 | `smallaios-bench` | Benchmarks (dev-dependency only) |

## DSM Evidence

Design Structure Matrix analysis of the workspace dependency graph. Generated by `tools/dsm/` from `build/analysis/dsm-matrix.json`. Run `just dsm-analyze` to regenerate.

### Propagation Cost

Propagation cost measures the percentage of crates affected when a crate changes (transitively).

| Crate | Propagation Cost | Notes |
|-------|-----------------|-------|
| `security` | 100% | Foundation — all crates transitively affected |
| `kernel` | 94% | 18 of 21 crates depend on it directly |
| `arch/nvidia` | 22% | Used by onnx-rt and aarch64 |
| `net` | 17% | Used by container and posix |
| `onnx-rt` | 17% | Used by container and aarch64 |
| `ipc` | 11% | Used by container |
| `posix` | 11% | Used by container |
| `usb` | 11% | Used by sdr |
| All other Layer 2 | 6% | Leaf or near-leaf crates |
| `container`, `bench` | 6% | Top-level consumers |

### Fan-In / Fan-Out Summary

| Crate | Fan-In | Fan-Out | Layer |
|-------|--------|---------|-------|
| `kernel` | 16 | 1 | 0 |
| `security` | 5 | 0 | 0 |
| `net` | 2 | 1 | 1 |
| `ipc` | 1 | 2 | 1 |
| `posix` | 1 | 2 | 1 |
| `onnx-rt` | 2 | 3 | 1 |
| `usb` | 1 | 1 | 1 |
| `arch/x86_64` | 0 | 1 | 2 |
| `arch/aarch64` | 0 | 3 | 2 |
| `arch/riscv64` | 0 | 1 | 2 |
| `arch/nvidia` | 3 | 1 | 2 |
| `arch/amd` | 0 | 1 | 2 |
| `arch/intel-gpu` | 0 | 1 | 2 |
| `peripheral` | 0 | 2 | 2 |
| `bus` | 0 | 1 | 2 |
| `sdr` | 0 | 2 | 2 |
| `container` | 0 | 7 | 3 |
| `bench` | 0 | 1 | 3 |

Fan-in = number of crates that depend on this crate (production deps only).
Fan-out = number of crates this crate depends on (production deps only).

### Key DSM Observations

- **Hub crate:** `kernel` is the central hub with fan-in=16. This is intentional — it provides the foundation (memory, scheduling, syscalls) that all other crates need.
- **High propagation risk:** `security` has the highest propagation cost (~94%) because `kernel` depends on it and nearly everything depends on `kernel`. Changes to crypto primitives or the capability system ripple through the entire workspace.
- **Clean leaf layer:** All Layer 2 (HAL/driver) crates have fan-in of 0-2, keeping hardware-specific changes isolated.
- **Narrow integration point:** `container` is the sole top-level integrator with fan-out=7, providing a single composition root.

## Design Rationale

### Unikernel Architecture

SmallAIOS runs in a single address space with no kernel/user mode split. This eliminates context switch overhead and IPC marshalling costs that would add latency to inference hot paths. The tradeoff — no process isolation — is acceptable because the system runs a single workload (ONNX inference) and uses capability-based security instead of address space isolation.

### Cooperative Scheduling

The scheduler is cooperative, not preemptive. Tasks yield at ONNX operator boundaries (after each Conv, MatMul, Relu, etc.). This avoids the overhead and complexity of preemption (saving/restoring SIMD/GPU state mid-operator) while providing natural scheduling points that align with the inference workload.

### `#![no_std]` Throughout

All 21 crates are `#![no_std]`. This enables bare-metal deployment on x86-64, ARM64, and RISC-V without a host OS. The same crates also compile for musl targets for container deployment, giving a single codebase for both deployment modes.

### Size Goals

- **<8 MB base kernel:** Fits in embedded flash/SRAM. Achieved via `opt-level = "z"`, LTO, and single codegen unit.
- **<15 MB container image:** Built `FROM scratch` with static musl binary. Current size: ~594 KB.
- **<50 ms container boot:** No init system, no dynamic linking, no filesystem setup. Boot straight to inference.

### Post-Quantum Cryptography by Default

The `pqc-hybrid` feature is on by default, providing ML-KEM-768 (key encapsulation) + ML-DSA-65 (signatures) alongside classical X25519/Ed25519. This future-proofs deployed systems against quantum attacks without waiting for a migration event. The `classical-only` and `pqc-only` feature flags allow operators to choose.

### DO-178C DAL A Compliance Target

Safety-critical aviation certification requires MC/DC 100% coverage on critical paths, formal verification (19 TLA+ models, 6 SPIN/Promela models), and traceability from requirements to tests. The `formal-gate` feature flag enables compile-time formal verification checks. This is a design target, not yet achieved.

## Dependency Rules

### Layer Rules

1. **Downward only:** A crate at Layer N may depend on crates at Layer N or below.
2. **No upward dependencies:** Layer 0 crates never depend on Layer 1+. Layer 1 crates never depend on Layer 2+.
3. **Same-layer allowed:** Crates within the same layer may depend on each other (e.g., `kernel` → `security` within Layer 0).
4. **Dev-dependencies exempt:** Test-only dependencies may cross layers in any direction. These are not compiled into production builds.

### Specific Dependency Edges

```
Layer 0 (internal):
  kernel → security

Layer 1 → Layer 0:
  net → kernel
  ipc → kernel, security
  posix → kernel
  onnx-rt → kernel, security
  usb → kernel

Layer 2 → Layer 0:
  All arch/* crates → kernel
  nvidia → kernel, security
  peripheral → kernel
  bus → kernel
  sdr → kernel

Accepted cross-layer exceptions:
  aarch64 (L2) → nvidia (L2), onnx-rt (L1) — GPU dispatch on ARM64
  onnx-rt (L1) → nvidia (L2) — CUDA execution provider (upward, accepted)
  posix (L1) → net (L1) — POSIX socket layer needs network stack

Layer 3 → Layer 0/1:
  container → kernel, security, net, ipc, posix, onnx-rt, nvidia
```

### Enforcement

- **Cargo workspace:** The `Cargo.toml` dependency graph is the source of truth. Any new dependency edge is visible in PR diffs.
- **`cargo-modules --acyclic`:** Run in CI to verify no production-dependency cycles exist.
- **Code review:** Layer violations are flagged during review. A crate's layer assignment is documented in this file and in its own `Cargo.toml` metadata.

## Acyclicity Guarantee

### Zero Production Cycles

The production dependency graph (normal + build dependencies) contains **zero cycles**. This is verified by `cargo-modules` and is structurally enforced by the layered architecture: since dependencies only flow downward (Layer 3 → 2 → 1 → 0), cycles cannot form across layers. Within Layer 0, the single edge `kernel → security` is unidirectional.

### Benign Dev-Dependency Cycle

One dev-dependency cycle exists and is intentional:

```
security ──[dev-dep]──▶ net ──[normal]──▶ kernel ──[normal]──▶ security
```

`security` has a dev-dependency on `net` for integration testing (verifying that TLS handshakes work with the real network stack). This creates a cycle in the full dependency graph but **not** in the production graph. Cargo handles this correctly — dev-dependencies are only compiled for `cargo test` of that specific crate and are never included in release builds.

### Enforcement Mechanism

- **CI check:** `cargo-modules` with the `--acyclic` flag runs in CI and will fail the build if any production cycle is introduced.
- **Cargo itself:** Cargo forbids cycles in normal dependencies at the workspace level. A PR that introduces a production cycle will fail `cargo check`.
- **Structural guarantee:** The 4-layer model makes cycles structurally unlikely. A cycle would require a lower-layer crate to depend on a higher-layer crate, which violates the documented dependency rules and would be caught in review.

## GPU Residency: Hybrid Inference Path

The CUDA execution provider supports two operator-routing modes,
selectable per-session via `SessionConfig::gpu_residency`:

- `GpuResidency::OpByOp` (default) — each GPU-eligible op copies its
  inputs to device, runs, and copies the output back to host before
  the next op starts.
- `GpuResidency::Hybrid` — per-tensor location tracking keeps
  intermediate activations device-resident across consecutive
  GPU-supported ops, eliminating the round-trip.

### Value-location tracking

Inside the hybrid executor (`onnx-rt/src/executor_hybrid.rs`), the
tensor value map binds each named graph value to one of two states:

```rust
enum ValueLocation {
    Host(Tensor),
    Device(Arc<DeviceTensor>),
}
```

A name lives on exactly one side at any moment, never both.
`Arc<DeviceTensor>` lets branching graphs (one tensor feeding
multiple consumers) share a device buffer without device→device
memcpy.

### Dispatch decision

For each node:

1. Examine the first input's residency to determine its current dtype.
2. Consult `gpu_op_supported(op_type, dtype)` — a table covering
   `Conv`, `Gemm`, `MatMul`, `BatchNormalization`, `Relu`, `Clip`,
   `MaxPool`, `AveragePool`, `GlobalAveragePool`, and `Add` for
   `Float` / `BFloat16` inputs.
3. If supported, ensure all inputs are on device (uploading any
   host-resident input via `cudaMemcpyHostToDevice`), call the
   device-side kernel via `try_gpu_dispatch`, and store the output
   as `ValueLocation::Device`.
4. Otherwise, ensure all inputs are on host (copying device-resident
   ones back via `cudaMemcpyDeviceToHost`), run the existing CPU
   dispatcher, and store the output as `ValueLocation::Host`.
5. The next op picks up from whichever residency its inputs hold.
   The hybrid path never collapses to all-CPU after the first CPU op
   — subsequent GPU-eligible ops still dispatch to GPU.

### Boundary memcpys

Memcpys happen at exactly four moments:

- **Graph input → first GPU op.** A user-provided host tensor is
  uploaded the first time a GPU op consumes it.
- **CPU op → next GPU op.** A CPU op's host output is uploaded when
  the next GPU op needs it.
- **GPU op → next CPU op.** A device-resident value is downloaded to
  host before the CPU op runs.
- **Graph output.** Any device-resident graph output is downloaded to
  host before being returned to the caller.

Steady-state vision-graph inference (e.g. ResNet-50) does **zero**
intermediate device→host copies in hybrid mode — only the input and
the final classifier output cross the boundary.

### Initializer caching

Initializers (model weights, BN parameters, etc.) are uploaded to
device exactly once per session via
`Session::device_initializer_cache`, lazy-populated on the first
hybrid-mode `run()` call. The cache stores
`Arc<DeviceTensor>` keyed by tensor name; the executor's value map
populates initializer entries by `Arc::clone`-ing from the cache,
which is a refcount bump rather than a buffer copy. This avoids
re-decoding `TensorProto` bytes and re-uploading all weights on every
inference.

For ResNet-50 v2 (~25 M parameters), this saves ~100 MB of
host→device transfers per inference and pushed the measured
end-to-end speedup from ~111× to ~419× (see
`docs/benchmarks/arm64-gpu-cpu-vs-gpu.md`).

### Opt-in rollout

`GpuResidency::OpByOp` remains the default for backward compatibility.
Users opt into the hybrid path by setting
`SessionConfig::gpu_residency = GpuResidency::Hybrid` at session
creation time. The runtime falls back to `OpByOp` cleanly if no CUDA
runtime is attached.

### Limitations

- BFloat16 path is wired but less tested than Float32; the bench
  numbers in the cited doc are FP32/TF32.
- Operators not in the `gpu_op_supported` table (Reshape, Concat,
  Gather, Cast, Shape, Unsqueeze, Dropout, etc.) always run on CPU.
  Their outputs are typically tiny shape-path tensors and the boundary
  memcpy is cheap.
- A device-side bias-add kernel is implemented for `Conv`; other ops
  do not support an in-graph bias (model authors typically lift bias
  into a separate `Add` node, which itself dispatches to GPU).
- Async DMA / multi-stream overlap is not yet wired — every cuDNN
  call is fully synchronous.

### CUDA Graph Capture (cuda-graphs-v1)

Layered on top of the hybrid path, `SessionConfig::cuda_graph =
CudaGraphMode::Capture` enables CUDA Graph capture / replay. The
hybrid executor's per-op dispatch loop pays ~10–50 µs of host-side
launch overhead per cuDNN / cuBLAS call. With ~170 ops per ResNet-50
inference that's ~5 ms of pure overhead per run. Graph capture
collapses N kernel launches into a single `cudaGraphLaunch` call.

**How it works:**

1. **First inference (cache miss).** Runs the existing per-op path
   to produce a correct output, then attempts to capture the same op
   sequence on a dedicated CUDA stream. The capture path:
   - allocates persistent input `DeviceTensor`s and pre-loads the
     user inputs (so the first replay reproduces the per-op output)
   - binds cuDNN + cuBLAS handles to the capture stream via
     `cudnnSetStream` / `cublasSetStream_v2`
   - wraps the dispatch loop in `cudaStreamBeginCapture` /
     `cudaStreamEndCapture`
   - calls `cudaGraphInstantiate` to build the executable graph
   - stores the graph + persistent buffers + every intermediate
     `DeviceTensor` (so the captured pointers stay live for replay)
     in a per-Session `CudaGraphCache` keyed by input
     `(shape, dtype)` tuple.
2. **Subsequent inferences (cache hit).** `try_replay` does
   `cudaMemcpyAsync` of the new user input into the persistent input
   buffer, calls `cudaGraphLaunch`, syncs, and `cudaMemcpyAsync`s
   outputs back to host. One host call replaces ~170.
3. **Cache invalidation.** A new input shape produces a new
   `GraphKey`, missing the cache and triggering a re-capture for
   that shape. Multiple shapes can coexist in the cache (e.g.
   batch 1, 4, 16, 64 each get their own captured graph).
4. **Disable on thrash.** After 32 inferences, if rebuilds exceed
   1% of inferences, the cache disables itself for the remaining
   Session lifetime — capturing and discarding is slower than just
   running per-op.
5. **Graceful fallback.** Capture / instantiation / replay failures
   never propagate. They log a single warning, evict the bad entry
   (replay) or mark the cache disabled (capture), and fall through
   to per-op execution. Only actual op failures (e.g. cuDNN
   `BAD_PARAM`) surface as `SessionError`.

The cache lives in `Session::cuda_graph_cache: RefCell<Option<...>>`,
lazily created on the first capture-mode run. `CudaGraphMode::Off`
(the default) is a complete no-op — byte-for-byte identical to the
hybrid path before this change.

**Targets:** ≥1.5× ResNet-50 over hybrid alone (~33 ms → ~22 ms),
≥1.2× on MLP / SqueezeNet / MobileNetV2. Output `max_abs_diff`
between capture and per-op modes ≤ 1e-4 (same compute, just
different launch mechanism). See
`onnx-rt/tests/bench_vision_models.rs` for the
`*_hybrid_with_graph` benchmark variants.

### Multi-Stream Overlap (async-multistream-v1)

Layered on top of the hybrid + graph-capture path,
`SessionConfig::stream_config = StreamConfig::Overlap {
transfer_streams }` allocates a per-Session [`StreamPool`] holding:

* one **compute** stream for cuDNN / cuBLAS / `cudaGraphLaunch`
* `transfer_streams` **H2D** streams for asynchronous host→device
  input upload
* `transfer_streams` **D2H** streams for asynchronous device→host
  output download
* a small reusable [`Event`] pool

`transfer_streams` is capped at 2 (`Session::ensure_stream_pool`
returns `SessionError::InvalidConfig` if exceeded). Beyond ~2
transfer streams, contention on the PCIe / NVLink fabric eats the
overlap window.

**How it composes with capture (cuda-graphs-v1):**

The capture path is unchanged — it always captures on the cache's
internal capture stream so the resulting `cudaGraph_t` is
stream-agnostic. The replay path (`try_replay`) is what changes:

1. **H2D phase.** `cudaMemcpyAsync` the new inputs into the
   persistent input buffers on `pool.h2d[0]`.
2. **Cross-stream gate.** `cudaEventRecord` on `pool.h2d[0]`,
   `cudaStreamWaitEvent` on `pool.compute` — the compute stream
   queues a barrier waiting for the H2D event without the host
   blocking.
3. **Compute phase.** `cudaGraphLaunch` on `pool.compute`. The
   captured kernel sequence runs there, fed by the input pointers
   the H2D just filled.
4. **Cross-stream gate.** `cudaEventRecord` on `pool.compute`,
   `cudaStreamWaitEvent` on `pool.d2h[0]`.
5. **D2H phase.** `cudaMemcpyAsync` the persistent output buffers
   to host on `pool.d2h[0]`.
6. **One host sync.** `cudaStreamSynchronize(pool.d2h[0])` blocks
   the calling thread once at the end. The H2D, compute, and D2H
   for the *next* inference can already be in flight on different
   streams while this sync is waiting.

`StreamConfig::SingleStream` (the default) is a complete no-op —
every memcpy and kernel still goes to the cache's capture stream
(or the default stream when capture is also off), preserving the
exact byte-for-byte output of every previous mode.

**Targets:** ≥1.3× throughput vs single-stream on B=1 ResNet-50
serving loops; ≥1.5× combined with graph capture. Single-request
latency must not regress more than 5% under `Overlap`. See
`onnx-rt/tests/bench_vision_models.rs` once the multi-stream
bench variants land alongside the dynamic-batching change.

**Limitations:**

* The per-op (no-graph) path does not currently route through the
  pool — there's no clean wiring point because each op allocates
  intermediate device buffers via synchronous `cudaMalloc`. Pair
  `Overlap` with `CudaGraphMode::Capture` to see speedup.
* Sessions running multiple concurrent inference threads are
  outside scope — the pool is not yet thread-safe across `run`
  calls (use one Session per worker thread until that lands).

## Storage Layout & GPT Partition Type GUIDs

The `embedded-filesystem-v1` change establishes the on-disk
layout. SmallAIOS-specific partitions use registered type GUIDs so
external tooling (`gdisk`, `parted`, `lsblk`) can identify them
without colliding with any existing GUID in the public registry.

### v1 Partition Layout

| Idx | Type GUID                              | Size      | Purpose                                    |
|----:|----------------------------------------|-----------|--------------------------------------------|
|   1 | `C12A7328-F81F-11D2-BA4B-00A0C93EC93B` | 256 MiB   | UEFI ESP — bootloader + kernel image       |
|   2 | `A3F7C2E0-FACE-4FFF-AAAA-000000000001` | ~4 GiB    | SmallAIOS squashfs `/models/` slot A       |
|   3 | `A3F7C2E0-FACE-4FFF-AAAA-000000000002` | ~4 GiB    | SmallAIOS squashfs `/models/` slot B       |
|   4 | `8DA63339-0007-60C0-C436-083AC8230908` | remainder | Linux F2FS `/data/`                        |
|   5 | `A3F7C2E0-FACE-4FFF-BBBB-000000000000` | 8 MiB     | SmallAIOS A/B boot config (double-buffer)  |

Partitions 2 and 3 are equal-size and swappable per the A/B update
mechanism. The active slot is selected at boot via partition 5
(see `embedded-filesystem-v1`'s `fs-ab-boot` capability) and a
mirrored UEFI variable when available.

### SmallAIOS-registered GUID prefix

All SmallAIOS-specific partition type GUIDs share the prefix
`A3F7C2E0-FACE-4FFF-`, with a discriminator nibble identifying the
purpose:

| Nibble pattern   | Purpose                                      | Owning change                |
|------------------|----------------------------------------------|------------------------------|
| `AAAA-000000000001` | squashfs `/models/` slot A                | `embedded-filesystem-v1`     |
| `AAAA-000000000002` | squashfs `/models/` slot B                | `embedded-filesystem-v1`     |
| `BBBB-000000000000` | A/B boot config double-buffer             | `embedded-filesystem-v1`     |
| `CCCC-000000000001` | _reserved_ — overlay upper-layer partition (if ever split off `/data/`) | `embedded-overlay-v1` (deferred) |
| `DDDD-000000000001` | _reserved_ — raw-flash littlefs partition | `embedded-flash-fs-v1`       |

The `CCCC-...` and `DDDD-...` slots are pre-reserved here so the
overlay and flash-fs implementation phases can use them without
re-litigating the prefix scheme. v1 of overlay places its upper
layer as a subdir under `/data/` rather than its own partition;
the `CCCC` GUID is held in case a future v2 splits it out.

### Writable filesystem alternatives

| Target class                         | Writable FS         | Mount point  |
|--------------------------------------|---------------------|--------------|
| Block-device (eMMC, NVMe, SATA)      | F2FS (Linux 6.6)    | `/data/`     |
| Raw-flash MCU/FPGA (NOR via QSPI, NAND via ONFI) | littlefs v2.x | `/flash/` |

Targets MAY support both simultaneously (e.g., an embedded ARM
SoC with eMMC for bulk plus QSPI NOR for secure config). When
both are present, `/data/auth/shadow` lives on `/data/` (F2FS) and
the small-config `/flash/` is a separate read-write surface.

### Syscall numbering reservation

The `wave0-scaffolding-stubs` change reserved syscall numbers
ahead of implementation so the four management/filesystem changes
can land their phase PRs without colliding on `kernel/src/syscall/mod.rs`:

| Range          | Category | Owning change            | Status |
|----------------|----------|--------------------------|--------|
| 0x36–0x37      | ONNX     | `embedded-overlay-v1`    | reserved (model_add, model_remove) |
| 0x57           | System   | `embedded-filesystem-v1` | reserved (boot_success) |
| 0x90–0x95      | Auth (NEW) | `management-login-v1`  | reserved (login, logout, change_password, create_user, whoami, totp_setup) |
| 0x96–0x9F      | Auth     | future                   | reserved for additional auth syscalls |

`SYSCALL_TABLE_SIZE` was bumped from `0x90` to `0xA0` to cover the
new Auth category.
