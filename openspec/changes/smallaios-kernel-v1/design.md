## Context

No existing operating system is purpose-built for AI inference with safety-critical assurance. Current approaches run inference on general-purpose Linux — carrying ~450 syscalls, massive attack surface, and no formal verification of core properties. SmallAIOS fills this gap: a clean-room Rust unikernel that boots to ONNX inference on x86-64, ARM64, and NVIDIA GPU with < 50ms startup, < 8 MB base footprint, and formal proofs of critical properties.

Target hardware ranges from Raspberry Pi (edge testing) to NVIDIA DGX Spark (production inference). The kernel is Apache-2.0 licensed, uses no GPL code, and follows DO-178C DAL A development practices.

## Goals / Non-Goals

**Goals:**
- Boot to ONNX inference in < 50ms (container mode)
- ~46 syscalls (vs Linux ~450) — minimal attack surface
- Native PQC cryptography (ML-KEM-768, ML-DSA-65)
- Hardware acceleration: AVX2/AVX-512/AMX (x86), NEON/SVE (ARM), NVIDIA tensor cores
- 100% MC/DC structural coverage on safety-critical paths
- Formal verification of scheduler, allocator, and IPC protocol properties
- Sphinx-needs traceability from requirements through code to test results
- Container-native (Docker/K8s) with < 15 MB images

**Non-Goals:**
- General-purpose computing (no shell, no package manager, no users)
- Training workloads (inference only)
- Non-ONNX model formats
- Non-NVIDIA GPU vendors (future work)
- Full POSIX compliance (only the minimal subset needed)
- GUI or display support

## Decisions

### Decision 1: Unikernel architecture (not microkernel or monolithic)

**Choice**: Library OS / unikernel — single address space, single binary.

**Rationale**: SmallAIOS runs exactly one workload (ONNX inference). A unikernel eliminates TLB flush overhead, ring transitions, and IPC costs between kernel and userspace. The entire system compiles to a single static binary that boots in milliseconds.

**Alternatives considered**:
- Microkernel (Mach/L4 style): Adds IPC overhead (~1-5us per message) for component isolation we don't need — all components are equally trusted.
- Monolithic (Linux-style): Carries unnecessary subsystems. Impossible to build clean-room in reasonable time.

### Decision 2: Cooperative async scheduling (not preemptive)

**Choice**: Rust async/await with work-stealing executor, cooperative yielding at operator boundaries.

**Rationale**: AI inference is a pipeline of operators. Preemption mid-operator wastes cache state. Cooperative yielding at operator boundaries provides natural scheduling points. Rust's zero-cost futures eliminate allocation overhead.

**Risk**: A buggy operator that never yields could starve other tasks → Mitigated with watchdog timer that logs warnings after configurable timeout.

### Decision 3: Clean-room ONNX runtime (not wrapping onnxruntime)

**Choice**: Write ONNX parser, optimizer, and execution providers from scratch.

**Rationale**: The Microsoft onnxruntime is MIT-licensed (compatible), but it's C++ with heavy dependencies (protobuf, MLAS, etc.) that can't easily run in a `#![no_std]` kernel. A from-scratch Rust implementation gives us zero-copy data paths, no C++ FFI, and full control over memory layout.

**Risk**: Operator coverage takes time → Mitigate by prioritizing Tier 1 ops (covers ~90% of production models).

### Decision 4: Zenoh-compatible IPC (not gRPC/REST)

**Choice**: Zenoh-inspired pub/sub with key-expression routing, wire-protocol compatible with Zenoh clients.

**Rationale**: Zenoh's pub/sub model is lighter than gRPC (no protobuf dependency, no HTTP/2) and supports zero-copy shared memory transport. Wire compatibility means existing Zenoh clients work without modification.

### Decision 5: Post-quantum cryptography as default

**Choice**: ML-KEM-768 + ML-DSA-65 in hybrid mode (combined with X25519 + Ed25519).

**Rationale**: Harvest-now-decrypt-later attacks threaten long-lived deployments. NIST FIPS 203/204 are finalized standards. Hybrid mode provides defense-in-depth — both classical and PQC must be broken.

### Decision 6: DO-178C DAL A with MC/DC 100% coverage

**Choice**: Follow DO-178C (aerospace software assurance) at Design Assurance Level A (most stringent) for all safety-critical kernel paths.

**Rationale**: While SmallAIOS isn't avionics, DAL A provides the most rigorous framework for high-assurance software: full requirements traceability, MC/DC coverage, independence of verification, and formal methods. This level of assurance is appropriate for AI systems making safety-relevant decisions (autonomous vehicles, medical devices, industrial control).

**Adaptation for Rust**: MISRA-C:2023 rules are adapted for Rust idioms — many MISRA rules (null pointer checks, array bounds, memory management) are enforced by the Rust compiler. The remaining rules focus on unsafe code discipline, error handling, and coding style.

### Decision 7: Multi-tool formal verification (TLA+, Lean 4, SPIN)

**Choice**: Use different verification tools for different properties based on their strengths.

**Rationale**:
- **TLA+** (Leslie Lamport): Best for concurrent systems — scheduler properties (deadlock freedom, starvation freedom, fairness), allocator properties (no double-free, conservation of memory). TLA+'s TLC model checker exhaustively explores state spaces.
- **Lean 4**: Best for mathematical proofs about types and invariants — tensor shape correctness, capability type safety, cryptographic protocol properties. Lean's dependent type system enables machine-checked proofs.
- **SPIN** (Promela): Best for protocol verification — IPC message delivery guarantees, TCP state machine correctness, pub/sub routing correctness. SPIN's LTL model checking verifies temporal properties.

### Decision 8: Sphinx-needs + PlantUML documentation

**Choice**: Sphinx-needs for requirements engineering, PlantUML for architecture diagrams, integrated into CI.

**Rationale**: Sphinx-needs provides bidirectional traceability (REQ ↔ SPEC ↔ IMPL ↔ TEST ↔ VERIFY) required by DO-178C. PlantUML generates consistent diagrams from text (version-controllable, diff-able). Both integrate with Sphinx for a single documentation build.

### Decision 9: Soft real-time scheduling with operator-level preemption

**Choice**: Three-class priority scheduler (SYSTEM > IPC > INFERENCE) with mandatory yield points between every ONNX operator and per-operator time budgets.

**Rationale**: ONNX inference is inherently variable-time — model sizes range from 1 MB (MobileNet, ~2ms) to multi-GB (LLM, seconds). A traditional hard-RTOS model doesn't fit because operator execution times are data-dependent and GPU offload introduces non-deterministic latency. Instead, SmallAIOS provides:
- **Hard-RT for system health**: Watchdog, health probes, and syslog always preempt inference
- **Soft-RT for IPC**: External communication preempts at operator boundaries (< 10ms target)
- **Observability for inference**: Per-operator time budgets with logging, not hard deadlines
- **Edge WCET calibration**: For constrained hardware (Jetson, RPi), runtime calibration provides per-operator worst-case estimates

**Alternatives considered**:
- Hard RTOS with fixed time slots: Doesn't work — operator times vary by orders of magnitude
- Fully preemptive scheduling: Wastes cache state mid-operator (GEMM kernels are cache-blocked)
- No priority classes: Health checks could be starved by long inference chains

**Risk**: A buggy operator that never returns will block its core → Mitigated with hardware watchdog (30s default) that triggers system reset, plus per-operator hard timeout (configurable, default 10x budget).

## Risks / Trade-offs

- [NVIDIA GPU driver complexity] → Start with container mode (host driver), add bare metal driver incrementally. Reference open-gpu-kernel-modules MIT portions.
- [Clean-room ONNX runtime performance gap] → Benchmark continuously against onnxruntime. Focus SIMD kernel optimization on GEMM (the dominant cost).
- [Formal verification state explosion] → Bound model sizes. Verify abstract models, not full implementation. Use compositional verification.
- [MC/DC 100% coverage effort] → Start with kernel core and safety-critical paths. Expand coverage incrementally. Use coverage-guided fuzzing to reach edge cases.
- [Rust nightly dependency] → Pin exact nightly version. Track stabilization of required features. Maintain fallback for feature regressions.

## Open Questions

1. Should we support ONNX model training (backward pass) in a future version?
2. Should we add AMD ROCm GPU support alongside NVIDIA?
3. What certification body (if any) should review the DO-178C artifacts?
4. Should the formal verification models be executable specifications that generate test cases?
