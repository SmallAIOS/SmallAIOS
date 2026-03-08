## Why

SonarCloud flags 21 functions across 8 crates for cognitive complexity exceeding the threshold of 15 (rule `rust:S3776`). The worst offender scores 58 — nearly 4x the limit. While total technical debt is modest (~126 minutes), these functions are concentrated in safety-critical code paths (crypto signing, TCP state machine, memory allocator, ONNX operators) where readability directly impacts correctness and auditability. Reducing complexity now improves DO-178C compliance posture and makes these functions easier to review, test, and maintain.

## What Changes

- Refactor 21 functions to reduce cognitive complexity to ≤ 15 each
- Extract helper functions from deeply nested loops and match arms
- Decompose large state machines into per-state handler methods
- Introduce shared broadcast/iteration utilities for ONNX operator patterns
- No public API changes — all refactoring is internal to function bodies
- No behavioral changes — all existing tests must continue to pass

### Functions by severity

**Tier 1 — Severely over threshold (>30):**
- `ml_dsa_65_sign()` (58) — `security/src/crypto/ml_dsa.rs:1130`
- `ModeCodeProcessor::process()` (42) — `bus/src/mil1553/mode_code.rs:174`
- `op_conv()` (40) — `onnx-rt/src/operators.rs:667`
- `TcpConnection::on_segment()` (33) — `net/src/tcp.rs:334`
- `parse_dtb()` (32) — `kernel/src/mem/phys.rs:291`

**Tier 2 — Moderately over (20–30):**
- `KernelAllocator::alloc()` (29) — `kernel/src/mem/global.rs:104`
- `LongHeader::decode()` (29) — `net/src/quic/packet.rs:207`
- `op_add()` (27) — `onnx-rt/src/operators.rs:351`
- `op_softmax()` (27) — `onnx-rt/src/operators.rs:507`
- `validate_model()` (23) — `onnx-rt/src/session.rs:200`
- `VfsTree::lookup()` (23) — `posix/src/vfs.rs:174`
- `build_execution_graph()` (23) — `onnx-rt/src/graph.rs:157`
- `KeccakState::permute()` (20) — `security/src/crypto/sha3.rs:127`
- `Scheduler::poll()` (20) — `bus/src/arinc429/scheduler.rs:152`

**Tier 3 — Slightly over (16–19):**
- `ml_dsa_65_verify()` (19) — `security/src/crypto/ml_dsa.rs:1348`
- `parse_ndp_options()` (19) — `net/src/ndp.rs:408`
- `op_reshape()` (19) — `onnx-rt/src/operators.rs:595`
- `plan_memory()` (18) — `onnx-rt/src/memory_planner.rs:201`
- `CanZenohAdapter::transmit()` (17) — `bus/src/can/adapter.rs:100`
- `topological_sort()` (16) — `onnx-rt/src/graph.rs:239`
- `trace_status()` (16) — `kernel/src/safety/traceability.rs:248`

## Capabilities

### New Capabilities

- `onnx-operator-complexity`: Refactor 8 ONNX runtime functions (op_conv, op_add, op_softmax, op_reshape, validate_model, build_execution_graph, plan_memory, topological_sort) — extract shared broadcast iteration, decompose validation logic
- `crypto-complexity`: Refactor 3 security/crypto functions (ml_dsa_65_sign, ml_dsa_65_verify, KeccakState::permute) — extract NTT preparation, rejection sampling, and round helpers
- `kernel-complexity`: Refactor 3 kernel functions (parse_dtb, KernelAllocator::alloc, trace_status) — extract token handlers and allocation fallback chains
- `network-complexity`: Refactor 3 network functions (TcpConnection::on_segment, LongHeader::decode, parse_ndp_options) — extract per-state TCP handlers and per-type QUIC decoders
- `bus-complexity`: Refactor 3 bus functions (ModeCodeProcessor::process, Scheduler::poll, CanZenohAdapter::transmit) — extract broadcast response helpers and frame construction
- `posix-complexity`: Refactor VfsTree::lookup — simplify path traversal logic

### Modified Capabilities

_(No spec-level behavioral changes — all modifications are internal refactoring.)_

## Impact

- **Code:** 21 functions across 8 crates (onnx-rt, security, kernel, bus, net, posix)
- **APIs:** No public API changes; all refactoring is internal
- **Tests:** All existing tests must pass unchanged; new unit tests for extracted helpers
- **Dependencies:** No new dependencies
- **CI:** SonarCloud code smell count should drop from 21 to 0
