## 1. ONNX Runtime — Shared Utilities & Operators (onnx-rt)

- [x] 1.1 Extract `BroadcastIter` coordinate iterator utility in `onnx-rt/src/operators.rs` (shared by op_add, op_softmax, op_reshape)
- [x] 1.2 Refactor `op_conv()` — extract `convolve_at()` inner kernel helper (complexity 40 → ≤ 15)
- [x] 1.3 Refactor `op_add()` — use `BroadcastIter` for coordinate iteration (complexity 27 → ≤ 15)
- [x] 1.4 Refactor `op_softmax()` — use `BroadcastIter` for coordinate iteration (complexity 27 → ≤ 15)
- [x] 1.5 Refactor `op_reshape()` — simplify shape inference logic (complexity 19 → ≤ 15)
- [x] 1.6 Refactor `validate_model()` in `session.rs` — extract validation sub-checks into helpers (complexity 23 → ≤ 15)
- [x] 1.7 Refactor `build_execution_graph()` in `graph.rs` — extract graph construction phases (complexity 23 → ≤ 15)
- [x] 1.8 Refactor `plan_memory()` in `memory_planner.rs` — simplify allocation planning (complexity 18 → ≤ 15)
- [x] 1.9 Refactor `topological_sort()` in `graph.rs` — flatten nested logic (complexity 16 → ≤ 15)
- [x] 1.10 Run `cargo test -p smallaios-onnx-rt` and `cargo clippy -p smallaios-onnx-rt -- -D warnings` — verify all pass

## 2. Security — Crypto Functions (security)

- [x] 2.1 Refactor `ml_dsa_65_sign()` — extract `prepare_ntt_vectors()`, `compute_challenge()`, `check_signature_norms()`, `pack_ml_dsa_signature()` (complexity 58 → ≤ 15)
- [x] 2.2 Refactor `ml_dsa_65_verify()` — extract `reconstruct_and_check()` helper (complexity 19 → ≤ 15)
- [x] 2.3 Refactor `KeccakState::permute()` — extract theta/rho/pi/chi/iota round step helpers (complexity 20 → ≤ 15)
- [x] 2.4 Run `cargo test -p smallaios-security` and `cargo clippy -p smallaios-security -- -D warnings` — verify all pass

## 3. Kernel — Memory & Traceability (kernel)

- [x] 3.1 Refactor `parse_dtb()` — extract `handle_begin_node()`, `handle_prop()`, `handle_end_node()`, `is_memory_node()`, `parse_reg_property()` (complexity 32 → ≤ 15)
- [x] 3.2 Refactor `KernelAllocator::alloc()` — extract `try_slab_alloc()`, `try_buddy_alloc()`, compose with `or_else` chain (complexity 29 → ≤ 15)
- [x] 3.3 Refactor `trace_status()` — simplify conditional logic (complexity 16 → ≤ 15)
- [x] 3.4 Run `cargo test -p smallaios-kernel` and `cargo clippy -p smallaios-kernel -- -D warnings` — verify all pass

## 4. Network — TCP, QUIC, NDP (net)

- [x] 4.1 Refactor `TcpConnection::on_segment()` — extract per-state handlers: `handle_listen()`, `handle_syn_sent()`, `handle_established()`, etc. (complexity 33 → ≤ 15)
- [x] 4.2 Refactor `LongHeader::decode()` — extract per-packet-type decoders: `decode_initial()`, `decode_handshake()`, `decode_zero_rtt()`, `decode_retry()` (complexity 29 → ≤ 15)
- [x] 4.3 Refactor `parse_ndp_options()` — extract per-option-type helpers (complexity 19 → ≤ 15)
- [x] 4.4 Run `cargo test -p smallaios-net` and `cargo clippy -p smallaios-net -- -D warnings` — verify all pass

## 5. Bus — MIL-STD-1553, ARINC 429, CAN (bus)

- [x] 5.1 Refactor `ModeCodeProcessor::process()` — extract `broadcast_response()` helper to deduplicate 13 match arms (complexity 42 → ≤ 15)
- [x] 5.2 Refactor `Scheduler::poll()` — extract scheduling phase helpers (complexity 20 → ≤ 15)
- [x] 5.3 Refactor `CanZenohAdapter::transmit()` — extract frame construction helper (complexity 17 → ≤ 15)
- [x] 5.4 Run `cargo test -p smallaios-bus` and `cargo clippy -p smallaios-bus -- -D warnings` — verify all pass

## 6. POSIX — VFS (posix)

- [x] 6.1 Refactor `VfsTree::lookup()` — extract component resolution helper, simplify path traversal (complexity 23 → ≤ 15)
- [x] 6.2 Run `cargo test -p smallaios-posix` and `cargo clippy -p smallaios-posix -- -D warnings` — verify all pass

## 7. Validation

- [x] 7.1 Run full workspace tests: `make test` — all pass
- [x] 7.2 Run full workspace clippy: `make clippy` — all pass
- [x] 7.3 Run `cargo fmt --check` — no formatting issues
- [ ] 7.4 Push branch and verify SonarCloud reports 0 code smells on PR
