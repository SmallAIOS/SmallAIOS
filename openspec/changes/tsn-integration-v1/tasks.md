# Tasks — tsn-integration-v1

> **Status: Future-facing.** No work has started. This change is a roadmap document for industrial-automation / in-vehicle TSN deployments. The Jetson Orin AI-inference sweet spot does not need TSN.

## 0. Trigger conditions (review before starting)

- [ ] 0.1 Confirm at least one production / customer target uses a TSN-orchestrated network (industrial PLC + TSN switch + SmallAIOS inference node, or in-vehicle central-compute + TSN switch).
- [ ] 0.2 Procure / borrow at least one TSN-capable NIC (Intel i210 minimum) and one TSN-capable Ethernet switch (Hirschmann, Cisco IE-3400, Beckhoff, or equivalent) for integration testing.
- [ ] 0.3 Identify a reference industrial workload to anchor the implementation (e.g., vision-inspection inference aligned to a 5 ms TSN cycle, 1 ms scheduled-traffic window).

## 1. Phase 1 — gPTP daemon (802.1AS)

### 1a. Cargo feature + module scaffolding

- [ ] 1.1 Add `tsn` Cargo feature to `net/Cargo.toml` with doc-comment describing the scope (industrial / in-vehicle TSN endpoints; Jetson out-of-scope for Qbv enforcement).
- [ ] 1.2 Create `net/src/tsn/mod.rs` exposing the public API (`Gptp`, `Qbv`, `TsnNicDriver` trait).
- [ ] 1.3 Add `PhcDriver` trait abstracting NIC hardware timestamping.

### 1b. PTPv2 packet handling

- [ ] 1.4 Implement PTPv2 message types: `Sync`, `Follow_Up`, `PDelay_Req`, `PDelay_Resp`, `PDelay_Resp_Follow_Up`. Layer 2 framing with destination MAC `01:80:C2:00:00:0E` and ethertype `0x88F7`.
- [ ] 1.5 Implement the 802.1AS state machine in `net/src/tsn/gptp.rs` — peer-delay measurement, offset computation, clock adjustment.
- [ ] 1.6 Hook the daemon into the existing `net` packet path with priority over best-effort traffic (gPTP is the foundation everything else depends on).

### 1c. Hardware-timestamp shim

- [ ] 1.7 Define `PhcDriver` methods: `get_phc_time()`, `adjust_phc_freq(ppb)`, `step_phc(offset_ns)`, `rx_timestamp(packet)`, `tx_timestamp(packet)`.
- [ ] 1.8 Implement `PhcDriver` for Intel i210 (canonical reference NIC). Reads / writes `IGC_SYSTIM`, `IGC_RXSTMPL`, `IGC_TXSTMPL` registers per datasheet section 7.4.
- [ ] 1.9 Implement software-timestamp fallback with a loud `warn!` at boot indicating sub-microsecond accuracy is unattainable in this mode.

### 1d. gPTP unit + integration tests

- [ ] 1.10 Unit tests for PTPv2 packet parsing / serialization (captured wire-format fixtures committed under `net/src/tsn/test-data/`).
- [ ] 1.11 Integration test against a real i210 + TSN switch: mean offset over 10 minutes SHALL be <500 ns.

## 2. Phase 2 — 802.1Qbv scheduled traffic

### 2a. Gate Control List

- [ ] 2.1 Define `GateControlList` and `GateInterval` types in `net/src/tsn/qbv.rs`.
- [ ] 2.2 Parse the GCL from TOML configuration (`SMALLAIOS_TSN_SCHEDULE`).
- [ ] 2.3 Validate the GCL: total cycle time matches `cycle_time_ns`, all gate-state bitmasks are well-formed.

### 2b. NIC-driver shim

- [ ] 2.4 Define `TsnNicDriver::set_gcl(gcl, base_time)` trait method.
- [ ] 2.5 Implement for Intel i210 — translate the GCL into i210's `IGC_BASE_TIME` / `IGC_LAUNCHTIME` register layout, up to 16 entries.
- [ ] 2.6 Document the per-NIC quirks (i210 gate-granularity is 1 µs; ns-precision GCL must round; etc.).

### 2c. Qbv tests

- [ ] 2.7 Unit tests for GCL parsing + translation to NIC registers.
- [ ] 2.8 Integration test: program a 5 ms cycle with a 1 ms inference window + 4 ms best-effort window; capture wire-line timing via an external TAP + Wireshark / `tcpdump`; confirm the inference-window frames are transmitted exclusively within the 1 ms gate-open interval.

## 3. Phase 3 — Scheduler deadline propagation

- [ ] 3.1 Add `OpDeadline` and `DeadlineMissAction` types to `kernel/src/sched/tsn.rs`.
- [ ] 3.2 Modify the cooperative-scheduler op-boundary yield to check the active deadline and emit `SchedulerAction::DeadlineMiss(action)` when the next op's estimated cost would exceed the deadline.
- [ ] 3.3 Implement the three `DeadlineMissAction` variants: `Warn`, `Abort`, `Continue`.
- [ ] 3.4 Add `estimated_op_cost_ns()` heuristic — table-driven, indexed by op type + tensor shape. Initial estimates can be conservative; refinement is a follow-up.

## 4. Phase 4 — Configuration + observability

- [ ] 4.1 Define the TOML schedule format (see `design.md`) and add parser + validator.
- [ ] 4.2 Wire `SMALLAIOS_TSN_SCHEDULE` env var (container path) and equivalent kernel boot argument.
- [ ] 4.3 Extend telemetry with `tsn.gptp.offset_ns_p50/p99`, `tsn.gptp.path_delay_ns`, `tsn.qbv.cycles_completed`, `tsn.qbv.gate_open_late_count`, `tsn.deadline.met_count{window_id}`, `tsn.deadline.miss_count{window_id}`.
- [ ] 4.4 Add a TSN-aware example to `examples/` and a TOML snippet to the docs.

## 5. Phase 5 — Documentation

- [ ] 5.1 Create `docs/tsn-integration.md` covering: standards (802.1AS, 802.1Qbv), hardware compatibility matrix, TOML schedule format, deadline-miss handling, troubleshooting (clock not syncing, gates not enforcing, deadlines missed).
- [ ] 5.2 Add a clear **Jetson out-of-scope** callout: Tegra234 EQOS supports gPTP hardware timestamping (so software gPTP works) but does NOT enforce 802.1Qbv gates. Jetson SmallAIOS can be a gPTP slave but not a scheduled-traffic endpoint with strong guarantees.
- [ ] 5.3 Add a row to the README hardware matrix for "TSN-capable industrial deployments" linking to `docs/tsn-integration.md`.

## 6. Phase 6 — CI

- [ ] 6.1 Add a host-build CI job that compiles `--features tsn` (no TSN hardware on GitHub-hosted runners; validates compile-time correctness).
- [ ] 6.2 Add a scheduled job `tsn-i210-smoke` that runs gPTP + Qbv against a self-hosted i210 + TSN switch (when available). Advisory initially.
- [ ] 6.3 Promote `tsn-i210-smoke` to `change-gates` when the self-hosted runner availability is reliable (separate change).

## 7. Close-out

- [ ] 7.1 PR title: `feat(net): tsn-integration-v1 — 802.1AS gPTP + 802.1Qbv scheduled traffic + deadline-aware scheduler`.
- [ ] 7.2 Reviewer sign-off + green CI + i210 + TSN switch integration evidence (clock offset histogram + Qbv wire-capture) in the PR description.
- [ ] 7.3 Update CLAUDE.md "Current state" to mention TSN endpoint capability with the explicit Jetson-out-of-scope note.
