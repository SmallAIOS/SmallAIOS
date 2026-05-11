# Design — tsn-integration-v1

## Goal

Make SmallAIOS a viable Time-Sensitive Networking (TSN) endpoint for industrial automation, in-vehicle networks, and similar deterministic control-loop deployments. Concretely: gPTP-synchronized to sub-microsecond accuracy with the TSN domain grandmaster; able to advertise its supported scheduled-traffic windows to upstream configurators; cooperative scheduler hooks that consume the schedule and meet deadlines at op-boundary granularity; explicit out-of-scope-on-Jetson positioning.

Success: (a) on an Intel i210-equipped industrial PC paired with a Hirschmann or Cisco TSN-capable switch, the unikernel synchronizes its clock to the grandmaster with mean offset <500 ns over a 10-minute window; (b) a configured 1 ms scheduled-traffic window opens predictably with NIC-enforced gate semantics; (c) an inference op tagged with that window's deadline either completes before the gate closes or the scheduler emits a structured warning + configured-action; (d) on the Jetson Orin production target without `--features tsn`, behavior is identical to develop.

## Alternatives considered

### A1. Treat TSN as a userspace concern (run `linuxptp` next to SmallAIOS)

**Not applicable.** SmallAIOS is a unikernel; there is no userspace partition where `linuxptp` could run, no shared `ethtool -T` plumbing. The gPTP daemon must live inside the kernel (or in the same address space — same thing in a unikernel). Furthermore, the scheduler integration requires kernel-side awareness of the TSN clock; deferring it to a userspace daemon would lose the deadline-propagation benefit.

### A2. Use 802.1Qav (credit-based shaping) instead of 802.1Qbv

**Considered, partially overlapping.** Qav is per-class credit-based shaping; Qbv is time-aware gate-based scheduling. Qav guarantees bounded latency per class but not a specific time window; Qbv guarantees specific time windows. For the **AI inference deadline** use case, Qbv is the better fit — we have a deadline aligned to a network schedule, not a bandwidth allocation. We may add Qav support in a follow-up if a use case (e.g., audio-style traffic class) demands it; v1 focuses on Qbv.

### A3. Implement TSN as a generic `bus` crate extension (alongside AFDX/ARINC664)

**Rejected.** TSN is fundamentally an Ethernet (802.1) standard suite, not a bus protocol. Its natural home is the `net` crate (where Ethernet, IPv4, IPv6, TCP, UDP live). The `bus` crate's AFDX support is also Ethernet-based, but it implements ARINC664 Part 7 virtual links — a specific avionics profile that, while conceptually similar, has its own conformance test suite and protocol semantics. Mixing them confuses the conformance story for both standards. They share the underlying Ethernet packet path but diverge at the upper layers.

### A4. Use a vendor-specific TSN library (e.g., Intel TSN Linux SDK port)

**Rejected.** The Intel TSN Linux SDK is GPL-licensed and Linux-kernel-specific (relies on `taprio` qdisc, `cbs` qdisc, ethtool TSN extensions). Porting it to a `#![no_std]` Rust unikernel is not feasible. We re-implement the standards-conformant pieces from the IEEE specs.

### A5. Implement gPTP without hardware timestamping (software-only)

**Considered as fallback.** Software-only PTP achieves ~100 µs accuracy at best on a busy system — well above the 1 µs target for TSN. We support software-only as a fallback for testing / development on NICs without PHC support, with a loud warning that sub-microsecond accuracy is unattainable. Production deployments must use a NIC with hardware timestamping (every NIC in the supported set has it; the issue arises only on QEMU virtio-net or on Jetson where Qbv enforcement is missing anyway).

## gPTP (802.1AS)

The 802.1AS profile of PTPv2 differs from base PTP in several ways important for the implementation:

- **Layer 2 multicast only** (destination MAC `01:80:C2:00:00:0E`) — no UDP. Simpler datapath than general PTP.
- **No best-master-clock-algorithm (BMCA) negotiation** by default — the grandmaster is configured / pinned, not elected.
- **Path delay measured via peer-delay mechanism** (PDelay_Req / PDelay_Resp / PDelay_Resp_Follow_Up between adjacent peers, not end-to-end Delay_Req as in base PTP).
- **Domain number** = 0 for the default 802.1AS profile; other domains are non-standard.

The daemon's state machine:

1. Send periodic `PDelay_Req` to the upstream peer. Capture TX timestamp.
2. Receive `PDelay_Resp` with peer's RX timestamp. Receive `PDelay_Resp_Follow_Up` with peer's TX timestamp. Compute mean propagation delay.
3. Receive `Sync` + `Follow_Up` from the grandmaster (forwarded by the upstream peer with residence-time correction). Compute clock offset.
4. Apply offset to local PHC (Precision Hardware Clock) via a NIC-driver-specific call.

Implementation: ~600 LOC in `net/src/tsn/gptp.rs`. Hardware-timestamp hooks abstracted via a `PhcDriver` trait that each supported NIC implements.

## 802.1Qbv Gate Control List

A Gate Control List is a cyclic schedule of `(gate_states, duration_ns)` tuples. `gate_states` is a bitmask of which traffic-class queues are open during that interval. The NIC enforces the schedule at the precise gPTP-synchronized boundary.

SmallAIOS-side responsibilities:

1. Read the schedule from `SMALLAIOS_TSN_SCHEDULE` (TOML).
2. Translate to the NIC's GCL register format.
3. Program the GCL via the `TsnNicDriver::set_gcl` trait method.
4. Acknowledge schedule activation (next-cycle vs immediate; vendor-specific).

The Gate Control List is **static** in v1 — we set it once at boot, and changes require a reboot or an explicit reconfiguration call. Dynamic schedules (e.g., schedule changes orchestrated by an upstream YANG-based configurator) are a follow-up.

NIC-specific notes:

- **Intel i210**: GCL programmed via `IGC_BASE_TIME` and `IGC_LAUNCHTIME` registers; up to 16 GCL entries; 1 µs gate-granularity. Documented in Intel's i210 datasheet section 7.4.
- **Intel i225/i226**: GCL up to 256 entries; ns-precision via Time Aware Scheduler hardware. Section 7.5 of the i225 datasheet.
- **Intel E810**: Datacenter-grade TSN; GCL up to 1024 entries.

Initial NIC support is i210 only; i225/i226/E810 are added in follow-up changes (each ~0.5-1 week).

## Scheduler deadline propagation

The cooperative scheduler today yields at op boundaries. We extend the op-boundary check with a deadline check:

```rust
pub struct OpDeadline {
    tsn_window_id: TsnWindowId,
    close_at_gptp_ns: u64,
    on_miss: DeadlineMissAction,  // Warn, Abort, or Continue
}

// Existing op boundary becomes:
fn yield_or_continue(&mut self) -> SchedulerAction {
    if let Some(deadline) = self.current_op_deadline() {
        let now_gptp = self.gptp_clock().now_ns();
        if now_gptp + self.estimated_next_op_cost_ns() > deadline.close_at_gptp_ns {
            return SchedulerAction::DeadlineMiss(deadline.on_miss);
        }
    }
    // ... existing logic
}
```

`estimated_next_op_cost_ns` is a heuristic based on op type + tensor sizes. We do not need it to be exact — we need it to be a useful warning signal so the operator can tune their schedule.

`DeadlineMissAction` is configurable per-window:

- `Warn`: log + emit a metric; continue executing. Suitable for soft-real-time use cases.
- `Abort`: abandon the op chain, emit a metric, await the next window. Suitable for hard-real-time where stale output is worse than no output.
- `Continue`: do nothing (most permissive — for early development).

## Schedule TOML format

```toml
[gptp]
domain = 0
grandmaster_priority1 = 128  # we are not the grandmaster

[qbv]
cycle_time_ns = 5_000_000  # 5 ms total cycle
nic = "i210"
interface = "eth0"

[[qbv.gate_control_list]]
duration_ns = 1_000_000  # 1 ms scheduled-traffic window
gates = ["TC7", "TC6"]
window_id = "inference-result"

[[qbv.gate_control_list]]
duration_ns = 4_000_000  # 4 ms best-effort window
gates = ["TC0", "TC1", "TC2", "TC3"]
window_id = "best-effort"

[[ops.deadline]]
op_pattern = "model:vision-inspect.onnx"
window_id = "inference-result"
on_miss = "warn"
```

## Observability

Extends the existing telemetry path with:

- `tsn.gptp.offset_ns_p50`, `tsn.gptp.offset_ns_p99`, `tsn.gptp.path_delay_ns`
- `tsn.qbv.cycles_completed`, `tsn.qbv.gate_open_late_count`
- `tsn.deadline.met_count{window_id=...}`, `tsn.deadline.miss_count{window_id=...}`

## What this change explicitly does NOT do

- Does not modify the Jetson Orin Ethernet code path (Tegra234 EQOS does not enforce Qbv anyway).
- Does not become a gPTP grandmaster.
- Does not implement 802.1Qci, 802.1CB, or 802.1Qbu in v1.
- Does not implement YANG-based dynamic configuration in v1 (TOML only).
- Does not modify the existing AFDX / ARINC664 path in the `bus` crate.
- Does not target TSN-over-wireless (802.11be, 5G URLLC).
- Does not change the existing `net` crate behavior when `--features tsn` is off.
