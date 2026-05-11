# tsn-integration-v1

## Summary

Integrate **Time-Sensitive Networking (TSN)** support into the SmallAIOS networking stack and cooperative scheduler so that the unikernel can participate as a deadline-driven node in industrial-automation control loops where the network schedule and the inference schedule must align. TSN is a set of IEEE 802.1 standards that turn ordinary Ethernet into a deterministic-latency fabric — the canonical scale-out shape for factory floors, robotic cells, automotive in-vehicle networks, and time-sensitive measurement systems. This change adds:

- **IEEE 802.1AS (generalized Precision Time Protocol / gPTP)** clock synchronization to sub-microsecond accuracy across TSN domain peers. The existing `bus` crate already touches similar concepts via ARINC664 / AFDX, but the 802.1AS profile is the standard industrial / automotive shape.
- **IEEE 802.1Qbv (time-aware scheduled traffic gates)** so that scheduled traffic on a TSN-enabled NIC is gated open / closed at the gPTP-synchronized clock boundaries. SmallAIOS-side bookkeeping for the schedule; the NIC enforces the gate.
- **Cooperative scheduler hooks** that consume the TSN schedule and emit inference deadlines aligned with the scheduled-traffic windows. Inference op N must complete before window K closes; the scheduler honors the deadline as a soft preemption point at op boundaries.

This is a Tier 3 industrial-automation feature. The Jetson Orin AI-inference sweet spot (a Jetson NX running an ONNX model behind an HTTP endpoint over best-effort Ethernet) does not need TSN. The proposal documents the integration shape for when SmallAIOS is deployed as a factory-floor controller or in-vehicle inference node.

## Why

- **Industrial AI inference is increasingly TSN-orchestrated.** The 2024-2026 industrial-automation shift is from PROFINET / EtherCAT (deterministic but proprietary or limited-interop fieldbuses) to TSN over standard Ethernet (deterministic *and* interoperable). Major industrial-controls vendors (Siemens, Beckhoff, Rockwell, B&R) are converging on TSN. AI vision inspection, predictive-maintenance inference, robotic-control regression — all of these are increasingly TSN endpoints. SmallAIOS as an inference engine on a factory floor needs to speak the same time-aware protocol as the PLCs around it.
- **The unikernel is the right shape for TSN.** A cooperative scheduler with deterministic op boundaries is closer to a TSN endpoint's natural shape than a Linux real-time-patched kernel. SmallAIOS doesn't need to fight scheduler jitter, page-fault latency, or interrupt threads competing for CPU — the boundaries are explicit. This is the same reason RTOSes (FreeRTOS, Zephyr, VxWorks) have TSN integrations: deterministic schedulers compose cleanly with deterministic networks.
- **Bus crate is partway there.** The existing `bus` crate's AFDX / ARINC664 path implements bounded-latency virtual link concepts that share DNA with TSN's time-aware shaping. The cooperative scheduler already has the notion of an op boundary as a yield point. The new code is **(a)** gPTP daemon (clock sync over the network), **(b)** NIC-driver hooks for 802.1Qbv schedule programming, **(c)** scheduler-side deadline propagation. Each piece is bounded.
- **Jetson does not need this for typical AI use cases.** Jetson Orin running a customer-facing inference endpoint over best-effort TCP/IP has zero TSN requirements; the latency budget is dominated by GPU compute time, not network jitter. The TSN path adds value only when SmallAIOS is the slow link in a 1 ms control loop — that's a factory-floor or automotive workload, not the Jetson sweet spot.

## Hardware prerequisites

### TSN-capable NICs

| NIC family | TSN features supported | Notes |
|------------|------------------------|-------|
| **Intel i210** | 802.1AS, 802.1Qbv, 802.1Qav | Common entry-level TSN NIC. Single-port 1 GbE; well-documented. |
| **Intel i225 / i226 (Foxville)** | 802.1AS, 802.1Qbv, 802.1Qav, 802.1Qbu (Frame Preemption) | 2.5 GbE; common on industrial PCs (e.g., Siemens IPC227G) |
| **Intel I350** | 802.1AS, 802.1Qbv (limited) | Older but widely deployed |
| **Intel E810** | Full TSN: 802.1AS / Qbv / Qav / Qbu / Qci | Datacenter / industrial; 10-100 GbE |
| **Marvell Prestera DX / Falcon** | 802.1AS / Qbv / Qav | Industrial / automotive switch SoCs |
| **NXP S32G with built-in TSN switch** | 802.1AS / Qbv / Qav | Automotive central compute |
| **Microchip LAN9662 / LAN9668** | 802.1AS / Qbv / Qav | Industrial Ethernet switches |

### TSN-incapable NICs (out of scope)

| NIC | Reason |
|-----|--------|
| **Tegra234 EQOS Ethernet (Jetson Orin)** | No 802.1Qbv gate enforcement in hardware; only 802.1AS PHC support |
| **Realtek RTL8111 / RTL8125** | Consumer; no TSN features |
| **Most cloud-VM virtio-net adapters** | Paravirtualized; no TSN guarantees |
| **Apple silicon Ethernet** | No public TSN feature support |

Note: the Jetson Orin's NIC supports gPTP hardware timestamping (so 802.1AS can run), but it does not enforce 802.1Qbv gates. A Jetson can be a gPTP **time slave** in a TSN domain but cannot be a scheduled-traffic endpoint with strong guarantees. This is one of the reasons TSN is out of scope for the typical Jetson workload.

### Domain partners

A TSN deployment requires at minimum:

- One gPTP grandmaster (often a TSN switch like Cisco IE-3400, Hirschmann RSPE, or Beckhoff CU2208).
- TSN-aware Ethernet switches between endpoints (or a single switch).
- Network configuration via the IEEE 802.1Qcc Network Configuration model (typically a YANG-based netconf flow or a vendor-specific tool like TTTech's Slate XNS).

SmallAIOS interoperates with this topology as an **endpoint**, not as a switch or grandmaster.

## What changes

- **Capability `net-tsn` (new):** owned by the `net` crate, gated behind a new `tsn` Cargo feature.
- **gPTP daemon** (`net/src/tsn/gptp.rs`): IEEE 802.1AS profile of PTPv2. Hardware-timestamping aware (uses NIC PHC where available, falls back to software timestamping otherwise — with a loud warning that sub-microsecond accuracy is unattainable in software-only mode).
- **802.1Qbv schedule programming** (`net/src/tsn/qbv.rs`): Gate Control List (GCL) construction; NIC-driver shim that writes the GCL to the hardware (vendor-specific, abstracted behind a trait). Initial NIC support: Intel i210 (canonical reference for "minimum viable TSN NIC"). Subsequent NICs added in follow-up changes.
- **Scheduler hooks** (`kernel/src/sched/tsn.rs`): the cooperative scheduler exposes a `Deadline { tsn_window_id, close_at_gptp_time }` type. Inference ops associated with a TSN window are dispatched with the deadline; the scheduler tracks remaining time at each op-boundary yield and emits a warning (or aborts the chain, configurable) if a deadline is at risk of being missed.
- **Configuration**: a TOML schedule file (`SMALLAIOS_TSN_SCHEDULE`) describes the Gate Control List, the gPTP domain, and the mapping from inference ops to windows.
- **Documentation**: `docs/tsn-integration.md` covering the standards landscape, the SmallAIOS configuration model, hardware compatibility, and an end-to-end example (industrial vision inspection with a 1 ms inference budget aligned to a 5 ms TSN cycle).

## Out of scope

- **Becoming a gPTP grandmaster.** SmallAIOS is a TSN endpoint, not a clock source. Grandmaster duties belong on a dedicated TSN switch or appliance.
- **802.1Qci (Per-Stream Filtering and Policing).** Adds ingress traffic-policing per-stream; useful for switch nodes but less so for endpoints. Defer.
- **802.1CB (Frame Replication and Elimination for Reliability).** Frame replication / duplicate elimination across redundant paths; targeted at high-reliability industrial / automotive deployments. Defer to a follow-up change once basic TSN endpoint capability is in.
- **802.1Qbu (Frame Preemption).** Allows large best-effort frames to be preempted by an urgent express frame mid-transmission. Hardware-dependent; defer until at least one NIC with Qbu enforcement is in the supported set and a use case demands it.
- **TSN over wireless (802.11be Wi-Fi 7 with TSN, 5G URLLC).** Wired-Ethernet TSN only for v1.
- **Building a TSN configuration GUI / YANG client.** Configuration is via static TOML in v1; YANG-based dynamic configuration is a follow-up.
- **Modifying the Jetson Orin Ethernet path.** Tegra234 EQOS does not enforce 802.1Qbv. The unikernel will run gPTP-only mode on Jetson (clock sync, no scheduled traffic) — useful in some research scenarios but not the production target for this change.
- **AFDX / ARINC664 changes.** The bus crate's existing avionics-bus paths are not modified by this change. TSN and AFDX share concepts but are distinct standards stacks. TSN-ization of the bus crate is a future concern.

## When this becomes important

- **Now (deferred):** Jetson Orin AI inference behind an HTTP endpoint — zero TSN requirement. Treat as roadmap documentation.
- **Trigger event 1:** SmallAIOS deployment as an industrial vision-inspection node, predictive-maintenance inference engine, or robotic-control coprocessor where the upstream PLC enforces a TSN cycle.
- **Trigger event 2:** SmallAIOS deployment in an automotive central-compute platform (e.g., NXP S32G-based zonal controller) where TSN is the in-vehicle backbone.
- **Likely horizon:** 12-24 months out, contingent on industrial-automation or automotive design wins. If those segments don't materialize, this change can be archived.

## Effort estimate

| Sub-phase | Scope | Estimate |
|-----------|-------|----------|
| 1 | gPTP daemon (802.1AS profile, hardware-timestamping shim) | ~1.5 weeks |
| 2 | 802.1Qbv GCL construction + Intel i210 driver shim | ~1 week |
| 3 | Scheduler deadline hooks + op-boundary deadline tracking | ~0.5 week |
| 4 | TOML schedule format + bench harness + docs | ~0.5-1 week |
| **Total** | | **~3-4 weeks** |

Each additional supported NIC adds ~0.5-1 week (mostly vendor-specific register programming for the gate control list).
