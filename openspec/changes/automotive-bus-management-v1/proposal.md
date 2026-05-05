## Why

SmallAIOS already ships a CAN bus inference bridge
(`SMALLAIOS_BUS_BACKEND=can`, `docs/can-inference.md`,
`examples/can-routes.toml`) — the data-plane path of "ONNX in,
CAN frame out" is working. What is missing is the **management
plane on the same wire**: when a SmallAIOS unit is deployed as
an ECU on a vehicle bus, an industrial controller on a CANopen
segment, or a fleet sensor reachable only through the
diagnostic port, there is no IP network for the
`management-login-v1` Zenoh keyspace to ride on.

The automotive and industrial worlds have spent four decades
standardizing exactly the management surface we just designed
(login / authenticate / read telemetry / reset / firmware
update) — it is just framed as **UDS over ISO-TP** (ISO 14229
+ ISO 15765-2) for vehicles and **CANopen NMT/SDO** (CiA 301)
for industrial automation. Operators in those environments
already have tooling (Vector CANoe, Intrepid Vehicle Spy, jPO,
SocketCAN + `isotp-tools`) that speaks these protocols; if we
invent a custom wire format we strand ourselves.

This change is **research-and-design first**. Before we write
code, we need to:

1. Understand the actual standards well enough to map our
   verbs to them faithfully (not "approximately UDS-like").
2. Pick a Rust-native, `#![no_std]`, clean-room implementation
   strategy for ISO-TP and the UDS subset we need.
3. Settle the **security** story: classic CAN has zero
   authentication or encryption, so we need to choose between
   AUTOSAR SecOC (truncated MAC + freshness counter, fits in
   8-byte frames), CAN XL with TLS (modern but requires CAN XL
   silicon), or "trust the physical bus" (only acceptable for
   isolated diagnostic-only buses, must be explicit).
4. Decide whether SmallAIOS should also speak CANopen for
   industrial deployments, or only UDS for automotive — these
   are different protocol stacks and we may not need both in
   v1.

The deliverable of v1 is therefore **the design and a clean-room
ISO-TP implementation**, plus the UDS subset for `Reset (0x11)`,
`Read Data By Identifier (0x22)`, `Security Access (0x27)`, and
the `Request Download / Transfer Data / Transfer Exit (0x34 /
0x36 / 0x37)` flow that maps onto `remote-update-v1`'s transport
trait. CANopen, AUTOSAR SecOC, and J1939 are explicit non-goals
for v1 and tracked as follow-ups.

## What Changes

### Research deliverables (gating; produced before any code)

- `docs/automotive-bus-research.md` — synthesis of:
  - ISO 15765-2 ISO-TP framing (single-frame, first-frame,
    consecutive-frame, flow-control, padding rules,
    classical-CAN vs CAN-FD differences).
  - ISO 14229 UDS service catalog with the subset we will
    implement and the subset we will explicitly **not**
    implement (and why).
  - Security options: AUTOSAR SecOC (E2E MAC), CAN XL TLS,
    physically-isolated diagnostic bus. Recommended choice
    with a clear tradeoff table.
  - Survey of existing Rust crates: `socketcan`,
    `embedded-can`, `isotp-rs`, `automotive-rs` — which are
    `no_std`, which we can use, which we replace with
    clean-room.
- `docs/automotive-bus-design.md` — concrete Rust trait /
  module layout, mapping of every `management-login-v1` verb
  to a UDS service ID, and the SecOC/CAN-XL/isolated-bus
  decision.

### ISO-TP transport (clean-room)

- New `automotive/` Layer 1 crate (depends only on
  `peripheral` for the CAN HAL).
- `automotive/src/isotp.rs` — `no_std` ISO 15765-2
  implementation:
  - Single-frame (≤7 byte payload on classical CAN).
  - First-frame + consecutive-frame for ≤4095-byte payloads
    (UDS extended addressing pushes this further; v1 caps at
    4 KiB).
  - Flow-control frame with block-size and STmin negotiation.
  - Pad-to-8-bytes option (required by some controllers).
- ~400 lines of clean-room Rust; mirrors the parser style
  used elsewhere in the workspace.

### UDS subset (clean-room)

- `automotive/src/uds.rs` — service handlers:
  - `0x11 ECU Reset` → wraps `system_power(REBOOT)` from
    `system-power-control-v1`.
  - `0x22 Read Data By Identifier` → reads the same telemetry
    Zenoh exposes on `smallaios/metrics/**`. DID assignments
    in a config file (operator-defined, no automotive-OEM
    assumed).
  - `0x27 Security Access` (level 1) → seed/key challenge that
    bridges to the `auth_login` syscall under
    `management-login-v1`. The "key" derivation function uses
    SHA-3 + a per-unit pre-shared secret stored alongside the
    shadow file.
  - `0x34 Request Download` / `0x36 Transfer Data` / `0x37
    Request Transfer Exit` → implements the
    `update::Transport` trait from `remote-update-v1`, so the
    same A/B-slot machinery handles a UDS-driven update.
  - `0x3E Tester Present` → keeps the session alive.

### SecOC / CAN-XL / isolated-bus selection

- Default for v1: **physically-isolated diagnostic bus only**.
  This is the conservative choice — explicit in docs, refuses
  to bind to a bus marked `non-isolated` in config without an
  override flag.
- Optional: a **SecOC-equivalent** layer using our existing
  AES-256-GCM truncated to 32-bit MAC + 16-bit freshness
  counter (compatible with AUTOSAR SecOC framing). Adds 6
  bytes to every UDS request/response payload.
- CAN XL TLS deferred — requires CAN XL silicon we cannot test
  on yet.

### Configuration

- `automotive` config (TOML) defines:
  - CAN interface (loopback, `mcp2515:/dev/spidev0.0`,
    `axi:0xa0010000`, etc. — same selectors the inference
    bridge already uses).
  - Diagnostic CAN ID (request and response).
  - Isolated-bus assertion (`isolated = true|false`).
  - Optional SecOC key (path to a key file).
  - DID table for `0x22 Read Data By Identifier`.

### Out of scope for v1 (flagged)

- **CANopen** (NMT/SDO/PDO/EDS). Industrial-automation
  protocol; substantial second stack. Tracked for v2.
- **J1939** (heavy-duty trucks/agriculture, 29-bit ID, PGN-
  based). Different protocol family, separate change.
- **DoIP** (ISO 13400, UDS over IP). Already covered by the
  Zenoh path on a real IP network.
- **Full UDS** — only the five service IDs above plus
  `0x3E`. Other services (programming sessions beyond
  download, security level 2+, dynamic DID definition,
  routine control, ReadDTC) deferred.
- **Classical-CAN max-payload optimization** — CAN FD support
  is a v2 enhancement once the classical path is solid.
- **Hardware-backed key storage** for SecOC.

## Capabilities

### New Capabilities

- `automotive-iso-tp`: ISO 15765-2 framing rules, FC
  negotiation, padding policy, error-recovery semantics.
- `automotive-uds-subset`: the v1 service subset, DID-table
  config schema, Security Access seed/key derivation, mapping
  to existing SmallAIOS verbs.
- `automotive-secoc-mac`: optional SecOC-equivalent
  authentication layer (AES-256-GCM truncated MAC + freshness
  counter).
- `automotive-bus-isolation-policy`: explicit declaration that
  a SmallAIOS unit refuses non-isolated buses without a SecOC
  key or an override flag, and the boot-time check that
  enforces it.

### Modified Capabilities

- `update-transport-trait` (from `remote-update-v1`): adds
  `UdsIsoTpTransport` as a third implementation.
- `mgmt` (the cross-cutting management surface): documents that
  there are now two equivalent control planes (Zenoh-on-IP and
  UDS-on-ISO-TP) sharing the same verbs and audit log.
- `bus-can` (the existing inference bridge): documents that
  the management ISO-TP listener and the inference bridge can
  coexist on the same physical interface using disjoint CAN-ID
  ranges.

## Impact

- **Code:**
  - New `automotive/` crate (Layer 1): ISO-TP, UDS handler,
    SecOC-equivalent (optional), DID table loader.
  - Reuses the existing CAN HAL in `peripheral/` and the
    `bus/` selectors.
  - `container/src/mgmt_uds.rs` to dispatch UDS requests to
    the same admin core that handles Zenoh requests.
- **Tests:** golden ISO-TP frame vectors (cross-checked
  against `socketcan`'s `isotp-tools` on the developer
  workstation), UDS service handlers, SecOC MAC verification,
  end-to-end "drive an A/B update over loopback CAN" test.
  Aim for ~60 new passing tests.
- **Boot footprint:** ~10 KB code; zero footprint when
  `bus_backend != can`.
- **Downstream:** unblocks SmallAIOS deployments on vehicle /
  industrial CAN segments without an IP gateway, and gives
  operators a path to use existing automotive tooling
  (CANoe, jPO) for management.
- **Dependencies:** `management-login-v1` — provides auth, the
  audit log, **and the management surface convention** (`Config`
  model, `ConfigSurface` trait, `/data/` layout). This change
  adds `automotive/uds.toml` to the `Config` model **and** adds
  a fourth `ConfigSurface` impl (`UdsConfigSurface`) so every
  existing option is automatically reachable over the CAN bus
  by the universal-exposure invariant — no per-feature UDS
  plumbing. `system-power-control-v1` (reboot path) and
  `remote-update-v1` (transport trait + A/B slots) are also
  dependencies.
- **Risks:** (1) Mis-implementing ISO-TP flow control would
  silently corrupt large transfers — the design.md must enumerate
  every state transition and the test plan must replay vectors
  from the standard. (2) Choosing "isolated bus only" by default
  could surprise operators who expected SecOC by default;
  needs a clear log line at boot. (3) SecOC compatibility with
  third-party AUTOSAR stacks is not a v1 goal — we implement
  the algorithm, not the wire compatibility with a specific
  vendor's SecOC profile.

## Open Questions

1. **Do we need CANopen in v1**, or only UDS? UDS covers
   automotive; CANopen covers industrial. They're different
   stacks (different framing, different state machines, both
   ride on CAN). Doing both doubles scope. Leaning UDS-only
   for v1, CANopen as a follow-up.
2. **Should SecOC be on by default when not isolated**, or
   only when explicitly configured? Defaulting to SecOC is
   safer but the operator must provide a key — there is no
   way for two endpoints to negotiate one over a no-IP bus.
3. **DID assignments**: do we ship a default table mapping our
   metrics to a SmallAIOS-specific DID range (and document
   it), or strictly require the operator to define every DID?
   Leaning ship-default-table with reserved-range documented.
4. **Loopback CAN testing in CI** — Linux SocketCAN's `vcan`
   loopback works, but the kernel-mode SmallAIOS path runs in
   QEMU. Do we wire up QEMU's CAN device emulation, or test
   ISO-TP and UDS purely as `#[cfg(test)]` host-mode unit
   tests with a fake CAN HAL? Leaning host-mode for v1; QEMU
   CAN integration as a follow-up.
