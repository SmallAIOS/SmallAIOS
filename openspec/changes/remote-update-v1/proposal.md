## Why

Today, updating SmallAIOS on a deployed box means **physically
removing the storage medium** — pull the USB drive, flash a new
image on a workstation, plug it back in, power-cycle. That is
fine for a developer workstation but unworkable for any deployed
appliance: a Jetson Orin in a rack, a serial-only industrial
controller, a vehicle ECU, or a remote sensor node.

We need an in-place update path that:

1. **Cannot brick the box** — a bad image must auto-revert.
2. **Cannot run an unsigned image** — the existing
   `verified-boot` ML-DSA-65 chain must extend to in-field
   updates.
3. **Works over multiple transports** — Zenoh-over-IP for
   networked deployments, **YMODEM-1K over the serial console**
   for boxes reachable only by a USB-TTL cable, and (deferred to
   `automotive-bus-management-v1`) UDS-over-ISO-TP for CAN-bus
   environments.
4. Is **size-honest**: an 8 MB unikernel does not need a Linux-
   class update framework (Mender / RAUC / SWUpdate). Two slots
   + a boot pointer + a watchdog is enough.

The standard answer for embedded/automotive/Chrome-OS-style
in-field updates is **A/B slots with verified boot and watchdog
rollback**. We already have ML-DSA-65 signing and the
`verified-boot` feature flag — this change wires them into the
update pipeline.

## What Changes

### A/B boot slots

- Two on-disk image slots, `slot_a` and `slot_b`, each sized
  for the unikernel (~16 MB headroom).
- A small **boot-pointer record** (separate from the slots)
  records:
  - `active: A | B`
  - `pending: Option<A | B>` — set during update, cleared on
    successful boot
  - `tries_remaining: u8` — decremented on each boot attempt;
    zero ⇒ boot loader falls back to `active`
  - `manifest_hash: [u8; 32]` of the active image
- The boot loader (UEFI app on Orin / x86-EFI, board-specific on
  bare-metal) reads this record before each boot.

### Image format

- Self-describing manifest:

  ```text
  smallaios-img v1
  ---
  arch:        aarch64-unknown-none | x86_64-unknown-none | ...
  version:     0.2.3
  build_date:  2026-05-05T14:00:00Z
  payload_len: <bytes>
  payload_hash: <32B SHA3-256>
  signature:   <ML-DSA-65 over manifest+payload, base64>
  ---
  <payload bytes>
  ```
- Manifest is human-readable (debuggable on a flaky link); the
  payload is opaque.

### Watchdog-backed rollback

- After flashing the inactive slot, the boot pointer is set to
  `active=<new>, pending=<new>, tries_remaining=3`.
- On boot, the new image **must** call
  `system_update_confirm()` within 60 s (or successfully respond
  to a `smallaios/admin/system/healthy` ping over Zenoh) — that
  call clears `pending` and resets the counter.
- If it doesn't, the platform watchdog fires, the boot loader
  decrements `tries_remaining`, and after exhaustion reverts to
  the prior slot. **Net result:** a bad image auto-reverts after
  ~3 minutes of nobody-home.

### Transport: TTY / YMODEM-1K

- After login as root over the serial console, the operator runs
  `update`. The kernel enters **YMODEM-1K receive mode** on the
  same TTY:
  - 1024-byte data blocks, CRC-16, `<NAK>`/`<C>`/`<ACK>`/`<EOT>`
    state machine.
  - Clean-room `no_std` Rust (~200 lines) — no third-party
    `xmodem` / `ymodem` crates.
  - Operator triggers the upload from any standard terminal:
    `minicom` `Ctrl-A S → ymodem`, `picocom -t '!! sx -k'`,
    `tio --send`, etc.
- At ~921600 baud an 8 MB image lands in ~90 seconds; at the
  conservative 115200 baud, ~12 minutes. Both acceptable.
- After `<EOT>`, the receiver hands the bytes off to the manifest
  parser → signature verifier → slot writer.
- Failure modes (CRC mismatch, signature fail, wrong arch,
  insufficient slot space) abort cleanly without touching the
  boot pointer.

### Transport: Zenoh chunked upload

- New keyspace under the `management-login-v1` admin tree:
  - `smallaios/admin/update/begin` — request: image manifest
    metadata, response: opaque session id + chunk size.
  - `smallaios/admin/update/chunk/<session>/<index>` — payload
    bytes. CRC-32 per chunk for early-fail.
  - `smallaios/admin/update/commit/<session>` — finalizes,
    triggers signature verify + slot write + boot pointer
    update.
  - `smallaios/admin/update/abort/<session>` — drops the staged
    bytes.
- Re-uses the existing PQC-backed Zenoh transport. Chunked
  rather than monolithic so the 8 MB image fits over an MTU
  that may be ~1 KB on lossy links, and so progress is
  observable in `smallaios/metrics/update`.

### Transport: UDS-over-ISO-TP-over-CAN

- **Deferred to `automotive-bus-management-v1`** — same
  manifest, same signature check, same boot pointer; only the
  wire framing differs. The transport-plugin trait below makes
  it pluggable.

### Transport plugin trait

- New `update::Transport` trait:
  - `fn begin(manifest: &Manifest) -> Result<SessionId>`
  - `fn recv_chunk(session: SessionId, index: u32) ->
    Result<&[u8]>`
  - `fn commit(session: SessionId) -> Result<()>`
  - `fn abort(session: SessionId)`
- Implementations: `TtyYmodemTransport`, `ZenohChunkedTransport`,
  `UdsIsoTpTransport` (later).

### Health-check / commit endpoint

- `system_update_confirm()` — syscall (#47). Called by the
  user-space process responsible for "did the new image
  actually come up correctly," typically the inference server
  after one successful inference round-trip.
- `smallaios/admin/system/healthy` — Zenoh equivalent for
  remote operators to mark the boot good.

### Out of scope for v1 (flagged)

- Differential / delta updates (BSDIFF, courgette). 8 MB full
  image is small enough that delta is not worth the complexity.
- Multi-image bundles (kernel + model files in one update).
  Models update via the existing `SMALLAIOS_MODEL_DIR` flow.
- Update over USB mass-storage (drag-drop a file onto a
  partition). Possible but a different UX from "stream over a
  channel."
- Self-updating bootloader. The boot loader stays as a separate
  artifact updated only via physical reflash for v1.
- `automotive-bus-management-v1` provides the third transport
  (UDS-over-ISO-TP); not part of this change.

## Capabilities

### New Capabilities

- `update-image-format`: manifest schema, payload-hash rules,
  signature algorithm (ML-DSA-65 over manifest+payload).
- `update-boot-slots`: A/B slot layout, boot-pointer record,
  atomic-switch rules, `tries_remaining` semantics.
- `update-watchdog-rollback`: confirm window, watchdog wiring
  per arch, rollback decision tree.
- `update-tty-ymodem`: YMODEM-1K state machine, CRC-16, block
  numbering, `<C>`-mode handshake, error-recovery rules.
- `update-zenoh-chunked`: keyspace, session lifecycle,
  per-chunk CRC-32, abort semantics.
- `update-transport-trait`: the `Transport` plugin contract
  every wire format must satisfy.

### Modified Capabilities

- `kernel-syscalls`: adds `system_update_confirm` (#47).
- `mgmt-zenoh-admin`: adds the `smallaios/admin/update/**`
  keyspace.
- `verified-boot`: extends the existing boot-time signature
  check to apply at update-commit time too (same key, same
  algorithm).
- `peripheral-uart`: adds raw binary I/O mode (no terminal
  cooking) needed by the YMODEM receiver.
- `console-login` (from `management-login-v1`): adds the
  `update` command.

## Impact

- **Code:**
  - New crate `update/` (Layer 1): manifest parser, slot writer,
    transport trait, watchdog wiring.
  - `update/src/ymodem.rs` — clean-room YMODEM-1K, ~200 lines.
  - `container/src/mgmt_update.rs` — Zenoh handler.
  - Boot loader changes (UEFI app on Orin / x86, board-specific
    elsewhere) to read the boot-pointer record.
  - `peripheral/src/uart.rs` — raw mode.
- **Tests:** ~80 new tests targeted: YMODEM golden vectors,
  manifest round-trip, signature verification, slot-write
  atomicity, boot-pointer state machine, rollback-after-N-fails,
  Zenoh chunk reassembly, transport-trait conformance per
  implementation. QEMU integration test that flashes a new
  image and observes the second boot.
- **Disk footprint:** doubles the on-disk image storage
  (~16 MB → ~32 MB). Acceptable on every target.
- **Boot loader:** the EFI app gains ~1 KB to read the boot
  pointer.
- **Container image:** unchanged — containers are updated
  through the registry, not this path.
- **Downstream:** unblocks running SmallAIOS as a
  field-deployable appliance. Required by serving teams who
  cannot send someone with a USB stick to every box.
- **Dependencies:** `management-login-v1` (auth gate),
  `system-power-control-v1` (the post-update reboot uses
  `system_power`).
- **Risks:** (1) Boot-pointer corruption is fatal — must be
  CRC-protected and written with a journaled-replace scheme.
  (2) Watchdog tuning per platform: too short and false
  rollbacks; too long and bricked-update window grows. v1
  targets 60 s confirm window across all platforms.
  (3) YMODEM over a physically-noisy serial cable still
  retransmits; the protocol is robust but slow on bad lines.

## Open Questions

1. Where does the boot-pointer record live? Options: (a) raw
   sectors at a known offset (simplest, breaks if disk geometry
   changes), (b) a small dedicated FAT partition (Orin EFI
   partition is convenient), (c) UEFI variable (atomic, durable,
   but bare-metal-only). Leaning per-platform: UEFI variable on
   Orin/x86-EFI, dedicated raw region on bare-metal.
2. Should `update` over TTY *also* require the bearer-token from
   `management-login-v1`, or is "physical access to the serial
   port" considered sufficient authorization on its own? Leaning
   require login; physical-presence-only is cheaper but
   inconsistent with the Zenoh path.
3. Should we keep the previous slot's image after a successful
   boot, or reuse the space immediately? Leaning keep — three
   slot states (active / standby / staging) gives one-shot
   manual rollback for free if the operator notices a problem
   in the first hour.
