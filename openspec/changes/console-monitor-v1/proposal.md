## Why

Once `management-login-v1` lands, an operator who SSHes — sorry,
logs in over the serial console as a `Viewer` — has nothing to
*do*. The role exists to give serving / SRE teams a read-only
window into the running box, but with no first-class TUI the
only path is "scrape the Zenoh metrics keyspace from another
machine," which defeats the purpose of having a console session
in the first place. SmallAIOS needs the same primary tool every
operator already has muscle memory for: a `top`-style live
monitor that shows what the box is doing right now, on the same
TTY they just logged into.

Beyond the operator-UX argument, this is also where the
*viewability* of all the SmallAIOS-specific telemetry actually
gets exercised. We already publish per-model QPS, p50/p99,
GPU utilization, per-NIC counters, per-mount IOPS, and per-
peripheral byte counts via `mgmt-zenoh-telemetry` — but if no
on-box consumer of that data exists, regressions in the
collectors go unnoticed until someone external connects. The
console monitor is the canary: if `top` shows zero GPU
utilization while a model is clearly serving, the collector is
broken.

This change keeps strictly to **consumption**: it does not add
collectors, does not add data sources, and does not bypass the
role gate. It is a TTY rendering of the existing telemetry
stream, gated by `Role::Viewer` or higher.

## What Changes

### Command surface

- `top` — start the live monitor on the current TTY. Available
  to `Role::Viewer`, `Role::Operator`, and `Role::Root`.
- `top --once` — print one snapshot and exit (script-friendly,
  no terminal control sequences).
- `top --interval N` — set refresh interval to N seconds
  (default 1, min 0.5, max 60).
- `top --filter <pattern>` — show only models / interfaces /
  mounts matching the substring.
- Keys while running:
  | Key | Action |
  |-----|--------|
  | `q` / Ctrl-C / Esc | Quit, restore screen |
  | `h` / `?` | Help overlay |
  | `s` | Set refresh interval (prompts) |
  | `f` | Set filter (prompts) |
  | `P` | Sort process / model list by CPU |
  | `M` | Sort by memory (or VRAM for GPU section) |
  | `L` | Sort by p99 latency |
  | `Q` | Sort by QPS |
  | `1` | Toggle per-core CPU expansion |
  | `g` | Toggle GPU section |
  | `n` | Toggle network section |
  | `d` | Toggle filesystem (disk) section |
  | `i` | Toggle peripheral I/O section |
  | `c` | Cycle color schemes (default, mono, high-contrast) |
  | space | Force refresh now |

### Layout

A single full-screen render, redrawn at the configured interval.
Sections collapse gracefully when the terminal is narrow
(monitors over a 80-column serial console must stay readable):

```
SmallAIOS  hostname=orin-01  up 4h 12m  load 0.31 0.28 0.22       [q]uit  [h]elp
─────────────────────────────────────────────────────────────────────────────────
CPU       4× Cortex-A78AE   18%  ▆▃▂▅                            (press 1 to expand per-core)
MEM       2.1 / 8.0 GiB     26%  ▇▇▃░░
GPU       Orin GPU (8.7)    72%  ▇▇▇▇▇▇░░     VRAM 1.2 / 8.0 GiB  Streams 4  Graphs 12
NET   eth0   ↓ 4.2 MB/s  ↑ 1.1 MB/s   bond0   ↓ 0 B/s   ↑ 0 B/s   errs 0/0
FS    /data  18 / 64 GiB    used     r 12 IOPS / w 4 IOPS         lat 1.2 ms
I/O   I2C0  142 KB/s   SPI1  0 B/s   GPIO0  4 events/s
─────────────────────────────────────────────────────────────────────────────────
MODELS                            QPS    p50    p99   BATCH    LAST ERR
resnet50_v2.onnx                   42   33ms   58ms   16/16    —
mobilenet_v2.onnx                  91   12ms   24ms    4/4     —
squeezenet_v1.onnx                  6   18ms   31ms    1/4     —
─────────────────────────────────────────────────────────────────────────────────
SESSIONS  root@tty0 (1m)   sre1@zenoh (12m)
```

### Data sources (all already published; this change consumes,
does not collect)

- **CPU**: per-core utilization, load average. From the
  scheduler's existing run-queue stats; topology from `kernel`.
- **Memory**: page-allocator counts (`kernel`).
- **GPU**: `arch/nvidia` already exposes utilization via the
  CUDA driver profiling counters when `gpu-profile` is on, and
  the `gpu-resident-vision-hybrid-v1` graph cache exposes hit
  rate and capture count. Future AMD / Intel GPU crates expose
  the same trait. The monitor shows whatever is implemented;
  unimplemented backends render as `n/a`.
- **Network**: per-interface RX/TX byte/packet/error counters
  from `net` (existing).
- **Filesystem**: per-mount IOPS / bytes / latency from `peripheral`
  block-device drivers and the future block layer (today the
  numbers may be zeroed on platforms without persistent
  storage; render `n/a` cleanly).
- **Peripheral I/O**: per-bus byte / event counters from the
  existing `peripheral::{i2c, spi, gpio, uart, camera_csi,
  audio_i2s}` modules. Only buses compiled in (per the feature
  flags) appear.
- **Models**: per-model QPS / p50 / p99 / batch fill rate / error
  count from the existing `mgmt-zenoh-telemetry` publishers in
  `container/`.
- **Sessions**: live session table from `auth/` (read-only
  view).

The monitor reads these via the same `mgmt::Config` /
telemetry channel a Zenoh subscriber would — no new kernel-
internal interface, no privilege escalation. If a metric is
not exposed (e.g. AMD GPU utilization in v1), the monitor
displays `n/a` rather than failing.

### Implementation: VT100 mini-emulator

- Clean-room `no_std` Rust module `console_monitor::vt100`
  emitting only the subset we need:
  - Cursor positioning (CSI `H`, CSI `;H`).
  - Clear screen / line (CSI `J`, CSI `K`).
  - SGR colors (CSI `Nm`) — 16-color and the optional
    `[c]ycle` mono / high-contrast modes.
  - Alternate screen buffer enter / leave (CSI `?1049h` /
    `?1049l`) — preserves the operator's pre-`top` shell
    history when they quit.
  - Cursor hide / show (CSI `?25l` / `?25h`).
- ~150 LOC; no third-party crate (`crossterm`, `tui-rs`,
  `ratatui` all pull in a std-only dep tree).
- Renderer is double-buffered: build the next frame as a string,
  diff against the previous, emit only the changed cells. This
  matters on a 115200-baud serial console where redrawing the
  whole 80×24 grid every second wastes ~3 KB/s.

### Resource budget

- ~6 KB live memory for the frame buffers + sort-key tables.
- Refresh latency target: <50 ms wall-clock from "tick" to
  "rendered" on Orin; <200 ms on x86-64 baseline.
- CPU: <1% of one core at 1 Hz refresh on Orin.
- The monitor reuses the active session's idle-timeout slot —
  any keypress (including the configured refresh tick is
  *not* a keypress) resets the timer (closes open question 6
  on `management-login-v1`).

### Out of scope for v1 (flagged)

- **Process-level inspection** (Linux `top`'s per-PID list).
  SmallAIOS is a unikernel — there are no processes in the
  Linux sense. The "models" section is the closest analogue.
- **Writable actions** (`top`'s `k` to kill, `r` to renice).
  `Viewer` is read-only by definition; `Operator` and `Root`
  use separate commands (`model unload <name>`, `system reboot`)
  outside the monitor.
- **Mouse support**. Serial consoles rarely have it; over
  Zenoh the operator runs a real terminal. v2 maybe.
- **Log-tail pane** (`journalctl -f` overlay). Useful but big;
  separate change (`console-log-tail-v1`).
- **GPU-vendor-specific deep dives** (per-SM utilization,
  per-tensor-core occupancy). The summary `%util / VRAM /
  graphs / streams` line is enough for v1; per-vendor pages
  follow the same pattern as Linux `nvtop` — a follow-on.
- **Recording / replay** of a monitor session.
- **Color customization beyond three built-in schemes**.

## Capabilities

### New Capabilities

- `console-monitor-tui`: the `top` command, full keybinding
  catalog, layout / collapse rules for narrow terminals, the
  collapse-on-`n/a` behavior for missing metric sources.
- `console-monitor-vt100-emitter`: the clean-room VT100 subset,
  alternate-screen-buffer guarantee, double-buffered renderer
  rules, and the "no third-party TUI crate" boundary.
- `console-monitor-data-bindings`: the precise mapping from each
  on-screen field to its publishing source in the existing
  telemetry pipeline (so the monitor breaks loudly when a
  source is removed or renamed).

### Modified Capabilities

- `console-login` (from `management-login-v1`): registers `top`
  as a built-in command available to every role (with the
  read-only / writable distinction enforced inside, not at the
  command boundary — every role can *run* `top`, only the
  rendering and the keybind catalog differ).
- `mgmt-zenoh-telemetry` (from `management-login-v1`): the
  metrics keyspace is the canonical schema; this change does
  not modify it but adds a documented "consumed-by"
  relationship (regressions in published fields break the
  monitor's CI test).
- `auth-roles` (from `management-login-v1`): documents that
  `Role::Viewer` is the *primary* user of this tool and that
  `top`'s observed surface is the v1 yardstick for what
  "read-only" means.

## Impact

- **Code:**
  - New crate `console_monitor/` (Layer 2): VT100 emitter,
    layout engine, data-source bindings, keybind dispatcher.
  - `peripheral/src/uart.rs`: gains a raw-input mode (already
    needed by `remote-update-v1`'s YMODEM receiver — confirms
    the v1 design).
  - `container/src/bin/top.rs` — the user-space command (thin
    wrapper that opens the telemetry channel + spawns the
    renderer).
- **Tests:** ~40 new tests targeted: VT100 emitter golden
  vectors (cursor moves, colors, alt-screen toggles), layout
  collapse-at-narrow-width, sort-key correctness for each
  `M`/`P`/`L`/`Q` mode, `n/a` rendering when a source is
  missing, frame-diff renderer minimal-bytes test, role gate
  enforcement (a `Viewer` who somehow sends a `model_unload`
  keystroke gets `EPERM`), idle-timer reset on keypress.
  Aim 4,143 → ≥4,260 once `management-login-v1` lands first.
- **Boot footprint:** ~12 KB (VT100 + layout + data bindings).
- **Container image:** unchanged.
- **Downstream:** unblocks the `Viewer` role from
  `management-login-v1` actually being useful; gives
  developers a one-key sanity check ("is the GPU doing
  anything?") without external tooling; doubles as a
  collector-regression alarm.
- **Dependencies:** `management-login-v1` — provides the
  `Viewer` role, the telemetry pipeline this monitor
  consumes, the TTY shell that hosts the `top` command, the
  audit log for the session-ran-monitor record, and the
  management surface convention. This change adds **no**
  fields to `Config` (purely a viewer of existing telemetry).
- **Risks:**
  (1) On a slow serial console (115200 baud, ~11 KB/s
  effective) a full redraw every second uses ~30 % of
  bandwidth. The double-buffered diff renderer is non-
  optional, not a nice-to-have. (2) Telemetry source
  rot — if a future change renames a published key, the
  monitor silently shows zero. The `console-monitor-data-
  bindings` capability mandates a CI test that asserts every
  bound source is still published. (3) GPU vendor coverage
  is uneven (NVIDIA real, AMD/Intel stubs) — the v1 monitor
  must render `n/a` cleanly and document the gap, not
  pretend zeros.

## Open Questions

1. **Default refresh interval**: 1 s matches Linux `top`,
   but on a 115200-baud serial console even the diff
   renderer noticeably flickers. Options: (a) 1 s default
   on Zenoh-attached terminals, 2 s on serial; (b) 1 s
   universal with a clear "set with `s` if it's choppy"
   message. Leaning (a) — auto-detect transport.
2. **Where does the monitor render `n/a`?** A dash, the
   literal string `n/a`, or the column collapses entirely?
   Linux convention is `--`; we lean `n/a` for clarity but
   width-aware collapse for the GPU section if no GPU is
   detected at boot.
3. **Should `top` be the only command name, or also alias
   `htop` / `monitor`?** Real estate is cheap; aliases are
   friendly.
4. **Idle-timer behavior**: does any keypress reset the
   timer (yes, almost certainly), or only "non-trivial"
   keys (excluding the auto-refresh tick which is not a
   keypress anyway)? Documented in the spec; expected
   answer is "any keypress."
5. **Per-session vs global filter**: when an operator sets
   `f resnet*`, does that filter persist after they `q`uit
   and re-launch `top` in the same session? Linux `top`
   forgets; `htop` remembers. Leaning forget for v1.
6. **Sort default**: Linux `top` defaults to %CPU. For
   SmallAIOS the analogue is "highest QPS model on top." Is
   that the right default, or should it be alphabetical so
   names don't shuffle while you read? Leaning QPS-desc
   like Linux.
