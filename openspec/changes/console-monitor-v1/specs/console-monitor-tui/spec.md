## ADDED Requirements

### Requirement: `top` Command Availability by Role

The console shell SHALL provide a `top` command that starts a live, full-screen monitor on the current TTY. The command SHALL be available to `Role::Viewer`, `Role::Operator`, and `Role::Root`. The monitor SHALL be strictly a consumer of existing telemetry: it SHALL NOT add collectors, SHALL NOT add data sources, and SHALL NOT bypass the role gate.

#### Scenario: Viewer starts the live monitor

- **WHEN** an authenticated `Role::Viewer` session enters `top` on the serial console
- **THEN** the monitor SHALL start rendering on that TTY
- **AND** no privilege beyond `Role::Viewer` SHALL be required

#### Scenario: Operator and Root run the same monitor

- **WHEN** an authenticated `Role::Operator` or `Role::Root` session enters `top`
- **THEN** the monitor SHALL start with the same observation surface as for `Role::Viewer`
- **AND** the role distinction SHALL be enforced inside the monitor, not at the command boundary

### Requirement: Snapshot Mode (`--once`)

`top --once` SHALL print exactly one snapshot of the monitor content to the TTY and exit. Snapshot output SHALL contain no terminal control sequences so it is script-friendly.

#### Scenario: One snapshot then exit

- **WHEN** `top --once` is executed
- **THEN** exactly one rendered snapshot SHALL be written
- **AND** the command SHALL exit without entering the interactive refresh loop

#### Scenario: Snapshot output carries no control sequences

- **WHEN** the output of `top --once` is captured to a buffer
- **THEN** the buffer SHALL contain no ESC (0x1B) bytes and no CSI sequences
- **AND** the buffer SHALL be plain text suitable for scripting

### Requirement: Refresh Interval Control

The monitor SHALL refresh at a configurable interval. The default SHALL be 1 second; the effective interval SHALL be constrained to the inclusive range 0.5–60 seconds. The interval SHALL be settable at launch via `top --interval N` and interactively via the `s` key, which SHALL prompt for a new value. Pressing space SHALL force an immediate refresh without changing the configured interval.

#### Scenario: Default interval is 1 second

- **WHEN** `top` is started with no `--interval` argument
- **THEN** the monitor SHALL redraw once per second

#### Scenario: Interval set at launch

- **WHEN** `top --interval 5` is executed
- **THEN** the monitor SHALL redraw once every 5 seconds

#### Scenario: Interval bounds enforced

- **WHEN** an interval below 0.5 or above 60 seconds is requested via `--interval` or the `s` prompt
- **THEN** the effective refresh interval SHALL remain within the 0.5–60 second range

#### Scenario: Space forces immediate refresh

- **WHEN** the operator presses space between refresh ticks
- **THEN** the monitor SHALL redraw immediately
- **AND** the configured interval SHALL remain unchanged

### Requirement: Filter Control

The monitor SHALL support a substring filter over models, network interfaces, and mounts. The filter SHALL be settable at launch via `top --filter <pattern>` and interactively via the `f` key, which SHALL prompt for a pattern.

#### Scenario: Launch-time filter narrows the model list

- **WHEN** `top --filter resnet` is executed while `resnet50_v2.onnx` and `mobilenet_v2.onnx` are loaded
- **THEN** the MODELS section SHALL list only `resnet50_v2.onnx`

#### Scenario: Interactive filter via `f`

- **WHEN** the operator presses `f` and enters `eth` at the prompt
- **THEN** the NET section SHALL show only interfaces whose name contains `eth`

### Requirement: Keybinding Catalog

While the monitor is running, it SHALL dispatch the following keys: `q`, Ctrl-C, and Esc SHALL quit and restore the screen; `h` and `?` SHALL show a help overlay; `s` SHALL prompt for the refresh interval; `f` SHALL prompt for the filter; `P` SHALL sort the model list by CPU; `M` SHALL sort by memory (VRAM for the GPU section); `L` SHALL sort by p99 latency; `Q` SHALL sort by QPS; `1` SHALL toggle per-core CPU expansion; `g` SHALL toggle the GPU section; `n` SHALL toggle the network section; `d` SHALL toggle the filesystem section; `i` SHALL toggle the peripheral I/O section; `c` SHALL cycle color schemes (default, mono, high-contrast); space SHALL force an immediate refresh.

#### Scenario: Sort by QPS

- **WHEN** the operator presses `Q` while three models are serving at different QPS
- **THEN** the MODELS section SHALL be reordered by the QPS sort key

#### Scenario: Sort by p99 latency

- **WHEN** the operator presses `L`
- **THEN** the MODELS section SHALL be reordered by the p99 latency sort key

#### Scenario: Toggle a section off and back on

- **WHEN** the operator presses `g` once and then `g` again
- **THEN** the GPU section SHALL disappear from the layout after the first press
- **AND** SHALL reappear after the second press

#### Scenario: Cycle the three color schemes

- **WHEN** the operator presses `c` three times from the default scheme
- **THEN** the monitor SHALL render mono, then high-contrast, then return to the default scheme

#### Scenario: Help overlay

- **WHEN** the operator presses `h` or `?`
- **THEN** a help overlay listing the keybinding catalog SHALL be drawn

#### Scenario: Quit keys restore the screen

- **WHEN** the operator presses any of `q`, Ctrl-C, or Esc
- **THEN** the monitor SHALL exit
- **AND** the screen SHALL be restored per the alternate-screen-buffer guarantee in `console-monitor-vt100-emitter`

### Requirement: Full-Screen Layout with Narrow-Terminal Collapse

The monitor SHALL render a single full-screen frame, redrawn at the configured interval, comprising: a header line (hostname, uptime, load average, quit/help hints), CPU, MEM, GPU, NET, FS, and peripheral I/O summary lines, a MODELS table (QPS, p50, p99, batch fill, last error per model), and a SESSIONS line. Sections SHALL collapse gracefully when the terminal is narrow: the layout SHALL remain readable on an 80-column serial console.

#### Scenario: 80-column serial console stays readable

- **WHEN** the monitor renders on an 80-column terminal
- **THEN** no rendered line SHALL exceed 80 columns
- **AND** every enabled section SHALL remain identifiable (collapsed where necessary)

#### Scenario: Per-core CPU expansion

- **WHEN** the operator presses `1` on a 4-core host
- **THEN** the CPU section SHALL expand to show one utilization entry per core
- **AND** pressing `1` again SHALL collapse it back to the summary line

#### Scenario: MODELS table carries the per-model columns

- **WHEN** a model is serving traffic
- **THEN** its MODELS row SHALL show QPS, p50, p99, batch fill rate, and last error

### Requirement: Missing-Metric `n/a` Rendering

When a bound metric source is not exposed on the running platform, the monitor SHALL render `n/a` for the affected fields rather than failing or rendering fabricated zeros. GPU backends that do not implement the utilization surface (AMD and Intel stubs in v1) SHALL render as `n/a`. Filesystem metrics on platforms without persistent storage SHALL render `n/a` cleanly. Only peripheral buses compiled in via their feature flags SHALL appear in the peripheral I/O section.

#### Scenario: Stub GPU backend renders n/a

- **WHEN** the monitor runs on a platform whose GPU crate is an architectural stub with no utilization counters
- **THEN** the GPU section SHALL render `n/a` for utilization and VRAM
- **AND** the monitor SHALL continue running without error

#### Scenario: No persistent storage renders n/a

- **WHEN** the monitor runs on a platform with no block devices
- **THEN** the FS section SHALL render `n/a` rather than zeroed IOPS figures

#### Scenario: Feature-gated buses are omitted

- **WHEN** the `peripheral` crate is built without the `spi` feature
- **THEN** no SPI entry SHALL appear in the peripheral I/O section

### Requirement: Read-Only Enforcement Inside the Monitor

The monitor SHALL be strictly read-only: it SHALL NOT issue any state-mutating operation. Writable actions (model unload, reboot, and similar) SHALL NOT be reachable from the monitor's keybinding catalog. If a mutating request nonetheless reaches the kernel from a `Role::Viewer` monitor session, the existing role gate SHALL reject it.

#### Scenario: Viewer mutation attempt is rejected

- **WHEN** a `Role::Viewer` session running the monitor somehow submits a `model_unload` request
- **THEN** the kernel SHALL return `-EPERM`
- **AND** no model SHALL be unloaded

#### Scenario: No writable keybindings exist

- **WHEN** a reviewer inspects the keybind dispatcher
- **THEN** no key SHALL map to a state-mutating operation for any role

### Requirement: Idle-Timer Interaction

Any keypress received while the monitor is running SHALL reset the active session's idle-logout timer. The automatic refresh tick SHALL NOT count as a keypress and SHALL NOT reset the timer.

#### Scenario: Keypress resets the idle timer

- **WHEN** a Viewer session running the monitor presses any key at minute 59 of a 60-minute idle window
- **THEN** the idle timer SHALL reset to zero

#### Scenario: Refresh tick does not reset the timer

- **WHEN** the monitor auto-refreshes every second with no operator keypress
- **THEN** the session idle timer SHALL continue to advance
- **AND** the session SHALL auto-logout at the configured idle window despite the ongoing refreshes

### Requirement: Resource Budget

The monitor SHALL stay within its resource budget: approximately 6 KB of live memory for frame buffers and sort-key tables; refresh latency under 50 ms wall-clock from tick to rendered frame on Jetson Orin and under 200 ms on the x86-64 baseline; CPU consumption under 1% of one core at a 1 Hz refresh on Orin.

#### Scenario: Refresh latency within target

- **WHEN** the benchmark harness measures tick-to-rendered latency at a 1 Hz refresh
- **THEN** the measured latency SHALL be under 50 ms on Jetson Orin
- **AND** under 200 ms on the x86-64 baseline

#### Scenario: CPU budget at 1 Hz

- **WHEN** the monitor runs at a 1 Hz refresh on Orin with all sections enabled
- **THEN** the monitor SHALL consume less than 1% of one core
