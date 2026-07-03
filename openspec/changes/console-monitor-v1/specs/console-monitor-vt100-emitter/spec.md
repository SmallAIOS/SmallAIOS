## ADDED Requirements

### Requirement: Clean-Room VT100 Subset Emitter

The `console_monitor` crate SHALL provide a clean-room `#![no_std]` module `console_monitor::vt100` that emits only the following VT100/ANSI subset: cursor positioning (CSI `H`, CSI `<row>;<col>H`), clear screen and clear line (CSI `J`, CSI `K`), SGR colors (CSI `<N>m`, 16-color, covering the default, mono, and high-contrast schemes), alternate screen buffer enter/leave (CSI `?1049h` / CSI `?1049l`), and cursor hide/show (CSI `?25l` / CSI `?25h`). The module SHALL NOT emit escape sequences outside this subset.

#### Scenario: Golden vectors for cursor moves

- **WHEN** the unit tests drive `console_monitor::vt100` cursor positioning for a set of row/column pairs
- **THEN** the emitted bytes SHALL match the golden vectors for CSI `<row>;<col>H` exactly

#### Scenario: Golden vectors for colors and alt-screen toggles

- **WHEN** the unit tests drive SGR color emission and alternate-screen enter/leave
- **THEN** the emitted bytes SHALL match the golden vectors for CSI `<N>m`, CSI `?1049h`, and CSI `?1049l` exactly

#### Scenario: No sequences outside the subset

- **WHEN** a full monitor session's output is captured in test
- **THEN** every escape sequence present SHALL belong to the specified subset

### Requirement: No Third-Party TUI Crate

The console monitor SHALL implement its terminal handling in-tree. It SHALL NOT depend on `crossterm`, `tui-rs`, `ratatui`, or any other third-party terminal/TUI crate, because those pull in std-only dependency trees incompatible with the `#![no_std]` workspace.

#### Scenario: Dependency tree stays clean

- **WHEN** `cargo tree` is run for the `console_monitor` crate
- **THEN** no third-party terminal or TUI crate SHALL appear in the dependency graph

### Requirement: Alternate Screen Buffer Guarantee

Starting the interactive monitor SHALL enter the alternate screen buffer (CSI `?1049h`) and hide the cursor (CSI `?25l`). Quitting by any path — `q`, Ctrl-C, or Esc — SHALL leave the alternate screen buffer (CSI `?1049l`) and show the cursor (CSI `?25h`), preserving the operator's pre-`top` shell history.

#### Scenario: Shell history preserved across a monitor session

- **WHEN** an operator with visible shell scrollback starts `top` and later presses `q`
- **THEN** the TTY SHALL return to the pre-`top` screen contents
- **AND** the cursor SHALL be visible again

#### Scenario: Ctrl-C also restores the screen

- **WHEN** the operator quits the monitor with Ctrl-C instead of `q`
- **THEN** the alternate screen buffer SHALL be left and the cursor shown, identically to a `q` quit

### Requirement: Double-Buffered Diff Renderer

The renderer SHALL be double-buffered: it SHALL build the next frame in memory, diff it against the previous frame, and emit only the changed cells. Full-frame redraws every tick SHALL NOT occur in steady state, so that a 115200-baud serial console is not saturated by refresh traffic.

#### Scenario: Unchanged frame emits no drawing bytes

- **WHEN** two consecutive refresh ticks produce identical frame content
- **THEN** the second tick SHALL emit no cell-drawing output

#### Scenario: Single-cell change emits minimal bytes

- **WHEN** exactly one cell's content changes between two frames
- **THEN** the emitted output SHALL cover only the changed cell (cursor positioning plus the new cell content)
- **AND** the frame-diff minimal-bytes test SHALL assert the emitted byte count is far below a full 80×24 redraw
