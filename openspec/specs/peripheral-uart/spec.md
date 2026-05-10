# peripheral-uart Specification

## Purpose
TBD - created by archiving change management-login-v1. Update Purpose after archive.
## Requirements
### Requirement: Per-read echo and raw mode options
The UART driver SHALL accept per-read options on `read_line` (or equivalent) that select echo behavior and raw mode. Defaults SHALL be `echo=true, raw=false` (the existing visible-character line-buffered behavior). When `echo=false`, no character SHALL be echoed back to the terminal — not even a placeholder character — to avoid leaking password length. When `raw=true`, control characters SHALL be delivered to the caller verbatim instead of being interpreted by the line discipline. The driver SHALL NOT hold global mode state; every read SHALL declare its own options.

```rust
pub struct ReadOptions {
    pub echo: bool,
    pub raw: bool,
    pub max_len: usize,
}

pub fn read_line(buf: &mut [u8], opts: ReadOptions) -> Result<usize, Error>;
```

#### Scenario: Password prompt suppresses echo
- **WHEN** the console-login code calls `read_line` with `echo=false`
- **THEN** no characters SHALL appear on the TTY as the operator types

#### Scenario: Backspace honored in echo-off password mode
- **WHEN** the operator types `a`, `b`, then backspace during an echo-off read
- **THEN** the buffered result SHALL contain only `a`
- **AND** no character SHALL be echoed for any keypress including the backspace

#### Scenario: Default read still echoes
- **WHEN** code calls `read_line` with default options
- **THEN** typed characters SHALL be echoed as before
- **AND** behavior SHALL be byte-identical to the previous API for any caller that does not use the new options

#### Scenario: Stateless across reads
- **WHEN** an echo-off read is followed by a default read
- **THEN** the default read SHALL behave with `echo=true` regardless of any prior call's options

