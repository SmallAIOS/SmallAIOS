## ADDED Requirements

### Requirement: DPAUX DDC initialization

The EDID driver SHALL enable the DPAUX1 clock, configure the DPAUX1 hybrid pad for I2C/DDC mode, and prepare the AUX controller for DDC transactions to the HDMI sink.

#### Scenario: DPAUX1 clock and pad enable

WHEN `dpaux_init()` is called
THEN DPAUX1 clock SHALL be enabled via CAR, DPAUX1 reset deasserted, and `DPAUX_HYBRID_PADCTL` configured for DDC (I2C) mode.

### Requirement: EDID block read via DDC

The driver SHALL read the 128-byte base EDID block from DDC address `0x50` (slave address on the I2C bus). The read SHALL use DPAUX1 AUX transactions in I2C mode.

#### Scenario: Successful 128-byte EDID read

WHEN a monitor is connected and responds to DDC
THEN `read_edid()` SHALL return a 128-byte array containing the EDID data block, validated by the magic header `[0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x00]`.

#### Scenario: DDC timeout

WHEN no monitor is connected or DDC lines are unresponsive
THEN `read_edid()` SHALL return an error after a bounded timeout (no more than 100 ms total). The function SHALL NOT hang.

#### Scenario: Invalid EDID magic

WHEN the DDC read completes but bytes 0-7 do not match the EDID magic header
THEN `read_edid()` SHALL return an error indicating corrupt EDID data.

### Requirement: EDID preferred timing extraction

The driver SHALL parse the first detailed timing descriptor (EDID bytes 54-71) to extract: pixel clock, horizontal active/blanking/sync, and vertical active/blanking/sync.

#### Scenario: Parse 1080p preferred timing

WHEN the EDID preferred timing descriptor contains 1920x1080 at 148.5 MHz
THEN `parse_preferred_timing(&edid)` SHALL return a `VideoMode` with width=1920, height=1080, pixel_clock_khz=148500, and correct H/V sync parameters.

#### Scenario: Parse 720p preferred timing

WHEN the EDID preferred timing descriptor contains 1280x720 at 74.25 MHz
THEN `parse_preferred_timing(&edid)` SHALL return a `VideoMode` with width=1280, height=720, pixel_clock_khz=74250.

### Requirement: VideoMode struct

The driver SHALL define a `VideoMode` struct containing: width, height, pixel_clock_khz, h_front_porch, h_sync_width, h_back_porch, v_front_porch, v_sync_width, v_back_porch.

#### Scenario: Default 1080p mode

WHEN `VideoMode::default_1080p()` is called
THEN it SHALL return the standard CEA-861 1080p60 timing: 1920x1080, 148500 kHz, H(88/44/148), V(4/5/36).

### Requirement: Safe fallback on EDID failure

The driver SHALL provide a `detect_mode()` function that attempts EDID read and parse, falling back to `VideoMode::default_1080p()` on any failure.

#### Scenario: EDID succeeds

WHEN `detect_mode()` is called and EDID read + parse both succeed
THEN it SHALL return the EDID-derived `VideoMode`.

#### Scenario: EDID fails gracefully

WHEN `detect_mode()` is called and EDID read fails (timeout, corrupt, no monitor)
THEN it SHALL return `VideoMode::default_1080p()` and log a warning via UART.
