## ADDED Requirements

### Requirement: DC clock and reset initialization

The Display Controller driver SHALL enable the DISP0 clock and deassert its reset via CAR registers before accessing any DC0 registers. The driver SHALL verify clock stability before proceeding.

#### Scenario: DC clock enable sequence

WHEN `dc_init()` is called
THEN the driver writes CAR clock enable for DISP0, deasserts DISP0 reset, and waits for clock lock before returning success.

#### Scenario: DC init called before clocks

WHEN any DC register is read or written before `dc_init()` completes
THEN the driver SHALL return an error rather than accessing ungated hardware.

### Requirement: Framebuffer window configuration

The driver SHALL configure DC0 Window A with the framebuffer physical address, pixel dimensions, stride, and RGBA8888 format. The stride MUST be 64-byte aligned.

#### Scenario: Configure 1080p RGBA8888 framebuffer

WHEN `dc_set_framebuffer(addr, 1920, 1080, Format::RGBA8888)` is called
THEN Window A registers SHALL be programmed: `WIN_A_START_ADDR` = `addr`, `WIN_A_SIZE` = 1920 | (1080 << 16), `WIN_A_LINE_STRIDE` = 7680, `WIN_A_BUFFER_CTRL` format bits = `0x3`.

#### Scenario: Non-64-byte-aligned stride

WHEN the computed stride (width x bytes_per_pixel) is not 64-byte aligned
THEN the driver SHALL round up to the next 64-byte boundary.

### Requirement: Display timing configuration

The driver SHALL program horizontal and vertical timing registers with active area, front porch, sync width, and back porch values derived from the selected video mode.

#### Scenario: Standard 1080p60 timings

WHEN the video mode is 1920x1080 at 60 Hz
THEN the DC timing registers SHALL be set to: H-active=1920, H-front-porch=88, H-sync=44, H-back-porch=148, V-active=1080, V-front-porch=4, V-sync=5, V-back-porch=36.

#### Scenario: Custom mode from EDID

WHEN EDID provides a different preferred timing
THEN the DC timing registers SHALL use the EDID-derived values for active area, porches, and sync widths.

### Requirement: Display controller enable and disable

The driver SHALL provide `dc_enable()` to start DMA scanning and `dc_disable()` to stop it. Enabling SHALL set the DISP_CTRL enable bit. Disabling SHALL clear it and wait for the current frame to complete.

#### Scenario: Enable DC output

WHEN `dc_enable()` is called after framebuffer and timing configuration
THEN the DISP_CTRL enable bit SHALL be set and the display controller begins scanning the framebuffer to the output.

#### Scenario: Disable DC output

WHEN `dc_disable()` is called
THEN the DISP_CTRL enable bit SHALL be cleared.

### Requirement: Register definitions with MMIO safety

All DC register accesses SHALL use volatile MMIO read/write functions. Register offset constants SHALL be defined for all accessed registers. All MMIO functions SHALL be marked `unsafe`.

#### Scenario: Register offset correctness

WHEN the module is compiled
THEN all register offset constants SHALL match the Tegra X1 TRM chapter 32 definitions for DC0 at base `0x5420_0000`.
