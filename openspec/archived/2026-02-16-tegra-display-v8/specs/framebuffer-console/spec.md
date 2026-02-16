## ADDED Requirements

### Requirement: Framebuffer initialization

The framebuffer console SHALL initialize with the framebuffer physical address, width, height, and stride. It SHALL clear the framebuffer to black (all zeros) on init.

#### Scenario: Init 1080p framebuffer

WHEN `fb_console_init(addr, 1920, 1080, 7680)` is called
THEN the framebuffer memory (1920 x 1080 x 4 = 8,294,400 bytes) SHALL be zeroed and the console cursor set to row 0, column 0.

### Requirement: 8x16 bitmap font

The console SHALL embed a 256-glyph bitmap font where each glyph is 8 pixels wide and 16 pixels tall, stored as 16 bytes per glyph (one byte per row, MSB = leftmost pixel). The font SHALL cover ASCII printable characters (0x20-0x7E) at minimum.

#### Scenario: Font data size

WHEN the font is compiled
THEN the font data SHALL occupy exactly 4,096 bytes (256 glyphs x 16 bytes).

#### Scenario: Glyph lookup

WHEN rendering character `'A'` (0x41)
THEN the font lookup SHALL return the 16-byte bitmap at offset `0x41 * 16 = 1040`.

### Requirement: Character rendering

The console SHALL render characters by writing foreground pixels (white, `0xFFFFFFFF`) for set bits and background pixels (black, `0x00000000`) for clear bits in the glyph bitmap, directly into the framebuffer at the current cursor position.

#### Scenario: Render single character

WHEN `fb_putc(b'X')` is called at cursor position (row=0, col=0)
THEN 8x16 pixels SHALL be written to the framebuffer starting at offset 0, advancing the cursor to column 1.

#### Scenario: Foreground and background colors

WHEN a character is rendered
THEN set bits in the glyph bitmap produce white pixels (0xFFFFFFFF RGBA) and clear bits produce black pixels (0x00000000 RGBA).

### Requirement: Text output with puts

The console SHALL provide `fb_puts(s: &str)` that iterates over bytes and calls `fb_putc` for each. This is the primary text output API.

#### Scenario: Print string

WHEN `fb_puts("Hello\n")` is called
THEN characters H, e, l, l, o are rendered at successive cursor positions, followed by a newline advancing to the next row.

### Requirement: Line wrapping

The console SHALL wrap to the next line when the cursor column exceeds the screen width in characters (width / 8).

#### Scenario: Wrap at screen edge

WHEN the cursor is at column 239 (last column at 1920/8=240 chars) and a character is printed
THEN the character SHALL be rendered at column 239, and the cursor SHALL advance to column 0 of the next row.

### Requirement: Newline handling

The console SHALL treat byte `0x0A` (newline) by moving the cursor to column 0 of the next row without rendering a glyph.

#### Scenario: Newline character

WHEN `fb_putc(b'\n')` is called at cursor (row=5, col=30)
THEN the cursor SHALL move to (row=6, col=0) with no glyph rendered.

### Requirement: Vertical scrolling

When the cursor advances past the last row (height / 16), the console SHALL scroll the framebuffer content up by one text row (16 pixel rows) and clear the new bottom row.

#### Scenario: Scroll at bottom

WHEN the cursor is at the last row (row 66 at 1080/16=67 rows) and a newline occurs
THEN all framebuffer content SHALL shift up by 16 pixel rows (memcpy), the bottom 16 pixel rows SHALL be cleared to black, and the cursor SHALL remain at the last row, column 0.

### Requirement: Dual output console abstraction

A `console` module SHALL provide `console::puts(s)` and `console::putc(b)` that write to both UART and framebuffer simultaneously. On platforms without a framebuffer (e.g., QEMU), it SHALL fall back to UART-only via `#[cfg]` gating.

#### Scenario: Dual output on Tegra

WHEN `console::puts("boot")` is called on a `tegra-x1` build after framebuffer init
THEN "boot" SHALL appear on both the UART serial output and the HDMI framebuffer console.

#### Scenario: UART-only fallback

WHEN `console::puts("boot")` is called on a `qemu-virt` build
THEN "boot" SHALL appear on UART only, with no framebuffer writes.

### Requirement: Boot sequence integration

The `kernel_main` function SHALL call display init (DC + SOR + framebuffer console) as a new boot stage after PCIe enumeration. After framebuffer init, all subsequent boot messages SHALL use `console::puts` for dual output.

#### Scenario: Display init in boot

WHEN `kernel_main` executes on a `tegra-x1` build
THEN display initialization SHALL occur after PCIe enumeration (Stage 4) and before the final "ready" banner. The "ready" banner SHALL appear on both UART and HDMI.
