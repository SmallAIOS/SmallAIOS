## 1. Platform Constants and Linker Script

- [x] 1.1 Add `DC0_BASE`, `SOR0_BASE`, `DPAUX1_BASE`, `PMC_BASE` constants to `arch/aarch64/src/platform.rs` under `#[cfg(feature = "tegra-x1")]`
- [x] 1.2 Add `FRAMEBUFFER_BASE` constant (`0x8F00_0000`) and `FRAMEBUFFER_SIZE` (8 MiB) to `platform.rs`
- [x] 1.3 Update `arch/aarch64/linker-tegra.ld` to reserve the 8 MiB framebuffer region with a `__fb_start` / `__fb_end` symbol pair
- [x] 1.4 Write unit tests verifying all new platform constants match TRM addresses

## 2. Display Controller (tegra_dc.rs)

- [x] 2.1 Create `arch/aarch64/src/tegra_dc.rs` with DC0 register offset constants: `DISP_CTRL`, `H_TIMING`, `V_TIMING`, `DI_SIZE`, `WIN_A_START_ADDR`, `WIN_A_SIZE`, `WIN_A_LINE_STRIDE`, `WIN_A_BYTE_SWAP`, `WIN_A_BUFFER_CTRL`
- [x] 2.2 Implement MMIO read/write helpers (`dc_read32`, `dc_write32`) using volatile ops at `DC0_BASE`
- [x] 2.3 Implement CAR clock enable for DISP0: `dc_enable_clock()` — enable DISP0 clock bit, deassert DISP0 reset, poll for stability
- [x] 2.4 Define `PixelFormat` enum (`RGBA8888 = 0x3`, `RGB888 = 0x0`, `RGB565 = 0x1`) and stride alignment helper (`align_stride_64`)
- [x] 2.5 Implement `dc_set_framebuffer(addr, width, height, format)`: program `WIN_A_START_ADDR`, `WIN_A_SIZE`, `WIN_A_LINE_STRIDE` (64-byte aligned), `WIN_A_BUFFER_CTRL`
- [x] 2.6 Implement `dc_set_timing(mode: &VideoMode)`: program horizontal timing (active, front porch, sync, back porch) and vertical timing registers
- [x] 2.7 Implement `dc_enable()`: set `DISP_CTRL` enable bit to start framebuffer DMA scanning
- [x] 2.8 Implement `dc_disable()`: clear `DISP_CTRL` enable bit
- [x] 2.9 Implement `dc_init(mode: &VideoMode, fb_addr: usize)`: enable clock → set framebuffer → set timing → enable (full init sequence)
- [x] 2.10 Register `#[cfg(feature = "tegra-x1")] pub mod tegra_dc;` in `lib.rs`
- [x] 2.11 Write unit tests: register offset values, stride alignment (7680 → 7680, 5760 → 5824), `PixelFormat` codes, `WIN_A_SIZE` packing (width | height << 16)
- [x] 2.12 Write unit tests: 1080p60 timing register values match CEA-861 (H: 88/44/148, V: 4/5/36)

## 3. SOR HDMI Output (tegra_sor.rs)

- [x] 3.1 Create `arch/aarch64/src/tegra_sor.rs` with SOR0 register offset constants: `SOR_CTRL`, `SOR_STATE_1`, `SOR_CLK_CNTRL`, `SOR_HDMI_CTRL`, `SOR_PLL_CTRL`, `SOR_PLL_STATUS`, `SOR_PHY_CNTRL`, `SOR_HEAD_STATE_0`
- [x] 3.2 Implement MMIO helpers (`sor_read32`, `sor_write32`) at `SOR0_BASE`
- [x] 3.3 Implement CAR clock enable for SOR0: `sor_enable_clock()` — enable SOR0 clock bit, deassert SOR0 reset
- [x] 3.4 Implement PLLD configuration: `sor_configure_plld(pixel_clock_khz)` — set PLLD feedback/input dividers for target pixel clock, poll PLL lock with bounded timeout
- [x] 3.5 Implement TMDS PHY init: `sor_init_phy()` — configure drive strength, pre-emphasis, enable data lanes, verify PLL lock
- [x] 3.6 Implement `sor_enable_hdmi()`: set `SOR_HDMI_CTRL` to HDMI mode, configure head mux to route DC0 → SOR0
- [x] 3.7 Implement `sor_init(mode: &VideoMode)`: enable clock → configure PLLD → init PHY → enable HDMI (full init sequence); return error if PLL fails to lock
- [x] 3.8 Register `#[cfg(feature = "tegra-x1")] pub mod tegra_sor;` in `lib.rs`
- [x] 3.9 Write unit tests: register offset values match TRM, PLL lock timeout constant, HDMI mode bit position
- [x] 3.10 Write unit tests: PLLD divider calculation for 148.5 MHz (1080p60) and 74.25 MHz (720p60)

## 4. EDID and Mode Detection (tegra_edid.rs)

- [x] 4.1 Create `arch/aarch64/src/tegra_edid.rs` with DPAUX1 register offset constants: `DPAUX_HYBRID_PADCTL`, `DPAUX_DP_AUXDATA`, `DPAUX_DP_AUXADDR`, `DPAUX_DP_AUXCTL`, `DPAUX_DP_AUXSTAT`
- [x] 4.2 Implement MMIO helpers (`dpaux_read32`, `dpaux_write32`) at `DPAUX1_BASE`
- [x] 4.3 Implement `dpaux_init()`: enable DPAUX1 clock via CAR, deassert reset, configure hybrid pad for I2C/DDC mode
- [x] 4.4 Implement `dpaux_i2c_read(slave_addr, offset, buf)`: perform I2C read transaction via AUX controller with bounded polling timeout
- [x] 4.5 Implement `read_edid() -> Result<[u8; 128], EdidError>`: read 128 bytes from DDC address 0x50, validate magic header `[0x00, 0xFF, ..., 0x00]`
- [x] 4.6 Define `VideoMode` struct: `width`, `height`, `pixel_clock_khz`, `h_front_porch`, `h_sync_width`, `h_back_porch`, `v_front_porch`, `v_sync_width`, `v_back_porch`
- [x] 4.7 Implement `VideoMode::default_1080p()`: return CEA-861 1920x1080@60Hz timing constants
- [x] 4.8 Implement `VideoMode::default_720p()`: return CEA-861 1280x720@60Hz timing constants
- [x] 4.9 Implement `parse_preferred_timing(edid: &[u8; 128]) -> Result<VideoMode, EdidError>`: extract pixel clock and H/V timing from EDID bytes 54-71 (first detailed timing descriptor)
- [x] 4.10 Implement `detect_mode() -> VideoMode`: try `read_edid()` + `parse_preferred_timing()`, fall back to `default_1080p()` on any error; log UART warning on fallback
- [x] 4.11 Define `EdidError` enum: `Timeout`, `InvalidMagic`, `ParseError`, `DdcNak`
- [x] 4.12 Register `#[cfg(feature = "tegra-x1")] pub mod tegra_edid;` in `lib.rs`
- [x] 4.13 Write unit tests: `VideoMode::default_1080p()` returns correct CEA-861 values (148500 kHz, H 88/44/148, V 4/5/36)
- [x] 4.14 Write unit tests: parse real 1080p EDID bytes (construct test vector for bytes 54-71), verify extracted `VideoMode` matches
- [x] 4.15 Write unit tests: parse 720p EDID test vector, verify width=1280 height=720 pixel_clock=74250
- [x] 4.16 Write unit tests: invalid EDID magic returns `EdidError::InvalidMagic`
- [x] 4.17 Write unit tests: EDID checksum validation (sum of all 128 bytes mod 256 = 0)

## 5. Framebuffer Console (fb_console.rs)

- [x] 5.1 Create `arch/aarch64/src/fb_console.rs` with `FbConsole` struct: `fb_addr`, `width`, `height`, `stride`, `cursor_row`, `cursor_col`, `cols`, `rows`, `initialized`
- [x] 5.2 Embed 8x16 bitmap font as `static FONT_8X16: [u8; 4096]` covering glyphs 0x00-0xFF; use VGA/CP437-compatible bitmaps for ASCII range
- [x] 5.3 Implement `fb_console_init(addr, width, height, stride)`: zero framebuffer memory, set cursor to (0, 0), compute `cols = width/8` and `rows = height/16`
- [x] 5.4 Implement `render_glyph(row, col, byte)`: write 8x16 pixels into framebuffer at pixel position `(col*8, row*16)` using font lookup; white foreground (0xFFFFFFFF), black background (0x00000000)
- [x] 5.5 Implement `fb_putc(byte)`: handle `\n` (newline → next row col 0), printable chars (render + advance col), wrap at `cols`, scroll if past `rows`
- [x] 5.6 Implement `scroll_up()`: memmove framebuffer up by `stride * 16` bytes, zero the bottom 16 rows of pixels
- [x] 5.7 Implement `fb_puts(s: &str)`: iterate bytes and call `fb_putc` for each
- [x] 5.8 Use a `static mut` global `FbConsole` instance (or `AtomicBool` + raw pointer) for single-core early boot access
- [x] 5.9 Register `#[cfg(feature = "tegra-x1")] pub mod fb_console;` in `lib.rs`
- [x] 5.10 Write unit tests: font data is exactly 4096 bytes, glyph 0x41 ('A') offset is 1040
- [x] 5.11 Write unit tests: `cols` and `rows` computation (1920/8=240, 1080/16=67)
- [x] 5.12 Write unit tests: cursor advancement — single char, line wrap at col 240, newline resets col to 0
- [x] 5.13 Write unit tests: scroll detection — newline at row 66 triggers scroll

## 6. Console Dual-Output Abstraction

- [x] 6.1 Create `arch/aarch64/src/console.rs` with `console::putc(byte)` and `console::puts(s)` that delegate to `uart::putc` + `fb_console::fb_putc` on `tegra-x1`, or `uart` only on `qemu-virt`
- [x] 6.2 Add `console::init()` that calls `fb_console_init` on `tegra-x1` (no-op on `qemu-virt`)
- [x] 6.3 Add `pub mod console;` to `lib.rs` (unconditional — works on both platforms)
- [x] 6.4 Write unit tests: on `qemu-virt` cfg, `console::puts` compiles and delegates to `uart::puts` only

## 7. Boot Sequence Integration

- [x] 7.1 Add Stage 5 block to `kernel_main` (behind `tegra-x1`): detect mode → init SOR → init DC → init framebuffer console; print `[hdmi] ...` status messages to UART during init
- [x] 7.2 Replace all `uart::puts` calls after Stage 5 with `console::puts` so the "ready" banner appears on both UART and HDMI
- [x] 7.3 Handle display init failure gracefully: if SOR PLL lock fails, print UART warning and continue boot without HDMI (do not panic)
- [x] 7.4 Verify QEMU build (`--features qemu-virt`) still compiles and boots unchanged
- [x] 7.5 Verify Tegra build (`--features tegra-x1`) compiles with zero warnings and produces correct Image

## 8. Testing and CI

- [x] 8.1 Run `cargo test -p smallaios-arch-aarch64` and verify all new unit tests pass
- [x] 8.2 Run `cargo clippy -p smallaios-arch-aarch64 --features tegra-x1` and verify zero warnings
- [x] 8.3 Run `make build-kernel-jetson` and verify Image builds cleanly
- [x] 8.4 Run `make check-size-jetson` and verify Image stays under 15 MB
- [x] 8.5 Verify CI: both QEMU and Tegra build jobs pass, clippy clean
