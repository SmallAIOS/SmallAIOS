## Why

The Jetson Nano boots to UART-only serial output — the HDMI port shows nothing. For development, debugging, and standalone deployment, a framebuffer console over HDMI removes the need for a USB-to-serial adapter and makes SmallAIOS immediately visible on any monitor. The Tegra X1 has dedicated display hardware (DC + SOR) that can drive HDMI without GPU rendering, requiring only MMIO register programming and a linear framebuffer in DRAM.

## What Changes

- Add Tegra X1 Display Controller (DC) driver: clock enable, framebuffer window programming, H/V timing configuration
- Add Tegra X1 SOR (Serial Output Resource) driver for HDMI output: PHY init, TMDS serializer, HDMI mode enable
- Add DPAUX DDC driver for EDID reading: I2C-over-AUX transactions to read monitor capabilities
- Add EDID parser: extract preferred resolution, pixel clock, and sync timings from 128-byte EDID block
- Add framebuffer console: 8x16 bitmap font rasterizer, scrolling text output, `puts`/`putc` API matching UART interface
- Add CAR (Clock and Reset) helpers for display power domains: DISP0, SOR0, DPAUX clock enable and reset deassert
- Wire HDMI init into Tegra boot sequence: after PCIe enumeration, before halt — boot messages appear on both UART and HDMI
- Safe 1080p@60Hz fallback when EDID read fails (1920x1080, 148.5 MHz pixel clock)

## Capabilities

### New Capabilities
- `tegra-dc`: Display Controller driver — framebuffer DMA, window configuration, timing generator, pixel format selection
- `tegra-sor-hdmi`: SOR HDMI output — serializer init, TMDS PHY, pixel clock programming, hot-plug detect
- `tegra-edid`: DPAUX DDC interface and EDID parser — monitor detection, preferred mode extraction, safe fallback
- `framebuffer-console`: Text console over linear framebuffer — bitmap font, scroll buffer, dual-output with UART

### Modified Capabilities
<!-- No existing spec-level requirements change -->

## Impact

- **New code:** `arch/aarch64/src/tegra_dc.rs`, `tegra_sor.rs`, `tegra_edid.rs`, `fb_console.rs` (all behind `#[cfg(feature = "tegra-x1")]`)
- **Modified:** `arch/aarch64/src/lib.rs` (wire HDMI init into `kernel_main`), `arch/aarch64/src/platform.rs` (add DC/SOR/DPAUX base address constants)
- **Memory:** 8 MiB framebuffer allocation at boot (1920x1080 RGBA8888)
- **Binary size:** ~2-4 KB additional `.text` (MMIO register writes + font data in `.rodata`)
- **Dependencies:** None — pure MMIO, no external crates
- **CI:** Tegra cross-build already in CI; new modules gated behind existing `tegra-x1` feature
