## Context

SmallAIOS boots on Jetson Nano to a UART-only serial console. The Tegra X1 SoC has a dedicated display pipeline — Display Controller (DC) reads a linear framebuffer via DMA, Serial Output Resource (SOR) serializes pixels to HDMI TMDS — that operates independently of the GPU. U-Boot does not initialize display hardware; all DISP/SOR/DPAUX clocks are gated at kernel entry. The existing `arch/aarch64` crate already has platform constants, CAR register helpers, and UART output behind the `tegra-x1` feature flag.

Key hardware addresses (Tegra X1 / Tegra210):
- DC0 (Display Controller 0): `0x5420_0000`
- SOR0 (HDMI serializer): `0x5454_0000`
- DPAUX1 (DDC for HDMI): `0x545C_0000`
- CAR (Clock & Reset): `0x6000_6000`
- PMC (Power Management): `0x7000_E000`

## Goals / Non-Goals

**Goals:**
- HDMI output with text console at boot — boot messages visible on any HDMI monitor
- Dual output: all `uart::puts` calls also render to the framebuffer console
- EDID-based resolution detection with safe 1080p@60Hz fallback
- Minimal binary size impact (<4 KB `.text`, ~4 KB font in `.rodata`)
- Clean integration into existing boot flow (after PCIe, before halt)

**Non-Goals:**
- GPU-accelerated rendering (no 3D, no Tegra GR3D/GPU involvement)
- Multiple display head support (DC1/SOR1 unused)
- Audio over HDMI (SOR audio registers untouched)
- Display hot-plug after boot (init once, no runtime HPD handling)
- Window compositing or hardware cursor (single full-screen window A only)
- User-space framebuffer API or `/dev/fb0` device node

## Decisions

### 1. All display code in `arch/aarch64` behind `tegra-x1` feature

**Decision:** Four new modules in `arch/aarch64/src/`: `tegra_dc.rs`, `tegra_sor.rs`, `tegra_edid.rs`, `fb_console.rs`. All gated on `#[cfg(feature = "tegra-x1")]`.

**Rationale:** The display controller is Tegra-specific MMIO — it belongs in the arch HAL alongside `tegra_pcie.rs` and `gicv2.rs`. No cross-platform abstraction needed since only Tegra X1 has this hardware. Keeps the `kernel` and `net` crates unaffected.

**Alternative considered:** Separate `display` crate. Rejected — premature abstraction for a single SoC's display controller. Can be extracted later if multi-platform display support is added.

### 2. Single window (Window A), RGBA8888 format

**Decision:** Use DC Window A with `T_RGBA8888` (format code `0x3`, 32 bits per pixel). No tiling, no rotation.

**Rationale:** Simplest format — 4 bytes per pixel with natural alignment. Stride is automatically 64-byte aligned at 1080p (1920 × 4 = 7680). The extra alpha byte costs 2 MB more than RGB888 but eliminates stride padding complexity and byte-swap issues. Window A is the primary overlay and sufficient for a full-screen console.

### 3. Static framebuffer at fixed DRAM offset

**Decision:** Reserve framebuffer at a fixed physical address in the linker script (e.g., `0x8F00_0000`), avoiding dynamic allocation.

**Rationale:** The buddy allocator in `kernel/src/mem` is designed for general-purpose allocation and may not be initialized this early in boot. A fixed address in the linker script guarantees the framebuffer is available before any allocator setup. 8 MiB reserved at the end of the first 256 MiB of DRAM, below the kernel's working memory. The DTB memory parser can exclude this region.

**Alternative considered:** `buddy_allocate()` at runtime. Rejected for initial implementation — adds a dependency on allocator init ordering. Can switch to dynamic allocation once boot sequence matures.

### 4. 8×16 bitmap font embedded in `.rodata`

**Decision:** Ship a 256-glyph, 8×16 bitmap font as a `[u8; 4096]` constant array (256 glyphs × 16 rows × 1 byte per row).

**Rationale:** 4 KiB is negligible. 8×16 is the standard VGA text mode cell size — gives 240×67 characters at 1920×1080. Each glyph row is a single byte (8 pixels wide), making the rasterizer a tight bitwise loop. No external font file needed.

### 5. Dual output via `console` abstraction

**Decision:** Introduce a `console::puts(s)` function that writes to both UART and framebuffer. Replace direct `uart::puts` calls in `kernel_main` with `console::puts`.

**Rationale:** The user wants to see boot output on both serial and HDMI simultaneously. A thin `console` module that delegates to `uart::puts` + `fb_console::puts` avoids duplicating every print call. On QEMU (no framebuffer), `console::puts` falls back to UART-only via `#[cfg]`.

### 6. EDID reading with graceful fallback

**Decision:** Attempt DDC read via DPAUX1. If it fails (no monitor, DDC timeout, invalid EDID magic), fall back to 1920×1080@60Hz.

**Rationale:** Most HDMI monitors support 1080p. Reading EDID is best-effort — a failed read should not prevent boot. The fallback timings (148.5 MHz pixel clock, standard CEA-861 sync parameters) are universally compatible. EDID parsing extracts only the preferred detailed timing descriptor (bytes 54–71) — no need for a full EDID library.

### 7. Init sequence ordering

**Decision:** Display init happens in `kernel_main` after PCIe enumeration and before the "ready" banner:

```
Stage 1: Early init (UART, BSS, EL detection)
Stage 2: Memory detection (DTB parse)
Stage 3: Interrupt controller (GICv2)
Stage 4: PCIe enumeration
Stage 5: Display init (DC + SOR + HDMI)  ← NEW
Boot complete banner (on both UART and HDMI)
```

**Rationale:** Display init needs CAR clocks but no PCIe or interrupts. Placing it after PCIe keeps the existing stages untouched and lets the HDMI output show the final "ready" banner. Early boot messages (stages 1–4) appear on UART only; stage 5 onward appears on both.

## Risks / Trade-offs

- **[No monitor connected]** → SOR init completes but TMDS has no sink. Harmless — UART still works, DC just DMA's to an unseen framebuffer. No hang risk.
- **[Monitor doesn't support 1080p]** → Fallback timings may produce no image. Mitigation: EDID read gives us the monitor's preferred mode. If EDID fails AND the monitor is below 1080p, there's no output. Acceptable for v1 — 720p fallback could be added later.
- **[Framebuffer memory overlap]** → Fixed address could conflict with kernel heap growth. Mitigation: place at `0x8F00_0000` (240 MiB offset), well above the kernel's initial working set (~16 MiB). Document the reservation in the linker script.
- **[Pixel clock accuracy]** → CAR dividers may not produce exactly 148.5 MHz. Mitigation: HDMI spec allows ±0.5% tolerance. Integer dividers from PLLP (408 MHz) give 136 MHz (÷3) or 204 MHz (÷2) — neither is exact. Use PLLD (display PLL) with fractional divider for precise 148.5 MHz.
- **[Binary size]** → Font data adds ~4 KiB to `.rodata`. Acceptable — current Image is 9 KiB total, even doubling it stays well under the 15 MiB limit.
