## Tegra GPU Platform Init

### Overview

Initialize the GM20B GPU's power, clock, reset, and PLL subsystems on the Tegra X1 SoC. This is the foundation for all subsequent GPU operations — without power and clocks, the GPU registers are inaccessible.

### Hardware Details

- **GPU BAR0:** `0x5700_0000` (16 MB) — GPU control/status registers
- **GPU BAR1:** `0x5800_0000` (16 MB) — GPU framebuffer/memory aperture
- **PMC base:** `0x7000_E400` — Power Management Controller (GPU partition at bit 14)
- **CAR base:** `0x6000_6000` — Clock and Reset Controller
- **GPU IRQs:** SPI 189 (stall interrupt), SPI 190 (nonstall interrupt) via GICv2

### Power Sequence

1. Read `PMC_PWRGATE_STATUS` to check if GPU partition (partition 14) is powered
2. If not powered, write to `PMC_PWRGATE_TOGGLE` with partition 14 and START bit
3. Poll `PMC_PWRGATE_STATUS` until bit 14 is set (timeout: 10 ms)
4. Remove GPU clamps via `PMC_REMOVE_CLAMPING_CMD`

### Clock and Reset Sequence

1. Enable GPU clock source in CAR: `CLK_RST_CLK_ENB_SET_W` for GPU
2. Set GPU clock source to PLLP_OUT0 (408 MHz) as initial safe clock
3. Deassert GPU reset: `CLK_RST_RST_DEV_CLR_W` for GPU
4. Wait 10 us for reset propagation

### GPCPLL Configuration

The GPU PLL (GPCPLL) is the primary clock source for GPU cores:

- **Reference clock:** 38.4 MHz oscillator
- **Formula:** `f_gpu = f_ref * N / (M * 2^PL)`
- **Range:** 76.8 MHz (N=1) to 921.6 MHz (N=12) with M=1, PL=0
- **Default:** 614.4 MHz (N=8) — balanced performance and thermal margin

Configuration sequence:
1. Bypass GPCPLL (switch GPU to ref clock)
2. Program GPCPLL_COEFF: M, N, PL values
3. Enable GPCPLL and wait for lock (poll GPCPLL_CFG LOCK bit, timeout: 1 ms)
4. Switch GPU clock source from ref to GPCPLL output
5. Disable bypass

### Interrupt Setup

1. Enable GICv2 SPI 189 (stall) and SPI 190 (nonstall) via `gicv2::enable_irq()`
2. Configure GPU interrupt tree: `NV_PMC_INTR_EN_0` to enable stall/nonstall sources
3. Clear any pending interrupts via `NV_PMC_INTR_0`

### Interface

```rust
pub struct TegraGpuPlatform {
    bar0_base: usize,
    bar1_base: usize,
    gpcpll_freq_mhz: u32,
    powered: bool,
    clocks_enabled: bool,
}

impl TegraGpuPlatform {
    pub fn new() -> Self;
    pub fn power_on() -> Result<(), GpuError>;
    pub fn enable_clocks() -> Result<(), GpuError>;
    pub fn configure_gpcpll(freq_step: u8) -> Result<(), GpuError>;
    pub fn enable_interrupts() -> Result<(), GpuError>;
    pub fn power_off() -> Result<(), GpuError>;
}
```

### Verification

- Unit tests for PLL coefficient calculations (all 12 steps)
- Unit tests for power partition bit manipulation
- Unit tests for clock enable/reset deassert sequencing
- Mock MMIO tests for full init sequence state machine
