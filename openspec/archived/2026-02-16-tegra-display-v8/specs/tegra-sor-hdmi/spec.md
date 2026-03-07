## ADDED Requirements

### Requirement: SOR clock and reset initialization

The SOR HDMI driver SHALL enable the SOR0 clock and deassert its reset via CAR registers. The driver SHALL configure the display PLL (PLLD) to produce the required pixel clock frequency.

#### Scenario: SOR0 clock enable

WHEN `sor_init()` is called
THEN the driver enables SOR0 clock via CAR, deasserts SOR0 reset, and configures PLLD for the target pixel clock.

#### Scenario: 148.5 MHz pixel clock for 1080p60

WHEN the target mode is 1920x1080 at 60 Hz
THEN PLLD SHALL be configured to produce 148.5 MHz (within 0.5% tolerance per HDMI spec).

### Requirement: HDMI mode enable

The driver SHALL configure SOR0 for HDMI output mode (not DisplayPort). The HDMI control register SHALL be programmed to enable TMDS output.

#### Scenario: Enable HDMI output

WHEN `sor_enable_hdmi()` is called after clock init
THEN SOR0_HDMI_CTRL SHALL be set to enable HDMI mode and TMDS serialization begins.

#### Scenario: SOR head mux routing

WHEN HDMI output is enabled
THEN SOR0 head state registers SHALL route DC0 output to SOR0 (DISP0 → SOR0 path).

### Requirement: TMDS PHY configuration

The driver SHALL configure the SOR0 PHY for the appropriate TMDS lane settings: pre-emphasis, voltage swing, and lane enable. The PHY PLL SHALL be locked before enabling output.

#### Scenario: PHY PLL lock

WHEN the PHY is configured
THEN the driver SHALL poll `SOR_PLL_STATUS` until the lock bit is set, with a bounded timeout. If the PLL fails to lock, `sor_init()` SHALL return an error.

#### Scenario: Standard TMDS drive strength

WHEN HDMI output is at 1080p60 (pixel clock 148.5 MHz, TMDS clock 1.485 GHz)
THEN PHY drive strength and pre-emphasis SHALL be set to Tegra X1 recommended values for HDMI 1.4.

### Requirement: SOR register definitions

All SOR register accesses SHALL use volatile MMIO read/write functions. Register offset constants SHALL match the Tegra X1 TRM chapter 34 for SOR0 at base `0x5454_0000`.

#### Scenario: Register base and offsets

WHEN the module is compiled
THEN `SOR0_BASE` SHALL equal `0x5454_0000` and all register offsets SHALL match TRM definitions for SOR_CTRL, SOR_HDMI_CTRL, SOR_PLL_CTRL, SOR_PLL_STATUS, SOR_PHY_CNTRL.

### Requirement: Graceful failure on no sink

The driver SHALL not hang or panic if no HDMI monitor is connected. TMDS output proceeds regardless — the DC continues to DMA the framebuffer.

#### Scenario: No monitor connected

WHEN `sor_init()` completes but no HDMI sink is present
THEN the function SHALL return success. The display pipeline runs without a sink; UART output is unaffected.
