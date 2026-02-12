# Hardware Peripheral Interfaces - Design Document

## Architecture Overview

A new `peripheral` crate is added to the workspace, containing all peripheral bus drivers (I2C, SPI, GPIO, UART) and higher-level subsystems (CSI camera, I2S audio). The kernel crate is extended with new HAL traits and HalError variants. The security crate gains new capability types. Feature flags gate everything at compile time.

```
peripheral/
  src/
    lib.rs              -- Crate root, feature-gated module declarations
    i2c/
      mod.rs            -- I2cController trait re-export, config types
      arm_mmio.rs       -- ARM Designware/generic MMIO I2C driver
      riscv_mmio.rs     -- RISC-V SiFive/OpenCores I2C driver
      axi_iic.rs        -- Xilinx AXI IIC FPGA driver (uses FpgaFabric trait)
      bitbang.rs        -- GPIO bitbang fallback (requires gpio feature)
    spi/
      mod.rs            -- SpiController trait re-export, config types
      arm_mmio.rs       -- ARM PL022/Designware SSI driver
      riscv_mmio.rs     -- RISC-V SiFive SPI driver
      axi_quad_spi.rs   -- Xilinx AXI Quad SPI FPGA driver
    gpio/
      mod.rs            -- GpioController trait re-export, config types
      arm_pl061.rs      -- ARM PL061 GPIO driver
      riscv_mmio.rs     -- RISC-V SiFive GPIO driver
      axi_gpio.rs       -- Xilinx AXI GPIO FPGA driver
    uart/
      mod.rs            -- UartController trait re-export, config types
      pl011.rs          -- ARM PL011 full-featured driver
      ns16550a.rs       -- NS16550A-compatible driver
      sifive.rs         -- SiFive UART driver
      axi_uart_lite.rs  -- Xilinx AXI UART Lite FPGA driver
      nmea.rs           -- Optional NMEA 0183 parser
    camera/
      mod.rs            -- CsiReceiver trait re-export, config types
      sensor.rs         -- I2C sensor detection and configuration tables
      ov5640.rs         -- OV5640 register table
      imx219.rs         -- IMX219 register table
      imx477.rs         -- IMX477 register table
      tegra_vi.rs       -- NVIDIA Jetson Tegra VI/CSI driver
      broadcom_unicam.rs -- Raspberry Pi Unicam CSI driver
      fpga_csi.rs       -- FPGA CSI receiver IP driver
      preprocess.rs     -- Optional YUV→RGB, resize, normalize for ONNX
    audio/
      mod.rs            -- I2sController trait re-export, config types
      codec.rs          -- I2C codec detection and configuration tables
      tlv320aic3x.rs    -- TLV320AIC3x register table
      wm8960.rs         -- WM8960 register table
      es8388.rs         -- ES8388 register table
      arm_i2s.rs        -- ARM MMIO I2S driver
      riscv_i2s.rs      -- RISC-V MMIO I2S driver
      fpga_i2s.rs       -- FPGA I2S IP driver
      preprocess.rs     -- Optional PCM→float32, mel spectrogram for ONNX
```

## Key Design Decisions

### 1. Separate `peripheral` Crate (Not Extending `bus`)

The existing `bus` crate handles safety-critical field buses (CAN, ARINC, MIL-STD-1553) with Zenoh transport adapters. Peripheral interfaces (I2C, SPI, GPIO, UART) are fundamentally different:

- **Different abstraction level**: Bus protocols are message-oriented (frames); peripherals are byte/register-oriented
- **Different security posture**: Bus protocols are part of the safety-critical data path; peripherals are sensor I/O
- **Different consumers**: Bus crate feeds IPC; peripheral crate feeds ONNX input pipeline and device control
- **Clean feature isolation**: Entire peripheral crate compiles to zero when all features are off

### 2. HAL Traits in `kernel/src/hal.rs`, Drivers in `peripheral/`

Following the existing pattern where `BusController` and `FpgaFabric` traits live in the kernel crate while implementations live in architecture/bus crates:

- **Traits** (`I2cController`, `SpiController`, `GpioController`, `CsiReceiver`, `I2sController`, `UartController`) are defined in `kernel/src/hal.rs`
- **Config types** and **error variants** also in `kernel/src/hal.rs`
- **Platform drivers** (concrete implementations) in `peripheral/` crate
- This allows the kernel to reference traits without depending on driver implementations

### 3. All Features Default OFF (Security-by-Default)

```toml
[features]
default = []
i2c = []
spi = []
gpio = []
camera-csi = ["i2c"]    # CSI depends on I2C for sensor config
audio-i2s = ["i2c"]     # I2S depends on I2C for codec config
uart = []

# Convenience bundles
sensor-io = ["i2c", "spi", "gpio"]           # Basic sensor access
vision = ["camera-csi", "i2c", "gpio"]       # Camera + inference
audio = ["audio-i2s", "i2c"]                 # Audio + inference
full-peripheral = ["i2c", "spi", "gpio", "camera-csi", "audio-i2s", "uart"]
```

This produces zero binary overhead when features are disabled. Datacenter and avionics builds simply omit peripheral features.

### 4. Capability-Based Access Control

Each peripheral type has a dedicated capability in the security crate:

```rust
pub enum PeripheralCapability {
    I2c { bus: u8 },        // Access to specific I2C bus
    Spi { bus: u8 },        // Access to specific SPI bus
    Gpio { controller: u8 }, // Access to specific GPIO controller
    Camera { device: u8 },  // Access to specific camera
    Audio { device: u8 },   // Access to specific audio device
    Uart { port: u8 },      // Access to specific UART port
}
```

The device syscall layer checks capabilities before granting access. Capabilities are granted at container/process creation time via the container config.

### 5. DTB-Based Peripheral Discovery

All platform drivers use Device Tree Blob (DTB) parsing for hardware discovery, reusing the existing DTB infrastructure from the `arch/aarch64` and `arch/riscv64` crates:

```
DTB Node → compatible string match → extract base_addr, irq, clock → register device
```

Discovery flow:
1. At boot, the kernel parses the DTB (already implemented)
2. For each enabled peripheral feature, scan DTB for matching `compatible` strings
3. Instantiate the appropriate platform driver with extracted MMIO base and IRQ
4. Register the device with the kernel device manager (extends `sys_dev_enumerate`)

### 6. Zero-Copy ONNX Integration for Camera and Audio

Camera frames and audio buffers are allocated in DMA-capable physically contiguous memory. The ONNX runtime can access them directly via physical addresses:

```
Camera: CSI RX → DMA → Frame Buffer → (optional preprocess) → ONNX input tensor
Audio:  I2S RX → DMA → Ring Buffer  → (optional preprocess) → ONNX input tensor
```

No memory copies in the fast path. Preprocessing (format conversion, normalization) is optional and runs in-place or into a separate pre-allocated buffer.

### 7. Boot Console UARTs Are Separate

The existing minimal UART drivers in `arch/aarch64` (PL011) and `arch/riscv64` (NS16550A) are boot-time console drivers — polling TX only, no RX, no configuration. The new `UartController` in the `peripheral` crate is a full-featured driver with interrupt/DMA RX, flow control, and configurable baud rates. They coexist without conflict:

- Boot console: initialized early in boot, uses polling, always available
- Peripheral UART: initialized after DTB parsing, uses interrupts/DMA, feature-gated

### 8. FPGA Drivers Use Existing `FpgaFabric` Trait

All FPGA-based peripheral drivers (AXI IIC, AXI Quad SPI, AXI GPIO, AXI UART Lite) access hardware via the `FpgaFabric` trait already defined in `kernel/src/hal.rs`. This avoids duplicating MMIO/DMA infrastructure:

```rust
// FPGA I2C driver wraps FpgaFabric
pub struct AxiIicDriver<F: FpgaFabric> {
    fabric: F,
    base_addr: u64,
}

impl<F: FpgaFabric> I2cController for AxiIicDriver<F> {
    fn init(&mut self, config: I2cConfig) -> Result<(), HalError> {
        // Use self.fabric.write_reg() to configure AXI IIC registers
    }
    // ...
}
```

## New HalError Variants

The following variants are added to `HalError` in `kernel/src/hal.rs`:

```rust
pub enum HalError {
    // ... existing variants ...

    /// I2C: target device did not acknowledge (address or data NACK).
    NackReceived,
    /// I2C: another master won bus arbitration.
    ArbitrationLost,
    /// UART: stop bit not detected.
    FramingError,
    /// UART: parity mismatch.
    ParityError,
    /// UART/I2S: receive FIFO overflow, data lost.
    Overrun,
}
```

## Workspace Integration

### Cargo.toml Changes

**Root `Cargo.toml`** — add `peripheral` to workspace members:
```toml
members = [
    "kernel",
    "security",
    # ... existing crates ...
    "peripheral",    # NEW
]
```

**`peripheral/Cargo.toml`**:
```toml
[package]
name = "smallaios-peripheral"
version = "0.1.0"
edition = "2024"

[dependencies]
smallaios-kernel = { path = "../kernel" }
smallaios-security = { path = "../security" }

[features]
default = []
i2c = []
spi = []
gpio = []
camera-csi = ["i2c"]
audio-i2s = ["i2c"]
uart = []
sensor-io = ["i2c", "spi", "gpio"]
vision = ["camera-csi", "i2c", "gpio"]
audio = ["audio-i2s", "i2c"]
full-peripheral = ["i2c", "spi", "gpio", "camera-csi", "audio-i2s", "uart"]
```

**`kernel/Cargo.toml`** — no changes needed (traits are unconditionally defined; feature gating happens in `peripheral` crate).

**`security/Cargo.toml`** — add peripheral capability types (always compiled, zero-cost if unused).

## Platform Driver Matrix

| Interface | ARM (Jetson/RPi) | RISC-V (SiFive/PolarFire) | FPGA (Zynq/PolarFire fabric) | x86 |
|-----------|-----------------|--------------------------|------------------------------|-----|
| I2C | Designware MMIO | SiFive/OpenCores MMIO | Xilinx AXI IIC | Bitbang (if GPIO available) |
| SPI | PL022/Designware | SiFive SPI | Xilinx AXI Quad SPI | N/A (use USB-SPI adapter) |
| GPIO | PL061/Broadcom/Tegra | SiFive GPIO | Xilinx AXI GPIO | N/A |
| UART | PL011 | SiFive UART | Xilinx AXI UART Lite | NS16550A (COM ports) |
| CSI Camera | Tegra VI / Unicam | N/A (no standard CSI) | FPGA CSI IP | N/A |
| I2S Audio | Tegra I2S / BCM PCM | MMIO I2S | FPGA I2S IP | N/A |

## Formal Verification Models

| Model | Property | Tool |
|-------|----------|------|
| I2C multi-master arbitration | Correctness, liveness, no deadlock | TLA+ |
| SPI clock/data phase | Mode correctness, CS ordering | TLA+ |
| GPIO interrupt handling | No missed interrupts, atomic set_mask | TLA+ |
| UART RX FIFO overflow | Flow control prevents data loss | TLA+ |
| CSI frame buffer lifecycle | No use-after-enqueue, no double-free | TLA+ |
| I2S ring buffer | Gapless capture, overrun detection | TLA+ |
