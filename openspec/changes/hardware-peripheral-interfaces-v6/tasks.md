# Hardware Peripheral Interfaces - Tasks

## Phase 1: Foundation (kernel HAL + crate scaffolding)

- [x] T1: Add new HalError variants (NackReceived, ArbitrationLost, FramingError, ParityError, Overrun) to kernel/src/hal.rs with Display impls and tests
- [x] T2: Add I2cController HAL trait + config types (I2cSpeed, I2cAddressMode, I2cConfig) to kernel/src/hal.rs
- [x] T3: Add SpiController HAL trait + config types (SpiMode, SpiBitOrder, SpiConfig) to kernel/src/hal.rs
- [x] T4: Add GpioController HAL trait + config types (GpioPinMode, GpioPull, GpioInterruptEdge, GpioConfig) to kernel/src/hal.rs
- [x] T5: Add UartController HAL trait + config types (UartParity, UartStopBits, UartFlowControl, UartRxMode, UartConfig, UartIrqSource) to kernel/src/hal.rs
- [x] T6: Add CsiReceiver HAL trait + config types (CsiPixelFormat, CsiConfig, CsiFrame, CsiIrqSource) to kernel/src/hal.rs
- [x] T7: Add I2sController HAL trait + config types (I2sMode, I2sRole, I2sBitDepth, I2sConfig, AudioBuffer, I2sIrqSource) to kernel/src/hal.rs
- [x] T8: Create peripheral/Cargo.toml with feature flags (i2c, spi, gpio, uart, camera-csi, audio-i2s, convenience bundles), all default OFF
- [x] T9: Create peripheral/src/lib.rs with feature-gated module declarations
- [x] T10: Add peripheral crate to root Cargo.toml workspace members
- [x] T11: Add PeripheralCapability enum (I2c, Spi, Gpio, Camera, Audio, Uart) to security crate
- [x] T12: Extend device syscall stubs in kernel/src/syscall/device.rs with peripheral ioctl command constants

## Phase 2: I2C Drivers

- [x] T13: Implement peripheral/src/i2c/mod.rs — I2cController re-export, shared helpers (bus recovery, address validation)
- [x] T14: Implement peripheral/src/i2c/arm_mmio.rs — ARM Designware-compatible I2C driver with MMIO access
- [x] T15: Implement peripheral/src/i2c/riscv_mmio.rs — RISC-V SiFive/OpenCores I2C driver
- [x] T16: Implement peripheral/src/i2c/axi_iic.rs — Xilinx AXI IIC FPGA driver (wraps FpgaFabric trait)
- [x] T17: Implement peripheral/src/i2c/bitbang.rs — GPIO bitbang I2C fallback (requires gpio feature, Standard mode only)
- [x] T18: Write comprehensive unit tests for all I2C drivers (mock MMIO, NACK handling, arbitration, clock stretching, bus recovery)

## Phase 3: SPI Drivers

- [x] T19: Implement peripheral/src/spi/mod.rs — SpiController re-export, CS management helpers, clock divider calculation
- [x] T20: Implement peripheral/src/spi/arm_mmio.rs — ARM PL022/Designware SSI SPI driver
- [x] T21: Implement peripheral/src/spi/riscv_mmio.rs — RISC-V SiFive SPI driver
- [x] T22: Implement peripheral/src/spi/axi_quad_spi.rs — Xilinx AXI Quad SPI FPGA driver (wraps FpgaFabric, DMA support)
- [x] T23: Write comprehensive unit tests for all SPI drivers (all 4 modes, CS assertion, DMA, clock divider edge cases)

## Phase 4: GPIO Drivers

- [x] T24: Implement peripheral/src/gpio/mod.rs — GpioController re-export, shared interrupt dispatch helpers
- [x] T25: Implement peripheral/src/gpio/arm_pl061.rs — ARM PL061 GPIO driver with atomic address-masked data access
- [x] T26: Implement peripheral/src/gpio/riscv_mmio.rs — RISC-V SiFive GPIO driver with PLIC integration
- [x] T27: Implement peripheral/src/gpio/axi_gpio.rs — Xilinx AXI GPIO FPGA driver (dual-channel)
- [x] T28: Write comprehensive unit tests for all GPIO drivers (pin modes, atomic set_mask, interrupt edge detection, debounce)

## Phase 5: UART Drivers

- [x] T29: Implement peripheral/src/uart/mod.rs — UartController re-export, baud rate divisor calculation helpers
- [x] T30: Implement peripheral/src/uart/pl011.rs — ARM PL011 full-featured UART driver (interrupt + DMA RX, flow control)
- [x] T31: Implement peripheral/src/uart/ns16550a.rs — NS16550A-compatible UART driver
- [x] T32: Implement peripheral/src/uart/sifive.rs — SiFive UART driver with PLIC integration
- [x] T33: Implement peripheral/src/uart/axi_uart_lite.rs — Xilinx AXI UART Lite FPGA driver
- [x] T34: Implement peripheral/src/uart/nmea.rs — Optional NMEA 0183 sentence parser (checksum validation, coordinate extraction)
- [x] T35: Write comprehensive unit tests for all UART drivers (baud rates, framing errors, parity, overrun, flow control, DMA ring buffer, NMEA parser)

## Phase 6: Camera CSI Subsystem

- [x] T36: Implement peripheral/src/camera/mod.rs — CsiReceiver re-export, sensor detection logic
- [x] T37: Implement peripheral/src/camera/sensor.rs — I2C sensor auto-detection (scan known addresses, read chip ID)
- [x] T38: Implement peripheral/src/camera/ov5640.rs — OV5640 register tables (init, resolution, pixel format, exposure, test pattern)
- [x] T39: Implement peripheral/src/camera/imx219.rs — IMX219 register tables
- [x] T40: Implement peripheral/src/camera/imx477.rs — IMX477 register tables
- [x] T41: Implement peripheral/src/camera/tegra_vi.rs — NVIDIA Jetson Tegra VI/CSI receiver driver
- [x] T42: Implement peripheral/src/camera/broadcom_unicam.rs — Raspberry Pi Unicam CSI receiver driver
- [x] T43: Implement peripheral/src/camera/fpga_csi.rs — FPGA CSI receiver IP driver (wraps FpgaFabric, AXI DMA)
- [x] T44: Implement peripheral/src/camera/preprocess.rs — Optional YUV→RGB conversion, resize, float32 normalization for ONNX
- [x] T45: Write comprehensive unit tests for camera subsystem (sensor detection, register table validation, frame buffer lifecycle, overflow, preprocessing)

## Phase 7: Audio I2S Subsystem

- [x] T46: Implement peripheral/src/audio/mod.rs — I2sController re-export, codec detection logic
- [x] T47: Implement peripheral/src/audio/codec.rs — I2C codec auto-detection (scan known addresses, read chip ID)
- [x] T48: Implement peripheral/src/audio/tlv320aic3x.rs — TLV320AIC3x register tables (init, sample rate, input source, gain)
- [x] T49: Implement peripheral/src/audio/wm8960.rs — WM8960 register tables
- [x] T50: Implement peripheral/src/audio/es8388.rs — ES8388 register tables
- [x] T51: Implement peripheral/src/audio/arm_i2s.rs — ARM MMIO I2S driver (Tegra I2S, BCM PCM) with DMA ring buffer
- [x] T52: Implement peripheral/src/audio/riscv_i2s.rs — RISC-V MMIO I2S driver
- [x] T53: Implement peripheral/src/audio/fpga_i2s.rs — FPGA I2S IP driver (wraps FpgaFabric, AXI DMA)
- [x] T54: Implement peripheral/src/audio/preprocess.rs — Optional PCM→float32 normalization, windowing, mel spectrogram for ONNX
- [x] T55: Write comprehensive unit tests for audio subsystem (codec detection, register table validation, ring buffer lifecycle, overrun, preprocessing)

## Phase 8: Integration and Verification

- [x] T56: Implement DTB-based peripheral discovery — scan DTB for all peripheral compatible strings, instantiate drivers, register with device manager
- [x] T57: Wire peripheral capability checks into device syscall handlers (sys_dev_open checks CAP_I2C/SPI/GPIO/CAMERA/AUDIO/UART)
- [x] T58: Implement device ioctl dispatch for all peripheral types (I2C_WRITE, SPI_TRANSFER, GPIO_READ, UART_READ, CSI_DEQUEUE_FRAME, I2S_DEQUEUE_BUFFER, etc.)
- [x] T59: Write TLA+ formal model for I2C multi-master arbitration
- [x] T60: Write TLA+ formal model for SPI clock/data phase and CS ordering
- [x] T61: Write TLA+ formal model for GPIO interrupt handling and atomic set_mask
- [x] T62: Write TLA+ formal model for UART RX FIFO overflow and flow control
- [x] T63: Write TLA+ formal model for CSI frame buffer lifecycle (no use-after-enqueue)
- [x] T64: Write TLA+ formal model for I2S ring buffer (gapless capture, overrun detection)
- [x] T65: End-to-end integration test: I2C sensor read via syscall with capability check
- [x] T66: End-to-end integration test: SPI radar data read via DMA with capability check
- [x] T67: End-to-end integration test: GPIO interrupt triggers inference pipeline
- [x] T68: End-to-end integration test: UART NMEA parse from GPS receiver
- [x] T69: End-to-end integration test: CSI camera frame → ONNX MobileNetV2 inference (zero-copy)
- [x] T70: End-to-end integration test: I2S audio buffer → ONNX Whisper-tiny inference (zero-copy)
- [x] T71: Verify all features compile to zero when disabled (binary size comparison)
- [x] T72: Run cargo clippy and cargo fmt on all new code
- [x] T73: Update CLAUDE.md with new peripheral crate documentation and feature flags
- [x] T74: Commit and push to branch
