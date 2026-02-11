## Why

SmallAIOS runs AI inference on embedded, edge, and industrial platforms — many of which have cameras, microphones, IMUs, temperature sensors, and other peripherals connected via I2C, SPI, and GPIO. Today, SmallAIOS has no way to access these devices. The existing HAL covers safety-critical buses (CAN, ARINC, MIL-STD-1553, SpaceWire) and GPU accelerators, but the fundamental embedded peripheral buses — I2C, SPI, GPIO — are absent. Without them:

- **Camera input** is impossible. MIPI CSI-2 cameras (Jetson, RPi, industrial vision) need I2C for sensor configuration and a CSI/MIPI receiver for pixel data. USB cameras need a host controller. Vision inference (MobileNetV2, YOLO) cannot run on live frames.
- **Audio input** is impossible. I2S/TDM codecs (used in industrial and edge devices) need I2C for codec configuration and I2S for sample streaming. Audio inference (Whisper-tiny) cannot run on live audio.
- **Sensor fusion** is impossible. IMUs, temperature sensors, pressure sensors, and ADCs on I2C/SPI buses cannot be read.
- **GPIO control** is impossible. Inference-triggered actuation (alerts, relays, indicator LEDs) cannot happen.

These are required for real-world AI inference deployments, especially on Jetson, Raspberry Pi, FPGA SoCs, and industrial ARM/RISC-V platforms already targeted by SmallAIOS.

**Security concern:** Many deployments (datacenter, avionics, space) must NOT expose peripheral buses — a rogue I2C/GPIO driver is an attack surface. All peripheral interfaces must be individually disableable at compile time via Cargo feature flags, producing zero code for disabled interfaces.

## What Changes

- Add I2C controller HAL trait and platform drivers (ARM MMIO, RISC-V MMIO, FPGA AXI soft-IP, bitbang GPIO fallback)
- Add SPI controller HAL trait and platform drivers (ARM MMIO, RISC-V MMIO, FPGA AXI soft-IP)
- Add GPIO controller HAL trait and platform drivers (ARM MMIO, RISC-V MMIO, FPGA AXI)
- Add MIPI CSI-2 camera interface: I2C-based sensor configuration, CSI receiver HAL trait, frame buffer management, V4L2-style capture API for ONNX input pipeline
- Add I2S/TDM audio interface: I2C-based codec configuration, I2S/TDM HAL trait, ring buffer DMA streaming, PCM capture API for ONNX input pipeline
- Extend kernel HAL (`kernel/src/hal.rs`) with new traits: `I2cController`, `SpiController`, `GpioController`, `CsiReceiver`, `I2sController`
- Add compile-time feature flags: `i2c`, `spi`, `gpio`, `camera-csi`, `audio-i2s` — all default-off
- Integrate with existing capability system (`security` crate) — peripheral access requires explicit capabilities
- Extend device syscalls (`kernel/src/syscall/device.rs`) to open/control peripheral devices
- Add DTB-based peripheral discovery for I2C/SPI/GPIO controllers on ARM, RISC-V, and FPGA platforms

## Capabilities

### New Capabilities

- `peripheral-i2c`: I2C master controller — 7-bit and 10-bit addressing, standard (100 kHz) / fast (400 kHz) / fast-plus (1 MHz) modes, multi-byte read/write transactions, repeated start, clock stretching support. Platform drivers for ARM PL061/generic MMIO, RISC-V MMIO, Xilinx AXI IIC soft-IP, bitbang GPIO fallback.
- `peripheral-spi`: SPI master controller — modes 0-3, configurable clock (up to 50 MHz), chip-select management, full-duplex transfer, DMA support for bulk transfers. Platform drivers for ARM MMIO, RISC-V MMIO, Xilinx AXI Quad SPI soft-IP.
- `peripheral-gpio`: GPIO controller — input/output/alternate-function pin modes, pull-up/pull-down configuration, interrupt-on-edge (rising/falling/both), debounce, atomic pin set/clear. Platform drivers for ARM PL061 / generic MMIO, RISC-V MMIO, Xilinx AXI GPIO soft-IP.
- `camera-csi`: MIPI CSI-2 camera interface — I2C sensor configuration (register read/write), CSI-2 receiver (1-4 data lanes, up to 1.5 Gbps/lane), frame capture (RAW8/10/12, YUV422, RGB888), frame buffer allocation via DMA, V4L2-compatible ioctl subset for userspace/ONNX input pipeline integration, supported sensors: OV5640, IMX219, IMX477 (config tables).
- `audio-i2s`: I2S/TDM audio interface — I2C codec configuration, I2S master/slave mode, TDM multi-channel (up to 8 channels), sample rates 8-192 kHz, bit depths 16/24/32, DMA ring-buffer streaming, PCM capture API for ONNX input pipeline, supported codecs: TLV320AIC3x, WM8960, ES8388 (config tables).

### Modified Capabilities

- `05-device-hal`: Extend HAL trait set with `I2cController`, `SpiController`, `GpioController`, `CsiReceiver`, `I2sController`. Add new `HalError` variants as needed (`NackReceived`, `ArbitrationLost`, `FrameError`).
- `02-security-model`: Add capability types for peripheral access — `CAP_I2C`, `CAP_SPI`, `CAP_GPIO`, `CAP_CAMERA`, `CAP_AUDIO`. Peripheral open/read/write/ioctl requires matching capability.
- `07-container-interface`: Add build-time feature flag documentation. Add container config for peripheral passthrough.
- `10-hardware-platforms`: Document peripheral availability per platform (Jetson: I2C/SPI/GPIO/CSI native; RPi: I2C/SPI/GPIO/CSI native; Zynq: I2C/SPI/GPIO via FPGA fabric; PolarFire: I2C/SPI/GPIO via FPGA fabric).

## Impact

- **Rust workspace**: New `peripheral` crate (I2C, SPI, GPIO drivers), extend `kernel` crate (HAL traits, syscall handlers), extend `security` crate (new capability types), extend `onnx-rt` crate (camera/audio input providers)
- **Feature flags**: `i2c`, `spi`, `gpio`, `camera-csi`, `audio-i2s` — all default OFF (security-by-default). Enable per-deployment as needed. Zero binary size cost when disabled.
- **Build targets**: No new targets — uses existing ARM64, RISC-V, and FPGA platforms
- **Hardware dependencies**: I2C peripherals (sensors, camera modules, audio codecs), SPI peripherals (flash, ADCs), CSI camera modules (OV5640, IMX219), I2S audio codecs (TLV320AIC3x, WM8960)
- **Security**: All peripheral access gated by capability system. Feature flags provide compile-time elimination. No peripheral code in datacenter/avionics builds unless explicitly enabled.
- **Safety certification**: I2C/SPI/GPIO drivers subject to same DO-178C/ISO 26262 coverage requirements as bus protocol drivers
- **Formal verification**: TLA+ models for I2C arbitration, SPI protocol state machine, GPIO interrupt handling

## Non-Goals

- USB host controller stack (complex, separate change)
- Display/HDMI/DisplayPort output (not needed for inference)
- Bluetooth/WiFi wireless interfaces (separate change)
- Analog audio output / speaker drivers (inference input only)
- Video encoding / streaming output
- Touchscreen or HID input devices
