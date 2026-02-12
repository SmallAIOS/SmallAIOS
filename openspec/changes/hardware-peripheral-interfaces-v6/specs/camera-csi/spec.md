# Camera CSI Specification

## Overview

MIPI CSI-2 camera interface for SmallAIOS. Provides a HAL trait for CSI-2 receiver control, I2C-based image sensor configuration, frame buffer management via DMA, and a V4L2-compatible capture API that feeds frames directly into the ONNX input pipeline. Gated by the `camera-csi` feature flag (default OFF). Depends on the `i2c` feature for sensor register access. All camera operations require the `CAP_CAMERA` capability.

Primary use cases: live vision inference (MobileNetV2, YOLO, object detection/classification) on embedded platforms with MIPI CSI-2 camera connectors (Jetson, Raspberry Pi, FPGA SoCs with CSI IP).

Reference: MIPI Alliance CSI-2 specification v3.0, MIPI D-PHY specification v2.5.

## Feature Gate

- **Cargo feature:** `camera-csi` (default OFF)
- **Dependency:** requires `i2c` feature to also be enabled (for sensor configuration over I2C)
- WHEN the `camera-csi` feature is disabled, THEN zero CSI code SHALL be compiled into the binary
- WHEN `camera-csi` is enabled but `i2c` is not, THEN compilation SHALL fail with a clear error message

## Image Sensor Configuration (I2C)

### REQ-CSI-001: Sensor Register Access

The system SHALL configure image sensors via I2C using the `I2cController` trait:
- Sensors are addressed at their I2C slave address (e.g., OV5640 = 0x3C, IMX219 = 0x10, IMX477 = 0x1A)
- Register access uses 16-bit register addresses and 8-bit or 16-bit values (sensor-dependent)
- The `write_read()` I2C method SHALL be used for register reads (write register address, repeated-start, read value)

### REQ-CSI-002: Sensor Configuration Tables

The system SHALL include static configuration tables for supported sensors:

| Sensor | Resolution | Max FPS | Interface | I2C Addr |
|--------|-----------|---------|-----------|----------|
| OV5640 | 2592x1944 (5MP) | 15 fps (full), 30 fps (1080p), 60 fps (720p) | 1-2 lanes | 0x3C |
| IMX219 | 3280x2464 (8MP) | 21 fps (full), 30 fps (1080p), 60 fps (720p) | 2 lanes | 0x10 |
| IMX477 | 4056x3040 (12MP) | 10 fps (full), 40 fps (1080p), 60 fps (720p) | 2-4 lanes | 0x1A |

Each table SHALL include register sequences for:
- Sensor power-on initialization
- Resolution/frame rate configuration
- Pixel format selection (RAW8, RAW10, RAW12, YUV422, RGB888)
- Exposure/gain control (manual mode for deterministic inference)
- Test pattern generation (for integration testing without physical sensor)

### REQ-CSI-003: Sensor Auto-Detection

WHEN the camera subsystem initializes, THEN it SHALL:
1. Scan known I2C addresses for supported sensors
2. Read the sensor chip ID register to identify the exact model
3. Return `HalError::NotSupported` if no supported sensor is found

## CSI-2 Receiver HAL

### REQ-CSI-010: CsiConfig Structure

```rust
pub enum CsiPixelFormat {
    Raw8,
    Raw10,
    Raw12,
    Yuv422,
    Rgb888,
}

pub struct CsiConfig {
    pub lanes: u8,              // Number of data lanes: 1, 2, or 4
    pub lane_speed_mbps: u16,   // Per-lane data rate (up to 1500 Mbps)
    pub width: u16,             // Frame width in pixels
    pub height: u16,            // Frame height in pixels
    pub pixel_format: CsiPixelFormat,
    pub fps: u8,                // Target frame rate
    pub num_buffers: u8,        // Number of frame buffers (double/triple buffering)
}
```

### REQ-CSI-011: CsiReceiver Trait

```rust
pub trait CsiReceiver {
    /// Initialize the CSI-2 receiver with the given configuration.
    /// Allocates frame buffers via DMA-capable memory.
    fn init(&mut self, config: CsiConfig) -> Result<(), HalError>;

    /// Start frame capture (continuous mode).
    /// Frames are written to DMA buffers in a round-robin.
    fn start_capture(&mut self) -> Result<(), HalError>;

    /// Stop frame capture.
    fn stop_capture(&mut self) -> Result<(), HalError>;

    /// Dequeue a completed frame buffer. Returns the buffer index and timestamp.
    /// Returns HalError::RxEmpty if no frame is ready.
    fn dequeue_frame(&mut self) -> Result<CsiFrame, HalError>;

    /// Return a frame buffer to the capture queue after processing.
    fn enqueue_frame(&mut self, buffer_index: u8) -> Result<(), HalError>;

    /// Get the physical address of a frame buffer (for zero-copy ONNX input).
    fn buffer_address(&self, buffer_index: u8) -> Result<u64, HalError>;

    /// Get the byte size of a single frame buffer.
    fn buffer_size(&self) -> usize;

    /// Handle a CSI interrupt (frame complete, error, overflow).
    fn irq_handler(&mut self) -> Result<CsiIrqSource, HalError>;

    /// Reset the CSI receiver and release all buffers.
    fn reset(&mut self) -> Result<(), HalError>;
}

pub struct CsiFrame {
    pub buffer_index: u8,
    pub timestamp_us: u64,
    pub sequence: u32,          // Frame sequence number
    pub bytes_used: u32,        // Actual bytes written (may differ for compressed formats)
}

pub enum CsiIrqSource {
    FrameComplete(u8),          // Buffer index of completed frame
    Overflow,                   // Receiver FIFO overflow (frame dropped)
    CrcError,                   // CSI-2 CRC mismatch
    EccError,                   // CSI-2 ECC error in packet header
}
```

### REQ-CSI-012: Frame Buffer Management

- Frame buffers SHALL be allocated from DMA-capable physically contiguous memory
- The system SHALL support double buffering (2 buffers) or triple buffering (3 buffers)
- WHEN all buffers are in-flight (not returned via `enqueue_frame()`), THEN the oldest buffer SHALL be overwritten and an overflow event logged
- Buffer physical addresses SHALL be accessible for zero-copy handoff to the ONNX runtime

### REQ-CSI-013: ONNX Input Pipeline Integration

- Camera frames SHALL be directly usable as ONNX model inputs without memory copies
- The `buffer_address()` method provides the physical address for the ONNX runtime's input tensor
- For models expecting specific input formats (e.g., float32 RGB normalized), a lightweight preprocessing step SHALL be available:
  - YUV422 → RGB888 conversion
  - Resize to model input dimensions
  - Normalize pixel values (0–255 → 0.0–1.0)
- Preprocessing SHALL be optional and configurable per-model

## Platform Drivers

### REQ-CSI-020: NVIDIA Jetson CSI Driver

WHEN running on NVIDIA Jetson platforms (Tegra), THEN the driver SHALL:
- Access the Tegra VI (Video Input) and CSI hardware blocks via MMIO
- Support 1-4 CSI data lanes
- Support ISP (Image Signal Processor) bypass mode for raw frame capture
- Integrate with the Jetson DMA engine for frame buffer fill

### REQ-CSI-021: Raspberry Pi CSI Driver

WHEN running on Raspberry Pi platforms (Broadcom), THEN the driver SHALL:
- Access the Unicam CSI-2 receiver via MMIO
- Support 2 CSI data lanes (RPi Camera connector)
- Use the Broadcom DMA controller for frame buffer fill

### REQ-CSI-022: FPGA CSI IP Driver

WHEN running on FPGA platforms with MIPI CSI-2 receiver IP, THEN the driver SHALL:
- Access the CSI receiver IP registers via the `FpgaFabric` trait
- Use AXI DMA for frame buffer fill
- Support configurable lane count and pixel format via IP parameters
- Support DTB-based discovery

## Capability Integration

### REQ-CSI-030: Capability Gating

- A new capability type `CAP_CAMERA` SHALL be defined in the `security` crate
- WHEN a process opens a camera device, THEN the kernel SHALL verify `CAP_CAMERA`
- Camera access is particularly security-sensitive — capability SHALL NOT be granted by default

### REQ-CSI-031: Syscall Interface

Camera devices SHALL be accessible via the existing device syscall interface:
- `sys_dev_enumerate()` — lists discovered camera devices (sensor + CSI receiver)
- `sys_dev_open(id)` — opens a camera (checks `CAP_CAMERA`)
- `sys_dev_ioctl(handle, CSI_SET_CONFIG, &config)` — configure resolution/format/fps
- `sys_dev_ioctl(handle, CSI_START_CAPTURE, 0)` — start continuous capture
- `sys_dev_ioctl(handle, CSI_STOP_CAPTURE, 0)` — stop capture
- `sys_dev_ioctl(handle, CSI_DEQUEUE_FRAME, &frame)` — get completed frame
- `sys_dev_ioctl(handle, CSI_ENQUEUE_FRAME, buffer_index)` — return buffer
- `sys_dev_ioctl(handle, CSI_GET_BUFFER_ADDR, buffer_index)` — get physical addr for zero-copy
- `sys_dev_close(handle)` — stop capture, release buffers, close device

## Safety and Verification

### REQ-CSI-040: Test Coverage

- Unit tests with mock sensor I2C responses and CSI frame generation
- Sensor configuration table verification (register sequences validated against datasheets)
- Frame buffer lifecycle tests (alloc, dequeue, enqueue, overflow)
- MC/DC coverage on error handling paths (CRC error, overflow, NACK on I2C)

### REQ-CSI-041: Deterministic Capture

- For safety-critical applications, the camera subsystem SHALL provide deterministic frame delivery
- Frame timestamps SHALL use the kernel monotonic clock
- Frame drops SHALL be logged and counted
- WCET for `dequeue_frame()`: O(1) — direct buffer pointer return, no allocation
