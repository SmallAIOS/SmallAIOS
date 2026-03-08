# Audio I2S Specification

## Overview

I2S/TDM audio interface for SmallAIOS. Provides a HAL trait for I2S/TDM controller access, I2C-based audio codec configuration, DMA ring-buffer streaming, and a PCM capture API that feeds audio samples directly into the ONNX input pipeline. Gated by the `audio-i2s` feature flag (default OFF). Depends on the `i2c` feature for codec register access. All audio operations require the `CAP_AUDIO` capability.

Primary use cases: live audio inference (Whisper-tiny for speech recognition, keyword spotting, anomaly detection) on embedded platforms with I2S-connected audio codecs.

Reference: Philips I2S bus specification (1996), TDM (Time-Division Multiplexing) industry conventions.

## Feature Gate

- **Cargo feature:** `audio-i2s` (default OFF)
- **Dependency:** requires `i2c` feature to also be enabled (for codec configuration over I2C)
- WHEN the `audio-i2s` feature is disabled, THEN zero I2S/audio code SHALL be compiled into the binary
- WHEN `audio-i2s` is enabled but `i2c` is not, THEN compilation SHALL fail with a clear error message

## Audio Codec Configuration (I2C)

### REQ-I2S-001: Codec Register Access

The system SHALL configure audio codecs via I2C using the `I2cController` trait:
- Codecs are addressed at their I2C slave address (e.g., TLV320AIC3x = 0x18, WM8960 = 0x1A, ES8388 = 0x10)
- Register access uses 8-bit register addresses and 8-bit values (codec-dependent)

### REQ-I2S-002: Codec Configuration Tables

The system SHALL include static configuration tables for supported codecs:

| Codec | Channels | Max Sample Rate | Bit Depth | I2C Addr | Use Case |
|-------|----------|----------------|-----------|----------|----------|
| TLV320AIC3x | 2 (stereo) | 96 kHz | 16/24/32 | 0x18 | Industrial, Jetson |
| WM8960 | 2 (stereo) | 48 kHz | 16/24/32 | 0x1A | Raspberry Pi, general |
| ES8388 | 2 (stereo) | 96 kHz | 16/24/32 | 0x10 | Low-cost edge devices |

Each table SHALL include register sequences for:
- Codec power-on initialization and PLL configuration
- Sample rate and bit depth selection
- Input source selection (line-in, microphone with preamp gain)
- ADC enable (capture path)
- Digital volume/gain control

### REQ-I2S-003: Codec Auto-Detection

WHEN the audio subsystem initializes, THEN it SHALL:
1. Scan known I2C addresses for supported codecs
2. Read the codec chip ID / revision register to identify the exact model
3. Return `HalError::NotSupported` if no supported codec is found

## I2S/TDM Controller HAL

### REQ-I2S-010: I2sConfig Structure

```rust
pub enum I2sMode {
    /// Standard I2S (Philips) — data delayed by 1 BCLK after WS transition.
    Standard,
    /// Left-justified — data starts immediately on WS transition.
    LeftJustified,
    /// Right-justified — data is right-aligned within the WS slot.
    RightJustified,
    /// TDM — multiple channels time-multiplexed on a single data line.
    Tdm { num_slots: u8 },
}

pub enum I2sRole {
    /// I2S controller generates BCLK and WS (LRCK).
    Master,
    /// I2S controller receives BCLK and WS from external source.
    Slave,
}

pub enum I2sBitDepth {
    Bits16,
    Bits24,
    Bits32,
}

pub struct I2sConfig {
    pub mode: I2sMode,
    pub role: I2sRole,
    pub sample_rate_hz: u32,    // 8000, 16000, 22050, 44100, 48000, 96000, 192000
    pub bit_depth: I2sBitDepth,
    pub channels: u8,           // 1 (mono), 2 (stereo), up to 8 (TDM)
    pub buffer_frames: u32,     // Number of frames per DMA buffer
    pub num_buffers: u8,        // Number of ring buffers (2-4)
}
```

### REQ-I2S-011: I2sController Trait

```rust
pub trait I2sController {
    /// Initialize the I2S controller with the given configuration.
    /// Allocates DMA ring buffers for audio streaming.
    fn init(&mut self, config: I2sConfig) -> Result<(), HalError>;

    /// Start audio capture (continuous DMA streaming).
    fn start_capture(&mut self) -> Result<(), HalError>;

    /// Stop audio capture.
    fn stop_capture(&mut self) -> Result<(), HalError>;

    /// Dequeue a completed audio buffer. Returns buffer index and sample count.
    /// Returns HalError::RxEmpty if no buffer is ready.
    fn dequeue_buffer(&mut self) -> Result<AudioBuffer, HalError>;

    /// Return a buffer to the capture ring after processing.
    fn enqueue_buffer(&mut self, buffer_index: u8) -> Result<(), HalError>;

    /// Get the physical address of an audio buffer (for zero-copy ONNX input).
    fn buffer_address(&self, buffer_index: u8) -> Result<u64, HalError>;

    /// Get the byte size of a single audio buffer.
    fn buffer_size(&self) -> usize;

    /// Handle an I2S interrupt (buffer complete, overrun, underrun).
    fn irq_handler(&mut self) -> Result<I2sIrqSource, HalError>;

    /// Reset the I2S controller and release all buffers.
    fn reset(&mut self) -> Result<(), HalError>;
}

pub struct AudioBuffer {
    pub buffer_index: u8,
    pub timestamp_us: u64,      // Timestamp of first sample in buffer
    pub sample_count: u32,      // Number of samples (frames × channels)
    pub bytes_used: u32,        // Actual bytes in buffer
}

pub enum I2sIrqSource {
    BufferComplete(u8),         // Buffer index of completed buffer
    Overrun,                    // DMA overrun (data lost)
    Underrun,                   // DMA underrun (for playback, if ever added)
    ClockError,                 // BCLK/WS mismatch or loss
}
```

### REQ-I2S-012: DMA Ring Buffer Streaming

- Audio buffers SHALL be allocated from DMA-capable physically contiguous memory
- The system SHALL use a ring of 2-4 buffers for gapless capture
- DMA SHALL automatically cycle through buffers, raising an interrupt on each completion
- WHEN all buffers are full (not returned via `enqueue_buffer()`), THEN the oldest buffer SHALL be overwritten and an Overrun event logged
- Buffer sizes SHALL be aligned to DMA requirements (typically 32-byte or 64-byte alignment)

### REQ-I2S-013: Sample Rate and Format

Supported sample rates:
- 8 kHz (telephony, keyword spotting)
- 16 kHz (Whisper-tiny default, speech recognition)
- 22050 Hz (low-quality audio ML)
- 44100 Hz (CD quality)
- 48 kHz (professional audio, standard codec default)
- 96 kHz (high-resolution)
- 192 kHz (ultrasonic, vibration analysis)

WHEN a sample rate is not achievable by the hardware PLL/divider, THEN the driver SHALL select the nearest achievable rate and report the actual rate.

### REQ-I2S-014: ONNX Input Pipeline Integration

- Audio buffers SHALL be directly usable as ONNX model inputs without memory copies
- The `buffer_address()` method provides the physical address for the ONNX runtime's input tensor
- For models expecting specific formats (e.g., float32 normalized, mel spectrogram), optional preprocessing:
  - int16/int24/int32 PCM → float32 normalization (-1.0 to 1.0)
  - Windowing (Hann, Hamming) for spectrogram computation
  - FFT → mel filterbank → log-mel spectrogram (for Whisper-style models)
- Preprocessing SHALL be optional and configurable per-model

## Platform Drivers

### REQ-I2S-020: ARM MMIO I2S Driver

WHEN running on ARM64 platforms with memory-mapped I2S controllers, THEN the driver SHALL:
- Support Tegra I2S (Jetson), Broadcom PCM/I2S (RPi), generic I2S MMIO
- Configure DMA for continuous ring-buffer capture
- Support DTB-based discovery

### REQ-I2S-021: RISC-V MMIO I2S Driver

WHEN running on RISC-V platforms with I2S peripherals, THEN the driver SHALL:
- Support I2S controllers accessible via MMIO
- Use PLIC for interrupt delivery
- Support DTB-based discovery

### REQ-I2S-022: FPGA I2S IP Driver

WHEN running on FPGA platforms with I2S receiver soft-IP, THEN the driver SHALL:
- Access I2S IP registers via the `FpgaFabric` trait
- Use AXI DMA for buffer fill
- Support configurable sample rate and channel count via IP parameters

## Capability Integration

### REQ-I2S-030: Capability Gating

- A new capability type `CAP_AUDIO` SHALL be defined in the `security` crate
- WHEN a process opens an audio device, THEN the kernel SHALL verify `CAP_AUDIO`
- Audio capture is privacy-sensitive — capability SHALL NOT be granted by default

### REQ-I2S-031: Syscall Interface

Audio devices SHALL be accessible via the existing device syscall interface:
- `sys_dev_enumerate()` — lists discovered audio devices (codec + I2S controller)
- `sys_dev_open(id)` — opens an audio capture device (checks `CAP_AUDIO`)
- `sys_dev_ioctl(handle, I2S_SET_CONFIG, &config)` — configure sample rate/format
- `sys_dev_ioctl(handle, I2S_START_CAPTURE, 0)` — start streaming
- `sys_dev_ioctl(handle, I2S_STOP_CAPTURE, 0)` — stop streaming
- `sys_dev_ioctl(handle, I2S_DEQUEUE_BUFFER, &buf)` — get completed buffer
- `sys_dev_ioctl(handle, I2S_ENQUEUE_BUFFER, index)` — return buffer
- `sys_dev_ioctl(handle, I2S_GET_BUFFER_ADDR, index)` — get physical addr for zero-copy
- `sys_dev_close(handle)` — stop capture, release buffers, close device

## Safety and Verification

### REQ-I2S-040: Test Coverage

- Unit tests with mock codec I2C responses and I2S sample generation
- Codec configuration table verification against datasheets
- Ring buffer lifecycle tests (alloc, fill, dequeue, enqueue, overrun)
- MC/DC coverage on sample rate configuration and error paths

### REQ-I2S-041: Deterministic Audio Capture

- Audio capture SHALL maintain gapless recording with no sample loss under normal operation
- Buffer overrun events SHALL be counted and reported
- Timestamps SHALL use the kernel monotonic clock for correlation with inference results
- WCET for `dequeue_buffer()`: O(1) — direct buffer pointer return
