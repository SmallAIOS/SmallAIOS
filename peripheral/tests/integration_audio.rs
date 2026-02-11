// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the I2S audio peripheral subsystem.
//!
//! Tests exercise a mock I2S controller implementing the
//! `smallaios_kernel::hal::I2sController` trait, covering configuration,
//! capture start/stop, buffer dequeue/enqueue ring buffer lifecycle,
//! buffer address queries, sample rate verification, and interrupt
//! handling.

#![cfg(feature = "audio-i2s")]

extern crate alloc;
use alloc::collections::VecDeque;

use smallaios_kernel::hal::{
    AudioBuffer, HalError, I2sBitDepth, I2sConfig, I2sController, I2sIrqSource, I2sMode, I2sRole,
};
use smallaios_peripheral::PeripheralType;

// ---------------------------------------------------------------------------
// Mock I2S controller
// ---------------------------------------------------------------------------

/// Mock I2S controller for testing.
struct MockI2s {
    initialized: bool,
    capturing: bool,
    config: Option<I2sConfig>,
    /// Simulated DMA buffer addresses.
    buffer_addrs: [u64; 4],
    /// Buffer size in bytes.
    buf_size: usize,
    /// Completed audio buffers ready for dequeue.
    completed_buffers: VecDeque<AudioBuffer>,
    /// Buffers returned by the application (available for capture).
    available_buffers: VecDeque<u8>,
    /// Buffer sequence counter (for simulating captures).
    buffer_seq: u32,
    /// Pending IRQ source.
    pending_irq: Option<I2sIrqSource>,
}

impl MockI2s {
    fn new() -> Self {
        Self {
            initialized: false,
            capturing: false,
            config: None,
            buffer_addrs: [0; 4],
            buf_size: 0,
            completed_buffers: VecDeque::new(),
            available_buffers: VecDeque::new(),
            buffer_seq: 0,
            pending_irq: None,
        }
    }

    /// Simulate a DMA buffer capture completing on the given buffer index.
    fn simulate_buffer_complete(&mut self, buffer_index: u8) {
        let config = self.config.unwrap();
        let bytes_per_sample = match config.bit_depth {
            I2sBitDepth::Bits16 => 2u32,
            I2sBitDepth::Bits24 => 3,
            I2sBitDepth::Bits32 => 4,
        };
        let sample_count = config.buffer_frames * config.channels as u32;
        let bytes_used = sample_count * bytes_per_sample;

        let buf = AudioBuffer {
            buffer_index,
            timestamp_us: self.buffer_seq as u64 * 64_000, // ~64ms per buffer at 16kHz/1024 frames
            sample_count,
            bytes_used,
        };
        self.buffer_seq += 1;
        self.completed_buffers.push_back(buf);
        self.pending_irq = Some(I2sIrqSource::BufferComplete(buffer_index));
    }
}

impl I2sController for MockI2s {
    fn init(&mut self, config: I2sConfig) -> Result<(), HalError> {
        if config.num_buffers < 2 || config.num_buffers > 4 {
            return Err(HalError::InvalidConfig);
        }
        if config.channels == 0 || config.channels > 8 {
            return Err(HalError::InvalidConfig);
        }

        let bytes_per_sample = match config.bit_depth {
            I2sBitDepth::Bits16 => 2usize,
            I2sBitDepth::Bits24 => 3,
            I2sBitDepth::Bits32 => 4,
        };
        self.buf_size = config.buffer_frames as usize * config.channels as usize * bytes_per_sample;

        // Assign buffer addresses.
        let base = 0xA000_0000u64;
        for i in 0..config.num_buffers as usize {
            self.buffer_addrs[i] = base + (i as u64 * self.buf_size as u64);
            self.available_buffers.push_back(i as u8);
        }

        self.config = Some(config);
        self.initialized = true;
        Ok(())
    }

    fn start_capture(&mut self) -> Result<(), HalError> {
        if !self.initialized {
            return Err(HalError::InitFailed);
        }
        self.capturing = true;
        Ok(())
    }

    fn stop_capture(&mut self) -> Result<(), HalError> {
        self.capturing = false;
        Ok(())
    }

    fn dequeue_buffer(&mut self) -> Result<AudioBuffer, HalError> {
        if let Some(buf) = self.completed_buffers.pop_front() {
            Ok(buf)
        } else {
            Err(HalError::RxEmpty)
        }
    }

    fn enqueue_buffer(&mut self, buffer_index: u8) -> Result<(), HalError> {
        let num_buffers = self.config.map(|c| c.num_buffers).unwrap_or(0);
        if buffer_index >= num_buffers {
            return Err(HalError::OutOfRange);
        }
        self.available_buffers.push_back(buffer_index);
        Ok(())
    }

    fn buffer_address(&self, buffer_index: u8) -> Result<u64, HalError> {
        let num_buffers = self.config.map(|c| c.num_buffers).unwrap_or(0);
        if buffer_index >= num_buffers {
            return Err(HalError::OutOfRange);
        }
        Ok(self.buffer_addrs[buffer_index as usize])
    }

    fn buffer_size(&self) -> usize {
        self.buf_size
    }

    fn irq_handler(&mut self) -> Result<I2sIrqSource, HalError> {
        if let Some(src) = self.pending_irq.take() {
            Ok(src)
        } else {
            Err(HalError::InterruptError)
        }
    }

    fn reset(&mut self) -> Result<(), HalError> {
        self.initialized = false;
        self.capturing = false;
        self.config = None;
        self.buffer_addrs = [0; 4];
        self.buf_size = 0;
        self.completed_buffers.clear();
        self.available_buffers.clear();
        self.buffer_seq = 0;
        self.pending_irq = None;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_i2s_init_16khz_mono() {
    let mut i2s = MockI2s::new();
    let config = I2sConfig {
        mode: I2sMode::Standard,
        role: I2sRole::Master,
        sample_rate_hz: 16_000,
        bit_depth: I2sBitDepth::Bits16,
        channels: 1,
        buffer_frames: 1024,
        num_buffers: 2,
    };
    assert!(i2s.init(config).is_ok());
    assert!(i2s.initialized);
    // Buffer size = 1024 frames * 1 channel * 2 bytes = 2048.
    assert_eq!(i2s.buffer_size(), 2048);
}

#[test]
fn test_i2s_init_48khz_stereo() {
    let mut i2s = MockI2s::new();
    let config = I2sConfig {
        mode: I2sMode::Standard,
        role: I2sRole::Master,
        sample_rate_hz: 48_000,
        bit_depth: I2sBitDepth::Bits24,
        channels: 2,
        buffer_frames: 512,
        num_buffers: 3,
    };
    assert!(i2s.init(config).is_ok());
    // Buffer size = 512 * 2 * 3 = 3072 bytes.
    assert_eq!(i2s.buffer_size(), 3072);
}

#[test]
fn test_i2s_start_stop_capture() {
    let mut i2s = MockI2s::new();
    i2s.init(I2sConfig {
        mode: I2sMode::Standard,
        role: I2sRole::Master,
        sample_rate_hz: 16_000,
        bit_depth: I2sBitDepth::Bits16,
        channels: 1,
        buffer_frames: 1024,
        num_buffers: 2,
    })
    .unwrap();

    assert!(!i2s.capturing);
    i2s.start_capture().unwrap();
    assert!(i2s.capturing);
    i2s.stop_capture().unwrap();
    assert!(!i2s.capturing);
}

#[test]
fn test_i2s_dequeue_buffer() {
    let mut i2s = MockI2s::new();
    i2s.init(I2sConfig {
        mode: I2sMode::Standard,
        role: I2sRole::Master,
        sample_rate_hz: 16_000,
        bit_depth: I2sBitDepth::Bits16,
        channels: 1,
        buffer_frames: 1024,
        num_buffers: 2,
    })
    .unwrap();
    i2s.start_capture().unwrap();

    // No buffers ready yet.
    assert_eq!(i2s.dequeue_buffer(), Err(HalError::RxEmpty));

    // Simulate buffer 0 completing.
    i2s.simulate_buffer_complete(0);
    let buf = i2s.dequeue_buffer().unwrap();
    assert_eq!(buf.buffer_index, 0);
    assert_eq!(buf.sample_count, 1024); // 1024 frames * 1 channel
    assert_eq!(buf.bytes_used, 2048); // 1024 samples * 2 bytes
}

#[test]
fn test_i2s_ring_buffer_lifecycle() {
    let mut i2s = MockI2s::new();
    i2s.init(I2sConfig {
        mode: I2sMode::Standard,
        role: I2sRole::Master,
        sample_rate_hz: 16_000,
        bit_depth: I2sBitDepth::Bits16,
        channels: 1,
        buffer_frames: 256,
        num_buffers: 3,
    })
    .unwrap();
    i2s.start_capture().unwrap();

    // Simulate completing all 3 buffers.
    i2s.simulate_buffer_complete(0);
    i2s.simulate_buffer_complete(1);
    i2s.simulate_buffer_complete(2);

    // Dequeue all 3.
    let b0 = i2s.dequeue_buffer().unwrap();
    let b1 = i2s.dequeue_buffer().unwrap();
    let b2 = i2s.dequeue_buffer().unwrap();

    // Verify sequential timestamps.
    assert!(b1.timestamp_us > b0.timestamp_us);
    assert!(b2.timestamp_us > b1.timestamp_us);

    // Return buffers.
    i2s.enqueue_buffer(b0.buffer_index).unwrap();
    i2s.enqueue_buffer(b1.buffer_index).unwrap();
    i2s.enqueue_buffer(b2.buffer_index).unwrap();

    // Out of range.
    assert_eq!(i2s.enqueue_buffer(3), Err(HalError::OutOfRange));

    // No more completed.
    assert_eq!(i2s.dequeue_buffer(), Err(HalError::RxEmpty));
}

#[test]
fn test_i2s_buffer_addresses() {
    let mut i2s = MockI2s::new();
    i2s.init(I2sConfig {
        mode: I2sMode::Standard,
        role: I2sRole::Master,
        sample_rate_hz: 16_000,
        bit_depth: I2sBitDepth::Bits16,
        channels: 1,
        buffer_frames: 1024,
        num_buffers: 3,
    })
    .unwrap();

    let buf_size = i2s.buffer_size() as u64;
    let addr0 = i2s.buffer_address(0).unwrap();
    let addr1 = i2s.buffer_address(1).unwrap();
    let addr2 = i2s.buffer_address(2).unwrap();

    assert_eq!(addr0, 0xA000_0000);
    assert_eq!(addr1, 0xA000_0000 + buf_size);
    assert_eq!(addr2, 0xA000_0000 + 2 * buf_size);

    // Out of range.
    assert_eq!(i2s.buffer_address(3), Err(HalError::OutOfRange));
}

#[test]
fn test_i2s_irq_handler() {
    let mut i2s = MockI2s::new();
    i2s.init(I2sConfig {
        mode: I2sMode::Standard,
        role: I2sRole::Master,
        sample_rate_hz: 16_000,
        bit_depth: I2sBitDepth::Bits16,
        channels: 1,
        buffer_frames: 1024,
        num_buffers: 2,
    })
    .unwrap();
    i2s.start_capture().unwrap();

    i2s.simulate_buffer_complete(0);
    let irq = i2s.irq_handler().unwrap();
    assert_eq!(irq, I2sIrqSource::BufferComplete(0));

    // No more pending.
    assert_eq!(i2s.irq_handler(), Err(HalError::InterruptError));
}

#[test]
fn test_i2s_tdm_mode_8_channel() {
    let mut i2s = MockI2s::new();
    let config = I2sConfig {
        mode: I2sMode::Tdm { num_slots: 8 },
        role: I2sRole::Master,
        sample_rate_hz: 48_000,
        bit_depth: I2sBitDepth::Bits32,
        channels: 8,
        buffer_frames: 128,
        num_buffers: 4,
    };
    assert!(i2s.init(config).is_ok());
    // Buffer size = 128 frames * 8 channels * 4 bytes = 4096.
    assert_eq!(i2s.buffer_size(), 4096);

    let cfg = i2s.config.unwrap();
    match cfg.mode {
        I2sMode::Tdm { num_slots } => assert_eq!(num_slots, 8),
        _ => panic!("expected TDM mode"),
    }
}

#[test]
fn test_i2s_invalid_num_buffers() {
    let mut i2s = MockI2s::new();
    let config = I2sConfig {
        mode: I2sMode::Standard,
        role: I2sRole::Master,
        sample_rate_hz: 16_000,
        bit_depth: I2sBitDepth::Bits16,
        channels: 1,
        buffer_frames: 1024,
        num_buffers: 5, // Max is 4.
    };
    assert_eq!(i2s.init(config), Err(HalError::InvalidConfig));

    let config_zero = I2sConfig {
        mode: I2sMode::Standard,
        role: I2sRole::Master,
        sample_rate_hz: 16_000,
        bit_depth: I2sBitDepth::Bits16,
        channels: 1,
        buffer_frames: 1024,
        num_buffers: 0,
    };
    assert_eq!(i2s.init(config_zero), Err(HalError::InvalidConfig));
}

#[test]
fn test_i2s_start_without_init() {
    let mut i2s = MockI2s::new();
    assert_eq!(i2s.start_capture(), Err(HalError::InitFailed));
}

#[test]
fn test_i2s_reset() {
    let mut i2s = MockI2s::new();
    i2s.init(I2sConfig {
        mode: I2sMode::Standard,
        role: I2sRole::Master,
        sample_rate_hz: 16_000,
        bit_depth: I2sBitDepth::Bits16,
        channels: 1,
        buffer_frames: 1024,
        num_buffers: 2,
    })
    .unwrap();
    i2s.start_capture().unwrap();
    i2s.simulate_buffer_complete(0);

    i2s.reset().unwrap();
    assert!(!i2s.initialized);
    assert!(!i2s.capturing);
    assert_eq!(i2s.buffer_size(), 0);
    assert_eq!(i2s.dequeue_buffer(), Err(HalError::RxEmpty));
}

#[test]
fn test_i2s_slave_mode() {
    let mut i2s = MockI2s::new();
    let config = I2sConfig {
        mode: I2sMode::Standard,
        role: I2sRole::Slave,
        sample_rate_hz: 44_100,
        bit_depth: I2sBitDepth::Bits16,
        channels: 2,
        buffer_frames: 512,
        num_buffers: 2,
    };
    assert!(i2s.init(config).is_ok());
    let cfg = i2s.config.unwrap();
    assert_eq!(cfg.role, I2sRole::Slave);
    assert_eq!(cfg.sample_rate_hz, 44_100);
}

#[test]
fn test_i2s_peripheral_type() {
    assert_eq!(PeripheralType::from_u8(5), Some(PeripheralType::Audio));
    assert_eq!(PeripheralType::Audio as u8, 5);
}
