// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the CSI camera peripheral subsystem.
//!
//! Tests exercise a mock CSI-2 receiver implementing the
//! `smallaios_kernel::hal::CsiReceiver` trait, covering configuration,
//! capture start/stop, frame dequeue/enqueue lifecycle, buffer address
//! queries, and interrupt handling.

#![cfg(feature = "camera-csi")]

extern crate alloc;
use alloc::collections::VecDeque;

use smallaios_kernel::hal::{
    CsiConfig, CsiFrame, CsiIrqSource, CsiPixelFormat, CsiReceiver, HalError,
};
use smallaios_peripheral::PeripheralType;

// ---------------------------------------------------------------------------
// Mock CSI receiver
// ---------------------------------------------------------------------------

/// Mock CSI-2 receiver for testing.
struct MockCsi {
    initialized: bool,
    capturing: bool,
    config: Option<CsiConfig>,
    /// Simulated frame buffers (physical addresses).
    buffer_addrs: [u64; 4],
    /// Size of a single buffer in bytes.
    buf_size: usize,
    /// Queue of completed frames ready for dequeue.
    completed_frames: VecDeque<CsiFrame>,
    /// Buffers returned by the application (available for capture).
    available_buffers: VecDeque<u8>,
    /// Frame sequence counter.
    sequence: u32,
    /// Pending IRQ source.
    pending_irq: Option<CsiIrqSource>,
}

impl MockCsi {
    fn new() -> Self {
        Self {
            initialized: false,
            capturing: false,
            config: None,
            buffer_addrs: [0; 4],
            buf_size: 0,
            completed_frames: VecDeque::new(),
            available_buffers: VecDeque::new(),
            sequence: 0,
            pending_irq: None,
        }
    }

    /// Simulate a frame capture completing on the given buffer index.
    fn simulate_frame_complete(&mut self, buffer_index: u8) {
        let frame = CsiFrame {
            buffer_index,
            timestamp_us: 16_667 * (self.sequence as u64 + 1), // ~60fps timestamps
            sequence: self.sequence,
            bytes_used: self.buf_size as u32,
        };
        self.sequence += 1;
        self.completed_frames.push_back(frame);
        self.pending_irq = Some(CsiIrqSource::FrameComplete(buffer_index));
    }
}

impl CsiReceiver for MockCsi {
    fn init(&mut self, config: CsiConfig) -> Result<(), HalError> {
        if config.lanes != 1 && config.lanes != 2 && config.lanes != 4 {
            return Err(HalError::InvalidConfig);
        }
        if config.num_buffers == 0 || config.num_buffers > 4 {
            return Err(HalError::InvalidConfig);
        }
        // Calculate buffer size.
        let bytes_per_pixel = match config.pixel_format {
            CsiPixelFormat::Raw8 => 1,
            CsiPixelFormat::Raw10 => 2, // packed into 16-bit
            CsiPixelFormat::Raw12 => 2,
            CsiPixelFormat::Yuv422 => 2,
            CsiPixelFormat::Rgb888 => 3,
        };
        self.buf_size = config.width as usize * config.height as usize * bytes_per_pixel;

        // Assign buffer addresses (simulated contiguous memory).
        let base = 0x8000_0000u64;
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

    fn dequeue_frame(&mut self) -> Result<CsiFrame, HalError> {
        if let Some(frame) = self.completed_frames.pop_front() {
            Ok(frame)
        } else {
            Err(HalError::RxEmpty)
        }
    }

    fn enqueue_frame(&mut self, buffer_index: u8) -> Result<(), HalError> {
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

    fn irq_handler(&mut self) -> Result<CsiIrqSource, HalError> {
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
        self.completed_frames.clear();
        self.available_buffers.clear();
        self.sequence = 0;
        self.pending_irq = None;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_csi_init_1080p_yuv422() {
    let mut csi = MockCsi::new();
    let config = CsiConfig {
        lanes: 2,
        lane_speed_mbps: 1500,
        width: 1920,
        height: 1080,
        pixel_format: CsiPixelFormat::Yuv422,
        fps: 30,
        num_buffers: 3,
    };
    assert!(csi.init(config).is_ok());
    assert!(csi.initialized);
    let expected_size = 1920 * 1080 * 2; // YUV422 = 2 bytes/pixel
    assert_eq!(csi.buffer_size(), expected_size);
}

#[test]
fn test_csi_init_720p_raw10() {
    let mut csi = MockCsi::new();
    let config = CsiConfig {
        lanes: 4,
        lane_speed_mbps: 800,
        width: 1280,
        height: 720,
        pixel_format: CsiPixelFormat::Raw10,
        fps: 60,
        num_buffers: 2,
    };
    assert!(csi.init(config).is_ok());
    assert_eq!(csi.buffer_size(), 1280 * 720 * 2);
}

#[test]
fn test_csi_start_stop_capture() {
    let mut csi = MockCsi::new();
    csi.init(CsiConfig {
        lanes: 2,
        lane_speed_mbps: 1500,
        width: 640,
        height: 480,
        pixel_format: CsiPixelFormat::Rgb888,
        fps: 30,
        num_buffers: 2,
    })
    .unwrap();

    assert!(!csi.capturing);
    csi.start_capture().unwrap();
    assert!(csi.capturing);
    csi.stop_capture().unwrap();
    assert!(!csi.capturing);
}

#[test]
fn test_csi_dequeue_frame() {
    let mut csi = MockCsi::new();
    csi.init(CsiConfig {
        lanes: 2,
        lane_speed_mbps: 1500,
        width: 640,
        height: 480,
        pixel_format: CsiPixelFormat::Yuv422,
        fps: 30,
        num_buffers: 3,
    })
    .unwrap();
    csi.start_capture().unwrap();

    // No frames yet.
    assert_eq!(csi.dequeue_frame(), Err(HalError::RxEmpty));

    // Simulate a frame completing on buffer 0.
    csi.simulate_frame_complete(0);
    let frame = csi.dequeue_frame().unwrap();
    assert_eq!(frame.buffer_index, 0);
    assert_eq!(frame.sequence, 0);
    assert_eq!(frame.bytes_used, (640 * 480 * 2) as u32);
    assert!(frame.timestamp_us > 0);
}

#[test]
fn test_csi_frame_buffer_lifecycle() {
    let mut csi = MockCsi::new();
    csi.init(CsiConfig {
        lanes: 2,
        lane_speed_mbps: 1500,
        width: 320,
        height: 240,
        pixel_format: CsiPixelFormat::Raw8,
        fps: 30,
        num_buffers: 3,
    })
    .unwrap();
    csi.start_capture().unwrap();

    // Simulate 3 frames.
    csi.simulate_frame_complete(0);
    csi.simulate_frame_complete(1);
    csi.simulate_frame_complete(2);

    // Dequeue all 3.
    let f0 = csi.dequeue_frame().unwrap();
    let f1 = csi.dequeue_frame().unwrap();
    let f2 = csi.dequeue_frame().unwrap();
    assert_eq!(f0.sequence, 0);
    assert_eq!(f1.sequence, 1);
    assert_eq!(f2.sequence, 2);

    // Return buffers for reuse.
    csi.enqueue_frame(f0.buffer_index).unwrap();
    csi.enqueue_frame(f1.buffer_index).unwrap();
    csi.enqueue_frame(f2.buffer_index).unwrap();

    // Verify buffers are available again.
    assert_eq!(csi.available_buffers.len(), 6); // 3 initial + 3 returned
}

#[test]
fn test_csi_buffer_addresses() {
    let mut csi = MockCsi::new();
    csi.init(CsiConfig {
        lanes: 2,
        lane_speed_mbps: 1500,
        width: 640,
        height: 480,
        pixel_format: CsiPixelFormat::Yuv422,
        fps: 30,
        num_buffers: 3,
    })
    .unwrap();

    let buf_size = csi.buffer_size() as u64;
    let addr0 = csi.buffer_address(0).unwrap();
    let addr1 = csi.buffer_address(1).unwrap();
    let addr2 = csi.buffer_address(2).unwrap();

    assert_eq!(addr0, 0x8000_0000);
    assert_eq!(addr1, 0x8000_0000 + buf_size);
    assert_eq!(addr2, 0x8000_0000 + 2 * buf_size);

    // Out of range.
    assert_eq!(csi.buffer_address(3), Err(HalError::OutOfRange));
}

#[test]
fn test_csi_irq_handler() {
    let mut csi = MockCsi::new();
    csi.init(CsiConfig {
        lanes: 2,
        lane_speed_mbps: 1500,
        width: 640,
        height: 480,
        pixel_format: CsiPixelFormat::Yuv422,
        fps: 30,
        num_buffers: 2,
    })
    .unwrap();
    csi.start_capture().unwrap();

    csi.simulate_frame_complete(0);
    let irq = csi.irq_handler().unwrap();
    assert_eq!(irq, CsiIrqSource::FrameComplete(0));

    // No more pending.
    assert_eq!(csi.irq_handler(), Err(HalError::InterruptError));
}

#[test]
fn test_csi_invalid_lane_count() {
    let mut csi = MockCsi::new();
    let config = CsiConfig {
        lanes: 3, // Invalid: must be 1, 2, or 4.
        lane_speed_mbps: 1500,
        width: 640,
        height: 480,
        pixel_format: CsiPixelFormat::Yuv422,
        fps: 30,
        num_buffers: 2,
    };
    assert_eq!(csi.init(config), Err(HalError::InvalidConfig));
}

#[test]
fn test_csi_start_without_init() {
    let mut csi = MockCsi::new();
    assert_eq!(csi.start_capture(), Err(HalError::InitFailed));
}

#[test]
fn test_csi_reset() {
    let mut csi = MockCsi::new();
    csi.init(CsiConfig {
        lanes: 2,
        lane_speed_mbps: 1500,
        width: 640,
        height: 480,
        pixel_format: CsiPixelFormat::Yuv422,
        fps: 30,
        num_buffers: 2,
    })
    .unwrap();
    csi.start_capture().unwrap();
    csi.simulate_frame_complete(0);

    csi.reset().unwrap();
    assert!(!csi.initialized);
    assert!(!csi.capturing);
    assert_eq!(csi.buffer_size(), 0);
    assert_eq!(csi.dequeue_frame(), Err(HalError::RxEmpty));
}

#[test]
fn test_csi_peripheral_type() {
    assert_eq!(PeripheralType::from_u8(4), Some(PeripheralType::Camera));
    assert_eq!(PeripheralType::Camera as u8, 4);
}
