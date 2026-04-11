// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Bridges CAN frames to inference input tensors and vice versa.
//!
//! Maps CAN IDs to byte offsets within inference input tensors via a
//! configurable routing table. Supports batch triggers (frame count,
//! time window, trigger frame), CANaerospace decoding, and stale frame
//! detection.
//!
//! # Usage
//!
//! 1. Build an [`AdapterConfig`] describing input routes (CAN ID →
//!    tensor offset), output routes (tensor offset → CAN ID), batch
//!    triggers, and per-topic tensor sizes.
//! 2. Construct a [`CanInferenceAdapter`] from the config.
//! 3. Feed received CAN frames to [`CanInferenceAdapter::process_frame`]
//!    along with a monotonically increasing microsecond timestamp. When
//!    a batch trigger fires, the call returns `Some((topic, payload))`.
//! 4. Pass `payload` to the inference runner. When it returns an
//!    output tensor, call [`CanInferenceAdapter::on_inference_output`]
//!    to encode it back into CAN frames for transmission.
//!
//! The adapter also implements [`CanFrameSink`] with a zero timestamp,
//! which is useful for caller pattern wiring where timestamps are
//! tracked by a higher-level scheduler.

use alloc::collections::BTreeMap;
use alloc::string::String;
#[cfg(test)]
use alloc::string::ToString;
use alloc::vec;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};

use super::canaerospace::DataType;
use super::controller::CanFrameSink;
use super::frame::CanFrame;

/// How a CAN frame's payload is decoded into the tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameDecoder {
    /// Copy raw bytes verbatim.
    RawBytes,
    /// Decode as CANaerospace float32 (data type `DataType::Float`, 4-byte
    /// data field at bytes 4..8 per ARINC 825).
    CanAerospaceFloat32,
    /// Decode as CANaerospace signed int32 (data type `DataType::Long`,
    /// 4-byte data field at bytes 4..8 per ARINC 825).
    CanAerospaceInt32,
}

/// Routing from a CAN ID to a slice of an inference input tensor.
#[derive(Debug, Clone)]
pub struct RouteSpec {
    /// Logical topic name (tensor identifier) this route writes into.
    pub topic: String,
    /// Byte offset within the tensor payload at which to write the
    /// decoded frame data.
    pub tensor_offset: usize,
    /// Number of decoded bytes to copy from frame into tensor.
    pub frame_size: usize,
    /// How to decode the CAN frame payload before copying.
    pub decoder: FrameDecoder,
}

/// Routing from a slice of an inference output tensor back to a CAN
/// frame to be transmitted on the bus.
#[derive(Debug, Clone)]
pub struct OutputRouteSpec {
    /// Topic (output tensor identifier) this route reads from.
    pub source_topic: String,
    /// Byte offset within the output tensor payload to read from.
    pub tensor_offset: usize,
    /// Number of bytes to read from the tensor.
    pub frame_size: usize,
    /// CAN identifier of the emitted frame (standard 11-bit).
    pub can_id: u32,
    /// Encoding scheme (uses the same enum, interpreted as encoder).
    pub encoder: FrameDecoder,
}

/// When to flush a batched tensor to the inference runner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchTrigger {
    /// Flush after N frames received for the topic.
    FrameCount(usize),
    /// Flush after T microseconds since first frame in the batch.
    TimeWindow(u64),
    /// Flush when a specific trigger CAN ID is received.
    OnFrame(u32),
}

/// Adapter configuration: input routes, output routes, triggers, sizes.
#[derive(Debug, Clone)]
pub struct AdapterConfig {
    /// Routes from CAN ID to tensor position.
    pub input_routes: BTreeMap<u32, RouteSpec>,
    /// Routes from inference output to CAN frames.
    pub output_routes: Vec<OutputRouteSpec>,
    /// Per-topic batch trigger configuration.
    pub batch_triggers: BTreeMap<String, BatchTrigger>,
    /// Per-topic tensor size (bytes) for buffer pre-allocation.
    pub tensor_sizes: BTreeMap<String, usize>,
    /// Maximum age of any frame in a batch (microseconds). Batches in
    /// which the oldest frame exceeds this age are dropped when the
    /// trigger fires, with `stale_batch_drops_total` incremented.
    pub freshness_window_us: u64,
}

impl AdapterConfig {
    /// Create an empty configuration. Callers populate the fields
    /// directly or through a builder/loader.
    pub fn new() -> Self {
        Self {
            input_routes: BTreeMap::new(),
            output_routes: Vec::new(),
            batch_triggers: BTreeMap::new(),
            tensor_sizes: BTreeMap::new(),
            freshness_window_us: u64::MAX,
        }
    }
}

impl Default for AdapterConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct BatchBuffer {
    payload: Vec<u8>,
    frames_received: usize,
    first_frame_timestamp: u64,
    frame_timestamps: BTreeMap<u32, u64>,
}

/// Adapts CAN traffic to inference tensors and back again.
pub struct CanInferenceAdapter {
    config: AdapterConfig,
    batches: BTreeMap<String, BatchBuffer>,

    /// Frames dropped because the decoder produced insufficient bytes
    /// or the write would overflow the tensor buffer.
    pub dropped_frames_total: AtomicUsize,
    /// Number of partial batches reset without flushing (reserved for
    /// future use by higher-level schedulers).
    pub partial_batches_total: AtomicUsize,
    /// Frames that did not match any input route.
    pub unrouted_frames_total: AtomicUsize,
    /// Batches dropped because one or more frames were older than
    /// `freshness_window_us` at the moment the trigger fired.
    pub stale_batch_drops_total: AtomicUsize,
}

impl CanInferenceAdapter {
    /// Construct a new adapter from a configuration. Pre-allocates a
    /// zero-filled buffer for each topic in `config.tensor_sizes`.
    pub fn new(config: AdapterConfig) -> Self {
        let batches = config
            .tensor_sizes
            .iter()
            .map(|(topic, &size)| {
                let buf = BatchBuffer {
                    payload: vec![0u8; size],
                    frames_received: 0,
                    first_frame_timestamp: 0,
                    frame_timestamps: BTreeMap::new(),
                };
                (topic.clone(), buf)
            })
            .collect();
        Self {
            config,
            batches,
            dropped_frames_total: AtomicUsize::new(0),
            partial_batches_total: AtomicUsize::new(0),
            unrouted_frames_total: AtomicUsize::new(0),
            stale_batch_drops_total: AtomicUsize::new(0),
        }
    }

    /// Read-only access to the underlying configuration.
    pub fn config(&self) -> &AdapterConfig {
        &self.config
    }

    /// Process a single CAN frame with an explicit microsecond
    /// timestamp. Returns `Some((topic, payload))` if a batch flushed
    /// as a result of this frame.
    pub fn process_frame(
        &mut self,
        frame: &CanFrame,
        timestamp_us: u64,
    ) -> Option<(String, Vec<u8>)> {
        let route = match self.config.input_routes.get(&frame.id) {
            Some(r) => r.clone(),
            None => {
                self.unrouted_frames_total.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        };

        // Decode frame payload per route decoder.
        let decoded = match Self::decode_frame(frame, route.decoder) {
            Some(d) => d,
            None => {
                self.dropped_frames_total.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        };

        if decoded.len() < route.frame_size {
            self.dropped_frames_total.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        let topic = route.topic.clone();
        let buf = match self.batches.get_mut(&topic) {
            Some(b) => b,
            None => {
                self.dropped_frames_total.fetch_add(1, Ordering::Relaxed);
                return None;
            }
        };

        // Write decoded bytes into tensor at offset.
        let end = route.tensor_offset + route.frame_size;
        if end > buf.payload.len() {
            self.dropped_frames_total.fetch_add(1, Ordering::Relaxed);
            return None;
        }
        buf.payload[route.tensor_offset..end].copy_from_slice(&decoded[..route.frame_size]);

        // Track timestamps.
        if buf.frames_received == 0 {
            buf.first_frame_timestamp = timestamp_us;
        }
        buf.frame_timestamps.insert(frame.id, timestamp_us);
        buf.frames_received += 1;

        // Resolve the batch trigger for this topic. If the user did
        // not specify one, default to the number of input routes that
        // feed this topic (i.e. "flush when we have seen each route
        // once").
        let trigger = self.config.batch_triggers.get(&topic).copied().unwrap_or({
            let default_count = self
                .config
                .input_routes
                .values()
                .filter(|r| r.topic == topic)
                .count();
            BatchTrigger::FrameCount(default_count.max(1))
        });

        let should_flush = match trigger {
            BatchTrigger::FrameCount(n) => buf.frames_received >= n,
            BatchTrigger::TimeWindow(t) => {
                timestamp_us.saturating_sub(buf.first_frame_timestamp) >= t
            }
            BatchTrigger::OnFrame(id) => frame.id == id,
        };

        if should_flush {
            // Freshness check: the oldest frame in the batch must be
            // within the freshness window relative to `timestamp_us`.
            let oldest = buf
                .frame_timestamps
                .values()
                .min()
                .copied()
                .unwrap_or(timestamp_us);
            if timestamp_us.saturating_sub(oldest) > self.config.freshness_window_us {
                self.stale_batch_drops_total.fetch_add(1, Ordering::Relaxed);
                Self::reset_buf(buf);
                return None;
            }

            let payload = buf.payload.clone();
            Self::reset_buf(buf);
            return Some((topic, payload));
        }

        None
    }

    /// Decode a frame according to the given decoder. Returns `None`
    /// if the frame is malformed for the decoder (e.g. too short, or
    /// a CANaerospace frame with a mismatched `data_type`).
    fn decode_frame(frame: &CanFrame, decoder: FrameDecoder) -> Option<Vec<u8>> {
        match decoder {
            FrameDecoder::RawBytes => Some(frame.data.clone()),
            FrameDecoder::CanAerospaceFloat32 => {
                // CANaerospace header is bytes 0..4 (node_id, data_type,
                // service_code, message_code); data is bytes 4..8.
                if frame.data.len() < 8 {
                    return None;
                }
                if frame.data[1] != DataType::Float.raw() {
                    return None;
                }
                Some(frame.data[4..8].to_vec())
            }
            FrameDecoder::CanAerospaceInt32 => {
                if frame.data.len() < 8 {
                    return None;
                }
                if frame.data[1] != DataType::Long.raw() {
                    return None;
                }
                Some(frame.data[4..8].to_vec())
            }
        }
    }

    fn reset_buf(buf: &mut BatchBuffer) {
        buf.frames_received = 0;
        buf.first_frame_timestamp = 0;
        buf.frame_timestamps.clear();
        for b in buf.payload.iter_mut() {
            *b = 0;
        }
    }

    /// Reset the batch state for a specific topic without flushing.
    pub fn reset_batch(&mut self, topic: &str) {
        if let Some(buf) = self.batches.get_mut(topic) {
            Self::reset_buf(buf);
            self.partial_batches_total.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Convert an inference output tensor back to CAN frames, one per
    /// matching output route. Frames whose source range lies outside
    /// `payload` are silently skipped.
    pub fn on_inference_output(&self, topic: &str, payload: &[u8]) -> Vec<CanFrame> {
        let mut frames = Vec::new();
        for route in &self.config.output_routes {
            if route.source_topic != topic {
                continue;
            }
            if route.tensor_offset + route.frame_size > payload.len() {
                continue;
            }

            let src = &payload[route.tensor_offset..route.tensor_offset + route.frame_size];
            let mut data = vec![0u8; 8];

            match route.encoder {
                FrameDecoder::RawBytes => {
                    let copy_len = route.frame_size.min(8);
                    data[..copy_len].copy_from_slice(&src[..copy_len]);
                    // Trim to the actual copied size so the CAN frame
                    // reflects the configured length.
                    data.truncate(copy_len.max(1));
                }
                FrameDecoder::CanAerospaceFloat32 | FrameDecoder::CanAerospaceInt32 => {
                    // Build a full 8-byte CANaerospace frame:
                    //   [node_id, data_type, service_code, msg_code, d0..d3]
                    data[0] = 1; // default node_id
                    data[1] = if matches!(route.encoder, FrameDecoder::CanAerospaceFloat32) {
                        DataType::Float.raw()
                    } else {
                        DataType::Long.raw()
                    };
                    data[2] = 0;
                    data[3] = 0;
                    let copy_len = route.frame_size.min(4);
                    data[4..4 + copy_len].copy_from_slice(&src[..copy_len]);
                }
            }

            if let Ok(frame) = CanFrame::new_standard(route.can_id, &data) {
                frames.push(frame);
            }
        }
        frames
    }
}

impl CanFrameSink for CanInferenceAdapter {
    fn on_frame(&mut self, frame: &CanFrame) {
        // Trait variant without explicit timestamp — real integrations
        // should prefer [`CanInferenceAdapter::process_frame`] so that
        // freshness checks and time-window triggers behave correctly.
        let _ = self.process_frame(frame, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::can::canaerospace::{params, CanaMessage, DataType};

    fn topic(name: &str) -> String {
        name.to_string()
    }

    /// Build a bare 4-byte payload tensor with two input routes:
    /// CAN ID 0x100 → bytes 0..2, CAN ID 0x101 → bytes 2..4.
    fn simple_two_route_config() -> AdapterConfig {
        let mut input_routes = BTreeMap::new();
        input_routes.insert(
            0x100,
            RouteSpec {
                topic: topic("sensors"),
                tensor_offset: 0,
                frame_size: 2,
                decoder: FrameDecoder::RawBytes,
            },
        );
        input_routes.insert(
            0x101,
            RouteSpec {
                topic: topic("sensors"),
                tensor_offset: 2,
                frame_size: 2,
                decoder: FrameDecoder::RawBytes,
            },
        );

        let mut tensor_sizes = BTreeMap::new();
        tensor_sizes.insert(topic("sensors"), 4);

        AdapterConfig {
            input_routes,
            output_routes: Vec::new(),
            batch_triggers: BTreeMap::new(),
            tensor_sizes,
            freshness_window_us: u64::MAX,
        }
    }

    #[test]
    fn test_default_frame_count_trigger_flushes_on_all_routes() {
        // With no explicit trigger configured, the adapter should flush
        // once every input route for the topic has been seen.
        let mut adapter = CanInferenceAdapter::new(simple_two_route_config());

        let f1 = CanFrame::new_standard(0x100, &[0xAA, 0xBB]).unwrap();
        let r1 = adapter.process_frame(&f1, 1000);
        assert!(r1.is_none());

        let f2 = CanFrame::new_standard(0x101, &[0xCC, 0xDD]).unwrap();
        let r2 = adapter.process_frame(&f2, 1100);
        let (topic, payload) = r2.expect("batch should flush after two routes filled");
        assert_eq!(topic, "sensors");
        assert_eq!(payload, vec![0xAA, 0xBB, 0xCC, 0xDD]);
    }

    #[test]
    fn test_frame_count_trigger() {
        let mut cfg = simple_two_route_config();
        cfg.batch_triggers
            .insert(topic("sensors"), BatchTrigger::FrameCount(2));
        let mut adapter = CanInferenceAdapter::new(cfg);

        let f1 = CanFrame::new_standard(0x100, &[0x11, 0x22]).unwrap();
        assert!(adapter.process_frame(&f1, 0).is_none());

        let f2 = CanFrame::new_standard(0x101, &[0x33, 0x44]).unwrap();
        let (t, payload) = adapter.process_frame(&f2, 1).unwrap();
        assert_eq!(t, "sensors");
        assert_eq!(payload, vec![0x11, 0x22, 0x33, 0x44]);
    }

    #[test]
    fn test_time_window_trigger() {
        let mut cfg = simple_two_route_config();
        cfg.batch_triggers
            .insert(topic("sensors"), BatchTrigger::TimeWindow(500));
        let mut adapter = CanInferenceAdapter::new(cfg);

        let f1 = CanFrame::new_standard(0x100, &[0xAA, 0xBB]).unwrap();
        assert!(adapter.process_frame(&f1, 1_000).is_none());

        // Not yet 500 us elapsed.
        let f2 = CanFrame::new_standard(0x101, &[0xCC, 0xDD]).unwrap();
        assert!(adapter.process_frame(&f2, 1_100).is_none());

        // Now enough time has passed: another frame at 1_600 flushes.
        let f3 = CanFrame::new_standard(0x100, &[0x11, 0x22]).unwrap();
        let flushed = adapter.process_frame(&f3, 1_600);
        let (t, payload) = flushed.expect("time window should have fired");
        assert_eq!(t, "sensors");
        assert_eq!(payload, vec![0x11, 0x22, 0xCC, 0xDD]);
    }

    #[test]
    fn test_on_frame_trigger() {
        let mut cfg = simple_two_route_config();
        cfg.batch_triggers
            .insert(topic("sensors"), BatchTrigger::OnFrame(0x101));
        let mut adapter = CanInferenceAdapter::new(cfg);

        let f1 = CanFrame::new_standard(0x100, &[0xAA, 0xBB]).unwrap();
        assert!(adapter.process_frame(&f1, 0).is_none());

        let trigger = CanFrame::new_standard(0x101, &[0xCC, 0xDD]).unwrap();
        let (t, _) = adapter.process_frame(&trigger, 1).unwrap();
        assert_eq!(t, "sensors");
    }

    #[test]
    fn test_unrouted_frames_counter() {
        let mut adapter = CanInferenceAdapter::new(simple_two_route_config());
        let f = CanFrame::new_standard(0x7FF, &[0x01]).unwrap();
        assert!(adapter.process_frame(&f, 0).is_none());
        assert_eq!(adapter.unrouted_frames_total.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_stale_batch_dropped() {
        let mut cfg = simple_two_route_config();
        cfg.batch_triggers
            .insert(topic("sensors"), BatchTrigger::FrameCount(2));
        cfg.freshness_window_us = 100;
        let mut adapter = CanInferenceAdapter::new(cfg);

        // First frame at t=0, second at t=500 → oldest age 500 > 100.
        let f1 = CanFrame::new_standard(0x100, &[0xAA, 0xBB]).unwrap();
        assert!(adapter.process_frame(&f1, 0).is_none());

        let f2 = CanFrame::new_standard(0x101, &[0xCC, 0xDD]).unwrap();
        let flushed = adapter.process_frame(&f2, 500);
        assert!(flushed.is_none(), "stale batch should be dropped");
        assert_eq!(adapter.stale_batch_drops_total.load(Ordering::Relaxed), 1);

        // Buffer was reset — a fresh pair should now flush normally.
        let f3 = CanFrame::new_standard(0x100, &[0x01, 0x02]).unwrap();
        assert!(adapter.process_frame(&f3, 600).is_none());
        let f4 = CanFrame::new_standard(0x101, &[0x03, 0x04]).unwrap();
        let (_, payload) = adapter.process_frame(&f4, 650).unwrap();
        assert_eq!(payload, vec![0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn test_canaerospace_float32_decoded() {
        // Build a real CANaerospace NOD frame for airspeed (CAN ID 200).
        let speed: f32 = 123.5;
        let msg = CanaMessage::new_nod(
            params::AIRSPEED_IAS,
            1,
            DataType::Float,
            &speed.to_be_bytes(),
        )
        .unwrap();
        let frame = msg.to_can_frame().unwrap();
        assert_eq!(frame.data.len(), 8);

        let mut input_routes = BTreeMap::new();
        input_routes.insert(
            params::AIRSPEED_IAS,
            RouteSpec {
                topic: topic("flight"),
                tensor_offset: 0,
                frame_size: 4,
                decoder: FrameDecoder::CanAerospaceFloat32,
            },
        );
        let mut tensor_sizes = BTreeMap::new();
        tensor_sizes.insert(topic("flight"), 4);
        let mut batch_triggers = BTreeMap::new();
        batch_triggers.insert(topic("flight"), BatchTrigger::FrameCount(1));

        let cfg = AdapterConfig {
            input_routes,
            output_routes: Vec::new(),
            batch_triggers,
            tensor_sizes,
            freshness_window_us: u64::MAX,
        };
        let mut adapter = CanInferenceAdapter::new(cfg);

        let (t, payload) = adapter.process_frame(&frame, 0).unwrap();
        assert_eq!(t, "flight");
        let decoded = f32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]);
        assert_eq!(decoded, 123.5);
    }

    #[test]
    fn test_canaerospace_int32_decoded() {
        let value: i32 = -42;
        // Build CANaerospace frame with Long data type (signed 32-bit).
        let msg = CanaMessage::new_nod(params::ENGINE_RPM, 2, DataType::Long, &value.to_be_bytes())
            .unwrap();
        let frame = msg.to_can_frame().unwrap();

        let mut input_routes = BTreeMap::new();
        input_routes.insert(
            params::ENGINE_RPM,
            RouteSpec {
                topic: topic("engine"),
                tensor_offset: 0,
                frame_size: 4,
                decoder: FrameDecoder::CanAerospaceInt32,
            },
        );
        let mut tensor_sizes = BTreeMap::new();
        tensor_sizes.insert(topic("engine"), 4);
        let mut batch_triggers = BTreeMap::new();
        batch_triggers.insert(topic("engine"), BatchTrigger::FrameCount(1));
        let cfg = AdapterConfig {
            input_routes,
            output_routes: Vec::new(),
            batch_triggers,
            tensor_sizes,
            freshness_window_us: u64::MAX,
        };

        let mut adapter = CanInferenceAdapter::new(cfg);
        let (_, payload) = adapter.process_frame(&frame, 0).unwrap();
        let decoded = i32::from_be_bytes([payload[0], payload[1], payload[2], payload[3]]);
        assert_eq!(decoded, -42);
    }

    #[test]
    fn test_canaerospace_rejects_wrong_data_type() {
        // Float decoder applied to a Long-typed frame should reject.
        let msg =
            CanaMessage::new_nod(params::AIRSPEED_IAS, 1, DataType::Long, &1i32.to_be_bytes())
                .unwrap();
        let frame = msg.to_can_frame().unwrap();

        let mut input_routes = BTreeMap::new();
        input_routes.insert(
            params::AIRSPEED_IAS,
            RouteSpec {
                topic: topic("flight"),
                tensor_offset: 0,
                frame_size: 4,
                decoder: FrameDecoder::CanAerospaceFloat32,
            },
        );
        let mut tensor_sizes = BTreeMap::new();
        tensor_sizes.insert(topic("flight"), 4);
        let cfg = AdapterConfig {
            input_routes,
            output_routes: Vec::new(),
            batch_triggers: BTreeMap::new(),
            tensor_sizes,
            freshness_window_us: u64::MAX,
        };
        let mut adapter = CanInferenceAdapter::new(cfg);

        assert!(adapter.process_frame(&frame, 0).is_none());
        assert_eq!(adapter.dropped_frames_total.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_round_trip_input_to_output() {
        // Input: one raw-bytes route packs 2 frames into a 4-byte tensor.
        // Output: same 4 bytes are emitted back as two CAN frames.
        let mut input_routes = BTreeMap::new();
        input_routes.insert(
            0x200,
            RouteSpec {
                topic: topic("io"),
                tensor_offset: 0,
                frame_size: 2,
                decoder: FrameDecoder::RawBytes,
            },
        );
        input_routes.insert(
            0x201,
            RouteSpec {
                topic: topic("io"),
                tensor_offset: 2,
                frame_size: 2,
                decoder: FrameDecoder::RawBytes,
            },
        );

        let output_routes = vec![
            OutputRouteSpec {
                source_topic: topic("io_out"),
                tensor_offset: 0,
                frame_size: 2,
                can_id: 0x300,
                encoder: FrameDecoder::RawBytes,
            },
            OutputRouteSpec {
                source_topic: topic("io_out"),
                tensor_offset: 2,
                frame_size: 2,
                can_id: 0x301,
                encoder: FrameDecoder::RawBytes,
            },
        ];

        let mut tensor_sizes = BTreeMap::new();
        tensor_sizes.insert(topic("io"), 4);

        let cfg = AdapterConfig {
            input_routes,
            output_routes,
            batch_triggers: BTreeMap::new(),
            tensor_sizes,
            freshness_window_us: u64::MAX,
        };
        let mut adapter = CanInferenceAdapter::new(cfg);

        let f1 = CanFrame::new_standard(0x200, &[0xDE, 0xAD]).unwrap();
        let f2 = CanFrame::new_standard(0x201, &[0xBE, 0xEF]).unwrap();
        assert!(adapter.process_frame(&f1, 0).is_none());
        let (_, payload) = adapter.process_frame(&f2, 1).unwrap();
        assert_eq!(payload, vec![0xDE, 0xAD, 0xBE, 0xEF]);

        // Pretend an inference runner passed `payload` through and
        // returned it under a different topic name.
        let frames = adapter.on_inference_output("io_out", &payload);
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].id, 0x300);
        assert_eq!(frames[0].data, vec![0xDE, 0xAD]);
        assert_eq!(frames[1].id, 0x301);
        assert_eq!(frames[1].data, vec![0xBE, 0xEF]);
    }

    #[test]
    fn test_round_trip_canaerospace_output_encoding() {
        // Output encoder builds a full 8-byte CANaerospace frame.
        let output_routes = vec![OutputRouteSpec {
            source_topic: topic("out"),
            tensor_offset: 0,
            frame_size: 4,
            can_id: params::ALTITUDE_BARO,
            encoder: FrameDecoder::CanAerospaceFloat32,
        }];

        let cfg = AdapterConfig {
            input_routes: BTreeMap::new(),
            output_routes,
            batch_triggers: BTreeMap::new(),
            tensor_sizes: BTreeMap::new(),
            freshness_window_us: u64::MAX,
        };
        let adapter = CanInferenceAdapter::new(cfg);

        let altitude: f32 = 35_000.0;
        let payload = altitude.to_be_bytes().to_vec();
        let frames = adapter.on_inference_output("out", &payload);
        assert_eq!(frames.len(), 1);

        let decoded = CanaMessage::from_can_frame(&frames[0]).unwrap();
        assert_eq!(decoded.data_type, DataType::Float.raw());
        assert_eq!(decoded.as_f32(), Some(35_000.0));
    }

    #[test]
    fn test_can_frame_sink_impl() {
        // The trait variant uses timestamp 0 and therefore cannot use
        // time-window triggers; frame-count triggers should still work.
        let mut cfg = simple_two_route_config();
        cfg.batch_triggers
            .insert(topic("sensors"), BatchTrigger::FrameCount(2));
        let mut adapter = CanInferenceAdapter::new(cfg);

        let f1 = CanFrame::new_standard(0x100, &[0x01, 0x02]).unwrap();
        let f2 = CanFrame::new_standard(0x101, &[0x03, 0x04]).unwrap();

        // Drive through the CanFrameSink trait method.
        let sink: &mut dyn CanFrameSink = &mut adapter;
        sink.on_frame(&f1);
        sink.on_frame(&f2);

        // Via the trait, the result is swallowed — but the counters
        // and buffer state should still reflect successful processing.
        assert_eq!(adapter.unrouted_frames_total.load(Ordering::Relaxed), 0);
        assert_eq!(adapter.dropped_frames_total.load(Ordering::Relaxed), 0);
    }
}
