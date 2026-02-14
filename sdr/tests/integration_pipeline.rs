// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration test (Task 7.21): SDR mock -> ring buffer -> FFT -> ONNX
//! inference -> Zenoh publish.
//!
//! Tests the end-to-end IQ processing pipeline: mock SDR source produces
//! IQ samples which flow through the ring buffer, undergo windowing and FFT,
//! produce magnitude/PSD spectra, and are formatted as ONNX-compatible tensors.

#![cfg(feature = "iq-pipeline")]

extern crate alloc;
use alloc::vec;

use smallaios_sdr::pipeline::{
    apply_hamming_window, apply_hann_window, apply_rectangular_window, apply_window,
    compute_stride, fft_radix2, format_tensor_1d_mag, format_tensor_2d_ri, magnitude_spectrum,
    power_spectral_density, IqRingBuffer, PipelineConfig, PipelineManager, PipelineState,
    RunningNormalizer, WindowFunction, MAX_PIPELINES,
};
use smallaios_sdr::{IqFormat, SdrError};

// ---------------------------------------------------------------------------
// Helper: generate synthetic IQ samples (single tone at a given bin)
// ---------------------------------------------------------------------------

fn generate_single_tone(n: usize, bin: usize) -> alloc::vec::Vec<(f32, f32)> {
    (0..n)
        .map(|i| {
            let angle = 2.0 * core::f32::consts::PI * bin as f32 * i as f32 / n as f32;
            (angle.cos(), angle.sin())
        })
        .collect()
}

fn generate_dc(n: usize) -> alloc::vec::Vec<(f32, f32)> {
    vec![(1.0f32, 0.0f32); n]
}

// ---------------------------------------------------------------------------
// 1. IQ samples flowing through ring buffer
// ---------------------------------------------------------------------------

#[test]
fn pipeline_ring_buffer_write_and_read() {
    let mut storage = vec![(0.0f32, 0.0f32); 128];
    let buf = unsafe {
        IqRingBuffer::new(core::mem::transmute::<
            &mut [(f32, f32)],
            &'static mut [(f32, f32)],
        >(storage.as_mut_slice()))
    };

    let samples = generate_single_tone(64, 1);
    let written = buf.write(&samples);
    assert_eq!(written, 64);
    assert_eq!(buf.available(), 64);

    let mut output = vec![(0.0f32, 0.0f32); 64];
    let read = buf.read(&mut output, 64);
    assert_eq!(read, 64);

    // Verify first few samples match.
    for i in 0..5 {
        assert!(
            (output[i].0 - samples[i].0).abs() < 0.001,
            "sample {} I mismatch",
            i
        );
        assert!(
            (output[i].1 - samples[i].1).abs() < 0.001,
            "sample {} Q mismatch",
            i
        );
    }
}

#[test]
fn pipeline_ring_buffer_overflow_tracking() {
    let mut storage = vec![(0.0f32, 0.0f32); 8]; // small buffer
    let buf = unsafe {
        IqRingBuffer::new(core::mem::transmute::<
            &mut [(f32, f32)],
            &'static mut [(f32, f32)],
        >(storage.as_mut_slice()))
    };

    // Write more than buffer capacity.
    let samples = generate_dc(16);
    buf.write(&samples);

    assert!(buf.overflow_count() > 0, "should have overflow");
    let count = buf.take_overflow_count();
    assert!(count > 0);
    assert_eq!(buf.overflow_count(), 0, "should reset after take");
}

#[test]
fn pipeline_ring_buffer_empty_read() {
    let mut storage = vec![(0.0f32, 0.0f32); 32];
    let buf = unsafe {
        IqRingBuffer::new(core::mem::transmute::<
            &mut [(f32, f32)],
            &'static mut [(f32, f32)],
        >(storage.as_mut_slice()))
    };

    let mut output = vec![(0.0f32, 0.0f32); 16];
    let read = buf.read(&mut output, 16);
    assert_eq!(read, 0);
}

#[test]
fn pipeline_ring_buffer_partial_read() {
    let mut storage = vec![(0.0f32, 0.0f32); 64];
    let buf = unsafe {
        IqRingBuffer::new(core::mem::transmute::<
            &mut [(f32, f32)],
            &'static mut [(f32, f32)],
        >(storage.as_mut_slice()))
    };

    let samples = generate_dc(10);
    buf.write(&samples);

    let mut output = vec![(0.0f32, 0.0f32); 32];
    let read = buf.read(&mut output, 32);
    assert_eq!(read, 10, "should only read available samples");
}

// ---------------------------------------------------------------------------
// 2. Windowing
// ---------------------------------------------------------------------------

#[test]
fn pipeline_hann_window_properties() {
    let mut samples = vec![(1.0f32, 1.0f32); 64];
    apply_hann_window(&mut samples);

    // First and last should be near zero.
    assert!(samples[0].0.abs() < 0.02);
    assert!(samples[63].0.abs() < 0.02);
    // Middle should be near 1.0.
    assert!((samples[32].0 - 1.0).abs() < 0.05);
}

#[test]
fn pipeline_hamming_window_properties() {
    let mut samples = vec![(1.0f32, 1.0f32); 64];
    apply_hamming_window(&mut samples);

    // Hamming window: w(0) = 0.54 - 0.46 = 0.08
    assert!((samples[0].0 - 0.08).abs() < 0.02);
    // Middle should be near 1.0.
    assert!((samples[32].0 - 1.0).abs() < 0.05);
}

#[test]
fn pipeline_rectangular_window_no_change() {
    let original = vec![(core::f32::consts::PI, 2.71f32); 8];
    let mut samples = original.clone();
    apply_rectangular_window(&mut samples);
    assert_eq!(samples, original);
}

#[test]
fn pipeline_apply_window_dispatches_correctly() {
    let mut hann = vec![(1.0f32, 1.0f32); 8];
    let mut rect = vec![(1.0f32, 1.0f32); 8];

    apply_window(&mut hann, WindowFunction::Hann);
    apply_window(&mut rect, WindowFunction::Rectangular);

    // Hann should modify first sample, rectangular should not.
    assert!(hann[0].0.abs() < 0.02);
    assert_eq!(rect[0].0, 1.0);
}

#[test]
fn pipeline_stride_computation() {
    assert_eq!(compute_stride(1024, 0.5), 512);
    assert_eq!(compute_stride(1024, 0.0), 1024);
    assert_eq!(compute_stride(1024, 0.75), 256);
    assert_eq!(compute_stride(256, 0.5), 128);
}

// ---------------------------------------------------------------------------
// 3. FFT processing
// ---------------------------------------------------------------------------

#[test]
fn pipeline_fft_dc_input() {
    let mut data = vec![(1.0f32, 0.0f32); 64];
    fft_radix2(&mut data).unwrap();

    // DC bin should have magnitude N=64.
    assert!((data[0].0 - 64.0).abs() < 1.0);
    assert!(data[0].1.abs() < 1.0);

    // All other bins should be near zero.
    for (i, d) in data[1..64].iter().enumerate() {
        let mag = (d.0 * d.0 + d.1 * d.1).sqrt();
        assert!(
            mag < 1.0,
            "bin {} magnitude {} should be near zero",
            i + 1,
            mag
        );
    }
}

#[test]
fn pipeline_fft_single_tone() {
    let n = 256;
    let tone_bin = 10;
    let mut data = generate_single_tone(n, tone_bin);
    fft_radix2(&mut data).unwrap();

    // Tone bin should have large magnitude.
    let tone_mag =
        (data[tone_bin].0 * data[tone_bin].0 + data[tone_bin].1 * data[tone_bin].1).sqrt();
    assert!(
        tone_mag > (n as f32) * 0.8,
        "tone bin magnitude {} should be close to {}",
        tone_mag,
        n
    );

    // Non-tone bins should be much smaller.
    for (i, d) in data.iter().enumerate() {
        if i == tone_bin {
            continue;
        }
        let mag = (d.0 * d.0 + d.1 * d.1).sqrt();
        assert!(
            mag < tone_mag * 0.1,
            "non-tone bin {} magnitude {} too large",
            i,
            mag
        );
    }
}

#[test]
fn pipeline_fft_rejects_non_power_of_2() {
    let mut data = vec![(0.0f32, 0.0f32); 100];
    assert_eq!(fft_radix2(&mut data), Err(SdrError::InvalidParameter));
}

#[test]
fn pipeline_fft_single_sample() {
    let mut data = vec![(42.0f32, 7.0f32)];
    fft_radix2(&mut data).unwrap();
    assert_eq!(data[0], (42.0, 7.0));
}

// ---------------------------------------------------------------------------
// 4. Magnitude and PSD spectrum
// ---------------------------------------------------------------------------

#[test]
fn pipeline_magnitude_spectrum_from_fft() {
    let mut data = generate_dc(64);
    fft_radix2(&mut data).unwrap();

    let mut mag = vec![0.0f32; 64];
    let count = magnitude_spectrum(&data, &mut mag);
    assert_eq!(count, 64);

    // DC bin should have magnitude ~64.
    assert!((mag[0] - 64.0).abs() < 2.0);

    // Other bins should be near zero.
    for (i, &m) in mag[1..64].iter().enumerate() {
        assert!(m < 1.0, "bin {} magnitude {} should be near zero", i + 1, m);
    }
}

#[test]
fn pipeline_psd_from_fft() {
    let mut data = generate_dc(64);
    fft_radix2(&mut data).unwrap();

    let mut psd = vec![0.0f32; 64];
    let count = power_spectral_density(&data, &mut psd, -120.0);
    assert_eq!(count, 64);

    // DC bin: 10*log10(64^2) = 10*log10(4096) ~ 36 dB.
    assert!(psd[0] > 30.0, "DC PSD {} should be > 30 dB", psd[0]);

    // Zero-power bins should be at the floor.
    for i in 1..64 {
        // Allow some tolerance from FFT imprecision
        assert!(
            psd[i] <= psd[0],
            "bin {} PSD {} should be <= DC PSD {}",
            i,
            psd[i],
            psd[0]
        );
    }
}

// ---------------------------------------------------------------------------
// 5. Tensor output format and shape
// ---------------------------------------------------------------------------

#[test]
fn pipeline_tensor_2d_ri_format() {
    let samples = [(1.0f32, 2.0), (3.0, 4.0), (5.0, 6.0)];
    let mut output = [0.0f32; 6];
    let count = format_tensor_2d_ri(&samples, &mut output);

    // Shape: [1, 3, 2] -> 6 values.
    assert_eq!(count, 6);
    assert_eq!(output, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn pipeline_tensor_2d_ri_insufficient_output() {
    let samples = [(1.0f32, 2.0), (3.0, 4.0)];
    let mut output = [0.0f32; 2]; // Too small for N*2=4
    let count = format_tensor_2d_ri(&samples, &mut output);
    assert_eq!(count, 0);
}

#[test]
fn pipeline_tensor_1d_magnitude_format() {
    let mags = [5.0f32, 3.0, 1.0, 0.5];
    let mut output = [0.0f32; 4];
    let count = format_tensor_1d_mag(&mags, &mut output);

    // Shape: [1, 4] -> 4 values.
    assert_eq!(count, 4);
    assert_eq!(output, [5.0, 3.0, 1.0, 0.5]);
}

#[test]
fn pipeline_tensor_1d_output_clipping() {
    let mags = [1.0f32, 2.0, 3.0, 4.0, 5.0];
    let mut output = [0.0f32; 3]; // Only room for 3
    let count = format_tensor_1d_mag(&mags, &mut output);
    assert_eq!(count, 3);
    assert_eq!(output, [1.0, 2.0, 3.0]);
}

// ---------------------------------------------------------------------------
// 6. End-to-end pipeline: SDR -> ring buffer -> window -> FFT -> tensor
// ---------------------------------------------------------------------------

#[test]
fn pipeline_end_to_end_sdr_to_tensor() {
    let window_size = 64;
    let tone_bin = 5;

    // Step 1: Generate mock SDR IQ samples (single tone).
    let iq_samples = generate_single_tone(window_size, tone_bin);

    // Step 2: Write to ring buffer.
    let mut storage = vec![(0.0f32, 0.0f32); 128];
    let ring_buf = unsafe {
        IqRingBuffer::new(core::mem::transmute::<
            &mut [(f32, f32)],
            &'static mut [(f32, f32)],
        >(storage.as_mut_slice()))
    };
    ring_buf.write(&iq_samples);
    assert_eq!(ring_buf.available(), window_size);

    // Step 3: Read a window of samples.
    let mut window = vec![(0.0f32, 0.0f32); window_size];
    let read = ring_buf.read(&mut window, window_size);
    assert_eq!(read, window_size);

    // Step 4: Apply Hann window.
    apply_hann_window(&mut window);

    // Step 5: Compute FFT.
    fft_radix2(&mut window).unwrap();

    // Step 6: Compute magnitude spectrum.
    let mut mag = vec![0.0f32; window_size];
    let count = magnitude_spectrum(&window, &mut mag);
    assert_eq!(count, window_size);

    // The tone bin should still dominate.
    let max_bin = mag
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    assert_eq!(
        max_bin, tone_bin,
        "peak bin should be at {}, got {}",
        tone_bin, max_bin
    );

    // Step 7: Format as 1D magnitude tensor for ONNX.
    let mut tensor = vec![0.0f32; window_size];
    let tensor_count = format_tensor_1d_mag(&mag, &mut tensor);
    assert_eq!(tensor_count, window_size);

    // Verify tensor has correct shape semantics: [1, N].
    assert_eq!(tensor.len(), window_size);
    assert!(
        tensor[tone_bin] > 0.0,
        "tone bin should have positive magnitude"
    );
}

#[test]
fn pipeline_end_to_end_with_2d_ri_tensor() {
    let window_size = 32;

    // Generate DC samples, push through pipeline.
    let iq_samples = generate_dc(window_size);
    let mut window = iq_samples.clone();

    // Apply rectangular window (no-op).
    apply_window(&mut window, WindowFunction::Rectangular);

    // FFT.
    fft_radix2(&mut window).unwrap();

    // Format as 2D [re, im] tensor.
    let mut tensor = vec![0.0f32; window_size * 2];
    let count = format_tensor_2d_ri(&window, &mut tensor);
    assert_eq!(count, window_size * 2);

    // DC bin (index 0): re should be ~32, im should be ~0.
    assert!((tensor[0] - 32.0).abs() < 1.0, "DC re = {}", tensor[0]);
    assert!(tensor[1].abs() < 1.0, "DC im = {}", tensor[1]);
}

// ---------------------------------------------------------------------------
// 7. Pipeline configuration and validation
// ---------------------------------------------------------------------------

#[test]
fn pipeline_config_valid() {
    let cfg = PipelineConfig {
        window_size: 1024,
        window_function: WindowFunction::Hann,
        overlap_fraction: 0.5,
        enable_fft: true,
        psd_floor_db: -120.0,
        normalization_window: 100,
        iq_format: IqFormat::Signed8,
    };
    assert!(cfg.validate().is_ok());
    assert_eq!(cfg.stride(), 512);
}

#[test]
fn pipeline_config_invalid_window_size() {
    let cfg = PipelineConfig {
        window_size: 1000,
        window_function: WindowFunction::Hann,
        overlap_fraction: 0.5,
        enable_fft: true,
        psd_floor_db: -120.0,
        normalization_window: 100,
        iq_format: IqFormat::Signed8,
    };
    assert_eq!(cfg.validate(), Err(SdrError::InvalidParameter));
}

#[test]
fn pipeline_config_invalid_overlap() {
    let cfg = PipelineConfig {
        window_size: 1024,
        window_function: WindowFunction::Hann,
        overlap_fraction: 1.0,
        enable_fft: true,
        psd_floor_db: -120.0,
        normalization_window: 100,
        iq_format: IqFormat::Signed8,
    };
    assert_eq!(cfg.validate(), Err(SdrError::InvalidParameter));
}

// ---------------------------------------------------------------------------
// 8. Pipeline state and backpressure
// ---------------------------------------------------------------------------

#[test]
fn pipeline_state_backpressure_handling() {
    let cfg = PipelineConfig {
        window_size: 256,
        window_function: WindowFunction::Hamming,
        overlap_fraction: 0.5,
        enable_fft: true,
        psd_floor_db: -120.0,
        normalization_window: 0,
        iq_format: IqFormat::Signed16,
    };
    let mut state = PipelineState::new(cfg);

    // Initially should process.
    assert!(state.should_process());
    state.record_processed();
    assert_eq!(state.windows_processed, 1);

    // Simulate inference backpressure.
    state.inference_busy = true;
    assert!(!state.should_process());
    state.record_skipped();
    state.record_skipped();
    state.record_skipped();
    assert_eq!(state.take_skip_count(), 3);
    assert_eq!(state.windows_skipped, 0);

    // Release backpressure.
    state.inference_busy = false;
    assert!(state.should_process());
    state.record_processed();
    assert_eq!(state.windows_processed, 2);
}

// ---------------------------------------------------------------------------
// 9. Multi-device pipeline manager
// ---------------------------------------------------------------------------

#[test]
fn pipeline_manager_add_remove() {
    let mut mgr = PipelineManager::new();
    assert_eq!(mgr.active_count(), 0);

    let cfg = PipelineConfig {
        window_size: 256,
        window_function: WindowFunction::Hann,
        overlap_fraction: 0.25,
        enable_fft: true,
        psd_floor_db: -100.0,
        normalization_window: 50,
        iq_format: IqFormat::Signed8,
    };

    let idx = mgr.add_pipeline(cfg).unwrap();
    assert_eq!(idx, 0);
    assert_eq!(mgr.active_count(), 1);

    // Fill to max.
    for _ in 1..MAX_PIPELINES {
        mgr.add_pipeline(cfg).unwrap();
    }
    assert_eq!(mgr.active_count(), MAX_PIPELINES);

    // Should be full.
    assert_eq!(mgr.add_pipeline(cfg), Err(SdrError::DeviceBusy));

    // Remove one and add again.
    mgr.remove_pipeline(2).unwrap();
    assert_eq!(mgr.active_count(), MAX_PIPELINES - 1);
    let idx = mgr.add_pipeline(cfg).unwrap();
    assert_eq!(idx, 2, "should reuse slot 2");
}

#[test]
fn pipeline_manager_remove_nonexistent() {
    let mut mgr = PipelineManager::new();
    assert_eq!(mgr.remove_pipeline(0), Err(SdrError::DeviceNotFound));
}

#[test]
fn pipeline_manager_pipeline_state_access() {
    let mut mgr = PipelineManager::new();
    let cfg = PipelineConfig {
        window_size: 512,
        window_function: WindowFunction::Hamming,
        overlap_fraction: 0.5,
        enable_fft: true,
        psd_floor_db: -120.0,
        normalization_window: 100,
        iq_format: IqFormat::Signed16,
    };

    let idx = mgr.add_pipeline(cfg).unwrap();
    assert!(mgr.pipeline(idx).is_some());

    // Modify pipeline state.
    let state = mgr.pipeline_mut(idx).unwrap();
    state.record_processed();
    state.record_processed();
    assert_eq!(mgr.pipeline(idx).unwrap().windows_processed, 2);
}

// ---------------------------------------------------------------------------
// 10. Running normalizer
// ---------------------------------------------------------------------------

#[test]
fn pipeline_normalizer_constant_input() {
    let mut norm = RunningNormalizer::new(100);
    for _ in 0..50 {
        norm.update(5.0);
    }

    assert!((norm.mean() - 5.0).abs() < 0.01);
    let normalized = norm.normalize(5.0);
    assert!(
        normalized.abs() < 0.1,
        "constant input normalized to {}",
        normalized
    );
}

#[test]
fn pipeline_normalizer_buffer() {
    let mut norm = RunningNormalizer::new(100);
    for i in 0..50 {
        norm.update(i as f32);
    }

    let mean = norm.mean();
    let mut buf = [mean, mean, mean];
    norm.normalize_buffer(&mut buf);
    for v in &buf {
        assert!(v.abs() < 0.1, "mean value normalized to {}", v);
    }
}

#[test]
fn pipeline_normalizer_reset() {
    let mut norm = RunningNormalizer::new(50);
    for _ in 0..20 {
        norm.update(42.0);
    }
    assert!((norm.mean() - 42.0).abs() < 0.01);

    norm.reset();
    assert!((norm.mean() - 0.0).abs() < 0.001);
}
