//! SDR-to-ONNX inference pipeline.
//!
//! Implements a lock-free IQ ring buffer, windowing functions (Hann, Hamming,
//! rectangular), radix-2 FFT, magnitude/PSD computation, tensor formatting,
//! and a continuous inference pipeline with backpressure handling.

use crate::{IqFormat, SdrError};
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

// ────────────── Ring Buffer ──────────────

/// Lock-free IQ ring buffer with lossy overwrite semantics.
/// Uses atomic indices for single-producer single-consumer operation.
pub struct IqRingBuffer {
    /// Storage for I/Q sample pairs.
    buffer: &'static mut [(f32, f32)],
    /// Capacity (number of complex samples).
    capacity: usize,
    /// Write index (wraps around).
    write_idx: AtomicU32,
    /// Read index (wraps around).
    read_idx: AtomicU32,
    /// Overflow counter.
    overflow_count: AtomicU64,
}

impl IqRingBuffer {
    /// Creates a new ring buffer backed by the provided storage.
    ///
    /// # Safety
    /// The provided buffer must be exclusively owned by this ring buffer.
    pub unsafe fn new(buffer: &'static mut [(f32, f32)]) -> Self {
        let capacity = buffer.len();
        Self {
            buffer,
            capacity,
            write_idx: AtomicU32::new(0),
            read_idx: AtomicU32::new(0),
            overflow_count: AtomicU64::new(0),
        }
    }

    /// Returns the buffer capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns the number of available samples to read.
    pub fn available(&self) -> usize {
        let w = self.write_idx.load(Ordering::Acquire) as usize;
        let r = self.read_idx.load(Ordering::Acquire) as usize;
        if w >= r {
            w - r
        } else {
            self.capacity - r + w
        }
    }

    /// Writes samples into the ring buffer. Overwrites oldest data on overflow (lossy).
    /// Returns the number of samples written.
    pub fn write(&self, samples: &[(f32, f32)]) -> usize {
        let mut w = self.write_idx.load(Ordering::Relaxed) as usize;
        let r = self.read_idx.load(Ordering::Acquire) as usize;

        for &sample in samples {
            let next_w = (w + 1) % self.capacity;
            if next_w == r {
                // Overflow: advance read pointer to make room (lossy overwrite)
                self.overflow_count.fetch_add(1, Ordering::Relaxed);
                self.read_idx
                    .store(((r + 1) % self.capacity) as u32, Ordering::Release);
            }
            // Safety: single producer, and we just checked bounds
            unsafe {
                let ptr = self.buffer.as_ptr() as *mut (f32, f32);
                core::ptr::write(ptr.add(w), sample);
            }
            w = next_w;
        }

        self.write_idx.store(w as u32, Ordering::Release);
        samples.len()
    }

    /// Reads up to `count` samples into the output buffer.
    /// Returns the number of samples actually read.
    pub fn read(&self, output: &mut [(f32, f32)], count: usize) -> usize {
        let w = self.write_idx.load(Ordering::Acquire) as usize;
        let mut r = self.read_idx.load(Ordering::Relaxed) as usize;

        let avail = if w >= r { w - r } else { self.capacity - r + w };

        let to_read = count.min(avail).min(output.len());
        for out in output.iter_mut().take(to_read) {
            *out = unsafe {
                let ptr = self.buffer.as_ptr();
                core::ptr::read(ptr.add(r))
            };
            r = (r + 1) % self.capacity;
        }

        self.read_idx.store(r as u32, Ordering::Release);
        to_read
    }

    /// Returns and resets the overflow counter.
    pub fn take_overflow_count(&self) -> u64 {
        self.overflow_count.swap(0, Ordering::Relaxed)
    }

    /// Returns the current overflow count without resetting.
    pub fn overflow_count(&self) -> u64 {
        self.overflow_count.load(Ordering::Relaxed)
    }
}

// ────────────── Window Functions ──────────────

/// Applies a Hann window to a buffer of samples in-place.
/// w(n) = 0.5 * (1 - cos(2*pi*n / (N-1)))
pub fn apply_hann_window(samples: &mut [(f32, f32)]) {
    let n = samples.len();
    if n <= 1 {
        return;
    }
    let n_minus_1 = (n - 1) as f32;
    for (i, sample) in samples.iter_mut().enumerate() {
        let w = 0.5 * (1.0 - cos_approx(2.0 * core::f32::consts::PI * i as f32 / n_minus_1));
        sample.0 *= w;
        sample.1 *= w;
    }
}

/// Applies a Hamming window to a buffer of samples in-place.
/// w(n) = 0.54 - 0.46 * cos(2*pi*n / (N-1))
pub fn apply_hamming_window(samples: &mut [(f32, f32)]) {
    let n = samples.len();
    if n <= 1 {
        return;
    }
    let n_minus_1 = (n - 1) as f32;
    for (i, sample) in samples.iter_mut().enumerate() {
        let w = 0.54 - 0.46 * cos_approx(2.0 * core::f32::consts::PI * i as f32 / n_minus_1);
        sample.0 *= w;
        sample.1 *= w;
    }
}

/// Applies a rectangular (passthrough) window — no-op.
pub fn apply_rectangular_window(_samples: &mut [(f32, f32)]) {
    // Identity transform
}

/// Window function type for configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowFunction {
    Hann,
    Hamming,
    Rectangular,
}

/// Applies the selected window function.
pub fn apply_window(samples: &mut [(f32, f32)], window: WindowFunction) {
    match window {
        WindowFunction::Hann => apply_hann_window(samples),
        WindowFunction::Hamming => apply_hamming_window(samples),
        WindowFunction::Rectangular => apply_rectangular_window(samples),
    }
}

/// Computes the overlap stride for a given window size and overlap fraction.
/// overlap_fraction: 0.0 (no overlap) to <1.0 (nearly full overlap).
pub fn compute_stride(window_size: usize, overlap_fraction: f32) -> usize {
    let stride = (window_size as f32 * (1.0 - overlap_fraction)) as usize;
    if stride == 0 {
        1
    } else {
        stride
    }
}

// ────────────── FFT ──────────────

/// Performs an in-place radix-2 Cooley-Tukey FFT on complex data.
/// Input length must be a power of 2.
pub fn fft_radix2(data: &mut [(f32, f32)]) -> Result<(), SdrError> {
    let n = data.len();
    if n == 0 || n & (n - 1) != 0 {
        return Err(SdrError::InvalidParameter);
    }
    if n == 1 {
        return Ok(());
    }

    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 0..n {
        if i < j {
            data.swap(i, j);
        }
        let mut m = n >> 1;
        while m >= 1 && j >= m {
            j -= m;
            m >>= 1;
        }
        j += m;
    }

    // Butterfly stages
    let mut step = 2;
    while step <= n {
        let half = step / 2;
        let angle_step = -2.0 * core::f32::consts::PI / step as f32;
        for k in 0..half {
            let angle = angle_step * k as f32;
            let tw_re = cos_approx(angle);
            let tw_im = sin_approx(angle);

            let mut i = k;
            while i < n {
                let j = i + half;
                let t_re = data[j].0 * tw_re - data[j].1 * tw_im;
                let t_im = data[j].0 * tw_im + data[j].1 * tw_re;
                data[j].0 = data[i].0 - t_re;
                data[j].1 = data[i].1 - t_im;
                data[i].0 += t_re;
                data[i].1 += t_im;
                i += step;
            }
        }
        step <<= 1;
    }

    Ok(())
}

// ────────────── Spectrum Computation ──────────────

/// Computes magnitude spectrum: sqrt(re^2 + im^2) for each bin.
pub fn magnitude_spectrum(fft_output: &[(f32, f32)], output: &mut [f32]) -> usize {
    let count = fft_output.len().min(output.len());
    for i in 0..count {
        let re = fft_output[i].0;
        let im = fft_output[i].1;
        output[i] = sqrt_approx(re * re + im * im);
    }
    count
}

/// Computes power spectral density: 10 * log10(re^2 + im^2) with floor clamping.
pub fn power_spectral_density(
    fft_output: &[(f32, f32)],
    output: &mut [f32],
    floor_db: f32,
) -> usize {
    let count = fft_output.len().min(output.len());
    for i in 0..count {
        let re = fft_output[i].0;
        let im = fft_output[i].1;
        let power = re * re + im * im;
        let db = if power > 0.0 {
            10.0 * log10_approx(power)
        } else {
            floor_db
        };
        output[i] = if db < floor_db { floor_db } else { db };
    }
    count
}

// ────────────── Tensor Formatting ──────────────

/// Formats IQ data as a 2D real/imaginary tensor [1, N, 2].
/// Output layout: [re_0, im_0, re_1, im_1, ...] as f32.
/// Returns the number of f32 values written (= N * 2).
pub fn format_tensor_2d_ri(samples: &[(f32, f32)], output: &mut [f32]) -> usize {
    let n = samples.len();
    let needed = n * 2;
    if output.len() < needed {
        return 0;
    }
    for i in 0..n {
        output[i * 2] = samples[i].0;
        output[i * 2 + 1] = samples[i].1;
    }
    needed
}

/// Formats magnitude data as a 1D tensor [1, N].
/// Input: pre-computed magnitude values.
/// Returns the number of f32 values written (= N).
pub fn format_tensor_1d_mag(magnitudes: &[f32], output: &mut [f32]) -> usize {
    let n = magnitudes.len().min(output.len());
    output[..n].copy_from_slice(&magnitudes[..n]);
    n
}

// ────────────── Normalization ──────────────

/// Running normalizer that tracks mean and variance over a configurable window.
pub struct RunningNormalizer {
    /// Running sum for mean calculation.
    sum: f32,
    /// Running sum of squares for variance.
    sum_sq: f32,
    /// Number of samples in the window.
    count: u32,
    /// Maximum window size.
    window_size: u32,
}

impl RunningNormalizer {
    /// Creates a new normalizer with the given window size.
    pub fn new(window_size: u32) -> Self {
        Self {
            sum: 0.0,
            sum_sq: 0.0,
            count: 0,
            window_size,
        }
    }

    /// Updates statistics with a new value.
    pub fn update(&mut self, value: f32) {
        if self.count >= self.window_size {
            // Simple decay: reduce influence of old samples
            let decay = (self.window_size - 1) as f32 / self.window_size as f32;
            self.sum *= decay;
            self.sum_sq *= decay;
            self.count = self.window_size - 1;
        }
        self.sum += value;
        self.sum_sq += value * value;
        self.count += 1;
    }

    /// Returns the current mean.
    pub fn mean(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        self.sum / self.count as f32
    }

    /// Returns the current standard deviation.
    pub fn std_dev(&self) -> f32 {
        if self.count < 2 {
            return 1.0;
        }
        let mean = self.mean();
        let variance = self.sum_sq / self.count as f32 - mean * mean;
        if variance <= 0.0 {
            1.0
        } else {
            sqrt_approx(variance)
        }
    }

    /// Normalizes a value to zero-mean, unit-variance.
    pub fn normalize(&self, value: f32) -> f32 {
        (value - self.mean()) / self.std_dev()
    }

    /// Normalizes a buffer of values in-place.
    pub fn normalize_buffer(&self, buffer: &mut [f32]) {
        let mean = self.mean();
        let std = self.std_dev();
        for v in buffer.iter_mut() {
            *v = (*v - mean) / std;
        }
    }

    /// Resets the normalizer state.
    pub fn reset(&mut self) {
        self.sum = 0.0;
        self.sum_sq = 0.0;
        self.count = 0;
    }
}

// ────────────── Pipeline Configuration ──────────────

/// Pipeline configuration for SDR-to-ONNX inference.
#[derive(Debug, Clone, Copy)]
pub struct PipelineConfig {
    /// Window size for FFT (must be power of 2).
    pub window_size: usize,
    /// Window function to apply.
    pub window_function: WindowFunction,
    /// Overlap fraction (0.0 to <1.0).
    pub overlap_fraction: f32,
    /// Whether to compute FFT (false = time-domain passthrough).
    pub enable_fft: bool,
    /// PSD floor in dB.
    pub psd_floor_db: f32,
    /// Normalization window size (0 = disabled).
    pub normalization_window: u32,
    /// IQ format from the SDR device.
    pub iq_format: IqFormat,
}

impl PipelineConfig {
    /// Validates the pipeline configuration.
    pub fn validate(&self) -> Result<(), SdrError> {
        if self.window_size == 0 || (self.window_size & (self.window_size - 1)) != 0 {
            return Err(SdrError::InvalidParameter);
        }
        if self.overlap_fraction < 0.0 || self.overlap_fraction >= 1.0 {
            return Err(SdrError::InvalidParameter);
        }
        Ok(())
    }

    /// Returns the stride in samples.
    pub fn stride(&self) -> usize {
        compute_stride(self.window_size, self.overlap_fraction)
    }
}

/// Pipeline state tracking.
#[derive(Debug)]
pub struct PipelineState {
    /// Total windows processed.
    pub windows_processed: u64,
    /// Windows skipped due to backpressure.
    pub windows_skipped: u64,
    /// Whether inference is currently running (backpressure signal).
    pub inference_busy: bool,
    /// Configuration for this pipeline.
    pub config: PipelineConfig,
}

impl PipelineState {
    /// Creates a new pipeline state with the given configuration.
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            windows_processed: 0,
            windows_skipped: 0,
            inference_busy: false,
            config,
        }
    }

    /// Checks if a window should be processed or skipped (backpressure).
    pub fn should_process(&self) -> bool {
        !self.inference_busy
    }

    /// Records a processed window.
    pub fn record_processed(&mut self) {
        self.windows_processed += 1;
    }

    /// Records a skipped window.
    pub fn record_skipped(&mut self) {
        self.windows_skipped += 1;
    }

    /// Returns and resets the skip counter.
    pub fn take_skip_count(&mut self) -> u64 {
        let count = self.windows_skipped;
        self.windows_skipped = 0;
        count
    }
}

// ────────────── Multi-device Pipeline Manager ──────────────

/// Maximum number of concurrent SDR pipelines.
pub const MAX_PIPELINES: usize = 4;

/// Multi-device pipeline manager.
pub struct PipelineManager {
    /// Active pipeline states (None = slot unused).
    pipelines: [Option<PipelineState>; MAX_PIPELINES],
    /// Number of active pipelines.
    active_count: usize,
}

impl Default for PipelineManager {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineManager {
    /// Creates a new empty pipeline manager.
    pub const fn new() -> Self {
        Self {
            pipelines: [None, None, None, None],
            active_count: 0,
        }
    }

    /// Adds a pipeline with the given configuration. Returns the pipeline index.
    pub fn add_pipeline(&mut self, config: PipelineConfig) -> Result<usize, SdrError> {
        config.validate()?;
        for (i, slot) in self.pipelines.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(PipelineState::new(config));
                self.active_count += 1;
                return Ok(i);
            }
        }
        Err(SdrError::DeviceBusy)
    }

    /// Removes a pipeline by index.
    pub fn remove_pipeline(&mut self, index: usize) -> Result<(), SdrError> {
        if index >= MAX_PIPELINES || self.pipelines[index].is_none() {
            return Err(SdrError::DeviceNotFound);
        }
        self.pipelines[index] = None;
        self.active_count -= 1;
        Ok(())
    }

    /// Returns a reference to a pipeline state.
    pub fn pipeline(&self, index: usize) -> Option<&PipelineState> {
        if index >= MAX_PIPELINES {
            return None;
        }
        self.pipelines[index].as_ref()
    }

    /// Returns a mutable reference to a pipeline state.
    pub fn pipeline_mut(&mut self, index: usize) -> Option<&mut PipelineState> {
        if index >= MAX_PIPELINES {
            return None;
        }
        self.pipelines[index].as_mut()
    }

    /// Returns the number of active pipelines.
    pub fn active_count(&self) -> usize {
        self.active_count
    }
}

// ────────────── Math approximations (no_std) ──────────────

/// Approximate cosine using Taylor series (good for |x| < 2*PI).
fn cos_approx(x: f32) -> f32 {
    // Reduce x to [-PI, PI]
    let mut x = x % (2.0 * core::f32::consts::PI);
    if x > core::f32::consts::PI {
        x -= 2.0 * core::f32::consts::PI;
    }
    if x < -core::f32::consts::PI {
        x += 2.0 * core::f32::consts::PI;
    }
    // Taylor series: cos(x) ≈ 1 - x²/2! + x⁴/4! - x⁶/6! + x⁸/8! - x¹⁰/10!
    let x2 = x * x;
    let x4 = x2 * x2;
    let x6 = x4 * x2;
    let x8 = x6 * x2;
    let x10 = x8 * x2;
    1.0 - x2 / 2.0 + x4 / 24.0 - x6 / 720.0 + x8 / 40320.0 - x10 / 3628800.0
}

/// Approximate sine using Taylor series.
fn sin_approx(x: f32) -> f32 {
    // sin(x) = cos(x - PI/2)
    cos_approx(x - core::f32::consts::FRAC_PI_2)
}

/// Approximate square root using Newton's method.
fn sqrt_approx(x: f32) -> f32 {
    if x <= 0.0 {
        return 0.0;
    }
    // Use bit manipulation for initial guess (Quake III fast inverse sqrt variant)
    let mut guess = x;
    let x_half = 0.5 * x;
    let mut i = guess.to_bits();
    i = 0x5f3759df - (i >> 1); // Fast inverse sqrt magic number
    guess = f32::from_bits(i);
    // Newton iterations for 1/sqrt(x)
    guess = guess * (1.5 - x_half * guess * guess);
    guess = guess * (1.5 - x_half * guess * guess);
    // sqrt(x) = x * (1/sqrt(x))
    x * guess
}

/// Approximate log10 using natural log approximation.
fn log10_approx(x: f32) -> f32 {
    if x <= 0.0 {
        return -120.0; // Floor value
    }
    // log10(x) = ln(x) / ln(10)
    // ln(x) approximated via: ln(x) = (exponent) * ln(2) + ln(mantissa)
    let bits = x.to_bits();
    let exponent = ((bits >> 23) & 0xFF) as i32 - 127;
    // Reconstruct mantissa in [1, 2)
    let mantissa_bits = (bits & 0x007FFFFF) | 0x3F800000;
    let mantissa = f32::from_bits(mantissa_bits);
    // ln(mantissa) ≈ (mantissa - 1) - 0.5*(mantissa-1)^2 + ... (Padé approximation)
    let m = mantissa - 1.0;
    let ln_mantissa = m * (1.0 + m * (-0.5 + m * (1.0 / 3.0 - m * 0.25)));
    let ln_x = exponent as f32 * core::f32::consts::LN_2 + ln_mantissa;
    ln_x / core::f32::consts::LN_10
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // ── Ring Buffer Tests ──

    #[test]
    fn test_ring_buffer_write_read() {
        let mut storage = vec![(0.0f32, 0.0f32); 16];
        let buf = unsafe {
            IqRingBuffer::new(core::mem::transmute::<
                &mut [(f32, f32)],
                &'static mut [(f32, f32)],
            >(storage.as_mut_slice()))
        };

        let samples = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)];
        let written = buf.write(&samples);
        assert_eq!(written, 3);
        assert_eq!(buf.available(), 3);

        let mut out = [(0.0f32, 0.0f32); 4];
        let read = buf.read(&mut out, 4);
        assert_eq!(read, 3);
        assert_eq!(out[0], (1.0, 2.0));
        assert_eq!(out[1], (3.0, 4.0));
        assert_eq!(out[2], (5.0, 6.0));
    }

    #[test]
    fn test_ring_buffer_overflow() {
        let mut storage = vec![(0.0f32, 0.0f32); 4]; // capacity 4, usable 3
        let buf = unsafe {
            IqRingBuffer::new(core::mem::transmute::<
                &mut [(f32, f32)],
                &'static mut [(f32, f32)],
            >(storage.as_mut_slice()))
        };

        // Write 5 samples into buffer of effective capacity 3
        let samples: [(f32, f32); 5] = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0), (4.0, 4.0), (5.0, 5.0)];
        buf.write(&samples);

        // Should have overflowed
        assert!(buf.overflow_count() > 0);

        let count = buf.take_overflow_count();
        assert!(count > 0);
        // After take, counter resets
        assert_eq!(buf.overflow_count(), 0);
    }

    #[test]
    fn test_ring_buffer_empty_read() {
        let mut storage = vec![(0.0f32, 0.0f32); 8];
        let buf = unsafe {
            IqRingBuffer::new(core::mem::transmute::<
                &mut [(f32, f32)],
                &'static mut [(f32, f32)],
            >(storage.as_mut_slice()))
        };

        let mut out = [(0.0f32, 0.0f32); 4];
        let read = buf.read(&mut out, 4);
        assert_eq!(read, 0);
    }

    // ── Window Function Tests ──

    #[test]
    fn test_hann_window() {
        // For N=5: w = [0, 0.5, 1.0, 0.5, 0]
        let mut samples = [
            (1.0f32, 1.0f32),
            (1.0, 1.0),
            (1.0, 1.0),
            (1.0, 1.0),
            (1.0, 1.0),
        ];
        apply_hann_window(&mut samples);

        // First and last should be ~0
        assert!(samples[0].0.abs() < 0.02);
        assert!(samples[4].0.abs() < 0.02);
        // Middle should be ~1.0 (Taylor series cos has ~2% error at PI)
        assert!((samples[2].0 - 1.0).abs() < 0.05);
        // Quarter points should be ~0.5
        assert!((samples[1].0 - 0.5).abs() < 0.05);
        assert!((samples[3].0 - 0.5).abs() < 0.05);
    }

    #[test]
    fn test_hamming_window() {
        // For N=5: first/last should be ~0.08, middle should be ~1.0
        let mut samples = [
            (1.0f32, 1.0f32),
            (1.0, 1.0),
            (1.0, 1.0),
            (1.0, 1.0),
            (1.0, 1.0),
        ];
        apply_hamming_window(&mut samples);

        // Hamming: w(0) = 0.54 - 0.46 = 0.08
        assert!((samples[0].0 - 0.08).abs() < 0.02);
        // w(2) = 0.54 + 0.46 = 1.0
        assert!((samples[2].0 - 1.0).abs() < 0.02);
    }

    #[test]
    fn test_rectangular_window() {
        let mut samples = [(1.0f32, 2.0f32), (3.0, 4.0)];
        apply_rectangular_window(&mut samples);
        // No change
        assert_eq!(samples[0], (1.0, 2.0));
        assert_eq!(samples[1], (3.0, 4.0));
    }

    #[test]
    fn test_stride_computation() {
        assert_eq!(compute_stride(1024, 0.5), 512); // 50% overlap
        assert_eq!(compute_stride(1024, 0.0), 1024); // No overlap
        assert_eq!(compute_stride(1024, 0.75), 256); // 75% overlap
        assert_eq!(compute_stride(1, 0.99), 1); // Min stride is 1
    }

    // ── FFT Tests ──

    #[test]
    fn test_fft_dc() {
        // DC input: all samples = (1, 0) → FFT bin 0 should have magnitude N
        let mut data = [(1.0f32, 0.0f32); 8];
        fft_radix2(&mut data).unwrap();

        // Bin 0 (DC) should be approximately (8, 0)
        assert!((data[0].0 - 8.0).abs() < 0.1);
        assert!(data[0].1.abs() < 0.1);

        // All other bins should be approximately 0
        for (i, d) in data[1..8].iter().enumerate() {
            assert!(d.0.abs() < 0.1, "bin {} re = {}", i + 1, d.0);
            assert!(d.1.abs() < 0.1, "bin {} im = {}", i + 1, d.1);
        }
    }

    #[test]
    fn test_fft_single_tone() {
        // Single tone at bin 1: x[n] = exp(j*2*pi*n/N) = (cos, sin)
        let n = 8;
        let mut data = [(0.0f32, 0.0f32); 8];
        for (i, d) in data[..n].iter_mut().enumerate() {
            let angle = 2.0 * core::f32::consts::PI * i as f32 / n as f32;
            *d = (cos_approx(angle), sin_approx(angle));
        }
        fft_radix2(&mut data).unwrap();

        // Bin 1 should have magnitude N=8
        let mag1 = sqrt_approx(data[1].0 * data[1].0 + data[1].1 * data[1].1);
        assert!((mag1 - 8.0).abs() < 0.5, "bin 1 magnitude = {}", mag1);

        // Other bins should be near 0
        for i in [0, 2, 3, 4, 5, 6, 7] {
            let mag = sqrt_approx(data[i].0 * data[i].0 + data[i].1 * data[i].1);
            assert!(mag < 0.5, "bin {} magnitude = {}", i, mag);
        }
    }

    #[test]
    fn test_fft_non_power_of_2_rejected() {
        let mut data = [(0.0f32, 0.0f32); 6];
        assert_eq!(fft_radix2(&mut data), Err(SdrError::InvalidParameter));
    }

    #[test]
    fn test_fft_single_sample() {
        let mut data = [(42.0f32, 7.0f32)];
        fft_radix2(&mut data).unwrap();
        assert_eq!(data[0], (42.0, 7.0));
    }

    // ── Spectrum Tests ──

    #[test]
    fn test_magnitude_spectrum() {
        let fft_out = [(3.0f32, 4.0f32), (0.0, 1.0), (1.0, 0.0)];
        let mut mag = [0.0f32; 3];
        let count = magnitude_spectrum(&fft_out, &mut mag);
        assert_eq!(count, 3);
        assert!((mag[0] - 5.0).abs() < 0.01); // sqrt(9+16) = 5
        assert!((mag[1] - 1.0).abs() < 0.01); // sqrt(0+1) = 1
        assert!((mag[2] - 1.0).abs() < 0.01); // sqrt(1+0) = 1
    }

    #[test]
    fn test_psd_computation() {
        let fft_out = [(10.0f32, 0.0f32), (0.0, 0.0)];
        let mut psd = [0.0f32; 2];
        let count = power_spectral_density(&fft_out, &mut psd, -120.0);
        assert_eq!(count, 2);
        // 10*log10(100) = 20 dB
        assert!((psd[0] - 20.0).abs() < 1.0);
        // Zero power → floor
        assert_eq!(psd[1], -120.0);
    }

    // ── Tensor Formatting Tests ──

    #[test]
    fn test_format_tensor_2d_ri() {
        let samples = [(1.0f32, 2.0f32), (3.0, 4.0), (5.0, 6.0)];
        let mut output = [0.0f32; 6];
        let count = format_tensor_2d_ri(&samples, &mut output);
        assert_eq!(count, 6); // N*2
        assert_eq!(output, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_format_tensor_2d_ri_insufficient_output() {
        let samples = [(1.0f32, 2.0f32), (3.0, 4.0)];
        let mut output = [0.0f32; 2]; // Too small
        let count = format_tensor_2d_ri(&samples, &mut output);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_format_tensor_1d_mag() {
        let mags = [1.0f32, 2.0, 3.0];
        let mut output = [0.0f32; 3];
        let count = format_tensor_1d_mag(&mags, &mut output);
        assert_eq!(count, 3);
        assert_eq!(output, [1.0, 2.0, 3.0]);
    }

    // ── Normalization Tests ──

    #[test]
    fn test_normalizer() {
        let mut norm = RunningNormalizer::new(100);
        // Feed constant value = 5.0
        for _ in 0..10 {
            norm.update(5.0);
        }
        assert!((norm.mean() - 5.0).abs() < 0.01);

        // Normalized 5.0 should be ~0 (zero-mean)
        let n = norm.normalize(5.0);
        assert!(n.abs() < 0.1);
    }

    #[test]
    fn test_normalizer_buffer() {
        let mut norm = RunningNormalizer::new(100);
        for i in 0..50 {
            norm.update(i as f32);
        }
        let mean = norm.mean();
        let mut buf = [mean, mean, mean];
        norm.normalize_buffer(&mut buf);
        // All values equal to mean should normalize to ~0
        for v in &buf {
            assert!(v.abs() < 0.1);
        }
    }

    // ── Pipeline Config Tests ──

    #[test]
    fn test_pipeline_config_valid() {
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
    fn test_pipeline_config_invalid_window_size() {
        let cfg = PipelineConfig {
            window_size: 1000, // Not power of 2
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
    fn test_pipeline_config_invalid_overlap() {
        let cfg = PipelineConfig {
            window_size: 1024,
            window_function: WindowFunction::Hann,
            overlap_fraction: 1.0, // Invalid
            enable_fft: true,
            psd_floor_db: -120.0,
            normalization_window: 100,
            iq_format: IqFormat::Signed8,
        };
        assert_eq!(cfg.validate(), Err(SdrError::InvalidParameter));
    }

    // ── Pipeline State Tests ──

    #[test]
    fn test_pipeline_state_backpressure() {
        let cfg = PipelineConfig {
            window_size: 1024,
            window_function: WindowFunction::Hann,
            overlap_fraction: 0.5,
            enable_fft: true,
            psd_floor_db: -120.0,
            normalization_window: 0,
            iq_format: IqFormat::Signed8,
        };
        let mut state = PipelineState::new(cfg);

        assert!(state.should_process());
        state.record_processed();
        assert_eq!(state.windows_processed, 1);

        // Simulate backpressure
        state.inference_busy = true;
        assert!(!state.should_process());
        state.record_skipped();
        state.record_skipped();
        assert_eq!(state.take_skip_count(), 2);
        assert_eq!(state.windows_skipped, 0); // Reset after take
    }

    // ── Pipeline Manager Tests ──

    #[test]
    fn test_pipeline_manager() {
        let mut mgr = PipelineManager::new();
        assert_eq!(mgr.active_count(), 0);

        let cfg = PipelineConfig {
            window_size: 256,
            window_function: WindowFunction::Hamming,
            overlap_fraction: 0.25,
            enable_fft: true,
            psd_floor_db: -100.0,
            normalization_window: 50,
            iq_format: IqFormat::Signed16,
        };

        let idx = mgr.add_pipeline(cfg).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(mgr.active_count(), 1);
        assert!(mgr.pipeline(0).is_some());

        // Add up to max
        let _ = mgr.add_pipeline(cfg).unwrap();
        let _ = mgr.add_pipeline(cfg).unwrap();
        let _ = mgr.add_pipeline(cfg).unwrap();
        assert_eq!(mgr.active_count(), 4);

        // Should be full
        assert_eq!(mgr.add_pipeline(cfg), Err(SdrError::DeviceBusy));

        // Remove one
        mgr.remove_pipeline(1).unwrap();
        assert_eq!(mgr.active_count(), 3);
        assert!(mgr.pipeline(1).is_none());

        // Now can add again
        let idx = mgr.add_pipeline(cfg).unwrap();
        assert_eq!(idx, 1); // Reuses slot
    }

    // ── Math Approximation Tests ──

    #[test]
    fn test_cos_approx() {
        assert!((cos_approx(0.0) - 1.0).abs() < 0.001);
        assert!(cos_approx(core::f32::consts::PI).abs() - 1.0 < 0.01);
        assert!(cos_approx(core::f32::consts::FRAC_PI_2).abs() < 0.01);
    }

    #[test]
    fn test_sin_approx() {
        assert!(sin_approx(0.0).abs() < 0.01);
        assert!((sin_approx(core::f32::consts::FRAC_PI_2) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_sqrt_approx() {
        assert!((sqrt_approx(4.0) - 2.0).abs() < 0.01);
        assert!((sqrt_approx(9.0) - 3.0).abs() < 0.01);
        assert!((sqrt_approx(2.0) - 1.414).abs() < 0.02);
        assert_eq!(sqrt_approx(0.0), 0.0);
    }

    #[test]
    fn test_log10_approx() {
        assert!((log10_approx(100.0) - 2.0).abs() < 0.1);
        assert!((log10_approx(1000.0) - 3.0).abs() < 0.1);
        assert!((log10_approx(1.0) - 0.0).abs() < 0.1);
    }
}
