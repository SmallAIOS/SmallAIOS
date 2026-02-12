// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Image preprocessing utilities for ONNX inference input.
//!
//! Provides `#![no_std]`-compatible functions for:
//! - YUV422 to RGB888 conversion
//! - Bilinear image resize
//! - Float32 normalization (ImageNet-style)
//!
//! These run on the CPU and operate on byte slices. For zero-copy pipelines,
//! the CSI receiver DMA buffers can be passed directly.

#![allow(dead_code)]

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

// ---------------------------------------------------------------------------
// YUV422 to RGB888 conversion
// ---------------------------------------------------------------------------

/// Convert a YUV422 (YUYV interleaved) image to RGB888.
///
/// Input format: [Y0, U0, Y1, V0, Y2, U1, Y3, V1, ...]
/// where each (Y, U, V) group encodes 2 pixels sharing the same U/V.
///
/// Output format: [R0, G0, B0, R1, G1, B1, ...]
///
/// # Parameters
///
/// - `yuv`: Input YUV422 data (2 bytes per pixel = width * height * 2).
/// - `width`: Image width in pixels (must be even).
/// - `height`: Image height in pixels.
///
/// # Returns
///
/// RGB888 byte vector of length `width * height * 3`.
pub fn yuv422_to_rgb888(yuv: &[u8], width: usize, height: usize) -> Vec<u8> {
    let expected_len = width * height * 2;
    if yuv.len() < expected_len || width == 0 || height == 0 || !width.is_multiple_of(2) {
        return Vec::new();
    }

    let mut rgb = vec![0u8; width * height * 3];

    let mut src_idx = 0;
    let mut dst_idx = 0;

    for _row in 0..height {
        for _pair in 0..(width / 2) {
            let y0 = yuv[src_idx] as i32;
            let u = yuv[src_idx + 1] as i32;
            let y1 = yuv[src_idx + 2] as i32;
            let v = yuv[src_idx + 3] as i32;
            src_idx += 4;

            // BT.601 conversion (integer approximation).
            let c0 = y0 - 16;
            let c1 = y1 - 16;
            let d = u - 128;
            let e = v - 128;

            // Pixel 0.
            rgb[dst_idx] = clamp_u8((298 * c0 + 409 * e + 128) >> 8);
            rgb[dst_idx + 1] = clamp_u8((298 * c0 - 100 * d - 208 * e + 128) >> 8);
            rgb[dst_idx + 2] = clamp_u8((298 * c0 + 516 * d + 128) >> 8);
            dst_idx += 3;

            // Pixel 1.
            rgb[dst_idx] = clamp_u8((298 * c1 + 409 * e + 128) >> 8);
            rgb[dst_idx + 1] = clamp_u8((298 * c1 - 100 * d - 208 * e + 128) >> 8);
            rgb[dst_idx + 2] = clamp_u8((298 * c1 + 516 * d + 128) >> 8);
            dst_idx += 3;
        }
    }

    rgb
}

/// Clamp an i32 to u8 range.
#[inline(always)]
fn clamp_u8(val: i32) -> u8 {
    if val < 0 {
        0
    } else if val > 255 {
        255
    } else {
        val as u8
    }
}

// ---------------------------------------------------------------------------
// Bilinear image resize
// ---------------------------------------------------------------------------

/// Resize an RGB888 image using bilinear interpolation.
///
/// # Parameters
///
/// - `src`: Source RGB888 data.
/// - `src_w`, `src_h`: Source dimensions.
/// - `dst_w`, `dst_h`: Target dimensions.
///
/// # Returns
///
/// Resized RGB888 byte vector of length `dst_w * dst_h * 3`.
pub fn resize_bilinear(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    dst_w: usize,
    dst_h: usize,
) -> Vec<u8> {
    if src.len() < src_w * src_h * 3 || dst_w == 0 || dst_h == 0 || src_w == 0 || src_h == 0 {
        return Vec::new();
    }

    let mut dst = vec![0u8; dst_w * dst_h * 3];

    // Fixed-point scaling factors (16.16).
    let x_ratio = if dst_w > 1 {
        ((src_w - 1) << 16) / (dst_w - 1)
    } else {
        0
    };
    let y_ratio = if dst_h > 1 {
        ((src_h - 1) << 16) / (dst_h - 1)
    } else {
        0
    };

    for dy in 0..dst_h {
        let y_fp = dy * y_ratio;
        let y0 = y_fp >> 16;
        let y1 = (y0 + 1).min(src_h - 1);
        let y_frac = (y_fp & 0xFFFF) as u32;

        for dx in 0..dst_w {
            let x_fp = dx * x_ratio;
            let x0 = x_fp >> 16;
            let x1 = (x0 + 1).min(src_w - 1);
            let x_frac = (x_fp & 0xFFFF) as u32;

            let dst_off = (dy * dst_w + dx) * 3;

            for c in 0..3 {
                let p00 = src[(y0 * src_w + x0) * 3 + c] as u32;
                let p10 = src[(y0 * src_w + x1) * 3 + c] as u32;
                let p01 = src[(y1 * src_w + x0) * 3 + c] as u32;
                let p11 = src[(y1 * src_w + x1) * 3 + c] as u32;

                // Bilinear interpolation in fixed-point (u64 to avoid overflow).
                let top = p00 as u64 * (65536 - x_frac) as u64 + p10 as u64 * x_frac as u64;
                let bot = p01 as u64 * (65536 - x_frac) as u64 + p11 as u64 * x_frac as u64;
                let val = ((top * (65536 - y_frac) as u64 + bot * y_frac as u64) >> 32) as u32;

                dst[dst_off + c] = val.min(255) as u8;
            }
        }
    }

    dst
}

// ---------------------------------------------------------------------------
// Float32 normalization
// ---------------------------------------------------------------------------

/// Normalize an RGB888 image to f32 with ImageNet-style mean/std normalization.
///
/// Output layout: CHW (channels-first) as required by most ONNX models.
///
/// Normalization: `output[c] = (pixel[c] / 255.0 - mean[c]) / std[c]`
///
/// Default ImageNet values:
/// - mean = [0.485, 0.456, 0.406]
/// - std  = [0.229, 0.224, 0.225]
///
/// # Parameters
///
/// - `rgb`: Input RGB888 data (HWC layout).
/// - `width`, `height`: Image dimensions.
/// - `mean`: Per-channel mean values [R, G, B].
/// - `std_dev`: Per-channel standard deviation values [R, G, B].
///
/// # Returns
///
/// f32 vector in CHW layout of length `3 * width * height`.
pub fn normalize_f32(
    rgb: &[u8],
    width: usize,
    height: usize,
    mean: [f32; 3],
    std_dev: [f32; 3],
) -> Vec<f32> {
    let pixels = width * height;
    if rgb.len() < pixels * 3 || pixels == 0 {
        return Vec::new();
    }

    // Precompute inverse std for multiplication instead of division.
    let inv_std = [
        if std_dev[0] != 0.0 {
            1.0 / std_dev[0]
        } else {
            1.0
        },
        if std_dev[1] != 0.0 {
            1.0 / std_dev[1]
        } else {
            1.0
        },
        if std_dev[2] != 0.0 {
            1.0 / std_dev[2]
        } else {
            1.0
        },
    ];

    let mut output = vec![0.0f32; 3 * pixels];

    for i in 0..pixels {
        for c in 0..3 {
            let val = rgb[i * 3 + c] as f32 / 255.0;
            output[c * pixels + i] = (val - mean[c]) * inv_std[c];
        }
    }

    output
}

/// ImageNet default mean values.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// ImageNet default standard deviation values.
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Convenience function: normalize RGB888 with ImageNet defaults.
pub fn normalize_imagenet(rgb: &[u8], width: usize, height: usize) -> Vec<f32> {
    normalize_f32(rgb, width, height, IMAGENET_MEAN, IMAGENET_STD)
}

// ---------------------------------------------------------------------------
// Center crop
// ---------------------------------------------------------------------------

/// Center-crop an RGB888 image to the specified dimensions.
///
/// If the crop size is larger than the source, returns the original.
pub fn center_crop_rgb(
    src: &[u8],
    src_w: usize,
    src_h: usize,
    crop_w: usize,
    crop_h: usize,
) -> Vec<u8> {
    if crop_w >= src_w && crop_h >= src_h {
        return src.to_vec();
    }
    let crop_w = crop_w.min(src_w);
    let crop_h = crop_h.min(src_h);

    let x_off = (src_w - crop_w) / 2;
    let y_off = (src_h - crop_h) / 2;

    let mut dst = vec![0u8; crop_w * crop_h * 3];
    for row in 0..crop_h {
        let src_start = ((y_off + row) * src_w + x_off) * 3;
        let dst_start = row * crop_w * 3;
        dst[dst_start..dst_start + crop_w * 3]
            .copy_from_slice(&src[src_start..src_start + crop_w * 3]);
    }
    dst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clamp_u8() {
        assert_eq!(clamp_u8(-10), 0);
        assert_eq!(clamp_u8(0), 0);
        assert_eq!(clamp_u8(128), 128);
        assert_eq!(clamp_u8(255), 255);
        assert_eq!(clamp_u8(300), 255);
    }

    #[test]
    fn test_yuv422_to_rgb888_empty() {
        let result = yuv422_to_rgb888(&[], 0, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_yuv422_to_rgb888_odd_width() {
        // Odd width is invalid for YUV422.
        let result = yuv422_to_rgb888(&[0; 6], 3, 1);
        assert!(result.is_empty());
    }

    #[test]
    fn test_yuv422_to_rgb888_black() {
        // Y=16, U=128, V=128 is black in BT.601.
        let yuv = [16, 128, 16, 128];
        let rgb = yuv422_to_rgb888(&yuv, 2, 1);
        assert_eq!(rgb.len(), 6);
        // Should be near black (R,G,B close to 0).
        assert!(rgb[0] <= 2); // R
        assert!(rgb[1] <= 2); // G
        assert!(rgb[2] <= 2); // B
    }

    #[test]
    fn test_yuv422_to_rgb888_white() {
        // Y=235, U=128, V=128 is white in BT.601.
        let yuv = [235, 128, 235, 128];
        let rgb = yuv422_to_rgb888(&yuv, 2, 1);
        assert_eq!(rgb.len(), 6);
        // Should be near white (R,G,B close to 255).
        assert!(rgb[0] >= 250);
        assert!(rgb[1] >= 250);
        assert!(rgb[2] >= 250);
    }

    #[test]
    fn test_yuv422_to_rgb888_output_length() {
        let yuv = vec![128u8; 640 * 480 * 2];
        let rgb = yuv422_to_rgb888(&yuv, 640, 480);
        assert_eq!(rgb.len(), 640 * 480 * 3);
    }

    #[test]
    fn test_resize_bilinear_same_size() {
        let src = vec![100u8; 4 * 4 * 3]; // 4x4 solid gray
        let dst = resize_bilinear(&src, 4, 4, 4, 4);
        assert_eq!(dst.len(), 4 * 4 * 3);
        // All pixels should remain ~100.
        for &v in dst.iter() {
            assert!((v as i32 - 100).unsigned_abs() <= 1);
        }
    }

    #[test]
    fn test_resize_bilinear_downscale() {
        // 4x4 solid color -> 2x2.
        let src = vec![200u8; 4 * 4 * 3];
        let dst = resize_bilinear(&src, 4, 4, 2, 2);
        assert_eq!(dst.len(), 2 * 2 * 3);
        for &v in dst.iter() {
            assert!((v as i32 - 200).unsigned_abs() <= 1);
        }
    }

    #[test]
    fn test_resize_bilinear_upscale() {
        // 2x2 -> 4x4 solid color.
        let src = vec![150u8; 2 * 2 * 3];
        let dst = resize_bilinear(&src, 2, 2, 4, 4);
        assert_eq!(dst.len(), 4 * 4 * 3);
        for &v in dst.iter() {
            assert!((v as i32 - 150).unsigned_abs() <= 1);
        }
    }

    #[test]
    fn test_resize_bilinear_empty() {
        let dst = resize_bilinear(&[], 0, 0, 10, 10);
        assert!(dst.is_empty());
    }

    #[test]
    fn test_normalize_f32_output_size() {
        let rgb = vec![128u8; 4 * 4 * 3];
        let out = normalize_f32(&rgb, 4, 4, IMAGENET_MEAN, IMAGENET_STD);
        assert_eq!(out.len(), 3 * 4 * 4); // CHW layout.
    }

    #[test]
    fn test_normalize_f32_chw_layout() {
        // Single pixel: R=128, G=64, B=255.
        let rgb = [128u8, 64, 255];
        let out = normalize_f32(&rgb, 1, 1, [0.0; 3], [1.0; 3]);
        // With mean=0, std=1: output = pixel/255.
        assert_eq!(out.len(), 3);
        let eps = 0.01;
        assert!((out[0] - 128.0 / 255.0).abs() < eps); // R channel
        assert!((out[1] - 64.0 / 255.0).abs() < eps); // G channel
        assert!((out[2] - 255.0 / 255.0).abs() < eps); // B channel
    }

    #[test]
    fn test_normalize_imagenet_range() {
        // With ImageNet normalization, black pixels should be negative.
        let rgb = vec![0u8; 2 * 2 * 3]; // All black.
        let out = normalize_imagenet(&rgb, 2, 2);
        // R: (0/255 - 0.485) / 0.229 = -2.117...
        assert!(out[0] < -2.0);
    }

    #[test]
    fn test_normalize_f32_empty() {
        let out = normalize_f32(&[], 0, 0, IMAGENET_MEAN, IMAGENET_STD);
        assert!(out.is_empty());
    }

    #[test]
    fn test_center_crop_rgb() {
        // 4x4 image with pixel values indicating position.
        let mut src = vec![0u8; 4 * 4 * 3];
        for y in 0..4 {
            for x in 0..4 {
                let idx = (y * 4 + x) * 3;
                src[idx] = (y * 4 + x) as u8; // R = position
            }
        }
        let cropped = center_crop_rgb(&src, 4, 4, 2, 2);
        assert_eq!(cropped.len(), 2 * 2 * 3);
        // Center crop from (1,1) to (2,2).
        assert_eq!(cropped[0], (1 * 4 + 1) as u8); // Top-left of crop.
    }

    #[test]
    fn test_center_crop_larger_than_source() {
        let src = vec![42u8; 2 * 2 * 3];
        let cropped = center_crop_rgb(&src, 2, 2, 4, 4);
        // Should return original since crop is larger.
        assert_eq!(cropped, src);
    }
}
