// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Vision-specific operators used by image-heavy models (ViT, Swin,
//! ConvNeXt, Mask R-CNN, …).
//!
//! These ops target the common-case attribute combinations used by
//! recent CV exports. Many ONNX vision ops accept a large attribute
//! matrix; we implement the pragmatic subset documented on each
//! function and return `InvalidAttribute` for unsupported modes.

use alloc::string::String;
use alloc::vec::Vec;

use crate::byte_io::{allocate_tensor_data, read_f32, write_f32};
use crate::operators::OpError;
use crate::tensor::{DataType, Tensor, TensorShape};

fn require_float(t: &Tensor, op: &str) -> Result<(), OpError> {
    if t.data_type != DataType::Float {
        return Err(OpError::ShapeMismatch(alloc::format!(
            "{} requires Float input",
            op
        )));
    }
    Ok(())
}

// no_std-safe math helpers (the libcore f32 type has no .floor/.round/.ceil).
fn floor_f32(x: f32) -> f32 {
    let i = x as i64 as f32;
    if x < i {
        i - 1.0
    } else {
        i
    }
}
fn ceil_f32(x: f32) -> f32 {
    let i = x as i64 as f32;
    if x > i {
        i + 1.0
    } else {
        i
    }
}
fn round_f32(x: f32) -> f32 {
    floor_f32(x + 0.5)
}

fn require_4d(t: &Tensor, op: &str) -> Result<(usize, usize, usize, usize), OpError> {
    require_float(t, op)?;
    if t.shape.dims.len() != 4 {
        return Err(OpError::ShapeMismatch(alloc::format!(
            "{} requires 4D NCHW input",
            op
        )));
    }
    Ok((
        t.shape.dims[0] as usize,
        t.shape.dims[1] as usize,
        t.shape.dims[2] as usize,
        t.shape.dims[3] as usize,
    ))
}

// ---------------------------------------------------------------------------
// Resize
// ---------------------------------------------------------------------------

/// 2D NCHW image resize.
///
/// Supported attributes (matches the common case used by most vision
/// exports):
/// - `mode = "nearest"` or `"linear"` (bilinear)
/// - `coordinate_transformation_mode = "half_pixel"`
/// - no antialias, no cubic interpolation, no `exclude_outside`
///
/// Output sizes are taken from `sizes` (input #3) when present, else
/// derived from `scales` (input #2). Only the spatial dimensions
/// (H, W) are rescaled; N and C are preserved.
pub fn op_resize(
    input: &Tensor,
    scales: Option<&[f32]>,
    sizes: Option<&[i64]>,
    mode: &str,
) -> Result<Tensor, OpError> {
    let (n, c, h, w) = require_4d(input, "Resize")?;
    let (oh, ow) = resolve_resize_output_dims(h, w, scales, sizes)?;
    if oh == 0 || ow == 0 {
        return Err(OpError::ShapeMismatch(String::from(
            "Resize output dims must be > 0",
        )));
    }
    let total_out = checked_total_elements(n, c, oh, ow)?;
    let out_shape = TensorShape::new(alloc::vec![n as i64, c as i64, oh as i64, ow as i64]);
    let mut data = allocate_tensor_data(total_out, DataType::Float);
    let h_scale = h as f32 / oh as f32;
    let w_scale = w as f32 / ow as f32;
    for bn in 0..n {
        for ch in 0..c {
            let plane = bn * c * h * w + ch * h * w;
            let out_plane = bn * c * oh * ow + ch * oh * ow;
            resize_plane(
                &input.raw_data,
                &mut data,
                plane,
                out_plane,
                (h, w),
                (oh, ow),
                (h_scale, w_scale),
                mode,
            )?;
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: out_shape,
        name: String::new(),
        raw_data: data,
    })
}

/// Resolve the (oh, ow) output spatial dimensions for `Resize` from the
/// `sizes` (preferred) or `scales` input. Returns an error if the
/// attribute is malformed or would overflow the output buffer.
fn resolve_resize_output_dims(
    h: usize,
    w: usize,
    scales: Option<&[f32]>,
    sizes: Option<&[i64]>,
) -> Result<(usize, usize), OpError> {
    if let Some(s) = sizes {
        return resize_dims_from_sizes(s);
    }
    if let Some(s) = scales {
        return resize_dims_from_scales(h, w, s);
    }
    Err(OpError::InvalidAttribute(String::from(
        "Resize requires scales or sizes",
    )))
}

fn resize_dims_from_sizes(s: &[i64]) -> Result<(usize, usize), OpError> {
    if s.len() != 4 {
        return Err(OpError::InvalidAttribute(String::from(
            "Resize sizes must have rank 4",
        )));
    }
    if s.iter().any(|&d| d < 0) {
        return Err(OpError::InvalidAttribute(String::from(
            "Resize sizes must be non-negative",
        )));
    }
    let oh_i = s[2];
    let ow_i = s[3];
    if oh_i > (i32::MAX as i64) || ow_i > (i32::MAX as i64) {
        return Err(OpError::InvalidAttribute(String::from(
            "Resize output dims exceed addressable range",
        )));
    }
    Ok((oh_i as usize, ow_i as usize))
}

fn resize_dims_from_scales(h: usize, w: usize, s: &[f32]) -> Result<(usize, usize), OpError> {
    if s.len() != 4 {
        return Err(OpError::InvalidAttribute(String::from(
            "Resize scales must have rank 4",
        )));
    }
    if !s.iter().all(|v| v.is_finite() && *v > 0.0) {
        return Err(OpError::InvalidAttribute(String::from(
            "Resize scales must be finite and > 0",
        )));
    }
    let oh_f = round_f32((h as f32) * s[2]);
    let ow_f = round_f32((w as f32) * s[3]);
    if !oh_f.is_finite() || !ow_f.is_finite() || oh_f < 0.0 || ow_f < 0.0 {
        return Err(OpError::InvalidAttribute(String::from(
            "Resize computed output dims are invalid",
        )));
    }
    if oh_f > (i32::MAX as f32) || ow_f > (i32::MAX as f32) {
        return Err(OpError::InvalidAttribute(String::from(
            "Resize output dims exceed addressable range",
        )));
    }
    Ok((oh_f as usize, ow_f as usize))
}

fn checked_total_elements(n: usize, c: usize, oh: usize, ow: usize) -> Result<usize, OpError> {
    n.checked_mul(c)
        .and_then(|v| v.checked_mul(oh))
        .and_then(|v| v.checked_mul(ow))
        .ok_or_else(|| {
            OpError::InvalidAttribute(String::from("Resize output tensor size overflows usize"))
        })
}

#[allow(clippy::too_many_arguments)]
fn resize_plane(
    raw: &[u8],
    data: &mut [u8],
    plane: usize,
    out_plane: usize,
    in_hw: (usize, usize),
    out_hw: (usize, usize),
    scales: (f32, f32),
    mode: &str,
) -> Result<(), OpError> {
    let (h, w) = in_hw;
    let (oh, ow) = out_hw;
    let (h_scale, w_scale) = scales;
    for oy in 0..oh {
        for ox in 0..ow {
            // half_pixel: src = (out + 0.5) * scale - 0.5
            let sy = (oy as f32 + 0.5) * h_scale - 0.5;
            let sx = (ox as f32 + 0.5) * w_scale - 0.5;
            let v = sample_resize_pixel(raw, plane, h, w, sy, sx, mode)?;
            write_f32(data, out_plane + oy * ow + ox, v);
        }
    }
    Ok(())
}

fn sample_resize_pixel(
    raw: &[u8],
    plane: usize,
    h: usize,
    w: usize,
    sy: f32,
    sx: f32,
    mode: &str,
) -> Result<f32, OpError> {
    match mode {
        "nearest" => {
            let iy = round_f32(sy).clamp(0.0, (h - 1) as f32) as usize;
            let ix = round_f32(sx).clamp(0.0, (w - 1) as f32) as usize;
            Ok(read_f32(raw, plane + iy * w + ix))
        }
        "linear" | "bilinear" => Ok(bilinear_sample(raw, plane, h, w, sy, sx, false)),
        other => Err(OpError::InvalidAttribute(alloc::format!(
            "Resize mode '{}' not supported",
            other
        ))),
    }
}

/// Bilinear sample from a single HxW plane with clamp-to-edge (or
/// zero-pad) semantics when `zero_pad` is true.
fn bilinear_sample(
    raw: &[u8],
    plane_off: usize,
    h: usize,
    w: usize,
    sy: f32,
    sx: f32,
    zero_pad: bool,
) -> f32 {
    let x0 = floor_f32(sx) as i64;
    let y0 = floor_f32(sy) as i64;
    let x1 = x0 + 1;
    let y1 = y0 + 1;
    let dx = sx - x0 as f32;
    let dy = sy - y0 as f32;
    let sample = |yy: i64, xx: i64| -> f32 {
        if yy < 0 || yy >= h as i64 || xx < 0 || xx >= w as i64 {
            if zero_pad {
                0.0
            } else {
                let yc = yy.clamp(0, h as i64 - 1) as usize;
                let xc = xx.clamp(0, w as i64 - 1) as usize;
                read_f32(raw, plane_off + yc * w + xc)
            }
        } else {
            read_f32(raw, plane_off + (yy as usize) * w + (xx as usize))
        }
    };
    let v00 = sample(y0, x0);
    let v01 = sample(y0, x1);
    let v10 = sample(y1, x0);
    let v11 = sample(y1, x1);
    let v0 = v00 * (1.0 - dx) + v01 * dx;
    let v1 = v10 * (1.0 - dx) + v11 * dx;
    v0 * (1.0 - dy) + v1 * dy
}

// ---------------------------------------------------------------------------
// DepthToSpace / SpaceToDepth
// ---------------------------------------------------------------------------

/// DepthToSpace: [N, C*r*r, H, W] -> [N, C, H*r, W*r]. Both "DCR" and
/// "CRD" orderings are supported.
pub fn op_depth_to_space(input: &Tensor, blocksize: i64, mode: &str) -> Result<Tensor, OpError> {
    let (n, c_in, h, w) = require_4d(input, "DepthToSpace")?;
    if blocksize <= 0 {
        return Err(OpError::InvalidAttribute(String::from(
            "DepthToSpace blocksize must be > 0",
        )));
    }
    let r = blocksize as usize;
    if (c_in % (r * r)) != 0 {
        return Err(OpError::ShapeMismatch(String::from(
            "DepthToSpace: C must be divisible by blocksize^2",
        )));
    }
    let c_out = c_in / (r * r);
    let oh = h * r;
    let ow = w * r;
    let out_shape = TensorShape::new(alloc::vec![n as i64, c_out as i64, oh as i64, ow as i64]);
    let mut data = allocate_tensor_data(n * c_out * oh * ow, DataType::Float);
    for bn in 0..n {
        for co in 0..c_out {
            for yy in 0..oh {
                for xx in 0..ow {
                    let ci = depth_to_space_channel_index(mode, r, c_out, yy, xx, co)?;
                    let (ih, iw) = (yy / r, xx / r);
                    let src = bn * c_in * h * w + ci * h * w + ih * w + iw;
                    let dst = bn * c_out * oh * ow + co * oh * ow + yy * ow + xx;
                    write_f32(&mut data, dst, read_f32(&input.raw_data, src));
                }
            }
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: out_shape,
        name: String::new(),
        raw_data: data,
    })
}

/// Compute the source channel index for `DepthToSpace` given the mode
/// (`DCR` or `CRD`), block size `r`, output channel count, and the
/// current output (yy, xx, co) coordinate. Returns `InvalidAttribute`
/// for unsupported modes.
fn depth_to_space_channel_index(
    mode: &str,
    r: usize,
    c_out: usize,
    yy: usize,
    xx: usize,
    co: usize,
) -> Result<usize, OpError> {
    let by_ = yy % r;
    let bx_ = xx % r;
    match mode {
        "DCR" => Ok((by_ * r + bx_) * c_out + co),
        "CRD" => Ok(co * r * r + by_ * r + bx_),
        other => Err(OpError::InvalidAttribute(alloc::format!(
            "DepthToSpace mode '{}' not supported",
            other
        ))),
    }
}

/// SpaceToDepth: [N, C, H*r, W*r] -> [N, C*r*r, H, W] (DCR-style order).
pub fn op_space_to_depth(input: &Tensor, blocksize: i64) -> Result<Tensor, OpError> {
    let (n, c_in, h, w) = require_4d(input, "SpaceToDepth")?;
    if blocksize <= 0 {
        return Err(OpError::InvalidAttribute(String::from(
            "SpaceToDepth blocksize must be > 0",
        )));
    }
    let r = blocksize as usize;
    if (h % r) != 0 || (w % r) != 0 {
        return Err(OpError::ShapeMismatch(String::from(
            "SpaceToDepth: H and W must be divisible by blocksize",
        )));
    }
    let oh = h / r;
    let ow = w / r;
    let c_out = c_in * r * r;
    let out_shape = TensorShape::new(alloc::vec![n as i64, c_out as i64, oh as i64, ow as i64]);
    let mut data = allocate_tensor_data(n * c_out * oh * ow, DataType::Float);
    for bn in 0..n {
        for ci in 0..c_in {
            space_to_depth_channel(
                &input.raw_data,
                &mut data,
                bn,
                ci,
                r,
                (c_in, c_out),
                (h, w),
                (oh, ow),
            );
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: out_shape,
        name: String::new(),
        raw_data: data,
    })
}

/// Copy one input channel into its `r*r` output sub-channels under the
/// DCR-style ordering used by `SpaceToDepth`.
#[allow(clippy::too_many_arguments)]
fn space_to_depth_channel(
    raw: &[u8],
    data: &mut [u8],
    bn: usize,
    ci: usize,
    r: usize,
    c_in_out: (usize, usize),
    in_hw: (usize, usize),
    out_hw: (usize, usize),
) {
    let (c_in, c_out) = c_in_out;
    let (h, w) = in_hw;
    let (oh, ow) = out_hw;
    for by_ in 0..r {
        for bx_ in 0..r {
            let co = (by_ * r + bx_) * c_in + ci;
            copy_space_to_depth_block(
                raw,
                data,
                bn,
                ci,
                co,
                (by_, bx_),
                r,
                (c_in, c_out),
                (h, w),
                (oh, ow),
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn copy_space_to_depth_block(
    raw: &[u8],
    data: &mut [u8],
    bn: usize,
    ci: usize,
    co: usize,
    block: (usize, usize),
    r: usize,
    c_in_out: (usize, usize),
    in_hw: (usize, usize),
    out_hw: (usize, usize),
) {
    let (by_, bx_) = block;
    let (c_in, c_out) = c_in_out;
    let (h, w) = in_hw;
    let (oh, ow) = out_hw;
    for yy in 0..oh {
        for xx in 0..ow {
            let src_y = yy * r + by_;
            let src_x = xx * r + bx_;
            let src = bn * c_in * h * w + ci * h * w + src_y * w + src_x;
            let dst = bn * c_out * oh * ow + co * oh * ow + yy * ow + xx;
            write_f32(data, dst, read_f32(raw, src));
        }
    }
}

// ---------------------------------------------------------------------------
// RoiAlign
// ---------------------------------------------------------------------------

/// Bilinear RoiAlign, average-pooled ("avg" mode only).
///
/// `x` is NCHW, `rois` is `[num_rois, 4]` with `[x1, y1, x2, y2]` in
/// continuous image coordinates. `batch_indices` is `[num_rois]`
/// selecting which batch element each ROI comes from.
///
/// The returned tensor has shape `[num_rois, C, output_height, output_width]`.
/// Only the bilinear average pooling path is implemented; `max` pooling
/// is not supported.
#[allow(clippy::too_many_arguments)]
pub fn op_roi_align(
    x: &Tensor,
    rois: &Tensor,
    batch_indices: &Tensor,
    output_height: i64,
    output_width: i64,
    sampling_ratio: i64,
    spatial_scale: f32,
    mode: &str,
) -> Result<Tensor, OpError> {
    if mode != "avg" {
        return Err(OpError::InvalidAttribute(alloc::format!(
            "RoiAlign mode '{}' not supported (only 'avg')",
            mode
        )));
    }
    let (n_batch, c, h, w) = require_4d(x, "RoiAlign")?;
    validate_roi_align_rois(rois)?;
    let num_rois = rois.shape.dims[0] as usize;
    let pooled_h = output_height.max(1) as usize;
    let pooled_w = output_width.max(1) as usize;

    let bi = decode_roi_batch_indices(batch_indices, num_rois)?;
    validate_roi_batch_ids(&bi, n_batch)?;

    let out_shape = TensorShape::new(alloc::vec![
        num_rois as i64,
        c as i64,
        pooled_h as i64,
        pooled_w as i64,
    ]);
    let mut data = allocate_tensor_data(num_rois * c * pooled_h * pooled_w, DataType::Float);

    for (r_idx, &batch_id) in bi.iter().enumerate() {
        pool_one_roi(
            x,
            rois,
            &mut data,
            r_idx,
            batch_id as usize,
            (c, h, w),
            (pooled_h, pooled_w),
            sampling_ratio,
            spatial_scale,
        );
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: out_shape,
        name: String::new(),
        raw_data: data,
    })
}

fn validate_roi_align_rois(rois: &Tensor) -> Result<(), OpError> {
    require_float(rois, "RoiAlign")?;
    if rois.shape.dims.len() != 2 || rois.shape.dims[1] != 4 {
        return Err(OpError::ShapeMismatch(String::from(
            "RoiAlign rois must be [num_rois, 4]",
        )));
    }
    Ok(())
}

fn decode_roi_batch_indices(batch_indices: &Tensor, num_rois: usize) -> Result<Vec<i64>, OpError> {
    if batch_indices.shape.total_elements() != num_rois {
        return Err(OpError::ShapeMismatch(String::from(
            "RoiAlign batch_indices length must equal num_rois",
        )));
    }
    match batch_indices.data_type {
        DataType::Int64 => Ok((0..num_rois)
            .map(|i| crate::byte_io::read_i64(&batch_indices.raw_data, i))
            .collect()),
        DataType::Int32 => Ok((0..num_rois)
            .map(|i| crate::byte_io::read_i32(&batch_indices.raw_data, i) as i64)
            .collect()),
        _ => Err(OpError::ShapeMismatch(String::from(
            "RoiAlign batch_indices must be Int32/Int64",
        ))),
    }
}

fn validate_roi_batch_ids(bi: &[i64], n_batch: usize) -> Result<(), OpError> {
    for &b in bi {
        if b < 0 || (b as usize) >= n_batch {
            return Err(OpError::ShapeMismatch(String::from(
                "RoiAlign batch_indices value out of range [0, N)",
            )));
        }
    }
    Ok(())
}

struct RoiBinGeometry {
    x1: f32,
    y1: f32,
    bin_h: f32,
    bin_w: f32,
    sr_h: usize,
    sr_w: usize,
}

/// Decode the ROI box for `r_idx` and the sampling lattice (bin sizes,
/// sub-sample counts) given the global pooling configuration.
fn roi_bin_geometry(
    rois: &Tensor,
    r_idx: usize,
    pooled_h: usize,
    pooled_w: usize,
    sampling_ratio: i64,
    spatial_scale: f32,
) -> RoiBinGeometry {
    let base = r_idx * 4;
    let x1 = read_f32(&rois.raw_data, base) * spatial_scale;
    let y1 = read_f32(&rois.raw_data, base + 1) * spatial_scale;
    let x2 = read_f32(&rois.raw_data, base + 2) * spatial_scale;
    let y2 = read_f32(&rois.raw_data, base + 3) * spatial_scale;
    let roi_w = (x2 - x1).max(1.0);
    let roi_h = (y2 - y1).max(1.0);
    let bin_h = roi_h / pooled_h as f32;
    let bin_w = roi_w / pooled_w as f32;
    let sr_h = sampling_count(sampling_ratio, roi_h, pooled_h);
    let sr_w = sampling_count(sampling_ratio, roi_w, pooled_w);
    RoiBinGeometry {
        x1,
        y1,
        bin_h,
        bin_w,
        sr_h,
        sr_w,
    }
}

fn sampling_count(sampling_ratio: i64, roi_side: f32, pooled_side: usize) -> usize {
    if sampling_ratio > 0 {
        sampling_ratio as usize
    } else {
        ceil_f32(roi_side / pooled_side as f32).max(1.0) as usize
    }
}

#[allow(clippy::too_many_arguments)]
fn pool_one_roi(
    x: &Tensor,
    rois: &Tensor,
    data: &mut [u8],
    r_idx: usize,
    bn: usize,
    chw: (usize, usize, usize),
    pooled_hw: (usize, usize),
    sampling_ratio: i64,
    spatial_scale: f32,
) {
    let (c, h, w) = chw;
    let (pooled_h, pooled_w) = pooled_hw;
    let geom = roi_bin_geometry(
        rois,
        r_idx,
        pooled_h,
        pooled_w,
        sampling_ratio,
        spatial_scale,
    );
    for ch in 0..c {
        let plane = bn * c * h * w + ch * h * w;
        for py in 0..pooled_h {
            for px in 0..pooled_w {
                let v = average_roi_bin(&x.raw_data, plane, h, w, &geom, py, px);
                let dst =
                    r_idx * c * pooled_h * pooled_w + ch * pooled_h * pooled_w + py * pooled_w + px;
                write_f32(data, dst, v);
            }
        }
    }
}

fn average_roi_bin(
    raw: &[u8],
    plane: usize,
    h: usize,
    w: usize,
    geom: &RoiBinGeometry,
    py: usize,
    px: usize,
) -> f32 {
    let mut acc = 0.0f32;
    let mut count = 0u32;
    for iy in 0..geom.sr_h {
        for ix in 0..geom.sr_w {
            let sy = geom.y1
                + (py as f32) * geom.bin_h
                + ((iy as f32) + 0.5) * geom.bin_h / (geom.sr_h as f32);
            let sx = geom.x1
                + (px as f32) * geom.bin_w
                + ((ix as f32) + 0.5) * geom.bin_w / (geom.sr_w as f32);
            acc += bilinear_sample(raw, plane, h, w, sy, sx, true);
            count += 1;
        }
    }
    if count > 0 {
        acc / count as f32
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// GridSample
// ---------------------------------------------------------------------------

/// 2D bilinear GridSample.
///
/// Supported attributes:
/// - `mode = "bilinear"` or `"nearest"`
/// - `padding_mode = "zeros"` (other modes not implemented)
/// - `align_corners` = 0 or 1
pub fn op_grid_sample(
    input: &Tensor,
    grid: &Tensor,
    mode: &str,
    padding_mode: &str,
    align_corners: bool,
) -> Result<Tensor, OpError> {
    let (n, c, h, w) = require_4d(input, "GridSample")?;
    validate_grid_sample_grid(grid, n, padding_mode)?;
    let oh = grid.shape.dims[1] as usize;
    let ow = grid.shape.dims[2] as usize;
    let out_shape = TensorShape::new(alloc::vec![n as i64, c as i64, oh as i64, ow as i64]);
    let mut data = allocate_tensor_data(n * c * oh * ow, DataType::Float);
    for bn in 0..n {
        for yy in 0..oh {
            for xx in 0..ow {
                let g_off = bn * oh * ow * 2 + yy * ow * 2 + xx * 2;
                let gx = read_f32(&grid.raw_data, g_off);
                let gy = read_f32(&grid.raw_data, g_off + 1);
                let (sx, sy) = grid_to_pixel(gx, gy, w, h, align_corners);
                write_grid_sample_pixels(
                    &input.raw_data,
                    &mut data,
                    bn,
                    yy,
                    xx,
                    (n, c, h, w),
                    (oh, ow),
                    (sy, sx),
                    mode,
                )?;
            }
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: out_shape,
        name: String::new(),
        raw_data: data,
    })
}

fn validate_grid_sample_grid(grid: &Tensor, n: usize, padding_mode: &str) -> Result<(), OpError> {
    require_float(grid, "GridSample")?;
    if padding_mode != "zeros" {
        return Err(OpError::InvalidAttribute(alloc::format!(
            "GridSample padding_mode '{}' not supported",
            padding_mode
        )));
    }
    // grid is [N, oh, ow, 2]
    if grid.shape.dims.len() != 4 || grid.shape.dims[3] != 2 {
        return Err(OpError::ShapeMismatch(String::from(
            "GridSample grid must be [N, H_out, W_out, 2]",
        )));
    }
    if grid.shape.dims[0] as usize != n {
        return Err(OpError::ShapeMismatch(String::from(
            "GridSample grid batch must match input batch",
        )));
    }
    Ok(())
}

/// Map a `[-1, 1]` grid coordinate `(gx, gy)` to fractional pixel
/// coordinates given the source image size and `align_corners` flag.
fn grid_to_pixel(gx: f32, gy: f32, w: usize, h: usize, align_corners: bool) -> (f32, f32) {
    if align_corners {
        (
            (gx + 1.0) * 0.5 * ((w - 1) as f32),
            (gy + 1.0) * 0.5 * ((h - 1) as f32),
        )
    } else {
        (
            ((gx + 1.0) * (w as f32) - 1.0) * 0.5,
            ((gy + 1.0) * (h as f32) - 1.0) * 0.5,
        )
    }
}

#[allow(clippy::too_many_arguments)]
fn write_grid_sample_pixels(
    raw: &[u8],
    data: &mut [u8],
    bn: usize,
    yy: usize,
    xx: usize,
    nchw: (usize, usize, usize, usize),
    out_hw: (usize, usize),
    sy_sx: (f32, f32),
    mode: &str,
) -> Result<(), OpError> {
    let (_, c, h, w) = nchw;
    let (oh, ow) = out_hw;
    let (sy, sx) = sy_sx;
    for ch in 0..c {
        let plane = bn * c * h * w + ch * h * w;
        let v = sample_grid_pixel(raw, plane, h, w, sy, sx, mode)?;
        let dst = bn * c * oh * ow + ch * oh * ow + yy * ow + xx;
        write_f32(data, dst, v);
    }
    Ok(())
}

fn sample_grid_pixel(
    raw: &[u8],
    plane: usize,
    h: usize,
    w: usize,
    sy: f32,
    sx: f32,
    mode: &str,
) -> Result<f32, OpError> {
    match mode {
        "bilinear" | "linear" => Ok(bilinear_sample(raw, plane, h, w, sy, sx, true)),
        "nearest" => Ok(grid_nearest_sample(raw, plane, h, w, sy, sx)),
        other => Err(OpError::InvalidAttribute(alloc::format!(
            "GridSample mode '{}' not supported",
            other
        ))),
    }
}

fn grid_nearest_sample(raw: &[u8], plane: usize, h: usize, w: usize, sy: f32, sx: f32) -> f32 {
    let iy = round_f32(sy);
    let ix = round_f32(sx);
    if iy < 0.0 || iy >= h as f32 || ix < 0.0 || ix >= w as f32 {
        0.0
    } else {
        read_f32(raw, plane + (iy as usize) * w + (ix as usize))
    }
}

// ---------------------------------------------------------------------------
// GlobalMaxPool
// ---------------------------------------------------------------------------

/// Global max pooling over NCHW → NC11.
pub fn op_global_max_pool(input: &Tensor) -> Result<Tensor, OpError> {
    let (n, c, h, w) = require_4d(input, "GlobalMaxPool")?;
    let spatial = h * w;
    let out_shape = TensorShape::new(alloc::vec![n as i64, c as i64, 1, 1]);
    let mut data = allocate_tensor_data(n * c, DataType::Float);
    for bn in 0..n {
        for ch in 0..c {
            let mut m = f32::NEG_INFINITY;
            let off = bn * c * spatial + ch * spatial;
            for i in 0..spatial {
                let v = read_f32(&input.raw_data, off + i);
                if v > m {
                    m = v;
                }
            }
            write_f32(&mut data, bn * c + ch, m);
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: out_shape,
        name: String::new(),
        raw_data: data,
    })
}

// ---------------------------------------------------------------------------
// CenterCropPad
// ---------------------------------------------------------------------------

/// Symmetrically crops or pads `input` along `axes` so that each
/// selected axis ends up at the matching size in `shape`.
///
/// When `axes` is `None`, `shape` must have rank equal to the input and
/// applies to every axis. Padding uses zeros.
pub fn op_center_crop_pad(
    input: &Tensor,
    shape: &[i64],
    axes: Option<&[i64]>,
) -> Result<Tensor, OpError> {
    require_float(input, "CenterCropPad")?;
    let in_rank = input.shape.dims.len();
    let axes_vec = resolve_crop_axes(axes, shape, in_rank)?;
    let mut out_dims = input.shape.dims.clone();
    apply_crop_target_dims(&mut out_dims, &axes_vec, shape, in_rank)?;

    // Allocate zero-filled output.
    let total: usize = out_dims
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let mut data = allocate_tensor_data(total, DataType::Float);

    let (src_starts, dst_starts, copy_dims) = compute_crop_windows(input, &out_dims, in_rank);
    let total_copy: usize = copy_dims.iter().product();
    if total_copy == 0 {
        return Ok(Tensor {
            data_type: DataType::Float,
            shape: TensorShape::new(out_dims),
            name: String::new(),
            raw_data: data,
        });
    }
    let in_strides = row_major_strides(&input.shape.dims, in_rank);
    let out_strides = row_major_strides(&out_dims, in_rank);
    copy_crop_region(
        &input.raw_data,
        &mut data,
        &src_starts,
        &dst_starts,
        &copy_dims,
        &in_strides,
        &out_strides,
        total_copy,
        in_rank,
    );
    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(out_dims),
        name: String::new(),
        raw_data: data,
    })
}

fn resolve_crop_axes(
    axes: Option<&[i64]>,
    shape: &[i64],
    in_rank: usize,
) -> Result<Vec<usize>, OpError> {
    if let Some(a) = axes {
        if a.len() != shape.len() {
            return Err(OpError::InvalidAttribute(String::from(
                "CenterCropPad: axes length must match shape length",
            )));
        }
        Ok(a.iter()
            .map(|&ax| {
                if ax < 0 {
                    (ax + in_rank as i64) as usize
                } else {
                    ax as usize
                }
            })
            .collect())
    } else {
        if shape.len() != in_rank {
            return Err(OpError::InvalidAttribute(String::from(
                "CenterCropPad: shape length must equal input rank when axes omitted",
            )));
        }
        Ok((0..in_rank).collect())
    }
}

fn apply_crop_target_dims(
    out_dims: &mut [i64],
    axes_vec: &[usize],
    shape: &[i64],
    in_rank: usize,
) -> Result<(), OpError> {
    for (i, &ax) in axes_vec.iter().enumerate() {
        if ax >= in_rank {
            return Err(OpError::InvalidAttribute(String::from(
                "CenterCropPad: axis out of range",
            )));
        }
        if shape[i] < 0 {
            return Err(OpError::InvalidAttribute(String::from(
                "CenterCropPad: target shape dims must be non-negative",
            )));
        }
        out_dims[ax] = shape[i];
    }
    Ok(())
}

fn compute_crop_windows(
    input: &Tensor,
    out_dims: &[i64],
    in_rank: usize,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut src_starts = alloc::vec![0usize; in_rank];
    let mut dst_starts = alloc::vec![0usize; in_rank];
    let mut copy_dims = alloc::vec![0usize; in_rank];
    for d in 0..in_rank {
        let id = input.shape.dims[d] as usize;
        let od = out_dims[d] as usize;
        let c = id.min(od);
        copy_dims[d] = c;
        src_starts[d] = (id - c) / 2;
        dst_starts[d] = (od - c) / 2;
    }
    (src_starts, dst_starts, copy_dims)
}

fn row_major_strides(dims: &[i64], rank: usize) -> Vec<usize> {
    let mut s = alloc::vec![1usize; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        s[i] = s[i + 1] * dims[i + 1] as usize;
    }
    s
}

#[allow(clippy::too_many_arguments)]
fn copy_crop_region(
    raw: &[u8],
    data: &mut [u8],
    src_starts: &[usize],
    dst_starts: &[usize],
    copy_dims: &[usize],
    in_strides: &[usize],
    out_strides: &[usize],
    total_copy: usize,
    in_rank: usize,
) {
    let mut coord = alloc::vec![0usize; in_rank];
    for _ in 0..total_copy {
        let mut src = 0usize;
        let mut dst = 0usize;
        for d in 0..in_rank {
            src += (src_starts[d] + coord[d]) * in_strides[d];
            dst += (dst_starts[d] + coord[d]) * out_strides[d];
        }
        write_f32(data, dst, read_f32(raw, src));
        increment_coord(&mut coord, copy_dims, in_rank);
    }
}

fn increment_coord(coord: &mut [usize], copy_dims: &[usize], in_rank: usize) {
    for d in (0..in_rank).rev() {
        coord[d] += 1;
        if coord[d] < copy_dims[d] {
            break;
        }
        coord[d] = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_f32(dims: &[i64], vals: &[f32]) -> Tensor {
        let mut data = allocate_tensor_data(vals.len(), DataType::Float);
        for (i, &v) in vals.iter().enumerate() {
            write_f32(&mut data, i, v);
        }
        Tensor {
            data_type: DataType::Float,
            shape: TensorShape::new(dims.to_vec()),
            name: String::new(),
            raw_data: data,
        }
    }

    fn make_i64(dims: &[i64], vals: &[i64]) -> Tensor {
        let mut data = allocate_tensor_data(vals.len(), DataType::Int64);
        for (i, &v) in vals.iter().enumerate() {
            crate::byte_io::write_i64(&mut data, i, v);
        }
        Tensor {
            data_type: DataType::Int64,
            shape: TensorShape::new(dims.to_vec()),
            name: String::new(),
            raw_data: data,
        }
    }

    fn read_all(t: &Tensor) -> Vec<f32> {
        (0..t.shape.total_elements())
            .map(|i| read_f32(&t.raw_data, i))
            .collect()
    }

    #[test]
    fn test_resize_nearest_upsample() {
        let t = make_f32(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let out = op_resize(&t, Some(&[1.0, 1.0, 2.0, 2.0]), None, "nearest").unwrap();
        assert_eq!(out.shape.dims, alloc::vec![1, 1, 4, 4]);
        let v = read_all(&out);
        assert_eq!(
            v,
            alloc::vec![
                1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
            ]
        );
    }

    #[test]
    fn test_resize_bilinear_identity() {
        // scale 1x1 should return the same tensor.
        let t = make_f32(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let out = op_resize(&t, None, Some(&[1, 1, 2, 2]), "linear").unwrap();
        let v = read_all(&out);
        for (a, b) in v.iter().zip([1.0f32, 2.0, 3.0, 4.0].iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_resize_rejects_unknown_mode() {
        let t = make_f32(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        assert!(op_resize(&t, None, Some(&[1, 1, 4, 4]), "cubic").is_err());
    }

    #[test]
    fn test_depth_to_space_dcr() {
        // [1, 4, 1, 1] (4 channels) -> [1, 1, 2, 2]
        let t = make_f32(&[1, 4, 1, 1], &[1.0, 2.0, 3.0, 4.0]);
        let out = op_depth_to_space(&t, 2, "DCR").unwrap();
        assert_eq!(out.shape.dims, alloc::vec![1, 1, 2, 2]);
        let v = read_all(&out);
        // DCR: byx uses channels in order
        assert_eq!(v.len(), 4);
    }

    #[test]
    fn test_depth_to_space_crd_small() {
        let t = make_f32(&[1, 4, 1, 1], &[10.0, 20.0, 30.0, 40.0]);
        let out = op_depth_to_space(&t, 2, "CRD").unwrap();
        assert_eq!(out.shape.dims, alloc::vec![1, 1, 2, 2]);
        let v = read_all(&out);
        assert_eq!(v, alloc::vec![10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    fn test_space_to_depth_roundtrip() {
        // Build a 1x1x4x4 ramp, s2d with block 2, verify shape.
        let vals: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let t = make_f32(&[1, 1, 4, 4], &vals);
        let out = op_space_to_depth(&t, 2).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![1, 4, 2, 2]);
    }

    #[test]
    fn test_space_to_depth_rejects_nondivisible() {
        let t = make_f32(&[1, 1, 3, 3], &alloc::vec![0.0; 9]);
        assert!(op_space_to_depth(&t, 2).is_err());
    }

    #[test]
    fn test_global_max_pool() {
        let t = make_f32(&[1, 2, 2, 2], &[1.0, 5.0, 3.0, 2.0, 10.0, 8.0, 6.0, 9.0]);
        let out = op_global_max_pool(&t).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![1, 2, 1, 1]);
        let v = read_all(&out);
        assert_eq!(v, alloc::vec![5.0, 10.0]);
    }

    #[test]
    fn test_global_max_pool_rejects_3d() {
        let t = make_f32(&[1, 2, 4], &alloc::vec![0.0; 8]);
        assert!(op_global_max_pool(&t).is_err());
    }

    #[test]
    fn test_center_crop_pad_crop() {
        // [1,1,4,4] cropped to [1,1,2,2] — center 2x2.
        let vals: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let t = make_f32(&[1, 1, 4, 4], &vals);
        let out = op_center_crop_pad(&t, &[1, 1, 2, 2], None).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![1, 1, 2, 2]);
        let v = read_all(&out);
        // Center window is rows 1-2, cols 1-2 → indices 5,6,9,10
        assert_eq!(v, alloc::vec![5.0, 6.0, 9.0, 10.0]);
    }

    #[test]
    fn test_center_crop_pad_pad() {
        // [1,1,2,2] padded to [1,1,4,4] with zeros.
        let t = make_f32(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let out = op_center_crop_pad(&t, &[1, 1, 4, 4], None).unwrap();
        let v = read_all(&out);
        // Center 2x2 should hold the originals, others zero.
        assert_eq!(v[5], 1.0);
        assert_eq!(v[6], 2.0);
        assert_eq!(v[9], 3.0);
        assert_eq!(v[10], 4.0);
        assert_eq!(v[0], 0.0);
    }

    #[test]
    fn test_center_crop_pad_with_axes() {
        // Only crop axis 3 from 4 to 2.
        let vals: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let t = make_f32(&[1, 1, 2, 4], &vals);
        let out = op_center_crop_pad(&t, &[2], Some(&[3])).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![1, 1, 2, 2]);
        let v = read_all(&out);
        // center cols of each row: row0 cols [1,2], row1 cols [5,6]
        assert_eq!(v, alloc::vec![1.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    fn test_grid_sample_identity() {
        // 1x1x2x2 input, grid that maps to exactly (0,0) corner — should
        // sample around the corners.
        let t = make_f32(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        // grid [-1,-1] -> (0,0), [1,1] -> (1,1)
        let grid = make_f32(&[1, 2, 2, 2], &[-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0]);
        let out = op_grid_sample(&t, &grid, "bilinear", "zeros", true).unwrap();
        assert_eq!(out.shape.dims, alloc::vec![1, 1, 2, 2]);
        let v = read_all(&out);
        assert!((v[0] - 1.0).abs() < 1e-5);
        assert!((v[1] - 2.0).abs() < 1e-5);
        assert!((v[2] - 3.0).abs() < 1e-5);
        assert!((v[3] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_grid_sample_rejects_border_pad() {
        let t = make_f32(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let grid = make_f32(&[1, 1, 1, 2], &[0.0, 0.0]);
        assert!(op_grid_sample(&t, &grid, "bilinear", "border", true).is_err());
    }

    #[test]
    fn test_roi_align_single_roi() {
        // 1x1x4x4 ramp, one ROI covering the whole image, pool 2x2.
        let vals: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let x = make_f32(&[1, 1, 4, 4], &vals);
        let rois = make_f32(&[1, 4], &[0.0, 0.0, 3.0, 3.0]);
        let bi = make_i64(&[1], &[0]);
        let out = op_roi_align(&x, &rois, &bi, 2, 2, 2, 1.0, "avg").unwrap();
        assert_eq!(out.shape.dims, alloc::vec![1, 1, 2, 2]);
        // The outputs are bilinear means of 4 samples each; all positive.
        let v = read_all(&out);
        for x in v {
            assert!(x > 0.0 && x < 16.0);
        }
    }

    #[test]
    fn test_resize_rejects_negative_sizes() {
        let t = make_f32(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let err = op_resize(&t, None, Some(&[1, 1, -4, 4]), "nearest");
        assert!(matches!(err, Err(OpError::InvalidAttribute(_))));
    }

    #[test]
    fn test_resize_rejects_nonpositive_scales() {
        let t = make_f32(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        assert!(matches!(
            op_resize(&t, Some(&[1.0, 1.0, 0.0, 2.0]), None, "nearest"),
            Err(OpError::InvalidAttribute(_))
        ));
        assert!(matches!(
            op_resize(&t, Some(&[1.0, 1.0, -1.0, 2.0]), None, "nearest"),
            Err(OpError::InvalidAttribute(_))
        ));
        assert!(matches!(
            op_resize(&t, Some(&[1.0, 1.0, f32::NAN, 2.0]), None, "nearest"),
            Err(OpError::InvalidAttribute(_))
        ));
    }

    #[test]
    fn test_resize_rejects_huge_output_dims() {
        let t = make_f32(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let big = i32::MAX as i64 + 1;
        assert!(matches!(
            op_resize(&t, None, Some(&[1, 1, big, 2]), "nearest"),
            Err(OpError::InvalidAttribute(_))
        ));
    }

    #[test]
    fn test_roi_align_rejects_short_batch_indices() {
        let x = make_f32(&[2, 1, 2, 2], &alloc::vec![0.0; 8]);
        let rois = make_f32(&[2, 4], &[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
        let bi = make_i64(&[1], &[0]);
        assert!(matches!(
            op_roi_align(&x, &rois, &bi, 2, 2, 1, 1.0, "avg"),
            Err(OpError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn test_roi_align_rejects_out_of_range_batch_id() {
        let x = make_f32(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let rois = make_f32(&[1, 4], &[0.0, 0.0, 1.0, 1.0]);
        let bi = make_i64(&[1], &[5]);
        assert!(matches!(
            op_roi_align(&x, &rois, &bi, 2, 2, 1, 1.0, "avg"),
            Err(OpError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn test_roi_align_rejects_negative_batch_id() {
        let x = make_f32(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let rois = make_f32(&[1, 4], &[0.0, 0.0, 1.0, 1.0]);
        let bi = make_i64(&[1], &[-1]);
        assert!(matches!(
            op_roi_align(&x, &rois, &bi, 2, 2, 1, 1.0, "avg"),
            Err(OpError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn test_grid_sample_rejects_batch_mismatch() {
        let t = make_f32(&[2, 1, 2, 2], &alloc::vec![0.0; 8]);
        // grid batch is 1 but input batch is 2.
        let grid = make_f32(&[1, 1, 1, 2], &[0.0, 0.0]);
        assert!(matches!(
            op_grid_sample(&t, &grid, "bilinear", "zeros", true),
            Err(OpError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn test_center_crop_pad_rejects_negative_shape() {
        let t = make_f32(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        assert!(matches!(
            op_center_crop_pad(&t, &[1, 1, -2, 2], None),
            Err(OpError::InvalidAttribute(_))
        ));
    }

    #[test]
    fn test_roi_align_rejects_max_mode() {
        let x = make_f32(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let rois = make_f32(&[1, 4], &[0.0, 0.0, 1.0, 1.0]);
        let bi = make_i64(&[1], &[0]);
        assert!(op_roi_align(&x, &rois, &bi, 2, 2, 1, 1.0, "max").is_err());
    }
}
