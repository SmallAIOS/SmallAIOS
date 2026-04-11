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
    let (oh, ow) = if let Some(s) = sizes {
        if s.len() != 4 {
            return Err(OpError::InvalidAttribute(String::from(
                "Resize sizes must have rank 4",
            )));
        }
        (s[2] as usize, s[3] as usize)
    } else if let Some(s) = scales {
        if s.len() != 4 {
            return Err(OpError::InvalidAttribute(String::from(
                "Resize scales must have rank 4",
            )));
        }
        (
            round_f32((h as f32) * s[2]) as usize,
            round_f32((w as f32) * s[3]) as usize,
        )
    } else {
        return Err(OpError::InvalidAttribute(String::from(
            "Resize requires scales or sizes",
        )));
    };
    if oh == 0 || ow == 0 {
        return Err(OpError::ShapeMismatch(String::from(
            "Resize output dims must be > 0",
        )));
    }
    let out_shape = TensorShape::new(alloc::vec![n as i64, c as i64, oh as i64, ow as i64]);
    let mut data = allocate_tensor_data(n * c * oh * ow, DataType::Float);
    let h_scale = h as f32 / oh as f32;
    let w_scale = w as f32 / ow as f32;
    for bn in 0..n {
        for ch in 0..c {
            let plane = bn * c * h * w + ch * h * w;
            let out_plane = bn * c * oh * ow + ch * oh * ow;
            for oy in 0..oh {
                for ox in 0..ow {
                    // half_pixel: src = (out + 0.5) * scale - 0.5
                    let sy = (oy as f32 + 0.5) * h_scale - 0.5;
                    let sx = (ox as f32 + 0.5) * w_scale - 0.5;
                    let v = match mode {
                        "nearest" => {
                            let iy = round_f32(sy).clamp(0.0, (h - 1) as f32) as usize;
                            let ix = round_f32(sx).clamp(0.0, (w - 1) as f32) as usize;
                            read_f32(&input.raw_data, plane + iy * w + ix)
                        }
                        "linear" | "bilinear" => bilinear_sample(
                            &input.raw_data,
                            plane,
                            h,
                            w,
                            sy,
                            sx,
                            /*zero_pad=*/ false,
                        ),
                        other => {
                            return Err(OpError::InvalidAttribute(alloc::format!(
                                "Resize mode '{}' not supported",
                                other
                            )));
                        }
                    };
                    write_f32(&mut data, out_plane + oy * ow + ox, v);
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
                    let (ih, iw) = (yy / r, xx / r);
                    let (by_, bx_) = (yy % r, xx % r);
                    let ci = match mode {
                        "DCR" => (by_ * r + bx_) * c_out + co,
                        "CRD" => co * r * r + by_ * r + bx_,
                        other => {
                            return Err(OpError::InvalidAttribute(alloc::format!(
                                "DepthToSpace mode '{}' not supported",
                                other
                            )));
                        }
                    };
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
            for by_ in 0..r {
                for bx_ in 0..r {
                    let co = (by_ * r + bx_) * c_in + ci;
                    for yy in 0..oh {
                        for xx in 0..ow {
                            let src_y = yy * r + by_;
                            let src_x = xx * r + bx_;
                            let src = bn * c_in * h * w + ci * h * w + src_y * w + src_x;
                            let dst = bn * c_out * oh * ow + co * oh * ow + yy * ow + xx;
                            write_f32(&mut data, dst, read_f32(&input.raw_data, src));
                        }
                    }
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
    let (_n, c, h, w) = require_4d(x, "RoiAlign")?;
    require_float(rois, "RoiAlign")?;
    if rois.shape.dims.len() != 2 || rois.shape.dims[1] != 4 {
        return Err(OpError::ShapeMismatch(String::from(
            "RoiAlign rois must be [num_rois, 4]",
        )));
    }
    let num_rois = rois.shape.dims[0] as usize;
    let pooled_h = output_height.max(1) as usize;
    let pooled_w = output_width.max(1) as usize;

    // Decode batch indices (Int64 or Int32).
    let bi: Vec<i64> = match batch_indices.data_type {
        DataType::Int64 => (0..num_rois)
            .map(|i| crate::byte_io::read_i64(&batch_indices.raw_data, i))
            .collect(),
        DataType::Int32 => (0..num_rois)
            .map(|i| crate::byte_io::read_i32(&batch_indices.raw_data, i) as i64)
            .collect(),
        _ => {
            return Err(OpError::ShapeMismatch(String::from(
                "RoiAlign batch_indices must be Int32/Int64",
            )));
        }
    };

    let out_shape = TensorShape::new(alloc::vec![
        num_rois as i64,
        c as i64,
        pooled_h as i64,
        pooled_w as i64,
    ]);
    let mut data = allocate_tensor_data(num_rois * c * pooled_h * pooled_w, DataType::Float);

    for (r_idx, &batch_id) in bi.iter().enumerate() {
        let base = r_idx * 4;
        let x1 = read_f32(&rois.raw_data, base) * spatial_scale;
        let y1 = read_f32(&rois.raw_data, base + 1) * spatial_scale;
        let x2 = read_f32(&rois.raw_data, base + 2) * spatial_scale;
        let y2 = read_f32(&rois.raw_data, base + 3) * spatial_scale;
        let roi_w = (x2 - x1).max(1.0);
        let roi_h = (y2 - y1).max(1.0);
        let bin_h = roi_h / pooled_h as f32;
        let bin_w = roi_w / pooled_w as f32;
        let sr_h = if sampling_ratio > 0 {
            sampling_ratio as usize
        } else {
            ceil_f32(roi_h / pooled_h as f32).max(1.0) as usize
        };
        let sr_w = if sampling_ratio > 0 {
            sampling_ratio as usize
        } else {
            ceil_f32(roi_w / pooled_w as f32).max(1.0) as usize
        };
        let bn = batch_id.max(0) as usize;
        for ch in 0..c {
            let plane = bn * c * h * w + ch * h * w;
            for py in 0..pooled_h {
                for px in 0..pooled_w {
                    let mut acc = 0.0f32;
                    let mut count = 0u32;
                    for iy in 0..sr_h {
                        for ix in 0..sr_w {
                            let sy = y1
                                + (py as f32) * bin_h
                                + ((iy as f32) + 0.5) * bin_h / (sr_h as f32);
                            let sx = x1
                                + (px as f32) * bin_w
                                + ((ix as f32) + 0.5) * bin_w / (sr_w as f32);
                            acc += bilinear_sample(&x.raw_data, plane, h, w, sy, sx, true);
                            count += 1;
                        }
                    }
                    let v = if count > 0 { acc / count as f32 } else { 0.0 };
                    let dst = r_idx * c * pooled_h * pooled_w
                        + ch * pooled_h * pooled_w
                        + py * pooled_w
                        + px;
                    write_f32(&mut data, dst, v);
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
                // Map from [-1,1] to pixel coordinates.
                let (sx, sy) = if align_corners {
                    (
                        (gx + 1.0) * 0.5 * ((w - 1) as f32),
                        (gy + 1.0) * 0.5 * ((h - 1) as f32),
                    )
                } else {
                    (
                        ((gx + 1.0) * (w as f32) - 1.0) * 0.5,
                        ((gy + 1.0) * (h as f32) - 1.0) * 0.5,
                    )
                };
                for ch in 0..c {
                    let plane = bn * c * h * w + ch * h * w;
                    let v = match mode {
                        "bilinear" | "linear" => {
                            bilinear_sample(&input.raw_data, plane, h, w, sy, sx, true)
                        }
                        "nearest" => {
                            let iy = round_f32(sy);
                            let ix = round_f32(sx);
                            if iy < 0.0 || iy >= h as f32 || ix < 0.0 || ix >= w as f32 {
                                0.0
                            } else {
                                read_f32(&input.raw_data, plane + (iy as usize) * w + (ix as usize))
                            }
                        }
                        other => {
                            return Err(OpError::InvalidAttribute(alloc::format!(
                                "GridSample mode '{}' not supported",
                                other
                            )));
                        }
                    };
                    let dst = bn * c * oh * ow + ch * oh * ow + yy * ow + xx;
                    write_f32(&mut data, dst, v);
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
    let mut out_dims = input.shape.dims.clone();
    let axes_vec: Vec<usize> = if let Some(a) = axes {
        if a.len() != shape.len() {
            return Err(OpError::InvalidAttribute(String::from(
                "CenterCropPad: axes length must match shape length",
            )));
        }
        a.iter()
            .map(|&ax| {
                if ax < 0 {
                    (ax + in_rank as i64) as usize
                } else {
                    ax as usize
                }
            })
            .collect()
    } else {
        if shape.len() != in_rank {
            return Err(OpError::InvalidAttribute(String::from(
                "CenterCropPad: shape length must equal input rank when axes omitted",
            )));
        }
        (0..in_rank).collect()
    };
    for (i, &ax) in axes_vec.iter().enumerate() {
        if ax >= in_rank {
            return Err(OpError::InvalidAttribute(String::from(
                "CenterCropPad: axis out of range",
            )));
        }
        out_dims[ax] = shape[i];
    }
    // Allocate zero-filled output.
    let total: usize = out_dims
        .iter()
        .map(|&d| d as usize)
        .product::<usize>()
        .max(1);
    let mut data = allocate_tensor_data(total, DataType::Float);
    // For each axis, compute source/dest start offsets (symmetric).
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
    // Iterate the copy region.
    let total_copy: usize = copy_dims.iter().product();
    if total_copy == 0 {
        return Ok(Tensor {
            data_type: DataType::Float,
            shape: TensorShape::new(out_dims),
            name: String::new(),
            raw_data: data,
        });
    }
    let mut coord = alloc::vec![0usize; in_rank];
    let in_strides: Vec<usize> = {
        let mut s = alloc::vec![1usize; in_rank];
        for i in (0..in_rank.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * input.shape.dims[i + 1] as usize;
        }
        s
    };
    let out_strides: Vec<usize> = {
        let mut s = alloc::vec![1usize; in_rank];
        for i in (0..in_rank.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * out_dims[i + 1] as usize;
        }
        s
    };
    for _ in 0..total_copy {
        let mut src = 0usize;
        let mut dst = 0usize;
        for d in 0..in_rank {
            src += (src_starts[d] + coord[d]) * in_strides[d];
            dst += (dst_starts[d] + coord[d]) * out_strides[d];
        }
        write_f32(&mut data, dst, read_f32(&input.raw_data, src));
        // Increment coord
        for d in (0..in_rank).rev() {
            coord[d] += 1;
            if coord[d] < copy_dims[d] {
                break;
            }
            coord[d] = 0;
        }
    }
    Ok(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(out_dims),
        name: String::new(),
        raw_data: data,
    })
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
    fn test_roi_align_rejects_max_mode() {
        let x = make_f32(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let rois = make_f32(&[1, 4], &[0.0, 0.0, 1.0, 1.0]);
        let bi = make_i64(&[1], &[0]);
        assert!(op_roi_align(&x, &rois, &bi, 2, 2, 1, 1.0, "max").is_err());
    }
}
