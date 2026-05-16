// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Recurrent operators: RNN, LSTM, GRU.
//!
//! All ops accept ONNX-style weight layouts where the leading dimension is
//! the direction (1 for forward/reverse, 2 for bidirectional). The shapes
//! follow the ONNX opset 14+ spec.
//!
//! Implementations are sequence-major and use a simple per-timestep loop.
//! Output Y has shape `[seq_length, num_directions, batch, hidden_size]`.

#![allow(clippy::too_many_arguments)]

use alloc::string::String;
use alloc::vec::Vec;

use crate::byte_io::{allocate_tensor_data, read_f32, write_f32};
use crate::operators::{expf_approx, OpError};
use crate::ops::common::require_float;
use crate::tensor::{DataType, Tensor, TensorShape};

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + expf_approx(-x))
}

fn tanh_approx(x: f32) -> f32 {
    let ep = expf_approx(x);
    let em = expf_approx(-x);
    (ep - em) / (ep + em)
}

fn parse_direction(direction: &str) -> Result<(usize, bool), OpError> {
    // returns (num_directions, is_bidirectional)
    match direction {
        "forward" | "reverse" => Ok((1, false)),
        "bidirectional" => Ok((2, true)),
        _ => Err(OpError::InvalidAttribute(alloc::format!(
            "unknown direction: {}",
            direction
        ))),
    }
}

/// Reads a slice of the form `data[base..base+n]` as f32.
fn slice_f32(data: &[u8], base: usize, n: usize) -> Vec<f32> {
    (0..n).map(|i| read_f32(data, base + i)).collect()
}

/// Whether this direction iterates the sequence in reverse.
#[inline]
fn dir_is_reverse(direction: &str, d: usize) -> bool {
    direction == "reverse" || (direction == "bidirectional" && d == 1)
}

/// Shared shape validation for the `[seq, batch, input]` X tensor.
/// Returns `(seq, batch, input_size)`.
fn validate_x_shape(x: &Tensor, op: &str) -> Result<(usize, usize, usize), OpError> {
    if x.shape.dims.len() != 3 {
        return Err(OpError::ShapeMismatch(alloc::format!(
            "{}: X must be 3D [seq, batch, input]",
            op
        )));
    }
    Ok((
        x.shape.dims[0] as usize,
        x.shape.dims[1] as usize,
        x.shape.dims[2] as usize,
    ))
}

/// Validate the `[num_dir, gates*hidden, hidden]` R tensor.
fn validate_r_shape(
    r: &Tensor,
    num_dir: usize,
    gates: usize,
    hidden: usize,
    op: &str,
    shape_msg: &str,
) -> Result<(), OpError> {
    if r.shape.dims.len() != 3
        || r.shape.dims[0] as usize != num_dir
        || r.shape.dims[1] as usize != gates * hidden
        || r.shape.dims[2] as usize != hidden
    {
        return Err(OpError::ShapeMismatch(alloc::format!(
            "{}: R shape must be {}",
            op,
            shape_msg
        )));
    }
    Ok(())
}

/// Validate the `[num_dir, total]` B tensor, if provided.
fn validate_b_shape(
    b: Option<&Tensor>,
    num_dir: usize,
    total: usize,
    op: &str,
    shape_msg: &str,
) -> Result<(), OpError> {
    if let Some(bt) = b {
        if bt.shape.dims.len() != 2
            || bt.shape.dims[0] as usize != num_dir
            || bt.shape.dims[1] as usize != total
        {
            return Err(OpError::ShapeMismatch(alloc::format!(
                "{}: B shape must be {}",
                op,
                shape_msg
            )));
        }
    }
    Ok(())
}

/// Validate the `[num_dir, batch, hidden]` initial-state tensor (h0 or c0).
fn validate_initial_state(
    state: Option<&Tensor>,
    num_dir: usize,
    batch: usize,
    hidden: usize,
    op: &str,
    name: &str,
) -> Result<(), OpError> {
    if let Some(st) = state {
        if st.shape.dims.len() != 3
            || st.shape.dims[0] as usize != num_dir
            || st.shape.dims[1] as usize != batch
            || st.shape.dims[2] as usize != hidden
        {
            return Err(OpError::ShapeMismatch(alloc::format!(
                "{}: {} shape must be [num_dir, batch, hidden]",
                op,
                name
            )));
        }
    }
    Ok(())
}

/// Read the per-direction initial state slice, defaulting to zeros.
fn read_initial_state(
    state: Option<&Tensor>,
    d: usize,
    batch: usize,
    hidden: usize,
    op: &str,
) -> Result<Vec<f32>, OpError> {
    if let Some(st) = state {
        require_float(st, op)?;
        Ok(slice_f32(&st.raw_data, d * batch * hidden, batch * hidden))
    } else {
        Ok(alloc::vec![0.0f32; batch * hidden])
    }
}

/// Merge `[W bias | R bias]` halves into a single bias of length `half`.
fn merge_wr_bias(b: Option<&Tensor>, d: usize, half: usize, op: &str) -> Result<Vec<f32>, OpError> {
    if let Some(bt) = b {
        require_float(bt, op)?;
        let off = d * 2 * half;
        let wb = slice_f32(&bt.raw_data, off, half);
        let rb = slice_f32(&bt.raw_data, off + half, half);
        Ok(wb.iter().zip(rb.iter()).map(|(a, b)| a + b).collect())
    } else {
        Ok(alloc::vec![0.0f32; half])
    }
}

/// Write the final per-direction hidden state into the `Y_h`-style buffer.
fn write_final_state(yh_data: &mut [u8], h: &[f32], d: usize, batch: usize, hidden: usize) {
    let off = d * batch * hidden;
    for (i, &v) in h.iter().enumerate() {
        write_f32(yh_data, off + i, v);
    }
}

/// Write the per-direction output slice into `Y[t, d, :, :]`.
fn write_y_timestep(
    y_data: &mut [u8],
    h: &[f32],
    t: usize,
    d: usize,
    num_dir: usize,
    batch: usize,
    hidden: usize,
) {
    let y_off = t * num_dir * batch * hidden + d * batch * hidden;
    for (i, &v) in h.iter().enumerate() {
        write_f32(y_data, y_off + i, v);
    }
}

/// Build the output Y tensor with shape `[seq, num_dir, batch, hidden]`.
fn make_y_tensor(
    y_data: Vec<u8>,
    seq: usize,
    num_dir: usize,
    batch: usize,
    hidden: usize,
) -> Tensor {
    Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(alloc::vec![
            seq as i64,
            num_dir as i64,
            batch as i64,
            hidden as i64
        ]),
        name: String::new(),
        raw_data: y_data,
    }
}

/// Build the hidden-state tensor with shape `[num_dir, batch, hidden]`.
fn make_state_tensor(data: Vec<u8>, num_dir: usize, batch: usize, hidden: usize) -> Tensor {
    Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(alloc::vec![num_dir as i64, batch as i64, hidden as i64]),
        name: String::new(),
        raw_data: data,
    }
}

// ---------------------------------------------------------------------------
// RNN
// ---------------------------------------------------------------------------

/// Validate W (and derive `hidden`) and the related R/B/h0 shapes for RNN.
fn rnn_validate_shapes(
    w: &Tensor,
    r: &Tensor,
    b: Option<&Tensor>,
    h0: Option<&Tensor>,
    num_dir: usize,
    batch: usize,
    input_size: usize,
) -> Result<usize, OpError> {
    if w.shape.dims.len() != 3 || w.shape.dims[0] as usize != num_dir {
        return Err(OpError::ShapeMismatch(String::from(
            "RNN: W shape must be [num_dir, hidden, input]",
        )));
    }
    let hidden = w.shape.dims[1] as usize;
    if w.shape.dims[2] as usize != input_size {
        return Err(OpError::ShapeMismatch(String::from(
            "RNN: W input dim mismatch",
        )));
    }
    validate_r_shape(r, num_dir, 1, hidden, "RNN", "[num_dir, hidden, hidden]")?;
    validate_b_shape(b, num_dir, 2 * hidden, "RNN", "[num_dir, 2*hidden]")?;
    validate_initial_state(h0, num_dir, batch, hidden, "RNN", "h0")?;
    Ok(hidden)
}

/// Compute one RNN timestep: `h_t = tanh(x_t W^T + h_{t-1} R^T + b)`.
fn rnn_step(
    x_raw: &[u8],
    x_off: usize,
    h_prev: &[f32],
    w_dir: &[f32],
    r_dir: &[f32],
    b_dir: &[f32],
    batch: usize,
    input_size: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut new_h = alloc::vec![0.0f32; batch * hidden];
    for bi in 0..batch {
        for hi in 0..hidden {
            let mut acc = b_dir[hi];
            for ii in 0..input_size {
                acc += read_f32(x_raw, x_off + bi * input_size + ii) * w_dir[hi * input_size + ii];
            }
            for hh in 0..hidden {
                acc += h_prev[bi * hidden + hh] * r_dir[hi * hidden + hh];
            }
            new_h[bi * hidden + hi] = tanh_approx(acc);
        }
    }
    new_h
}

/// Basic RNN with tanh activation.
///
/// X: [seq_length, batch, input_size]
/// W: [num_directions, hidden_size, input_size]
/// R: [num_directions, hidden_size, hidden_size]
/// B: [num_directions, 2*hidden_size] (concat of W bias and R bias) — optional
/// h0: [num_directions, batch, hidden_size] — optional, default zeros
///
/// Returns:
/// - Y: [seq_length, num_directions, batch, hidden_size]
/// - Y_h: [num_directions, batch, hidden_size]
pub fn op_rnn(
    x: &Tensor,
    w: &Tensor,
    r: &Tensor,
    b: Option<&Tensor>,
    h0: Option<&Tensor>,
    direction: &str,
) -> Result<Vec<Tensor>, OpError> {
    require_float(x, "RNN")?;
    require_float(w, "RNN")?;
    require_float(r, "RNN")?;
    let (num_dir, _bi) = parse_direction(direction)?;
    let (seq, batch, input_size) = validate_x_shape(x, "RNN")?;
    let hidden = rnn_validate_shapes(w, r, b, h0, num_dir, batch, input_size)?;

    let y_total = seq * num_dir * batch * hidden;
    let mut y_data = allocate_tensor_data(y_total, DataType::Float);
    let mut yh_data = allocate_tensor_data(num_dir * batch * hidden, DataType::Float);

    for d in 0..num_dir {
        let reverse = dir_is_reverse(direction, d);
        let w_off = d * hidden * input_size;
        let r_off = d * hidden * hidden;
        let w_dir = slice_f32(&w.raw_data, w_off, hidden * input_size);
        let r_dir = slice_f32(&r.raw_data, r_off, hidden * hidden);
        let b_dir = merge_wr_bias(b, d, hidden, "RNN")?;
        let mut h = read_initial_state(h0, d, batch, hidden, "RNN")?;

        for step in 0..seq {
            let t = if reverse { seq - 1 - step } else { step };
            let x_off = t * batch * input_size;
            h = rnn_step(
                &x.raw_data,
                x_off,
                &h,
                &w_dir,
                &r_dir,
                &b_dir,
                batch,
                input_size,
                hidden,
            );
            write_y_timestep(&mut y_data, &h, t, d, num_dir, batch, hidden);
        }
        write_final_state(&mut yh_data, &h, d, batch, hidden);
    }

    Ok(alloc::vec![
        make_y_tensor(y_data, seq, num_dir, batch, hidden),
        make_state_tensor(yh_data, num_dir, batch, hidden),
    ])
}

// ---------------------------------------------------------------------------
// LSTM
// ---------------------------------------------------------------------------

/// Validate W (and derive `hidden`) and the related R/B/h0/c0 shapes for LSTM.
fn lstm_validate_shapes(
    w: &Tensor,
    r: &Tensor,
    b: Option<&Tensor>,
    h0: Option<&Tensor>,
    c0: Option<&Tensor>,
    num_dir: usize,
    batch: usize,
    input_size: usize,
) -> Result<usize, OpError> {
    if w.shape.dims.len() != 3 || w.shape.dims[0] as usize != num_dir {
        return Err(OpError::ShapeMismatch(String::from(
            "LSTM: W shape must be [num_dir, 4*hidden, input]",
        )));
    }
    if w.shape.dims[2] as usize != input_size {
        return Err(OpError::ShapeMismatch(String::from(
            "LSTM: W input dim mismatch",
        )));
    }
    let four_h = w.shape.dims[1] as usize;
    if !four_h.is_multiple_of(4) {
        return Err(OpError::ShapeMismatch(String::from(
            "LSTM: W second dim must be 4*hidden",
        )));
    }
    let hidden = four_h / 4;
    validate_r_shape(r, num_dir, 4, hidden, "LSTM", "[num_dir, 4*hidden, hidden]")?;
    validate_b_shape(b, num_dir, 8 * hidden, "LSTM", "[num_dir, 8*hidden]")?;
    validate_initial_state(h0, num_dir, batch, hidden, "LSTM", "h0")?;
    validate_initial_state(c0, num_dir, batch, hidden, "LSTM", "c0")?;
    Ok(hidden)
}

/// Compute the LSTM gate pre-activations for one batch element:
/// `g[k] = X·W^T_k + H·R^T_k + b_k` for `k in 0..4*hidden`.
fn lstm_gate_preacts(
    x_raw: &[u8],
    x_row_off: usize,
    h_prev_row: &[f32],
    w_dir: &[f32],
    r_dir: &[f32],
    b_dir: &[f32],
    input_size: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut gates = alloc::vec![0.0f32; 4 * hidden];
    for k in 0..4 * hidden {
        let mut acc = b_dir[k];
        for ii in 0..input_size {
            acc += read_f32(x_raw, x_row_off + ii) * w_dir[k * input_size + ii];
        }
        for hh in 0..hidden {
            acc += h_prev_row[hh] * r_dir[k * hidden + hh];
        }
        gates[k] = acc;
    }
    gates
}

/// Apply the LSTM gate non-linearities and update `(h, c)` for one batch row.
/// Gate layout per ONNX: i, o, f, c (input, output, forget, cell).
fn lstm_apply_gates(
    gates: &[f32],
    c_prev_row: &[f32],
    new_h_row: &mut [f32],
    new_c_row: &mut [f32],
    hidden: usize,
) {
    for hi in 0..hidden {
        let i_g = sigmoid(gates[hi]);
        let o_g = sigmoid(gates[hidden + hi]);
        let f_g = sigmoid(gates[2 * hidden + hi]);
        let c_g = tanh_approx(gates[3 * hidden + hi]);
        let c_new = f_g * c_prev_row[hi] + i_g * c_g;
        new_c_row[hi] = c_new;
        new_h_row[hi] = o_g * tanh_approx(c_new);
    }
}

/// Compute one LSTM timestep across the batch, returning the updated
/// `(h, c)` slices.
#[allow(clippy::too_many_arguments)]
fn lstm_step(
    x_raw: &[u8],
    x_off: usize,
    h_prev: &[f32],
    c_prev: &[f32],
    w_dir: &[f32],
    r_dir: &[f32],
    b_dir: &[f32],
    batch: usize,
    input_size: usize,
    hidden: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut new_h = alloc::vec![0.0f32; batch * hidden];
    let mut new_c = alloc::vec![0.0f32; batch * hidden];
    for bi in 0..batch {
        let off = bi * hidden;
        let gates = lstm_gate_preacts(
            x_raw,
            x_off + bi * input_size,
            &h_prev[off..off + hidden],
            w_dir,
            r_dir,
            b_dir,
            input_size,
            hidden,
        );
        lstm_apply_gates(
            &gates,
            &c_prev[off..off + hidden],
            &mut new_h[off..off + hidden],
            &mut new_c[off..off + hidden],
            hidden,
        );
    }
    (new_h, new_c)
}

/// LSTM with input, forget, cell, output gates (i, o, f, c order per ONNX).
///
/// W: [num_dir, 4*hidden, input]
/// R: [num_dir, 4*hidden, hidden]
/// B: [num_dir, 8*hidden]
///
/// Returns: Y, Y_h, Y_c
#[allow(clippy::too_many_arguments)]
pub fn op_lstm(
    x: &Tensor,
    w: &Tensor,
    r: &Tensor,
    b: Option<&Tensor>,
    h0: Option<&Tensor>,
    c0: Option<&Tensor>,
    direction: &str,
) -> Result<Vec<Tensor>, OpError> {
    require_float(x, "LSTM")?;
    require_float(w, "LSTM")?;
    require_float(r, "LSTM")?;
    let (num_dir, _bi) = parse_direction(direction)?;
    let (seq, batch, input_size) = validate_x_shape(x, "LSTM")?;
    let hidden = lstm_validate_shapes(w, r, b, h0, c0, num_dir, batch, input_size)?;

    let y_total = seq * num_dir * batch * hidden;
    let mut y_data = allocate_tensor_data(y_total, DataType::Float);
    let mut yh_data = allocate_tensor_data(num_dir * batch * hidden, DataType::Float);
    let mut yc_data = allocate_tensor_data(num_dir * batch * hidden, DataType::Float);

    for d in 0..num_dir {
        let reverse = dir_is_reverse(direction, d);
        let w_off = d * 4 * hidden * input_size;
        let r_off = d * 4 * hidden * hidden;
        let w_dir = slice_f32(&w.raw_data, w_off, 4 * hidden * input_size);
        let r_dir = slice_f32(&r.raw_data, r_off, 4 * hidden * hidden);
        let b_dir = merge_wr_bias(b, d, 4 * hidden, "LSTM")?;
        let mut h = read_initial_state(h0, d, batch, hidden, "LSTM")?;
        let mut c = read_initial_state(c0, d, batch, hidden, "LSTM")?;

        for step in 0..seq {
            let t = if reverse { seq - 1 - step } else { step };
            let x_off = t * batch * input_size;
            let (new_h, new_c) = lstm_step(
                &x.raw_data,
                x_off,
                &h,
                &c,
                &w_dir,
                &r_dir,
                &b_dir,
                batch,
                input_size,
                hidden,
            );
            h = new_h;
            c = new_c;
            write_y_timestep(&mut y_data, &h, t, d, num_dir, batch, hidden);
        }
        write_final_state(&mut yh_data, &h, d, batch, hidden);
        write_final_state(&mut yc_data, &c, d, batch, hidden);
    }

    Ok(alloc::vec![
        make_y_tensor(y_data, seq, num_dir, batch, hidden),
        make_state_tensor(yh_data, num_dir, batch, hidden),
        make_state_tensor(yc_data, num_dir, batch, hidden),
    ])
}

// ---------------------------------------------------------------------------
// GRU
// ---------------------------------------------------------------------------

/// Validate W (and derive `hidden`) and the related R/B/h0 shapes for GRU.
fn gru_validate_shapes(
    w: &Tensor,
    r: &Tensor,
    b: Option<&Tensor>,
    h0: Option<&Tensor>,
    num_dir: usize,
    batch: usize,
    input_size: usize,
) -> Result<usize, OpError> {
    if w.shape.dims.len() != 3 || w.shape.dims[0] as usize != num_dir {
        return Err(OpError::ShapeMismatch(String::from(
            "GRU: W shape must be [num_dir, 3*hidden, input]",
        )));
    }
    if w.shape.dims[2] as usize != input_size {
        return Err(OpError::ShapeMismatch(String::from(
            "GRU: W input dim mismatch",
        )));
    }
    let three_h = w.shape.dims[1] as usize;
    if !three_h.is_multiple_of(3) {
        return Err(OpError::ShapeMismatch(String::from(
            "GRU: W second dim must be 3*hidden",
        )));
    }
    let hidden = three_h / 3;
    validate_r_shape(r, num_dir, 3, hidden, "GRU", "[num_dir, 3*hidden, hidden]")?;
    validate_b_shape(b, num_dir, 6 * hidden, "GRU", "[num_dir, 6*hidden]")?;
    validate_initial_state(h0, num_dir, batch, hidden, "GRU", "h0")?;
    Ok(hidden)
}

/// Split the GRU bias into separate `(W bias, R bias)` halves for direction `d`.
/// ONNX keeps these separate because the hidden gate adds `R bias` only after
/// being scaled by the reset gate.
fn gru_split_bias(
    b: Option<&Tensor>,
    d: usize,
    hidden: usize,
) -> Result<(Vec<f32>, Vec<f32>), OpError> {
    if let Some(bt) = b {
        require_float(bt, "GRU")?;
        let off = d * 6 * hidden;
        Ok((
            slice_f32(&bt.raw_data, off, 3 * hidden),
            slice_f32(&bt.raw_data, off + 3 * hidden, 3 * hidden),
        ))
    } else {
        Ok((
            alloc::vec![0.0f32; 3 * hidden],
            alloc::vec![0.0f32; 3 * hidden],
        ))
    }
}

/// Compute the GRU `(z, r)` gates and the split hidden pre-activations
/// `(h_pre_w, h_pre_r)` for one batch row.
#[allow(clippy::too_many_arguments)]
fn gru_zr_and_h_preacts(
    x_raw: &[u8],
    x_row_off: usize,
    h_prev_row: &[f32],
    w_dir: &[f32],
    r_dir: &[f32],
    wb_dir: &[f32],
    rb_dir: &[f32],
    input_size: usize,
    hidden: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut z_g = alloc::vec![0.0f32; hidden];
    let mut r_g = alloc::vec![0.0f32; hidden];
    let mut h_pre_w = alloc::vec![0.0f32; hidden];
    let mut h_pre_r = alloc::vec![0.0f32; hidden];
    for hi in 0..hidden {
        let mut z_acc = wb_dir[hi] + rb_dir[hi];
        let mut r_acc = wb_dir[hidden + hi] + rb_dir[hidden + hi];
        let mut hw_acc = wb_dir[2 * hidden + hi];
        let mut hr_acc = rb_dir[2 * hidden + hi];
        for ii in 0..input_size {
            let xv = read_f32(x_raw, x_row_off + ii);
            z_acc += xv * w_dir[hi * input_size + ii];
            r_acc += xv * w_dir[(hidden + hi) * input_size + ii];
            hw_acc += xv * w_dir[(2 * hidden + hi) * input_size + ii];
        }
        for hh in 0..hidden {
            let hv = h_prev_row[hh];
            z_acc += hv * r_dir[hi * hidden + hh];
            r_acc += hv * r_dir[(hidden + hi) * hidden + hh];
            hr_acc += hv * r_dir[(2 * hidden + hi) * hidden + hh];
        }
        z_g[hi] = sigmoid(z_acc);
        r_g[hi] = sigmoid(r_acc);
        h_pre_w[hi] = hw_acc;
        h_pre_r[hi] = hr_acc;
    }
    (z_g, r_g, h_pre_w, h_pre_r)
}

/// Compute one GRU timestep across the batch.
#[allow(clippy::too_many_arguments)]
fn gru_step(
    x_raw: &[u8],
    x_off: usize,
    h_prev: &[f32],
    w_dir: &[f32],
    r_dir: &[f32],
    wb_dir: &[f32],
    rb_dir: &[f32],
    batch: usize,
    input_size: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut new_h = alloc::vec![0.0f32; batch * hidden];
    for bi in 0..batch {
        let off = bi * hidden;
        let (z_g, r_g, h_pre_w, h_pre_r) = gru_zr_and_h_preacts(
            x_raw,
            x_off + bi * input_size,
            &h_prev[off..off + hidden],
            w_dir,
            r_dir,
            wb_dir,
            rb_dir,
            input_size,
            hidden,
        );
        // Hidden gate: h_tilde = tanh(h_pre_w + r * h_pre_r)
        for hi in 0..hidden {
            let h_tilde = tanh_approx(h_pre_w[hi] + r_g[hi] * h_pre_r[hi]);
            let h_prev_v = h_prev[off + hi];
            new_h[off + hi] = (1.0 - z_g[hi]) * h_tilde + z_g[hi] * h_prev_v;
        }
    }
    new_h
}

/// GRU with reset, update, and hidden gates (z, r, h order per ONNX).
///
/// W: [num_dir, 3*hidden, input]
/// R: [num_dir, 3*hidden, hidden]
/// B: [num_dir, 6*hidden]
pub fn op_gru(
    x: &Tensor,
    w: &Tensor,
    r: &Tensor,
    b: Option<&Tensor>,
    h0: Option<&Tensor>,
    direction: &str,
) -> Result<Vec<Tensor>, OpError> {
    require_float(x, "GRU")?;
    require_float(w, "GRU")?;
    require_float(r, "GRU")?;
    let (num_dir, _bi) = parse_direction(direction)?;
    let (seq, batch, input_size) = validate_x_shape(x, "GRU")?;
    let hidden = gru_validate_shapes(w, r, b, h0, num_dir, batch, input_size)?;

    let y_total = seq * num_dir * batch * hidden;
    let mut y_data = allocate_tensor_data(y_total, DataType::Float);
    let mut yh_data = allocate_tensor_data(num_dir * batch * hidden, DataType::Float);

    for d in 0..num_dir {
        let reverse = dir_is_reverse(direction, d);
        let w_off = d * 3 * hidden * input_size;
        let r_off = d * 3 * hidden * hidden;
        let w_dir = slice_f32(&w.raw_data, w_off, 3 * hidden * input_size);
        let r_dir = slice_f32(&r.raw_data, r_off, 3 * hidden * hidden);
        let (wb_dir, rb_dir) = gru_split_bias(b, d, hidden)?;
        let mut h = read_initial_state(h0, d, batch, hidden, "GRU")?;

        for step in 0..seq {
            let t = if reverse { seq - 1 - step } else { step };
            let x_off = t * batch * input_size;
            h = gru_step(
                &x.raw_data,
                x_off,
                &h,
                &w_dir,
                &r_dir,
                &wb_dir,
                &rb_dir,
                batch,
                input_size,
                hidden,
            );
            write_y_timestep(&mut y_data, &h, t, d, num_dir, batch, hidden);
        }
        write_final_state(&mut yh_data, &h, d, batch, hidden);
    }

    Ok(alloc::vec![
        make_y_tensor(y_data, seq, num_dir, batch, hidden),
        make_state_tensor(yh_data, num_dir, batch, hidden),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::common::test_helpers::make_f32;

    #[test]
    fn test_rnn_forward_shape() {
        // seq=2, batch=1, input=3, hidden=4
        let x = make_f32(&[2, 1, 3], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let w = make_f32(&[1, 4, 3], &alloc::vec![0.1f32; 12]);
        let r = make_f32(&[1, 4, 4], &alloc::vec![0.1f32; 16]);
        let outs = op_rnn(&x, &w, &r, None, None, "forward").unwrap();
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0].shape.dims, alloc::vec![2, 1, 1, 4]);
        assert_eq!(outs[1].shape.dims, alloc::vec![1, 1, 4]);
    }

    #[test]
    fn test_rnn_zero_input_zero_state_yields_zero_h() {
        let x = make_f32(&[3, 2, 4], &alloc::vec![0.0f32; 24]);
        let w = make_f32(&[1, 5, 4], &alloc::vec![0.5f32; 20]);
        let r = make_f32(&[1, 5, 5], &alloc::vec![0.5f32; 25]);
        let outs = op_rnn(&x, &w, &r, None, None, "forward").unwrap();
        let yh = &outs[1];
        for i in 0..yh.shape.total_elements() {
            assert!(read_f32(&yh.raw_data, i).abs() < 1e-6);
        }
    }

    #[test]
    fn test_rnn_reverse_runs() {
        let x = make_f32(&[2, 1, 2], &[1.0, 0.0, 0.0, 1.0]);
        let w = make_f32(&[1, 3, 2], &alloc::vec![0.1f32; 6]);
        let r = make_f32(&[1, 3, 3], &alloc::vec![0.1f32; 9]);
        let outs = op_rnn(&x, &w, &r, None, None, "reverse").unwrap();
        assert_eq!(outs[0].shape.dims, alloc::vec![2, 1, 1, 3]);
    }

    #[test]
    fn test_rnn_bidirectional_shape() {
        let x = make_f32(&[2, 1, 2], &[0.1, 0.2, 0.3, 0.4]);
        let w = make_f32(&[2, 3, 2], &alloc::vec![0.1f32; 12]);
        let r = make_f32(&[2, 3, 3], &alloc::vec![0.1f32; 18]);
        let outs = op_rnn(&x, &w, &r, None, None, "bidirectional").unwrap();
        assert_eq!(outs[0].shape.dims, alloc::vec![2, 2, 1, 3]);
        assert_eq!(outs[1].shape.dims, alloc::vec![2, 1, 3]);
    }

    #[test]
    fn test_lstm_forward_shape() {
        // seq=2, batch=1, input=3, hidden=4
        let x = make_f32(&[2, 1, 3], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let w = make_f32(&[1, 16, 3], &alloc::vec![0.05f32; 48]);
        let r = make_f32(&[1, 16, 4], &alloc::vec![0.05f32; 64]);
        let outs = op_lstm(&x, &w, &r, None, None, None, "forward").unwrap();
        assert_eq!(outs.len(), 3);
        assert_eq!(outs[0].shape.dims, alloc::vec![2, 1, 1, 4]);
        assert_eq!(outs[1].shape.dims, alloc::vec![1, 1, 4]);
        assert_eq!(outs[2].shape.dims, alloc::vec![1, 1, 4]);
    }

    #[test]
    fn test_lstm_zero_input_state_stays_zero() {
        let x = make_f32(&[3, 1, 2], &alloc::vec![0.0f32; 6]);
        let w = make_f32(&[1, 12, 2], &alloc::vec![0.0f32; 24]);
        let r = make_f32(&[1, 12, 3], &alloc::vec![0.0f32; 36]);
        let outs = op_lstm(&x, &w, &r, None, None, None, "forward").unwrap();
        let yh = &outs[1];
        let yc = &outs[2];
        for i in 0..yh.shape.total_elements() {
            // With zero W/R/B and zero state, gates are sigmoid(0)=0.5,
            // cell candidate tanh(0)=0, so c stays 0 and h = 0.5 * tanh(0) = 0.
            assert!(read_f32(&yh.raw_data, i).abs() < 1e-6);
            assert!(read_f32(&yc.raw_data, i).abs() < 1e-6);
        }
    }

    #[test]
    fn test_lstm_bidirectional_shape() {
        let x = make_f32(&[2, 1, 2], &[0.1, 0.2, 0.3, 0.4]);
        let w = make_f32(&[2, 8, 2], &alloc::vec![0.05f32; 32]);
        let r = make_f32(&[2, 8, 2], &alloc::vec![0.05f32; 32]);
        let outs = op_lstm(&x, &w, &r, None, None, None, "bidirectional").unwrap();
        assert_eq!(outs[0].shape.dims, alloc::vec![2, 2, 1, 2]);
        assert_eq!(outs[1].shape.dims, alloc::vec![2, 1, 2]);
        assert_eq!(outs[2].shape.dims, alloc::vec![2, 1, 2]);
    }

    #[test]
    fn test_gru_forward_shape() {
        // seq=2, batch=1, input=3, hidden=4
        let x = make_f32(&[2, 1, 3], &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let w = make_f32(&[1, 12, 3], &alloc::vec![0.05f32; 36]);
        let r = make_f32(&[1, 12, 4], &alloc::vec![0.05f32; 48]);
        let outs = op_gru(&x, &w, &r, None, None, "forward").unwrap();
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0].shape.dims, alloc::vec![2, 1, 1, 4]);
        assert_eq!(outs[1].shape.dims, alloc::vec![1, 1, 4]);
    }

    #[test]
    fn test_gru_zero_input_zero_state() {
        let x = make_f32(&[3, 1, 2], &alloc::vec![0.0f32; 6]);
        let w = make_f32(&[1, 9, 2], &alloc::vec![0.0f32; 18]);
        let r = make_f32(&[1, 9, 3], &alloc::vec![0.0f32; 27]);
        let outs = op_gru(&x, &w, &r, None, None, "forward").unwrap();
        // z = sigmoid(0)=0.5, h_tilde = tanh(0+0)=0, h_new = 0.5 * 0 + 0.5 * 0 = 0
        for i in 0..outs[1].shape.total_elements() {
            assert!(read_f32(&outs[1].raw_data, i).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gru_with_bias() {
        let x = make_f32(&[1, 1, 2], &[0.0, 0.0]);
        let w = make_f32(&[1, 6, 2], &alloc::vec![0.0f32; 12]);
        let r = make_f32(&[1, 6, 2], &alloc::vec![0.0f32; 12]);
        let b = make_f32(&[1, 12], &alloc::vec![0.0f32; 12]);
        let outs = op_gru(&x, &w, &r, Some(&b), None, "forward").unwrap();
        assert_eq!(outs[1].shape.dims, alloc::vec![1, 1, 2]);
    }

    #[test]
    fn test_rnn_with_initial_state_and_bias() {
        // seq=1, batch=1, input=2, hidden=2
        // x = [1, 0]; W = I; R = 0; B = 0; h0 = [0, 1]
        // h_t = tanh(W·x + R·h0 + b) = tanh([1, 0]) = [tanh(1), 0] ≈ [0.7616, 0]
        let x = make_f32(&[1, 1, 2], &[1.0, 0.0]);
        let w = make_f32(&[1, 2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let r = make_f32(&[1, 2, 2], &[0.0, 0.0, 0.0, 0.0]);
        let h0 = make_f32(&[1, 1, 2], &[0.0, 1.0]);
        let outs = op_rnn(&x, &w, &r, None, Some(&h0), "forward").unwrap();
        let yh = &outs[1];
        let h0_out = read_f32(&yh.raw_data, 0);
        let h1_out = read_f32(&yh.raw_data, 1);
        assert!((h0_out - 0.7616).abs() < 1e-3);
        assert!(h1_out.abs() < 1e-3);
    }

    #[test]
    fn test_rnn_rejects_mismatched_w_input_dim() {
        // X has input_size=3 but W claims input_size=2 → must error, not panic.
        let x = make_f32(&[1, 1, 3], &[0.0; 3]);
        let w = make_f32(&[1, 2, 2], &alloc::vec![0.0f32; 4]);
        let r = make_f32(&[1, 2, 2], &alloc::vec![0.0f32; 4]);
        assert!(matches!(
            op_rnn(&x, &w, &r, None, None, "forward"),
            Err(OpError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn test_lstm_rejects_mismatched_r_hidden_dim() {
        let x = make_f32(&[1, 1, 2], &[0.0; 2]);
        let w = make_f32(&[1, 8, 2], &alloc::vec![0.0f32; 16]);
        // R claims hidden=5 but W implies hidden=2 → must error.
        let r = make_f32(&[1, 8, 5], &alloc::vec![0.0f32; 40]);
        assert!(matches!(
            op_lstm(&x, &w, &r, None, None, None, "forward"),
            Err(OpError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn test_lstm_rejects_mismatched_bias_shape() {
        let x = make_f32(&[1, 1, 2], &[0.0; 2]);
        let w = make_f32(&[1, 8, 2], &alloc::vec![0.0f32; 16]);
        let r = make_f32(&[1, 8, 2], &alloc::vec![0.0f32; 16]);
        // B should be [1, 16] but we give [1, 4] — must error, not panic slicing.
        let b = make_f32(&[1, 4], &alloc::vec![0.0f32; 4]);
        assert!(matches!(
            op_lstm(&x, &w, &r, Some(&b), None, None, "forward"),
            Err(OpError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn test_gru_rejects_mismatched_initial_state() {
        let x = make_f32(&[1, 1, 2], &[0.0; 2]);
        let w = make_f32(&[1, 6, 2], &alloc::vec![0.0f32; 12]);
        let r = make_f32(&[1, 6, 2], &alloc::vec![0.0f32; 12]);
        // h0 should be [1,1,2] but we give [1,1,5] — must error.
        let h0 = make_f32(&[1, 1, 5], &alloc::vec![0.0f32; 5]);
        assert!(matches!(
            op_gru(&x, &w, &r, None, Some(&h0), "forward"),
            Err(OpError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn test_lstm_validation_errors() {
        let x = make_f32(&[2, 1, 3], &[0.0; 6]);
        // Wrong direction
        let w = make_f32(&[1, 16, 3], &alloc::vec![0.0f32; 48]);
        let r = make_f32(&[1, 16, 4], &alloc::vec![0.0f32; 64]);
        assert!(op_lstm(&x, &w, &r, None, None, None, "sideways").is_err());
        // Wrong W rank
        let w_bad = make_f32(&[16, 3], &alloc::vec![0.0f32; 48]);
        assert!(op_lstm(&x, &w_bad, &r, None, None, None, "forward").is_err());
    }
}
