// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tier 2 operator implementations for the SmallAIOS ONNX runtime.
//!
//! These operators expand support beyond the original Tier 1 set to cover
//! transformer building blocks (Einsum, Split, Expand, …), recurrent layers
//! (RNN, LSTM, GRU), composite activations (GELU, LeakyReLU, ELU, Swish),
//! quantized INT8 ops (QuantizeLinear, DequantizeLinear, QLinearMatMul,
//! QLinearConv), comparison/selection ops, and additional element-wise math
//! primitives (Pow, Sqrt, Exp, Log, Erf, Neg, Abs, Floor, Ceil, Round).
//!
//! Each submodule is independent and depends only on `tensor`, `byte_io`,
//! and `operators::OpError`. Tier 1 operators continue to live in
//! `crate::operators` to keep the diff focused and to avoid churning
//! existing call sites.

pub mod activations;
pub mod common;
pub mod compare;
pub mod control_flow;
pub mod data_select;
pub mod generative;
pub mod math;
pub mod microsoft;
pub mod normalization;
pub mod quantized;
pub mod recurrent;
pub mod reduction;
pub mod shape_data;
pub mod transformer;
pub mod vision;
