// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SmallAIOS ONNX Runtime
//!
//! Native ONNX model loading, optimization, and inference execution.
//!
//! Components:
//! - Protobuf parser (minimal, ONNX-specific)
//! - Model validation (opset, operators, shapes)
//! - Execution graph builder (topological sort, DAG)
//! - Graph optimizer (fusion, constant folding, memory planning)
//! - CPU execution provider (x86-64 AVX/AVX-512, ARM64 NEON/SVE)
//! - CUDA execution provider (NVIDIA PTX kernels)
//! - Tensor type system and memory management

#![no_std]

extern crate alloc;

pub mod byte_io;
pub mod executor;
pub mod gemm;
pub mod graph;
pub mod memory_planner;
#[cfg(feature = "formal-gate")]
pub mod model_policy;
#[cfg(feature = "verified-boot")]
pub mod model_verify;
pub mod onnx_types;
pub mod operators;
pub mod optimizer;
pub mod parallel;
pub mod protobuf;
pub mod session;
pub mod tensor;
