// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Task scheduler with async executor.
//!
//! Cooperative multitasking using Rust async/await:
//! - Work-stealing executor with one worker per CPU core
//! - Lock-free per-core run queues
//! - Tasks yield at ONNX operator boundaries
