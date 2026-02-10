// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Task scheduler with soft real-time cooperative executor.
//!
//! Three scheduling classes (priority order):
//! 1. SYSTEM — watchdog, health, syslog, timer (hard-RT, < 1 ms)
//! 2. IPC — message routing, TCP, model reload (soft-RT, < 10 ms target)
//! 3. INFERENCE — ONNX execution, GPU commands (soft-RT, throughput)
//!
//! Cooperative multitasking with mandatory yield at ONNX operator boundaries.
//! Work-stealing executor with lock-free per-core run queues.

pub mod executor;
pub mod task;
pub mod timer;

pub use task::{CpuAffinity, SchedulingClass, Task, TaskId, TaskState, TaskType};
