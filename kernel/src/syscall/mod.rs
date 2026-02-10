// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Syscall dispatch interface.
//!
//! ~46 syscalls organized into categories:
//! - Memory (0x00-0x0F): allocation, tensor buffers, GPU mapping
//! - Task (0x10-0x1F): spawn, yield, join, affinity
//! - IPC (0x20-0x2F): pub/sub, request/reply
//! - ONNX (0x30-0x3F): model load, session, inference
//! - Device (0x40-0x4F): enumerate, open, ioctl, DMA
//! - System (0x50-0x5F): info, time, shutdown, random
//! - Capability (0x60-0x6F): create, revoke, delegate, check
