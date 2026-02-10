// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! x86-64 Hardware Abstraction Layer
//!
//! Provides platform-specific initialization and hardware access:
//! - UEFI/Multiboot2 boot entry
//! - GDT and IDT setup
//! - Local APIC and I/O APIC
//! - 4/5-level page table management
//! - CPUID feature detection (SSE, AVX, AVX-512, AMX)
//! - Serial console (COM1)

#![no_std]
