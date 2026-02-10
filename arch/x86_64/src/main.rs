// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SmallAIOS x86-64 kernel binary.
//!
//! This is the binary entry point. All code lives in the library crate;
//! this file just ensures the crate compiles as an executable with the
//! correct `_start` entry point linked in.

#![no_std]
#![no_main]
#![feature(naked_functions)]

// Pull in the library crate to ensure _start and kernel_main are linked.
extern crate smallaios_arch_x86_64;
