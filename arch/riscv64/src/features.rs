// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! RISC-V ISA extension detection (23.10).
//!
//! Detects available ISA extensions by reading the `misa` CSR (or its
//! S-mode accessible equivalent) and parsing DTB `riscv,isa` strings.
//!
//! Standard extensions indicated by misa bits [25:0]:
//!   A=0  Atomic
//!   B=1  (reserved)
//!   C=2  Compressed instructions (16-bit)
//!   D=3  Double-precision FP
//!   F=5  Single-precision FP
//!   G    = IMAFD (no separate bit)
//!   H=7  Hypervisor
//!   I=8  Base integer ISA
//!   M=12 Integer multiply/divide
//!   S=18 Supervisor mode
//!   U=20 User mode
//!   V=21 Vector extension

/// Bitmask for each extension in the misa register.
pub const EXT_A: u64 = 1 << 0;  // Atomic
pub const EXT_C: u64 = 1 << 2;  // Compressed
pub const EXT_D: u64 = 1 << 3;  // Double-precision FP
pub const EXT_F: u64 = 1 << 5;  // Single-precision FP
pub const EXT_H: u64 = 1 << 7;  // Hypervisor
pub const EXT_I: u64 = 1 << 8;  // Base integer
pub const EXT_M: u64 = 1 << 12; // Multiply/divide
pub const EXT_S: u64 = 1 << 18; // Supervisor mode
pub const EXT_U: u64 = 1 << 20; // User mode
pub const EXT_V: u64 = 1 << 21; // Vector

/// XLEN (register width) encoded in misa[63:62].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Xlen {
    Rv32 = 1,
    Rv64 = 2,
    Rv128 = 3,
}

/// Detected RISC-V CPU features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RiscvFeatures {
    /// Raw misa value (or reconstructed from DTB).
    pub misa: u64,
    /// XLEN detected from misa[63:62].
    pub xlen: Xlen,
}

impl RiscvFeatures {
    /// Construct features from a raw misa value.
    pub fn from_misa(misa: u64) -> Self {
        let xlen_bits = (misa >> 62) & 0x3;
        let xlen = match xlen_bits {
            1 => Xlen::Rv32,
            2 => Xlen::Rv64,
            3 => Xlen::Rv128,
            _ => Xlen::Rv64, // default
        };
        Self { misa, xlen }
    }

    /// Construct features from a DTB `riscv,isa` string like "rv64imafdc".
    pub fn from_isa_string(s: &[u8]) -> Self {
        let mut misa: u64 = 0;
        let mut xlen = Xlen::Rv64;

        // Parse "rv32" or "rv64" or "rv128" prefix
        let rest = if starts_with_ci(s, b"rv32") {
            xlen = Xlen::Rv32;
            &s[4..]
        } else if starts_with_ci(s, b"rv64") {
            xlen = Xlen::Rv64;
            &s[4..]
        } else if starts_with_ci(s, b"rv128") {
            xlen = Xlen::Rv128;
            &s[5..]
        } else {
            s
        };

        // Each character maps to an extension bit
        for &ch in rest {
            let lower = ch.to_ascii_lowercase();
            if lower >= b'a' && lower <= b'z' {
                misa |= 1u64 << (lower - b'a');
            } else if ch == b'_' {
                // Multi-letter extensions start with '_' — skip for now
                break;
            }
        }

        // Set XLEN in misa[63:62]
        let xlen_bits = match xlen {
            Xlen::Rv32 => 1u64,
            Xlen::Rv64 => 2u64,
            Xlen::Rv128 => 3u64,
        };
        misa |= xlen_bits << 62;

        Self { misa, xlen }
    }

    /// Check if the Atomic extension (A) is present.
    pub fn has_atomic(&self) -> bool {
        self.misa & EXT_A != 0
    }

    /// Check if the Compressed extension (C) is present.
    pub fn has_compressed(&self) -> bool {
        self.misa & EXT_C != 0
    }

    /// Check if the Double-precision FP extension (D) is present.
    pub fn has_double_fp(&self) -> bool {
        self.misa & EXT_D != 0
    }

    /// Check if the Single-precision FP extension (F) is present.
    pub fn has_single_fp(&self) -> bool {
        self.misa & EXT_F != 0
    }

    /// Check if the Hypervisor extension (H) is present.
    pub fn has_hypervisor(&self) -> bool {
        self.misa & EXT_H != 0
    }

    /// Check if the Multiply/Divide extension (M) is present.
    pub fn has_multiply(&self) -> bool {
        self.misa & EXT_M != 0
    }

    /// Check if Supervisor mode is available.
    pub fn has_supervisor(&self) -> bool {
        self.misa & EXT_S != 0
    }

    /// Check if User mode is available.
    pub fn has_user(&self) -> bool {
        self.misa & EXT_U != 0
    }

    /// Check if the Vector extension (V) is present.
    pub fn has_vector(&self) -> bool {
        self.misa & EXT_V != 0
    }

    /// Check if the CPU supports the "G" profile (IMAFD).
    pub fn is_gc(&self) -> bool {
        let gc_mask = EXT_I | EXT_M | EXT_A | EXT_F | EXT_D | EXT_C;
        self.misa & gc_mask == gc_mask
    }

    /// Check if this is RV64GC (the standard SmallAIOS target).
    pub fn is_rv64gc(&self) -> bool {
        self.xlen == Xlen::Rv64 && self.is_gc()
    }
}

/// Case-insensitive starts_with for byte slices.
fn starts_with_ci(haystack: &[u8], prefix: &[u8]) -> bool {
    if haystack.len() < prefix.len() {
        return false;
    }
    for (a, b) in haystack.iter().zip(prefix.iter()) {
        if a.to_ascii_lowercase() != b.to_ascii_lowercase() {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_isa_string_rv64gc() {
        let f = RiscvFeatures::from_isa_string(b"rv64imafdc");
        assert_eq!(f.xlen, Xlen::Rv64);
        assert!(f.has_atomic());
        assert!(f.has_compressed());
        assert!(f.has_double_fp());
        assert!(f.has_single_fp());
        assert!(f.has_multiply());
        assert!(f.is_gc());
        assert!(f.is_rv64gc());
    }

    #[test]
    fn test_from_isa_string_rv64imafdcsu() {
        let f = RiscvFeatures::from_isa_string(b"rv64imafdcsu");
        assert!(f.has_supervisor());
        assert!(f.has_user());
        assert!(f.is_rv64gc());
    }

    #[test]
    fn test_from_isa_string_rv32i() {
        let f = RiscvFeatures::from_isa_string(b"rv32i");
        assert_eq!(f.xlen, Xlen::Rv32);
        assert!(!f.has_atomic());
        assert!(!f.is_gc());
        assert!(!f.is_rv64gc());
    }

    #[test]
    fn test_from_isa_string_with_vector() {
        let f = RiscvFeatures::from_isa_string(b"rv64imafdcv");
        assert!(f.has_vector());
        assert!(f.is_rv64gc());
    }

    #[test]
    fn test_from_misa() {
        // Simulate RV64GC misa value
        let misa = (2u64 << 62) | EXT_I | EXT_M | EXT_A | EXT_F | EXT_D | EXT_C;
        let f = RiscvFeatures::from_misa(misa);
        assert_eq!(f.xlen, Xlen::Rv64);
        assert!(f.is_gc());
        assert!(f.is_rv64gc());
    }

    #[test]
    fn test_extension_bits() {
        assert_eq!(EXT_A, 1 << 0);
        assert_eq!(EXT_C, 1 << 2);
        assert_eq!(EXT_D, 1 << 3);
        assert_eq!(EXT_F, 1 << 5);
        assert_eq!(EXT_I, 1 << 8);
        assert_eq!(EXT_M, 1 << 12);
        assert_eq!(EXT_V, 1 << 21);
    }

    #[test]
    fn test_not_gc_without_compressed() {
        let f = RiscvFeatures::from_isa_string(b"rv64imafd");
        assert!(f.has_atomic());
        assert!(!f.has_compressed());
        assert!(!f.is_gc()); // Missing C
    }

    #[test]
    fn test_starts_with_ci() {
        assert!(starts_with_ci(b"RV64IMAFDC", b"rv64"));
        assert!(starts_with_ci(b"rv32i", b"RV32"));
        assert!(!starts_with_ci(b"rv64", b"rv128"));
    }

    #[test]
    fn test_hypervisor_extension() {
        let f = RiscvFeatures::from_isa_string(b"rv64imafdch");
        assert!(f.has_hypervisor());
    }
}
