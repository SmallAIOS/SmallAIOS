// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Byte-level I/O helpers for tensor data buffers.
//!
//! Centralizes the conversion between raw byte slices and primitive
//! numeric types. Eliminates duplicated `f32::from_le_bytes([...])`
//! patterns scattered across operators.rs and executor.rs.

use crate::tensor::DataType;
use alloc::vec::Vec;

/// Size in bytes of an f32 value.
pub const F32_SIZE: usize = 4;
/// Size in bytes of an i32 value.
pub const I32_SIZE: usize = 4;
/// Size in bytes of an i64 value.
pub const I64_SIZE: usize = 8;
/// Size in bytes of an f64 value.
pub const F64_SIZE: usize = 8;

/// Read an f32 value from a tensor data buffer at the given element index.
#[inline]
pub fn read_f32(data: &[u8], idx: usize) -> f32 {
    let off = idx * F32_SIZE;
    f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

/// Write an f32 value to a tensor data buffer at the given element index.
#[inline]
pub fn write_f32(data: &mut [u8], idx: usize, val: f32) {
    let off = idx * F32_SIZE;
    let bytes = val.to_le_bytes();
    data[off..off + F32_SIZE].copy_from_slice(&bytes);
}

/// Read an i32 value from a tensor data buffer at the given element index.
#[inline]
pub fn read_i32(data: &[u8], idx: usize) -> i32 {
    let off = idx * I32_SIZE;
    i32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]])
}

/// Write an i32 value to a tensor data buffer at the given element index.
#[inline]
pub fn write_i32(data: &mut [u8], idx: usize, val: i32) {
    let off = idx * I32_SIZE;
    data[off..off + I32_SIZE].copy_from_slice(&val.to_le_bytes());
}

/// Read an i64 value from a tensor data buffer at the given element index.
#[inline]
pub fn read_i64(data: &[u8], idx: usize) -> i64 {
    let off = idx * I64_SIZE;
    i64::from_le_bytes([
        data[off],
        data[off + 1],
        data[off + 2],
        data[off + 3],
        data[off + 4],
        data[off + 5],
        data[off + 6],
        data[off + 7],
    ])
}

/// Write an i64 value to a tensor data buffer at the given element index.
#[inline]
pub fn write_i64(data: &mut [u8], idx: usize, val: i64) {
    let off = idx * I64_SIZE;
    data[off..off + I64_SIZE].copy_from_slice(&val.to_le_bytes());
}

/// Read an f64 value from a tensor data buffer at the given element index.
#[inline]
pub fn read_f64(data: &[u8], idx: usize) -> f64 {
    let off = idx * F64_SIZE;
    f64::from_le_bytes([
        data[off],
        data[off + 1],
        data[off + 2],
        data[off + 3],
        data[off + 4],
        data[off + 5],
        data[off + 6],
        data[off + 7],
    ])
}

/// Write an f64 value to a tensor data buffer at the given element index.
#[inline]
pub fn write_f64(data: &mut [u8], idx: usize, val: f64) {
    let off = idx * F64_SIZE;
    data[off..off + F64_SIZE].copy_from_slice(&val.to_le_bytes());
}

/// Allocate a zero-filled tensor data buffer for the given element count and dtype.
pub fn allocate_tensor_data(elements: usize, dtype: DataType) -> Vec<u8> {
    alloc::vec![0u8; elements * dtype.element_size()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_f32_round_trip() {
        let mut buf = vec![0u8; 16];
        for i in 0..4 {
            write_f32(&mut buf, i, (i as f32) * 1.5);
        }
        for i in 0..4 {
            assert_eq!(read_f32(&buf, i), (i as f32) * 1.5);
        }
    }

    #[test]
    fn test_i32_round_trip() {
        let mut buf = vec![0u8; 16];
        write_i32(&mut buf, 0, -42);
        write_i32(&mut buf, 1, 123_456);
        assert_eq!(read_i32(&buf, 0), -42);
        assert_eq!(read_i32(&buf, 1), 123_456);
    }

    #[test]
    fn test_i64_round_trip() {
        let mut buf = vec![0u8; 16];
        write_i64(&mut buf, 0, -12345i64);
        write_i64(&mut buf, 1, 67890i64);
        assert_eq!(read_i64(&buf, 0), -12345);
        assert_eq!(read_i64(&buf, 1), 67890);
    }

    #[test]
    fn test_f64_round_trip() {
        let mut buf = vec![0u8; 16];
        write_f64(&mut buf, 0, 3.14159);
        write_f64(&mut buf, 1, -2.71828);
        assert_eq!(read_f64(&buf, 0), 3.14159);
        assert_eq!(read_f64(&buf, 1), -2.71828);
    }

    #[test]
    fn test_allocate_tensor_data() {
        let buf = allocate_tensor_data(10, DataType::Float);
        assert_eq!(buf.len(), 40);
        assert!(buf.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_allocate_tensor_data_int64() {
        let buf = allocate_tensor_data(5, DataType::Int64);
        assert_eq!(buf.len(), 40);
    }
}
