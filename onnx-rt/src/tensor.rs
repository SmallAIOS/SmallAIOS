// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tensor type system for the SmallAIOS ONNX runtime.
//!
//! Provides the core data types matching the ONNX specification,
//! tensor shapes with dimension tracking, and the Tensor container
//! for model inputs, outputs, and intermediate values.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

/// ONNX data types with their protocol buffer enum values.
///
/// Values match the ONNX TensorProto.DataType enum exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum DataType {
    /// 32-bit IEEE 754 floating point.
    Float = 1,
    /// 8-bit unsigned integer.
    Uint8 = 2,
    /// 8-bit signed integer.
    Int8 = 3,
    /// 16-bit unsigned integer.
    Uint16 = 4,
    /// 16-bit signed integer.
    Int16 = 5,
    /// 32-bit signed integer.
    Int32 = 6,
    /// 64-bit signed integer.
    Int64 = 7,
    /// Variable-length string (UTF-8).
    String = 8,
    /// Boolean.
    Bool = 9,
    /// 16-bit IEEE 754 floating point.
    Float16 = 10,
    /// 64-bit IEEE 754 floating point.
    Double = 11,
    /// 32-bit unsigned integer.
    Uint32 = 12,
    /// 64-bit unsigned integer.
    Uint64 = 13,
    /// 16-bit brain floating point.
    BFloat16 = 16,
    /// 8-bit floating point, E4M3 format (4-bit exponent, 3-bit mantissa, finite-only).
    ///
    /// Matches the ONNX `FLOAT8E4M3FN` TensorProto type.
    Float8E4M3 = 17,
    /// 8-bit floating point, E5M2 format (5-bit exponent, 2-bit mantissa).
    ///
    /// Matches the ONNX `FLOAT8E5M2` TensorProto type.
    Float8E5M2 = 19,
    /// Unsigned 4-bit integer (packed 2 values per byte).
    ///
    /// Matches the ONNX `UINT4` TensorProto type. `element_size` returns 0
    /// because 4-bit data cannot be expressed as an integer byte width;
    /// callers handling packed 4-bit buffers must compute `(n + 1) / 2`
    /// bytes explicitly.
    Uint4 = 21,
    /// Signed 4-bit integer (packed 2 values per byte).
    ///
    /// Matches the ONNX `INT4` TensorProto type. See `Uint4` for the
    /// `element_size` note.
    Int4 = 22,
    /// 4-bit floating point, E2M1 format (Blackwell NVFP4).
    ///
    /// Matches the ONNX `FLOAT4E2M1` TensorProto type. Packed 2 values
    /// per byte; see `Uint4` for the `element_size` note.
    Float4E2M1 = 23,
}

impl DataType {
    /// Converts an ONNX protobuf data type integer to a `DataType`.
    ///
    /// Returns `None` for unknown or unsupported type codes.
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            1 => Some(DataType::Float),
            2 => Some(DataType::Uint8),
            3 => Some(DataType::Int8),
            4 => Some(DataType::Uint16),
            5 => Some(DataType::Int16),
            6 => Some(DataType::Int32),
            7 => Some(DataType::Int64),
            8 => Some(DataType::String),
            9 => Some(DataType::Bool),
            10 => Some(DataType::Float16),
            11 => Some(DataType::Double),
            12 => Some(DataType::Uint32),
            13 => Some(DataType::Uint64),
            16 => Some(DataType::BFloat16),
            17 => Some(DataType::Float8E4M3),
            19 => Some(DataType::Float8E5M2),
            21 => Some(DataType::Uint4),
            22 => Some(DataType::Int4),
            23 => Some(DataType::Float4E2M1),
            _ => None,
        }
    }

    /// Returns the size in bytes of a single element of this data type.
    ///
    /// Returns 0 for `String` (variable-length) and for the sub-byte types
    /// `Int4`, `Uint4`, `Float4E2M1` (stored packed 2 per byte — callers
    /// must compute `(n + 1) / 2` bytes explicitly).
    pub fn element_size(&self) -> usize {
        match self {
            DataType::Bool
            | DataType::Int8
            | DataType::Uint8
            | DataType::Float8E4M3
            | DataType::Float8E5M2 => 1,
            DataType::Float16 | DataType::BFloat16 | DataType::Int16 | DataType::Uint16 => 2,
            DataType::Float | DataType::Int32 | DataType::Uint32 => 4,
            DataType::Double | DataType::Int64 | DataType::Uint64 => 8,
            DataType::String | DataType::Int4 | DataType::Uint4 | DataType::Float4E2M1 => 0,
        }
    }

    /// Returns the human-readable name of this data type.
    pub fn name(&self) -> &'static str {
        match self {
            DataType::Float => "float32",
            DataType::Uint8 => "uint8",
            DataType::Int8 => "int8",
            DataType::Uint16 => "uint16",
            DataType::Int16 => "int16",
            DataType::Int32 => "int32",
            DataType::Int64 => "int64",
            DataType::String => "string",
            DataType::Bool => "bool",
            DataType::Float16 => "float16",
            DataType::Double => "float64",
            DataType::Uint32 => "uint32",
            DataType::Uint64 => "uint64",
            DataType::BFloat16 => "bfloat16",
            DataType::Float8E4M3 => "float8e4m3",
            DataType::Float8E5M2 => "float8e5m2",
            DataType::Uint4 => "uint4",
            DataType::Int4 => "int4",
            DataType::Float4E2M1 => "float4e2m1",
        }
    }

    /// Returns `true` if this is an integer data type.
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            DataType::Uint8
                | DataType::Int8
                | DataType::Uint16
                | DataType::Int16
                | DataType::Int32
                | DataType::Int64
                | DataType::Uint32
                | DataType::Uint64
                | DataType::Int4
                | DataType::Uint4
        )
    }

    /// Returns `true` if this is a floating-point data type.
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            DataType::Float
                | DataType::Float16
                | DataType::Double
                | DataType::BFloat16
                | DataType::Float8E4M3
                | DataType::Float8E5M2
                | DataType::Float4E2M1
        )
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Multi-dimensional tensor shape with dimension tracking.
///
/// Dimensions are stored as `i64` to support symbolic dimensions:
/// - Positive values represent concrete sizes.
/// - `-1` represents a symbolic/dynamic dimension.
/// - Other negative values are invalid for concrete shapes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorShape {
    /// Dimension sizes. Positive for concrete, -1 for symbolic.
    pub dims: Vec<i64>,
}

impl TensorShape {
    /// Creates a new tensor shape from dimension sizes.
    pub fn new(dims: Vec<i64>) -> Self {
        Self { dims }
    }

    /// Creates a scalar (zero-dimensional) tensor shape.
    pub fn scalar() -> Self {
        Self { dims: Vec::new() }
    }

    /// Returns the number of dimensions (rank).
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }

    /// Returns the total number of elements in the tensor.
    ///
    /// For shapes with symbolic dimensions (`-1`), those dimensions
    /// are treated as size 1 for the purpose of this calculation.
    /// A scalar shape (zero dimensions) has exactly 1 element.
    pub fn total_elements(&self) -> usize {
        if self.dims.is_empty() {
            return 1;
        }
        self.dims
            .iter()
            .map(|&d| if d < 0 { 1usize } else { d as usize })
            .fold(1usize, |acc, d| acc.saturating_mul(d))
    }

    /// Returns `true` if this is a scalar (zero-dimensional) shape.
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }

    /// Validates the shape.
    ///
    /// A shape is valid if all dimensions are either positive (concrete)
    /// or exactly `-1` (symbolic). Zero-dimensional (scalar) shapes are valid.
    pub fn is_valid(&self) -> bool {
        self.dims.iter().all(|&d| d > 0 || d == -1)
    }
}

/// Errors that can occur during tensor operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    /// Shape dimensions do not match expected values.
    ShapeMismatch,
    /// An invalid or unsupported data type was specified.
    InvalidDataType,
    /// Raw data size does not match the expected byte size for the shape and type.
    DataSizeMismatch,
    /// Dimension computation overflowed.
    DimensionOverflow,
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch => f.write_str("tensor shape mismatch"),
            TensorError::InvalidDataType => f.write_str("invalid data type"),
            TensorError::DataSizeMismatch => f.write_str("data size mismatch"),
            TensorError::DimensionOverflow => f.write_str("dimension overflow"),
        }
    }
}

/// A multi-dimensional array with a data type and shape.
///
/// This is the primary data container for ONNX model inputs, outputs,
/// and intermediate computation results. Data is stored as raw bytes
/// to support zero-copy operations and all ONNX data types.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    /// The element data type.
    pub data_type: DataType,
    /// The tensor shape (dimensions).
    pub shape: TensorShape,
    /// Optional tensor name (from the ONNX graph).
    pub name: String,
    /// Raw byte data in little-endian format.
    pub raw_data: Vec<u8>,
}

/// Converts a byte buffer of little-endian BFloat16 values to `f32`.
///
/// BFloat16 is the high 16 bits of the IEEE 754 binary32 representation, so
/// zero-extending the 16-bit value into the high half of a `u32` and
/// reinterpreting via [`f32::from_bits`] recovers the corresponding `f32`
/// exactly (with 8-bit mantissa precision).
///
/// Trailing bytes (an odd-length input) are ignored; the returned `Vec`
/// contains `bytes.len() / 2` elements.
pub fn bf16_to_f32(bytes: &[u8]) -> Vec<f32> {
    let n = bytes.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let lo = bytes[2 * i] as u32;
        let hi = bytes[2 * i + 1] as u32;
        let bf = (hi << 8) | lo; // little-endian u16
                                 // Place BF16 value into the upper half of a u32 (mantissa zero-extend).
        let bits = bf << 16;
        out.push(f32::from_bits(bits));
    }
    out
}

/// Converts an `f32` slice to a packed little-endian BFloat16 byte buffer.
///
/// Uses IEEE 754 round-to-nearest-even on the 16 discarded mantissa bits.
/// NaN inputs are preserved as NaN (not flushed to zero); the canonical
/// BF16 NaN bit pattern with a non-zero payload is emitted.
pub fn f32_to_bf16(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        let bits = v.to_bits();
        let bf = if v.is_nan() {
            // Preserve NaN — set a non-zero mantissa bit in the BF16 payload
            // so the result remains a quiet NaN. Keep the sign bit.
            ((bits >> 16) as u16) | 0x0040
        } else {
            // Round-to-nearest-even on the low 16 bits.
            let rounding_bias = 0x0000_7FFFu32 + ((bits >> 16) & 1);
            ((bits.wrapping_add(rounding_bias)) >> 16) as u16
        };
        out.push((bf & 0xFF) as u8);
        out.push((bf >> 8) as u8);
    }
    out
}

impl Tensor {
    /// Creates a new empty tensor with the given data type, shape, and name.
    pub fn new(data_type: DataType, shape: TensorShape, name: String) -> Self {
        Self {
            data_type,
            shape,
            name,
            raw_data: Vec::new(),
        }
    }

    /// Returns the expected size in bytes for this tensor's shape and data type.
    ///
    /// For byte-sized and larger types this is
    /// `shape.total_elements() * data_type.element_size()`. For
    /// sub-byte packed types (`Int4`, `Uint4`, `Float4E2M1`) the
    /// payload stores 2 elements per byte, so the size is
    /// `(total_elements + 1) / 2`. `String` returns 0 because layout
    /// is not size-uniform.
    pub fn byte_size(&self) -> usize {
        let n = self.shape.total_elements();
        match self.data_type {
            DataType::Int4 | DataType::Uint4 | DataType::Float4E2M1 => n.div_ceil(2),
            DataType::String => 0,
            _ => n * self.data_type.element_size(),
        }
    }

    /// Returns `true` if the tensor contains no data.
    pub fn is_empty(&self) -> bool {
        self.raw_data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use alloc::vec;

    // ---- DataType tests ----

    #[test]
    fn test_data_type_from_i32_valid() {
        assert_eq!(DataType::from_i32(1), Some(DataType::Float));
        assert_eq!(DataType::from_i32(2), Some(DataType::Uint8));
        assert_eq!(DataType::from_i32(3), Some(DataType::Int8));
        assert_eq!(DataType::from_i32(6), Some(DataType::Int32));
        assert_eq!(DataType::from_i32(7), Some(DataType::Int64));
        assert_eq!(DataType::from_i32(9), Some(DataType::Bool));
        assert_eq!(DataType::from_i32(10), Some(DataType::Float16));
        assert_eq!(DataType::from_i32(11), Some(DataType::Double));
        assert_eq!(DataType::from_i32(16), Some(DataType::BFloat16));
        assert_eq!(DataType::from_i32(17), Some(DataType::Float8E4M3));
        assert_eq!(DataType::from_i32(19), Some(DataType::Float8E5M2));
        assert_eq!(DataType::from_i32(21), Some(DataType::Uint4));
        assert_eq!(DataType::from_i32(22), Some(DataType::Int4));
        assert_eq!(DataType::from_i32(23), Some(DataType::Float4E2M1));
    }

    #[test]
    fn test_data_type_from_i32_invalid() {
        assert_eq!(DataType::from_i32(0), None);
        assert_eq!(DataType::from_i32(14), None);
        assert_eq!(DataType::from_i32(15), None);
        assert_eq!(DataType::from_i32(18), None);
        assert_eq!(DataType::from_i32(20), None);
        assert_eq!(DataType::from_i32(24), None);
        assert_eq!(DataType::from_i32(-1), None);
        assert_eq!(DataType::from_i32(100), None);
    }

    #[test]
    fn test_data_type_element_size() {
        assert_eq!(DataType::Bool.element_size(), 1);
        assert_eq!(DataType::Int8.element_size(), 1);
        assert_eq!(DataType::Uint8.element_size(), 1);
        assert_eq!(DataType::Float8E4M3.element_size(), 1);
        assert_eq!(DataType::Float8E5M2.element_size(), 1);
        assert_eq!(DataType::Float16.element_size(), 2);
        assert_eq!(DataType::BFloat16.element_size(), 2);
        assert_eq!(DataType::Int16.element_size(), 2);
        assert_eq!(DataType::Float.element_size(), 4);
        assert_eq!(DataType::Int32.element_size(), 4);
        assert_eq!(DataType::Uint32.element_size(), 4);
        assert_eq!(DataType::Double.element_size(), 8);
        assert_eq!(DataType::Int64.element_size(), 8);
        assert_eq!(DataType::Uint64.element_size(), 8);
        assert_eq!(DataType::String.element_size(), 0);
        // Sub-byte types return 0 — see docstring.
        assert_eq!(DataType::Int4.element_size(), 0);
        assert_eq!(DataType::Uint4.element_size(), 0);
        assert_eq!(DataType::Float4E2M1.element_size(), 0);
    }

    #[test]
    fn test_data_type_name() {
        assert_eq!(DataType::Float.name(), "float32");
        assert_eq!(DataType::Double.name(), "float64");
        assert_eq!(DataType::Int32.name(), "int32");
        assert_eq!(DataType::BFloat16.name(), "bfloat16");
        assert_eq!(DataType::Float8E4M3.name(), "float8e4m3");
        assert_eq!(DataType::Float8E5M2.name(), "float8e5m2");
        assert_eq!(DataType::Int4.name(), "int4");
        assert_eq!(DataType::Uint4.name(), "uint4");
        assert_eq!(DataType::Float4E2M1.name(), "float4e2m1");
        assert_eq!(DataType::Bool.name(), "bool");
        assert_eq!(DataType::String.name(), "string");
    }

    #[test]
    fn test_data_type_is_integer() {
        assert!(DataType::Int8.is_integer());
        assert!(DataType::Uint8.is_integer());
        assert!(DataType::Int16.is_integer());
        assert!(DataType::Uint16.is_integer());
        assert!(DataType::Int32.is_integer());
        assert!(DataType::Int64.is_integer());
        assert!(DataType::Uint32.is_integer());
        assert!(DataType::Uint64.is_integer());
        assert!(DataType::Int4.is_integer());
        assert!(DataType::Uint4.is_integer());
        assert!(!DataType::Float.is_integer());
        assert!(!DataType::Double.is_integer());
        assert!(!DataType::Bool.is_integer());
        assert!(!DataType::String.is_integer());
        assert!(!DataType::Float4E2M1.is_integer());
    }

    #[test]
    fn test_data_type_is_float() {
        assert!(DataType::Float.is_float());
        assert!(DataType::Float16.is_float());
        assert!(DataType::Double.is_float());
        assert!(DataType::BFloat16.is_float());
        assert!(DataType::Float8E4M3.is_float());
        assert!(DataType::Float8E5M2.is_float());
        assert!(DataType::Float4E2M1.is_float());
        assert!(!DataType::Int4.is_float());
        assert!(!DataType::Uint4.is_float());
        assert!(!DataType::Int32.is_float());
        assert!(!DataType::Bool.is_float());
        assert!(!DataType::String.is_float());
        assert!(!DataType::Uint64.is_float());
    }

    #[test]
    fn test_data_type_display() {
        use alloc::format;
        assert_eq!(format!("{}", DataType::Float), "float32");
        assert_eq!(format!("{}", DataType::BFloat16), "bfloat16");
        assert_eq!(format!("{}", DataType::Int64), "int64");
    }

    // ---- TensorShape tests ----

    #[test]
    fn test_shape_new_and_ndim() {
        let shape = TensorShape::new(vec![2, 3, 4]);
        assert_eq!(shape.ndim(), 3);
        assert_eq!(shape.dims, vec![2, 3, 4]);
    }

    #[test]
    fn test_shape_scalar() {
        let shape = TensorShape::scalar();
        assert!(shape.is_scalar());
        assert_eq!(shape.ndim(), 0);
        assert_eq!(shape.total_elements(), 1);
        assert!(shape.is_valid());
    }

    #[test]
    fn test_shape_total_elements() {
        let shape = TensorShape::new(vec![2, 3, 4]);
        assert_eq!(shape.total_elements(), 24);

        let shape_1d = TensorShape::new(vec![10]);
        assert_eq!(shape_1d.total_elements(), 10);

        let shape_with_symbolic = TensorShape::new(vec![-1, 3, 224, 224]);
        // Symbolic dims treated as 1
        assert_eq!(shape_with_symbolic.total_elements(), 3 * 224 * 224);
    }

    #[test]
    fn test_shape_is_valid() {
        assert!(TensorShape::new(vec![1, 2, 3]).is_valid());
        assert!(TensorShape::new(vec![-1, 3, 224]).is_valid());
        assert!(TensorShape::scalar().is_valid());

        // Invalid: negative dims other than -1
        assert!(!TensorShape::new(vec![-2, 3]).is_valid());
        assert!(!TensorShape::new(vec![3, -5]).is_valid());

        // Invalid: zero-sized dimension
        assert!(!TensorShape::new(vec![0, 3]).is_valid());
    }

    // ---- Tensor tests ----

    #[test]
    fn test_tensor_new() {
        let tensor = Tensor::new(
            DataType::Float,
            TensorShape::new(vec![2, 3]),
            String::from("input"),
        );
        assert_eq!(tensor.data_type, DataType::Float);
        assert_eq!(tensor.shape.ndim(), 2);
        assert_eq!(tensor.name, "input");
        assert!(tensor.is_empty());
    }

    #[test]
    fn test_tensor_byte_size() {
        let tensor = Tensor::new(
            DataType::Float,
            TensorShape::new(vec![2, 3]),
            String::from("t"),
        );
        // 2 * 3 = 6 elements * 4 bytes = 24
        assert_eq!(tensor.byte_size(), 24);

        let tensor_i64 = Tensor::new(
            DataType::Int64,
            TensorShape::new(vec![10]),
            String::from("ids"),
        );
        // 10 elements * 8 bytes = 80
        assert_eq!(tensor_i64.byte_size(), 80);

        let scalar = Tensor::new(
            DataType::Double,
            TensorShape::scalar(),
            String::from("loss"),
        );
        // 1 element * 8 bytes = 8
        assert_eq!(scalar.byte_size(), 8);
    }

    #[test]
    fn test_tensor_is_empty() {
        let mut tensor = Tensor::new(
            DataType::Uint8,
            TensorShape::new(vec![4]),
            String::from("mask"),
        );
        assert!(tensor.is_empty());

        tensor.raw_data = vec![0, 1, 1, 0];
        assert!(!tensor.is_empty());
    }

    #[test]
    fn test_tensor_bf16_byte_size() {
        // BF16 element_size is 2 bytes.
        assert_eq!(DataType::BFloat16.element_size(), 2);
        let t = Tensor::new(
            DataType::BFloat16,
            TensorShape::new(vec![4, 8]),
            String::from("weights"),
        );
        // 4 * 8 = 32 elements * 2 bytes = 64
        assert_eq!(t.byte_size(), 64);

        let scalar = Tensor::new(DataType::BFloat16, TensorShape::scalar(), String::from("s"));
        assert_eq!(scalar.byte_size(), 2);
    }

    // ---- BF16 conversion tests ----

    fn f32_to_bf16_roundtrip(v: f32) -> f32 {
        let bytes = f32_to_bf16(&[v]);
        bf16_to_f32(&bytes)[0]
    }

    #[test]
    fn test_bf16_roundtrip_exact_zero() {
        assert_eq!(f32_to_bf16_roundtrip(0.0).to_bits(), 0.0f32.to_bits());
        assert_eq!(f32_to_bf16_roundtrip(-0.0).to_bits(), (-0.0f32).to_bits());
    }

    #[test]
    fn test_bf16_roundtrip_exact_one() {
        // 1.0 has zero mantissa below the BF16 cut, so round-trip is exact.
        assert_eq!(f32_to_bf16_roundtrip(1.0), 1.0);
        assert_eq!(f32_to_bf16_roundtrip(-1.0), -1.0);
        assert_eq!(f32_to_bf16_roundtrip(2.0), 2.0);
        assert_eq!(f32_to_bf16_roundtrip(0.5), 0.5);
    }

    #[test]
    fn test_bf16_roundtrip_approximate() {
        // Values with low-bit mantissa detail lose precision — check ~1e-2.
        #[allow(clippy::approx_constant)]
        let cases = [3.14159_f32, -2.71828, 0.1, -0.01, 1234.5];
        for &v in &cases {
            let r = f32_to_bf16_roundtrip(v);
            let rel = (r - v).abs() / v.abs().max(1e-6);
            assert!(rel < 1e-2, "round-trip {} -> {} rel={}", v, r, rel);
        }
    }

    #[test]
    fn test_bf16_infinity_preserved() {
        assert!(f32_to_bf16_roundtrip(f32::INFINITY).is_infinite());
        assert!(f32_to_bf16_roundtrip(f32::INFINITY).is_sign_positive());
        assert!(f32_to_bf16_roundtrip(f32::NEG_INFINITY).is_infinite());
        assert!(f32_to_bf16_roundtrip(f32::NEG_INFINITY).is_sign_negative());
    }

    #[test]
    fn test_bf16_nan_preserved() {
        let r = f32_to_bf16_roundtrip(f32::NAN);
        assert!(r.is_nan(), "NaN must round-trip as NaN");
    }

    #[test]
    fn test_bf16_round_to_nearest_even_midpoint() {
        // Build an f32 whose low 16 bits form an exact-halfway midpoint:
        // mantissa low word = 0x8000. Round-to-nearest-even breaks ties
        // toward the value whose LSB is zero.
        //
        // Case A: upper BF16 even (LSB of high word == 0) -> round down.
        // Case B: upper BF16 odd  (LSB of high word == 1) -> round up.

        // Choose high-word patterns that unambiguously preserve the LSB.
        // f32 bits = 0x3F80_0000 is 1.0; high word LSB = 0 (even). Adding
        // 0x8000 produces a halfway case that should round *down* to 1.0.
        let even_mid = f32::from_bits(0x3F80_8000);
        let r = f32_to_bf16_roundtrip(even_mid);
        assert_eq!(r.to_bits(), 0x3F80_0000, "even midpoint must round down");

        // f32 bits = 0x3F81_0000 (BF16 = 0x3F81, LSB = 1, odd). Halfway
        // produces 0x3F81_8000; RNE rounds *up* to 0x3F82_0000.
        let odd_mid = f32::from_bits(0x3F81_8000);
        let r = f32_to_bf16_roundtrip(odd_mid);
        assert_eq!(r.to_bits(), 0x3F82_0000, "odd midpoint must round up");
    }

    #[test]
    fn test_bf16_to_f32_handles_odd_length() {
        // Odd-length input: trailing byte is ignored. 2 bytes = 1 element.
        let bytes = [0x80u8, 0x3F, 0xAB];
        let out = bf16_to_f32(&bytes);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0], 1.0);
    }

    #[test]
    fn test_bf16_vector_roundtrip_small() {
        let values = [0.0f32, 1.0, -1.0, 2.0, -2.0, 0.5, -0.5];
        let bytes = f32_to_bf16(&values);
        assert_eq!(bytes.len(), values.len() * 2);
        let decoded = bf16_to_f32(&bytes);
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_tensor_error_display() {
        use alloc::format;
        assert_eq!(
            format!("{}", TensorError::ShapeMismatch),
            "tensor shape mismatch"
        );
        assert_eq!(
            format!("{}", TensorError::InvalidDataType),
            "invalid data type"
        );
        assert_eq!(
            format!("{}", TensorError::DataSizeMismatch),
            "data size mismatch"
        );
        assert_eq!(
            format!("{}", TensorError::DimensionOverflow),
            "dimension overflow"
        );
    }
}
