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
            _ => None,
        }
    }

    /// Returns the size in bytes of a single element of this data type.
    ///
    /// Returns 0 for `String` since strings are variable-length.
    pub fn element_size(&self) -> usize {
        match self {
            DataType::Bool | DataType::Int8 | DataType::Uint8 => 1,
            DataType::Float16 | DataType::BFloat16 | DataType::Int16 | DataType::Uint16 => 2,
            DataType::Float | DataType::Int32 | DataType::Uint32 => 4,
            DataType::Double | DataType::Int64 | DataType::Uint64 => 8,
            DataType::String => 0,
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
        )
    }

    /// Returns `true` if this is a floating-point data type.
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            DataType::Float | DataType::Float16 | DataType::Double | DataType::BFloat16
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
    /// This is `shape.total_elements() * data_type.element_size()`.
    pub fn byte_size(&self) -> usize {
        self.shape.total_elements() * self.data_type.element_size()
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
    }

    #[test]
    fn test_data_type_from_i32_invalid() {
        assert_eq!(DataType::from_i32(0), None);
        assert_eq!(DataType::from_i32(14), None);
        assert_eq!(DataType::from_i32(15), None);
        assert_eq!(DataType::from_i32(17), None);
        assert_eq!(DataType::from_i32(-1), None);
        assert_eq!(DataType::from_i32(100), None);
    }

    #[test]
    fn test_data_type_element_size() {
        assert_eq!(DataType::Bool.element_size(), 1);
        assert_eq!(DataType::Int8.element_size(), 1);
        assert_eq!(DataType::Uint8.element_size(), 1);
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
    }

    #[test]
    fn test_data_type_name() {
        assert_eq!(DataType::Float.name(), "float32");
        assert_eq!(DataType::Double.name(), "float64");
        assert_eq!(DataType::Int32.name(), "int32");
        assert_eq!(DataType::BFloat16.name(), "bfloat16");
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
        assert!(!DataType::Float.is_integer());
        assert!(!DataType::Double.is_integer());
        assert!(!DataType::Bool.is_integer());
        assert!(!DataType::String.is_integer());
    }

    #[test]
    fn test_data_type_is_float() {
        assert!(DataType::Float.is_float());
        assert!(DataType::Float16.is_float());
        assert!(DataType::Double.is_float());
        assert!(DataType::BFloat16.is_float());
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
    fn test_tensor_error_display() {
        use alloc::format;
        assert_eq!(format!("{}", TensorError::ShapeMismatch), "tensor shape mismatch");
        assert_eq!(format!("{}", TensorError::InvalidDataType), "invalid data type");
        assert_eq!(format!("{}", TensorError::DataSizeMismatch), "data size mismatch");
        assert_eq!(format!("{}", TensorError::DimensionOverflow), "dimension overflow");
    }
}
