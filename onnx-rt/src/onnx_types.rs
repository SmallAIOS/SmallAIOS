// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ONNX model types (code-generated equivalent).
//!
//! These Rust types represent the ONNX protobuf schema without requiring
//! a protobuf dependency. They mirror the structure of the `.proto3`
//! definitions from the ONNX specification and are used throughout
//! the runtime for model loading, validation, and graph construction.

use alloc::string::String;
use alloc::vec::Vec;

/// Current ONNX IR version supported by this runtime.
pub const CURRENT_IR_VERSION: i64 = 9;

/// ONNX attribute types corresponding to `AttributeProto.AttributeType`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum AttributeType {
    /// Undefined / not set.
    Undefined = 0,
    /// Single 32-bit float.
    Float = 1,
    /// Single 64-bit integer.
    Int = 2,
    /// Single byte string.
    String = 3,
    /// Single tensor value.
    Tensor = 4,
    /// Single sub-graph.
    Graph = 5,
    /// List of 32-bit floats.
    Floats = 6,
    /// List of 64-bit integers.
    Ints = 7,
    /// List of byte strings.
    Strings = 8,
    /// List of tensor values.
    Tensors = 9,
    /// List of sub-graphs.
    Graphs = 10,
}

impl AttributeType {
    /// Converts an ONNX protobuf attribute type integer to an `AttributeType`.
    ///
    /// Returns `None` for unknown type codes.
    pub fn from_i32(value: i32) -> Option<Self> {
        match value {
            0 => Some(AttributeType::Undefined),
            1 => Some(AttributeType::Float),
            2 => Some(AttributeType::Int),
            3 => Some(AttributeType::String),
            4 => Some(AttributeType::Tensor),
            5 => Some(AttributeType::Graph),
            6 => Some(AttributeType::Floats),
            7 => Some(AttributeType::Ints),
            8 => Some(AttributeType::Strings),
            9 => Some(AttributeType::Tensors),
            10 => Some(AttributeType::Graphs),
            _ => None,
        }
    }
}

/// An attribute of an ONNX node, carrying typed constant data.
///
/// Mirrors `onnx.AttributeProto` from the ONNX protobuf schema.
/// Only the field matching `attr_type` is meaningful; other fields
/// retain their default (zero/empty) values.
#[derive(Debug, Clone, PartialEq)]
pub struct AttributeProto {
    /// Attribute name.
    pub name: String,
    /// Discriminator for which value field is active.
    pub attr_type: AttributeType,
    /// Float value (when `attr_type == Float`).
    pub f: f32,
    /// Integer value (when `attr_type == Int`).
    pub i: i64,
    /// Byte string value (when `attr_type == String`).
    pub s: Vec<u8>,
    /// Float list (when `attr_type == Floats`).
    pub floats: Vec<f32>,
    /// Integer list (when `attr_type == Ints`).
    pub ints: Vec<i64>,
    /// Byte string list (when `attr_type == Strings`).
    pub strings: Vec<Vec<u8>>,
    /// Tensor value (when `attr_type == Tensor`). Boxed to keep
    /// `AttributeProto` small when the common scalar case applies.
    pub t: Option<alloc::boxed::Box<TensorProto>>,
    /// Nested sub-graph value (when `attr_type == Graph`). Boxed because
    /// `GraphProto` is much larger than the scalar attribute fields and
    /// the common case does not carry a sub-graph. Used by `If`/`Loop`/
    /// `Scan` control-flow operators whose body is an embedded graph.
    pub g: Option<alloc::boxed::Box<GraphProto>>,
}

impl Default for AttributeProto {
    fn default() -> Self {
        Self {
            name: String::new(),
            attr_type: AttributeType::Undefined,
            f: 0.0,
            i: 0,
            s: Vec::new(),
            floats: Vec::new(),
            ints: Vec::new(),
            strings: Vec::new(),
            t: None,
            g: None,
        }
    }
}

/// A serialized tensor value embedded in the model.
///
/// Mirrors `onnx.TensorProto`. Data can be stored in `raw_data`
/// (preferred, compact binary) or in the typed data fields.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TensorProto {
    /// Tensor dimensions.
    pub dims: Vec<i64>,
    /// Data type as ONNX `TensorProto.DataType` integer.
    pub data_type: i32,
    /// Optional tensor name.
    pub name: String,
    /// Raw binary data (little-endian).
    pub raw_data: Vec<u8>,
    /// Float element data (when data_type == FLOAT).
    pub float_data: Vec<f32>,
    /// Int32 element data (when data_type == INT32).
    pub int32_data: Vec<i32>,
    /// Int64 element data (when data_type == INT64).
    pub int64_data: Vec<i64>,
    /// Double element data (when data_type == DOUBLE).
    pub double_data: Vec<f64>,
}

/// Type and shape information for a graph value (input/output).
///
/// Simplified representation of `onnx.ValueInfoProto` combined with
/// its nested `TypeProto.Tensor` fields.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ValueInfoProto {
    /// Value name (matches corresponding tensor names).
    pub name: String,
    /// Element data type as ONNX integer code.
    pub elem_type: i32,
    /// Shape dimensions (may include -1 for symbolic dims).
    pub shape: Vec<i64>,
}

/// A single computation node in the ONNX graph.
///
/// Mirrors `onnx.NodeProto`. Each node represents one operator
/// invocation with named inputs, outputs, and operator-specific attributes.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NodeProto {
    /// Input tensor names.
    pub input: Vec<String>,
    /// Output tensor names.
    pub output: Vec<String>,
    /// Optional node name (for debugging).
    pub name: String,
    /// Operator type (e.g., "Conv", "Relu", "MatMul").
    pub op_type: String,
    /// Operator domain (empty string means the default ONNX domain).
    pub domain: String,
    /// Operator-specific attributes.
    pub attribute: Vec<AttributeProto>,
}

/// The computation graph of an ONNX model.
///
/// Mirrors `onnx.GraphProto`. Contains the topologically ordered
/// list of nodes, graph inputs/outputs, and pre-trained weight tensors.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct GraphProto {
    /// Computation nodes in topological order.
    pub node: Vec<NodeProto>,
    /// Graph name.
    pub name: String,
    /// Graph input value descriptors.
    pub input: Vec<ValueInfoProto>,
    /// Graph output value descriptors.
    pub output: Vec<ValueInfoProto>,
    /// Pre-trained weight tensors (initializers).
    pub initializer: Vec<TensorProto>,
}

/// An operator set version binding.
///
/// Mirrors `onnx.OperatorSetIdProto`. Specifies which version of
/// an operator set domain is used by the model.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct OperatorSetIdProto {
    /// Operator set domain (empty string = default ONNX domain).
    pub domain: String,
    /// Operator set version number.
    pub version: i64,
}

/// Top-level ONNX model container.
///
/// Mirrors `onnx.ModelProto`. Contains model metadata, operator set
/// imports, and the computation graph.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ModelProto {
    /// ONNX IR version used to encode this model.
    pub ir_version: i64,
    /// Operator set version bindings.
    pub opset_import: Vec<OperatorSetIdProto>,
    /// Name of the tool that produced this model.
    pub producer_name: String,
    /// Version of the producing tool.
    pub producer_version: String,
    /// Model domain (for model versioning).
    pub domain: String,
    /// Model version number.
    pub model_version: i64,
    /// Human-readable documentation.
    pub doc_string: String,
    /// The computation graph.
    pub graph: Option<GraphProto>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;
    use alloc::vec;

    // ---- AttributeType tests ----

    #[test]
    fn test_attribute_type_from_i32_valid() {
        assert_eq!(AttributeType::from_i32(0), Some(AttributeType::Undefined));
        assert_eq!(AttributeType::from_i32(1), Some(AttributeType::Float));
        assert_eq!(AttributeType::from_i32(2), Some(AttributeType::Int));
        assert_eq!(AttributeType::from_i32(3), Some(AttributeType::String));
        assert_eq!(AttributeType::from_i32(4), Some(AttributeType::Tensor));
        assert_eq!(AttributeType::from_i32(5), Some(AttributeType::Graph));
        assert_eq!(AttributeType::from_i32(6), Some(AttributeType::Floats));
        assert_eq!(AttributeType::from_i32(7), Some(AttributeType::Ints));
        assert_eq!(AttributeType::from_i32(8), Some(AttributeType::Strings));
        assert_eq!(AttributeType::from_i32(9), Some(AttributeType::Tensors));
        assert_eq!(AttributeType::from_i32(10), Some(AttributeType::Graphs));
    }

    #[test]
    fn test_attribute_type_from_i32_invalid() {
        assert_eq!(AttributeType::from_i32(-1), None);
        assert_eq!(AttributeType::from_i32(11), None);
        assert_eq!(AttributeType::from_i32(100), None);
    }

    // ---- AttributeProto tests ----

    #[test]
    fn test_attribute_proto_default() {
        let attr = AttributeProto::default();
        assert_eq!(attr.name, "");
        assert_eq!(attr.attr_type, AttributeType::Undefined);
        assert_eq!(attr.f, 0.0);
        assert_eq!(attr.i, 0);
        assert!(attr.s.is_empty());
        assert!(attr.floats.is_empty());
        assert!(attr.ints.is_empty());
        assert!(attr.strings.is_empty());
        assert!(attr.t.is_none());
        assert!(attr.g.is_none());
    }

    #[test]
    fn test_attribute_proto_graph_valued() {
        // An attribute with a nested body GraphProto mimics the `body`
        // attribute of a `Loop` / `Scan` node, or the `then_branch` /
        // `else_branch` attributes of an `If` node.
        let inner = GraphProto {
            name: String::from("body"),
            node: vec![NodeProto {
                op_type: String::from("Add"),
                input: vec![String::from("a"), String::from("b")],
                output: vec![String::from("y")],
                ..NodeProto::default()
            }],
            ..GraphProto::default()
        };
        let attr = AttributeProto {
            name: String::from("body"),
            attr_type: AttributeType::Graph,
            g: Some(alloc::boxed::Box::new(inner)),
            ..AttributeProto::default()
        };
        assert_eq!(attr.attr_type, AttributeType::Graph);
        let g = attr.g.as_ref().expect("g should be populated");
        assert_eq!(g.name, "body");
        assert_eq!(g.node.len(), 1);
        assert_eq!(g.node[0].op_type, "Add");
    }

    #[test]
    fn test_attribute_proto_float() {
        let attr = AttributeProto {
            name: String::from("alpha"),
            attr_type: AttributeType::Float,
            f: 0.01,
            ..AttributeProto::default()
        };
        assert_eq!(attr.name, "alpha");
        assert_eq!(attr.attr_type, AttributeType::Float);
        assert!((attr.f - 0.01).abs() < f32::EPSILON);
    }

    #[test]
    fn test_attribute_proto_ints_list() {
        let attr = AttributeProto {
            name: String::from("kernel_shape"),
            attr_type: AttributeType::Ints,
            ints: vec![3, 3],
            ..AttributeProto::default()
        };
        assert_eq!(attr.ints, vec![3, 3]);
        assert_eq!(attr.attr_type, AttributeType::Ints);
    }

    // ---- TensorProto tests ----

    #[test]
    fn test_tensor_proto_default() {
        let tp = TensorProto::default();
        assert!(tp.dims.is_empty());
        assert_eq!(tp.data_type, 0);
        assert_eq!(tp.name, "");
        assert!(tp.raw_data.is_empty());
        assert!(tp.float_data.is_empty());
        assert!(tp.int32_data.is_empty());
        assert!(tp.int64_data.is_empty());
        assert!(tp.double_data.is_empty());
    }

    #[test]
    fn test_tensor_proto_with_float_data() {
        let tp = TensorProto {
            dims: vec![2, 3],
            data_type: 1, // FLOAT
            name: String::from("weight"),
            float_data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            ..TensorProto::default()
        };
        assert_eq!(tp.dims, vec![2, 3]);
        assert_eq!(tp.data_type, 1);
        assert_eq!(tp.float_data.len(), 6);
        assert_eq!(tp.name, "weight");
    }

    #[test]
    fn test_tensor_proto_with_raw_data() {
        let tp = TensorProto {
            dims: vec![4],
            data_type: 2, // UINT8
            raw_data: vec![10, 20, 30, 40],
            ..TensorProto::default()
        };
        assert_eq!(tp.raw_data, vec![10, 20, 30, 40]);
        assert_eq!(tp.dims, vec![4]);
    }

    // ---- NodeProto tests ----

    #[test]
    fn test_node_proto_construction() {
        let node = NodeProto {
            input: vec![String::from("X"), String::from("W")],
            output: vec![String::from("Y")],
            name: String::from("conv1"),
            op_type: String::from("Conv"),
            domain: String::new(),
            attribute: vec![AttributeProto {
                name: String::from("kernel_shape"),
                attr_type: AttributeType::Ints,
                ints: vec![3, 3],
                ..AttributeProto::default()
            }],
        };
        assert_eq!(node.op_type, "Conv");
        assert_eq!(node.input.len(), 2);
        assert_eq!(node.output.len(), 1);
        assert_eq!(node.attribute.len(), 1);
        assert_eq!(node.attribute[0].name, "kernel_shape");
    }

    // ---- GraphProto tests ----

    #[test]
    fn test_graph_proto_with_nodes() {
        let relu_node = NodeProto {
            input: vec![String::from("X")],
            output: vec![String::from("Y")],
            op_type: String::from("Relu"),
            ..NodeProto::default()
        };

        let graph = GraphProto {
            node: vec![relu_node],
            name: String::from("test_graph"),
            input: vec![ValueInfoProto {
                name: String::from("X"),
                elem_type: 1,
                shape: vec![1, 3, 224, 224],
            }],
            output: vec![ValueInfoProto {
                name: String::from("Y"),
                elem_type: 1,
                shape: vec![1, 3, 224, 224],
            }],
            initializer: Vec::new(),
        };
        assert_eq!(graph.name, "test_graph");
        assert_eq!(graph.node.len(), 1);
        assert_eq!(graph.node[0].op_type, "Relu");
        assert_eq!(graph.input.len(), 1);
        assert_eq!(graph.output.len(), 1);
        assert_eq!(graph.input[0].shape, vec![1, 3, 224, 224]);
    }

    // ---- ModelProto tests ----

    #[test]
    fn test_model_proto_default() {
        let model = ModelProto::default();
        assert_eq!(model.ir_version, 0);
        assert!(model.opset_import.is_empty());
        assert_eq!(model.producer_name, "");
        assert_eq!(model.producer_version, "");
        assert_eq!(model.domain, "");
        assert_eq!(model.model_version, 0);
        assert_eq!(model.doc_string, "");
        assert!(model.graph.is_none());
    }

    #[test]
    fn test_model_proto_full() {
        let model = ModelProto {
            ir_version: CURRENT_IR_VERSION,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            }],
            producer_name: String::from("smallaios-test"),
            producer_version: String::from("1.0"),
            domain: String::from("ai.smallaios"),
            model_version: 1,
            doc_string: String::from("Test model"),
            graph: Some(GraphProto {
                name: String::from("main"),
                ..GraphProto::default()
            }),
        };
        assert_eq!(model.ir_version, 9);
        assert_eq!(model.opset_import.len(), 1);
        assert_eq!(model.opset_import[0].version, 17);
        assert_eq!(model.producer_name, "smallaios-test");
        assert!(model.graph.is_some());
        assert_eq!(model.graph.as_ref().unwrap().name, "main");
    }

    #[test]
    fn test_current_ir_version() {
        assert_eq!(CURRENT_IR_VERSION, 9);
    }

    // ---- ValueInfoProto tests ----

    #[test]
    fn test_value_info_proto() {
        let vi = ValueInfoProto {
            name: String::from("input_0"),
            elem_type: 1,
            shape: vec![-1, 3, 224, 224],
        };
        assert_eq!(vi.name, "input_0");
        assert_eq!(vi.elem_type, 1);
        assert_eq!(vi.shape.len(), 4);
        assert_eq!(vi.shape[0], -1); // batch dim is symbolic
    }
}
