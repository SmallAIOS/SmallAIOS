#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz the protobuf decoder with arbitrary bytes
    let mut decoder = smallaios_onnx_rt::protobuf::ProtoDecoder::new(data);
    while !decoder.is_empty() {
        if let Ok(header) = decoder.read_tag() {
            match header.wire_type {
                smallaios_onnx_rt::protobuf::WireType::Varint => {
                    let _ = decoder.read_varint();
                }
                smallaios_onnx_rt::protobuf::WireType::Fixed64 => {
                    let _ = decoder.read_fixed64();
                }
                smallaios_onnx_rt::protobuf::WireType::LengthDelimited => {
                    let _ = decoder.read_length_delimited();
                }
                smallaios_onnx_rt::protobuf::WireType::Fixed32 => {
                    let _ = decoder.read_fixed32();
                }
                // Deprecated group wire types — real ONNX exports don't use
                // them, but the hardened decoder (PR #98) exposes them so
                // the streaming scanner can skip over group-encoded fields.
                // For fuzzing we just consume-and-ignore.
                smallaios_onnx_rt::protobuf::WireType::StartGroup
                | smallaios_onnx_rt::protobuf::WireType::EndGroup => {
                    break;
                }
            }
        } else {
            break;
        }
    }

    // Also fuzz model loading
    let _ = smallaios_onnx_rt::session::load_model(data);
});
