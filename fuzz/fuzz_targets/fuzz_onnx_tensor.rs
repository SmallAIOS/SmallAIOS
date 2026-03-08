#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz tensor shape construction with arbitrary dimensions
    if data.len() >= 2 {
        let ndims = (data[0] % 6) as usize; // 0-5 dimensions
        let mut dims = Vec::new();
        for i in 0..ndims {
            if 1 + i * 2 + 1 < data.len() {
                let d = i16::from_le_bytes([data[1 + i * 2], data[1 + i * 2 + 1]]);
                dims.push(d as i64);
            }
        }
        let shape = smallaios_onnx_rt::tensor::TensorShape::new(dims);
        let _ = shape.total_elements();
        let _ = shape.is_valid();
        let _ = shape.ndim();
    }
});
