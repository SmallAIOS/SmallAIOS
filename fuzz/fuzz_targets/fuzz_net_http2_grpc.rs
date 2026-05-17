#![no_main]
use libfuzzer_sys::fuzz_target;

// gRPC length-prefixed-message decoder. Cap matches the default
// audit-export inbound message size (4 MiB).
fuzz_target!(|data: &[u8]| {
    let _ = smallaios_net::http2::grpc::try_decode_message(
        data,
        smallaios_net::http2::grpc::DEFAULT_MAX_MESSAGE_LEN,
    );
});
