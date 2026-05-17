#![no_main]
use libfuzzer_sys::fuzz_target;

// Phase 2 of verifiable-audit-log-v1: HTTP/2 frame-header parser
// is the first attacker-controlled surface on the gRPC path.
fuzz_target!(|data: &[u8]| {
    let _ = smallaios_net::http2::frame::FrameHeader::parse(data);
});
