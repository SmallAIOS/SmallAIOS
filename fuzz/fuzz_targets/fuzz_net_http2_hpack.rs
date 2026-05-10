#![no_main]
use libfuzzer_sys::fuzz_target;

// HPACK static-table-only decoder. The historic "HPACK Bomb" CVEs
// involve dynamic-table state amplification — we reject every
// dynamic-table reference, so this fuzzer should converge quickly
// on the literal-string overflow / oversized-integer path instead.
fuzz_target!(|data: &[u8]| {
    let _ = smallaios_net::http2::hpack::decode_block(data);
});
