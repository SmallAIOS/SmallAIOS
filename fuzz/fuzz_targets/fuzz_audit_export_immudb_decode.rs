#![no_main]
use libfuzzer_sys::fuzz_target;
use smallaios_audit_export::immudb::schema;

// Phase 3 of verifiable-audit-log-v1: every immudb response
// decoder runs against attacker-controlled bytes. None of these
// may panic, allocate beyond the input length, or read past the
// end of the buffer.
fuzz_target!(|data: &[u8]| {
    let _ = schema::ImmutableState::decode(data);
    let _ = schema::TxHeader::decode(data);
    let _ = schema::TxEntry::decode(data);
    let _ = schema::Tx::decode(data);
    let _ = schema::LinearProof::decode(data);
    let _ = schema::InclusionProof::decode(data);
    let _ = schema::DualProof::decode(data);
    let _ = schema::VerifiableTx::decode(data);
    let _ = schema::Signature::decode(data);
    let _ = schema::KVMetadata::decode(data);
    let _ = schema::KeyValue::decode(data);
});
