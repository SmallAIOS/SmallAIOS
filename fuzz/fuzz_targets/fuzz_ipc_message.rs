#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz IPC message queue with arbitrary payloads
    let mut queue = smallaios_ipc::pubsub::MessageQueue::new();
    for chunk in data.chunks(64) {
        let msg = smallaios_ipc::pubsub::Message {
            topic: smallaios_ipc::pubsub::KeyExpr::new(0),
            payload: chunk.to_vec(),
            timestamp: 0,
            publisher_id: 0,
        };
        let _ = queue.push(msg);
    }
    while queue.pop().is_some() {}
});
