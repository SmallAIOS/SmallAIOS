#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Fuzz IPC message queue with arbitrary payloads
    let mut queue = smallaios_ipc::pubsub::MessageQueue::new();
    let key = match smallaios_ipc::key_expr::KeyExpr::new("fuzz/test") {
        Ok(k) => k,
        Err(_) => return,
    };
    for chunk in data.chunks(64) {
        let msg = smallaios_ipc::pubsub::Message {
            key: key.clone(),
            payload: chunk.to_vec(),
            timestamp: 0,
            publisher: smallaios_ipc::pubsub::PublisherId(0),
        };
        let _ = queue.push(msg);
    }
    while queue.pop().is_some() {}
});
