// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Dataflow inference runner.
//!
//! Holds a collection of loaded ONNX `Session`s keyed by model name and
//! exposes a [`DataflowRunner::process_message`] entry point that decodes an
//! inference request from the wire, dispatches it to the right session, and
//! re-encodes the response.
//!
//! This module is feature-gated (`#[cfg(feature = "onnx")]`) so the base IPC
//! crate stays free of the `smallaios-onnx-rt` dependency.
//!
//! The runner is intentionally transport-agnostic: it never spawns threads
//! and never touches the network itself. Callers wire it up to whatever
//! transport they prefer (the in-process router in [`crate::router`], DDS via
//! a Zenoh adapter, TCP, ...). A thin helper that binds the runner to the
//! existing pub/sub primitives is provided in
//! [`serve_dataflow_runner`].

#![cfg(feature = "onnx")]

use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::sync::atomic::{AtomicUsize, Ordering};

use smallaios_onnx_rt::session::Session;

use crate::inference_proto::{
    handle_run_inference_with_session, InferenceRequest, InferenceResponse, ResponseStatus,
};
use crate::IpcError;

/// Default topic prefix for dataflow inference messaging.
pub const DEFAULT_TOPIC_PREFIX: &str = "smallaios/inference";

/// Default bounded-queue depth for the runner (drop-oldest semantics).
pub const DEFAULT_MAX_QUEUE_DEPTH: usize = 16;

/// Runtime configuration for a [`DataflowRunner`].
#[derive(Debug, Clone)]
pub struct DataflowRunnerConfig {
    /// Maximum number of in-flight inference messages before drop-oldest
    /// backpressure kicks in.
    pub max_queue_depth: usize,
    /// Topic prefix used to derive input/output/error topics per model.
    ///
    /// Given a prefix of `smallaios/inference` and a model named `relu`,
    /// the derived topics are:
    ///
    /// - `smallaios/inference/relu/input`
    /// - `smallaios/inference/relu/output`
    /// - `smallaios/inference/relu/error`
    pub topic_prefix: String,
}

impl Default for DataflowRunnerConfig {
    fn default() -> Self {
        Self {
            max_queue_depth: DEFAULT_MAX_QUEUE_DEPTH,
            topic_prefix: String::from(DEFAULT_TOPIC_PREFIX),
        }
    }
}

/// Dataflow inference runner.
///
/// Holds one or more loaded ONNX sessions and dispatches incoming binary
/// inference requests to the session matching the requested model name.
///
/// The runner does not own a scheduler or transport. Callers are expected
/// to drive it by calling [`DataflowRunner::process_message`] for each
/// incoming payload.
pub struct DataflowRunner {
    config: DataflowRunnerConfig,
    sessions: BTreeMap<String, Session>,
    dropped_messages: AtomicUsize,
}

impl DataflowRunner {
    /// Create a new runner with the given configuration.
    pub fn new(config: DataflowRunnerConfig) -> Self {
        Self {
            config,
            sessions: BTreeMap::new(),
            dropped_messages: AtomicUsize::new(0),
        }
    }

    /// Register a session under a model name. Replaces any existing session
    /// registered under that name.
    pub fn register_session(&mut self, model_name: String, session: Session) {
        self.sessions.insert(model_name, session);
    }

    /// Returns the configured maximum queue depth.
    pub fn max_queue_depth(&self) -> usize {
        self.config.max_queue_depth
    }

    /// Returns the configured topic prefix.
    pub fn topic_prefix(&self) -> &str {
        &self.config.topic_prefix
    }

    /// Returns `true` if a session is registered under the given model name.
    pub fn has_model(&self, model: &str) -> bool {
        self.sessions.contains_key(model)
    }

    /// Returns the list of registered model names, in sorted order.
    pub fn model_names(&self) -> Vec<String> {
        self.sessions.keys().cloned().collect()
    }

    /// Build the input topic for a given model: `<prefix>/<model>/input`.
    pub fn input_topic(&self, model: &str) -> String {
        let mut s = String::with_capacity(self.config.topic_prefix.len() + model.len() + 8);
        s.push_str(&self.config.topic_prefix);
        s.push('/');
        s.push_str(model);
        s.push_str("/input");
        s
    }

    /// Build the output topic for a given model: `<prefix>/<model>/output`.
    pub fn output_topic(&self, model: &str) -> String {
        let mut s = String::with_capacity(self.config.topic_prefix.len() + model.len() + 9);
        s.push_str(&self.config.topic_prefix);
        s.push('/');
        s.push_str(model);
        s.push_str("/output");
        s
    }

    /// Build the error topic for a given model: `<prefix>/<model>/error`.
    pub fn error_topic(&self, model: &str) -> String {
        let mut s = String::with_capacity(self.config.topic_prefix.len() + model.len() + 8);
        s.push_str(&self.config.topic_prefix);
        s.push('/');
        s.push_str(model);
        s.push_str("/error");
        s
    }

    /// Build the wildcard input subscription: `<prefix>/*/input`.
    pub fn input_wildcard(&self) -> String {
        let mut s = String::with_capacity(self.config.topic_prefix.len() + 10);
        s.push_str(&self.config.topic_prefix);
        s.push_str("/*/input");
        s
    }

    /// Extract the model name from a concrete input topic matching the
    /// runner's prefix. Returns `None` if the topic does not have the form
    /// `<prefix>/<model>/input`.
    pub fn extract_model_name<'a>(&self, topic: &'a str) -> Option<&'a str> {
        let rest = topic.strip_prefix(self.config.topic_prefix.as_str())?;
        let rest = rest.strip_prefix('/')?;
        let rest = rest.strip_suffix("/input")?;
        if rest.is_empty() || rest.contains('/') {
            return None;
        }
        Some(rest)
    }

    /// Number of messages dropped by backpressure (drop-oldest semantics).
    pub fn dropped_messages(&self) -> usize {
        self.dropped_messages.load(Ordering::Relaxed)
    }

    /// Record a drop event from an external scheduler.
    ///
    /// This is exposed so the pub/sub driver (which owns the bounded queue)
    /// can increment the runner's drop counter when it evicts the oldest
    /// message.
    pub fn record_drop(&self) {
        self.dropped_messages.fetch_add(1, Ordering::Relaxed);
    }

    /// Process a single inference message for a specific model.
    ///
    /// Decodes the binary payload, dispatches to the registered session,
    /// and encodes the response into the wire format. Errors are also
    /// encoded as wire-format responses so that callers can publish the
    /// bytes on the error topic instead of silently dropping the request.
    ///
    /// # Errors
    ///
    /// - [`IpcError::NotFound`] if no session is registered for `model`
    ///   *or* the session reports the request refers to an unknown model.
    /// - [`IpcError::InvalidProtocol`] on decode errors, wrong request type,
    ///   or shape/tensor mismatches reported by the ONNX runtime.
    pub fn process_message(&self, model: &str, payload: &[u8]) -> Result<Vec<u8>, IpcError> {
        let request = InferenceRequest::decode(payload)?;
        let session = self.sessions.get(model).ok_or(IpcError::NotFound)?;
        let response = handle_run_inference_with_session(session, &request)?;
        Ok(response.encode())
    }

    /// Like [`process_message`](Self::process_message) but never returns an
    /// error: transport-layer errors are packaged as `InferenceResponse`
    /// failure messages and encoded to bytes.
    ///
    /// This is the variant intended for pub/sub plumbing: every input
    /// message produces an output message, whether success or failure, so
    /// subscribers on the error topic can correlate request IDs.
    pub fn process_message_into_response(&self, model: &str, payload: &[u8]) -> Vec<u8> {
        // Try to decode the request so we can echo back the request ID.
        let decoded = InferenceRequest::decode(payload);
        let request_id = decoded.as_ref().map(|r| r.request_id).unwrap_or(0);

        let status = match &decoded {
            Ok(req) => {
                let session = match self.sessions.get(model) {
                    Some(s) => s,
                    None => {
                        return InferenceResponse {
                            request_id,
                            status: ResponseStatus::ModelNotFound,
                            outputs: Vec::new(),
                            elapsed_us: 0,
                        }
                        .encode();
                    }
                };
                match handle_run_inference_with_session(session, req) {
                    Ok(resp) => return resp.encode(),
                    Err(IpcError::NotFound) => ResponseStatus::ModelNotFound,
                    Err(IpcError::InvalidProtocol) => ResponseStatus::InvalidInput,
                    Err(_) => ResponseStatus::InternalError,
                }
            }
            Err(_) => ResponseStatus::InvalidInput,
        };

        InferenceResponse {
            request_id,
            status,
            outputs: Vec::new(),
            elapsed_us: 0,
        }
        .encode()
    }
}

/// Description of a pending dataflow request, as consumed by
/// [`serve_dataflow_runner`] from the existing pub/sub primitives.
pub struct PendingRequest {
    /// Fully qualified input topic the request arrived on.
    pub topic: String,
    /// Raw request payload.
    pub payload: Vec<u8>,
}

/// Drain a pub/sub [`crate::pubsub::Subscriber`] into the runner and produce
/// a list of messages to publish.
///
/// Returns a vector of `(topic, payload)` pairs representing the result
/// messages to publish for each input that was processed. Drop-oldest
/// backpressure is applied by only draining up to
/// [`DataflowRunnerConfig::max_queue_depth`] messages from the subscriber
/// per call; any excess older messages are counted as drops on the runner.
///
/// This is the free function form the architecture doc calls for: it takes
/// a runner and an existing pub/sub subscriber (which is the input side)
/// and returns publishable outputs the caller then pushes on its publisher.
/// Keeping it free-function lets the same runner serve multiple transports.
pub fn serve_dataflow_runner(
    runner: &DataflowRunner,
    subscriber: &mut crate::pubsub::Subscriber,
) -> Vec<(String, Vec<u8>)> {
    let mut out = Vec::new();

    // Drop-oldest: if the subscriber has more than max_queue_depth pending
    // messages, pop and discard the oldest ones until we're at the limit.
    let depth = runner.max_queue_depth();
    while subscriber.pending() > depth {
        if subscriber.receive().is_some() {
            runner.record_drop();
        } else {
            break;
        }
    }

    // Process remaining messages in order.
    while let Some(msg) = subscriber.receive() {
        let topic_str = msg.key.as_str().to_string();
        let model = match runner.extract_model_name(&topic_str) {
            Some(m) => m.to_string(),
            None => continue,
        };
        let bytes = runner.process_message_into_response(&model, &msg.payload);
        out.push((runner.output_topic(&model), bytes));
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use alloc::vec;

    use smallaios_onnx_rt::onnx_types::{
        GraphProto, ModelProto, NodeProto, OperatorSetIdProto, ValueInfoProto, CURRENT_IR_VERSION,
    };
    use smallaios_onnx_rt::session::{Session, SessionConfig};

    use crate::inference_proto::{
        InferenceRequest, InferenceRequestType, InferenceResponse, TensorData, TensorDataType,
    };
    use crate::key_expr::KeyExpr;
    use crate::pubsub::{Publisher, PublisherId, Subscriber};
    use crate::router::SubscriptionId;

    fn relu_session() -> Session {
        let model = ModelProto {
            ir_version: CURRENT_IR_VERSION,
            opset_import: vec![OperatorSetIdProto {
                domain: String::new(),
                version: 17,
            }],
            producer_name: String::from("runner-test"),
            producer_version: String::from("1.0"),
            domain: String::new(),
            model_version: 1,
            doc_string: String::new(),
            graph: Some(GraphProto {
                name: String::from("relu"),
                node: vec![NodeProto {
                    op_type: String::from("Relu"),
                    name: String::from("relu0"),
                    input: vec![String::from("x")],
                    output: vec![String::from("y")],
                    ..NodeProto::default()
                }],
                input: vec![ValueInfoProto {
                    name: String::from("x"),
                    elem_type: 1,
                    shape: vec![1, 3],
                }],
                output: vec![ValueInfoProto {
                    name: String::from("y"),
                    elem_type: 1,
                    shape: vec![1, 3],
                }],
                initializer: Vec::new(),
            }),
        };
        let mut session = Session::new(SessionConfig::default());
        session.initialize(&model).unwrap();
        session
    }

    fn relu_request() -> InferenceRequest {
        let mut data = Vec::new();
        for v in [-2.0f32, 0.0, 3.5].iter() {
            data.extend_from_slice(&v.to_le_bytes());
        }
        InferenceRequest {
            request_type: InferenceRequestType::RunInference,
            request_id: 99,
            model_id: 0,
            inputs: vec![TensorData {
                name: String::from("x"),
                data_type: TensorDataType::Float32,
                shape: vec![1, 3],
                data,
            }],
            timeout_ms: 100,
        }
    }

    fn bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    // ---- config ----

    #[test]
    fn default_config_values() {
        let c = DataflowRunnerConfig::default();
        assert_eq!(c.max_queue_depth, DEFAULT_MAX_QUEUE_DEPTH);
        assert_eq!(c.topic_prefix, DEFAULT_TOPIC_PREFIX);
    }

    // ---- topic naming ----

    #[test]
    fn topic_naming_follows_convention() {
        let runner = DataflowRunner::new(DataflowRunnerConfig::default());
        assert_eq!(runner.input_topic("relu"), "smallaios/inference/relu/input");
        assert_eq!(
            runner.output_topic("relu"),
            "smallaios/inference/relu/output"
        );
        assert_eq!(runner.error_topic("relu"), "smallaios/inference/relu/error");
        assert_eq!(runner.input_wildcard(), "smallaios/inference/*/input");
    }

    #[test]
    fn topic_naming_honours_custom_prefix() {
        let runner = DataflowRunner::new(DataflowRunnerConfig {
            max_queue_depth: 4,
            topic_prefix: String::from("my/models"),
        });
        assert_eq!(runner.input_topic("m1"), "my/models/m1/input");
        assert_eq!(runner.output_topic("m1"), "my/models/m1/output");
    }

    #[test]
    fn extract_model_name_from_topic() {
        let runner = DataflowRunner::new(DataflowRunnerConfig::default());
        assert_eq!(
            runner.extract_model_name("smallaios/inference/relu/input"),
            Some("relu")
        );
        // Output topic must not be parsed as input.
        assert_eq!(
            runner.extract_model_name("smallaios/inference/relu/output"),
            None
        );
        // Wrong prefix.
        assert_eq!(runner.extract_model_name("other/prefix/relu/input"), None);
        // Nested model name (contains slash).
        assert_eq!(
            runner.extract_model_name("smallaios/inference/a/b/input"),
            None
        );
    }

    // ---- session registration ----

    #[test]
    fn register_and_query_sessions() {
        let mut runner = DataflowRunner::new(DataflowRunnerConfig::default());
        assert!(!runner.has_model("relu"));
        runner.register_session(String::from("relu"), relu_session());
        assert!(runner.has_model("relu"));
        assert_eq!(runner.model_names(), vec![String::from("relu")]);
    }

    // ---- process_message ----

    #[test]
    fn process_message_runs_relu_and_encodes_response() {
        let mut runner = DataflowRunner::new(DataflowRunnerConfig::default());
        runner.register_session(String::from("relu"), relu_session());

        let payload = relu_request().encode();
        let out = runner.process_message("relu", &payload).unwrap();
        let resp = InferenceResponse::decode(&out).unwrap();

        assert_eq!(resp.request_id, 99);
        assert_eq!(resp.outputs.len(), 1);
        assert_eq!(resp.outputs[0].name, "y");
        assert_eq!(bytes_to_f32(&resp.outputs[0].data), vec![0.0, 0.0, 3.5]);
    }

    #[test]
    fn process_message_unknown_model_returns_not_found() {
        let runner = DataflowRunner::new(DataflowRunnerConfig::default());
        let payload = relu_request().encode();
        let err = runner.process_message("missing", &payload).unwrap_err();
        assert_eq!(err, IpcError::NotFound);
    }

    #[test]
    fn process_message_bad_payload_returns_invalid_protocol() {
        let runner = DataflowRunner::new(DataflowRunnerConfig::default());
        let err = runner.process_message("relu", &[0u8; 4]).unwrap_err();
        // Decoding fails first; we surface the decoder error.
        assert!(matches!(
            err,
            IpcError::PacketTooShort | IpcError::InvalidProtocol
        ));
    }

    // ---- process_message_into_response (pub/sub-friendly variant) ----

    #[test]
    fn process_message_into_response_missing_model_encodes_status() {
        let runner = DataflowRunner::new(DataflowRunnerConfig::default());
        let payload = relu_request().encode();
        let bytes = runner.process_message_into_response("none", &payload);
        let resp = InferenceResponse::decode(&bytes).unwrap();
        assert_eq!(resp.request_id, 99);
        assert_eq!(resp.status, ResponseStatus::ModelNotFound);
    }

    #[test]
    fn process_message_into_response_bad_payload_encodes_status() {
        let runner = DataflowRunner::new(DataflowRunnerConfig::default());
        let bytes = runner.process_message_into_response("relu", &[0u8; 2]);
        let resp = InferenceResponse::decode(&bytes).unwrap();
        assert_eq!(resp.request_id, 0);
        assert_eq!(resp.status, ResponseStatus::InvalidInput);
    }

    // ---- serve_dataflow_runner (pub/sub wiring) ----

    fn ke(s: &str) -> KeyExpr {
        KeyExpr::new(s).unwrap()
    }

    fn deliver(sub: &mut Subscriber, topic: &str, payload: Vec<u8>, ts: u64) {
        let publisher = Publisher::new(PublisherId(1), ke(topic));
        let msg = publisher.create_message(payload, ts);
        sub.deliver(msg).unwrap();
    }

    #[test]
    fn serve_runner_round_trip_on_loopback_pubsub() {
        let mut runner = DataflowRunner::new(DataflowRunnerConfig::default());
        runner.register_session(String::from("relu"), relu_session());

        let mut sub = Subscriber::new(SubscriptionId(1), ke("smallaios/inference/*/input"));
        deliver(
            &mut sub,
            "smallaios/inference/relu/input",
            relu_request().encode(),
            1,
        );

        let results = serve_dataflow_runner(&runner, &mut sub);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, "smallaios/inference/relu/output");

        let resp = InferenceResponse::decode(&results[0].1).unwrap();
        assert_eq!(resp.status, ResponseStatus::Ok);
        assert_eq!(bytes_to_f32(&resp.outputs[0].data), vec![0.0, 0.0, 3.5]);
        assert_eq!(runner.dropped_messages(), 0);
    }

    #[test]
    fn serve_runner_drops_oldest_when_queue_over_depth() {
        let mut runner = DataflowRunner::new(DataflowRunnerConfig {
            max_queue_depth: 2,
            topic_prefix: String::from("smallaios/inference"),
        });
        runner.register_session(String::from("relu"), relu_session());

        let mut sub = Subscriber::new(SubscriptionId(1), ke("smallaios/inference/*/input"));
        // Deliver 5 messages; depth is 2 so 3 oldest should be dropped.
        for i in 0..5u64 {
            let mut req = relu_request();
            req.request_id = 100 + i;
            deliver(&mut sub, "smallaios/inference/relu/input", req.encode(), i);
        }

        let results = serve_dataflow_runner(&runner, &mut sub);
        assert_eq!(results.len(), 2);
        assert_eq!(runner.dropped_messages(), 3);

        // The two most recent messages (ids 103 and 104) should survive.
        let r0 = InferenceResponse::decode(&results[0].1).unwrap();
        let r1 = InferenceResponse::decode(&results[1].1).unwrap();
        assert_eq!(r0.request_id, 103);
        assert_eq!(r1.request_id, 104);
    }

    #[test]
    fn serve_runner_dispatches_multiple_models_by_topic_name() {
        let mut runner = DataflowRunner::new(DataflowRunnerConfig::default());
        runner.register_session(String::from("relu"), relu_session());
        runner.register_session(String::from("relu2"), relu_session());

        let mut sub = Subscriber::new(SubscriptionId(1), ke("smallaios/inference/*/input"));
        deliver(
            &mut sub,
            "smallaios/inference/relu/input",
            relu_request().encode(),
            1,
        );
        deliver(
            &mut sub,
            "smallaios/inference/relu2/input",
            relu_request().encode(),
            2,
        );

        let results = serve_dataflow_runner(&runner, &mut sub);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "smallaios/inference/relu/output");
        assert_eq!(results[1].0, "smallaios/inference/relu2/output");
    }

    #[test]
    fn serve_runner_ignores_non_input_topics() {
        let runner = DataflowRunner::new(DataflowRunnerConfig::default());
        let mut sub = Subscriber::new(SubscriptionId(1), ke("smallaios/inference/**"));
        deliver(
            &mut sub,
            "smallaios/inference/relu/output",
            relu_request().encode(),
            1,
        );
        let results = serve_dataflow_runner(&runner, &mut sub);
        assert!(results.is_empty());
    }
}
