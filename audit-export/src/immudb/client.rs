// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Immudb gRPC client driver.
//!
//! Drives two unary RPCs on top of [`GrpcTransport`]:
//!
//! - `currentState` — `/immudb.schema.ImmuService/CurrentState`,
//!   empty request, returns [`schema::ImmutableState`].
//! - `verifiableSet` — `/immudb.schema.ImmuService/VerifiableSet`,
//!   carries a batch of [`schema::KeyValue`]s and the last-known
//!   `txId` for the consistency proof, returns
//!   [`schema::VerifiableTx`].
//!
//! Phase 4 wires the request / response shape and the
//! status-code → retry-class mapping. Verification of the
//! returned proof + signature is Phase 5 (`verify.rs`).
//! State persistence is delegated to [`StateStore`].

use super::schema::{ImmutableState, KeyValue, SetRequest, VerifiableSetRequest, VerifiableTx};
use super::state::{PersistedState, StateStore, StateStoreError};
use super::transport::{GrpcTransport, Header, TransportError, UnaryResponse};
use alloc::string::String;
use alloc::vec::Vec;

/// gRPC canonical status codes (RFC: see grpc.github.io).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GrpcStatus(pub u32);

impl GrpcStatus {
    pub const OK: GrpcStatus = GrpcStatus(0);
    pub const DEADLINE_EXCEEDED: GrpcStatus = GrpcStatus(4);
    pub const PERMISSION_DENIED: GrpcStatus = GrpcStatus(7);
    pub const RESOURCE_EXHAUSTED: GrpcStatus = GrpcStatus(8);
    pub const UNAVAILABLE: GrpcStatus = GrpcStatus(14);
    pub const UNAUTHENTICATED: GrpcStatus = GrpcStatus(16);

    pub fn classify(self) -> RetryClass {
        match self.0 {
            0 => RetryClass::Ok,
            4 | 8 | 14 => RetryClass::Retry,
            7 | 16 => RetryClass::HaltWithAudit,
            _ => RetryClass::HardFail,
        }
    }
}

/// How the pipeline should react to a gRPC status (D7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryClass {
    /// Status 0: success.
    Ok,
    /// Statuses 4, 8, 14: transient — back off and retry.
    Retry,
    /// Statuses 7, 16: do **not** retry; record once/minute as
    /// `audit_export_attempt` and stop attempting until the
    /// config or token changes.
    HaltWithAudit,
    /// Any other status: treat as a decoder / protocol error.
    HardFail,
}

/// Errors surfaced to the audit pipeline by the client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ClientError {
    Transport(TransportError),
    Wire(super::WireError),
    Grpc {
        status: u32,
        class: RetryClass,
        message: Option<String>,
    },
    /// State-store I/O failed before the batch could be acked.
    StatePersist(StateStoreError),
    /// `currentState` reported a `txId` strictly less than the
    /// locally stored `txId`. D6's rollback-suspected halt.
    RollbackSuspected {
        local_tx_id: u64,
        remote_tx_id: u64,
    },
}

impl From<TransportError> for ClientError {
    fn from(e: TransportError) -> Self {
        Self::Transport(e)
    }
}

impl From<super::WireError> for ClientError {
    fn from(e: super::WireError) -> Self {
        Self::Wire(e)
    }
}

impl From<StateStoreError> for ClientError {
    fn from(e: StateStoreError) -> Self {
        Self::StatePersist(e)
    }
}

/// Methods we know how to call.
pub mod method {
    pub const CURRENT_STATE: &str = "/immudb.schema.ImmuService/CurrentState";
    pub const VERIFIABLE_SET: &str = "/immudb.schema.ImmuService/VerifiableSet";
}

/// `ImmudbClient` wraps a [`GrpcTransport`] with the
/// immudb-specific request shaping and response parsing.
pub struct ImmudbClient<'a, T: GrpcTransport, S: StateStore> {
    transport: &'a mut T,
    state_store: &'a mut S,
    bearer_token: String,
    database: String,
    user_agent: String,
}

impl<'a, T: GrpcTransport, S: StateStore> ImmudbClient<'a, T, S> {
    /// Build a new client. `bearer_token` is the contents of
    /// `/data/audit_export/immudb.token` (loader enforces 0600
    /// per `audit-export-config-surface`'s rule).
    pub fn new(
        transport: &'a mut T,
        state_store: &'a mut S,
        bearer_token: String,
        database: String,
    ) -> Self {
        Self {
            transport,
            state_store,
            bearer_token,
            database,
            user_agent: build_user_agent(),
        }
    }

    /// Fetch the server's signed state. Does **not** verify the
    /// signature; the verifier in Phase 5 does that.
    pub fn current_state(&mut self) -> Result<ImmutableState, ClientError> {
        let resp = self.unary_call(method::CURRENT_STATE, &[])?;
        // gRPC frame: 5-byte prefix + body. Skip the prefix on
        // the response side (caller verified status == OK).
        let body = strip_grpc_prefix(&resp.body)?;
        Ok(ImmutableState::decode(body)?)
    }

    /// Send a batch of `KeyValue`s as a `verifiableSet` and
    /// return the server's `VerifiableTx`. Caller is responsible
    /// for invoking the verifier (Phase 5) on the response
    /// **before** asking [`Self::persist_state`] to commit the
    /// new state to disk.
    pub fn verifiable_set(
        &mut self,
        kvs: Vec<KeyValue>,
        prove_since_tx: u64,
    ) -> Result<VerifiableTx, ClientError> {
        let req = VerifiableSetRequest {
            set_request: SetRequest {
                kvs,
                no_wait: false,
            },
            prove_since_tx,
        };
        let body = req.encode();
        let mut framed = Vec::with_capacity(5 + body.len());
        framed.push(0); // compressed = false
        framed.extend_from_slice(&(body.len() as u32).to_be_bytes());
        framed.extend_from_slice(&body);
        let resp = self.unary_call(method::VERIFIABLE_SET, &framed)?;
        let inner = strip_grpc_prefix(&resp.body)?;
        Ok(VerifiableTx::decode(inner)?)
    }

    /// Compare a freshly-fetched `currentState` against the
    /// persisted state. Returns `Err(RollbackSuspected)` if the
    /// server's `txId` is strictly less than the local one
    /// (D6); returns `Ok(())` otherwise.
    pub fn check_no_rollback(&self, current: &ImmutableState) -> Result<(), ClientError> {
        if let Ok(Some(local)) = self.state_store.load() {
            if current.tx_id < local.tx_id {
                return Err(ClientError::RollbackSuspected {
                    local_tx_id: local.tx_id,
                    remote_tx_id: current.tx_id,
                });
            }
        }
        Ok(())
    }

    /// Persist a verified state. Called by the pipeline only
    /// **after** the verifier has accepted the proof / signature.
    pub fn persist_state(&mut self, state: &PersistedState) -> Result<(), ClientError> {
        self.state_store.save(state)?;
        Ok(())
    }

    /// Common request setup: HTTP/2 pseudo-headers are added by
    /// the transport layer; we attach the gRPC-required regular
    /// headers and the bearer token.
    fn unary_call(&mut self, full_method: &str, body: &[u8]) -> Result<UnaryResponse, ClientError> {
        let mut headers = alloc::vec![
            Header::new("content-type", "application/grpc+proto"),
            Header::new("te", "trailers"),
            Header::new("user-agent", self.user_agent.clone()),
        ];
        if !self.bearer_token.is_empty() {
            let mut v = String::from("Bearer ");
            v.push_str(&self.bearer_token);
            headers.push(Header::new("authorization", v));
        }
        if !self.database.is_empty() {
            headers.push(Header::new(
                "grpc-metadata-immudb-database",
                self.database.clone(),
            ));
        }
        let resp = self.transport.unary_call(full_method, &headers, body)?;
        let status = resp.grpc_status();
        let class = GrpcStatus(status).classify();
        if class != RetryClass::Ok {
            return Err(ClientError::Grpc {
                status,
                class,
                message: resp.grpc_message().map(Into::into),
            });
        }
        Ok(resp)
    }
}

fn strip_grpc_prefix(body: &[u8]) -> Result<&[u8], super::WireError> {
    if body.len() < 5 {
        return Err(super::WireError::Truncated);
    }
    if body[0] != 0 {
        // Compressed flag set; we never advertise compression.
        return Err(super::WireError::OversizedField);
    }
    let len = u32::from_be_bytes([body[1], body[2], body[3], body[4]]) as usize;
    if 5 + len > body.len() {
        return Err(super::WireError::Truncated);
    }
    Ok(&body[5..5 + len])
}

fn build_user_agent() -> String {
    let mut s = String::from("smallaios-audit-export/");
    s.push_str(crate::VERSION);
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::immudb::state::{PersistedState, StateStoreError};

    struct MockTransport {
        last_method: String,
        last_headers: Vec<Header>,
        last_body: Vec<u8>,
        canned: UnaryResponse,
        fail: Option<TransportError>,
    }

    impl MockTransport {
        fn ok(canned: UnaryResponse) -> Self {
            Self {
                last_method: String::new(),
                last_headers: Vec::new(),
                last_body: Vec::new(),
                canned,
                fail: None,
            }
        }
        fn failing(err: TransportError) -> Self {
            Self {
                last_method: String::new(),
                last_headers: Vec::new(),
                last_body: Vec::new(),
                canned: UnaryResponse::default(),
                fail: Some(err),
            }
        }
    }

    impl GrpcTransport for MockTransport {
        fn unary_call(
            &mut self,
            full_method: &str,
            headers: &[Header],
            body: &[u8],
        ) -> Result<UnaryResponse, TransportError> {
            if let Some(e) = self.fail {
                return Err(e);
            }
            self.last_method = full_method.into();
            self.last_headers = headers.to_vec();
            self.last_body = body.to_vec();
            Ok(self.canned.clone())
        }
    }

    struct MockStore {
        loaded: Option<PersistedState>,
        saved: Option<PersistedState>,
        save_fails: bool,
    }

    impl StateStore for MockStore {
        fn load(&self) -> Result<Option<PersistedState>, StateStoreError> {
            Ok(self.loaded.clone())
        }
        fn save(&mut self, state: &PersistedState) -> Result<(), StateStoreError> {
            if self.save_fails {
                return Err(StateStoreError::Io);
            }
            self.saved = Some(state.clone());
            Ok(())
        }
    }

    fn grpc_frame(body: Vec<u8>, status: u32) -> UnaryResponse {
        let mut framed = Vec::with_capacity(5 + body.len());
        framed.push(0);
        framed.extend_from_slice(&(body.len() as u32).to_be_bytes());
        framed.extend_from_slice(&body);
        UnaryResponse {
            initial_headers: alloc::vec![Header::new(":status", "200")],
            body: framed,
            trailers: alloc::vec![Header::new("grpc-status", alloc::format!("{status}"))],
        }
    }

    fn make_state_body(tx_id: u64) -> Vec<u8> {
        use crate::immudb::wire::{encode_key, encode_len_delim, encode_varint, WireType};
        let mut sig = Vec::new();
        encode_key(1, WireType::Len, &mut sig);
        encode_len_delim(b"pubkey", &mut sig);
        encode_key(2, WireType::Len, &mut sig);
        encode_len_delim(b"sigsig", &mut sig);
        let mut out = Vec::new();
        encode_key(1, WireType::Len, &mut out);
        encode_len_delim(b"smallaios_audit", &mut out);
        encode_key(2, WireType::Varint, &mut out);
        encode_varint(tx_id, &mut out);
        encode_key(3, WireType::Len, &mut out);
        encode_len_delim(b"txhash", &mut out);
        encode_key(4, WireType::Len, &mut out);
        encode_len_delim(&sig, &mut out);
        out
    }

    #[test]
    fn grpc_status_classification() {
        assert_eq!(GrpcStatus::OK.classify(), RetryClass::Ok);
        assert_eq!(GrpcStatus(4).classify(), RetryClass::Retry);
        assert_eq!(GrpcStatus(8).classify(), RetryClass::Retry);
        assert_eq!(GrpcStatus(14).classify(), RetryClass::Retry);
        assert_eq!(GrpcStatus(7).classify(), RetryClass::HaltWithAudit);
        assert_eq!(GrpcStatus(16).classify(), RetryClass::HaltWithAudit);
        assert_eq!(GrpcStatus(2).classify(), RetryClass::HardFail);
        assert_eq!(GrpcStatus(13).classify(), RetryClass::HardFail);
    }

    #[test]
    fn current_state_round_trip() {
        let body = make_state_body(42);
        let mut transport = MockTransport::ok(grpc_frame(body, 0));
        let mut store = MockStore {
            loaded: None,
            saved: None,
            save_fails: false,
        };
        let mut client = ImmudbClient::new(
            &mut transport,
            &mut store,
            "tok".into(),
            "smallaios_audit".into(),
        );
        let state = client.current_state().unwrap();
        assert_eq!(state.tx_id, 42);
        assert_eq!(state.db, "smallaios_audit");
        assert_eq!(transport.last_method, method::CURRENT_STATE);
    }

    #[test]
    fn bearer_token_injected() {
        let mut transport = MockTransport::ok(grpc_frame(make_state_body(1), 0));
        let mut store = MockStore {
            loaded: None,
            saved: None,
            save_fails: false,
        };
        let mut client = ImmudbClient::new(
            &mut transport,
            &mut store,
            "s3cret".into(),
            "smallaios_audit".into(),
        );
        let _ = client.current_state().unwrap();
        let auth = transport
            .last_headers
            .iter()
            .find(|h| h.name == "authorization")
            .unwrap();
        assert_eq!(auth.value, "Bearer s3cret");
    }

    #[test]
    fn database_metadata_attached() {
        let mut transport = MockTransport::ok(grpc_frame(make_state_body(1), 0));
        let mut store = MockStore {
            loaded: None,
            saved: None,
            save_fails: false,
        };
        let mut client = ImmudbClient::new(
            &mut transport,
            &mut store,
            String::new(),
            "smallaios_audit".into(),
        );
        let _ = client.current_state().unwrap();
        let dbh = transport
            .last_headers
            .iter()
            .find(|h| h.name == "grpc-metadata-immudb-database")
            .unwrap();
        assert_eq!(dbh.value, "smallaios_audit");
    }

    #[test]
    fn empty_token_omits_authorization() {
        let mut transport = MockTransport::ok(grpc_frame(make_state_body(1), 0));
        let mut store = MockStore {
            loaded: None,
            saved: None,
            save_fails: false,
        };
        let mut client = ImmudbClient::new(
            &mut transport,
            &mut store,
            String::new(),
            "smallaios_audit".into(),
        );
        let _ = client.current_state().unwrap();
        assert!(!transport
            .last_headers
            .iter()
            .any(|h| h.name == "authorization"));
    }

    #[test]
    fn unauthenticated_halts_with_audit() {
        let mut transport = MockTransport::ok(UnaryResponse {
            trailers: alloc::vec![
                Header::new("grpc-status", "16"),
                Header::new("grpc-message", "bad token"),
            ],
            ..UnaryResponse::default()
        });
        let mut store = MockStore {
            loaded: None,
            saved: None,
            save_fails: false,
        };
        let mut client = ImmudbClient::new(
            &mut transport,
            &mut store,
            "tok".into(),
            "smallaios_audit".into(),
        );
        match client.current_state().unwrap_err() {
            ClientError::Grpc { status, class, .. } => {
                assert_eq!(status, 16);
                assert_eq!(class, RetryClass::HaltWithAudit);
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn unavailable_classified_retry() {
        let mut transport = MockTransport::ok(UnaryResponse {
            trailers: alloc::vec![Header::new("grpc-status", "14")],
            ..UnaryResponse::default()
        });
        let mut store = MockStore {
            loaded: None,
            saved: None,
            save_fails: false,
        };
        let mut client = ImmudbClient::new(
            &mut transport,
            &mut store,
            "tok".into(),
            "smallaios_audit".into(),
        );
        match client.current_state().unwrap_err() {
            ClientError::Grpc { class, .. } => assert_eq!(class, RetryClass::Retry),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn transport_error_propagated() {
        let mut transport = MockTransport::failing(TransportError::TlsHandshake);
        let mut store = MockStore {
            loaded: None,
            saved: None,
            save_fails: false,
        };
        let mut client = ImmudbClient::new(
            &mut transport,
            &mut store,
            "tok".into(),
            "smallaios_audit".into(),
        );
        match client.current_state().unwrap_err() {
            ClientError::Transport(TransportError::TlsHandshake) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn rollback_detected_when_remote_behind() {
        let mut transport = MockTransport::ok(grpc_frame(make_state_body(50), 0));
        let mut store = MockStore {
            loaded: Some(PersistedState {
                db: "smallaios_audit".into(),
                tx_id: 100,
                tx_hash: alloc::vec![],
                public_key: alloc::vec![],
                signature: alloc::vec![],
            }),
            saved: None,
            save_fails: false,
        };
        let mut client = ImmudbClient::new(
            &mut transport,
            &mut store,
            "tok".into(),
            "smallaios_audit".into(),
        );
        let state = client.current_state().unwrap();
        let err = client.check_no_rollback(&state).unwrap_err();
        assert!(matches!(
            err,
            ClientError::RollbackSuspected {
                local_tx_id: 100,
                remote_tx_id: 50,
            }
        ));
    }

    #[test]
    fn no_rollback_when_remote_ahead() {
        let mut transport = MockTransport::ok(grpc_frame(make_state_body(200), 0));
        let mut store = MockStore {
            loaded: Some(PersistedState {
                db: "smallaios_audit".into(),
                tx_id: 100,
                tx_hash: alloc::vec![],
                public_key: alloc::vec![],
                signature: alloc::vec![],
            }),
            saved: None,
            save_fails: false,
        };
        let mut client = ImmudbClient::new(
            &mut transport,
            &mut store,
            "tok".into(),
            "smallaios_audit".into(),
        );
        let state = client.current_state().unwrap();
        client.check_no_rollback(&state).unwrap();
    }

    #[test]
    fn state_persist_io_failure_propagates() {
        let mut transport = MockTransport::ok(grpc_frame(make_state_body(1), 0));
        let mut store = MockStore {
            loaded: None,
            saved: None,
            save_fails: true,
        };
        let mut client = ImmudbClient::new(
            &mut transport,
            &mut store,
            "tok".into(),
            "smallaios_audit".into(),
        );
        let ps = PersistedState::default();
        match client.persist_state(&ps).unwrap_err() {
            ClientError::StatePersist(StateStoreError::Io) => {}
            other => panic!("unexpected: {other:?}"),
        }
    }
}
