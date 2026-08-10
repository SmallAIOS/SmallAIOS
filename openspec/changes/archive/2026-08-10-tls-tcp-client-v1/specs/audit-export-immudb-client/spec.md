## ADDED Requirements

### Requirement: TlsStreamLike trait bound by TLS 1.3 over TCP

The `TlsStreamLike` trait declared in `container::audit_export_runtime::transport` SHALL have at least one concrete implementation backed by `tls_client::std_io::TcpTlsStream`. When `[exporter] enabled = true` and `endpoint` resolves to a real TCP host, `TlsGrpcTransport::unary_call` SHALL run a full TLS 1.3 handshake (per `tls-client-handshake`) and a chain verification against the configured trust store (per `tls-client-cert-chain`) before sending the gRPC connection preface.

#### Scenario: Handshake completes before HTTP/2 preface
- **WHEN** the exporter opens a connection to `https://immudb.example.com:3322` for the first time
- **THEN** the order of bytes on the wire SHALL be: TCP SYN, TLS ClientHello, … server Finished, client Finished, HTTP/2 connection preface
- **AND** no HTTP/2 frame SHALL be sent before the TLS handshake reports `Done`

#### Scenario: Handshake failure aborts before connect retry classification
- **WHEN** the handshake returns `TlsClientError::ChainUntrusted`
- **THEN** the exporter SHALL surface `TransportError::TlsHandshake`
- **AND** the audit pipeline SHALL classify the result as `RetryClass::HardFail` (no point retrying a bad cert)
