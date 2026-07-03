## ADDED Requirements

### Requirement: Clean-Room `no_std` OTLP Metrics Encoder

A new `telemetry/` crate (Layer 1, depending on `mgmt`, `net`, and the existing protobuf encoder) SHALL provide a hand-written `#![no_std]` protobuf encoder in `telemetry/src/otlp/proto.rs` for the OTLP metrics signal, covering `opentelemetry.proto.collector.metrics.v1.ExportMetricsServiceRequest` and the `opentelemetry.proto.metrics.v1` message set (`ResourceMetrics`, `ScopeMetrics`, `Metric`, and their nested data-point messages). The encoder SHALL follow the same hand-rolled pattern as the existing protobuf parser. The crate SHALL NOT use `prost`, .proto code generation, `opentelemetry-rust`, `tokio`, `tonic`, or `hyper`.

#### Scenario: Golden vectors decode under a reference decoder

- **WHEN** the encoder serializes a golden-vector `ExportMetricsServiceRequest` fixture
- **THEN** decoding the bytes with `protoc --decode` (cross-checked against the `opentelemetry-collector-contrib` reference decoder on the developer workstation) SHALL reproduce the fixture's field values exactly

#### Scenario: No std-only dependencies in the telemetry crate

- **WHEN** the `telemetry` crate's dependency graph is inspected on a workspace build
- **THEN** it SHALL contain no `prost`, `tokio`, `tonic`, `hyper`, or `opentelemetry` crates
- **AND** the crate SHALL build for the bare-metal `#![no_std]` targets

#### Scenario: Layer placement holds

- **WHEN** `just arch-check` runs after the crate is added
- **THEN** `telemetry` SHALL sit at Layer 1 depending only on `mgmt`, `net`, and lower layers
- **AND** no dependency cycle SHALL be introduced

### Requirement: Counter, Gauge, and Histogram Mapping From Publisher Metadata

The exporter SHALL support exactly three OTel metric kinds in v1 and SHALL map each Zenoh metric publisher to its kind mechanically, by reading the publisher's metric-kind metadata sidecar — no per-metric exporter code SHALL exist:

- **Counter**: monotonic cumulative value (e.g. total inferences served, total bytes RX'd on `eth0`, total Argon2id verifications), encoded as a cumulative monotonic Sum.
- **Gauge**: current snapshot (e.g. CPU utilization, free memory, GPU utilization, active session count).
- **Histogram**: explicit-bounds latency distributions (the source data from which `console-monitor-v1`'s p50/p99 derive).

#### Scenario: Counter encodes as monotonic cumulative

- **WHEN** a publisher whose sidecar declares kind `counter` reports total inferences served across two consecutive pushes
- **THEN** both exported data points SHALL be encoded as a cumulative, monotonic Sum
- **AND** the second value SHALL be greater than or equal to the first

#### Scenario: Gauge encodes the current snapshot

- **WHEN** a publisher whose sidecar declares kind `gauge` reports CPU utilization
- **THEN** each exported data point SHALL carry the latest snapshot value at its export timestamp, not an accumulation

#### Scenario: Histogram encodes explicit bounds

- **WHEN** a publisher whose sidecar declares kind `histogram` reports a per-model latency distribution
- **THEN** the exported metric SHALL be an explicit-bounds Histogram whose bucket bounds and counts reproduce the source distribution
- **AND** p50/p99 computed from the exported buckets SHALL agree with the on-box values within bucket resolution

### Requirement: New Zenoh Metric Publishers Export Automatically

Adding a new metric anywhere in the kernel SHALL require only a Zenoh publisher plus one metric-kind metadata field. The OTLP path SHALL pick the new metric up automatically; no change to the `telemetry` crate SHALL be needed.

#### Scenario: New publisher appears in the next push

- **WHEN** a new publisher is registered on `smallaios/metrics/<new-key>` with a metric-kind metadata sidecar declaring `gauge`
- **AND** no code in the `telemetry` crate is modified
- **THEN** the next OTLP push SHALL include the new metric with the declared kind and the standard resource attributes

### Requirement: OTLP/HTTP Transport Over Mandatory TLS 1.3

All OTLP export requests (metrics and logs signals) SHALL be transmitted as HTTP/1.1 `POST` requests with content type `application/x-protobuf`, over TLS 1.3 using the existing TLS stack (PQC-hybrid available, not required by the backend). There SHALL be no default endpoint and no embedded credentials: the operator MUST configure both. Authentication SHALL follow the configured `auth_mode`: `basic` sends `Authorization: Basic <base64(instance_id:api_key)>` (Grafana Cloud's scheme); `bearer` sends `Authorization: Bearer <api_key>`.

#### Scenario: Basic auth header constructed from keyfile

- **WHEN** `auth_mode = "basic"` and the keyfile holds the API key
- **THEN** each export request SHALL carry `Authorization: Basic <base64(instance_id:api_key)>`
- **AND** the request SHALL use `Content-Type: application/x-protobuf` over HTTP/1.1

#### Scenario: Bearer mode supported for non-Grafana backends

- **WHEN** `auth_mode = "bearer"` is configured
- **THEN** each export request SHALL carry `Authorization: Bearer <api_key>` instead of the Basic form

#### Scenario: TLS 1.3 on the export path

- **WHEN** the exporter connects to the configured endpoint
- **THEN** the connection SHALL negotiate TLS 1.3 via the existing TLS stack before any OTLP bytes are sent

#### Scenario: No default endpoint is ever contacted

- **WHEN** no endpoint has been configured by the operator
- **THEN** the exporter SHALL emit no network traffic
- **AND** no built-in default endpoint SHALL exist anywhere in the `telemetry` crate
