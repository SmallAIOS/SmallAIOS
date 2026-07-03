## ADDED Requirements

### Requirement: Metric-kind metadata sidecar

Each Zenoh metric publisher SHALL declare its OTel metric kind — `counter`, `gauge`, or `histogram` — in a metadata sidecar attached to the publisher. The OTLP exporter (`telemetry-otel-exporter-metrics`) is a consumer of the `smallaios/metrics/<key>` keyspace and SHALL read the kind from the publisher metadata, so the Counter/Gauge/Histogram mapping is mechanical and requires no per-metric exporter code. Adding a new metric anywhere in the kernel SHALL consist of one publisher plus one metadata field.

#### Scenario: Existing publishers carry a declared kind

- **WHEN** the v1 publishers on `smallaios/metrics/cpu`, `smallaios/metrics/mem`, and `smallaios/metrics/inference` register
- **THEN** each SHALL expose a metadata sidecar declaring its OTel kind (e.g. `gauge` for CPU utilization, `histogram` for per-model latency)

#### Scenario: Exporter consumes the sidecar, not per-metric code

- **WHEN** the OTLP exporter enumerates metric publishers
- **THEN** it SHALL derive each metric's OTel kind solely from the publisher's metadata sidecar
- **AND** no per-metric mapping table SHALL exist in the exporter

#### Scenario: New metric needs only publisher plus metadata field

- **WHEN** a developer adds a new kernel metric by registering a publisher with a kind metadata field
- **THEN** both the Zenoh keyspace and the OTLP export path SHALL carry the new metric without further code changes
