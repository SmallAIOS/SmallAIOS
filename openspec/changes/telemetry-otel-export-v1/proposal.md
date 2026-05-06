## Why

`console-monitor-v1` gives an operator at the box a live look
at what it is doing. `mgmt-zenoh-telemetry` gives an operator
on the same network a structured pub/sub stream of the same
data. Neither answers questions like:

- "What did the GPU utilization on `orin-01` look like *last
  Tuesday* between 14:00 and 16:00, and how did it compare
  to `x86-bench-02` over the same window?"
- "Which model has the highest p99 latency variance across all
  three of my test boxes, averaged over the last 14 days?"
- "Which of my SmallAIOS units have not phoned home in the
  last 5 minutes — i.e. is something quiet because the
  inference load is low or because the box is wedged?"

Those are off-box, historical, multi-host fleet questions, and
the conventional answer is OpenTelemetry: stream metrics (and
later traces and logs) over OTLP to a backend that handles
ingest, storage, query, and dashboards. For a single-developer
prototype phase the cheap landing pad is **Grafana Cloud's free
tier** (10k active series, ~14-day retention, OTLP ingest over
HTTPS). For an enterprise customer the same wire protocol works
against Mimir / Prometheus / Tempo / Loki / Datadog / any
OTLP-compatible backend — switching is a config change, not a
code change.

A multi-host concern matters from day one: the user already
runs SmallAIOS on at least two distinct test boxes (Jetson Orin
and an x86 reference machine), so every metric must arrive in
the backend tagged with stable, automatically-derived per-host
identity. OTel's Resource Attribute model is precisely designed
for this — `service.instance.id`, `host.id`, `host.name`,
`host.arch` — and getting them right at proposal time avoids
"all my boxes are reporting as the same instance" noise later.

The hard architectural constraint is the same one we have hit
on every protocol: the mainstream `opentelemetry-rust` SDK is
std-only, drags in `tokio` + `tonic` + `hyper` + `prost`, and
assumes a hosted OS. We **cannot** use it. We can — and per
CLAUDE.md already do — ship a clean-room `#![no_std]`
protobuf encoder, QUIC + HTTP/3, TLS 1.3 + PQC. The OTel wire
protocol (OTLP/HTTP) is just protobuf-over-HTTP-POST. So the
exporter is "encode the metrics we already collect into OTLP
protobuf, POST them over the existing TLS stack on a
configurable interval." Real work, but no new mountain.

This change deliberately stays **first-party**: the operator
configures their own OTLP endpoint and their own credentials.
**No default endpoint, no embedded credentials.** A separate
proposal — `project-usage-telemetry-v1` — designs the
"telemetry the SmallAIOS *project* collects from users"
problem, which has different threat model, different consent
requirements, and different code paths.

## What Changes

### Resource attribute model (per-host identity)

Every exported metric / log carries an OTel Resource block
with the standard attributes:

| Attribute | Source | Example |
|-----------|--------|---------|
| `service.name` | constant | `smallaios` |
| `service.namespace` | constant | `smallaios` |
| `service.version` | build-time | `0.2.3+abcd123` |
| `service.instance.id` | per-host UUID, generated on first boot, persisted in `system.toml` | `f47ac10b-58cc-4372-a567-0e02b2c3d479` |
| `host.id` | same UUID (or a separately-derived stable ID; see open question) | same |
| `host.name` | hostname from `system.toml` | `orin-01` |
| `host.arch` | build-time arch triple component | `aarch64`, `x86_64`, `riscv64` |
| `os.type` | constant | `smallaios` |
| `os.version` | build-time | `0.2.3` |
| `deployment.environment` | operator-configured | `dev`, `staging`, `prod` |

Plus an arbitrary `labels` map from the operator's config —
useful for `rack=A4`, `customer=acme`, `experiment=batched-v2`,
etc. The labels map is the only source of free-form metadata;
it is never derived from inference content.

The per-host UUID is generated **once**, on first boot before
the first export, written atomically to `/data/system.toml`,
and never regenerated except by an explicit `auth-create-user`-
gated reset. A bricked unit's identity should not change just
because an operator re-flashed the OS, so the UUID lives in
`/data/`, not in the image.

### OTLP/HTTP exporter (clean-room)

- New `telemetry/` crate (Layer 1) that depends on `mgmt`,
  `net`, and the existing protobuf encoder.
- `telemetry/src/otlp/proto.rs` — clean-room `no_std` Rust
  encoder for the four protobuf messages we need:
  - `opentelemetry.proto.collector.metrics.v1.ExportMetricsServiceRequest`
  - `opentelemetry.proto.metrics.v1.{ResourceMetrics, ScopeMetrics, Metric, ...}`
  - `opentelemetry.proto.collector.logs.v1.ExportLogsServiceRequest`
  - `opentelemetry.proto.logs.v1.{ResourceLogs, ScopeLogs, LogRecord}`
- The protobuf field tags / wire types are stable and
  documented in the .proto files — we do not generate code
  from .proto (no `prost`); we hand-write the encoder
  following the same pattern as our existing protobuf parser.
- ~600 LOC for both metric and log message types.

### Metric data model

Three OTel metric kinds suffice for everything
`mgmt-zenoh-telemetry` already publishes:

- **Counter**: monotonic cumulative value (e.g. total
  inferences served, total bytes RX'd on `eth0`, total
  Argon2id verifications).
- **Gauge**: current snapshot (e.g. CPU utilization, free
  memory, GPU utilization, active session count).
- **Histogram** (explicit-bounds): latency distributions
  (p50/p99 in `console-monitor-v1` derive from these).

Mapping is mechanical: each Zenoh metric publisher declares
its OTel kind in a metadata sidecar, the exporter reads the
kind from the publisher metadata, no per-metric exporter
code. Adding a new metric anywhere in the kernel = one
publisher + one metadata field; the OTLP path picks it up
automatically.

### Logs export (in v1 scope)

- OTLP/Logs encodes the same audit-ring records the Zenoh
  `smallaios/metrics/log` keyspace publishes today.
- Body, severity, attributes, trace context (empty for v1
  since traces are deferred), timestamp.
- The audit ring is the *only* log source for v1 — no
  arbitrary application stdout shipping. This keeps the
  exported volume bounded and avoids the "free text gets
  PII" failure mode.

### Transport: OTLP/HTTP + protobuf over TLS

- Default content type: `application/x-protobuf`.
- Default endpoint: **none.** The operator must configure one.
- TLS 1.3 mandatory (and PQC-hybrid available since the
  whole stack supports it; Grafana Cloud does not require
  PQC but our stack offers it).
- HTTP basic auth: `Authorization: Basic <base64(instance_id:api_key)>` —
  Grafana Cloud's standard scheme. Other backends may use
  `Authorization: Bearer ...`, supported via a `auth_mode`
  config knob.
- HTTP/1.1 wire format for v1 (small, debuggable). HTTP/3
  upgrade is a v2 nice-to-have once the path is exercised.

### Push interval and buffering

- Default push interval: **10 seconds**. Range: 1 s – 10 min.
  Configurable per-deployment.
- **In-memory ring buffer** for unsent records: bounded
  (default 1 MiB, configurable), drop-oldest on overflow.
- **Exponential backoff** on transport failure: 10 s → 20 s
  → 40 s → … → cap at 5 min. Buffer keeps filling during
  backoff; oldest records are evicted first.
- **Persistent on-disk WAL is out of scope for v1** —
  prototype phase, occasional gaps acceptable. Flag for v2.

### Configuration: `telemetry/otel.toml`

```toml
[exporter]
enabled       = false               # default false; explicit opt-in
endpoint      = ""                  # required when enabled
auth_mode     = "basic"             # "basic" | "bearer"
api_key_path  = "/data/telemetry/otel.key"  # 0600, root-only
push_interval_seconds = 10
buffer_bytes  = 1048576

[exporter.signals]
metrics = true
logs    = true
traces  = false                     # v2

[resource]
deployment_environment = "dev"
labels = { rack = "lab-3", experiment = "baseline" }
```

The `api_key_path` rule (separate file at 0600) keeps the
key out of `otel.toml` itself — `otel.toml` is viewer-readable
(operators may need to inspect endpoint config) but the key
file is root-only. Loader rule from `mgmt-config-layout`
enforces both modes. The setup script's first action on a
new box is to prompt for the key and write it to the keyfile;
the proposal explicitly forbids checking either file into git.

### Role gate

- `Role::Root` may write `telemetry/otel.toml` and the keyfile.
- `Role::Operator` may **read** `otel.toml` (not the keyfile),
  may not modify exporter state.
- `Role::Viewer` may read `otel.toml`, may not modify, may not
  read the keyfile.
- Surfaces: TTY shell `telemetry status` (any role; shows
  endpoint + last push success), `telemetry config` (root
  only). Zenoh `smallaios/admin/telemetry/**` keyspace
  mirrors the same.

### Out of scope for v1 (flagged)

- **Traces.** Span model + cross-IPC context propagation +
  exemplar wiring is substantial (~800 extra LOC); v2.
- **OTLP/gRPC.** Needs a clean-room HTTP/2 + gRPC client.
  Most backends accept OTLP/HTTP equally, so the gRPC path
  is purely an optimization. v2.
- **Persistent on-disk WAL** for offline buffering. Useful
  for vehicle / disconnected-edge deployments; v2.
- **Per-metric filter / sampling rules** beyond on/off
  per-signal. v2.
- **mTLS** to backend. Available transitively (TLS supports
  it) but no dedicated config knob in v1; backends typically
  use bearer tokens or basic auth.
- **Traces' built-in correlation with Zenoh request IDs.**
  Requires the trace context, deferred with traces.
- **Exemplars** linking histogram samples to trace IDs.
  Deferred with traces.
- **Spec-mandated Prometheus pull endpoint** (`/metrics`
  scrape). Push-only in v1; pull is a sibling change
  (`telemetry-prometheus-pull-v1`) if ever needed.
- **`project-usage-telemetry-v1`** — the deliberately
  separate "telemetry the project collects from users"
  proposal. This change is strictly first-party.

## Capabilities

### New Capabilities

- `telemetry-otel-resource`: the per-host UUID generation +
  persistence rule, the resource-attribute schema, the
  build-time vs runtime attribute taxonomy, and the rule
  that resource attributes are populated automatically (no
  per-export code).
- `telemetry-otel-exporter-metrics`: clean-room OTLP/HTTP
  protobuf encoder for the metrics signal, the
  Counter/Gauge/Histogram mapping rules, and the contract
  that adding a Zenoh metric publisher automatically
  exports it.
- `telemetry-otel-exporter-logs`: the same for the logs
  signal, sourced exclusively from the audit ring.
- `telemetry-otel-buffering`: ring-buffer size, drop-oldest
  semantics, exponential-backoff retry curve, and the rule
  that overflow does not block the producer.
- `telemetry-otel-config-surface`: the `telemetry/otel.toml`
  schema, the separate keyfile rule (0600), the role gate,
  and the inheritance from `mgmt-config-layout`.

### Modified Capabilities

- `mgmt-zenoh-telemetry`: documents the OTLP exporter as a
  consumer; adds the metric-kind metadata sidecar required
  for Counter/Gauge/Histogram mapping.
- `mgmt-config-layout`: adds the `/data/telemetry/`
  directory, the keyfile permission rule (0600 + loader
  rejects laxer modes), and the secret-redaction rule for
  audit records that mention the keyfile.
- `mgmt-audit-log`: adds `telemetry_export_attempt`
  records (success / failure / dropped-from-buffer) — these
  are themselves exported, so the audit trail of "did
  telemetry work" is visible in the backend.
- `console-monitor-v1`: adds an optional one-line status
  in the header — `OTLP last 4s ok / 12s err` — visible to
  any role (read-only). Triggered only when
  `[exporter] enabled = true`.

## Impact

- **Code:**
  - New `telemetry/` crate (Layer 1, ~1500 LOC total).
    - `otlp/proto.rs` — protobuf encoder (~600 LOC).
    - `otlp/exporter.rs` — push loop, buffer, retry (~300).
    - `resource.rs` — attribute population (~150).
    - `metrics.rs` — Zenoh-to-OTLP translator (~250).
    - `logs.rs` — audit-ring-to-OTLP translator (~200).
  - `container/src/bin/telemetry.rs` — `telemetry status`
    / `telemetry config` user-space command.
  - Small additions to `mgmt-zenoh-telemetry` publishers
    for the metric-kind metadata sidecar.
- **Tests:** ~80 new tests targeted: protobuf encoder
  golden vectors (cross-checked against `protoc` decode +
  `opentelemetry-collector-contrib`'s reference decoder
  on the developer workstation), Counter monotonicity,
  Gauge snapshot semantics, Histogram bucket math, ring-
  buffer overflow drop-oldest, exponential-backoff curve,
  resource-attribute presence on every export, keyfile
  permission rejection, audit-record round-trip through
  OTLP/Logs. End-to-end test pushes against a mock OTLP
  server and replays the recorded traffic. Aim 4,143 →
  ≥4,310 once `management-login-v1` lands.
- **Boot footprint:** ~80 KB code, ~1 MiB live (configurable
  buffer). Zero CPU when `enabled = false`.
- **Container image:** unchanged.
- **Network:** at the default 10 s push interval with the
  `console-monitor-v1` data set, ~3 KB / push compressed,
  ~25 KB / minute, ~36 MB / day per host. Comfortably
  under any free-tier ingest quota for one-to-a-handful of
  hosts.
- **Downstream:** unblocks fleet-level historical
  monitoring across multiple SmallAIOS test boxes; sets
  up the integration story for enterprise customers with
  their own Mimir / Prometheus / Datadog stack; is the
  foundation `project-usage-telemetry-v1` will reuse for
  the on-box anonymizer plumbing (though not the exporter
  itself).
- **Dependencies:** `management-login-v1` — provides auth,
  roles, the Zenoh telemetry pipeline this exporter
  consumes, the audit ring this exporter consumes for
  logs, and the management surface convention (Config +
  ConfigSurface + atomic-rewrite). This change adds
  `telemetry/otel.toml` to `Config` and reuses the
  existing TOML / TTY / Zenoh / (future) UDS surfaces —
  no new transport plumbing.
- **Risks:**
  (1) Free-tier quota exhaustion on Grafana Cloud (10k
  active series). The cardinality of `host.id × metric_name
  × labels` must stay bounded — flag in the proposal that
  user-supplied labels are *attributes*, not part of the
  metric name, so they do not multiply series. Documented
  with a worked example.
  (2) Clock skew. OTLP timestamps are UTC nanoseconds;
  unsynchronized boxes will scatter on the time axis.
  Recommend NTP / PTP at deployment; do not assume.
  (3) Resource-attribute drift between the OTel spec and
  our implementation as the spec evolves. Pin to a specific
  spec version (currently OTel v1.32.0) and document the
  upgrade path.
  (4) The keyfile is the only secret in v1 — reviewer
  attention to its permission, audit-record redaction, and
  scrubbing-on-export rules is warranted.

## Open Questions

1. **`host.id` strategy**: is it (a) the same UUID as
   `service.instance.id` (simpler, one ID), or (b) derived
   from a hardware identifier (MAC of the lowest-numbered
   NIC, machine-id from EFI, etc., so that re-flashing
   `/data/` does not change `host.id`)? Leaning (a) for v1
   simplicity; (b) is more correct in the long run.
2. **Push interval default**: 10 s is OTel-standard, but
   on a slow link (deployed vehicle ECU on a cellular
   modem) 10 s × 3 KB = 1 MB/hour just for telemetry.
   Should the default scale with the detected link speed?
   Probably not for v1; document the trade.
3. **Endpoint validation**: should the exporter reject
   `http://` (non-TLS) endpoints, or just warn? Leaning
   reject for the production / customer path; warn-only
   for `dev` / `staging` `deployment.environment` since
   developers may run a local OTel collector in the clear.
4. **Keyfile location**: `/data/telemetry/otel.key` matches
   our convention, but Grafana Cloud documentation will
   say "put your API key in the env var
   `GRAFANA_CLOUD_API_KEY`" — should we read from env vars
   too? Leaning no (env vars are not durable across the
   unikernel boot model); documented as a deliberate
   deviation.
5. **Per-signal granularity** in `[exporter.signals]`:
   v1 is a per-signal on/off boolean. A future need is
   "metrics on, but only the network and GPU subsets" —
   add a `filter` array per signal? Leaning defer to v2;
   v1 is on/off.
6. **Audit-export feedback loop**: the exporter writes
   `telemetry_export_attempt` audit records, and those
   records get exported. A failure can recursively log a
   failure-of-failure-export. Cap recursion at one level —
   `telemetry_export_attempt` records about
   `telemetry_export_attempt` records do **not** themselves
   produce audit entries. Flagged for design.md.
