## Context

`mgmt-audit-log` already gives SmallAIOS a tamper-evident audit
trail on-box: a SHA-3-256 hash chain, optional ML-DSA-65
signed checkpoints, periodic flush to `/data/audit/log.jsonl`,
hybrid size+age rotation with a hard failsafe, and a live
chain-head published on `smallaios/metrics/audit_fingerprint`.
`telemetry-otel-export-v1` (pending) ships those records off-
box as OTLP/Logs over HTTPS for fleet observability.

What is still missing is a **verifiable external system of
record** — an off-box log a third party can inspect, demand
inclusion proofs against, and walk for consistency
without trusting the SmallAIOS box that produced the
records. The leading off-the-shelf answer is **immudb**: a
SHA-256 Merkle-tree-backed immutable store with signed
state and well-defined inclusion + dual-consistency proofs.

The hard constraint, set by the user, is **do not run the
immudb server binary on SmallAIOS**. Immudb's server is
Go + BadgerDB-backed, ~50 MiB stripped, and would dwarf
the entire SmallAIOS image. The viable shape is therefore
an immudb **client** that runs on-box, plus immudb
**server(s)** the operator runs on their own
infrastructure.

The proposal evaluated three design families and the
maintainer accepted **Option A1**: a clean-room
`#![no_std]` immudb client speaking gRPC over our existing
HTTP/2 + TLS 1.3 + protobuf stack. This design document
captures every decision needed to go from that direction
to spec deltas, tasks, and implementation crates.

## Goals / Non-Goals

**Goals:**
- New `audit-export/` Layer 1 crate (~1,200 LOC) that
  batches audit records, ships them to a configured immudb
  endpoint as `verifiableSet` requests, parses the signed
  state response, verifies inclusion / dual-consistency
  proofs, and records both successes and failures back
  into the audit ring.
- Reuse every transport pattern already in the workspace:
  bounded in-memory ring buffer, drop-oldest on overflow,
  exponential backoff identical to the OTLP exporter,
  role-gated `0600` keyfile separate from a viewer-readable
  TOML, and the universal `ConfigSurface` trait.
- A clean-room `#![no_std]` HTTP/2 client in `net/`,
  scoped to exactly the gRPC subset we need (length-
  prefixed protobuf frames, server-streaming for
  `currentState`, client-streaming for batched `setAll`,
  unary for everything else).
- Cross-validated proof verifier: every Merkle inclusion
  and dual-consistency proof verifier path covered by
  golden fixtures emitted by a small Rust binary that
  talks to a live immudb sidecar via this same
  `audit-export` client. SmallAIOS stays Rust-only —
  no Go anywhere in the repo.
  checked into the repo so CI does not depend on a live
  immudb server.
- Strict opt-in: default off, no embedded endpoint, no
  embedded credentials, no default DNS lookups, no
  out-of-the-box traffic.
- Two integrity fingerprints visible on every box:
  the local SHA-3-256 chain head and the remote
  immudb signed state. `console-monitor-v1` displays
  both side-by-side.
- **Immudb is optional in two independent layers.**
  See D10 below.

**Non-Goals (deferred to follow-on changes):**
- On-box Merkle tree (Option B from the proposal) —
  the local audit chain stays linear SHA-3-256 for v1.
  Revisit as `verifiable-audit-log-v2`.
- Immudb SQL surface, PostgreSQL wire protocol,
  embedded BadgerDB. Pure bloat for our use case.
- Backfill of pre-existing rotated `*.jsonl.gz`
  archives. The exporter ships only records produced
  after `enabled = true`.
- Sigstore Rekor / Google Trillian / AWS QLDB sinks.
  A future generic `VerifiableLogSink` trait can be
  retrofitted; v1 hardcodes immudb to keep the
  proof-vector test surface tractable.
- A bespoke `smallaios-audit-verify` CLI. v1 documents
  `immuclient audit` (immudb's first-party verifier)
  as the canonical off-box tool.
- Mutual TLS to immudb. Static-token auth in v1; mTLS
  retrofits cleanly once `management-login-v1`'s
  client-cert path is generally available.

## Decisions

### D1. The transport is HTTP/2 + gRPC, not REST gateway

Immudb's REST gateway (`immugw`) is convenient but
deprecated as the recommended production path. Direct
gRPC against `immudb:3322` keeps SmallAIOS aligned with
the way every other immudb client speaks to the server.
Cost: a new HTTP/2 framing layer in `net/`. Mitigation:
restrict it to the gRPC subset (no server push, no stream
priority, no trailers-only optimization, fixed-size frame
defaults). The same HTTP/2 layer becomes reusable for any
future gRPC service we choose to integrate.

Open Question 1 from the proposal: **resolved A1**.

### D2. Hash algorithms stay split — SHA-3-256 local, SHA-256 remote

Both algorithms run side-by-side. The on-box chain is
unchanged (SHA-3-256, ML-DSA-65 checkpoints). The off-box
chain follows immudb's wire requirement (SHA-256, Ed25519
signed state). Records carry both digests when written
into the export pipeline:
- `local_sha3` — copied directly from the source audit
  record's existing `hash` field.
- `remote_sha256` — computed by the batcher as it
  serializes the record for the wire.

Both digests appear in the immudb value (so an auditor
walking the immudb log can later assert the local chain
head from the SHA-3 values), but the immudb tree itself
is built over SHA-256 as the protocol requires.

Open Question 2 from the proposal: **resolved option (b)**,
dual-hash per record. The ~32 B / record overhead is
trivial at 10 records/s (≈ 28 MiB / year per host).

### D3. Authentication is static token in v1, mTLS deferred

A single bearer token, written to
`/data/audit_export/immudb.token` at 0600 mode and owned
by `Role::Root`, is presented in the gRPC metadata as
`authorization: Bearer <token>`. mTLS requires a client
cert + key on disk plus a renewal path; that work belongs
in `management-login-v1` follow-on if/when client-cert
auth lands fleet-wide. The TOML config has a forward-
compatible `auth_mode = "bearer" | "mtls"` knob.

Open Question 3 from the proposal: **resolved bearer-v1,
mTLS-v2**.

### D4. Static endpoint URL, no service discovery

A single `endpoint = "https://immudb.example.com:3322"`
in the TOML. Auto-discovery via the local Zenoh ring is a
security footgun for an audit pipeline (an adversary that
can publish a Zenoh announcement could redirect the
audit stream). If multiple immudb backends are needed,
the operator configures a primary + zero or more
fallbacks; the exporter rotates round-robin on connect
failure.

Open Question 4 from the proposal: **resolved static**.

### D5. One immudb database per fleet, keys prefixed with `host_id`

The TOML carries `database = "smallaios_audit"` (default).
Per-record keys are
`audit/{host_id}/{ts_ns:020}/{seq:08x}`, where:
- `host_id` is the per-host UUID established by
  `telemetry-otel-export-v1` (do not re-roll for this
  change).
- `ts_ns` is the UNIX nanosecond timestamp from the
  source audit record, zero-padded to 20 digits so
  lexicographic ordering equals chronological ordering.
- `seq` is an 8-hex-digit monotonic counter per
  `(host_id, ts_ns)` pair to disambiguate co-timestamped
  records.

A single database keeps admin objects bounded; the
`host_id` prefix gives the operator per-host range scans
for free.

Open Question 5 from the proposal: **resolved one DB,
host-prefixed keys**.

### D6. Failsafe: drop-oldest in v1, optional disk spool in v2

When the exporter cannot reach immudb, records pile up
in the in-memory ring buffer. The buffer is bounded
(default 4 MiB, range 1 MiB – 64 MiB, configurable).
On overflow the oldest record is evicted, **never** the
audit producer's call path. This matches the OTLP
exporter's behavior and keeps the audit-write path
non-blocking under sustained outage.

A persistent on-disk spool at `/data/audit_export/
spool/*.bin` is deliberately deferred. It is the right
v2 answer for vehicles and air-gapped edge boxes that
re-uplink on a schedule; v1 keeps the durability story
simple — what's lost is lost, and operators are warned
in `console-monitor-v1`.

Open Question 6 from the proposal: **resolved
drop-oldest-v1, spool-v2**.

### D7. Verifier story off-box uses immudb's `immuclient audit`

We do not ship a SmallAIOS-branded verifier CLI in v1.
The first-party tool `immuclient audit -d
smallaios_audit` is the documented path. The CI test
suite invokes this command against a real immudb
sidecar to assert zero divergence after each
end-to-end run.

If customer feedback later demands a single-binary
verifier, the proposal-then-design path for
`smallaios-audit-verify` reuses every chunk of the
clean-room verifier already living in
`audit-export/immudb/verify.rs`.

Open Question 7 from the proposal: **resolved use
`immuclient`**.

### D8. Other immutable stores evaluated, immudb wins for v1

Briefly, for the design.md record:

| Store | Self-hostable? | Wire | Footprint | Rust SDK | Verdict |
|-------|----------------|------|-----------|----------|---------|
| immudb | Yes | gRPC | medium server, tiny client | none (we write it) | **Chosen** — covers all four requirements. |
| Sigstore Rekor | Yes (federated by design) | REST + JSON | small server | partial | Strong runner-up; v2 sink. |
| Google Trillian | Yes | gRPC | medium | none | Equally credible; immudb's tighter scope wins for audit. |
| AWS QLDB | No (AWS-managed) | Proprietary | n/a | none | Out — cloud-only contradicts on-prem story. |

The design records this trade so a future reader does
not re-litigate it from scratch.

Open Question 8 from the proposal: **resolved as above**.

### D10. Immudb is optional in two independent layers

The `audit-export` crate, the `immudb` client wire layer, and
every line of code involved in shipping records off-box can
be removed from a SmallAIOS image at **compile time** and,
separately, the runtime exporter can be turned off without
rebuilding.

**Layer 1 — Cargo feature (compile-time).** The integration
crate (`container/`) gains a non-default cargo feature
`audit-export`. When it is **not** enabled:

- `audit-export/` is not in the dependency graph of the
  binary being built. No HTTP/2 code is linked, no
  immudb protobuf code is linked, no Ed25519 verifier
  code is linked.
- `net::http2` is also unreachable (it's gated by its own
  `http2` feature, which is enabled transitively by
  `audit-export`).
- Image size delta: zero relative to a build that never
  knew the feature existed. Verified by the
  `container-size-check` job and by `tasks.md` task 1.6.

**Layer 2 — TOML runtime config.** Even on an image built
**with** the feature enabled, the default `immudb.toml`
ships `[exporter] enabled = false`. In that state:

- The exporter never registers as a tap on the audit ring
  (per `audit-export-pipeline` spec, "exporter disabled =
  zero overhead" scenario).
- No file under `/data/audit_export/` is read except for
  the existence-check on the directory itself; the
  keyfile and `last_state.bin` are not opened.
- Zero CPU, zero allocations beyond static crate layout,
  zero network sockets.

**Default posture.** Containers and unikernel images ship
with the `audit-export` cargo feature **off** by default;
operators who want the verifiable-log feature enable both
layers (`--features audit-export` at build time **and**
`enabled = true` in the TOML at runtime). This matches the
`telemetry-otel-export-v1` opt-in pattern and the
"first-party, no embedded endpoint" stance of D9.

**Operationally:**

| Build feature | TOML `enabled` | Behavior                            |
|---------------|----------------|-------------------------------------|
| off (default) | (irrelevant)   | No exporter code linked. Zero cost. |
| on            | `false`        | Code linked, exporter idle. Near-zero cost. |
| on            | `true`         | Active export to the configured endpoint. |

The CI matrix runs both `--no-default-features` (Layer 1
off) and `--features audit-export` (Layer 1 on, Layer 2
default off + Layer 2 enabled) to guarantee both paths
stay green.

### D11. Upstream proto vendoring + pin-check

The vendored `audit-export/vendor/schema.proto` reproduces
only the message subset that `audit-export/src/immudb/
schema.rs` translates. The pin file
`audit-export/vendor/IMMUDB_SCHEMA_SHA` is the single
source of truth for the upstream commit; CI verifies that
the vendored proto matches the SHA's content in
`codenotary/immudb`. The hand-written Rust translation is
the authority for what bytes go on the wire, but every
field tag in `schema.rs` must match a tag in the
vendored proto — a CI pin-check enforces this.

The v0.2.1 vendoring is pinned at immudb **v1.11.0**
(commit `f07d3ac01c068e3d6e760afaaf1f1db20b36d0bc`).

A regression test
`upstream_proto_field_tags_pinned` in
`audit-export/src/immudb/schema.rs` explicitly encodes
the upstream field-tag layout of `Signature` and
`TxEntry`, two messages whose internal layout is easy
to get wrong (the human-readable names do not match
their tag order).

### D9. This is operator data, not project telemetry

Re-affirmed for the record: audit logs are first-party
operator data. They never flow to a SmallAIOS-project-
owned endpoint. The pending `project-usage-telemetry-v1`
change is the place anonymous usage data is designed —
not here. The exporter has no default endpoint, no
embedded credentials, and refuses to start without an
operator-provided endpoint and token.

Open Question 9 from the proposal: **resolved**.

## Architecture

```
                ┌──────────────────────────┐
                │  mgmt-audit-log ring     │
                │  (in-memory, 16 MiB)     │
                └────────────┬─────────────┘
                             │  fan-out
              ┌──────────────┼────────────────┐
              ▼              ▼                ▼
        /data/audit/   OTLP exporter     audit-export
        log.jsonl      (telemetry/)      (this change)
                                                │
                                       ┌────────┴────────┐
                                       │ batcher         │  100 records
                                       │ (1 s | 100 rec) │  or 1 s
                                       └────────┬────────┘
                                                │
                                       ┌────────┴────────┐
                                       │ ring buffer     │  drop-oldest
                                       │ (4 MiB default) │  never blocks
                                       └────────┬────────┘
                                                │
                                       ┌────────┴────────┐
                                       │ immudb client   │
                                       │ gRPC / HTTP2    │
                                       │ TLS 1.3 + PQC   │
                                       └────────┬────────┘
                                                ▼
                                         immudb server
                                         (off-box)
                                                │ signed state
                                                │ + proofs
                                                ▼
                                       ┌─────────────────┐
                                       │ verifier        │  inclusion
                                       │ (on-box)        │  + consistency
                                       └────────┬────────┘
                                                │
                                                ▼
                                       audit ring re-entry
                                       action="immudb_state"
                                       or "audit_export_attempt"
```

### Crate layout

```
audit-export/
├── Cargo.toml          # depends on: net, security (Ed25519+SHA-256), mgmt
├── src/
│   ├── lib.rs          # public API surface
│   ├── pipeline.rs     # batcher + ring buffer + retry loop
│   ├── config.rs       # TOML schema, ConfigSurface impl
│   ├── cli.rs          # `audit-export status|config` shell commands
│   └── immudb/
│       ├── mod.rs
│       ├── schema.rs   # clean-room protobuf encoder/decoder
│       ├── client.rs   # gRPC over HTTP/2 client
│       ├── verify.rs   # inclusion + dual consistency + signed state
│       └── keys.rs     # host_id / ts_ns / seq key formatting
└── tests/
    ├── proof_vectors/  # checked-in fixtures (binary blobs)
    └── e2e_immudb.rs   # docker-sidecar end-to-end test
```

### `net/` additions

```
net/
└── src/
    └── http2/          # new module, ~600 LOC
        ├── mod.rs
        ├── framing.rs  # HPACK-lite + frame layout
        ├── client.rs   # connection pool, stream multiplexing
        └── grpc.rs     # length-prefixed protobuf framing
```

Only the gRPC subset is implemented. Server push, stream
priority, trailers-only, and dynamic-table HPACK are out;
static-table HPACK is enough for the headers immudb
requires (`:method POST`, `:path /immudb.schema.ImmuService/...`,
`content-type application/grpc+proto`, `te trailers`,
`authorization Bearer ...`).

### Immudb protobuf subset

We hand-write `audit-export/src/immudb/schema.rs` for
exactly these messages (field tags pinned to immudb 1.9.x):

- `KeyValue { key, value, metadata }`
- `SetRequest { kvs[], preconditions[] }`
- `VerifiableSetRequest { setRequest, proveSinceTx }`
- `TxHeader { id, prevAlh, ts, eh, blTxId, blRoot, ... }`
- `TxEntry { key, kvDigest, hValue, ... }`
- `LinearProof { sourceTxId, targetTxId, terms[] }`
- `InclusionProof { leaf, width, terms[] }`
- `DualProof { sourceTxHeader, targetTxHeader,
              inclusionProof, consistencyProof,
              targetBlTxAlh, lastInclusionProof,
              linearProof, linearAdvanceProof }`
- `ImmutableState { db, txId, txHash, signature }`
- `VerifiableTx { tx, dualProof, signature }`

Tag values are documented in immudb's `schema.proto`
([repo link](https://github.com/codenotary/immudb/tree/master/pkg/api/schema)).
We pin to a specific commit SHA so unexpected
field-tag drift is a CI failure, not a silent serialization
break.

## Wire protocol decisions

### gRPC framing rules

- Unary RPC: client sends one HEADERS frame + one DATA
  frame (length-prefixed protobuf) + END_STREAM. Server
  replies with one HEADERS + one DATA + one trailing
  HEADERS (`grpc-status`). No server streaming for v1
  except `currentState`.
- `currentState` is server-streamed in immudb 1.9 for
  long-lived clients but the unary variant suffices for
  v1; we re-call it on every batch's first request.
- `verifiableSetAll` is the only RPC we issue for write
  paths. Each invocation receives back a `VerifiableTx`
  containing the new signed state and the dual proof
  from the previous-known state to the new one.
- The exporter holds the last-known state in memory and
  on disk (`/data/audit_export/last_state.bin`) so a
  cold start can request a consistency proof from the
  last-shipped state forward.

### TLS rules

- TLS 1.3 mandatory. TLS 1.2 and below rejected at
  handshake.
- PQC hybrid (`X25519+ML-KEM-768`) available; offered as
  the first ciphersuite. Pure-classical
  ciphersuites fall back at the operator's existing
  TLS-policy knob.
- HTTP/1.x / h2c / cleartext gRPC all refused.

### Error handling

- gRPC status `OK (0)` → record success in the audit
  ring (`action = "audit_export_attempt", code = 0`).
- `UNAVAILABLE (14)`, `DEADLINE_EXCEEDED (4)`,
  `RESOURCE_EXHAUSTED (8)` → backoff, retry.
- `UNAUTHENTICATED (16)`, `PERMISSION_DENIED (7)` →
  do **not** retry; record once per minute as
  `audit_export_attempt` with the gRPC code in the
  `code` field; raise on `console-monitor-v1`.
- Any 4xx HTTP status → treat as `UNAUTHENTICATED`.
- Verifier failure (proof did not validate) → record
  `audit_export_proof_failure` and stop the exporter
  until the operator acks. This is the loudest
  failure mode because it implies either bug or
  tampering.

## Verifier design

The verifier mirrors what a correct immudb client does on
`verifiedSet`. Steps:

1. Decode `VerifiableTx` from the gRPC reply.
2. Recompute the target `TxHeader.eh` from the
   transaction's entries (SHA-256 over `kvDigest`s).
3. Recompute `Alh` over `(id, ts, prevAlh, eh)`.
4. Walk the inclusion proof — for each `term` in
   `inclusionProof.terms`, hash the running node up
   the tree, applying left/right at the bit indicated
   by `leaf` and `width`.
5. Verify the dual / linear proof from the last-known
   state to the new state — bridging the Merkle and
   linear sections of immudb's commit log.
6. Verify the Ed25519 signature over
   `(db, txId, txHash)` against the pinned server
   public key. The key fingerprint is configured in
   the TOML, **not** TOFU-trusted.

Each step has a checked-in fixture set:
- `proof_vectors/inclusion_v1_*.bin` — N=20 transactions
  inserting 1..1024 keys each, varying tree depths.
- `proof_vectors/dual_v1_*.bin` — N=10 jumps over
  arbitrary `sourceTxId → targetTxId` gaps.
- `proof_vectors/state_v1_*.bin` — N=5 Ed25519 signed
  states with known good and known tampered variants.

Fixtures are produced by a deterministic harness in
`audit-export/tests/scripts/gen_fixtures.rs` — a Rust
binary built from the workspace, never linked into any
SmallAIOS image. It uses the same `audit-export::immudb`
client this crate ships, points at a developer-controlled
immudb sidecar, and writes the resulting `VerifiableTx`
bytes to `audit-export/tests/proof_vectors/*.bin`. The
binary blobs (not the generator) are what the per-PR
fixture-replay tests consume. SmallAIOS itself is
Rust-only; nothing in the repo or the production image
links a Go toolchain.

## Configuration

`/data/audit_export/immudb.toml`:

```toml
[exporter]
enabled        = false                       # default off
endpoint       = ""                          # required if enabled
fallback_endpoints = []                      # optional list
auth_mode      = "bearer"                    # "bearer" | (v2) "mtls"
token_path     = "/data/audit_export/immudb.token"
state_path     = "/data/audit_export/last_state.bin"
database       = "smallaios_audit"
batch_size     = 100                         # records
batch_interval_ms = 1000
buffer_bytes   = 4194304                     # 4 MiB
backoff_initial_ms = 10000
backoff_cap_ms     = 300000

[tls]
require_pqc    = false                       # opt-in PQC hybrid
server_pubkey_fingerprint = ""               # SHA-256 of immudb's Ed25519 pubkey, hex

[record_filter]
include_actions = []                         # empty = include all
exclude_actions = ["telemetry_export_attempt", "audit_export_attempt"]
```

The `exclude_actions` default breaks the
exporter-self-feedback loop flagged in
`telemetry-otel-export-v1`'s Open Question 6 — we do
not ship records *about* the export itself to the
remote log. Local `mgmt-audit-log` still records them.

Loader rules inherit from `mgmt-config-layout`:
- TOML readable by `Role::Viewer` and above.
- Keyfile readable only by `Role::Root`; mode laxer
  than 0600 is a boot-time fatal.
- All writes go through `ConfigSurface::write` with
  atomic-rewrite + audit entry.

## Resource budget

| Resource | Budget |
|----------|--------|
| Code | ~1,200 LOC in `audit-export/` + ~600 LOC in `net/http2/` |
| Live memory (default config) | ~5 MiB (4 MiB buffer + 1 MiB working) |
| Boot time | <50 ms additional when enabled (one TLS handshake; lazy until first batch) |
| Steady-state CPU | <0.1 % of one core at 10 rec/s |
| Network out | ~17 MiB / day per host at 10 rec/s and default batch policy |
| On-disk | `last_state.bin` <1 KiB; no other persistent state |
| Test count delta | 4,143 → ≥4,310 |

## Failure modes

| Failure | Detection | Response |
|---------|-----------|----------|
| Endpoint unreachable (TCP / TLS) | connect timeout / TLS error | exponential backoff, buffer fills, drop-oldest at cap |
| `UNAUTHENTICATED` | gRPC status 16 | stop retrying, raise in monitor, audit once/min |
| `PERMISSION_DENIED` | gRPC status 7 | same |
| Schema drift (unknown protobuf tag) | decode error | record `audit_export_decode_failure`, stop exporter |
| Server signature mismatch | Ed25519 verify fail | record `audit_export_proof_failure`, stop exporter, **loud** |
| Inclusion proof fails | verifier returns false | same |
| Consistency proof fails | verifier returns false | same |
| Cold start, no `last_state.bin` | file missing | request fresh `currentState`, log as `audit_export_state_init` |
| Cold start, stale `last_state.bin` | server `txId` < stored `txId` | record `audit_export_rollback_suspected`, stop exporter, require operator ack |

The "rollback suspected" case is the only one where the
exporter refuses to recover automatically — a remote
log that goes backwards is a strong tampering signal
and should never be papered over.

## Test strategy

1. **Unit** — protobuf encode/decode round-trips for
   every message, key formatter property tests, ring
   buffer drop-oldest under pressure, exponential
   backoff curve matches OTLP exporter, keyfile mode
   rejection, TOML schema validation.
2. **Vector-replay** — verifier validates every fixture
   in `proof_vectors/`; tampered fixtures fail with the
   right error variant.
3. **Fuzz** — `cargo-fuzz` target on the protobuf
   decoder (the largest attacker-controlled surface).
4. **End-to-end** — CI runs immudb 1.9.x as a Docker
   sidecar, exporter writes 10 K records over 5
   minutes, then `immuclient audit -d smallaios_audit`
   asserts zero divergence; a second pass tampers with
   one record and asserts the verifier surfaces the
   right error.
5. **Formal** — TLA+ model `audit_export.tla` covers
   the producer / batcher / exporter handoff invariants
   (producer never blocks, no record produced before
   `enabled = true` is shipped, no record is shipped
   twice).

## Open questions resolved

All nine open questions from the proposal are
resolved in the **Decisions** section above. Any
question that needs maintainer attention before
implementation begins is flagged inline; none are
left open.

## What this design does not commit to

- The exact field-tag values for immudb's schema —
  these are read from a pinned-commit copy of
  `schema.proto` in `audit-export/vendor/`; the
  pinned SHA is part of the implementation tasks.
- The Ed25519 server public key for any specific
  deployment — operators provide their own.
- Whether the Docker-sidecar end-to-end test runs on
  every PR or nightly only — CI cost vs coverage
  decision is in `tasks.md`.
