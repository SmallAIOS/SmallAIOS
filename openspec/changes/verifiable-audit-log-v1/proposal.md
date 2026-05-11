## Why

SmallAIOS already has more tamper-evident logging than most
embedded OSes ever ship: `mgmt-audit-log` keeps every audit
record in a SHA-3-256 hash chain, optionally stamps an
ML-DSA-65 signature every N records, flushes to
`/data/audit/log.jsonl` on a fixed cadence, rotates with a
hard failsafe, and publishes the chain head live on
`smallaios/metrics/audit_fingerprint`. `mgmt-zenoh-telemetry`
streams structured log records on `smallaios/metrics/log`.
`telemetry-otel-export-v1` (pending) will ship those same
records off-box as OTLP/Logs over HTTPS.

What we do **not** have:

1. **Compact cryptographic proofs.** A SHA-3 hash chain is
   linear: proving a record `r_i` is in the log requires
   replaying every record from `r_0` to `r_i`. That is fine
   for forensic review of a single box but does not scale to
   fleet auditing (1k boxes × 30 days × ~10 records/s ≈ 26 B
   records to walk).
2. **Online consistency proofs.** With a linear chain a
   verifier cannot answer "given chain head H1 last hour and
   chain head H2 now, prove H2 strictly extends H1" without
   replaying the diff. Merkle-tree commit logs (immudb,
   Certificate Transparency, Sigstore Rekor, Google Trillian)
   answer that in O(log n).
3. **A neutral external verifier.** Today the only consumer
   of the audit fingerprint is the box itself. There is no
   well-known protocol an off-box auditor can speak to say
   "show me an inclusion proof for record #42 against chain
   head H, and a consistency proof from H1 to H." This is
   exactly the property an immutable database like **immudb**
   bakes in.
4. **A clear separation between "what we write locally" and
   "what we ship for archival."** Today's audit ring is
   one design serving both jobs; remote archival to an
   immutable store wants a different cadence, different
   transport, and different durability properties.

The user proposal: extend the existing audit infrastructure
into a *verifiable* audit log that (a) keeps a compact
Merkle-tree commitment locally and (b) ships records plus
signed roots to an external immutable store —
**immudb is the leading candidate** — without running
immudb's server binary on-box. The constraints are the
familiar ones: `#![no_std]`, no `std`-only dependencies, no
multi-MB blobs in the image, no external C deps, and the
existing TLS 1.3 + PQC + (clean-room) protobuf stack is the
only transport we are willing to maintain.

This proposal is **exploratory**: it surfaces the design
options, the trade-offs, and the open questions, and asks
the reviewer to pick a direction before any code is written.

## What Changes

This proposal does not yet commit to a single implementation;
it presents three families of designs, the recommendation,
and a short list of decisions the maintainer should make
before we move to a design document.

### Background: how immudb's verifiability works

immudb is a key-value (and SQL) store built on a
**SHA-256 Merkle tree** with a "linear extension" that
covers transactions not yet folded into the tree. Each
transaction's hash (`Alh`) becomes a leaf; the server
returns a *signed* `state` (`{tx_id, alh, signature}`) on
every commit. A client can request:

- **Inclusion proof** — given a key/value at `tx_i`, prove
  it is committed under server state `S`.
- **Consistency proof** — given two server states `S1` and
  `S2` where `S2.tx_id > S1.tx_id`, prove `S2` strictly
  extends `S1` (no rewrite).
- **Signed state** — the server signs `S` with an Ed25519
  key on every reply; the client verifies once and can
  cache the public key.

Transport is gRPC by default; a REST gateway (`immugw`)
exists. First-party SDKs are Java, Go, .NET, Python,
Node.js — **no Rust SDK**. The wire protocol is the
standard `immudb.schema.ImmuService` proto file
([`schema.proto` in the repo](https://github.com/codenotary/immudb/tree/master/pkg/api/schema)).
Historic client-side proof-verification CVEs
([`GHSA-672p-m5jq-mrh8`](https://github.com/codenotary/immudb/security/advisories/GHSA-672p-m5jq-mrh8))
mean any clean-room Rust client must be tested against the
reference SDKs' proof vectors.

### Design Option A — *Speak immudb directly (recommended)*

Treat immudb as the off-box system of record, write a
clean-room `#![no_std]` immudb client, and keep the
on-box log close to what we already have.

- **On-box** (small delta to today):
  - Audit ring stays as-is for fast in-memory writes.
  - The SHA-3-256 hash chain is preserved as the *local*
    integrity primitive (we keep PQC alignment; we already
    pay this cost).
  - A new `audit_export` worker batches records (e.g. 100
    records or 1 s, whichever first), hashes each batch
    with **SHA-256** (immudb's required hash), and pushes
    a `setAll` / `verifiableSetAll` over the network.
  - Server's signed state response is recorded in the local
    audit ring (as a `action = "immudb_state"` record) and
    re-published on `smallaios/metrics/audit_fingerprint`
    alongside the SHA-3 chain head. Both fingerprints are
    visible from `console-monitor-v1`.
  - The same in-memory ring buffer + exponential-backoff
    pattern as `telemetry-otel-export-v1` applies: bounded,
    drop-oldest, never blocks the audit producer.
- **Wire protocol**: immudb is gRPC-first. Two sub-options:
  - **A1 — gRPC over our existing HTTP/3 + TLS 1.3 stack.**
    Real work: HTTP/2 frame parser (~600 LOC) plus a tiny
    gRPC framing layer (5-byte length-prefixed protobuf,
    ~50 LOC). We already speak HTTP/3, so HTTP/2 is the
    awkward sibling — but immudb does not advertise HTTP/3.
    Recommended target: HTTP/2 over TLS 1.3 (mandatory),
    h2c-only paths rejected.
  - **A2 — REST gateway (`immugw`) over HTTPS.** The
    operator deploys `immugw` next to their immudb instance;
    we POST JSON to `/db/{db}/items`. Trivial client (~150
    LOC, reuses the OTLP/HTTP exporter's transport). The
    cost: one extra hop in the operator's stack, plus the
    fact that `immugw` is itself flagged as not the
    recommended path by Codenotary today.
  - **Recommendation**: A1 for production, A2 for the
    bring-up demo. The HTTP/2 layer is reusable for any
    future gRPC service.
- **Client crate**: new `audit-export/` (Layer 1, ~1,200
  LOC). Hand-rolled protobuf encoder/decoder for the small
  subset of `schema.proto` we actually need (`KeyValue`,
  `SetRequest`, `VerifiableSetRequest`, `ImmutableState`,
  `DualProof`, `LinearProof`), modeled on the existing
  clean-room ONNX protobuf parser.
- **Verification**: clean-room Merkle inclusion + dual
  consistency proof verifier in `audit-export/verify.rs`,
  cross-tested against immudb-generated proof vectors and
  the reference auditor (`immuclient audit`).
- **Authn**: immudb supports static token + mTLS. We
  re-use the OTLP keyfile convention: token at
  `/data/audit_export/immudb.token` (mode 0600,
  `Role::Root`-only).
- **Failure modes**: if the immudb endpoint is offline, the
  on-box ring continues to fill, the SHA-3 chain is still
  intact, and once the endpoint comes back the exporter
  catches up. Cap the buffer at `audit.export_buffer_bytes`
  (default 4 MiB, configurable) with drop-oldest.

**Pros**
- Best fit for the user's stated goal: "use immudb for
  remote reporting".
- Reuses existing patterns: protobuf, TLS, bounded buffer,
  exponential backoff, role gate, audit-record observability
  of the exporter itself.
- Customer-friendly: an enterprise SOC that already runs
  immudb gets a drop-in.
- Cryptographically strong: real Merkle proofs from a
  signed third-party log.

**Cons**
- ~1,200 LOC of new code we have to test against a moving
  immudb proto target.
- Tight coupling to immudb's wire protocol (gRPC + their
  custom proof layout). If immudb breaks compatibility in
  a future major version, we follow.
- The local chain is still SHA-3 while the exported chain is
  SHA-256 — operators must understand they are two
  independent integrity systems, not one chain.

### Design Option B — *Make our local log Merkle-shaped, then export to anything*

Instead of speaking immudb specifically, evolve
`mgmt-audit-log` from a linear SHA-3 chain into a
**Merkle Mountain Range (MMR)** or RFC-6962 binary Merkle
tree, with periodic signed roots. Then write a generic
exporter that targets any verifiable log — immudb, Sigstore
Rekor, Google Trillian, or a private RFC-6962 endpoint.

- **On-box** (bigger delta):
  - The on-disk format gains MMR sibling hashes alongside
    each leaf, so an offline auditor can produce inclusion
    proofs without the network.
  - Signed checkpoints (already specified) become *Merkle
    roots*, not chain heads. The ML-DSA-65 signature is
    over the root, signing interval = every N leaves
    *and* every M seconds (whichever first).
  - `smallaios/metrics/audit_fingerprint` publishes
    `{root_hash, tree_size, sig}` instead of the linear
    chain head.
- **Exporter** is a thin adapter: one trait
  `VerifiableLogSink` with three implementations:
  - `ImmudbSink` (Option A1's gRPC client),
  - `RekorSink` (Sigstore Rekor's `/api/v1/log/entries` —
    RFC 6962-ish, easier wire format),
  - `TrillianSink` (Google Trillian's gRPC, very close to
    the immudb effort).
- **Authn**: per-sink.

**Pros**
- The local format is genuinely Merkle-verifiable on its
  own, independent of any remote service. Operators in
  air-gapped deployments still get O(log n) proofs.
- Hedges against immudb-the-product going away or pivoting.
- Aligns with the broader transparency-log ecosystem
  (Sigstore, CT, Trillian).
- The signed Merkle root is what every modern transparency
  log already standardizes on; future export targets are
  cheap.

**Cons**
- Bigger blast radius on `mgmt-audit-log` (an already-
  archived spec). The on-disk format changes, the
  fingerprint payload changes, every audit-aware test
  changes.
- "Generic verifiable log" is more code than "talk to
  immudb": at least three sink implementations to keep
  honest in CI.
- Local Merkle storage is heavier than a linear chain —
  rough math: 2N hashes for N leaves, so a 10 M-record log
  at SHA-256 (32 B) ≈ 640 MiB of sibling state, vs 320 MiB
  for the leaves alone. The rotation policy needs revisiting.

### Design Option C — *Don't change the local log; ship to immudb opportunistically*

Minimal change: leave `mgmt-audit-log` exactly as-is. Add a
side-car worker that tails `/data/audit/log.jsonl`, batches
records, and pushes them to immudb. No local Merkle tree, no
new cryptographic state on-box, no API surface changes.

**Pros**
- ~250 LOC. Smallest viable change.
- Local audit story is unchanged — zero regression risk.
- Easy to disable / swap.

**Cons**
- No local proof story improvement.
- The local SHA-3 chain and the remote immudb tree have *no*
  relationship; tampering with `/data/audit/log.jsonl`
  *before* the exporter ships it is undetectable.
- Falls short of the user's framing — "immutable logs
  locally **and** reporting remotely". This is only the
  remote half.

### Recommendation

**Option A1**: speak immudb directly, gRPC over our HTTP/2
+ TLS 1.3 + clean-room protobuf stack, REST gateway path as
a fallback knob. Keep the on-box SHA-3 chain unchanged for
v1; revisit Option B's Merkle-tree local format as a follow-
on (`verifiable-audit-log-v2`) once the export plumbing is
production-validated.

Rationale: A1 delivers the user's stated outcome with the
least disturbance to the already-archived `mgmt-audit-log`
spec, reuses every transport pattern we already maintain,
and leaves the door open to retro-fit Option B's on-box
Merkle tree later without re-doing the export side.

### Out of scope for v1 (flagged for follow-on changes)

- **SQL surface on the immudb side.** v1 writes immutable
  key-value records (`audit/<host_id>/<ts_ns>` → JSONL
  record). The immudb SQL feature is unused.
- **On-box Merkle tree.** Option B's local-format upgrade
  is deferred. The on-box log keeps its current SHA-3
  linear chain plus ML-DSA-65 checkpoints.
- **Backfill of pre-existing archives.** The exporter ships
  only new records starting from the boot at which it is
  enabled. Replaying old rotated `.jsonl.gz` files is a
  v2 nice-to-have.
- **Trillian / Rekor / CT sinks.** Possible follow-on if
  Option B is later chosen.
- **PostgreSQL wire protocol** path into immudb. Pure
  bloat for our use case.
- **Local immudb embedded mode.** Codenotary publishes
  an embedded library; it is Go and pulls in BadgerDB —
  not viable in `#![no_std]`. Explicitly out.
- **`syslog` RFC 5424 export.** Many legacy SIEMs want
  syslog. Out of scope here; if needed, a sibling
  proposal `telemetry-syslog-export-v1` reuses the same
  audit-ring tap and adds the RFC 5424 framing.
- **Bus integration with `bus-zenoh` / `bus-dds`** as a
  delivery transport. The export client uses TLS/HTTP/2
  directly; the bus crates are for inference dataflow,
  not for cryptographically-protected audit transport.

## Capabilities

### New Capabilities

- `audit-export-immudb-client`: clean-room `#![no_std]`
  immudb client; covers the protobuf subset, the
  HTTP/2 + TLS 1.3 transport, the signed-state response
  parsing, the inclusion-proof and dual-consistency-proof
  verifier, the test-vector contract against immudb-generated fixtures,
  and the failure-mode semantics (offline → buffer →
  catch-up).
- `audit-export-pipeline`: the batching policy (100
  records OR 1 s, whichever first), the in-memory ring
  buffer (default 4 MiB, drop-oldest), the exponential-
  backoff curve (10 s → 5 min cap, identical to OTLP
  exporter), the "exporter never blocks the audit
  producer" invariant, and the audit-record back-feed
  (`action = "audit_export_attempt"`).
- `audit-export-config-surface`: schema for
  `/data/audit_export/immudb.toml`, separate keyfile
  rule (`/data/audit_export/immudb.token`, 0600,
  `Role::Root`-only), role gate, and the rule that the
  exporter is opt-in (default off, no embedded endpoint).
- `audit-export-fingerprint-cross-binding`: the rule that
  both the local SHA-3 chain head and the remote immudb
  signed state are published on
  `smallaios/metrics/audit_fingerprint`, with stable JSON
  keys (`local_sha3`, `immudb_state`) so off-box auditors
  can correlate; and the rule that
  `console-monitor-v1` surfaces both side-by-side.

### Modified Capabilities

- `mgmt-audit-log` (currently archived): adds the
  `audit_export_attempt` record type and the
  `immudb_state` record type. **Does not** change the
  SHA-3 chain format, the rotation policy, or the
  on-disk JSONL schema otherwise.
- `mgmt-zenoh-telemetry`: adds the keyed sub-fields to
  `smallaios/metrics/audit_fingerprint` (additive,
  schema rule already mandates this).
- `mgmt-config-layout`: adds `/data/audit_export/`
  directory, the keyfile permission rule (identical to
  OTLP's keyfile rule), and the secret-redaction rule
  for audit records mentioning the token file.
- `console-monitor-v1` (active): adds a one-line
  "audit export" status to the header alongside the
  existing OTLP status: `IMMUDB ok tx=12345 last 4s`
  or `IMMUDB err last 38s`. Triggered only when the
  exporter is enabled.

## Impact

- **Code:**
  - New `audit-export/` crate (Layer 1, ~1,200 LOC).
    - `immudb/schema.rs` — clean-room protobuf encoder
      / decoder for the immudb message subset (~400 LOC).
    - `immudb/client.rs` — gRPC over HTTP/2 client,
      `set` / `verifiableSet` / `currentState` calls
      (~250 LOC).
    - `immudb/verify.rs` — inclusion + dual consistency
      proof verifier, signed-state Ed25519 check
      (~200 LOC).
    - `pipeline.rs` — batching, ring buffer, retry
      (~200 LOC).
    - `config.rs` + `cli.rs` — TOML schema + management
      shell commands (~150 LOC).
  - HTTP/2 layer: either reuse the existing HTTP/3
    crate's framing (no — different transport) or a
    new minimal HTTP/2 client (~600 LOC). Lives in
    `net/` to be reusable.
- **Tests:** ~80 new tests:
  - Protobuf encoder round-trips for each message.
  - Inclusion-proof verifier against ≥20 immudb-
    generated vectors (a small Rust binary in
    `audit-export/tests/scripts/gen_fixtures.rs` talks
    to a developer-controlled immudb sidecar and writes
    fixture blobs to `audit-export/tests/proof_vectors/`;
    SmallAIOS stays Rust-only — no Go in the repo).
  - Dual consistency proof verifier across ≥10 vectors.
  - Ed25519 signed-state verifier against the
    immudb-emitted state JSON.
  - Ring buffer drop-oldest under sustained 10 KHz
    audit writes.
  - Exponential backoff curve matches OTLP exporter.
  - Keyfile permission rejection (0644 → boot fails).
  - End-to-end: spin up immudb 1.x in a Docker
    sidecar in CI, run the exporter against it for
    5 minutes, walk the chain with `immuclient audit`,
    assert zero divergence.
  - Test count target: 4,143 → ≥4,310 (assumes
    `management-login-v1` and `telemetry-otel-export-
    v1` have already landed).
- **Boot footprint:** ~100 KB code, ~5 MiB live
  including the configurable buffer. Zero CPU when
  `enabled = false`.
- **Container image:** unchanged (the immudb server is
  **not** packaged; only the client).
- **Network:** at the default batching policy and an
  audit rate of ~10 records/s, ~2 KB / batch
  compressed, ~12 KB / minute, ~17 MB / day per host
  to immudb. Independent of OTLP volume.
- **Downstream:** unblocks the "compliance / SOC-ready"
  story for enterprise customers. Foundational for any
  later move to a full Merkle on-box format (Option B
  as `-v2`).
- **Dependencies:**
  - `management-login-v1` — for the audit ring,
    role gate, keyfile convention.
  - `telemetry-otel-export-v1` — borrows transport
    plumbing and the keyfile loader pattern (does not
    *require* OTel to be enabled).
  - `mgmt-config-layout` — for the keyfile + secret-
    redaction rules.
- **Risks:**
  1. **Proof-verifier bugs.** Immudb has historically
     shipped client SDKs with broken verifiers
     ([GHSA-672p-m5jq-mrh8](https://github.com/codenotary/immudb/security/advisories/GHSA-672p-m5jq-mrh8)).
     The CI fixture set generated against a live
     immudb sidecar is non-optional.
  2. **Hash-algorithm split.** SHA-3-256 locally,
     SHA-256 remotely. Document loudly; operators must
     understand they are two integrity systems and
     correlation is by `host_id × tx_id × ts_ns`,
     not by hash equality.
  3. **HTTP/2 stack.** New transport surface to
     maintain. Mitigation: scope strictly to the
     gRPC framing we need (no server-push, no
     trailers-only optimization, no stream priority);
     fuzz the parser.
  4. **immudb wire-protocol drift.** Pin to a specific
     immudb major version (likely v1.9.x) and document
     the upgrade path. CI runs against pinned image.
  5. **Sensitive content leakage.** Audit records can
     contain config diffs (`before` / `after` JSON).
     If the operator's `before` value is a secret
     (rare, but possible — e.g. a rotated API key in
     `mgmt-config-layout`'s redaction policy), it
     would be exported. Mitigation: the existing
     `mgmt-config-layout` redaction rule applies
     *before* the audit record is written, so
     redaction is already at the right boundary; the
     exporter just ships whatever the audit ring
     holds.

## Open Questions

1. **Recommend A1 vs A2 vs B?** This proposal recommends
   A1 (direct gRPC). Worth confirming before any code is
   written — the HTTP/2 effort is the largest single
   commitment.

2. **Hash-algorithm reconciliation.** Three options:
   (a) keep SHA-3 local + SHA-256 remote, two
   independent integrity systems (current
   recommendation); (b) introduce a dual-hash
   per-record so a single record carries both digests
   (~32 B/record overhead, ~860 KiB / 24 h at 10 rec/s
   — easily absorbed); (c) switch the on-box chain to
   SHA-256 (small win, breaks compatibility with the
   archived `mgmt-audit-log` spec). Leaning (a) for
   simplicity, (b) is the rigorous answer.

3. **Authentication to immudb.** Static token in v1
   keeps parity with OTLP's basic-auth keyfile. Should
   we also offer mTLS in v1? Cost is a config knob and
   maybe ~30 LOC; benefit is "no shared secret on
   disk." Leaning yes if `management-login-v1`'s
   client-cert path is ready in time, otherwise v2.

4. **Endpoint discovery.** Static endpoint URL in the
   TOML, or service discovery via the local
   `bus-zenoh` ring (an immudb collector that
   announces itself)? Static for v1 — auto-discovery
   is a footgun for security.

5. **Database / collection / "ledger" naming.** Immudb
   has multiple databases per server. Default DB name?
   Likely `smallaios_audit`. Per-host? Per-fleet?
   Leaning one DB per fleet, keys prefixed with
   `host_id` — fewer admin objects, same isolation.

6. **Failsafe behavior under sustained immudb
   unavailability.** If the endpoint is gone for a
   week, the in-memory ring buffer overflows. Three
   options: (a) drop-oldest (current recommendation,
   matches OTLP); (b) spool to `/data/audit_export/
   spool/` with size cap; (c) refuse to write new
   audit records until export drains (would defeat
   the whole "exporter never blocks the producer"
   invariant — out). Leaning (a) for v1, (b) for v2.

7. **Verifier story off-box.** Does SmallAIOS ship a
   `smallaios-audit-verify` CLI for the operator's
   workstation, or do we just document
   `immuclient audit` as the canonical tool? Leaning
   "document `immuclient` for v1, ship our own CLI
   in v2 only if customer feedback demands a
   single-binary verifier."

8. **What about other immutable databases?** This
   proposal evaluated only immudb against the
   constraint set. Worth at least a paragraph in
   `design.md` on Sigstore Rekor, Google Trillian, and
   AWS QLDB — to justify why immudb wins for
   "self-hostable, dual cloud + on-prem, embedded-
   friendly schema."

9. **Project usage telemetry overlap.** The pending
   `project-usage-telemetry-v1` change designs the
   "telemetry the SmallAIOS *project* collects from
   users" surface. Audit logs are *first-party*
   operator data; they must not be confused with
   project telemetry. Re-affirm here that this
   change is operator-controlled, opt-in, with no
   default endpoint and no embedded credentials —
   identical to `telemetry-otel-export-v1`.
