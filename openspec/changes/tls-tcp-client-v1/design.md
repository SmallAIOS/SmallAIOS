## Context

`verifiable-audit-log-v1` shipped `TlsStreamLike: Read + Write
+ close()` as the integration seam between the
`audit-export` pipeline and a future TLS 1.3 client. Until
that client exists, no audit records can leave the box —
the operator-facing config validator already refuses to
boot the exporter without `https://`.

The workspace has the *cryptographic* pieces of TLS 1.3
already. `net::quic::tls` ships:

- `TlsKeySchedule` — HKDF over SHA-256 / SHA-384 derivations
  (`derive_handshake_secrets`,
  `derive_application_secrets`, `protection_keys`).
- `HybridKeyShare` — `X25519 + ML-KEM-768` key share
  encode/decode.
- `HybridServerShare` — server's ciphertext reply.
- `PacketProtectionKeys` — AEAD adapter (currently scoped to
  QUIC's `EncryptionLevel`).

What's missing is the **wire transport** for non-QUIC TLS.
QUIC carries the TLS handshake in CRYPTO frames; standard
HTTPS carries it in TLS records, framed by a 5-byte
record header (RFC 8446 § 5). The handshake messages
themselves are wrapped in a TLS handshake protocol record
(opaque type 22), application data in a separate record
type (23). All records after the ServerHello are
AEAD-encrypted with the handshake-traffic keys, then
re-keyed to application-traffic keys after Finished.

`security::crypto::aes_gcm` exists; ChaCha20-Poly1305 does
not. That gap matters: TLS 1.3 cipher-suite negotiation
includes `TLS_CHACHA20_POLY1305_SHA256`, mandated for
environments without AES hardware (low-power ARM cores
without the Cryptography Extension, RISC-V boards before
the Zk* extensions).

The change is **layered in two crates**:

1. `tls-client/` — new Layer-1 crate, `#![no_std]` core +
   `std` adapter. Owns record framing, handshake state
   machine, certificate-chain verification, trust-store
   loader.
2. `security::crypto::chacha20_poly1305` — sub-add of one
   AEAD primitive, sized to the TLS 1.3 cipher-suite
   matrix.

## Goals / Non-Goals

**Goals:**

- Land a TLS 1.3 over TCP client that satisfies the
  `TlsStreamLike` trait the `audit-export` pipeline
  consumes.
- Reuse every byte of crypto already in `net::quic::tls`
  and `security::crypto`. **No new crypto primitives**
  except `chacha20_poly1305`, which is mandated by TLS
  1.3's cipher-suite matrix.
- Strict TLS 1.3 only — TLS 1.2 ClientHello reply triggers
  a hard abort.
- Trust-store-anchored cert chain verification; refusal of
  empty trust stores, expired certs, mismatched SAN,
  SHA-1-signed certs.
- Hostname matching per RFC 6125.
- PQC-hybrid (`X25519+ML-KEM-768`) negotiation when the
  operator opts in.
- One real working chain: open TCP, run handshake, expose
  a `Read + Write + close()` stream that `TlsGrpcTransport`
  consumes verbatim.
- Test coverage that runs entirely on host (`cargo test`):
  every handshake message round-trips, the X.509 parser
  passes ≥10 corpus vectors, hostname matcher exercises
  every RFC 6125 edge case.

**Non-Goals:**

- TLS server. Client only.
- 0-RTT (early data). Defer to `tls-tcp-client-v2`.
- Session resumption / PSK. Defer.
- mTLS — client cert presentation. Tracked as a separate
  follow-on `tls-mtls-v2` that the `audit-export`
  `auth_mode = "mtls"` path consumes.
- Hardware TLS offload. Future hardware-driver crate.
- CRL fetching, OCSP stapling, Certificate Transparency
  SCT verification. Defer to `tls-revocation-v1` (a
  separately-scoped change whose threat model is more
  complex).
- DTLS. Out — orthogonal to this stack.

## Decisions

### D1. Two crates, not one

**Decision:** New `tls-client/` Layer-1 crate + a
sub-add of `security::crypto::chacha20_poly1305`. Do
not extend `net::quic::tls` to cover TCP records.

**Why:** `quic::tls` is structurally scoped to QUIC's
CRYPTO-frame transport; adding a parallel
record-framing path would muddle the module. A separate
crate gives the TLS-over-TCP client its own DSM boundary,
its own fuzz target, its own audit story. Both crates
share crypto primitives via the existing `security/`
layer.

**Alternative considered:** "Just add a `record_io.rs`
to `quic/`." Rejected because QUIC's encryption levels
(Initial / Handshake / 1-RTT) don't map cleanly to TLS
records (handshake / application_data), and QUIC's
PacketProtectionKeys carries QUIC-specific header
protection that has no TLS counterpart.

### D2. Cipher-suite policy: AES-256-GCM + ChaCha20-Poly1305

**Decision:** The client advertises exactly two cipher
suites in ClientHello, in this preference order:

1. `TLS_AES_256_GCM_SHA384` (strongest classical AEAD;
   uses `security::crypto::aes_gcm`, AES-256 only)
2. `TLS_CHACHA20_POLY1305_SHA256` (constant-time on
   any 32-bit machine; preferred on AES-hardware-less
   targets; uses `security::crypto::chacha20_poly1305`
   added in this change)

**Why two, not three:** the existing
`security::crypto::aes_gcm` ships AES-256-GCM only.
TLS_AES_128_GCM_SHA256 is mandatory-to-implement per
RFC 8446 § 9.1; deferring it is a conscious
non-conformance trade for this PR's scope. A follow-on
change `security-aes-128-v1` would add the 128-bit
variant (which AES-NI hardware can do for free; only
the software path needs new code). Until then, the
client refuses to negotiate AES-128-GCM-SHA256 by
simply not advertising it.

**Alternative considered:** Add AES-128-GCM in this
change (~300 LOC for the software path + KAT corpus).
Rejected because (a) it's a separate well-bounded
addition that belongs in its own change, (b) every
production TLS 1.3 deployment we'd talk to negotiates
AES-256 or ChaCha20 over AES-128 when offered, (c) the
RFC 8446 § 9.1 MUST is at implementation scope, not
per-connection — a client advertising fewer suites
than the spec's MUST list is still functional, just
not strictly conformant.

### D3. Hybrid key exchange is **opt-in**, not default

**Decision:** ClientHello advertises `x25519` as the
primary key share by default. When the operator sets
`tls.require_pqc = true` in `immudb.toml`, the client
advertises `X25519+ML-KEM-768` first and refuses to
negotiate without it.

**Why:** Today most TLS 1.3 servers (including
production immudb deployments) do not yet support the
hybrid group. Defaulting to hybrid would break every
real deployment. Operators who run a hybrid-capable
peer (a known immudb fork, or a custom build) flip
the flag.

When `require_pqc = true` and the server replies with
`x25519` (rejecting hybrid), the client aborts the
handshake with a documented `TlsAbort::PqcDowngrade`.

**Alternative considered:** Always offer hybrid +
classical, accept whichever the server picks. Rejected
for the same reason: the operator's intent should be
unambiguous. Either you want PQC or you accept
classical — the policy belongs in config, not
opportunistic on every connection.

### D4. X.509 parser is "just enough"

**Decision:** Implement a minimal DER decoder + the
specific X.509v3 fields TLS 1.3 needs:

- `tbsCertificate.serialNumber` (we don't validate
  uniqueness, just log it).
- `tbsCertificate.signature.algorithm` — must be one of:
  `sha256WithRSAEncryption` (only for ≥3072-bit RSA),
  `ecdsa-with-SHA256` (P-256), `Ed25519`, `rsassaPss`
  (with SHA-256+ params). SHA-1 anywhere is refused.
- `tbsCertificate.validity.{notBefore, notAfter}`.
- `tbsCertificate.subject` and `tbsCertificate.issuer`
  (Distinguished Names; we only compare them, not
  pretty-print them).
- `tbsCertificate.subjectPublicKeyInfo` — extract the
  algorithm OID + public-key bytes.
- `tbsCertificate.extensions.SubjectAltName` — DNS +
  IP names. We **require** SAN; certs without it are
  refused (RFC 6125 § 6.4.4 advice).
- `tbsCertificate.extensions.basicConstraints` — CA?
  Path length?
- `tbsCertificate.extensions.keyUsage` and
  `extKeyUsage` — server-auth required on leaf.
- `signatureAlgorithm` + `signatureValue` — verified
  against the issuer's pubkey.

We **do not** parse: NameConstraints (out of scope —
documented limitation), CertificatePolicies,
AuthorityInfoAccess, CRLDistributionPoints,
SignedCertificateTimestamp.

**Why:** "Just enough" keeps the attacker-controlled
surface tiny. Every additional ASN.1 type is one more
parser the fuzzer has to cover. The features we omit
(CRL fetch, OCSP, SCT) are tracked in their own
follow-on.

**Alternative considered:** Pull in `webpki` (the
dominant Rust X.509 verifier). Rejected because it's
6,000+ LOC of `unsafe`-flecked dependency, doesn't fit
the SmallAIOS clean-room rule, and pulls
`ring`+`untrusted` transitively.

### D5. Trust store is operator-controlled, no system anchor by default

**Decision:** Trust store path is configured via
`/data/audit_export/immudb.toml` (already shipped) as
`tls.trust_store_path`. Default: empty (which
**refuses** every cert chain — the operator must
explicitly populate it).

**Why:** SmallAIOS doesn't have a Mozilla CA bundle
baked in; bundling one would mean tracking upstream
updates, which is a sustained ops burden. Operators who
talk to a public-CA-signed immudb point at a Mozilla CA
bundle they manage; operators on private CAs point at
their internal root. Either way, the operator chooses.

**Alternative considered:** Bundle a minimal subset
(e.g. only Let's Encrypt ISRG Root X1). Rejected
because (a) it accidentally implies SmallAIOS trusts
all sites that LE has issued for, which is too broad,
and (b) it forces us into a CA-bundle update cadence.

### D6. Hostname matching follows RFC 6125 § 6.4

**Decision:** Match the operator-supplied hostname (from
`endpoint`) against the leaf cert's SAN according to
RFC 6125 § 6.4.1 (DNS-ID) and § 6.4.2 (CN-ID is **not**
consulted — leaf certs without SAN are already refused
in D4).

Wildcard rules:

- `*.example.com` matches exactly one label:
  `foo.example.com` matches; `foo.bar.example.com`
  does not.
- Wildcard MUST be in the leftmost label and MUST be
  the entire leftmost label: `*foo.example.com`,
  `f*o.example.com`, `foo.*.example.com` all rejected.
- Wildcard never matches `example.com` itself.

IP literal endpoints (`endpoint = "https://192.0.2.1:3322"`)
match `iPAddress` SAN entries only — DNS-name SAN
entries cannot match IP literals.

**Why:** RFC 6125 is the canonical rule. Permissive
variants (multiple `*` labels, mid-label `*`) have led
to certificate-confusion vulnerabilities in the past;
we refuse them.

### D7. Handshake state machine is explicit, no async runtime

**Decision:** The handshake is a synchronous state
machine: each `step()` call reads / writes one record,
returns one of `{Continue, NeedMoreInput, Done, Abort(reason)}`.
The integration layer drives `step()` in a loop until
`Done`. No `async`, no executor, no `Pin<&mut Self>`.

**Why:** Matches the `Read + Write` ergonomics of
`TlsStreamLike` and `std::net::TcpStream`. Blocking I/O
in the integration layer; the TLS state machine is
zero-cost on top.

**Alternative considered:** An `async` API with
`tokio`. Rejected for the workspace-wide reason
(SmallAIOS is `#![no_std]` and the existing transport
is synchronous everywhere; introducing `tokio` here
would force it everywhere).

### D8. Authority binding is non-negotiable

**Decision:** Every TLS connection MUST be opened with
an explicit `authority` (hostname + port) drawn from
the operator's `endpoint`. SNI MUST be set to that
hostname. Hostname matching after handshake MUST be
against the same hostname. The `TlsGrpcTransport` already
passes the operator-supplied authority into the
`:authority` HTTP/2 pseudo-header — this change wires
the same string into the TLS layer below it.

**Why:** Cert validation against the **same name** the
operator typed in their config is the only way to
prevent a confused-deputy attack where, e.g., DNS
hijacks the endpoint and a "valid" cert for the
hijacker's domain passes.

### D9. Error class taxonomy

**Decision:** `TlsClientError` is a flat enum:

| Variant | When it fires |
|---|---|
| `TcpConnect` | TCP-layer connect failed. |
| `Io(IoErr)` | Read/write returned an I/O error. |
| `BadRecord` | Record header malformed or oversized. |
| `Version` | Peer advertised TLS < 1.3 anywhere in the handshake. |
| `BadHandshake` | Handshake message malformed or out of order. |
| `KeyExchange` | Hybrid / classical key-exchange math failed. |
| `BadCertificate` | DER parse failed, SHA-1 signature, no SAN, etc. |
| `ChainUntrusted` | No chain to a trust-store anchor. |
| `Expired` | leaf or intermediate `notAfter` in the past. |
| `NameMismatch` | SAN does not include the operator's hostname. |
| `PqcDowngrade` | `require_pqc = true` and peer rejected hybrid. |
| `Aead` | AEAD decryption / authentication failed. |

Each maps to a distinct `TransportError` so the
audit-export pipeline's retry-class classifier can
treat them differently:

- `TcpConnect` / `Io` / `BadHandshake` → `Retry` class
  (transient).
- Everything else → `HardFail` (no point retrying a
  bad cert).

### D10. Test strategy

**Decision:** Three test surfaces:

1. **Unit tests in `tls-client/`** — per-module round
   trips: record encode/decode, handshake message
   encode/decode, X.509 DER round-trip on synthetic
   certs, hostname matcher table.
2. **Cross-vector tests** — checked-in DER blobs from
   a known-good TLS 1.3 corpus
   (`tls-client/tests/corpus/`). Each file is one
   ServerHello + Certificate sequence and the expected
   verifier outcome.
3. **End-to-end smoke** — `tls-client/tests/e2e.rs`
   `#[ignore]`-gated, exercises the full client against
   a real `https://` host when `TLS_E2E_URL` is set.

X.509 parser gets a `cargo-fuzz` target (~60 s per PR).
The handshake state machine has no fuzz target in v1
(state-machine fuzzing has higher ROI in v2 once
session resumption / 0-RTT add states).

### D11. Build-feature gating

**Decision:** `tls-client/` is unconditional in the
workspace (no cargo feature). `container::audit-export`
already gates the audit-export feature; when on, it
transitively activates `tls-client`. When off,
`tls-client` is not in the dep graph.

**Why:** Matches D10 from `verifiable-audit-log-v1`'s
design — the only consumer of `tls-client` today is
the audit-export feature path, so the existing feature
flag already covers the "zero overhead when off" rule.
If future code wants TLS without audit-export, the
feature graph can be revisited.

### D12. Real RFC 8446 key schedule in `tls-client`; the QUIC stub stays out

**Decision:** Implement the TLS 1.3 key schedule (RFC 8446
§7.1), transcript hash, HKDF-Expand-Label, traffic-key and
Finished-key derivations in
`tls-client/src/handshake/key_schedule.rs`, suite-generic
over SHA-256/SHA-384. Add the missing primitives to
`security/`: `Sha384` (in `sha2`), `hmac_sha2`
(HMAC-SHA-256/384), and `hkdf` (RFC 5869 Extract/Expand over
both hashes). Do **not** route through
`net::quic::tls::TlsKeySchedule`.

**Why:** The Context section above (and task 4.4 as
originally written) assumed `net::quic::tls` shipped real
HKDF derivations. Implementation found it is an XOR-based
placeholder — `net/src/quic/tls.rs` says "Simplified:
XOR-based derivation for stub" — with QUIC-specific labels
("quic key") and a fake transcript hash. Keys derived from it
cannot interoperate with any RFC 8446 peer, which would
defeat this change's headline goal ("one real working
chain"). The SHA-384 path is not optional either:
`TLS_AES_256_GCM_SHA384` is the client's first-preference
suite per D2 and its schedule mathematically requires
HMAC/HKDF-SHA-384, which existed nowhere in the workspace.

**Validation:** every key-schedule vector checked in is
cross-validated against two independent oracles — OpenSSL
3.0's `TLS13-KDF` (EXTRACT_ONLY/EXPAND_ONLY modes, which
encode the "tls13 " label prefix and the internal
Derive-Secret-on-extract semantics) and a Python-stdlib
HMAC/HKDF reference — plus published RFC 4231 / RFC 5869 /
FIPS 180-4 vectors for the primitives.

**Alternative considered:** Fix `net::quic::tls` and share
it. Rejected for this change: the QUIC module's stub is
load-bearing for the QUIC test suite's deterministic
expectations, its `PacketProtectionKeys` carry QUIC header
protection with no TLS counterpart (same reasoning as D1),
and rewriting QUIC key derivation is its own change with its
own interop test surface. A follow-on can migrate `quic::tls`
onto `security::hkdf`.

**Hybrid-group wire format note:** the handshake driver
follows draft-ietf-tls-ecdhe-mlkem for `X25519MLKEM768`
(codepoint 0x11ec): client share = ML-KEM-768 encapsulation
key ‖ X25519 key; server share = ML-KEM-768 ciphertext ‖
X25519 key; shared secret = ML-KEM ss ‖ X25519 ss fed raw to
HKDF-Extract. This deviates from `quic::tls`'s internal
SHA-3 combiner — required for interop with standard hybrid
deployments (OpenSSL 3.5+, BoringSSL).

## Risks / Trade-offs

- **[X.509 parser CVEs]** Even minimal X.509 parsers
  have shipped CVEs (length-confusion, signature-suite
  downgrade, name-constraints bypass).
  **Mitigation:** Fuzz target runs per-PR; the SAN
  parser explicitly rejects malformed UTF-8 in DNS
  names; signature-suite matrix is allow-list, not
  deny-list.

- **[Trust-store contents stale]** The operator must
  rotate their trust store as CAs come and go.
  **Mitigation:** Documented in the operator guide;
  audit-export `audit_export_attempt code = -EACCES`
  fires on every connect failure so trust-store drift
  surfaces in the audit ring.

- **[PQC hybrid not yet supported by immudb]**
  Customers who set `require_pqc = true` against an
  unmodified immudb will see `TlsClientError::PqcDowngrade`
  on every connect.
  **Mitigation:** `require_pqc` defaults to false. The
  operator guide documents the test required to confirm
  the peer supports the hybrid group before flipping.

- **[Record-layer size mismatch]** RFC 8446 § 5.1
  caps `TLSPlaintext.length` at 2^14 and
  `TLSCiphertext.length` at 2^14 + 256. Misreading the
  cap leads to a remote denial-of-service.
  **Mitigation:** Per-record cap enforced before any
  allocation (no `Vec::with_capacity` driven by the
  length field); unit test confirms a forged
  oversized record returns `BadRecord` and consumes
  no memory.

- **[Clock dependency for cert validity]** Validating
  `notBefore`/`notAfter` requires a wall clock. SmallAIOS
  may boot with an unsynchronized clock.
  **Mitigation:** The handshake reads `kernel::clock()`
  for "now". If the clock is < 2026 (sentinel meaning
  "unsynced"), validity check is **bypassed with a
  loud audit record** (`audit_export_unsynced_clock`).
  Operators with strict policies set
  `tls.require_synced_clock = true` to refuse instead.

- **[Hybrid key-share parsing cost]** ML-KEM-768 public
  keys are 1,184 bytes; the hybrid concatenation pushes
  the ClientHello into 2 records. Some misconfigured
  middleboxes drop large ClientHellos.
  **Mitigation:** Document; falls under the same
  `require_pqc` knob as D3.

## Migration Plan

- **Deploy:**
  1. Land the `tls-client/` crate via this change.
  2. Container builds with `--features audit-export`
     gain the working `TcpTlsStream` automatically.
  3. Operators who already have `immudb.toml` configured
     can flip `enabled = true` for the first time and
     the exporter starts.

- **Rollback:**
  1. Set `[exporter] enabled = false` in `immudb.toml`.
     The pipeline unregisters its tap on the audit ring,
     the TLS client is never invoked. Local audit
     logging continues unchanged.
  2. If a compile-time rollback is needed, rebuild the
     container without `--features audit-export`. No
     TLS code is linked.

## Open Questions

1. **OCSP stapling stance.** Defer to `tls-revocation-v1`,
   or fold a minimal `must_staple` check in here? Leaning
   defer — the revocation story is its own threat model
   and audit story.

2. **AES-NI / NEON acceleration.** `security::crypto::aes_gcm`
   today is portable software AES. For SmallAIOS deployments
   that ship audit traffic at >10 K rec/s, AES-NI / NEON
   would matter. Tracking separately; not blocking for v1.

3. **Trust-store hot reload.** Should a trust-store update
   take effect without restarting the exporter? Leaning
   yes; the `mgmt-config-layout`-side reload notifier
   already supports the pattern. Implementation deferred
   to the same follow-on that wires the audit-export
   `ConfigSurface` writes.

4. **SNI for IP-literal endpoints.** RFC 6066 § 3 says
   SNI MUST NOT be sent for IP literals. Behavior is
   documented but worth confirming against the production
   immudb deployments customers actually use.

5. **`security::crypto::chacha20_poly1305` test corpus.**
   RFC 8439 has ~4 test vectors; we'll add those plus
   any extras from the existing
   `security::crypto::aes_gcm` test patterns. Coverage
   target: ≥85 % on first pass, matching the workspace
   ratchet.

6. **Workspace member count.** Adding `tls-client` brings
   the workspace from 27 → 28 crates. Update CLAUDE.md
   architecture diagram + DSM check accordingly when this
   change archives.
