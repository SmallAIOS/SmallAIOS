# tls-client-cert-chain Specification

## Purpose
TBD - created by archiving change tls-tcp-client-v1. Update Purpose after archive.
## Requirements
### Requirement: Minimal X.509v3 DER parser

The cert parser SHALL accept the X.509v3 fields the TLS 1.3 handshake consumes: serial number, signature algorithm, issuer, subject, validity window, SubjectPublicKeyInfo, and the SAN / BasicConstraints / KeyUsage / ExtKeyUsage extensions. The parser SHALL reject any cert where any of the following is true:

- The version is not `v3` (DER integer 2).
- The signature algorithm is SHA-1-based (`sha1WithRSAEncryption`, `ecdsa-with-SHA1`, etc.).
- The cert has no SubjectAltName extension.
- The cert's `notAfter` is before `notBefore`.
- Any length field exceeds the enclosing structure.

The parser SHALL NOT allocate buffers sized to attacker-controlled length fields until the field is bounded against the input remaining.

#### Scenario: Cert without SAN refused
- **WHEN** a cert carries no `subjectAltName` extension
- **THEN** the parser SHALL return `TlsClientError::BadCertificate`

#### Scenario: SHA-1 signature refused
- **WHEN** a cert's `signatureAlgorithm` is `sha1WithRSAEncryption`
- **THEN** the parser SHALL return `TlsClientError::BadCertificate`

#### Scenario: Oversized inner length rejected without panic
- **WHEN** an OCTET STRING declares length larger than the enclosing SEQUENCE
- **THEN** the parser SHALL return an error
- **AND** SHALL NOT panic or allocate beyond the input length

### Requirement: Trust-store-anchored chain verification

The verifier SHALL accept a leaf certificate only when a chain from the leaf to a root certificate in the operator-configured trust store can be constructed, where each intermediate's `Subject` matches the next certificate's `Issuer` AND each intermediate's `BasicConstraints` extension has `CA = true` AND each intermediate's `KeyUsage` includes `keyCertSign`. Self-signed leaves SHALL be rejected unless explicitly anchored in the trust store.

#### Scenario: Self-signed leaf without trust-store anchor rejected
- **WHEN** the server presents a self-signed leaf
- **AND** the leaf's fingerprint is NOT in the trust store
- **THEN** the verifier SHALL emit `TlsClientError::ChainUntrusted`

#### Scenario: Chain to trusted root accepted
- **WHEN** the server presents `[leaf, intermediate]` AND the trust store contains `root` whose `Subject` matches `intermediate.Issuer` AND the chain signatures validate
- **THEN** the verifier SHALL accept the chain

### Requirement: Validity-window check with clock sentinel

The verifier SHALL compare the current wall-clock time against every cert's `notBefore` and `notAfter`. When `kernel::clock()` reports a value before 2026-01-01 (the agreed "clock unsynced" sentinel) AND `tls.require_synced_clock = false`, the validity check SHALL be bypassed and a `audit_export_unsynced_clock` audit record SHALL be appended. When `tls.require_synced_clock = true`, an unsynced clock SHALL cause `TlsClientError::Expired` instead.

#### Scenario: Synced clock, expired cert → reject
- **WHEN** the wall clock is 2026-06-01 AND the leaf's `notAfter` is 2026-01-01
- **THEN** the verifier SHALL emit `TlsClientError::Expired`

#### Scenario: Unsynced clock, validity bypassed by default
- **WHEN** the wall clock is 1970-01-01 (sentinel) AND `tls.require_synced_clock = false`
- **THEN** the validity check SHALL be skipped
- **AND** an `audit_export_unsynced_clock` record SHALL be appended

#### Scenario: Unsynced clock with strict policy → reject
- **WHEN** the wall clock is 1970-01-01 AND `tls.require_synced_clock = true`
- **THEN** the verifier SHALL emit `TlsClientError::Expired`

### Requirement: RFC 6125 hostname matching

The verifier SHALL match the operator-supplied hostname against the leaf cert's SAN per RFC 6125 § 6.4.1 (DNS-ID) and § 6.4.2. The CN field SHALL NOT be consulted. Wildcards SHALL be accepted only when (a) the wildcard `*` is the entire leftmost label of a SAN DNS name, (b) the certificate has at least three labels total, and (c) the matched hostname has the same label count.

#### Scenario: Exact match
- **WHEN** hostname is `immudb.example.com` AND SAN contains `DNS:immudb.example.com`
- **THEN** the verifier SHALL accept the name binding

#### Scenario: Wildcard matches single label
- **WHEN** hostname is `foo.example.com` AND SAN contains `DNS:*.example.com`
- **THEN** the verifier SHALL accept the binding

#### Scenario: Wildcard does NOT match multiple labels
- **WHEN** hostname is `foo.bar.example.com` AND SAN contains `DNS:*.example.com`
- **THEN** the verifier SHALL emit `TlsClientError::NameMismatch`

#### Scenario: Mid-label wildcard refused
- **WHEN** SAN contains `DNS:f*o.example.com`
- **THEN** the verifier SHALL emit `TlsClientError::BadCertificate`

#### Scenario: IP-literal endpoint matches iPAddress SAN
- **WHEN** hostname is `192.0.2.1` AND SAN contains `iPAddress:192.0.2.1`
- **THEN** the verifier SHALL accept the binding

#### Scenario: IP-literal endpoint does NOT match DNS-name SAN
- **WHEN** hostname is `192.0.2.1` AND SAN contains only `DNS:example.com`
- **THEN** the verifier SHALL emit `TlsClientError::NameMismatch`

