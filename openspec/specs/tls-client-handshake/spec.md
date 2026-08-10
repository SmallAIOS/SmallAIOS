# tls-client-handshake Specification

## Purpose
TBD - created by archiving change tls-tcp-client-v1. Update Purpose after archive.
## Requirements
### Requirement: TLS 1.3 version pinning at handshake

The `tls-client` ClientHello SHALL advertise `legacy_version = 0x0303` (mandated by RFC 8446) and the `supported_versions` extension carrying exactly `0x0304` (TLS 1.3). On receipt of a ServerHello whose `supported_versions` extension does not select `0x0304`, the client SHALL abort the handshake with `TlsClientError::Version` BEFORE deriving any keys.

#### Scenario: Server selects TLS 1.2 → handshake aborts
- **WHEN** the ServerHello's `supported_versions` extension selects `0x0303`
- **THEN** the client SHALL emit `TlsClientError::Version`
- **AND** no further handshake bytes SHALL be processed

### Requirement: Cipher-suite preference list

The ClientHello SHALL advertise exactly two cipher suites, in this order: `TLS_AES_256_GCM_SHA384`, `TLS_CHACHA20_POLY1305_SHA256`. (AES-128-GCM-SHA256 is deferred to a follow-on change per design.md D2.) The client SHALL accept either of the two as the server's selection; selection of any other suite SHALL be rejected with `TlsClientError::BadHandshake`.

#### Scenario: Server selects unsupported suite → reject
- **WHEN** the ServerHello selects `TLS_AES_128_GCM_SHA256` (not advertised)
- **THEN** the client SHALL return `TlsClientError::BadHandshake`

### Requirement: PQC-hybrid key exchange behind operator opt-in

The default `key_share` extension SHALL advertise `x25519` alone. When `tls.require_pqc = true` is configured, the ClientHello SHALL advertise `X25519+ML-KEM-768` as the first key share and SHALL refuse a ServerHello that replies with any non-hybrid group, surfacing `TlsClientError::PqcDowngrade`.

#### Scenario: Default key share is X25519
- **WHEN** the operator config has `tls.require_pqc = false`
- **THEN** the ClientHello's primary `key_share` SHALL be `x25519` (group 0x001D)

#### Scenario: Hybrid required and server picks classical → reject
- **WHEN** `tls.require_pqc = true` AND the ServerHello selects `x25519` instead of the hybrid group
- **THEN** the client SHALL emit `TlsClientError::PqcDowngrade`
- **AND** the connection SHALL be terminated before deriving handshake-traffic keys

### Requirement: Server Name Indication

The ClientHello SHALL include the `server_name` extension carrying the operator-supplied hostname as `host_name` (RFC 6066 § 3) for every connection whose endpoint resolves to a DNS name. For IP-literal endpoints, the SNI extension SHALL be omitted entirely per RFC 6066 § 3.

#### Scenario: Hostname endpoint → SNI present
- **WHEN** the operator endpoint is `https://immudb.example.com:3322`
- **THEN** the ClientHello SHALL include `server_name = "immudb.example.com"`

#### Scenario: IP-literal endpoint → SNI omitted
- **WHEN** the operator endpoint is `https://192.0.2.1:3322`
- **THEN** the ClientHello SHALL omit the `server_name` extension

### Requirement: CertificateVerify signature-suite allow-list

The client SHALL accept CertificateVerify signatures only from this allow-list: `ed25519`, `ecdsa_secp256r1_sha256`, `rsa_pss_rsae_sha256` (when the signing key is RSA ≥ 3072 bits), `rsa_pss_rsae_sha384`, `rsa_pss_rsae_sha512`. Any other algorithm — including `rsa_pkcs1_sha1`, `ecdsa_sha1`, `dsa_sha1`, or any SHA-1-based suite — SHALL be rejected with `TlsClientError::BadCertificate`.

#### Scenario: SHA-1 signature refused
- **WHEN** the CertificateVerify uses `rsa_pkcs1_sha1`
- **THEN** the client SHALL emit `TlsClientError::BadCertificate`

#### Scenario: Ed25519 signature accepted
- **WHEN** the CertificateVerify is `ed25519` with a valid signature over the transcript
- **THEN** the client SHALL accept it

