// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Synthetic X.509 certificate generator for tests (task 5.8).
//!
//! Builds Ed25519-signed DER certificates with configurable
//! validity, SAN, BasicConstraints, KeyUsage and ExtKeyUsage so the
//! structure parser ([`super::x509`]) and the chain verifier
//! ([`super::verify`]) can be exercised end-to-end without any
//! external fixtures or a real CA. NOT compiled into the shipping
//! crate — `#[cfg(test)]` only.

#![cfg(test)]

use alloc::vec::Vec;
use smallaios_security::crypto::ed25519::{ed25519_keygen, ed25519_sign, Ed25519KeyPair};

// ─── DER encoding helpers ──────────────────────────────────────────

/// Encode a DER length (short form < 128, else minimal long form).
fn der_len(len: usize) -> Vec<u8> {
    if len < 128 {
        alloc::vec![len as u8]
    } else {
        let mut bytes = Vec::new();
        let mut n = len;
        while n > 0 {
            bytes.insert(0, (n & 0xff) as u8);
            n >>= 8;
        }
        let mut out = alloc::vec![0x80 | bytes.len() as u8];
        out.extend_from_slice(&bytes);
        out
    }
}

/// Build a TLV from a tag and contents.
fn tlv(tag: u8, contents: &[u8]) -> Vec<u8> {
    let mut out = alloc::vec![tag];
    out.extend_from_slice(&der_len(contents.len()));
    out.extend_from_slice(contents);
    out
}

fn seq(contents: &[u8]) -> Vec<u8> {
    tlv(0x30, contents)
}
fn set(contents: &[u8]) -> Vec<u8> {
    tlv(0x31, contents)
}
fn oid(body: &[u8]) -> Vec<u8> {
    tlv(0x06, body)
}
fn integer(body: &[u8]) -> Vec<u8> {
    tlv(0x02, body)
}

/// BIT STRING with zero unused bits.
fn bit_string(body: &[u8]) -> Vec<u8> {
    let mut contents = alloc::vec![0x00];
    contents.extend_from_slice(body);
    tlv(0x03, &contents)
}

fn octet_string(body: &[u8]) -> Vec<u8> {
    tlv(0x04, body)
}

fn explicit(tag: u8, contents: &[u8]) -> Vec<u8> {
    tlv(tag, contents)
}

/// commonName (2.5.4.3) DN with a single CN RDN.
fn distinguished_name(cn: &str) -> Vec<u8> {
    let atv = seq(&[oid(&[0x55, 0x04, 0x03]), tlv(0x0c, cn.as_bytes())].concat());
    seq(&set(&atv))
}

/// Ed25519 SubjectPublicKeyInfo for a 32-byte public key.
fn ed25519_spki(pk: &[u8; 32]) -> Vec<u8> {
    let alg = seq(&oid(&[0x2b, 0x65, 0x70]));
    seq(&[alg, bit_string(pk)].concat())
}

/// Ed25519 AlgorithmIdentifier (no parameters).
fn ed25519_alg() -> Vec<u8> {
    seq(&oid(&[0x2b, 0x65, 0x70]))
}

/// GeneralizedTime `YYYYMMDDHHMMSSZ` for a given calendar time.
fn gen_time(y: u32, mo: u32, d: u32, h: u32, mi: u32, s: u32) -> Vec<u8> {
    use alloc::format;
    let str = format!("{y:04}{mo:02}{d:02}{h:02}{mi:02}{s:02}Z");
    tlv(0x18, str.as_bytes())
}

// ─── Certificate builder ───────────────────────────────────────────

/// A SubjectAltName entry to embed.
pub enum San {
    Dns(&'static str),
    Ip(Vec<u8>),
}

/// Fluent builder for a synthetic Ed25519 certificate.
pub struct CertBuilder {
    pub subject: &'static str,
    pub issuer: &'static str,
    pub serial: u8,
    pub not_before: (u32, u32, u32, u32, u32, u32),
    pub not_after: (u32, u32, u32, u32, u32, u32),
    pub sans: Vec<San>,
    pub san_extension: bool,
    pub ca: bool,
    pub basic_constraints: bool,
    pub key_cert_sign: bool,
    pub digital_signature: bool,
    pub key_usage: bool,
    pub server_auth_eku: bool,
    pub ext_key_usage: bool,
    /// Override the version field with a raw value (None ⇒ v3).
    pub version_override: Option<u8>,
}

impl Default for CertBuilder {
    fn default() -> Self {
        Self {
            subject: "leaf.example.com",
            issuer: "Test CA",
            serial: 1,
            not_before: (2024, 1, 1, 0, 0, 0),
            not_after: (2099, 1, 1, 0, 0, 0),
            sans: Vec::new(),
            san_extension: true,
            ca: false,
            basic_constraints: true,
            key_cert_sign: false,
            digital_signature: true,
            key_usage: true,
            server_auth_eku: true,
            ext_key_usage: true,
            version_override: None,
        }
    }
}

impl CertBuilder {
    /// A typical leaf: SAN, serverAuth EKU, digitalSignature.
    pub fn leaf(subject: &'static str, dns: &'static str) -> Self {
        Self {
            subject,
            sans: alloc::vec![San::Dns(dns)],
            ..Self::default()
        }
    }

    /// A typical CA: cA=true, keyCertSign, no SAN/EKU.
    pub fn ca(name: &'static str) -> Self {
        Self {
            subject: name,
            issuer: name,
            sans: Vec::new(),
            san_extension: false,
            ca: true,
            key_cert_sign: true,
            digital_signature: false,
            server_auth_eku: false,
            ext_key_usage: false,
            ..Self::default()
        }
    }

    fn extensions(&self) -> Vec<u8> {
        let mut exts = Vec::new();

        if self.san_extension {
            let mut names = Vec::new();
            for san in &self.sans {
                match san {
                    San::Dns(d) => names.extend_from_slice(&tlv(0x82, d.as_bytes())),
                    San::Ip(a) => names.extend_from_slice(&tlv(0x87, a)),
                }
            }
            let san_value = octet_string(&seq(&names));
            exts.extend_from_slice(&seq(&[oid(&[0x55, 0x1d, 0x11]), san_value].concat()));
        }

        if self.basic_constraints {
            let ca_bool = if self.ca {
                tlv(0x01, &[0xff])
            } else {
                tlv(0x01, &[0x00])
            };
            let bc_value = octet_string(&seq(&ca_bool));
            let critical = tlv(0x01, &[0xff]);
            exts.extend_from_slice(&seq(
                &[oid(&[0x55, 0x1d, 0x13]), critical, bc_value].concat()
            ));
        }

        if self.key_usage {
            let mut b0 = 0u8;
            if self.digital_signature {
                b0 |= 0x80;
            }
            if self.key_cert_sign {
                b0 |= 0x04;
            }
            let ku_value = octet_string(&bit_string(&[b0]));
            let critical = tlv(0x01, &[0xff]);
            exts.extend_from_slice(&seq(
                &[oid(&[0x55, 0x1d, 0x0f]), critical, ku_value].concat()
            ));
        }

        if self.ext_key_usage {
            let mut ekus = Vec::new();
            if self.server_auth_eku {
                ekus.extend_from_slice(&oid(&[0x2b, 0x06, 0x01, 0x05, 0x05, 0x07, 0x03, 0x01]));
            }
            let eku_value = octet_string(&seq(&ekus));
            exts.extend_from_slice(&seq(&[oid(&[0x55, 0x1d, 0x25]), eku_value].concat()));
        }

        explicit(0xa3, &seq(&exts))
    }

    fn tbs(&self, subject_pk: &[u8; 32]) -> Vec<u8> {
        let version = explicit(0xa0, &integer(&[self.version_override.unwrap_or(2)]));
        let serial = integer(&[self.serial]);
        let sig_alg = ed25519_alg();
        let issuer = distinguished_name(self.issuer);
        let validity = {
            let (y, mo, d, h, mi, s) = self.not_before;
            let nb = gen_time(y, mo, d, h, mi, s);
            let (y, mo, d, h, mi, s) = self.not_after;
            let na = gen_time(y, mo, d, h, mi, s);
            seq(&[nb, na].concat())
        };
        let subject = distinguished_name(self.subject);
        let spki = ed25519_spki(subject_pk);
        let exts = self.extensions();
        seq(&[
            version, serial, sig_alg, issuer, validity, subject, spki, exts,
        ]
        .concat())
    }

    /// Build the DER certificate: `subject_kp` provides the SPKI;
    /// `issuer_kp` signs the TBS.
    pub fn build(&self, subject_kp: &Ed25519KeyPair, issuer_kp: &Ed25519KeyPair) -> Vec<u8> {
        let tbs = self.tbs(subject_kp.public_key.as_bytes());
        let sig = ed25519_sign(&issuer_kp.secret_key, &tbs);
        let sig_value = bit_string(sig.as_bytes());
        seq(&[tbs, ed25519_alg(), sig_value].concat())
    }

    /// Build a self-signed certificate (subject == issuer key).
    pub fn build_self_signed(&self, kp: &Ed25519KeyPair) -> Vec<u8> {
        self.build(kp, kp)
    }
}

/// Deterministic Ed25519 keypair from a one-byte seed fill.
pub fn keypair(seed_fill: u8) -> Ed25519KeyPair {
    ed25519_keygen(&[seed_fill; 32])
}
