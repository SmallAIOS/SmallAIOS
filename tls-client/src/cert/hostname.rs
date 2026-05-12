// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! RFC 6125 § 6.4 hostname matching.
//!
//! Used to bind the operator-supplied hostname (from
//! `endpoint`) to a SubjectAltName entry on the leaf
//! certificate.
//!
//! ## Rules (RFC 6125)
//!
//! For DNS names:
//!
//! - Exact match: `foo.example.com` matches
//!   `DNS:foo.example.com`.
//! - Wildcards are allowed ONLY when:
//!   1. The wildcard `*` is the entire leftmost label of
//!      the SAN (e.g. `*.example.com`, NOT `f*.example.com`
//!      or `f*o.example.com`).
//!   2. The certificate has at least three labels total.
//!   3. The hostname has the same label count as the SAN.
//!   4. The non-wildcard labels match case-insensitively.
//! - Wildcards never span multiple labels: `*.example.com`
//!   matches `foo.example.com` but NOT `foo.bar.example.com`
//!   or `example.com` itself.
//! - CNs (Common Name) MUST NOT be consulted when the cert
//!   has a SAN — this layer never looks at CN.
//!
//! For IP literals:
//!
//! - The hostname must be a valid IPv4 or IPv6 literal.
//! - Match against `iPAddress` SAN entries only (DNS-name
//!   SAN entries cannot match IP literals).

use crate::{Result, TlsClientError};

/// A single SubjectAltName entry the leaf cert exposes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SanEntry {
    /// DNS name (RFC 6125 § 1.8 dNSName). ASCII; non-IDN.
    Dns(alloc::string::String),
    /// IPv4 (4 bytes) or IPv6 (16 bytes) address.
    IpAddress(alloc::vec::Vec<u8>),
}

/// Match the operator-supplied hostname against the leaf
/// cert's SAN list. Returns `Ok(())` on a positive binding,
/// `Err(NameMismatch)` on no match.
pub fn match_hostname(hostname: &str, sans: &[SanEntry]) -> Result<()> {
    if hostname.is_empty() {
        return Err(TlsClientError::NameMismatch);
    }
    // First decide whether the hostname is an IP literal.
    if let Some(ip) = parse_ip_literal(hostname) {
        for san in sans {
            if let SanEntry::IpAddress(addr) = san {
                if addr.as_slice() == ip.as_slice() {
                    return Ok(());
                }
            }
        }
        return Err(TlsClientError::NameMismatch);
    }
    // Hostname is a DNS name. Validate basic shape (no leading
    // / trailing dot, no empty labels, no whitespace).
    if !is_valid_dns_name(hostname) {
        return Err(TlsClientError::NameMismatch);
    }
    for san in sans {
        if let SanEntry::Dns(san_dns) = san {
            if dns_match(hostname, san_dns) {
                return Ok(());
            }
        }
    }
    Err(TlsClientError::NameMismatch)
}

fn dns_match(hostname: &str, san: &str) -> bool {
    let host_labels: alloc::vec::Vec<&str> = hostname.split('.').collect();
    let san_labels: alloc::vec::Vec<&str> = san.split('.').collect();
    if host_labels.len() != san_labels.len() {
        return false;
    }
    for (i, (h, s)) in host_labels.iter().zip(san_labels.iter()).enumerate() {
        if *s == "*" {
            // Wildcard label allowed ONLY at position 0 AND
            // total labels ≥ 3.
            if i != 0 {
                return false;
            }
            if san_labels.len() < 3 {
                return false;
            }
            if h.is_empty() {
                return false;
            }
            continue;
        }
        // Reject mid-label wildcards explicitly: any `*` in a
        // label that isn't the whole label is illegal.
        if s.contains('*') {
            return false;
        }
        if !s.eq_ignore_ascii_case(h) {
            return false;
        }
    }
    true
}

fn is_valid_dns_name(s: &str) -> bool {
    if s.starts_with('.') || s.ends_with('.') {
        return false;
    }
    if s.len() > 253 {
        return false;
    }
    for label in s.split('.') {
        if label.is_empty() || label.len() > 63 {
            return false;
        }
        for c in label.chars() {
            if !(c.is_ascii_alphanumeric() || c == '-') {
                return false;
            }
        }
        if label.starts_with('-') || label.ends_with('-') {
            return false;
        }
    }
    true
}

/// Parse an IPv4 or IPv6 literal. Returns the raw bytes
/// (4 for v4, 16 for v6). None if not a literal.
fn parse_ip_literal(s: &str) -> Option<alloc::vec::Vec<u8>> {
    if s.contains(':') {
        // IPv6 — accept the bracketless form (the
        // `[…]:port` wrapping is stripped by the caller's
        // endpoint parser).
        parse_ipv6(s)
    } else if s.chars().all(|c| c.is_ascii_digit() || c == '.') {
        parse_ipv4(s)
    } else {
        None
    }
}

fn parse_ipv4(s: &str) -> Option<alloc::vec::Vec<u8>> {
    let parts: alloc::vec::Vec<&str> = s.split('.').collect();
    if parts.len() != 4 {
        return None;
    }
    let mut out = alloc::vec::Vec::with_capacity(4);
    for p in parts {
        if p.is_empty() || p.len() > 3 {
            return None;
        }
        if p.starts_with('0') && p.len() > 1 {
            // Reject leading zeros (RFC 6943 / hostname-spoofing
            // defense).
            return None;
        }
        let n: u32 = p.parse().ok()?;
        if n > 255 {
            return None;
        }
        out.push(n as u8);
    }
    Some(out)
}

fn parse_ipv6(s: &str) -> Option<alloc::vec::Vec<u8>> {
    // Permit at most one "::" elision. We do not accept
    // IPv4-mapped-IPv6 forms in v1 (`::ffff:192.0.2.1`).
    let groups_count = if let Some(pos) = s.find("::") {
        if s[pos + 2..].contains("::") {
            return None;
        }
        let (left, right) = (&s[..pos], &s[pos + 2..]);
        let left_groups: alloc::vec::Vec<&str> = if left.is_empty() {
            alloc::vec::Vec::new()
        } else {
            left.split(':').collect()
        };
        let right_groups: alloc::vec::Vec<&str> = if right.is_empty() {
            alloc::vec::Vec::new()
        } else {
            right.split(':').collect()
        };
        if left_groups.len() + right_groups.len() > 7 {
            return None;
        }
        Some((left_groups, right_groups))
    } else {
        None
    };
    let mut out = alloc::vec::Vec::with_capacity(16);
    if let Some((left, right)) = groups_count {
        for g in &left {
            push_group(g, &mut out)?;
        }
        let zeros = (8 - left.len() - right.len()) * 2;
        out.extend(core::iter::repeat_n(0u8, zeros));
        for g in &right {
            push_group(g, &mut out)?;
        }
    } else {
        let parts: alloc::vec::Vec<&str> = s.split(':').collect();
        if parts.len() != 8 {
            return None;
        }
        for g in &parts {
            push_group(g, &mut out)?;
        }
    }
    if out.len() != 16 {
        return None;
    }
    Some(out)
}

fn push_group(g: &str, out: &mut alloc::vec::Vec<u8>) -> Option<()> {
    if g.is_empty() || g.len() > 4 {
        return None;
    }
    let n = u16::from_str_radix(g, 16).ok()?;
    out.extend_from_slice(&n.to_be_bytes());
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    fn dns(s: &str) -> SanEntry {
        SanEntry::Dns(s.to_string())
    }

    fn ip(bytes: &[u8]) -> SanEntry {
        SanEntry::IpAddress(bytes.to_vec())
    }

    #[test]
    fn exact_dns_match() {
        match_hostname("immudb.example.com", &[dns("immudb.example.com")]).unwrap();
    }

    #[test]
    fn case_insensitive_dns_match() {
        match_hostname("IMMUDB.example.com", &[dns("immudb.EXAMPLE.com")]).unwrap();
    }

    #[test]
    fn dns_no_match_rejected() {
        assert_eq!(
            match_hostname("immudb.example.com", &[dns("other.example.com")]).unwrap_err(),
            TlsClientError::NameMismatch
        );
    }

    #[test]
    fn wildcard_matches_single_label() {
        match_hostname("foo.example.com", &[dns("*.example.com")]).unwrap();
    }

    #[test]
    fn wildcard_does_not_match_multiple_labels() {
        assert_eq!(
            match_hostname("foo.bar.example.com", &[dns("*.example.com")]).unwrap_err(),
            TlsClientError::NameMismatch
        );
    }

    #[test]
    fn wildcard_does_not_match_apex() {
        assert_eq!(
            match_hostname("example.com", &[dns("*.example.com")]).unwrap_err(),
            TlsClientError::NameMismatch
        );
    }

    #[test]
    fn mid_label_wildcard_rejected() {
        assert_eq!(
            match_hostname("foo.example.com", &[dns("f*.example.com")]).unwrap_err(),
            TlsClientError::NameMismatch
        );
        assert_eq!(
            match_hostname("foo.example.com", &[dns("f*o.example.com")]).unwrap_err(),
            TlsClientError::NameMismatch
        );
    }

    #[test]
    fn wildcard_not_in_leftmost_label_rejected() {
        assert_eq!(
            match_hostname("a.b.example.com", &[dns("a.*.example.com")]).unwrap_err(),
            TlsClientError::NameMismatch
        );
    }

    #[test]
    fn wildcard_requires_three_labels() {
        // *.com is too broad.
        assert_eq!(
            match_hostname("example.com", &[dns("*.com")]).unwrap_err(),
            TlsClientError::NameMismatch
        );
    }

    #[test]
    fn dns_does_not_match_ip_san() {
        assert_eq!(
            match_hostname("192.0.2.1", &[dns("192.0.2.1")]).unwrap_err(),
            TlsClientError::NameMismatch
        );
    }

    #[test]
    fn ip_literal_matches_ip_san() {
        match_hostname("192.0.2.1", &[ip(&[192, 0, 2, 1])]).unwrap();
    }

    #[test]
    fn ip_literal_does_not_match_dns_san() {
        assert_eq!(
            match_hostname("192.0.2.1", &[dns("example.com")]).unwrap_err(),
            TlsClientError::NameMismatch
        );
    }

    #[test]
    fn ipv6_literal_matches_ip_san() {
        let v6: alloc::vec::Vec<u8> =
            alloc::vec![0x20, 0x01, 0x0d, 0xb8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x01,];
        match_hostname("2001:db8::1", &[ip(&v6)]).unwrap();
    }

    #[test]
    fn empty_hostname_rejected() {
        assert!(match_hostname("", &[dns("example.com")]).is_err());
    }

    #[test]
    fn trailing_dot_rejected() {
        assert!(match_hostname("foo.example.com.", &[dns("foo.example.com")]).is_err());
    }

    #[test]
    fn empty_label_rejected() {
        assert!(match_hostname("foo..example.com", &[dns("foo..example.com")]).is_err());
    }

    #[test]
    fn label_starts_with_hyphen_rejected() {
        assert!(match_hostname("-foo.example.com", &[dns("-foo.example.com")]).is_err());
    }

    #[test]
    fn ipv4_invalid_octet_rejected() {
        assert!(parse_ipv4("999.0.0.1").is_none());
        assert!(parse_ipv4("1.2.3.4.5").is_none());
        // Leading zero rejected (RFC 6943 spoofing defense).
        assert!(parse_ipv4("01.2.3.4").is_none());
    }

    #[test]
    fn ipv6_invalid_rejected() {
        assert!(parse_ipv6("1:2:3").is_none());
        assert!(parse_ipv6("12345::").is_none());
        assert!(parse_ipv6("1::2::3").is_none());
    }
}
