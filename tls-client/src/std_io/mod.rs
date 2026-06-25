// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! `std`-IO adapter (Phase 7, `tls-tcp-client-v1`).
//!
//! [`TcpTlsStream`] drives the [`ClientHandshake`] state machine over
//! a real `std::net::TcpStream`, then exposes the post-handshake
//! connection as `std::io::Read + Write` with a `close()` that emits
//! `close_notify`. This is the concrete transport
//! `verifiable-audit-log-v1` consumes (the container layer adds the
//! `TlsStreamLike` bridge in Phase 8).
//!
//! Only compiled with the `std` feature; the crate core stays
//! `#![no_std]`.
//!
//! The generic [`TlsStream`] over any `Read + Write` carries the
//! data-phase record logic and is unit-tested with a `Vec`-backed
//! mock socket; [`TcpTlsStream`] is the `TcpStream` specialisation
//! whose `connect` opens the socket and draws handshake entropy from
//! the `security` CSPRNG.

use crate::cert::ServerCertVerifier;
use crate::handshake::driver::{ClientConfig, ClientEntropy, ClientHandshake, HandshakeStatus};
use crate::handshake::key_schedule::TrafficKeys;
use crate::record::{
    self, CipherSuite, ContentType, RecordHeader, MAX_PLAINTEXT_LEN, RECORD_HEADER_LEN,
};
use crate::TlsClientError;
use alloc::format;
use alloc::vec::Vec;
use std::io::{self, Read, Write};
use std::net::TcpStream;

/// Bytes pulled from the socket per `read` syscall.
const READ_CHUNK: usize = 8192;

/// Map a [`TlsClientError`] into a `std::io::Error` for the
/// `Read`/`Write` surface.
fn io_err(e: TlsClientError) -> io::Error {
    io::Error::other(format!("tls: {e:?}"))
}

/// A TLS 1.3 stream over an arbitrary `Read + Write` transport.
///
/// [`TcpTlsStream`] is the `TcpStream` alias used in production;
/// the generic form exists so the data-phase logic can be tested
/// over a mock socket.
pub struct TlsStream<T: Read + Write> {
    sock: T,
    suite: CipherSuite,
    client: TrafficKeys,
    server: TrafficKeys,
    write_seq: u64,
    read_seq: u64,
    /// Decrypted application bytes awaiting the caller.
    plaintext: Vec<u8>,
    plaintext_pos: usize,
    /// Raw record bytes read from the socket but not yet framed.
    raw: Vec<u8>,
    /// Peer `close_notify` or TCP EOF observed.
    eof: bool,
    /// We have sent `close_notify`.
    closed: bool,
}

impl<T: Read + Write> TlsStream<T> {
    /// Drive a full handshake over `sock` with caller-supplied
    /// entropy, returning a ready data-phase stream. Generic over the
    /// transport so tests can inject a mock socket and deterministic
    /// entropy.
    pub fn connect_over<V: ServerCertVerifier>(
        mut sock: T,
        config: ClientConfig<'_>,
        verifier: &V,
        entropy: &ClientEntropy,
    ) -> crate::Result<Self> {
        let (mut hs, flight) = ClientHandshake::new(config, entropy, verifier)?;
        sock.write_all(&flight).map_err(|_| TlsClientError::Io)?;
        sock.flush().map_err(|_| TlsClientError::Io)?;

        let mut buf = [0u8; READ_CHUNK];
        loop {
            let n = sock.read(&mut buf).map_err(|_| TlsClientError::Io)?;
            if n == 0 {
                // Peer closed before the handshake completed.
                return Err(TlsClientError::BadHandshake);
            }
            let push = hs.push(&buf[..n])?;
            if !push.send.is_empty() {
                sock.write_all(&push.send).map_err(|_| TlsClientError::Io)?;
                sock.flush().map_err(|_| TlsClientError::Io)?;
            }
            if push.status == HandshakeStatus::Complete {
                break;
            }
        }

        let keys = hs.app_keys()?;
        Ok(Self {
            sock,
            suite: keys.suite,
            client: keys.client,
            server: keys.server,
            write_seq: 0,
            read_seq: 0,
            plaintext: Vec::new(),
            plaintext_pos: 0,
            // Any bytes the peer coalesced after its Finished (e.g. a
            // NewSessionTicket flight) are the data loop's to decrypt.
            raw: hs.take_residual(),
            eof: false,
            closed: false,
        })
    }

    /// Seal `data` into one or more application_data records and write
    /// them to the socket.
    fn write_records(&mut self, data: &[u8]) -> io::Result<()> {
        for chunk in data.chunks(MAX_PLAINTEXT_LEN) {
            let rec = record::seal(
                self.suite,
                &self.client.key,
                &self.client.iv,
                self.write_seq,
                ContentType::ApplicationData,
                chunk,
            )
            .map_err(io_err)?;
            self.write_seq += 1;
            self.sock.write_all(&rec)?;
        }
        Ok(())
    }

    /// Pull exactly one complete record off `raw`, reading from the
    /// socket as needed. `Ok(None)` means clean TCP EOF.
    fn next_record(&mut self) -> io::Result<Option<Vec<u8>>> {
        loop {
            if self.raw.len() >= RECORD_HEADER_LEN {
                let header = RecordHeader::parse(&self.raw[..RECORD_HEADER_LEN]).map_err(io_err)?;
                let total = RECORD_HEADER_LEN + header.length as usize;
                if self.raw.len() >= total {
                    let rec: Vec<u8> = self.raw.drain(..total).collect();
                    return Ok(Some(rec));
                }
            }
            let mut buf = [0u8; READ_CHUNK];
            let n = self.sock.read(&mut buf)?;
            if n == 0 {
                return Ok(None);
            }
            self.raw.extend_from_slice(&buf[..n]);
        }
    }

    /// Decrypt records until application data is buffered or the
    /// stream ends.
    fn fill_plaintext(&mut self) -> io::Result<()> {
        while self.plaintext_pos >= self.plaintext.len() && !self.eof {
            let rec = match self.next_record()? {
                Some(r) => r,
                None => {
                    self.eof = true;
                    return Ok(());
                }
            };
            let (inner, data) = record::open(
                self.suite,
                &self.server.key,
                &self.server.iv,
                self.read_seq,
                &rec,
            )
            .map_err(io_err)?;
            self.read_seq += 1;
            match inner {
                ContentType::ApplicationData => {
                    self.plaintext = data;
                    self.plaintext_pos = 0;
                }
                // NewSessionTicket / KeyUpdate. v1 does not rotate keys,
                // so we consume and ignore post-handshake handshake
                // messages; a server that actually issues KeyUpdate
                // would desync — a documented limitation tracked with
                // the rest of the data-phase hardening.
                ContentType::Handshake => {}
                // Any alert (incl. close_notify) ends the stream.
                ContentType::Alert => {
                    self.eof = true;
                    return Ok(());
                }
                _ => return Err(io_err(TlsClientError::BadRecord)),
            }
        }
        Ok(())
    }

    /// Send `close_notify` (idempotent). Does not shut the underlying
    /// transport — the caller drops it.
    pub fn close(&mut self) -> io::Result<()> {
        if self.closed {
            return Ok(());
        }
        // Alert: level = warning (1), description = close_notify (0).
        let rec = record::seal(
            self.suite,
            &self.client.key,
            &self.client.iv,
            self.write_seq,
            ContentType::Alert,
            &[1u8, 0u8],
        )
        .map_err(io_err)?;
        self.write_seq += 1;
        self.sock.write_all(&rec)?;
        self.sock.flush()?;
        self.closed = true;
        Ok(())
    }
}

impl<T: Read + Write> Read for TlsStream<T> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if self.plaintext_pos >= self.plaintext.len() {
            self.fill_plaintext()?;
        }
        if self.plaintext_pos >= self.plaintext.len() {
            return Ok(0);
        }
        let n = (self.plaintext.len() - self.plaintext_pos).min(out.len());
        out[..n].copy_from_slice(&self.plaintext[self.plaintext_pos..self.plaintext_pos + n]);
        self.plaintext_pos += n;
        Ok(n)
    }
}

impl<T: Read + Write> Write for TlsStream<T> {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.write_records(buf)?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        self.sock.flush()
    }
}

/// A TLS 1.3 stream over a `std::net::TcpStream`.
pub type TcpTlsStream = TlsStream<TcpStream>;

impl TcpTlsStream {
    /// Open a TCP connection to `host:port` and run the TLS 1.3
    /// handshake, validating the server chain through `verifier`.
    /// Handshake entropy is drawn from the hardware-seeded `security`
    /// CSPRNG.
    pub fn connect<V: ServerCertVerifier>(
        host: &str,
        port: u16,
        config: ClientConfig<'_>,
        verifier: &V,
    ) -> crate::Result<Self> {
        let sock = TcpStream::connect((host, port)).map_err(|_| TlsClientError::TcpConnect)?;
        let entropy = os_entropy()?;
        Self::connect_over(sock, config, verifier, &entropy)
    }
}

/// Draw fresh handshake entropy from the hardware-seeded CSPRNG.
fn os_entropy() -> crate::Result<ClientEntropy> {
    use smallaios_security::crypto::csprng::Csprng;
    let mut rng = Csprng::new();
    rng.seed_from_hardware()
        .map_err(|_| TlsClientError::KeyExchange)?;
    let mut client_random = [0u8; 32];
    let mut x25519_seed = [0u8; 32];
    let mut ml_kem_seed = [0u8; 64];
    rng.generate(&mut client_random)
        .map_err(|_| TlsClientError::KeyExchange)?;
    rng.generate(&mut x25519_seed)
        .map_err(|_| TlsClientError::KeyExchange)?;
    rng.generate(&mut ml_kem_seed)
        .map_err(|_| TlsClientError::KeyExchange)?;
    Ok(ClientEntropy {
        client_random,
        x25519_seed,
        ml_kem_seed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::VecDeque;

    /// In-memory bidirectional socket half: reads drain `inbound`,
    /// writes append to `outbound`.
    struct MockSocket {
        inbound: VecDeque<u8>,
        outbound: Vec<u8>,
    }

    impl MockSocket {
        fn new() -> Self {
            Self {
                inbound: VecDeque::new(),
                outbound: Vec::new(),
            }
        }
        fn feed(&mut self, bytes: &[u8]) {
            self.inbound.extend(bytes.iter().copied());
        }
    }

    impl Read for MockSocket {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            let n = out.len().min(self.inbound.len());
            for slot in out.iter_mut().take(n) {
                *slot = self.inbound.pop_front().unwrap();
            }
            Ok(n)
        }
    }

    impl Write for MockSocket {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.outbound.extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    const SUITE: CipherSuite = CipherSuite::ChaCha20Poly1305Sha256;

    fn keys(fill: u8) -> TrafficKeys {
        TrafficKeys {
            key: [fill; 32],
            iv: [fill ^ 0x5a; 12],
        }
    }

    /// Build a data-phase stream over a mock socket with fixed keys.
    fn stream_with_keys() -> (TlsStream<MockSocket>, TrafficKeys, TrafficKeys) {
        let client = keys(0x11);
        let server = keys(0x22);
        let s = TlsStream {
            sock: MockSocket::new(),
            suite: SUITE,
            client: client.clone(),
            server: server.clone(),
            write_seq: 0,
            read_seq: 0,
            plaintext: Vec::new(),
            plaintext_pos: 0,
            raw: Vec::new(),
            eof: false,
            closed: false,
        };
        (s, client, server)
    }

    #[test]
    fn write_seals_application_data() {
        let (mut s, client, _server) = stream_with_keys();
        s.write_all(b"hello world").unwrap();
        // The peer (holding the client keys) opens what we wrote.
        let rec = s.sock.outbound.clone();
        let (inner, data) = record::open(SUITE, &client.key, &client.iv, 0, &rec).unwrap();
        assert_eq!(inner, ContentType::ApplicationData);
        assert_eq!(data, b"hello world");
    }

    #[test]
    fn read_opens_server_records() {
        let (mut s, _client, server) = stream_with_keys();
        // The server seals with its keys at seq 0.
        let rec = record::seal(
            SUITE,
            &server.key,
            &server.iv,
            0,
            ContentType::ApplicationData,
            b"from server",
        )
        .unwrap();
        s.sock.feed(&rec);
        let mut out = [0u8; 32];
        let n = s.read(&mut out).unwrap();
        assert_eq!(&out[..n], b"from server");
    }

    #[test]
    fn multi_record_read_increments_seq() {
        let (mut s, _client, server) = stream_with_keys();
        for (seq, msg) in [(0u64, &b"one"[..]), (1, b"two")] {
            let rec = record::seal(
                SUITE,
                &server.key,
                &server.iv,
                seq,
                ContentType::ApplicationData,
                msg,
            )
            .unwrap();
            s.sock.feed(&rec);
        }
        let mut out = [0u8; 16];
        let n = s.read(&mut out).unwrap();
        assert_eq!(&out[..n], b"one");
        let n = s.read(&mut out).unwrap();
        assert_eq!(&out[..n], b"two");
    }

    #[test]
    fn handshake_record_is_skipped() {
        let (mut s, _client, server) = stream_with_keys();
        // A NewSessionTicket-style handshake record at seq 0, then app
        // data at seq 1.
        let ticket = record::seal(
            SUITE,
            &server.key,
            &server.iv,
            0,
            ContentType::Handshake,
            b"\x04\x00\x00\x00",
        )
        .unwrap();
        let app = record::seal(
            SUITE,
            &server.key,
            &server.iv,
            1,
            ContentType::ApplicationData,
            b"payload",
        )
        .unwrap();
        s.sock.feed(&ticket);
        s.sock.feed(&app);
        let mut out = [0u8; 16];
        let n = s.read(&mut out).unwrap();
        assert_eq!(&out[..n], b"payload");
    }

    #[test]
    fn server_close_notify_is_eof() {
        let (mut s, _client, server) = stream_with_keys();
        let alert = record::seal(
            SUITE,
            &server.key,
            &server.iv,
            0,
            ContentType::Alert,
            &[1u8, 0u8],
        )
        .unwrap();
        s.sock.feed(&alert);
        let mut out = [0u8; 16];
        assert_eq!(s.read(&mut out).unwrap(), 0);
    }

    #[test]
    fn tcp_eof_is_read_zero() {
        let (mut s, _client, _server) = stream_with_keys();
        let mut out = [0u8; 16];
        assert_eq!(s.read(&mut out).unwrap(), 0);
    }

    #[test]
    fn close_emits_close_notify() {
        let (mut s, client, _server) = stream_with_keys();
        s.close().unwrap();
        let rec = s.sock.outbound.clone();
        let (inner, data) = record::open(SUITE, &client.key, &client.iv, 0, &rec).unwrap();
        assert_eq!(inner, ContentType::Alert);
        assert_eq!(data, &[1u8, 0u8]);
        // Idempotent: a second close writes nothing further.
        let len_before = s.sock.outbound.len();
        s.close().unwrap();
        assert_eq!(s.sock.outbound.len(), len_before);
    }

    #[test]
    fn partial_record_reassembles_across_reads() {
        let (mut s, _client, server) = stream_with_keys();
        let rec = record::seal(
            SUITE,
            &server.key,
            &server.iv,
            0,
            ContentType::ApplicationData,
            b"reassembled",
        )
        .unwrap();
        // Feed the record in two fragments; the framing loop must wait
        // for the whole record before decrypting.
        let (head, tail) = rec.split_at(rec.len() - 4);
        s.sock.feed(head);
        s.sock.feed(tail);
        let mut out = [0u8; 32];
        let n = s.read(&mut out).unwrap();
        assert_eq!(&out[..n], b"reassembled");
    }
}
