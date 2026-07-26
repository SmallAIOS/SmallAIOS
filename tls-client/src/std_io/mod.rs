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
//!
//! Every socket-blocking stage is bounded by [`TlsTimeouts`] so an
//! unresponsive or silently-dropping peer cannot stall the caller
//! indefinitely — see that type for what is and is not covered.

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
use std::net::{TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};

/// Bytes pulled from the socket per `read` syscall.
const READ_CHUNK: usize = 8192;

/// Default ceiling on the TCP handshake, across every address the
/// host name resolves to.
const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

/// Default ceiling on a single socket read or write, applied to both
/// the TLS handshake flights and the data phase.
const DEFAULT_IO_TIMEOUT: Duration = Duration::from_secs(30);

/// Per-stage deadlines for [`TcpTlsStream`].
///
/// Without these, a peer that completes the TCP handshake and then
/// goes silent parks the caller in `read` forever; for the audit
/// exporter that means the export pipeline stops making progress with
/// no error to back off from. Each field bounds one stage:
///
/// - `connect` — total budget for the TCP handshake. When the host
///   name resolves to several addresses the budget is shared across
///   all of them, so the worst case is `connect`, not `connect` per
///   address.
/// - `read` — one `read` syscall, via `SO_RCVTIMEO`. Applies to the
///   handshake flights *and* every post-handshake record read.
/// - `write` — one `write` or `flush`, via `SO_SNDTIMEO`. Same
///   coverage.
///
/// `Duration::ZERO` disables that stage's timeout (the platform
/// rejects a zero socket timeout, and `connect_timeout` rejects a
/// zero duration outright, so zero is read as "unbounded" rather
/// than passed down).
///
/// **Not covered:** name resolution. `ToSocketAddrs` offers no
/// deadline in `std`, so a wedged resolver still blocks — bounding it
/// needs a resolver thread, which is out of scope for a `#![no_std]`
/// crate's thin `std` shim. Pass an IP literal to skip resolution
/// entirely.
///
/// A read or write that trips its deadline surfaces as
/// [`TlsClientError::Io`] during the handshake, and as an
/// `io::Error` of kind `WouldBlock`/`TimedOut` in the data phase; a
/// `connect` that trips surfaces as [`TlsClientError::TcpConnect`].
/// Both map to the transient class, which is the correct retry
/// decision for a timeout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TlsTimeouts {
    /// Total budget for the TCP handshake.
    pub connect: Duration,
    /// Per-`read` budget (`SO_RCVTIMEO`).
    pub read: Duration,
    /// Per-`write`/`flush` budget (`SO_SNDTIMEO`).
    pub write: Duration,
}

impl Default for TlsTimeouts {
    fn default() -> Self {
        Self {
            connect: DEFAULT_CONNECT_TIMEOUT,
            read: DEFAULT_IO_TIMEOUT,
            write: DEFAULT_IO_TIMEOUT,
        }
    }
}

impl TlsTimeouts {
    /// Every stage unbounded. Only for tests and callers that supply
    /// their own deadline mechanism; production paths should take
    /// [`Default`].
    pub const fn unbounded() -> Self {
        Self {
            connect: Duration::ZERO,
            read: Duration::ZERO,
            write: Duration::ZERO,
        }
    }
}

/// Normalise a caller duration into the `Option` the socket setters
/// want: `ZERO` means "no deadline".
fn opt_timeout(d: Duration) -> Option<Duration> {
    if d.is_zero() {
        None
    } else {
        Some(d)
    }
}

/// Open a TCP connection to `host:port`, spending at most
/// `budget` in total across every resolved address.
///
/// `TcpStream::connect` has no deadline, so this resolves first and
/// then dials each candidate with `connect_timeout`, charging the
/// elapsed time against a single shared deadline. Without the shared
/// deadline a name resolving to N addresses would take N × budget in
/// the worst case.
fn dial(host: &str, port: u16, budget: Duration) -> crate::Result<TcpStream> {
    let addrs = (host, port)
        .to_socket_addrs()
        .map_err(|_| TlsClientError::TcpConnect)?;

    // No budget: fall back to the unbounded connect, which also
    // re-uses std's own address iteration.
    if budget.is_zero() {
        return TcpStream::connect((host, port)).map_err(|_| TlsClientError::TcpConnect);
    }

    let deadline = Instant::now() + budget;
    for addr in addrs {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            // Budget exhausted mid-iteration; stop rather than dial
            // the remaining addresses past the caller's deadline.
            break;
        }
        if let Ok(sock) = TcpStream::connect_timeout(&addr, remaining) {
            return Ok(sock);
        }
    }
    Err(TlsClientError::TcpConnect)
}

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
    ///
    /// Uses [`TlsTimeouts::default`]; call
    /// [`connect_with_timeouts`](Self::connect_with_timeouts) to set
    /// your own deadlines.
    pub fn connect<V: ServerCertVerifier>(
        host: &str,
        port: u16,
        config: ClientConfig<'_>,
        verifier: &V,
    ) -> crate::Result<Self> {
        Self::connect_with_timeouts(host, port, config, verifier, TlsTimeouts::default())
    }

    /// [`connect`](Self::connect) with explicit per-stage deadlines.
    ///
    /// The read/write deadlines are installed on the socket *before*
    /// the first handshake byte moves, so they bound the handshake as
    /// well as the data phase that outlives this call.
    ///
    /// One consequence worth naming: a `write` deadline can expire
    /// after a partial write, leaving a half-written TLS record on
    /// the wire. The record stream is then unusable and the caller
    /// must drop the connection rather than retry on the same stream
    /// — which is what the audit exporter does, since a timeout
    /// surfaces as a transient error that tears the connection down
    /// and reconnects. Preferable to blocking forever, but it does
    /// mean a `write` timeout is connection-fatal, not recoverable
    /// in place.
    pub fn connect_with_timeouts<V: ServerCertVerifier>(
        host: &str,
        port: u16,
        config: ClientConfig<'_>,
        verifier: &V,
        timeouts: TlsTimeouts,
    ) -> crate::Result<Self> {
        let sock = dial(host, port, timeouts.connect)?;
        sock.set_read_timeout(opt_timeout(timeouts.read))
            .map_err(|_| TlsClientError::TcpConnect)?;
        sock.set_write_timeout(opt_timeout(timeouts.write))
            .map_err(|_| TlsClientError::TcpConnect)?;
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

    // ---- TlsTimeouts (CodeRabbit review on the Phase 8 PR: the
    // connect path was previously unbounded at every stage) ----

    use crate::cert::{LeafPublicKey, ServerCertVerifier};
    // `no_std` crate: `ToString` is not in the prelude.
    use alloc::string::ToString;
    use std::net::{TcpListener, TcpStream as StdTcpStream};

    /// Refuses every chain. The timeout tests never reach chain
    /// verification — the socket deadline fires first — so the
    /// verdict only has to be deterministic, not correct.
    struct RejectAll;

    impl ServerCertVerifier for RejectAll {
        fn verify_chain(
            &self,
            _certs: &[Vec<u8>],
            _server_name: Option<&str>,
        ) -> crate::Result<LeafPublicKey> {
            Err(TlsClientError::ChainUntrusted)
        }
    }

    #[test]
    fn default_timeouts_are_bounded() {
        let t = TlsTimeouts::default();
        assert!(!t.connect.is_zero(), "connect must be bounded by default");
        assert!(!t.read.is_zero(), "read must be bounded by default");
        assert!(!t.write.is_zero(), "write must be bounded by default");
    }

    #[test]
    fn zero_duration_reads_as_unbounded() {
        assert_eq!(opt_timeout(Duration::ZERO), None);
        assert_eq!(
            opt_timeout(Duration::from_millis(250)),
            Some(Duration::from_millis(250))
        );
        let u = TlsTimeouts::unbounded();
        assert_eq!(opt_timeout(u.read), None);
        assert_eq!(opt_timeout(u.write), None);
    }

    /// The regression this fix exists for: a peer that completes the
    /// TCP handshake and then never sends a byte must fail on the
    /// read deadline instead of parking the caller forever.
    #[test]
    fn silent_peer_trips_the_read_deadline() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        // Accept, then hold the connection open and say nothing. The
        // guard keeps the accepted socket alive so the client sees
        // silence rather than EOF (EOF would fail via a different
        // path and prove nothing about the timeout).
        let accepted = std::thread::spawn(move || {
            let (sock, _) = listener.accept().unwrap();
            std::thread::sleep(Duration::from_secs(5));
            drop(sock);
        });

        let timeouts = TlsTimeouts {
            connect: Duration::from_secs(5),
            read: Duration::from_millis(300),
            write: Duration::from_secs(5),
        };
        let config = ClientConfig {
            server_name: None,
            require_pqc: false,
        };

        let start = Instant::now();
        let res = TcpTlsStream::connect_with_timeouts(
            &addr.ip().to_string(),
            addr.port(),
            config,
            &RejectAll,
            timeouts,
        );
        let elapsed = start.elapsed();

        // A timed-out handshake read is an I/O fault, which
        // `tls_retry_class` treats as transient.
        assert_eq!(res.err(), Some(TlsClientError::Io));
        // Generous ceiling: entropy draw and the ClientHello build
        // both precede the read, and CI runners are slow. The point
        // is that it returns at all rather than hanging.
        assert!(
            elapsed < Duration::from_secs(4),
            "read deadline did not bound the handshake: {elapsed:?}"
        );

        accepted.join().unwrap();
    }

    /// A refused connection is still a fast `TcpConnect`, not a
    /// wait-out-the-whole-budget stall.
    #[test]
    fn refused_port_fails_fast_as_tcp_connect() {
        // Bind then drop to get a port nothing is listening on.
        let addr = {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap()
        };
        // Confirm the port really is closed before asserting on it;
        // otherwise a racing bind would make this test lie.
        if StdTcpStream::connect(addr).is_ok() {
            return;
        }

        let config = ClientConfig {
            server_name: None,
            require_pqc: false,
        };
        let start = Instant::now();
        let res = TcpTlsStream::connect_with_timeouts(
            &addr.ip().to_string(),
            addr.port(),
            config,
            &RejectAll,
            TlsTimeouts {
                connect: Duration::from_secs(10),
                ..TlsTimeouts::default()
            },
        );
        assert_eq!(res.err(), Some(TlsClientError::TcpConnect));
        assert!(
            start.elapsed() < Duration::from_secs(5),
            "refusal should not consume the connect budget"
        );
    }

    #[test]
    fn dial_reaches_a_live_listener_within_budget() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let sock = dial(&addr.ip().to_string(), addr.port(), Duration::from_secs(5))
            .expect("dial to a live listener");
        assert_eq!(sock.peer_addr().unwrap(), addr);
    }

    #[test]
    fn dial_with_zero_budget_still_connects() {
        // ZERO means unbounded, not "fail immediately" — a zero
        // socket timeout is rejected by the platform, so the code
        // must route around `connect_timeout` entirely.
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let sock = dial(&addr.ip().to_string(), addr.port(), Duration::ZERO)
            .expect("zero budget means unbounded");
        assert_eq!(sock.peer_addr().unwrap(), addr);
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
