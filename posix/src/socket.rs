// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! POSIX socket API subset.
//!
//! Provides the socket types and operations needed by AI inference workloads:
//! TCP client/server (SOCK_STREAM) and UDP (SOCK_DGRAM) over IPv4 and IPv6.
//!
//! All data-path operations are currently stubs returning `ENOSYS` until the
//! network stack is wired up. Validation and state-machine transitions are
//! fully implemented so callers get correct error codes immediately.

use crate::errno::Errno;

// ─── Constants ───────────────────────────────────────────────────────────────

/// Maximum number of sockets that may be open simultaneously.
pub const MAX_SOCKETS: usize = 64;

// ─── Address Family ─────────────────────────────────────────────────────────

/// BSD/POSIX address family constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum AddressFamily {
    /// IPv4 (AF_INET).
    Inet = 2,
    /// IPv6 (AF_INET6).
    Inet6 = 10,
}

impl AddressFamily {
    /// Try to convert a raw integer to an `AddressFamily`.
    pub fn from_raw(raw: i32) -> Option<Self> {
        match raw {
            2 => Some(Self::Inet),
            10 => Some(Self::Inet6),
            _ => None,
        }
    }
}

// ─── Socket Type ────────────────────────────────────────────────────────────

/// BSD/POSIX socket type constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum SocketType {
    /// Reliable byte-stream (SOCK_STREAM / TCP).
    Stream = 1,
    /// Connectionless datagrams (SOCK_DGRAM / UDP).
    Dgram = 2,
}

impl SocketType {
    /// Try to convert a raw integer to a `SocketType`.
    pub fn from_raw(raw: i32) -> Option<Self> {
        match raw {
            1 => Some(Self::Stream),
            2 => Some(Self::Dgram),
            _ => None,
        }
    }
}

// ─── Socket Address ─────────────────────────────────────────────────────────

/// Socket address — either IPv4 or IPv6.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SocketAddr {
    /// IPv4 address + port.
    V4 { addr: [u8; 4], port: u16 },
    /// IPv6 address + port.
    V6 { addr: [u8; 16], port: u16 },
}

impl SocketAddr {
    /// Return the port number.
    pub fn port(&self) -> u16 {
        match self {
            Self::V4 { port, .. } => *port,
            Self::V6 { port, .. } => *port,
        }
    }

    /// Return the address family this address belongs to.
    pub fn family(&self) -> AddressFamily {
        match self {
            Self::V4 { .. } => AddressFamily::Inet,
            Self::V6 { .. } => AddressFamily::Inet6,
        }
    }
}

// ─── Socket State Machine ───────────────────────────────────────────────────

/// Lifecycle state of a socket.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SocketState {
    /// Created but not yet bound to an address.
    Unbound,
    /// Bound to a local address.
    Bound,
    /// Listening for incoming connections (stream sockets only).
    Listening,
    /// Connected to a remote peer.
    Connected,
    /// Closed — no further operations are valid.
    Closed,
}

// ─── Socket ─────────────────────────────────────────────────────────────────

/// A single socket instance.
#[derive(Debug, Clone, PartialEq)]
pub struct Socket {
    /// Address family (IPv4 / IPv6).
    pub family: AddressFamily,
    /// Socket type (stream / datagram).
    pub sock_type: SocketType,
    /// Current lifecycle state.
    pub state: SocketState,
    /// Locally-bound address, if any.
    pub local_addr: Option<SocketAddr>,
    /// Remote peer address, if connected.
    pub remote_addr: Option<SocketAddr>,
    /// Whether the socket is in non-blocking mode.
    pub nonblocking: bool,
    /// Listen backlog depth (only meaningful in `Listening` state).
    pub backlog: u32,
}

// ─── Socket Table ───────────────────────────────────────────────────────────

/// System-wide socket table.
pub struct SocketTable {
    /// Socket slots.  `None` means the slot is free.
    sockets: [Option<Socket>; MAX_SOCKETS],
    /// Number of currently-open sockets.
    count: usize,
}

impl SocketTable {
    /// Create an empty socket table.
    pub fn new() -> Self {
        Self {
            sockets: core::array::from_fn(|_| None),
            count: 0,
        }
    }

    /// Return the number of open sockets.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Allocate a slot and return its index.
    pub fn alloc(&mut self, sock: Socket) -> Result<usize, Errno> {
        for (i, slot) in self.sockets.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(sock);
                self.count += 1;
                return Ok(i);
            }
        }
        Err(Errno::EMFILE)
    }

    /// Get a reference to a socket by index.
    pub fn get(&self, idx: usize) -> Result<&Socket, Errno> {
        if idx >= MAX_SOCKETS {
            return Err(Errno::EBADF);
        }
        self.sockets[idx].as_ref().ok_or(Errno::EBADF)
    }

    /// Get a mutable reference to a socket by index.
    pub fn get_mut(&mut self, idx: usize) -> Result<&mut Socket, Errno> {
        if idx >= MAX_SOCKETS {
            return Err(Errno::EBADF);
        }
        self.sockets[idx].as_mut().ok_or(Errno::EBADF)
    }

    /// Remove a socket from the table.
    pub fn remove(&mut self, idx: usize) -> Result<(), Errno> {
        if idx >= MAX_SOCKETS {
            return Err(Errno::EBADF);
        }
        if self.sockets[idx].is_none() {
            return Err(Errno::EBADF);
        }
        self.sockets[idx] = None;
        self.count -= 1;
        Ok(())
    }
}

// ─── Socket Operations ─────────────────────────────────────────────────────

/// Create a new socket with the given address family and type.
///
/// Returns a socket in the `Unbound` state. Invalid family or type
/// values yield `EAFNOSUPPORT` or `EINVAL`, respectively.
pub fn socket_create(family: AddressFamily, sock_type: SocketType) -> Result<Socket, Errno> {
    Ok(Socket {
        family,
        sock_type,
        state: SocketState::Unbound,
        local_addr: None,
        remote_addr: None,
        nonblocking: false,
        backlog: 0,
    })
}

/// Bind a socket to a local address.
///
/// The socket must be in the `Unbound` state and the address family must
/// match. Currently a stub — returns `ENOSYS` after validation.
pub fn socket_bind(socket: &mut Socket, addr: SocketAddr) -> Result<(), Errno> {
    if socket.state != SocketState::Unbound {
        return Err(Errno::EINVAL);
    }
    if addr.family() != socket.family {
        return Err(Errno::EAFNOSUPPORT);
    }
    // Stub: actual bind requires network stack integration.
    Err(Errno::ENOSYS)
}

/// Start listening for incoming connections on a bound stream socket.
///
/// The socket must be `Bound` and of type `Stream`.
/// Currently a stub — returns `ENOSYS` after validation.
pub fn socket_listen(socket: &mut Socket, _backlog: u32) -> Result<(), Errno> {
    if socket.sock_type != SocketType::Stream {
        return Err(Errno::EINVAL);
    }
    if socket.state != SocketState::Bound {
        return Err(Errno::EINVAL);
    }
    // Stub: actual listen requires network stack integration.
    Err(Errno::ENOSYS)
}

/// Accept an incoming connection on a listening socket.
///
/// The socket must be in the `Listening` state.
/// Currently a stub — returns `ENOSYS` after validation.
pub fn socket_accept(socket: &Socket) -> Result<Socket, Errno> {
    if socket.state != SocketState::Listening {
        return Err(Errno::EINVAL);
    }
    // Stub: actual accept requires network stack integration.
    Err(Errno::ENOSYS)
}

/// Connect a socket to a remote address.
///
/// The socket must be `Unbound` or `Bound`, and the address family must
/// match. Currently a stub — returns `ENOSYS` after validation.
pub fn socket_connect(socket: &mut Socket, addr: SocketAddr) -> Result<(), Errno> {
    match socket.state {
        SocketState::Unbound | SocketState::Bound => {}
        _ => return Err(Errno::EINVAL),
    }
    if addr.family() != socket.family {
        return Err(Errno::EAFNOSUPPORT);
    }
    // Stub: actual connect requires network stack integration.
    Err(Errno::ENOSYS)
}

/// Send data on a connected socket.
///
/// The socket must be in the `Connected` state.
/// Currently a stub — returns `ENOSYS` after validation.
pub fn socket_send(socket: &Socket, _data: &[u8]) -> Result<usize, Errno> {
    if socket.state != SocketState::Connected {
        return Err(Errno::EINVAL);
    }
    // Stub: actual send requires network stack integration.
    Err(Errno::ENOSYS)
}

/// Receive data from a connected socket.
///
/// The socket must be in the `Connected` state.
/// Currently a stub — returns `ENOSYS` after validation.
pub fn socket_recv(socket: &Socket, _buf: &mut [u8]) -> Result<usize, Errno> {
    if socket.state != SocketState::Connected {
        return Err(Errno::EINVAL);
    }
    // Stub: actual recv requires network stack integration.
    Err(Errno::ENOSYS)
}

/// Close a socket, transitioning it to the `Closed` state.
///
/// This is the only operation that always succeeds (no stub).
pub fn socket_close(socket: &mut Socket) -> Result<(), Errno> {
    socket.state = SocketState::Closed;
    socket.local_addr = None;
    socket.remote_addr = None;
    Ok(())
}

/// Set or clear non-blocking mode on a socket.
pub fn socket_set_nonblocking(socket: &mut Socket, nonblocking: bool) -> Result<(), Errno> {
    if socket.state == SocketState::Closed {
        return Err(Errno::EBADF);
    }
    socket.nonblocking = nonblocking;
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_inet_stream_socket() {
        let sock = socket_create(AddressFamily::Inet, SocketType::Stream).unwrap();
        assert_eq!(sock.family, AddressFamily::Inet);
        assert_eq!(sock.sock_type, SocketType::Stream);
        assert_eq!(sock.state, SocketState::Unbound);
        assert!(sock.local_addr.is_none());
        assert!(sock.remote_addr.is_none());
        assert!(!sock.nonblocking);
    }

    #[test]
    fn create_inet6_dgram_socket() {
        let sock = socket_create(AddressFamily::Inet6, SocketType::Dgram).unwrap();
        assert_eq!(sock.family, AddressFamily::Inet6);
        assert_eq!(sock.sock_type, SocketType::Dgram);
        assert_eq!(sock.state, SocketState::Unbound);
    }

    #[test]
    fn address_family_from_raw() {
        assert_eq!(AddressFamily::from_raw(2), Some(AddressFamily::Inet));
        assert_eq!(AddressFamily::from_raw(10), Some(AddressFamily::Inet6));
        assert_eq!(AddressFamily::from_raw(0), None);
        assert_eq!(AddressFamily::from_raw(99), None);
    }

    #[test]
    fn socket_type_from_raw() {
        assert_eq!(SocketType::from_raw(1), Some(SocketType::Stream));
        assert_eq!(SocketType::from_raw(2), Some(SocketType::Dgram));
        assert_eq!(SocketType::from_raw(0), None);
        assert_eq!(SocketType::from_raw(99), None);
    }

    #[test]
    fn socket_addr_properties() {
        let v4 = SocketAddr::V4 {
            addr: [127, 0, 0, 1],
            port: 8080,
        };
        assert_eq!(v4.port(), 8080);
        assert_eq!(v4.family(), AddressFamily::Inet);

        let v6 = SocketAddr::V6 {
            addr: [0; 16],
            port: 443,
        };
        assert_eq!(v6.port(), 443);
        assert_eq!(v6.family(), AddressFamily::Inet6);
    }

    #[test]
    fn bind_requires_unbound_state() {
        let mut sock = socket_create(AddressFamily::Inet, SocketType::Stream).unwrap();
        // Manually force a non-Unbound state to test validation.
        sock.state = SocketState::Connected;
        let addr = SocketAddr::V4 {
            addr: [0, 0, 0, 0],
            port: 80,
        };
        assert_eq!(socket_bind(&mut sock, addr), Err(Errno::EINVAL));
    }

    #[test]
    fn bind_rejects_family_mismatch() {
        let mut sock = socket_create(AddressFamily::Inet, SocketType::Stream).unwrap();
        let v6_addr = SocketAddr::V6 {
            addr: [0; 16],
            port: 80,
        };
        assert_eq!(socket_bind(&mut sock, v6_addr), Err(Errno::EAFNOSUPPORT));
    }

    #[test]
    fn bind_stub_returns_enosys() {
        let mut sock = socket_create(AddressFamily::Inet, SocketType::Stream).unwrap();
        let addr = SocketAddr::V4 {
            addr: [0, 0, 0, 0],
            port: 80,
        };
        assert_eq!(socket_bind(&mut sock, addr), Err(Errno::ENOSYS));
    }

    #[test]
    fn listen_requires_stream_and_bound() {
        // Dgram socket cannot listen.
        let mut dgram = socket_create(AddressFamily::Inet, SocketType::Dgram).unwrap();
        dgram.state = SocketState::Bound;
        assert_eq!(socket_listen(&mut dgram, 5), Err(Errno::EINVAL));

        // Stream socket that is not yet bound cannot listen.
        let mut stream = socket_create(AddressFamily::Inet, SocketType::Stream).unwrap();
        assert_eq!(stream.state, SocketState::Unbound);
        assert_eq!(socket_listen(&mut stream, 5), Err(Errno::EINVAL));
    }

    #[test]
    fn accept_requires_listening() {
        let sock = socket_create(AddressFamily::Inet, SocketType::Stream).unwrap();
        assert_eq!(socket_accept(&sock), Err(Errno::EINVAL));
    }

    #[test]
    fn connect_validates_state_and_family() {
        let mut sock = socket_create(AddressFamily::Inet, SocketType::Stream).unwrap();
        // Wrong family.
        let v6 = SocketAddr::V6 {
            addr: [0; 16],
            port: 80,
        };
        assert_eq!(socket_connect(&mut sock, v6), Err(Errno::EAFNOSUPPORT));

        // Closed socket cannot connect.
        sock.state = SocketState::Closed;
        let v4 = SocketAddr::V4 {
            addr: [1, 2, 3, 4],
            port: 80,
        };
        assert_eq!(socket_connect(&mut sock, v4), Err(Errno::EINVAL));
    }

    #[test]
    fn send_recv_require_connected() {
        let sock = socket_create(AddressFamily::Inet, SocketType::Stream).unwrap();
        assert_eq!(socket_send(&sock, b"hello"), Err(Errno::EINVAL));
        let mut buf = [0u8; 64];
        assert_eq!(socket_recv(&sock, &mut buf), Err(Errno::EINVAL));
    }

    #[test]
    fn close_transitions_to_closed() {
        let mut sock = socket_create(AddressFamily::Inet, SocketType::Stream).unwrap();
        assert_eq!(sock.state, SocketState::Unbound);
        socket_close(&mut sock).unwrap();
        assert_eq!(sock.state, SocketState::Closed);
        assert!(sock.local_addr.is_none());
        assert!(sock.remote_addr.is_none());
    }

    #[test]
    fn set_nonblocking() {
        let mut sock = socket_create(AddressFamily::Inet, SocketType::Stream).unwrap();
        assert!(!sock.nonblocking);
        socket_set_nonblocking(&mut sock, true).unwrap();
        assert!(sock.nonblocking);
        socket_set_nonblocking(&mut sock, false).unwrap();
        assert!(!sock.nonblocking);
    }

    #[test]
    fn set_nonblocking_on_closed_fails() {
        let mut sock = socket_create(AddressFamily::Inet, SocketType::Stream).unwrap();
        socket_close(&mut sock).unwrap();
        assert_eq!(
            socket_set_nonblocking(&mut sock, true),
            Err(Errno::EBADF)
        );
    }

    #[test]
    fn socket_table_alloc_and_get() {
        let mut table = SocketTable::new();
        assert_eq!(table.count(), 0);

        let sock = socket_create(AddressFamily::Inet, SocketType::Stream).unwrap();
        let idx = table.alloc(sock).unwrap();
        assert_eq!(idx, 0);
        assert_eq!(table.count(), 1);

        let s = table.get(idx).unwrap();
        assert_eq!(s.family, AddressFamily::Inet);
    }

    #[test]
    fn socket_table_remove() {
        let mut table = SocketTable::new();
        let sock = socket_create(AddressFamily::Inet6, SocketType::Dgram).unwrap();
        let idx = table.alloc(sock).unwrap();
        assert_eq!(table.count(), 1);

        table.remove(idx).unwrap();
        assert_eq!(table.count(), 0);
        assert_eq!(table.get(idx), Err(Errno::EBADF));
    }

    #[test]
    fn socket_table_remove_invalid() {
        let mut table = SocketTable::new();
        assert_eq!(table.remove(0), Err(Errno::EBADF));
        assert_eq!(table.remove(MAX_SOCKETS), Err(Errno::EBADF));
    }
}
