// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! DHCP client for SmallAIOS.
//!
//! Implements RFC 2131 Dynamic Host Configuration Protocol client with a
//! full state machine (INIT -> SELECTING -> REQUESTING -> BOUND -> RENEWING).
//! All buffers are fixed-size for `#![no_std]` compatibility.
//!
//! Provides [`DhcpClient`] for driving the DHCP handshake, [`DhcpMessage`]
//! for constructing and parsing DHCP packets, and [`DhcpLease`] for storing
//! obtained lease information.

// ---------------------------------------------------------------------------
// Message type constants (RFC 2131 section 9.6)
// ---------------------------------------------------------------------------

/// DHCPDISCOVER — client broadcasts to locate servers.
pub const MSG_TYPE_DISCOVER: u8 = 1;
/// DHCPOFFER — server responds with an IP offer.
pub const MSG_TYPE_OFFER: u8 = 2;
/// DHCPREQUEST — client requests offered parameters.
pub const MSG_TYPE_REQUEST: u8 = 3;
/// DHCPDECLINE — client declines offered address.
pub const MSG_TYPE_DECLINE: u8 = 4;
/// DHCPACK — server confirms lease.
pub const MSG_TYPE_ACK: u8 = 5;
/// DHCPNAK — server rejects request.
pub const MSG_TYPE_NAK: u8 = 6;
/// DHCPRELEASE — client relinquishes lease.
pub const MSG_TYPE_RELEASE: u8 = 7;

// ---------------------------------------------------------------------------
// DHCP option codes (RFC 2132)
// ---------------------------------------------------------------------------

/// Subnet mask option.
pub const OPT_SUBNET_MASK: u8 = 1;
/// Router (default gateway) option.
pub const OPT_ROUTER: u8 = 3;
/// Domain Name Server option.
pub const OPT_DNS: u8 = 6;
/// IP address lease time in seconds.
pub const OPT_LEASE_TIME: u8 = 51;
/// DHCP message type option.
pub const OPT_MSG_TYPE: u8 = 53;
/// Server identifier (IP address of the DHCP server).
pub const OPT_SERVER_ID: u8 = 54;
/// Parameter request list.
pub const OPT_PARAM_REQUEST: u8 = 55;
/// End of options marker.
pub const OPT_END: u8 = 255;
/// Pad option (single byte, no length).
pub const OPT_PAD: u8 = 0;

// ---------------------------------------------------------------------------
// Protocol constants
// ---------------------------------------------------------------------------

/// DHCP magic cookie (RFC 2131 section 3).
pub const MAGIC_COOKIE: u32 = 0x6382_5363;

/// Boot request (client to server).
pub const BOOT_REQUEST: u8 = 1;
/// Boot reply (server to client).
pub const BOOT_REPLY: u8 = 2;

/// Hardware type: Ethernet (10 Mb).
pub const HTYPE_ETHERNET: u8 = 1;
/// Hardware address length for Ethernet.
pub const HLEN_ETHERNET: u8 = 6;

/// Fixed portion of a DHCP message (before options).
pub const FIXED_HEADER_LEN: usize = 236;
/// Magic cookie length.
pub const MAGIC_COOKIE_LEN: usize = 4;
/// Maximum options buffer size.
pub const MAX_OPTIONS_LEN: usize = 312;
/// Minimum valid DHCP message length (header + cookie + end option).
pub const MIN_MSG_LEN: usize = FIXED_HEADER_LEN + MAGIC_COOKIE_LEN;

/// DHCP client port.
pub const CLIENT_PORT: u16 = 68;
/// DHCP server port.
pub const SERVER_PORT: u16 = 67;

/// Maximum number of retries before giving up.
pub const MAX_RETRIES: u8 = 3;
/// Initial retry interval in seconds (doubles each retry).
pub const INITIAL_RETRY_SECS: u32 = 2;
/// Overall timeout in seconds.
pub const TIMEOUT_SECS: u32 = 10;
/// Renewal threshold as a percentage of lease time (T1 = 50%).
pub const RENEWAL_PERCENT: u32 = 50;

// ---------------------------------------------------------------------------
// DhcpError
// ---------------------------------------------------------------------------

/// Errors that can occur during DHCP operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DhcpError {
    /// Packet data is shorter than the minimum DHCP message length.
    PacketTooShort,
    /// Provided output buffer is too small.
    BufferTooSmall,
    /// Magic cookie does not match the expected value.
    InvalidMagicCookie,
    /// A required DHCP option is missing from the message.
    MissingOption,
    /// An option has an invalid or unexpected length.
    InvalidOptionLength,
    /// Message type field is not a recognized DHCP message type.
    InvalidMessageType,
    /// The DHCP state machine received a message in an unexpected state.
    InvalidState,
    /// Retry limit exceeded without receiving a valid response.
    RetryExhausted,
    /// Overall DHCP handshake timed out.
    Timeout,
    /// Options buffer overflow — too many options for the fixed buffer.
    OptionsOverflow,
}

impl core::fmt::Display for DhcpError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            DhcpError::PacketTooShort => write!(f, "dhcp: packet too short"),
            DhcpError::BufferTooSmall => write!(f, "dhcp: buffer too small"),
            DhcpError::InvalidMagicCookie => write!(f, "dhcp: invalid magic cookie"),
            DhcpError::MissingOption => write!(f, "dhcp: missing required option"),
            DhcpError::InvalidOptionLength => write!(f, "dhcp: invalid option length"),
            DhcpError::InvalidMessageType => write!(f, "dhcp: invalid message type"),
            DhcpError::InvalidState => write!(f, "dhcp: invalid state"),
            DhcpError::RetryExhausted => write!(f, "dhcp: retry limit exhausted"),
            DhcpError::Timeout => write!(f, "dhcp: timeout"),
            DhcpError::OptionsOverflow => write!(f, "dhcp: options buffer overflow"),
        }
    }
}

// ---------------------------------------------------------------------------
// DhcpOption — parsed TLV option
// ---------------------------------------------------------------------------

/// A single parsed DHCP option (type-length-value).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DhcpOption {
    /// Option code (type).
    pub code: u8,
    /// Option data length.
    pub len: u8,
    /// Option value bytes (zero-padded to fixed size).
    pub data: [u8; 255],
}

impl DhcpOption {
    /// Create a new option with the given code and data slice.
    pub fn new(code: u8, data: &[u8]) -> Self {
        let mut opt = DhcpOption {
            code,
            len: data.len() as u8,
            data: [0u8; 255],
        };
        opt.data[..data.len()].copy_from_slice(data);
        opt
    }
}

// ---------------------------------------------------------------------------
// DhcpMessage
// ---------------------------------------------------------------------------

/// A DHCP message with fixed-size buffers.
///
/// The message layout follows RFC 2131 section 2:
/// - Bytes 0..236: fixed fields (op, htype, hlen, hops, xid, secs, flags,
///   ciaddr, yiaddr, siaddr, giaddr, chaddr, sname, file)
/// - Bytes 236..240: magic cookie (0x63825363)
/// - Bytes 240..: options (TLV encoded, terminated with option 255)
#[derive(Clone)]
pub struct DhcpMessage {
    /// Message op code: 1 = BOOTREQUEST, 2 = BOOTREPLY.
    pub op: u8,
    /// Hardware address type (1 = Ethernet).
    pub htype: u8,
    /// Hardware address length (6 for Ethernet).
    pub hlen: u8,
    /// Number of relay agent hops.
    pub hops: u8,
    /// Transaction ID chosen by the client.
    pub xid: u32,
    /// Seconds elapsed since the client began the DHCP process.
    pub secs: u16,
    /// Flags (bit 0 = broadcast flag).
    pub flags: u16,
    /// Client IP address (filled in if client is bound).
    pub ciaddr: [u8; 4],
    /// "Your" (client) IP address offered by the server.
    pub yiaddr: [u8; 4],
    /// Server IP address.
    pub siaddr: [u8; 4],
    /// Relay agent IP address.
    pub giaddr: [u8; 4],
    /// Client hardware address (MAC in first 6 bytes for Ethernet).
    pub chaddr: [u8; 16],
    /// Optional server host name.
    pub sname: [u8; 64],
    /// Boot file name.
    pub file: [u8; 128],
    /// Raw options bytes (TLV encoded, without magic cookie).
    pub options: [u8; MAX_OPTIONS_LEN],
    /// Number of valid bytes in the options buffer.
    pub options_len: usize,
}

impl PartialEq for DhcpMessage {
    fn eq(&self, other: &Self) -> bool {
        self.op == other.op
            && self.htype == other.htype
            && self.hlen == other.hlen
            && self.hops == other.hops
            && self.xid == other.xid
            && self.secs == other.secs
            && self.flags == other.flags
            && self.ciaddr == other.ciaddr
            && self.yiaddr == other.yiaddr
            && self.siaddr == other.siaddr
            && self.giaddr == other.giaddr
            && self.chaddr == other.chaddr
            && self.sname[..] == other.sname[..]
            && self.file[..] == other.file[..]
            && self.options_len == other.options_len
            && self.options[..self.options_len] == other.options[..other.options_len]
    }
}

impl Eq for DhcpMessage {}

impl core::fmt::Debug for DhcpMessage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DhcpMessage")
            .field("op", &self.op)
            .field("htype", &self.htype)
            .field("hlen", &self.hlen)
            .field("xid", &self.xid)
            .field("yiaddr", &self.yiaddr)
            .field("siaddr", &self.siaddr)
            .field("options_len", &self.options_len)
            .finish()
    }
}

impl Default for DhcpMessage {
    fn default() -> Self {
        DhcpMessage {
            op: 0,
            htype: HTYPE_ETHERNET,
            hlen: HLEN_ETHERNET,
            hops: 0,
            xid: 0,
            secs: 0,
            flags: 0,
            ciaddr: [0; 4],
            yiaddr: [0; 4],
            siaddr: [0; 4],
            giaddr: [0; 4],
            chaddr: [0; 16],
            sname: [0; 64],
            file: [0; 128],
            options: [0; MAX_OPTIONS_LEN],
            options_len: 0,
        }
    }
}

impl DhcpMessage {
    /// Serialize this DHCP message into a byte buffer.
    ///
    /// Returns the number of bytes written on success, or
    /// [`DhcpError::BufferTooSmall`] if `buf` is too small.
    pub fn serialize(&self, buf: &mut [u8]) -> Result<usize, DhcpError> {
        let total = FIXED_HEADER_LEN + MAGIC_COOKIE_LEN + self.options_len;
        if buf.len() < total {
            return Err(DhcpError::BufferTooSmall);
        }

        // Fixed header fields
        buf[0] = self.op;
        buf[1] = self.htype;
        buf[2] = self.hlen;
        buf[3] = self.hops;
        buf[4..8].copy_from_slice(&self.xid.to_be_bytes());
        buf[8..10].copy_from_slice(&self.secs.to_be_bytes());
        buf[10..12].copy_from_slice(&self.flags.to_be_bytes());
        buf[12..16].copy_from_slice(&self.ciaddr);
        buf[16..20].copy_from_slice(&self.yiaddr);
        buf[20..24].copy_from_slice(&self.siaddr);
        buf[24..28].copy_from_slice(&self.giaddr);
        buf[28..44].copy_from_slice(&self.chaddr);
        buf[44..108].copy_from_slice(&self.sname);
        buf[108..236].copy_from_slice(&self.file);

        // Magic cookie
        buf[236..240].copy_from_slice(&MAGIC_COOKIE.to_be_bytes());

        // Options
        buf[240..240 + self.options_len].copy_from_slice(&self.options[..self.options_len]);

        Ok(total)
    }

    /// Parse a DHCP message from raw bytes.
    ///
    /// Validates the magic cookie and copies the fixed fields and options
    /// into a new [`DhcpMessage`].
    pub fn parse(data: &[u8]) -> Result<Self, DhcpError> {
        if data.len() < MIN_MSG_LEN {
            return Err(DhcpError::PacketTooShort);
        }

        // Validate magic cookie
        let cookie = u32::from_be_bytes([data[236], data[237], data[238], data[239]]);
        if cookie != MAGIC_COOKIE {
            return Err(DhcpError::InvalidMagicCookie);
        }

        let mut ciaddr = [0u8; 4];
        let mut yiaddr = [0u8; 4];
        let mut siaddr = [0u8; 4];
        let mut giaddr = [0u8; 4];
        let mut chaddr = [0u8; 16];
        let mut sname = [0u8; 64];
        let mut file = [0u8; 128];
        let mut options = [0u8; MAX_OPTIONS_LEN];

        ciaddr.copy_from_slice(&data[12..16]);
        yiaddr.copy_from_slice(&data[16..20]);
        siaddr.copy_from_slice(&data[20..24]);
        giaddr.copy_from_slice(&data[24..28]);
        chaddr.copy_from_slice(&data[28..44]);
        sname.copy_from_slice(&data[44..108]);
        file.copy_from_slice(&data[108..236]);

        let opts_start = FIXED_HEADER_LEN + MAGIC_COOKIE_LEN;
        let opts_len = core::cmp::min(data.len() - opts_start, MAX_OPTIONS_LEN);
        options[..opts_len].copy_from_slice(&data[opts_start..opts_start + opts_len]);

        let msg = DhcpMessage {
            op: data[0],
            htype: data[1],
            hlen: data[2],
            hops: data[3],
            xid: u32::from_be_bytes([data[4], data[5], data[6], data[7]]),
            secs: u16::from_be_bytes([data[8], data[9]]),
            flags: u16::from_be_bytes([data[10], data[11]]),
            ciaddr,
            yiaddr,
            siaddr,
            giaddr,
            chaddr,
            sname,
            file,
            options,
            options_len: opts_len,
        };

        Ok(msg)
    }

    /// Append a TLV option to the options buffer.
    ///
    /// Returns [`DhcpError::OptionsOverflow`] if there is not enough room.
    pub fn add_option(&mut self, code: u8, data: &[u8]) -> Result<(), DhcpError> {
        let needed = 2 + data.len(); // type + length + data
        if self.options_len + needed > MAX_OPTIONS_LEN {
            return Err(DhcpError::OptionsOverflow);
        }
        self.options[self.options_len] = code;
        self.options[self.options_len + 1] = data.len() as u8;
        self.options[self.options_len + 2..self.options_len + 2 + data.len()].copy_from_slice(data);
        self.options_len += needed;
        Ok(())
    }

    /// Append the END option marker (255) to the options buffer.
    pub fn add_end_option(&mut self) -> Result<(), DhcpError> {
        if self.options_len >= MAX_OPTIONS_LEN {
            return Err(DhcpError::OptionsOverflow);
        }
        self.options[self.options_len] = OPT_END;
        self.options_len += 1;
        Ok(())
    }

    /// Extract the DHCP message type from the options.
    pub fn get_message_type(&self) -> Option<u8> {
        get_option_value(&self.options[..self.options_len], OPT_MSG_TYPE).map(|opt| opt.data[0])
    }

    /// Extract a 4-byte IPv4 address from a given option code.
    pub fn get_option_ipv4(&self, code: u8) -> Option<[u8; 4]> {
        get_option_value(&self.options[..self.options_len], code).and_then(|opt| {
            if opt.len >= 4 {
                Some([opt.data[0], opt.data[1], opt.data[2], opt.data[3]])
            } else {
                None
            }
        })
    }

    /// Extract lease time (u32 seconds) from options.
    pub fn get_lease_time(&self) -> Option<u32> {
        get_option_value(&self.options[..self.options_len], OPT_LEASE_TIME).and_then(|opt| {
            if opt.len >= 4 {
                Some(u32::from_be_bytes([
                    opt.data[0],
                    opt.data[1],
                    opt.data[2],
                    opt.data[3],
                ]))
            } else {
                None
            }
        })
    }

    /// Extract the server identifier from options.
    pub fn get_server_id(&self) -> Option<[u8; 4]> {
        self.get_option_ipv4(OPT_SERVER_ID)
    }
}

// ---------------------------------------------------------------------------
// TLV options parser
// ---------------------------------------------------------------------------

/// Iterate over DHCP options in a raw options buffer.
///
/// Each yielded item is `Ok((option_code, option_value_bytes))` for a
/// well-formed option, or `Err(DhcpError::InvalidOptionLength)` if the
/// buffer is truncated mid-option. Stops at the END option (0xff) or end
/// of buffer. Skips PAD options (0x00).
///
/// Once an error is yielded, the iterator terminates.
fn iter_dhcp_options(data: &[u8]) -> impl Iterator<Item = Result<(u8, &[u8]), DhcpError>> + '_ {
    let mut pos = 0;
    let mut done = false;
    core::iter::from_fn(move || {
        if done {
            return None;
        }
        while pos < data.len() {
            let code = data[pos];

            // PAD option — skip
            if code == OPT_PAD {
                pos += 1;
                continue;
            }
            // END option — stop
            if code == OPT_END {
                done = true;
                return None;
            }
            // Other options have a length byte
            if pos + 1 >= data.len() {
                done = true;
                return Some(Err(DhcpError::InvalidOptionLength));
            }
            let len = data[pos + 1] as usize;
            if pos + 2 + len > data.len() {
                done = true;
                return Some(Err(DhcpError::InvalidOptionLength));
            }
            let value = &data[pos + 2..pos + 2 + len];
            pos += 2 + len;
            return Some(Ok((code, value)));
        }
        None
    })
}

/// Parse all TLV options from a raw options byte slice.
///
/// Returns options in a fixed-size array. Stops at the END option (255)
/// or when the buffer is exhausted. Pad options (0) are skipped.
pub fn parse_options(data: &[u8]) -> Result<([DhcpOption; 16], usize), DhcpError> {
    let mut opts = [DhcpOption {
        code: 0,
        len: 0,
        data: [0u8; 255],
    }; 16];
    let mut count = 0;

    for item in iter_dhcp_options(data) {
        let (code, value) = item?;
        if count < 16 {
            opts[count] = DhcpOption::new(code, value);
            count += 1;
        }
    }

    Ok((opts, count))
}

/// Find a specific option by code in a raw options byte slice.
fn get_option_value(data: &[u8], code: u8) -> Option<DhcpOption> {
    iter_dhcp_options(data)
        .filter_map(Result::ok)
        .find(|(c, _)| *c == code)
        .map(|(c, v)| DhcpOption::new(c, v))
}

// ---------------------------------------------------------------------------
// DhcpState
// ---------------------------------------------------------------------------

/// DHCP client state machine states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DhcpState {
    /// Initial state — no DHCP activity started.
    Init,
    /// DISCOVER sent, waiting for OFFER.
    Selecting,
    /// REQUEST sent, waiting for ACK.
    Requesting,
    /// Lease obtained and active.
    Bound,
    /// Lease renewal in progress (T1 elapsed).
    Renewing,
    /// Lease has expired.
    Expired,
}

// ---------------------------------------------------------------------------
// DhcpLease
// ---------------------------------------------------------------------------

/// Information obtained from a successful DHCP lease.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DhcpLease {
    /// Assigned IPv4 address.
    pub ip_addr: [u8; 4],
    /// Subnet mask.
    pub subnet_mask: [u8; 4],
    /// Default gateway.
    pub gateway: [u8; 4],
    /// DNS server address.
    pub dns: [u8; 4],
    /// Lease duration in seconds.
    pub lease_time_secs: u32,
    /// Server that granted the lease.
    pub server_id: [u8; 4],
}

// ---------------------------------------------------------------------------
// RetryState
// ---------------------------------------------------------------------------

/// Tracks exponential backoff retry state.
#[derive(Debug, Clone, Copy)]
struct RetryState {
    /// Number of retries attempted so far.
    attempts: u8,
    /// Next retry interval in seconds.
    next_interval_secs: u32,
    /// Total elapsed time in seconds since the process started.
    total_elapsed_secs: u32,
}

impl RetryState {
    fn new() -> Self {
        RetryState {
            attempts: 0,
            next_interval_secs: INITIAL_RETRY_SECS,
            total_elapsed_secs: 0,
        }
    }

    /// Check if a retry is allowed. If so, advance the state and return
    /// the interval to wait. Returns `None` if retries are exhausted or
    /// the overall timeout would be exceeded.
    fn try_retry(&mut self) -> Option<u32> {
        if self.attempts >= MAX_RETRIES {
            return None;
        }
        if self.total_elapsed_secs + self.next_interval_secs > TIMEOUT_SECS {
            return None;
        }
        let interval = self.next_interval_secs;
        self.attempts += 1;
        self.total_elapsed_secs += interval;
        self.next_interval_secs *= 2; // exponential backoff
        Some(interval)
    }
}

// ---------------------------------------------------------------------------
// DhcpClient
// ---------------------------------------------------------------------------

/// A DHCP client implementing the RFC 2131 state machine.
///
/// # Usage
///
/// ```ignore
/// let mut client = DhcpClient::new([0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01]);
/// let discover = client.start_discover();
/// // Send discover as UDP broadcast to 255.255.255.255:67 from 0.0.0.0:68
/// // ... receive OFFER ...
/// if let Some(request) = client.handle_offer(&offer_msg) {
///     // Send request ...
///     // ... receive ACK ...
///     if let Some(lease) = client.handle_ack(&ack_msg) {
///         // Interface is now configured
///     }
/// }
/// ```
pub struct DhcpClient {
    /// Current state machine state.
    pub state: DhcpState,
    /// Client hardware (MAC) address.
    mac: [u8; 6],
    /// Transaction ID for the current exchange.
    xid: u32,
    /// Retry tracking.
    retry: RetryState,
    /// Current lease (if bound).
    lease: Option<DhcpLease>,
    /// Server ID from the accepted offer.
    server_id: [u8; 4],
    /// Offered IP address (from OFFER, before ACK).
    offered_ip: [u8; 4],
    /// Elapsed seconds since lease was obtained (for renewal tracking).
    lease_elapsed_secs: u32,
}

impl DhcpClient {
    /// Create a new DHCP client with the given MAC address.
    pub fn new(mac: [u8; 6]) -> Self {
        DhcpClient {
            state: DhcpState::Init,
            mac,
            xid: Self::generate_xid(&mac),
            retry: RetryState::new(),
            lease: None,
            server_id: [0; 4],
            offered_ip: [0; 4],
            lease_elapsed_secs: 0,
        }
    }

    /// Generate a pseudo-random transaction ID from the MAC address.
    ///
    /// Uses a simple hash to produce a deterministic but unique-ish XID.
    fn generate_xid(mac: &[u8; 6]) -> u32 {
        let mut hash: u32 = 0x1234_5678;
        for &b in mac.iter() {
            hash = hash.wrapping_mul(31).wrapping_add(b as u32);
        }
        hash
    }

    /// Build and return a DHCPDISCOVER message.
    ///
    /// Transitions state from `Init` to `Selecting` and resets retry state.
    pub fn start_discover(&mut self) -> DhcpMessage {
        self.state = DhcpState::Selecting;
        self.retry = RetryState::new();
        self.lease_elapsed_secs = 0;
        self.build_discover()
    }

    /// Build a base DHCP request message with common fields.
    fn base_request(&self, flags: u16) -> DhcpMessage {
        let mut chaddr = [0u8; 16];
        chaddr[..6].copy_from_slice(&self.mac);
        DhcpMessage {
            op: BOOT_REQUEST,
            htype: HTYPE_ETHERNET,
            hlen: HLEN_ETHERNET,
            hops: 0,
            xid: self.xid,
            secs: 0,
            flags,
            ciaddr: [0; 4],
            yiaddr: [0; 4],
            siaddr: [0; 4],
            giaddr: [0; 4],
            chaddr,
            sname: [0; 64],
            file: [0; 128],
            options: [0; MAX_OPTIONS_LEN],
            options_len: 0,
        }
    }

    /// Build a DHCPDISCOVER message.
    fn build_discover(&self) -> DhcpMessage {
        let mut msg = self.base_request(0x8000);

        // Option 53: DHCP Message Type = DISCOVER
        let _ = msg.add_option(OPT_MSG_TYPE, &[MSG_TYPE_DISCOVER]);
        // Option 55: Parameter Request List
        let _ = msg.add_option(
            OPT_PARAM_REQUEST,
            &[OPT_SUBNET_MASK, OPT_ROUTER, OPT_DNS, OPT_LEASE_TIME],
        );
        let _ = msg.add_end_option();

        msg
    }

    /// Handle a DHCPOFFER message.
    ///
    /// If the client is in the `Selecting` state and the offer is valid,
    /// transitions to `Requesting` and returns a DHCPREQUEST message.
    /// Returns `None` if the offer should be ignored.
    pub fn handle_offer(&mut self, msg: &DhcpMessage) -> Option<DhcpMessage> {
        if self.state != DhcpState::Selecting {
            return None;
        }

        // Validate: must be a reply, same XID, with message type OFFER
        if msg.op != BOOT_REPLY || msg.xid != self.xid {
            return None;
        }
        if msg.get_message_type() != Some(MSG_TYPE_OFFER) {
            return None;
        }

        // Extract server ID
        let server_id = msg.get_server_id()?;

        self.offered_ip = msg.yiaddr;
        self.server_id = server_id;
        self.state = DhcpState::Requesting;
        self.retry = RetryState::new();

        Some(self.build_request())
    }

    /// Build a DHCPREQUEST message for the accepted offer.
    fn build_request(&self) -> DhcpMessage {
        let mut msg = self.base_request(0x8000);

        // Option 53: DHCP Message Type = REQUEST
        let _ = msg.add_option(OPT_MSG_TYPE, &[MSG_TYPE_REQUEST]);
        // Option 54: Server Identifier
        let _ = msg.add_option(OPT_SERVER_ID, &self.server_id);
        // Option 55: Parameter Request List
        let _ = msg.add_option(
            OPT_PARAM_REQUEST,
            &[OPT_SUBNET_MASK, OPT_ROUTER, OPT_DNS, OPT_LEASE_TIME],
        );
        let _ = msg.add_end_option();

        msg
    }

    /// Build a DHCPREQUEST for renewal (unicast to the server).
    fn build_renewal_request(&self) -> DhcpMessage {
        let mut msg = self.base_request(0); // no broadcast flag for unicast renewal

        // Set ciaddr to our current IP (RFC 2131 section 4.3.2)
        if let Some(ref lease) = self.lease {
            msg.ciaddr = lease.ip_addr;
        }

        // Option 53: DHCP Message Type = REQUEST
        let _ = msg.add_option(OPT_MSG_TYPE, &[MSG_TYPE_REQUEST]);
        let _ = msg.add_end_option();

        msg
    }

    /// Handle a DHCPACK message.
    ///
    /// If the client is in the `Requesting` or `Renewing` state and the
    /// ACK is valid, transitions to `Bound` and returns the lease.
    /// Returns `None` if the ACK should be ignored.
    pub fn handle_ack(&mut self, msg: &DhcpMessage) -> Option<DhcpLease> {
        if self.state != DhcpState::Requesting && self.state != DhcpState::Renewing {
            return None;
        }

        if msg.op != BOOT_REPLY || msg.xid != self.xid {
            return None;
        }
        if msg.get_message_type() != Some(MSG_TYPE_ACK) {
            return None;
        }

        let lease_time = msg.get_lease_time().unwrap_or(3600);
        let subnet_mask = msg
            .get_option_ipv4(OPT_SUBNET_MASK)
            .unwrap_or([255, 255, 255, 0]);
        let gateway = msg.get_option_ipv4(OPT_ROUTER).unwrap_or([0; 4]);
        let dns = msg.get_option_ipv4(OPT_DNS).unwrap_or([0; 4]);
        let server_id = msg.get_server_id().unwrap_or(self.server_id);

        let lease = DhcpLease {
            ip_addr: msg.yiaddr,
            subnet_mask,
            gateway,
            dns,
            lease_time_secs: lease_time,
            server_id,
        };

        self.lease = Some(lease);
        self.state = DhcpState::Bound;
        self.lease_elapsed_secs = 0;

        Some(lease)
    }

    /// Handle a DHCPNAK message.
    ///
    /// Resets the client back to `Init` state.
    pub fn handle_nak(&mut self, msg: &DhcpMessage) {
        if msg.op != BOOT_REPLY || msg.xid != self.xid {
            return;
        }
        if msg.get_message_type() == Some(MSG_TYPE_NAK) {
            self.state = DhcpState::Init;
            self.lease = None;
        }
    }

    /// Check whether a renewal request should be sent.
    ///
    /// Call this periodically, passing the number of seconds elapsed since
    /// the last call. If T1 (50% of lease time) has been reached, transitions
    /// to `Renewing` and returns a renewal REQUEST message.
    pub fn check_renewal(&mut self, elapsed_secs: u32) -> Option<DhcpMessage> {
        if self.state != DhcpState::Bound {
            return None;
        }

        let lease = self.lease.as_ref()?;
        self.lease_elapsed_secs += elapsed_secs;

        let t1 = lease.lease_time_secs * RENEWAL_PERCENT / 100;

        if self.lease_elapsed_secs >= lease.lease_time_secs {
            // Lease expired
            self.state = DhcpState::Expired;
            self.lease = None;
            return None;
        }

        if self.lease_elapsed_secs >= t1 {
            self.state = DhcpState::Renewing;
            self.retry = RetryState::new();
            return Some(self.build_renewal_request());
        }

        None
    }

    /// Attempt a retry. Returns the retry interval in seconds and a
    /// re-sent message if a retry is available, or `None` if retries
    /// are exhausted.
    pub fn try_retry(&mut self) -> Option<(u32, DhcpMessage)> {
        let interval = self.retry.try_retry()?;
        let msg = match self.state {
            DhcpState::Selecting => self.build_discover(),
            DhcpState::Requesting => self.build_request(),
            DhcpState::Renewing => self.build_renewal_request(),
            _ => return None,
        };
        Some((interval, msg))
    }

    /// Build a DHCPRELEASE message to relinquish the current lease.
    pub fn release(&mut self) -> Option<DhcpMessage> {
        let lease = self.lease.as_ref()?;
        let server_id = lease.server_id;
        let ip_addr = lease.ip_addr;

        let mut msg = self.base_request(0);
        msg.ciaddr = ip_addr;

        let _ = msg.add_option(OPT_MSG_TYPE, &[MSG_TYPE_RELEASE]);
        let _ = msg.add_option(OPT_SERVER_ID, &server_id);
        let _ = msg.add_end_option();

        self.state = DhcpState::Init;
        self.lease = None;

        Some(msg)
    }

    /// Return the current lease, if any.
    pub fn current_lease(&self) -> Option<&DhcpLease> {
        self.lease.as_ref()
    }

    /// Return the current state.
    pub fn current_state(&self) -> DhcpState {
        self.state
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper: build a minimal valid DHCP message buffer
    // -----------------------------------------------------------------------

    fn make_reply(xid: u32, yiaddr: [u8; 4], siaddr: [u8; 4]) -> DhcpMessage {
        DhcpMessage {
            op: BOOT_REPLY,
            htype: HTYPE_ETHERNET,
            hlen: HLEN_ETHERNET,
            hops: 0,
            xid,
            secs: 0,
            flags: 0,
            ciaddr: [0; 4],
            yiaddr,
            siaddr,
            giaddr: [0; 4],
            chaddr: [0; 16],
            sname: [0; 64],
            file: [0; 128],
            options: [0; MAX_OPTIONS_LEN],
            options_len: 0,
        }
    }

    fn make_offer(xid: u32, yiaddr: [u8; 4], server_id: [u8; 4]) -> DhcpMessage {
        let mut msg = make_reply(xid, yiaddr, server_id);

        let _ = msg.add_option(OPT_MSG_TYPE, &[MSG_TYPE_OFFER]);
        let _ = msg.add_option(OPT_SERVER_ID, &server_id);
        let _ = msg.add_option(OPT_SUBNET_MASK, &[255, 255, 255, 0]);
        let _ = msg.add_option(OPT_ROUTER, &[192, 168, 1, 1]);
        let _ = msg.add_option(OPT_DNS, &[8, 8, 8, 8]);
        let _ = msg.add_option(OPT_LEASE_TIME, &3600u32.to_be_bytes());
        let _ = msg.add_end_option();

        msg
    }

    fn make_ack(xid: u32, yiaddr: [u8; 4], server_id: [u8; 4]) -> DhcpMessage {
        let mut msg = make_reply(xid, yiaddr, server_id);

        let _ = msg.add_option(OPT_MSG_TYPE, &[MSG_TYPE_ACK]);
        let _ = msg.add_option(OPT_SERVER_ID, &server_id);
        let _ = msg.add_option(OPT_SUBNET_MASK, &[255, 255, 255, 0]);
        let _ = msg.add_option(OPT_ROUTER, &[192, 168, 1, 1]);
        let _ = msg.add_option(OPT_DNS, &[8, 8, 8, 8]);
        let _ = msg.add_option(OPT_LEASE_TIME, &3600u32.to_be_bytes());
        let _ = msg.add_end_option();

        msg
    }

    fn make_nak(xid: u32) -> DhcpMessage {
        let mut msg = make_reply(xid, [0; 4], [0; 4]);
        let _ = msg.add_option(OPT_MSG_TYPE, &[MSG_TYPE_NAK]);
        let _ = msg.add_end_option();
        msg
    }

    // -----------------------------------------------------------------------
    // Serialize / parse roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_serialize_parse_roundtrip() {
        let mac = [0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x01];
        let mut client = DhcpClient::new(mac);
        let discover = client.start_discover();

        let mut buf = [0u8; 576];
        let len = discover.serialize(&mut buf).unwrap();
        assert!(len >= MIN_MSG_LEN);

        let parsed = DhcpMessage::parse(&buf[..len]).unwrap();
        assert_eq!(parsed.op, BOOT_REQUEST);
        assert_eq!(parsed.xid, discover.xid);
        assert_eq!(parsed.htype, HTYPE_ETHERNET);
        assert_eq!(parsed.hlen, HLEN_ETHERNET);
        assert_eq!(parsed.flags, 0x8000);
        assert_eq!(&parsed.chaddr[..6], &mac);
        assert_eq!(parsed.get_message_type(), Some(MSG_TYPE_DISCOVER));
    }

    #[test]
    fn test_serialize_buffer_too_small() {
        let mut client = DhcpClient::new([1, 2, 3, 4, 5, 6]);
        let discover = client.start_discover();
        let mut tiny_buf = [0u8; 10];
        let result = discover.serialize(&mut tiny_buf);
        assert_eq!(result, Err(DhcpError::BufferTooSmall));
    }

    #[test]
    fn test_parse_too_short() {
        let short_buf = [0u8; 100];
        let result = DhcpMessage::parse(&short_buf);
        assert_eq!(result, Err(DhcpError::PacketTooShort));
    }

    #[test]
    fn test_parse_bad_magic_cookie() {
        let mut buf = [0u8; 300];
        // Set wrong magic cookie
        buf[236] = 0x00;
        buf[237] = 0x00;
        buf[238] = 0x00;
        buf[239] = 0x00;
        let result = DhcpMessage::parse(&buf);
        assert_eq!(result, Err(DhcpError::InvalidMagicCookie));
    }

    // -----------------------------------------------------------------------
    // TLV options parsing
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_options_basic() {
        // Build options: MSG_TYPE=DISCOVER, SUBNET_MASK=255.255.255.0, END
        let mut opts = [0u8; 64];
        opts[0] = OPT_MSG_TYPE;
        opts[1] = 1;
        opts[2] = MSG_TYPE_DISCOVER;
        opts[3] = OPT_SUBNET_MASK;
        opts[4] = 4;
        opts[5] = 255;
        opts[6] = 255;
        opts[7] = 255;
        opts[8] = 0;
        opts[9] = OPT_END;

        let (parsed, count) = parse_options(&opts[..10]).unwrap();
        assert_eq!(count, 2);
        assert_eq!(parsed[0].code, OPT_MSG_TYPE);
        assert_eq!(parsed[0].len, 1);
        assert_eq!(parsed[0].data[0], MSG_TYPE_DISCOVER);
        assert_eq!(parsed[1].code, OPT_SUBNET_MASK);
        assert_eq!(parsed[1].len, 4);
        assert_eq!(&parsed[1].data[..4], &[255, 255, 255, 0]);
    }

    #[test]
    fn test_parse_options_with_pad() {
        let mut opts = [0u8; 16];
        opts[0] = OPT_PAD;
        opts[1] = OPT_PAD;
        opts[2] = OPT_MSG_TYPE;
        opts[3] = 1;
        opts[4] = MSG_TYPE_OFFER;
        opts[5] = OPT_END;

        let (parsed, count) = parse_options(&opts[..6]).unwrap();
        assert_eq!(count, 1);
        assert_eq!(parsed[0].code, OPT_MSG_TYPE);
        assert_eq!(parsed[0].data[0], MSG_TYPE_OFFER);
    }

    #[test]
    fn test_parse_options_empty() {
        let opts = [OPT_END];
        let (_, count) = parse_options(&opts).unwrap();
        assert_eq!(count, 0);
    }

    #[test]
    fn test_parse_options_truncated_length() {
        // Option code without length byte
        let opts = [OPT_MSG_TYPE];
        let result = parse_options(&opts);
        assert_eq!(result, Err(DhcpError::InvalidOptionLength));
    }

    #[test]
    fn test_parse_options_truncated_data() {
        // Option says length=4 but only 2 bytes of data
        let opts = [OPT_SUBNET_MASK, 4, 255, 255];
        let result = parse_options(&opts);
        assert_eq!(result, Err(DhcpError::InvalidOptionLength));
    }

    #[test]
    fn test_message_get_options() {
        let offer = make_offer(0x1234, [192, 168, 1, 100], [192, 168, 1, 1]);
        assert_eq!(offer.get_message_type(), Some(MSG_TYPE_OFFER));
        assert_eq!(offer.get_server_id(), Some([192, 168, 1, 1]));
        assert_eq!(
            offer.get_option_ipv4(OPT_SUBNET_MASK),
            Some([255, 255, 255, 0])
        );
        assert_eq!(offer.get_option_ipv4(OPT_ROUTER), Some([192, 168, 1, 1]));
        assert_eq!(offer.get_option_ipv4(OPT_DNS), Some([8, 8, 8, 8]));
        assert_eq!(offer.get_lease_time(), Some(3600));
    }

    // -----------------------------------------------------------------------
    // Full state machine: DISCOVER -> OFFER -> REQUEST -> ACK
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_dhcp_handshake() {
        let mac = [0x00, 0x11, 0x22, 0x33, 0x44, 0x55];
        let mut client = DhcpClient::new(mac);
        assert_eq!(client.current_state(), DhcpState::Init);

        // Step 1: Start discover
        let discover = client.start_discover();
        assert_eq!(client.current_state(), DhcpState::Selecting);
        assert_eq!(discover.op, BOOT_REQUEST);
        assert_eq!(discover.get_message_type(), Some(MSG_TYPE_DISCOVER));

        // Step 2: Receive offer
        let offer = make_offer(client.xid, [10, 0, 0, 50], [10, 0, 0, 1]);
        let request = client.handle_offer(&offer);
        assert!(request.is_some());
        assert_eq!(client.current_state(), DhcpState::Requesting);
        let request = request.unwrap();
        assert_eq!(request.get_message_type(), Some(MSG_TYPE_REQUEST));
        assert_eq!(request.get_server_id(), Some([10, 0, 0, 1]));

        // Step 3: Receive ACK
        let ack = make_ack(client.xid, [10, 0, 0, 50], [10, 0, 0, 1]);
        let lease = client.handle_ack(&ack);
        assert!(lease.is_some());
        assert_eq!(client.current_state(), DhcpState::Bound);

        let lease = lease.unwrap();
        assert_eq!(lease.ip_addr, [10, 0, 0, 50]);
        assert_eq!(lease.subnet_mask, [255, 255, 255, 0]);
        assert_eq!(lease.gateway, [192, 168, 1, 1]);
        assert_eq!(lease.dns, [8, 8, 8, 8]);
        assert_eq!(lease.lease_time_secs, 3600);
        assert_eq!(lease.server_id, [10, 0, 0, 1]);
    }

    #[test]
    fn test_offer_wrong_xid_ignored() {
        let mac = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let mut client = DhcpClient::new(mac);
        let _discover = client.start_discover();

        let offer = make_offer(0xDEAD_BEEF, [10, 0, 0, 50], [10, 0, 0, 1]);
        assert!(client.handle_offer(&offer).is_none());
        assert_eq!(client.current_state(), DhcpState::Selecting);
    }

    #[test]
    fn test_offer_in_wrong_state_ignored() {
        let mac = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let mut client = DhcpClient::new(mac);
        // Still in Init state
        let offer = make_offer(client.xid, [10, 0, 0, 50], [10, 0, 0, 1]);
        assert!(client.handle_offer(&offer).is_none());
    }

    #[test]
    fn test_ack_in_wrong_state_ignored() {
        let mac = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let mut client = DhcpClient::new(mac);
        let _discover = client.start_discover();
        // Still in Selecting, not Requesting
        let ack = make_ack(client.xid, [10, 0, 0, 50], [10, 0, 0, 1]);
        assert!(client.handle_ack(&ack).is_none());
    }

    #[test]
    fn test_nak_resets_state() {
        let mac = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let mut client = DhcpClient::new(mac);
        let _discover = client.start_discover();
        let offer = make_offer(client.xid, [10, 0, 0, 50], [10, 0, 0, 1]);
        let _request = client.handle_offer(&offer);
        assert_eq!(client.current_state(), DhcpState::Requesting);

        let nak = make_nak(client.xid);
        client.handle_nak(&nak);
        assert_eq!(client.current_state(), DhcpState::Init);
        assert!(client.current_lease().is_none());
    }

    // -----------------------------------------------------------------------
    // Retry logic
    // -----------------------------------------------------------------------

    #[test]
    fn test_retry_exponential_backoff() {
        let mac = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let mut client = DhcpClient::new(mac);
        let _discover = client.start_discover();

        // First retry: 2s
        let (interval, msg) = client.try_retry().unwrap();
        assert_eq!(interval, 2);
        assert_eq!(msg.get_message_type(), Some(MSG_TYPE_DISCOVER));

        // Second retry: 4s (total: 6s)
        let (interval, _) = client.try_retry().unwrap();
        assert_eq!(interval, 4);

        // Third retry: would be 8s but total=6+8=14 > 10s timeout
        assert!(client.try_retry().is_none());
    }

    #[test]
    fn test_retry_max_attempts() {
        let mut retry = RetryState::new();
        // Override to large timeout so attempts limit is hit first
        let r1 = retry.try_retry();
        assert!(r1.is_some());
        assert_eq!(r1.unwrap(), 2);

        let r2 = retry.try_retry();
        assert!(r2.is_some());
        assert_eq!(r2.unwrap(), 4);

        // Third attempt: 8s, but 2+4+8=14 > 10
        let r3 = retry.try_retry();
        assert!(r3.is_none());
    }

    #[test]
    fn test_retry_in_requesting_state() {
        let mac = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let mut client = DhcpClient::new(mac);
        let _discover = client.start_discover();
        let offer = make_offer(client.xid, [10, 0, 0, 50], [10, 0, 0, 1]);
        let _request = client.handle_offer(&offer);
        assert_eq!(client.current_state(), DhcpState::Requesting);

        let (interval, msg) = client.try_retry().unwrap();
        assert_eq!(interval, 2);
        assert_eq!(msg.get_message_type(), Some(MSG_TYPE_REQUEST));
    }

    #[test]
    fn test_retry_not_possible_in_bound() {
        let mac = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let mut client = DhcpClient::new(mac);
        let _discover = client.start_discover();
        let offer = make_offer(client.xid, [10, 0, 0, 50], [10, 0, 0, 1]);
        let _request = client.handle_offer(&offer);
        let ack = make_ack(client.xid, [10, 0, 0, 50], [10, 0, 0, 1]);
        let _lease = client.handle_ack(&ack);
        assert_eq!(client.current_state(), DhcpState::Bound);

        assert!(client.try_retry().is_none());
    }

    // -----------------------------------------------------------------------
    // Renewal
    // -----------------------------------------------------------------------

    #[test]
    fn test_renewal_at_t1() {
        let mac = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let mut client = DhcpClient::new(mac);
        let _discover = client.start_discover();
        let offer = make_offer(client.xid, [10, 0, 0, 50], [10, 0, 0, 1]);
        let _request = client.handle_offer(&offer);
        let ack = make_ack(client.xid, [10, 0, 0, 50], [10, 0, 0, 1]);
        let _lease = client.handle_ack(&ack);

        // Lease is 3600s, T1 = 1800s
        // Advance 1799s — should not trigger renewal
        assert!(client.check_renewal(1799).is_none());
        assert_eq!(client.current_state(), DhcpState::Bound);

        // Advance 1 more second (total 1800s = T1)
        let renewal = client.check_renewal(1);
        assert!(renewal.is_some());
        assert_eq!(client.current_state(), DhcpState::Renewing);
        let renewal_msg = renewal.unwrap();
        assert_eq!(renewal_msg.get_message_type(), Some(MSG_TYPE_REQUEST));
        assert_eq!(renewal_msg.ciaddr, [10, 0, 0, 50]);
    }

    #[test]
    fn test_lease_expiry() {
        let mac = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let mut client = DhcpClient::new(mac);
        let _discover = client.start_discover();
        let offer = make_offer(client.xid, [10, 0, 0, 50], [10, 0, 0, 1]);
        let _request = client.handle_offer(&offer);
        let ack = make_ack(client.xid, [10, 0, 0, 50], [10, 0, 0, 1]);
        let _lease = client.handle_ack(&ack);

        // Jump past lease expiry
        let result = client.check_renewal(3601);
        assert!(result.is_none());
        assert_eq!(client.current_state(), DhcpState::Expired);
        assert!(client.current_lease().is_none());
    }

    #[test]
    fn test_renewal_not_in_bound() {
        let mac = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let mut client = DhcpClient::new(mac);
        assert!(client.check_renewal(100).is_none());
    }

    #[test]
    fn test_renewal_ack_returns_to_bound() {
        let mac = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let mut client = DhcpClient::new(mac);
        let _discover = client.start_discover();
        let offer = make_offer(client.xid, [10, 0, 0, 50], [10, 0, 0, 1]);
        let _request = client.handle_offer(&offer);
        let ack = make_ack(client.xid, [10, 0, 0, 50], [10, 0, 0, 1]);
        let _lease = client.handle_ack(&ack);

        // Trigger renewal
        let _renewal = client.check_renewal(1800);
        assert_eq!(client.current_state(), DhcpState::Renewing);

        // Server ACKs renewal with new lease
        let renewal_ack = make_ack(client.xid, [10, 0, 0, 50], [10, 0, 0, 1]);
        let new_lease = client.handle_ack(&renewal_ack);
        assert!(new_lease.is_some());
        assert_eq!(client.current_state(), DhcpState::Bound);
    }

    // -----------------------------------------------------------------------
    // Release
    // -----------------------------------------------------------------------

    #[test]
    fn test_release() {
        let mac = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let mut client = DhcpClient::new(mac);
        let _discover = client.start_discover();
        let offer = make_offer(client.xid, [10, 0, 0, 50], [10, 0, 0, 1]);
        let _request = client.handle_offer(&offer);
        let ack = make_ack(client.xid, [10, 0, 0, 50], [10, 0, 0, 1]);
        let _lease = client.handle_ack(&ack);

        let release = client.release();
        assert!(release.is_some());
        let release_msg = release.unwrap();
        assert_eq!(release_msg.get_message_type(), Some(MSG_TYPE_RELEASE));
        assert_eq!(release_msg.ciaddr, [10, 0, 0, 50]);
        assert_eq!(client.current_state(), DhcpState::Init);
        assert!(client.current_lease().is_none());
    }

    #[test]
    fn test_release_without_lease() {
        let mac = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let mut client = DhcpClient::new(mac);
        assert!(client.release().is_none());
    }

    // -----------------------------------------------------------------------
    // Error display
    // -----------------------------------------------------------------------

    #[test]
    fn test_dhcp_error_display() {
        assert_eq!(
            alloc::format!("{}", DhcpError::PacketTooShort),
            "dhcp: packet too short"
        );
        assert_eq!(
            alloc::format!("{}", DhcpError::InvalidMagicCookie),
            "dhcp: invalid magic cookie"
        );
        assert_eq!(alloc::format!("{}", DhcpError::Timeout), "dhcp: timeout");
    }

    #[test]
    fn test_dhcp_error_eq() {
        assert_eq!(DhcpError::PacketTooShort, DhcpError::PacketTooShort);
        assert_ne!(DhcpError::PacketTooShort, DhcpError::Timeout);
    }

    // -----------------------------------------------------------------------
    // Message type constants
    // -----------------------------------------------------------------------

    #[test]
    fn test_message_type_constants() {
        assert_eq!(MSG_TYPE_DISCOVER, 1);
        assert_eq!(MSG_TYPE_OFFER, 2);
        assert_eq!(MSG_TYPE_REQUEST, 3);
        assert_eq!(MSG_TYPE_DECLINE, 4);
        assert_eq!(MSG_TYPE_ACK, 5);
        assert_eq!(MSG_TYPE_NAK, 6);
        assert_eq!(MSG_TYPE_RELEASE, 7);
    }

    #[test]
    fn test_option_code_constants() {
        assert_eq!(OPT_SUBNET_MASK, 1);
        assert_eq!(OPT_ROUTER, 3);
        assert_eq!(OPT_DNS, 6);
        assert_eq!(OPT_LEASE_TIME, 51);
        assert_eq!(OPT_MSG_TYPE, 53);
        assert_eq!(OPT_SERVER_ID, 54);
        assert_eq!(OPT_END, 255);
    }

    #[test]
    fn test_magic_cookie() {
        assert_eq!(MAGIC_COOKIE, 0x6382_5363);
    }

    // -----------------------------------------------------------------------
    // Option overflow
    // -----------------------------------------------------------------------

    #[test]
    fn test_options_overflow() {
        let mut msg = DhcpMessage::default();
        // Fill options to near capacity
        let big_data = [0xABu8; 250];
        let _ = msg.add_option(1, &big_data); // 2 + 250 = 252
                                              // Try to add another large option
        let result = msg.add_option(2, &big_data);
        assert_eq!(result, Err(DhcpError::OptionsOverflow));
    }

    #[test]
    fn test_end_option_overflow() {
        let mut msg = DhcpMessage {
            options_len: MAX_OPTIONS_LEN,
            ..DhcpMessage::default()
        };
        let result = msg.add_end_option();
        assert_eq!(result, Err(DhcpError::OptionsOverflow));
    }

    // -----------------------------------------------------------------------
    // DhcpOption helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_dhcp_option_new() {
        let opt = DhcpOption::new(OPT_MSG_TYPE, &[MSG_TYPE_DISCOVER]);
        assert_eq!(opt.code, OPT_MSG_TYPE);
        assert_eq!(opt.len, 1);
        assert_eq!(opt.data[0], MSG_TYPE_DISCOVER);
    }

    // -----------------------------------------------------------------------
    // Multiple offers — first accepted
    // -----------------------------------------------------------------------

    #[test]
    fn test_second_offer_ignored_after_request() {
        let mac = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let mut client = DhcpClient::new(mac);
        let _discover = client.start_discover();

        let offer1 = make_offer(client.xid, [10, 0, 0, 50], [10, 0, 0, 1]);
        let _request = client.handle_offer(&offer1);
        assert_eq!(client.current_state(), DhcpState::Requesting);

        // Second offer arrives — should be ignored (we're no longer Selecting)
        let offer2 = make_offer(client.xid, [10, 0, 0, 60], [10, 0, 0, 2]);
        assert!(client.handle_offer(&offer2).is_none());
    }

    // -----------------------------------------------------------------------
    // Default message
    // -----------------------------------------------------------------------

    #[test]
    fn test_default_message() {
        let msg = DhcpMessage::default();
        assert_eq!(msg.op, 0);
        assert_eq!(msg.htype, HTYPE_ETHERNET);
        assert_eq!(msg.hlen, HLEN_ETHERNET);
        assert_eq!(msg.xid, 0);
        assert_eq!(msg.options_len, 0);
        assert_eq!(msg.yiaddr, [0; 4]);
    }

    // -----------------------------------------------------------------------
    // iter_dhcp_options direct tests
    // -----------------------------------------------------------------------

    use alloc::vec::Vec;

    #[test]
    fn test_iter_options_empty() {
        let buf: [u8; 0] = [];
        let items: Vec<_> = iter_dhcp_options(&buf).collect();
        assert!(items.is_empty());
    }

    #[test]
    fn test_iter_options_pad_then_end() {
        // Pad, Pad, End — should yield nothing and terminate cleanly
        let buf = [OPT_PAD, OPT_PAD, OPT_END];
        let items: Vec<_> = iter_dhcp_options(&buf).collect();
        assert!(items.is_empty());
    }

    #[test]
    fn test_iter_options_only_end() {
        let buf = [OPT_END];
        let items: Vec<_> = iter_dhcp_options(&buf).collect();
        assert!(items.is_empty());
    }

    #[test]
    fn test_iter_options_truncated_length() {
        // Option code present but no length byte
        let buf = [OPT_SUBNET_MASK];
        let items: Vec<_> = iter_dhcp_options(&buf).collect();
        assert_eq!(items.len(), 1);
        assert!(matches!(items[0], Err(DhcpError::InvalidOptionLength)));
    }

    #[test]
    fn test_iter_options_truncated_value() {
        // Option code + length says 4 bytes but only 2 follow
        let buf = [OPT_SUBNET_MASK, 4, 0xff, 0xff];
        let items: Vec<_> = iter_dhcp_options(&buf).collect();
        assert_eq!(items.len(), 1);
        assert!(matches!(items[0], Err(DhcpError::InvalidOptionLength)));
    }

    #[test]
    fn test_iter_options_multiple_options() {
        // subnet_mask=255.255.255.0, router=192.168.1.1, end
        let buf = [
            OPT_SUBNET_MASK,
            4,
            255,
            255,
            255,
            0,
            OPT_ROUTER,
            4,
            192,
            168,
            1,
            1,
            OPT_END,
        ];
        let items: Vec<_> = iter_dhcp_options(&buf).collect();
        assert_eq!(items.len(), 2);
        let (code0, val0) = items[0].as_ref().unwrap();
        assert_eq!(*code0, OPT_SUBNET_MASK);
        assert_eq!(*val0, &[255, 255, 255, 0][..]);
        let (code1, val1) = items[1].as_ref().unwrap();
        assert_eq!(*code1, OPT_ROUTER);
        assert_eq!(*val1, &[192, 168, 1, 1][..]);
    }

    #[test]
    fn test_iter_options_pad_between() {
        // Pad, option, pad, end
        let buf = [OPT_PAD, OPT_SUBNET_MASK, 4, 1, 2, 3, 4, OPT_PAD, OPT_END];
        let items: Vec<_> = iter_dhcp_options(&buf).collect();
        assert_eq!(items.len(), 1);
        let (code, val) = items[0].as_ref().unwrap();
        assert_eq!(*code, OPT_SUBNET_MASK);
        assert_eq!(*val, &[1, 2, 3, 4][..]);
    }

    #[test]
    fn test_iter_options_terminates_after_error() {
        // Truncated option followed by what would be a valid one — iterator
        // should stop after the error.
        let buf = [OPT_SUBNET_MASK, 10, 1, 2];
        let items: Vec<_> = iter_dhcp_options(&buf).collect();
        assert_eq!(items.len(), 1);
        assert!(items[0].is_err());
    }

    #[test]
    fn test_iter_options_zero_length_value() {
        // Option with zero-length value
        let buf = [OPT_SUBNET_MASK, 0, OPT_END];
        let items: Vec<_> = iter_dhcp_options(&buf).collect();
        assert_eq!(items.len(), 1);
        let (code, val) = items[0].as_ref().unwrap();
        assert_eq!(*code, OPT_SUBNET_MASK);
        assert_eq!(val.len(), 0);
    }
}
