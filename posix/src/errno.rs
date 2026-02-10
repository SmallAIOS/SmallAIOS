// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! POSIX error numbers.
//!
//! Standard errno values used throughout the POSIX compatibility layer.
//! Unsupported operations return `ENOSYS` (-38).

/// POSIX error number type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Errno {
    /// Operation not permitted.
    EPERM = 1,
    /// No such file or directory.
    ENOENT = 2,
    /// No such process.
    ESRCH = 3,
    /// Interrupted system call.
    EINTR = 4,
    /// I/O error.
    EIO = 5,
    /// No such device or address.
    ENXIO = 6,
    /// Bad file descriptor.
    EBADF = 9,
    /// Resource temporarily unavailable / would block.
    EAGAIN = 11,
    /// Out of memory.
    ENOMEM = 12,
    /// Permission denied.
    EACCES = 13,
    /// Bad address.
    EFAULT = 14,
    /// Device or resource busy.
    EBUSY = 16,
    /// File exists.
    EEXIST = 17,
    /// Invalid argument.
    EINVAL = 22,
    /// Too many open files.
    EMFILE = 24,
    /// No space left on device.
    ENOSPC = 28,
    /// Read-only file system.
    EROFS = 30,
    /// Function not implemented.
    ENOSYS = 38,
    /// Connection refused.
    ECONNREFUSED = 111,
    /// Connection reset by peer.
    ECONNRESET = 104,
    /// Connection timed out.
    ETIMEDOUT = 110,
    /// Address already in use.
    EADDRINUSE = 98,
    /// Network is unreachable.
    ENETUNREACH = 101,
    /// Operation already in progress.
    EALREADY = 114,
    /// Operation now in progress.
    EINPROGRESS = 115,
    /// Not a socket.
    ENOTSOCK = 88,
    /// Address family not supported.
    EAFNOSUPPORT = 97,
}

impl Errno {
    /// Return the negative errno value (as returned by Linux syscalls).
    pub const fn as_negative(self) -> i32 {
        -(self as i32)
    }

    /// Return the errno value as i64 for syscall return.
    pub const fn as_i64(self) -> i64 {
        -(self as i32 as i64)
    }

    /// Try to convert a raw errno number to an Errno variant.
    pub fn from_raw(raw: i32) -> Option<Self> {
        match raw {
            1 => Some(Self::EPERM),
            2 => Some(Self::ENOENT),
            3 => Some(Self::ESRCH),
            4 => Some(Self::EINTR),
            5 => Some(Self::EIO),
            6 => Some(Self::ENXIO),
            9 => Some(Self::EBADF),
            11 => Some(Self::EAGAIN),
            12 => Some(Self::ENOMEM),
            13 => Some(Self::EACCES),
            14 => Some(Self::EFAULT),
            16 => Some(Self::EBUSY),
            17 => Some(Self::EEXIST),
            22 => Some(Self::EINVAL),
            24 => Some(Self::EMFILE),
            28 => Some(Self::ENOSPC),
            30 => Some(Self::EROFS),
            38 => Some(Self::ENOSYS),
            88 => Some(Self::ENOTSOCK),
            97 => Some(Self::EAFNOSUPPORT),
            98 => Some(Self::EADDRINUSE),
            101 => Some(Self::ENETUNREACH),
            104 => Some(Self::ECONNRESET),
            110 => Some(Self::ETIMEDOUT),
            111 => Some(Self::ECONNREFUSED),
            114 => Some(Self::EALREADY),
            115 => Some(Self::EINPROGRESS),
            _ => None,
        }
    }
}

impl core::fmt::Display for Errno {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EPERM => write!(f, "Operation not permitted"),
            Self::ENOENT => write!(f, "No such file or directory"),
            Self::ESRCH => write!(f, "No such process"),
            Self::EINTR => write!(f, "Interrupted system call"),
            Self::EIO => write!(f, "I/O error"),
            Self::ENXIO => write!(f, "No such device or address"),
            Self::EBADF => write!(f, "Bad file descriptor"),
            Self::EAGAIN => write!(f, "Resource temporarily unavailable"),
            Self::ENOMEM => write!(f, "Out of memory"),
            Self::EACCES => write!(f, "Permission denied"),
            Self::EFAULT => write!(f, "Bad address"),
            Self::EBUSY => write!(f, "Device or resource busy"),
            Self::EEXIST => write!(f, "File exists"),
            Self::EINVAL => write!(f, "Invalid argument"),
            Self::EMFILE => write!(f, "Too many open files"),
            Self::ENOSPC => write!(f, "No space left on device"),
            Self::EROFS => write!(f, "Read-only file system"),
            Self::ENOSYS => write!(f, "Function not implemented"),
            Self::ECONNREFUSED => write!(f, "Connection refused"),
            Self::ECONNRESET => write!(f, "Connection reset by peer"),
            Self::ETIMEDOUT => write!(f, "Connection timed out"),
            Self::EADDRINUSE => write!(f, "Address already in use"),
            Self::ENETUNREACH => write!(f, "Network is unreachable"),
            Self::EALREADY => write!(f, "Operation already in progress"),
            Self::EINPROGRESS => write!(f, "Operation now in progress"),
            Self::ENOTSOCK => write!(f, "Not a socket"),
            Self::EAFNOSUPPORT => write!(f, "Address family not supported"),
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn errno_values_match_linux() {
        assert_eq!(Errno::EPERM as i32, 1);
        assert_eq!(Errno::ENOENT as i32, 2);
        assert_eq!(Errno::EBADF as i32, 9);
        assert_eq!(Errno::EAGAIN as i32, 11);
        assert_eq!(Errno::ENOMEM as i32, 12);
        assert_eq!(Errno::EACCES as i32, 13);
        assert_eq!(Errno::EINVAL as i32, 22);
        assert_eq!(Errno::ENOSYS as i32, 38);
        assert_eq!(Errno::ECONNREFUSED as i32, 111);
    }

    #[test]
    fn errno_as_negative() {
        assert_eq!(Errno::ENOSYS.as_negative(), -38);
        assert_eq!(Errno::EINVAL.as_negative(), -22);
        assert_eq!(Errno::EBADF.as_negative(), -9);
    }

    #[test]
    fn errno_as_i64() {
        assert_eq!(Errno::ENOSYS.as_i64(), -38);
        assert_eq!(Errno::ENOMEM.as_i64(), -12);
    }

    #[test]
    fn errno_from_raw_valid() {
        assert_eq!(Errno::from_raw(1), Some(Errno::EPERM));
        assert_eq!(Errno::from_raw(38), Some(Errno::ENOSYS));
        assert_eq!(Errno::from_raw(111), Some(Errno::ECONNREFUSED));
    }

    #[test]
    fn errno_from_raw_invalid() {
        assert_eq!(Errno::from_raw(0), None);
        assert_eq!(Errno::from_raw(999), None);
        assert_eq!(Errno::from_raw(-1), None);
    }

    #[test]
    fn errno_display() {
        let msg = alloc::format!("{}", Errno::ENOSYS);
        assert_eq!(msg, "Function not implemented");
    }
}
