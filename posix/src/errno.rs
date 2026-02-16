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
    fn errno_from_raw_all_variants() {
        assert_eq!(Errno::from_raw(2), Some(Errno::ENOENT));
        assert_eq!(Errno::from_raw(3), Some(Errno::ESRCH));
        assert_eq!(Errno::from_raw(4), Some(Errno::EINTR));
        assert_eq!(Errno::from_raw(5), Some(Errno::EIO));
        assert_eq!(Errno::from_raw(6), Some(Errno::ENXIO));
        assert_eq!(Errno::from_raw(9), Some(Errno::EBADF));
        assert_eq!(Errno::from_raw(11), Some(Errno::EAGAIN));
        assert_eq!(Errno::from_raw(12), Some(Errno::ENOMEM));
        assert_eq!(Errno::from_raw(13), Some(Errno::EACCES));
        assert_eq!(Errno::from_raw(14), Some(Errno::EFAULT));
        assert_eq!(Errno::from_raw(16), Some(Errno::EBUSY));
        assert_eq!(Errno::from_raw(17), Some(Errno::EEXIST));
        assert_eq!(Errno::from_raw(22), Some(Errno::EINVAL));
        assert_eq!(Errno::from_raw(24), Some(Errno::EMFILE));
        assert_eq!(Errno::from_raw(28), Some(Errno::ENOSPC));
        assert_eq!(Errno::from_raw(30), Some(Errno::EROFS));
        assert_eq!(Errno::from_raw(88), Some(Errno::ENOTSOCK));
        assert_eq!(Errno::from_raw(97), Some(Errno::EAFNOSUPPORT));
        assert_eq!(Errno::from_raw(98), Some(Errno::EADDRINUSE));
        assert_eq!(Errno::from_raw(101), Some(Errno::ENETUNREACH));
        assert_eq!(Errno::from_raw(104), Some(Errno::ECONNRESET));
        assert_eq!(Errno::from_raw(110), Some(Errno::ETIMEDOUT));
        assert_eq!(Errno::from_raw(114), Some(Errno::EALREADY));
        assert_eq!(Errno::from_raw(115), Some(Errno::EINPROGRESS));
    }

    #[test]
    fn errno_display_all_variants() {
        use alloc::format;
        assert_eq!(format!("{}", Errno::EPERM), "Operation not permitted");
        assert_eq!(format!("{}", Errno::ENOENT), "No such file or directory");
        assert_eq!(format!("{}", Errno::ESRCH), "No such process");
        assert_eq!(format!("{}", Errno::EINTR), "Interrupted system call");
        assert_eq!(format!("{}", Errno::EIO), "I/O error");
        assert_eq!(format!("{}", Errno::ENXIO), "No such device or address");
        assert_eq!(format!("{}", Errno::EBADF), "Bad file descriptor");
        assert_eq!(
            format!("{}", Errno::EAGAIN),
            "Resource temporarily unavailable"
        );
        assert_eq!(format!("{}", Errno::ENOMEM), "Out of memory");
        assert_eq!(format!("{}", Errno::EACCES), "Permission denied");
        assert_eq!(format!("{}", Errno::EFAULT), "Bad address");
        assert_eq!(format!("{}", Errno::EBUSY), "Device or resource busy");
        assert_eq!(format!("{}", Errno::EEXIST), "File exists");
        assert_eq!(format!("{}", Errno::EINVAL), "Invalid argument");
        assert_eq!(format!("{}", Errno::EMFILE), "Too many open files");
        assert_eq!(format!("{}", Errno::ENOSPC), "No space left on device");
        assert_eq!(format!("{}", Errno::EROFS), "Read-only file system");
        assert_eq!(format!("{}", Errno::ENOSYS), "Function not implemented");
        assert_eq!(format!("{}", Errno::ECONNREFUSED), "Connection refused");
        assert_eq!(format!("{}", Errno::ECONNRESET), "Connection reset by peer");
        assert_eq!(format!("{}", Errno::ETIMEDOUT), "Connection timed out");
        assert_eq!(format!("{}", Errno::EADDRINUSE), "Address already in use");
        assert_eq!(format!("{}", Errno::ENETUNREACH), "Network is unreachable");
        assert_eq!(
            format!("{}", Errno::EALREADY),
            "Operation already in progress"
        );
        assert_eq!(
            format!("{}", Errno::EINPROGRESS),
            "Operation now in progress"
        );
        assert_eq!(format!("{}", Errno::ENOTSOCK), "Not a socket");
        assert_eq!(
            format!("{}", Errno::EAFNOSUPPORT),
            "Address family not supported"
        );
    }

    #[test]
    fn errno_values_all_match_linux() {
        assert_eq!(Errno::ESRCH as i32, 3);
        assert_eq!(Errno::EINTR as i32, 4);
        assert_eq!(Errno::EIO as i32, 5);
        assert_eq!(Errno::ENXIO as i32, 6);
        assert_eq!(Errno::EFAULT as i32, 14);
        assert_eq!(Errno::EBUSY as i32, 16);
        assert_eq!(Errno::EEXIST as i32, 17);
        assert_eq!(Errno::EMFILE as i32, 24);
        assert_eq!(Errno::ENOSPC as i32, 28);
        assert_eq!(Errno::EROFS as i32, 30);
        assert_eq!(Errno::ENOTSOCK as i32, 88);
        assert_eq!(Errno::EAFNOSUPPORT as i32, 97);
        assert_eq!(Errno::EADDRINUSE as i32, 98);
        assert_eq!(Errno::ENETUNREACH as i32, 101);
        assert_eq!(Errno::ECONNRESET as i32, 104);
        assert_eq!(Errno::ETIMEDOUT as i32, 110);
        assert_eq!(Errno::EALREADY as i32, 114);
        assert_eq!(Errno::EINPROGRESS as i32, 115);
    }
}
