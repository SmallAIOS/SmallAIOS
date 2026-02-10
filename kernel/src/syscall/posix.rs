// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! POSIX compatibility syscall handlers.
//!
//! These handlers bridge Linux/POSIX syscall conventions to the SmallAIOS
//! POSIX compatibility layer. Each validates arguments per POSIX semantics
//! and delegates to the posix crate's implementation.
//!
//! Unsupported POSIX calls return -ENOSYS (-38).

use super::{SyscallArgs, SyscallError, SyscallResult};

/// POSIX ENOSYS value (-38) for unimplemented calls.
const POSIX_ENOSYS: i64 = -38;
/// POSIX EINVAL (-22).
const POSIX_EINVAL: i64 = -22;
/// POSIX EBADF (-9).
const POSIX_EBADF: i64 = -9;

// ─── File Operations ────────────────────────────────────────────────────────

/// open(path_ptr, path_len, flags, mode, 0, 0) -> fd or negative errno.
pub fn sys_posix_open(args: &SyscallArgs) -> SyscallResult {
    let path_ptr = args.args[0];
    let path_len = args.args[1];
    if path_ptr == 0 || path_len == 0 {
        return SyscallError::InvalidArgument.as_i64();
    }
    // Stub: will delegate to posix::vfs + posix::fd
    POSIX_ENOSYS
}

/// close(fd) -> 0 or negative errno.
pub fn sys_posix_close(args: &SyscallArgs) -> SyscallResult {
    let _fd = args.args[0] as i32;
    // Stub: will delegate to posix::fd::FdTable::close
    POSIX_ENOSYS
}

/// read(fd, buf_ptr, count) -> bytes_read or negative errno.
pub fn sys_posix_read(args: &SyscallArgs) -> SyscallResult {
    let fd = args.args[0] as i32;
    let buf_ptr = args.args[1];
    let count = args.args[2];
    if buf_ptr == 0 && count > 0 {
        return SyscallError::BadAddress.as_i64();
    }
    if fd < 0 {
        return POSIX_EBADF;
    }
    // Stub: will delegate to posix::vfs::read via fd table
    POSIX_ENOSYS
}

/// write(fd, buf_ptr, count) -> bytes_written or negative errno.
pub fn sys_posix_write(args: &SyscallArgs) -> SyscallResult {
    let fd = args.args[0] as i32;
    let buf_ptr = args.args[1];
    let count = args.args[2];
    if buf_ptr == 0 && count > 0 {
        return SyscallError::BadAddress.as_i64();
    }
    if fd < 0 {
        return POSIX_EBADF;
    }
    // Stub: will delegate to fd write handler
    POSIX_ENOSYS
}

/// fstat(fd, stat_buf_ptr) -> 0 or negative errno.
pub fn sys_posix_fstat(args: &SyscallArgs) -> SyscallResult {
    let fd = args.args[0] as i32;
    let stat_ptr = args.args[1];
    if stat_ptr == 0 {
        return SyscallError::BadAddress.as_i64();
    }
    if fd < 0 {
        return POSIX_EBADF;
    }
    // Stub: will delegate to posix::fd + posix::vfs
    POSIX_ENOSYS
}

/// dup(old_fd) -> new_fd or negative errno.
pub fn sys_posix_dup(args: &SyscallArgs) -> SyscallResult {
    let fd = args.args[0] as i32;
    if fd < 0 {
        return POSIX_EBADF;
    }
    // Stub: will delegate to posix::fd::FdTable::dup
    POSIX_ENOSYS
}

/// dup2(old_fd, new_fd) -> new_fd or negative errno.
pub fn sys_posix_dup2(args: &SyscallArgs) -> SyscallResult {
    let old_fd = args.args[0] as i32;
    let new_fd = args.args[1] as i32;
    if old_fd < 0 || new_fd < 0 {
        return POSIX_EBADF;
    }
    // Stub: will delegate to posix::fd::FdTable::dup2
    POSIX_ENOSYS
}

/// lseek(fd, offset, whence) -> new_offset or negative errno.
pub fn sys_posix_lseek(args: &SyscallArgs) -> SyscallResult {
    let fd = args.args[0] as i32;
    let _offset = args.args[1] as i64;
    let whence = args.args[2] as u32;
    if fd < 0 {
        return POSIX_EBADF;
    }
    // SEEK_SET=0, SEEK_CUR=1, SEEK_END=2
    if whence > 2 {
        return POSIX_EINVAL;
    }
    // Stub
    POSIX_ENOSYS
}

// ─── Memory Operations ─────────────────────────────────────────────────────

/// mmap(addr, length, prot, flags, fd, offset) -> address or MAP_FAILED.
pub fn sys_posix_mmap(args: &SyscallArgs) -> SyscallResult {
    let _addr = args.args[0];
    let length = args.args[1];
    let _prot = args.args[2] as u32;
    let _flags = args.args[3] as u32;
    let _fd = args.args[4] as i32;
    let _offset = args.args[5];
    if length == 0 {
        return POSIX_EINVAL;
    }
    // Stub: will delegate to posix::mmap
    POSIX_ENOSYS
}

/// munmap(addr, length) -> 0 or negative errno.
pub fn sys_posix_munmap(args: &SyscallArgs) -> SyscallResult {
    let addr = args.args[0];
    let length = args.args[1];
    if length == 0 || !addr.is_multiple_of(4096) {
        return POSIX_EINVAL;
    }
    // Stub: will delegate to posix::mmap
    POSIX_ENOSYS
}

/// mprotect(addr, length, prot) -> 0 or negative errno.
pub fn sys_posix_mprotect(args: &SyscallArgs) -> SyscallResult {
    let addr = args.args[0];
    let length = args.args[1];
    let _prot = args.args[2] as u32;
    if length == 0 || !addr.is_multiple_of(4096) {
        return POSIX_EINVAL;
    }
    // Stub: will delegate to posix::mmap
    POSIX_ENOSYS
}

// ─── Epoll Operations ──────────────────────────────────────────────────────

/// epoll_create1(flags) -> epoll_fd or negative errno.
pub fn sys_posix_epoll_create(args: &SyscallArgs) -> SyscallResult {
    let flags = args.args[0] as i32;
    // Only 0 or O_CLOEXEC (0o2000000) are valid
    if flags != 0 && flags != 0o2000000 {
        return POSIX_EINVAL;
    }
    // Stub: will delegate to posix::epoll
    POSIX_ENOSYS
}

/// epoll_ctl(epfd, op, fd, event_ptr) -> 0 or negative errno.
pub fn sys_posix_epoll_ctl(args: &SyscallArgs) -> SyscallResult {
    let epfd = args.args[0] as i32;
    let op = args.args[1] as i32;
    let _fd = args.args[2] as i32;
    if epfd < 0 {
        return POSIX_EBADF;
    }
    // op must be 1 (ADD), 2 (MOD), or 3 (DEL)
    if !(1..=3).contains(&op) {
        return POSIX_EINVAL;
    }
    // Stub: will delegate to posix::epoll
    POSIX_ENOSYS
}

/// epoll_wait(epfd, events_ptr, max_events, timeout_ms) -> count or negative errno.
pub fn sys_posix_epoll_wait(args: &SyscallArgs) -> SyscallResult {
    let epfd = args.args[0] as i32;
    let events_ptr = args.args[1];
    let max_events = args.args[2] as i32;
    if epfd < 0 {
        return POSIX_EBADF;
    }
    if events_ptr == 0 || max_events <= 0 {
        return POSIX_EINVAL;
    }
    // Stub: will delegate to posix::epoll
    POSIX_ENOSYS
}

// ─── Time and Random ────────────────────────────────────────────────────────

/// clock_gettime(clock_id, timespec_ptr) -> 0 or negative errno.
pub fn sys_posix_clock_gettime(args: &SyscallArgs) -> SyscallResult {
    let clock_id = args.args[0] as u32;
    let ts_ptr = args.args[1];
    if ts_ptr == 0 {
        return SyscallError::BadAddress.as_i64();
    }
    // clock_id: 0=REALTIME, 1=MONOTONIC, 2=PROCESS_CPUTIME, 3=THREAD_CPUTIME
    if clock_id > 3 {
        return POSIX_EINVAL;
    }
    // Stub: will delegate to posix::time
    POSIX_ENOSYS
}

/// nanosleep(req_ptr, rem_ptr) -> 0 or negative errno.
pub fn sys_posix_nanosleep(args: &SyscallArgs) -> SyscallResult {
    let req_ptr = args.args[0];
    if req_ptr == 0 {
        return SyscallError::BadAddress.as_i64();
    }
    // Stub: will delegate to posix::time
    POSIX_ENOSYS
}

/// getrandom(buf_ptr, buf_len, flags) -> bytes or negative errno.
pub fn sys_posix_getrandom(args: &SyscallArgs) -> SyscallResult {
    let buf_ptr = args.args[0];
    let buf_len = args.args[1];
    let _flags = args.args[2] as u32;
    if buf_ptr == 0 && buf_len > 0 {
        return SyscallError::BadAddress.as_i64();
    }
    if buf_len == 0 {
        return 0; // Zero bytes requested → zero bytes returned
    }
    // Stub: will delegate to posix::time::getrandom via CSPRNG
    POSIX_ENOSYS
}

// ─── Socket Operations ─────────────────────────────────────────────────────

/// socket(family, type, protocol) -> fd or negative errno.
pub fn sys_posix_socket(args: &SyscallArgs) -> SyscallResult {
    let family = args.args[0] as i32;
    let sock_type = args.args[1] as i32;
    // AF_INET=2, AF_INET6=10; SOCK_STREAM=1, SOCK_DGRAM=2
    if family != 2 && family != 10 {
        return POSIX_EINVAL;
    }
    if sock_type != 1 && sock_type != 2 {
        return POSIX_EINVAL;
    }
    // Stub: will delegate to posix::socket
    POSIX_ENOSYS
}

// ─── Catch-all for unimplemented POSIX calls ────────────────────────────────

/// Default handler for any POSIX syscall not individually routed.
/// Returns ENOSYS per POSIX convention.
pub fn sys_posix_nosys(_args: &SyscallArgs) -> SyscallResult {
    POSIX_ENOSYS
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::super::SyscallArgs;
    use super::*;

    #[test]
    fn posix_open_null_path() {
        let args = SyscallArgs::new(0, [0, 0, 0, 0, 0, 0]);
        assert_eq!(
            sys_posix_open(&args),
            SyscallError::InvalidArgument.as_i64()
        );
    }

    #[test]
    fn posix_open_valid_stub() {
        let args = SyscallArgs::new(0, [0x1000, 5, 0, 0, 0, 0]);
        assert_eq!(sys_posix_open(&args), POSIX_ENOSYS);
    }

    #[test]
    fn posix_close_stub() {
        let args = SyscallArgs::new(0, [3, 0, 0, 0, 0, 0]);
        assert_eq!(sys_posix_close(&args), POSIX_ENOSYS);
    }

    #[test]
    fn posix_read_null_buf() {
        let args = SyscallArgs::new(0, [0, 0, 100, 0, 0, 0]);
        assert_eq!(sys_posix_read(&args), SyscallError::BadAddress.as_i64());
    }

    #[test]
    fn posix_read_negative_fd() {
        let args = SyscallArgs::new(0, [usize::MAX, 0x1000, 100, 0, 0, 0]);
        assert_eq!(sys_posix_read(&args), POSIX_EBADF);
    }

    #[test]
    fn posix_read_valid_stub() {
        let args = SyscallArgs::new(0, [0, 0x1000, 100, 0, 0, 0]);
        assert_eq!(sys_posix_read(&args), POSIX_ENOSYS);
    }

    #[test]
    fn posix_write_null_buf() {
        let args = SyscallArgs::new(0, [1, 0, 100, 0, 0, 0]);
        assert_eq!(sys_posix_write(&args), SyscallError::BadAddress.as_i64());
    }

    #[test]
    fn posix_write_valid_stub() {
        let args = SyscallArgs::new(0, [1, 0x1000, 100, 0, 0, 0]);
        assert_eq!(sys_posix_write(&args), POSIX_ENOSYS);
    }

    #[test]
    fn posix_fstat_null_buf() {
        let args = SyscallArgs::new(0, [0, 0, 0, 0, 0, 0]);
        assert_eq!(sys_posix_fstat(&args), SyscallError::BadAddress.as_i64());
    }

    #[test]
    fn posix_fstat_negative_fd() {
        let args = SyscallArgs::new(0, [usize::MAX, 0x1000, 0, 0, 0, 0]);
        assert_eq!(sys_posix_fstat(&args), POSIX_EBADF);
    }

    #[test]
    fn posix_fstat_valid_stub() {
        let args = SyscallArgs::new(0, [3, 0x1000, 0, 0, 0, 0]);
        assert_eq!(sys_posix_fstat(&args), POSIX_ENOSYS);
    }

    #[test]
    fn posix_dup_negative_fd() {
        let args = SyscallArgs::new(0, [usize::MAX, 0, 0, 0, 0, 0]);
        assert_eq!(sys_posix_dup(&args), POSIX_EBADF);
    }

    #[test]
    fn posix_dup_valid_stub() {
        let args = SyscallArgs::new(0, [1, 0, 0, 0, 0, 0]);
        assert_eq!(sys_posix_dup(&args), POSIX_ENOSYS);
    }

    #[test]
    fn posix_dup2_negative_fd() {
        let args = SyscallArgs::new(0, [usize::MAX, 3, 0, 0, 0, 0]);
        assert_eq!(sys_posix_dup2(&args), POSIX_EBADF);
    }

    #[test]
    fn posix_lseek_invalid_whence() {
        let args = SyscallArgs::new(0, [3, 0, 5, 0, 0, 0]);
        assert_eq!(sys_posix_lseek(&args), POSIX_EINVAL);
    }

    #[test]
    fn posix_lseek_valid_stub() {
        let args = SyscallArgs::new(0, [3, 0, 0, 0, 0, 0]); // SEEK_SET
        assert_eq!(sys_posix_lseek(&args), POSIX_ENOSYS);
    }

    #[test]
    fn posix_mmap_zero_length() {
        let args = SyscallArgs::new(0, [0, 0, 1, 0x22, 0, 0]);
        assert_eq!(sys_posix_mmap(&args), POSIX_EINVAL);
    }

    #[test]
    fn posix_mmap_valid_stub() {
        let args = SyscallArgs::new(0, [0, 4096, 3, 0x22, 0, 0]);
        assert_eq!(sys_posix_mmap(&args), POSIX_ENOSYS);
    }

    #[test]
    fn posix_munmap_unaligned() {
        let args = SyscallArgs::new(0, [1, 4096, 0, 0, 0, 0]);
        assert_eq!(sys_posix_munmap(&args), POSIX_EINVAL);
    }

    #[test]
    fn posix_munmap_zero_length() {
        let args = SyscallArgs::new(0, [4096, 0, 0, 0, 0, 0]);
        assert_eq!(sys_posix_munmap(&args), POSIX_EINVAL);
    }

    #[test]
    fn posix_mprotect_valid_stub() {
        let args = SyscallArgs::new(0, [4096, 4096, 1, 0, 0, 0]);
        assert_eq!(sys_posix_mprotect(&args), POSIX_ENOSYS);
    }

    #[test]
    fn posix_epoll_create_invalid_flags() {
        let args = SyscallArgs::new(0, [0x42, 0, 0, 0, 0, 0]);
        assert_eq!(sys_posix_epoll_create(&args), POSIX_EINVAL);
    }

    #[test]
    fn posix_epoll_create_valid_stub() {
        let args = SyscallArgs::new(0, [0, 0, 0, 0, 0, 0]);
        assert_eq!(sys_posix_epoll_create(&args), POSIX_ENOSYS);
    }

    #[test]
    fn posix_epoll_ctl_invalid_op() {
        let args = SyscallArgs::new(0, [3, 99, 5, 0, 0, 0]);
        assert_eq!(sys_posix_epoll_ctl(&args), POSIX_EINVAL);
    }

    #[test]
    fn posix_epoll_wait_invalid_max_events() {
        let args = SyscallArgs::new(0, [3, 0x1000, 0, 100, 0, 0]);
        assert_eq!(sys_posix_epoll_wait(&args), POSIX_EINVAL);
    }

    #[test]
    fn posix_clock_gettime_null_ptr() {
        let args = SyscallArgs::new(0, [0, 0, 0, 0, 0, 0]);
        assert_eq!(
            sys_posix_clock_gettime(&args),
            SyscallError::BadAddress.as_i64()
        );
    }

    #[test]
    fn posix_clock_gettime_invalid_clock() {
        let args = SyscallArgs::new(0, [99, 0x1000, 0, 0, 0, 0]);
        assert_eq!(sys_posix_clock_gettime(&args), POSIX_EINVAL);
    }

    #[test]
    fn posix_clock_gettime_valid_stub() {
        let args = SyscallArgs::new(0, [1, 0x1000, 0, 0, 0, 0]);
        assert_eq!(sys_posix_clock_gettime(&args), POSIX_ENOSYS);
    }

    #[test]
    fn posix_nanosleep_null_ptr() {
        let args = SyscallArgs::new(0, [0, 0, 0, 0, 0, 0]);
        assert_eq!(
            sys_posix_nanosleep(&args),
            SyscallError::BadAddress.as_i64()
        );
    }

    #[test]
    fn posix_nanosleep_valid_stub() {
        let args = SyscallArgs::new(0, [0x1000, 0, 0, 0, 0, 0]);
        assert_eq!(sys_posix_nanosleep(&args), POSIX_ENOSYS);
    }

    #[test]
    fn posix_getrandom_null_buf() {
        let args = SyscallArgs::new(0, [0, 32, 0, 0, 0, 0]);
        assert_eq!(
            sys_posix_getrandom(&args),
            SyscallError::BadAddress.as_i64()
        );
    }

    #[test]
    fn posix_getrandom_zero_len() {
        let args = SyscallArgs::new(0, [0x1000, 0, 0, 0, 0, 0]);
        assert_eq!(sys_posix_getrandom(&args), 0);
    }

    #[test]
    fn posix_getrandom_valid_stub() {
        let args = SyscallArgs::new(0, [0x1000, 32, 0, 0, 0, 0]);
        assert_eq!(sys_posix_getrandom(&args), POSIX_ENOSYS);
    }

    #[test]
    fn posix_socket_invalid_family() {
        let args = SyscallArgs::new(0, [99, 1, 0, 0, 0, 0]);
        assert_eq!(sys_posix_socket(&args), POSIX_EINVAL);
    }

    #[test]
    fn posix_socket_invalid_type() {
        let args = SyscallArgs::new(0, [2, 99, 0, 0, 0, 0]);
        assert_eq!(sys_posix_socket(&args), POSIX_EINVAL);
    }

    #[test]
    fn posix_socket_valid_stub() {
        let args = SyscallArgs::new(0, [2, 1, 0, 0, 0, 0]); // AF_INET, SOCK_STREAM
        assert_eq!(sys_posix_socket(&args), POSIX_ENOSYS);
    }

    #[test]
    fn posix_nosys_returns_enosys() {
        let args = SyscallArgs::zero(0);
        assert_eq!(sys_posix_nosys(&args), POSIX_ENOSYS);
    }

    #[test]
    fn posix_all_handlers_no_panic_with_max_args() {
        let max_args = [usize::MAX; 6];
        let args = SyscallArgs::new(0, max_args);
        // None of these should panic
        let _ = sys_posix_open(&args);
        let _ = sys_posix_close(&args);
        let _ = sys_posix_read(&args);
        let _ = sys_posix_write(&args);
        let _ = sys_posix_fstat(&args);
        let _ = sys_posix_dup(&args);
        let _ = sys_posix_dup2(&args);
        let _ = sys_posix_lseek(&args);
        let _ = sys_posix_mmap(&args);
        let _ = sys_posix_munmap(&args);
        let _ = sys_posix_mprotect(&args);
        let _ = sys_posix_epoll_create(&args);
        let _ = sys_posix_epoll_ctl(&args);
        let _ = sys_posix_epoll_wait(&args);
        let _ = sys_posix_clock_gettime(&args);
        let _ = sys_posix_nanosleep(&args);
        let _ = sys_posix_getrandom(&args);
        let _ = sys_posix_socket(&args);
        let _ = sys_posix_nosys(&args);
    }
}
