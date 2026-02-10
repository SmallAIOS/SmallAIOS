# Spec 02: POSIX Compatibility Layer

## Overview

SmallAIOS implements a **minimal POSIX-compatible interface** — not full POSIX
compliance, but enough that Rust's `std` library and common AI runtime patterns
can function. The goal is to support the POSIX surface area that inference workloads
actually use, and nothing more.

## Design Philosophy

Full POSIX compliance requires ~1,200 interfaces. AI inference workloads use roughly
5% of them. SmallAIOS implements that 5% faithfully and returns `ENOSYS` for the rest.

This is a pragmatic choice: many Rust crates use `std` (which calls libc), and
rewriting all dependencies as `no_std` is impractical. The POSIX layer lets us
use `std`-based Rust code in user-space components (ONNX runtime, IPC) while keeping
the kernel minimal.

## Implemented POSIX Interfaces

### File Descriptors

File descriptors are the fundamental POSIX abstraction. SmallAIOS uses them for:
- Model file access (read-only virtual filesystem)
- IPC channels
- Device handles
- Logging (stdout/stderr)

```
open(path, flags) -> fd          # Only for /models/*, /dev/*, /proc/self/*
close(fd) -> result
read(fd, buf, count) -> ssize_t
write(fd, buf, count) -> ssize_t  # Only stdout(1), stderr(2), devices
lseek(fd, offset, whence) -> off_t
fstat(fd, stat) -> result
fcntl(fd, cmd, arg) -> result     # F_GETFL, F_SETFL only
dup(fd) -> fd
dup2(oldfd, newfd) -> fd
```

### Memory Mapping

Required for memory-mapped model files and GPU buffer sharing.

```
mmap(addr, len, prot, flags, fd, offset) -> *void
munmap(addr, len) -> result
mprotect(addr, len, prot) -> result
```

Supported flags: `MAP_PRIVATE`, `MAP_ANONYMOUS`, `MAP_FIXED`, `MAP_HUGETLB`
Supported prot: `PROT_READ`, `PROT_WRITE`, `PROT_EXEC` (for JIT), `PROT_NONE`

### Threading (pthreads subset)

Required for parallel inference execution.

```
pthread_create(thread, attr, start, arg) -> result
pthread_join(thread, retval) -> result
pthread_exit(retval)
pthread_self() -> pthread_t

pthread_mutex_init(mutex, attr) -> result
pthread_mutex_lock(mutex) -> result
pthread_mutex_trylock(mutex) -> result
pthread_mutex_unlock(mutex) -> result
pthread_mutex_destroy(mutex) -> result

pthread_cond_init(cond, attr) -> result
pthread_cond_wait(cond, mutex) -> result
pthread_cond_signal(cond) -> result
pthread_cond_broadcast(cond) -> result
pthread_cond_destroy(cond) -> result

pthread_rwlock_init(rwlock, attr) -> result
pthread_rwlock_rdlock(rwlock) -> result
pthread_rwlock_wrlock(rwlock) -> result
pthread_rwlock_unlock(rwlock) -> result
pthread_rwlock_destroy(rwlock) -> result
```

### Socket API (subset)

Required for IPC network transport and health check endpoints.

```
socket(domain, type, protocol) -> fd     # AF_INET/AF_INET6, SOCK_STREAM/SOCK_DGRAM
bind(fd, addr, addrlen) -> result
listen(fd, backlog) -> result
accept(fd, addr, addrlen) -> fd
connect(fd, addr, addrlen) -> result
send(fd, buf, len, flags) -> ssize_t
recv(fd, buf, len, flags) -> ssize_t
sendto(fd, buf, len, flags, addr, addrlen) -> ssize_t
recvfrom(fd, buf, len, flags, addr, addrlen) -> ssize_t
setsockopt(fd, level, optname, optval, optlen) -> result
getsockopt(fd, level, optname, optval, optlen) -> result
shutdown(fd, how) -> result
```

### Epoll (Linux extension, widely used)

Required for async I/O multiplexing in the IPC layer.

```
epoll_create1(flags) -> fd
epoll_ctl(epollfd, op, fd, event) -> result
epoll_wait(epollfd, events, maxevents, timeout) -> count
```

### Clock and Time

```
clock_gettime(clockid, timespec) -> result  # CLOCK_MONOTONIC, CLOCK_REALTIME
nanosleep(req, rem) -> result
gettimeofday(tv, tz) -> result              # Legacy compat
```

### Signals (minimal)

```
sigaction(signum, act, oldact) -> result    # SIGTERM, SIGINT, SIGKILL only
kill(pid, sig) -> result                     # Only pid 0 (self)
```

### Process (minimal)

SmallAIOS is a unikernel — there is exactly one "process". These are stubbed
for compatibility:

```
getpid() -> 1                    # Always returns 1
getuid() -> 0                    # Always root (capability-controlled)
getgid() -> 0
geteuid() -> 0
getegid() -> 0
exit(status)                     # Shuts down the kernel
exit_group(status)               # Same as exit
```

### Misc

```
getrandom(buf, len, flags) -> ssize_t   # CSPRNG
uname(buf) -> result                      # Returns "SmallAIOS" info
sysinfo(info) -> result                   # Memory/CPU stats
futex(uaddr, op, val, ...) -> result      # Required by Rust std
```

## Virtual Filesystem

SmallAIOS presents a minimal read-only virtual filesystem:

```
/
├── models/          # ONNX model files (from container image)
│   ├── model.onnx
│   └── ...
├── config/          # Runtime configuration
│   └── smallaios.toml
├── dev/
│   ├── null
│   ├── zero
│   ├── urandom
│   └── nvidia0     # GPU device (if present)
└── proc/
    └── self/
        ├── maps     # Memory map (for debugging)
        ├── status   # Process status
        └── fd/      # Open file descriptors
```

No writable filesystem. Logs go to a ring buffer accessible via IPC.

## Explicitly NOT Implemented

The following return `ENOSYS` (-38):

- `fork`, `vfork`, `clone` (no process creation)
- `exec*` family (no program execution)
- `pipe`, `pipe2` (use IPC channels instead)
- `chdir`, `chroot`, `pivot_root` (no filesystem navigation)
- `mount`, `umount` (no mountable filesystems)
- `chmod`, `chown` (no permission model beyond capabilities)
- `link`, `symlink`, `unlink`, `rename` (no writable filesystem)
- `mkdir`, `rmdir` (no writable filesystem)
- `iopl`, `ioperm` (no direct port I/O from user code)
- All `*xattr` (no extended attributes)
- `ptrace` (no debugging interface — use QEMU gdb stub instead)
- `personality`, `prctl` (most subcmds)

## Implementation Strategy

The POSIX layer is a **Rust crate** (`posix/`) that translates POSIX calls to
SmallAIOS native syscalls:

```
POSIX call (libc ABI) → posix crate → kernel native syscall
```

For unikernel mode, these are direct function calls with no ring transition.
For VM mode, the posix crate runs in ring 3 and traps to the kernel.

## Crate Structure

```
posix/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── fd.rs        # File descriptor table
    ├── fs.rs        # Virtual filesystem
    ├── mmap.rs      # Memory mapping
    ├── pthread.rs   # Threading primitives
    ├── socket.rs    # Socket API
    ├── epoll.rs     # Epoll multiplexer
    ├── time.rs      # Clock and time
    ├── signal.rs    # Signal stubs
    └── errno.rs     # POSIX error codes
```

## Compatibility Testing

- Run Rust's `std` test suite against the POSIX layer
- Test common ONNX runtime crate operations
- Verify `tokio` async runtime works (required for IPC)
- Fuzz the POSIX interface with random syscall sequences
