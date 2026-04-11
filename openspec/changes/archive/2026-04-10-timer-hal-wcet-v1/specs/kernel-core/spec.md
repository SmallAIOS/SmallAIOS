## MODIFIED Requirements

### Requirement: System Time Syscall
The kernel SHALL expose a `sys_time()` syscall that returns monotonic time since boot in nanoseconds.

#### Scenario: sys_time returns monotonic value
- **WHEN** `sys_time()` is called at times T1 and T2 with T2 > T1
- **THEN** the returned values MUST be monotonic (`time(T2) >= time(T1)`)
- **AND** MUST NOT go backwards

#### Scenario: sys_time uses architecture timer
- **WHEN** `sys_time()` is called on a bare-metal target
- **THEN** it MUST read from the architecture timer via `sched::timer::Timestamp::now()`
- **AND** MUST NOT return a hardcoded zero value
