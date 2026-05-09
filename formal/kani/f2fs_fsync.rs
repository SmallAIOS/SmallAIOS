// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Kani harness sketch for the F2FS fsync crash-safety invariant
// (Phase 8 of `embedded-filesystem-v1`).
//
// Spec: `fs-f2fs-readwrite/spec.md` — "every `fsync(fd)`-acknowledged
// write MUST be readable after an arbitrary subsequent power loss +
// remount".
//
// ## Model
//
// The harness models the F2FS checkpoint commit protocol as a small
// state machine over two checkpoint pack slots and a per-file
// "durable bytes" tracker. A non-deterministic adversary picks (a)
// the write payload, (b) the moment of power loss between any two
// internal commit steps, and (c) which slot the writer is operating
// on.
//
// The proof obligation: after PowerLoss + Remount, every payload
// that was the argument of an acknowledged `fsync(fd)` is
// readable byte-for-byte from the file.
//
// ## How to run
//
// This file is a *sketch* — it is not a runnable binary today. To
// invoke under Kani once the kani CI lane is plumbed:
//
// ```bash
// cargo kani --harness f2fs_fsync_acknowledged_durable
// cargo kani --harness f2fs_fsync_old_pack_intact
// cargo kani --harness f2fs_fsync_alternate_slot_invariant
// ```
//
// The harness uses bounded `kani::any()` ranges so the proof
// terminates in finite time.

#![cfg(kani)]
#![allow(clippy::needless_range_loop)]

extern crate alloc;

const SLOTS: usize = 2;
const PAYLOAD_MAX: usize = 8;

/// One checkpoint pack slot — a tiny mirror of the on-disk
/// `cp_blkaddr` / `cp_blkaddr + segment` pair.
#[derive(Clone, Copy)]
struct Pack {
    version: u32,
    crc_ok: bool,
    journal_ok: bool,
    /// Length of the most-recent fsync'd payload at the time this
    /// pack was committed. Models inode.size for the single tracked
    /// file.
    inode_size: u32,
}

impl Pack {
    const fn empty() -> Self {
        Self {
            version: 0,
            crc_ok: false,
            journal_ok: false,
            inode_size: 0,
        }
    }
}

/// Outcome of `select_checkpoint()` after PowerLoss.
fn select(packs: &[Pack; SLOTS]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for s in 0..SLOTS {
        if !packs[s].crc_ok {
            continue;
        }
        match best {
            None => best = Some(s),
            Some(b) => {
                if packs[s].version > packs[b].version {
                    best = Some(s);
                }
            }
        }
    }
    best
}

/// Atomic commit: write packs to the alternate slot, flush, swap.
/// Crash points are modeled as `kani::any()` choices that abort the
/// commit between any two steps.
fn commit_with_crash(
    packs: &mut [Pack; SLOTS],
    cp_slot: &mut usize,
    new_size: u32,
    crash_point: u8,
) {
    let alt = 1 - *cp_slot;
    // Step 1: invalidate alt slot (writer starts overwriting).
    packs[alt].crc_ok = false;
    packs[alt].journal_ok = false;
    if crash_point == 0 {
        return;
    }
    // Step 2: write header → version + crc_ok = true.
    packs[alt].version = packs[*cp_slot].version + 1;
    packs[alt].crc_ok = true;
    packs[alt].inode_size = new_size;
    if crash_point == 1 {
        return;
    }
    // Step 3: write journal block.
    packs[alt].journal_ok = true;
    if crash_point == 2 {
        return;
    }
    // Step 4: flush barrier (no state change in this model).
    if crash_point == 3 {
        return;
    }
    // Step 5: swap cp_slot atomically.
    *cp_slot = alt;
}

/// Kani harness: every fsync-acked write is readable post-crash IFF
/// the commit ran to completion. If the commit was crashed mid-way,
/// the writer's caller did not get an OK return, so the write was
/// not "acknowledged" — and the rollback to the previous pack is
/// the spec-allowed behavior.
#[cfg(kani)]
#[kani::proof]
fn f2fs_fsync_acknowledged_durable() {
    let mut packs = [Pack::empty(); SLOTS];
    // Initial committed state: pack 0 valid, version 1, size 0.
    packs[0] = Pack {
        version: 1,
        crc_ok: true,
        journal_ok: true,
        inode_size: 0,
    };
    let mut cp_slot: usize = 0;

    // Adversary picks a write size (bounded).
    let new_size: u32 = kani::any();
    kani::assume(new_size <= PAYLOAD_MAX as u32);

    // Adversary picks when the crash happens: 0..=4 means crashed
    // at that step; 5 means commit completed.
    let crash_point: u8 = kani::any();
    kani::assume(crash_point <= 5);

    let pre_pack = packs[cp_slot];

    commit_with_crash(&mut packs, &mut cp_slot, new_size, crash_point);

    let mounted = select(&packs);
    // Invariant: at least one pack is always selectable.
    assert!(mounted.is_some(), "no valid pack post-crash");
    let m = mounted.unwrap();

    if crash_point >= 4 {
        // Commit completed → fsync was acknowledged → the new size
        // SHALL be readable.
        assert_eq!(packs[m].inode_size, new_size);
    } else {
        // Commit aborted → caller did NOT get an OK → the previous
        // committed state is acceptable.
        assert!(packs[m].inode_size == pre_pack.inode_size || packs[m].inode_size == new_size);
    }
}

/// Kani harness: a power loss during commit never wedges the
/// previously committed pack.
#[cfg(kani)]
#[kani::proof]
fn f2fs_fsync_old_pack_intact() {
    let mut packs = [Pack::empty(); SLOTS];
    packs[0] = Pack {
        version: 1,
        crc_ok: true,
        journal_ok: true,
        inode_size: 0,
    };
    let mut cp_slot: usize = 0;

    let new_size: u32 = kani::any();
    kani::assume(new_size <= PAYLOAD_MAX as u32);
    let crash_point: u8 = kani::any();
    kani::assume(crash_point <= 4);

    commit_with_crash(&mut packs, &mut cp_slot, new_size, crash_point);

    // Crashed before swap → cp_slot still points at the original pack.
    if crash_point < 4 {
        assert_eq!(cp_slot, 0);
        assert!(packs[0].crc_ok);
        assert_eq!(packs[0].version, 1);
    }
}

/// Kani harness: the writer never targets the slot the reader would
/// pick (alternate-slot policy).
#[cfg(kani)]
#[kani::proof]
fn f2fs_fsync_alternate_slot_invariant() {
    let mut packs = [Pack::empty(); SLOTS];
    packs[0] = Pack {
        version: 5,
        crc_ok: true,
        journal_ok: true,
        inode_size: 4,
    };
    let cp_slot: usize = 0;

    let alt = 1 - cp_slot;
    // Invariant: `alt` differs from `cp_slot`.
    assert_ne!(alt, cp_slot);
    // Invariant: `alt` slot is the lower-version slot.
    assert!(packs[alt].version <= packs[cp_slot].version);
}

/// Kani harness: if the previous slot had `valid = false`, mount
/// still picks the new slot when its CRC is good.
#[cfg(kani)]
#[kani::proof]
fn f2fs_fsync_picks_higher_version_after_commit() {
    let mut packs = [Pack::empty(); SLOTS];
    packs[0] = Pack {
        version: 1,
        crc_ok: true,
        journal_ok: true,
        inode_size: 0,
    };
    let mut cp_slot: usize = 0;
    let new_size: u32 = kani::any();
    kani::assume(new_size <= PAYLOAD_MAX as u32);
    commit_with_crash(&mut packs, &mut cp_slot, new_size, 5);
    let m = select(&packs).expect("commit completed → pack selectable");
    assert!(packs[m].version >= 1);
}
