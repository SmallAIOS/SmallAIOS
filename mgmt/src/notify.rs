// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Cooperative `#![no_std]` MPMC broadcast channel for `Config` change
//! notifications.
//!
//! Spec: `openspec/changes/management-login-v1/specs/mgmt-config-model/spec.md`
//! Requirement "Apply lifecycle" — step (5) "notify subscribers via a
//! broadcast channel".
//!
//! ## Model
//!
//! The broadcast channel is **fan-out, monotonic**, and **cooperative**:
//!
//! - Every successful [`crate::ApplyEngine::write`] appends one
//!   [`NotifyEvent`] to a single shared ring per channel.
//! - Each [`Subscriber`] holds an independent read cursor; subscribers
//!   pull events on their own schedule (no preemption, no callbacks).
//! - When a slow subscriber lags by more than the ring capacity, the
//!   channel reports a [`NotifyError::Lagged`] count of dropped events
//!   the next time that subscriber polls. The subscriber's cursor jumps
//!   to the oldest still-resident event so polling can continue.
//!
//! `live` (re-read on event) and `boot` (deferred to reboot) reload
//! semantics are imposed by the consumer, not the channel — the channel
//! treats every event uniformly.
//!
//! ## Why a ring instead of `alloc::sync::Weak`?
//!
//! Subscribers are spread across cooperative tasks (e.g., the metrics
//! sweeper, the audit appender, the Zenoh admin handler). They are
//! pulled-from, not pushed-to. A bounded ring with per-subscriber
//! cursors gives:
//!
//! 1. Backpressure visibility (lag is a counted error, not silent drop).
//! 2. No allocation per event (capacity fixed at construction).
//! 3. No cross-task locking on the producer fast-path beyond a single
//!    [`RefCell`] borrow.

use alloc::collections::VecDeque;
use alloc::rc::Rc;
use alloc::string::String;
use alloc::vec::Vec;
use core::cell::RefCell;
use core::fmt;

use crate::surface::{ConfigPath, Value};

/// Default broadcast-channel capacity. Sized so a single mass-load of
/// the v1 `Config` schema (~30 fields) plus headroom for in-flight
/// changes never lags the audit subscriber.
pub const DEFAULT_CAPACITY: usize = 64;

/// One `Config` mutation event delivered to subscribers.
///
/// `before` and `after` are typed [`Value`]s so subscribers can react
/// without re-parsing. `who` is a free-form identity string supplied by
/// the originating surface (`"root"`, `"operator:alice"`, `"toml"`,
/// `"zenoh:peer-7"`, …) — the audit ring is the canonical authority
/// for identity provenance.
#[derive(Debug, Clone, PartialEq)]
pub struct NotifyEvent {
    /// The path that changed.
    pub path: ConfigPath,
    /// Previous value, before the write.
    pub before: Value,
    /// New value, after the write.
    pub after: Value,
    /// Free-form identity of the originator.
    pub who: String,
    /// Monotonic timestamp from the originating surface, in nanoseconds
    /// since boot. The surface picks the clock source; the channel does
    /// not interpret the value.
    pub when_ns: u64,
}

/// Errors raised by [`Subscriber::poll`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotifyError {
    /// The subscriber's cursor lagged the channel by more than the
    /// ring capacity. `dropped` events were missed; the cursor was
    /// advanced to the oldest still-resident event before this error
    /// was returned.
    Lagged { dropped: usize },
}

impl fmt::Display for NotifyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Lagged { dropped } => {
                write!(
                    f,
                    "mgmt::notify: subscriber lagged, dropped {dropped} events"
                )
            }
        }
    }
}

/// Shared ring state. Producer holds the only writable handle via
/// [`BroadcastTx::publish`]; subscribers borrow read-only.
struct RingInner {
    /// Bounded ring. Newest event is `events.back()`, oldest is
    /// `events.front()`. When `events.len() == capacity` the next push
    /// pops the front and increments `oldest_seq`.
    events: VecDeque<NotifyEvent>,
    /// Sequence number of `events.front()`. Sequence numbers are
    /// monotonic across the channel's lifetime; subscribers track their
    /// own cursor against the same sequence space.
    oldest_seq: u64,
    /// Sequence number of the **next** event to be published —
    /// `oldest_seq + events.len()`.
    next_seq: u64,
    /// Maximum number of events held in the ring at any time.
    capacity: usize,
}

/// Producer half. Cheap to clone (`Rc`) — every surface that publishes
/// holds one. The inner ring is `RefCell`-guarded; in `#![no_std]`
/// cooperative tasks this is sound because publishes never re-enter
/// publishes (the apply lifecycle is single-stepped per write).
#[derive(Clone)]
pub struct BroadcastTx {
    inner: Rc<RefCell<RingInner>>,
}

/// Read cursor over the ring. Each subscriber holds its own.
pub struct Subscriber {
    inner: Rc<RefCell<RingInner>>,
    /// Sequence number of the next event this subscriber will return.
    cursor: u64,
    /// Optional path filter. When `Some`, [`Subscriber::poll`] returns
    /// only events whose `path` matches; events that don't match are
    /// skipped (their sequence numbers are still consumed).
    filter: Option<ConfigPath>,
}

/// Construct a producer + first subscriber pair with [`DEFAULT_CAPACITY`].
pub fn channel() -> (BroadcastTx, Subscriber) {
    channel_with_capacity(DEFAULT_CAPACITY)
}

/// Construct a producer + first subscriber pair with a caller-chosen
/// ring capacity. `capacity` SHALL be ≥ 1.
pub fn channel_with_capacity(capacity: usize) -> (BroadcastTx, Subscriber) {
    let cap = capacity.max(1);
    let inner = Rc::new(RefCell::new(RingInner {
        events: VecDeque::with_capacity(cap),
        oldest_seq: 0,
        next_seq: 0,
        capacity: cap,
    }));
    let tx = BroadcastTx {
        inner: inner.clone(),
    };
    let rx = Subscriber {
        inner,
        cursor: 0,
        filter: None,
    };
    (tx, rx)
}

impl BroadcastTx {
    /// Publish one event. The next [`Subscriber::poll`] on each
    /// outstanding subscriber will observe it (subject to the
    /// subscriber's path filter and lag handling).
    pub fn publish(&self, event: NotifyEvent) {
        let mut g = self.inner.borrow_mut();
        if g.events.len() == g.capacity {
            g.events.pop_front();
            g.oldest_seq += 1;
        }
        g.events.push_back(event);
        g.next_seq += 1;
    }

    /// Spawn a new subscriber that begins from the **current head** —
    /// it sees only events published after this call.
    pub fn subscribe(&self) -> Subscriber {
        let cursor = self.inner.borrow().next_seq;
        Subscriber {
            inner: self.inner.clone(),
            cursor,
            filter: None,
        }
    }

    /// Spawn a new path-filtered subscriber from the current head.
    pub fn subscribe_path(&self, path: ConfigPath) -> Subscriber {
        let cursor = self.inner.borrow().next_seq;
        Subscriber {
            inner: self.inner.clone(),
            cursor,
            filter: Some(path),
        }
    }

    /// Total events published since channel construction. Sequence
    /// numbers and lag counts are derived from this.
    pub fn published_count(&self) -> u64 {
        self.inner.borrow().next_seq
    }
}

impl Subscriber {
    /// Pull the next event, if any, that matches this subscriber's
    /// filter. Returns:
    ///
    /// - `Ok(Some(event))` — a matching event was delivered.
    /// - `Ok(None)` — no events newer than the cursor are resident.
    /// - `Err(NotifyError::Lagged { dropped })` — the cursor fell
    ///   behind the oldest resident event by `dropped` slots; the
    ///   cursor has been advanced to the oldest resident event and
    ///   the next call will resume from there.
    pub fn poll(&mut self) -> Result<Option<NotifyEvent>, NotifyError> {
        let g = self.inner.borrow();

        // Lag detection — cursor older than the oldest resident.
        if self.cursor < g.oldest_seq {
            let dropped = (g.oldest_seq - self.cursor) as usize;
            self.cursor = g.oldest_seq;
            return Err(NotifyError::Lagged { dropped });
        }

        loop {
            if self.cursor >= g.next_seq {
                return Ok(None);
            }
            let idx = (self.cursor - g.oldest_seq) as usize;
            let event = g.events[idx].clone();
            self.cursor += 1;
            if let Some(filter) = &self.filter {
                if &event.path != filter {
                    continue;
                }
            }
            return Ok(Some(event));
        }
    }

    /// Drain every currently-resident event matching the filter.
    /// Convenience wrapper over [`Subscriber::poll`] that swallows a
    /// single `Lagged` error and keeps draining; the lag count is
    /// returned alongside the events.
    pub fn drain(&mut self) -> (Vec<NotifyEvent>, usize) {
        let mut out = Vec::new();
        let mut dropped = 0usize;
        loop {
            match self.poll() {
                Ok(Some(ev)) => out.push(ev),
                Ok(None) => break,
                Err(NotifyError::Lagged { dropped: d }) => {
                    dropped += d;
                    // Loop again to keep draining from the new cursor.
                }
            }
        }
        (out, dropped)
    }

    /// Restrict (or replace) this subscriber's path filter.
    pub fn set_filter(&mut self, filter: Option<ConfigPath>) {
        self.filter = filter;
    }

    /// The subscriber's current cursor sequence number.
    pub fn cursor(&self) -> u64 {
        self.cursor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surface::{ConfigPath, Value};
    use alloc::string::ToString;

    fn ev(path: &str, before: u64, after: u64, who: &str, when_ns: u64) -> NotifyEvent {
        NotifyEvent {
            path: ConfigPath::new(path).unwrap(),
            before: Value::U64(before),
            after: Value::U64(after),
            who: who.to_string(),
            when_ns,
        }
    }

    #[test]
    fn publish_then_poll_returns_event() {
        let (tx, mut rx) = channel();
        tx.publish(ev("idle.root_minutes", 15, 30, "root", 1));
        let got = rx.poll().expect("no lag").expect("event");
        assert_eq!(got.before, Value::U64(15));
        assert_eq!(got.after, Value::U64(30));
    }

    #[test]
    fn poll_returns_none_when_idle() {
        let (_tx, mut rx) = channel();
        assert_eq!(rx.poll(), Ok(None));
    }

    #[test]
    fn multiple_subscribers_each_see_every_event() {
        let (tx, mut rx1) = channel();
        let mut rx2 = tx.subscribe();
        tx.publish(ev("a", 1, 2, "x", 10));
        tx.publish(ev("a", 2, 3, "x", 20));

        let s1 = rx1.drain().0;
        let s2 = rx2.drain().0;
        assert_eq!(s1.len(), 2);
        assert_eq!(s2.len(), 2);
        assert_eq!(s1[0].after, Value::U64(2));
        assert_eq!(s2[1].after, Value::U64(3));
    }

    #[test]
    fn subscribe_after_publish_starts_at_head() {
        let (tx, _rx_initial) = channel();
        tx.publish(ev("a", 1, 2, "x", 10));
        let mut rx = tx.subscribe();
        // Late subscriber sees nothing yet.
        assert_eq!(rx.poll(), Ok(None));
        tx.publish(ev("a", 2, 3, "x", 20));
        let got = rx.poll().unwrap().unwrap();
        assert_eq!(got.after, Value::U64(3));
    }

    #[test]
    fn lagged_subscriber_reports_drop_count() {
        let (tx, mut rx) = channel_with_capacity(2);
        tx.publish(ev("a", 0, 1, "x", 1));
        tx.publish(ev("a", 1, 2, "x", 2));
        tx.publish(ev("a", 2, 3, "x", 3)); // evicts event 0
        tx.publish(ev("a", 3, 4, "x", 4)); // evicts event 1

        // Subscriber's cursor=0 is below oldest_seq=2 → lag.
        match rx.poll() {
            Err(NotifyError::Lagged { dropped }) => assert_eq!(dropped, 2),
            other => panic!("expected lag error, got {:?}", other),
        }
        // After lag, cursor jumped to oldest resident — should see
        // events 2 and 3 next.
        let next = rx.poll().unwrap().unwrap();
        assert_eq!(next.after, Value::U64(3));
        let next = rx.poll().unwrap().unwrap();
        assert_eq!(next.after, Value::U64(4));
        assert_eq!(rx.poll(), Ok(None));
    }

    #[test]
    fn path_filtered_subscriber_skips_other_paths() {
        let (tx, _rx0) = channel();
        let mut rx = tx.subscribe_path(ConfigPath::new("idle.root_minutes").unwrap());
        tx.publish(ev("password.min_length", 1, 2, "x", 1));
        tx.publish(ev("idle.root_minutes", 15, 30, "x", 2));
        tx.publish(ev("password.min_length", 2, 3, "x", 3));
        let got = rx.poll().unwrap().unwrap();
        assert_eq!(got.path, ConfigPath::new("idle.root_minutes").unwrap());
        assert_eq!(rx.poll(), Ok(None));
    }

    #[test]
    fn drain_swallows_lag_and_returns_drop_count() {
        let (tx, mut rx) = channel_with_capacity(2);
        for i in 0..5 {
            tx.publish(ev("a", i, i + 1, "x", i));
        }
        let (events, dropped) = rx.drain();
        assert_eq!(dropped, 3); // saw 0,1,2 evicted before head=3
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn published_count_matches_publishes() {
        let (tx, _rx) = channel();
        assert_eq!(tx.published_count(), 0);
        tx.publish(ev("a", 0, 1, "x", 1));
        tx.publish(ev("a", 1, 2, "x", 2));
        assert_eq!(tx.published_count(), 2);
    }

    #[test]
    fn cursor_advances_only_on_matching_event_when_filtered() {
        // Cursor tracking still consumes sequence numbers for skipped
        // events — test ensures we don't get stuck.
        let (tx, _rx0) = channel();
        let mut rx = tx.subscribe_path(ConfigPath::new("a").unwrap());
        tx.publish(ev("b", 0, 1, "x", 1));
        tx.publish(ev("b", 1, 2, "x", 2));
        tx.publish(ev("a", 2, 3, "x", 3));
        let got = rx.poll().unwrap().unwrap();
        assert_eq!(got.path, ConfigPath::new("a").unwrap());
        assert_eq!(rx.cursor(), 3);
    }

    #[test]
    fn capacity_one_ring_drops_immediately_predecessor() {
        let (tx, mut rx) = channel_with_capacity(1);
        tx.publish(ev("a", 0, 1, "x", 1));
        tx.publish(ev("a", 1, 2, "x", 2));
        match rx.poll() {
            Err(NotifyError::Lagged { dropped }) => assert_eq!(dropped, 1),
            other => panic!("expected lag, got {:?}", other),
        }
        let got = rx.poll().unwrap().unwrap();
        assert_eq!(got.after, Value::U64(2));
    }

    #[test]
    fn set_filter_changes_subsequent_polls() {
        let (tx, mut rx) = channel();
        tx.publish(ev("a", 0, 1, "x", 1));
        rx.set_filter(Some(ConfigPath::new("b").unwrap()));
        // Already-resident "a" event no longer matches new filter.
        assert_eq!(rx.poll(), Ok(None));
        tx.publish(ev("b", 0, 1, "x", 2));
        let got = rx.poll().unwrap().unwrap();
        assert_eq!(got.path, ConfigPath::new("b").unwrap());
    }

    #[test]
    fn lag_recovery_after_drain_resumes_normal_polling() {
        let (tx, mut rx) = channel_with_capacity(2);
        for i in 0..4 {
            tx.publish(ev("a", i, i + 1, "x", i));
        }
        let _ = rx.drain();
        // Subsequent publish should be observed normally.
        tx.publish(ev("a", 100, 101, "x", 100));
        let got = rx.poll().unwrap().unwrap();
        assert_eq!(got.after, Value::U64(101));
    }
}
