// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Message router — dispatches messages to subscribers by key expression.

#![allow(dead_code)]

use alloc::vec::Vec;

use crate::key_expr::KeyExpr;
use crate::IpcError;

/// Maximum number of active subscriptions the router can hold.
const MAX_SUBSCRIPTIONS: usize = 1024;

/// Unique identifier for a subscription.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SubscriptionId(pub u64);

/// A single subscription entry in the router.
#[derive(Debug, Clone)]
pub struct Subscription {
    /// Unique identifier for this subscription.
    pub id: SubscriptionId,
    /// Key expression that this subscription is interested in.
    pub key_expr: KeyExpr,
    /// Whether this subscription is still active.
    pub active: bool,
}

/// Message router that manages subscriptions and matches incoming keys
/// against registered key expressions.
///
/// The router maintains a flat list of subscriptions and performs linear
/// matching. This is appropriate for the expected subscription counts in
/// a unikernel environment (typically < 100).
#[derive(Debug)]
pub struct Router {
    /// All subscriptions (active and inactive).
    subscriptions: Vec<Subscription>,
    /// Next subscription ID to assign.
    next_id: u64,
}

impl Router {
    /// Create a new, empty router.
    pub fn new() -> Self {
        Router {
            subscriptions: Vec::new(),
            next_id: 1,
        }
    }

    /// Register a new subscription for the given key expression.
    ///
    /// # Errors
    ///
    /// Returns [`IpcError::TableFull`] if the number of active subscriptions
    /// has reached [`MAX_SUBSCRIPTIONS`].
    pub fn subscribe(&mut self, key_expr: KeyExpr) -> Result<SubscriptionId, IpcError> {
        let active_count = self.subscriptions.iter().filter(|s| s.active).count();
        if active_count >= MAX_SUBSCRIPTIONS {
            return Err(IpcError::TableFull);
        }

        let id = SubscriptionId(self.next_id);
        self.next_id += 1;

        self.subscriptions.push(Subscription {
            id,
            key_expr,
            active: true,
        });

        Ok(id)
    }

    /// Deactivate a subscription by ID.
    ///
    /// The subscription is marked inactive but not removed, preserving stable
    /// IDs for the lifetime of the router.
    ///
    /// # Errors
    ///
    /// Returns [`IpcError::NotFound`] if no active subscription with the
    /// given ID exists.
    pub fn unsubscribe(&mut self, id: SubscriptionId) -> Result<(), IpcError> {
        for sub in self.subscriptions.iter_mut() {
            if sub.id == id && sub.active {
                sub.active = false;
                return Ok(());
            }
        }
        Err(IpcError::NotFound)
    }

    /// Find all active subscriptions whose key expressions match the given key.
    ///
    /// Returns a list of [`SubscriptionId`]s for all matching subscriptions.
    pub fn match_key(&self, key: &KeyExpr) -> Vec<SubscriptionId> {
        self.subscriptions
            .iter()
            .filter(|s| s.active && s.key_expr.matches(key))
            .map(|s| s.id)
            .collect()
    }

    /// Return the number of currently active subscriptions.
    pub fn subscription_count(&self) -> usize {
        self.subscriptions.iter().filter(|s| s.active).count()
    }

    /// Check whether a subscription with the given ID is currently active.
    pub fn is_subscribed(&self, id: SubscriptionId) -> bool {
        self.subscriptions
            .iter()
            .any(|s| s.id == id && s.active)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn ke(s: &str) -> KeyExpr {
        KeyExpr::new(s).unwrap()
    }

    #[test]
    fn subscribe_returns_unique_ids() {
        let mut router = Router::new();
        let id1 = router.subscribe(ke("a/b")).unwrap();
        let id2 = router.subscribe(ke("c/d")).unwrap();
        assert_ne!(id1, id2);
    }

    #[test]
    fn subscription_count_tracks_active() {
        let mut router = Router::new();
        assert_eq!(router.subscription_count(), 0);
        let id1 = router.subscribe(ke("a")).unwrap();
        let _id2 = router.subscribe(ke("b")).unwrap();
        assert_eq!(router.subscription_count(), 2);
        router.unsubscribe(id1).unwrap();
        assert_eq!(router.subscription_count(), 1);
    }

    #[test]
    fn unsubscribe_success() {
        let mut router = Router::new();
        let id = router.subscribe(ke("x/y")).unwrap();
        assert!(router.is_subscribed(id));
        router.unsubscribe(id).unwrap();
        assert!(!router.is_subscribed(id));
    }

    #[test]
    fn unsubscribe_not_found() {
        let mut router = Router::new();
        let result = router.unsubscribe(SubscriptionId(999));
        assert_eq!(result, Err(IpcError::NotFound));
    }

    #[test]
    fn unsubscribe_already_inactive() {
        let mut router = Router::new();
        let id = router.subscribe(ke("a")).unwrap();
        router.unsubscribe(id).unwrap();
        // Second unsubscribe should fail — it's already inactive.
        assert_eq!(router.unsubscribe(id), Err(IpcError::NotFound));
    }

    #[test]
    fn match_exact_key() {
        let mut router = Router::new();
        let id = router.subscribe(ke("sensor/temp")).unwrap();
        let matches = router.match_key(&ke("sensor/temp"));
        assert_eq!(matches, vec![id]);
    }

    #[test]
    fn match_wildcard_single() {
        let mut router = Router::new();
        let id = router.subscribe(ke("sensor/*/data")).unwrap();
        let matches = router.match_key(&ke("sensor/room1/data"));
        assert_eq!(matches, vec![id]);
    }

    #[test]
    fn match_wildcard_double() {
        let mut router = Router::new();
        let id = router.subscribe(ke("sensor/**")).unwrap();
        let matches = router.match_key(&ke("sensor/room1/temp/celsius"));
        assert_eq!(matches, vec![id]);
    }

    #[test]
    fn match_no_match() {
        let mut router = Router::new();
        let _id = router.subscribe(ke("sensor/temp")).unwrap();
        let matches = router.match_key(&ke("actuator/motor"));
        assert!(matches.is_empty());
    }

    #[test]
    fn match_multiple_subscriptions() {
        let mut router = Router::new();
        let id1 = router.subscribe(ke("sensor/**")).unwrap();
        let id2 = router.subscribe(ke("sensor/temp")).unwrap();
        let _id3 = router.subscribe(ke("actuator/motor")).unwrap();
        let matches = router.match_key(&ke("sensor/temp"));
        assert!(matches.contains(&id1));
        assert!(matches.contains(&id2));
        assert_eq!(matches.len(), 2);
    }

    #[test]
    fn match_skips_inactive() {
        let mut router = Router::new();
        let id1 = router.subscribe(ke("a/b")).unwrap();
        let id2 = router.subscribe(ke("a/b")).unwrap();
        router.unsubscribe(id1).unwrap();
        let matches = router.match_key(&ke("a/b"));
        assert_eq!(matches, vec![id2]);
    }

    #[test]
    fn is_subscribed_false_for_unknown() {
        let router = Router::new();
        assert!(!router.is_subscribed(SubscriptionId(42)));
    }

    #[test]
    fn table_full_error() {
        let mut router = Router::new();
        for i in 0..MAX_SUBSCRIPTIONS {
            let name = alloc::format!("key{}", i);
            router.subscribe(ke(&name)).unwrap();
        }
        let result = router.subscribe(ke("overflow"));
        assert_eq!(result, Err(IpcError::TableFull));
    }

    #[test]
    fn resubscribe_after_unsubscribe_within_limit() {
        let mut router = Router::new();
        // Fill to capacity.
        let mut ids = Vec::new();
        for i in 0..MAX_SUBSCRIPTIONS {
            let name = alloc::format!("key{}", i);
            ids.push(router.subscribe(ke(&name)).unwrap());
        }
        // Unsubscribe one — active count drops below limit.
        router.unsubscribe(ids[0]).unwrap();
        // Should succeed now.
        let result = router.subscribe(ke("new-key"));
        assert!(result.is_ok());
    }
}
