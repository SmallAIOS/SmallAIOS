// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Request/reply messaging pattern.
//!
//! Provides a queryable-based request/reply model inspired by Zenoh's
//! query/reply semantics. A [`Queryable`] registers interest in a key
//! expression and receives [`Query`] messages that it can respond to with
//! [`Reply`] messages. The [`QueryRouter`] manages queryable registration
//! and dispatches incoming queries to matching queryables.

#![allow(dead_code)]

use alloc::vec::Vec;

use crate::key_expr::KeyExpr;
use crate::IpcError;

/// Maximum number of pending queries a single queryable can hold.
const MAX_PENDING_QUERIES: usize = 64;

/// Maximum number of queryables the router can manage.
const MAX_QUERYABLES: usize = 256;

/// Unique identifier for a query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueryId(pub u64);

/// Unique identifier for a queryable endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct QueryableId(pub u64);

/// A query message sent to a queryable.
#[derive(Debug, Clone, PartialEq)]
pub struct Query {
    /// Unique identifier for this query.
    pub id: QueryId,
    /// Key expression the query is targeting.
    pub key_expr: KeyExpr,
    /// Query payload (serialised request data).
    pub payload: Vec<u8>,
    /// Timestamp when the query was created (kernel ticks or TSC).
    pub timestamp: u64,
}

/// A reply message sent back in response to a query.
#[derive(Debug, Clone, PartialEq)]
pub struct Reply {
    /// The query ID this reply is responding to.
    pub query_id: QueryId,
    /// Key expression of the reply (may differ from the query key for
    /// wildcard queries).
    pub key_expr: KeyExpr,
    /// Reply payload (serialised response data).
    pub payload: Vec<u8>,
    /// Timestamp when the reply was created.
    pub timestamp: u64,
}

/// A queryable endpoint that receives queries matching its key expression.
///
/// Each queryable maintains a bounded queue of pending queries. When a
/// query arrives via the router, it is pushed into the pending queue.
/// The owner of the queryable drains queries with [`Queryable::next_query`]
/// and sends replies through the router (or directly).
#[derive(Debug)]
pub struct Queryable {
    /// Unique identifier for this queryable.
    pub id: QueryableId,
    /// Key expression this queryable is registered for.
    pub key_expr: KeyExpr,
    /// Bounded queue of pending queries (FIFO, max [`MAX_PENDING_QUERIES`]).
    pending_queries: Vec<Query>,
}

impl Queryable {
    /// Create a new queryable with the given ID and key expression.
    pub fn new(id: QueryableId, key_expr: KeyExpr) -> Self {
        Self {
            id,
            key_expr,
            pending_queries: Vec::new(),
        }
    }

    /// Enqueue a query for this queryable.
    ///
    /// # Errors
    ///
    /// Returns [`IpcError::BufferFull`] if the pending queue has reached
    /// [`MAX_PENDING_QUERIES`].
    pub fn receive_query(&mut self, query: Query) -> Result<(), IpcError> {
        if self.pending_queries.len() >= MAX_PENDING_QUERIES {
            return Err(IpcError::BufferFull);
        }
        self.pending_queries.push(query);
        Ok(())
    }

    /// Dequeue the next pending query (FIFO order).
    ///
    /// Returns `None` if no queries are pending.
    pub fn next_query(&mut self) -> Option<Query> {
        if self.pending_queries.is_empty() {
            None
        } else {
            Some(self.pending_queries.remove(0))
        }
    }

    /// Return the number of pending queries.
    pub fn pending_count(&self) -> usize {
        self.pending_queries.len()
    }
}

/// Router that manages queryable registration and query dispatch.
///
/// The router maintains a set of queryables and routes incoming queries to
/// the first queryable whose key expression matches the query's key
/// expression. Query IDs and queryable IDs are assigned monotonically.
#[derive(Debug)]
pub struct QueryRouter {
    /// Registered queryables.
    queryables: Vec<Queryable>,
    /// Next query ID to assign.
    next_qid: u64,
    /// Next queryable ID to assign.
    next_queryable_id: u64,
}

impl QueryRouter {
    /// Create a new, empty query router.
    pub fn new() -> Self {
        Self {
            queryables: Vec::new(),
            next_qid: 1,
            next_queryable_id: 1,
        }
    }

    /// Register a new queryable for the given key expression.
    ///
    /// # Errors
    ///
    /// Returns [`IpcError::TableFull`] if the number of registered
    /// queryables has reached [`MAX_QUERYABLES`].
    pub fn register(&mut self, key_expr: KeyExpr) -> Result<QueryableId, IpcError> {
        if self.queryables.len() >= MAX_QUERYABLES {
            return Err(IpcError::TableFull);
        }
        let id = QueryableId(self.next_queryable_id);
        self.next_queryable_id += 1;
        self.queryables.push(Queryable::new(id, key_expr));
        Ok(id)
    }

    /// Unregister a queryable by its ID.
    ///
    /// # Errors
    ///
    /// Returns [`IpcError::NotFound`] if no queryable with the given ID
    /// is registered.
    pub fn unregister(&mut self, id: QueryableId) -> Result<(), IpcError> {
        let pos = self
            .queryables
            .iter()
            .position(|q| q.id == id)
            .ok_or(IpcError::NotFound)?;
        self.queryables.remove(pos);
        Ok(())
    }

    /// Submit a query to be routed to the first matching queryable.
    ///
    /// The query is assigned a unique [`QueryId`] and delivered to the first
    /// queryable whose key expression matches `key_expr`.
    ///
    /// # Errors
    ///
    /// - [`IpcError::NotFound`] if no queryable matches the key expression.
    /// - [`IpcError::BufferFull`] if the matching queryable's pending queue
    ///   is full.
    pub fn submit_query(
        &mut self,
        key_expr: &KeyExpr,
        payload: Vec<u8>,
        timestamp: u64,
    ) -> Result<QueryId, IpcError> {
        let qid = QueryId(self.next_qid);
        self.next_qid += 1;

        let query = Query {
            id: qid,
            key_expr: key_expr.clone(),
            payload,
            timestamp,
        };

        // Find the first matching queryable.
        for queryable in self.queryables.iter_mut() {
            if queryable.key_expr.matches(key_expr) {
                queryable.receive_query(query)?;
                return Ok(qid);
            }
        }

        Err(IpcError::NotFound)
    }

    /// Return the number of registered queryables.
    pub fn queryable_count(&self) -> usize {
        self.queryables.len()
    }

    /// Get a mutable reference to a queryable by its ID.
    ///
    /// Returns `None` if no queryable with the given ID is registered.
    pub fn get_queryable_mut(&mut self, id: QueryableId) -> Option<&mut Queryable> {
        self.queryables.iter_mut().find(|q| q.id == id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn ke(s: &str) -> KeyExpr {
        KeyExpr::new(s).unwrap()
    }

    // --- Queryable unit tests ---

    #[test]
    fn queryable_new_has_no_pending() {
        let q = Queryable::new(QueryableId(1), ke("a/b"));
        assert_eq!(q.pending_count(), 0);
    }

    #[test]
    fn queryable_receive_and_next() {
        let mut q = Queryable::new(QueryableId(1), ke("a/b"));
        let query = Query {
            id: QueryId(10),
            key_expr: ke("a/b"),
            payload: vec![1, 2, 3],
            timestamp: 100,
        };
        q.receive_query(query.clone()).unwrap();
        assert_eq!(q.pending_count(), 1);

        let got = q.next_query().unwrap();
        assert_eq!(got, query);
        assert_eq!(q.pending_count(), 0);
    }

    #[test]
    fn queryable_next_empty_returns_none() {
        let mut q = Queryable::new(QueryableId(1), ke("x/y"));
        assert!(q.next_query().is_none());
    }

    #[test]
    fn queryable_fifo_ordering() {
        let mut q = Queryable::new(QueryableId(1), ke("a"));
        for i in 0..5u64 {
            q.receive_query(Query {
                id: QueryId(i),
                key_expr: ke("a"),
                payload: vec![i as u8],
                timestamp: i,
            })
            .unwrap();
        }
        for i in 0..5u64 {
            let got = q.next_query().unwrap();
            assert_eq!(got.id, QueryId(i));
        }
    }

    #[test]
    fn queryable_buffer_full() {
        let mut q = Queryable::new(QueryableId(1), ke("a"));
        for i in 0..MAX_PENDING_QUERIES as u64 {
            q.receive_query(Query {
                id: QueryId(i),
                key_expr: ke("a"),
                payload: Vec::new(),
                timestamp: i,
            })
            .unwrap();
        }
        let result = q.receive_query(Query {
            id: QueryId(999),
            key_expr: ke("a"),
            payload: Vec::new(),
            timestamp: 999,
        });
        assert_eq!(result, Err(IpcError::BufferFull));
    }

    // --- QueryRouter tests ---

    #[test]
    fn router_register_and_unregister() {
        let mut router = QueryRouter::new();
        assert_eq!(router.queryable_count(), 0);

        let id = router.register(ke("sensor/temp")).unwrap();
        assert_eq!(router.queryable_count(), 1);

        router.unregister(id).unwrap();
        assert_eq!(router.queryable_count(), 0);
    }

    #[test]
    fn router_unregister_not_found() {
        let mut router = QueryRouter::new();
        assert_eq!(
            router.unregister(QueryableId(999)),
            Err(IpcError::NotFound)
        );
    }

    #[test]
    fn router_submit_query_routes_to_matching() {
        let mut router = QueryRouter::new();
        let id = router.register(ke("sensor/temp")).unwrap();

        let qid = router
            .submit_query(&ke("sensor/temp"), vec![42], 100)
            .unwrap();
        assert_eq!(qid, QueryId(1));

        let queryable = router.get_queryable_mut(id).unwrap();
        assert_eq!(queryable.pending_count(), 1);
        let query = queryable.next_query().unwrap();
        assert_eq!(query.payload, vec![42]);
        assert_eq!(query.timestamp, 100);
    }

    #[test]
    fn router_submit_query_no_match() {
        let mut router = QueryRouter::new();
        router.register(ke("sensor/temp")).unwrap();

        let result = router.submit_query(&ke("actuator/motor"), vec![], 0);
        assert_eq!(result, Err(IpcError::NotFound));
    }

    #[test]
    fn router_submit_query_wildcard_match() {
        let mut router = QueryRouter::new();
        let id = router.register(ke("sensor/**")).unwrap();

        let qid = router
            .submit_query(&ke("sensor/temp/room1"), vec![1], 200)
            .unwrap();
        assert_eq!(qid, QueryId(1));

        let queryable = router.get_queryable_mut(id).unwrap();
        assert_eq!(queryable.pending_count(), 1);
    }

    #[test]
    fn router_routes_to_first_matching_queryable() {
        let mut router = QueryRouter::new();
        let id1 = router.register(ke("sensor/**")).unwrap();
        let id2 = router.register(ke("sensor/temp")).unwrap();

        // Should route to the first match (id1 with wildcard).
        router.submit_query(&ke("sensor/temp"), vec![], 0).unwrap();

        let q1 = router.get_queryable_mut(id1).unwrap();
        assert_eq!(q1.pending_count(), 1);

        let q2 = router.get_queryable_mut(id2).unwrap();
        assert_eq!(q2.pending_count(), 0);
    }

    #[test]
    fn router_table_full() {
        let mut router = QueryRouter::new();
        for i in 0..MAX_QUERYABLES {
            let name = alloc::format!("key{}", i);
            router.register(ke(&name)).unwrap();
        }
        let result = router.register(ke("overflow"));
        assert_eq!(result, Err(IpcError::TableFull));
    }

    #[test]
    fn router_unique_query_ids() {
        let mut router = QueryRouter::new();
        router.register(ke("a")).unwrap();

        let qid1 = router.submit_query(&ke("a"), vec![], 0).unwrap();
        let qid2 = router.submit_query(&ke("a"), vec![], 1).unwrap();
        assert_ne!(qid1, qid2);
    }

    #[test]
    fn router_unique_queryable_ids() {
        let mut router = QueryRouter::new();
        let id1 = router.register(ke("a")).unwrap();
        let id2 = router.register(ke("b")).unwrap();
        assert_ne!(id1, id2);
    }

    #[test]
    fn router_submit_to_full_queryable() {
        let mut router = QueryRouter::new();
        let id = router.register(ke("a")).unwrap();

        // Fill the queryable's pending queue.
        for i in 0..MAX_PENDING_QUERIES as u64 {
            router.submit_query(&ke("a"), vec![], i).unwrap();
        }

        // Next query should fail with BufferFull.
        let result = router.submit_query(&ke("a"), vec![], 999);
        assert_eq!(result, Err(IpcError::BufferFull));

        // The queryable should still have MAX_PENDING_QUERIES queries.
        let q = router.get_queryable_mut(id).unwrap();
        assert_eq!(q.pending_count(), MAX_PENDING_QUERIES);
    }

    // --- Reply struct tests ---

    #[test]
    fn reply_clone_and_debug() {
        let reply = Reply {
            query_id: QueryId(1),
            key_expr: ke("a/b"),
            payload: vec![10, 20],
            timestamp: 300,
        };
        let cloned = reply.clone();
        assert_eq!(reply, cloned);

        // Debug should not panic.
        let _debug_str = alloc::format!("{:?}", reply);
    }

    #[test]
    fn query_clone_and_debug() {
        let query = Query {
            id: QueryId(5),
            key_expr: ke("x/y/z"),
            payload: vec![0xFF],
            timestamp: 42,
        };
        let cloned = query.clone();
        assert_eq!(query, cloned);

        let _debug_str = alloc::format!("{:?}", query);
    }
}
