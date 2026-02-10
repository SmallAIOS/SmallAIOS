// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Pub/sub messaging pattern.
//!
//! Provides publisher/subscriber fire-and-forget messaging built on top of
//! the key-expression router. Publishers create messages tagged with a key
//! expression, and subscribers receive messages whose keys match their
//! subscription expression.

#![allow(dead_code)]

use alloc::vec::Vec;

use crate::key_expr::KeyExpr;
use crate::router::SubscriptionId;
use crate::IpcError;

/// Maximum number of messages that can be queued per subscriber.
const MAX_QUEUED: usize = 256;

/// Unique identifier for a publisher.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PublisherId(pub u64);

/// A message sent through the pub/sub system.
///
/// Messages carry a payload of arbitrary bytes, tagged with the key
/// expression they were published on, a timestamp, and the originating
/// publisher's identity.
#[derive(Clone, Debug, PartialEq)]
pub struct Message {
    /// Key expression this message was published on.
    pub key: KeyExpr,
    /// Opaque payload bytes.
    pub payload: Vec<u8>,
    /// Monotonic timestamp (e.g., tick count or TSC value).
    pub timestamp: u64,
    /// Identity of the publisher that created this message.
    pub publisher: PublisherId,
}

/// A bounded FIFO queue of messages for a single subscriber.
///
/// Messages are delivered in order and the queue rejects new messages
/// when full, applying back-pressure to the publisher side.
#[derive(Debug)]
pub struct MessageQueue {
    /// Internal storage.
    messages: Vec<Message>,
}

impl Default for MessageQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl MessageQueue {
    /// Create a new, empty message queue.
    pub fn new() -> Self {
        MessageQueue {
            messages: Vec::new(),
        }
    }

    /// Push a message onto the back of the queue.
    ///
    /// # Errors
    ///
    /// Returns [`IpcError::BufferFull`] if the queue already contains
    /// [`MAX_QUEUED`] messages.
    pub fn push(&mut self, msg: Message) -> Result<(), IpcError> {
        if self.messages.len() >= MAX_QUEUED {
            return Err(IpcError::BufferFull);
        }
        self.messages.push(msg);
        Ok(())
    }

    /// Pop the oldest message from the front of the queue.
    ///
    /// Returns `None` if the queue is empty.
    pub fn pop(&mut self) -> Option<Message> {
        if self.messages.is_empty() {
            None
        } else {
            Some(self.messages.remove(0))
        }
    }

    /// Return the number of messages currently in the queue.
    pub fn len(&self) -> usize {
        self.messages.len()
    }

    /// Return `true` if the queue contains no messages.
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }
}

/// A publisher that creates messages on a fixed key expression.
///
/// Publishers are lightweight handles that stamp messages with their
/// identity and key expression before handing them off to the routing
/// layer.
#[derive(Debug, Clone)]
pub struct Publisher {
    /// Unique identity of this publisher.
    pub id: PublisherId,
    /// Key expression this publisher writes to.
    pub key_expr: KeyExpr,
}

impl Publisher {
    /// Create a new publisher with the given ID and key expression.
    pub fn new(id: PublisherId, key_expr: KeyExpr) -> Self {
        Publisher { id, key_expr }
    }

    /// Build a [`Message`] from the given payload and timestamp.
    ///
    /// The message inherits this publisher's key expression and identity.
    pub fn create_message(&self, payload: Vec<u8>, timestamp: u64) -> Message {
        Message {
            key: self.key_expr.clone(),
            payload,
            timestamp,
            publisher: self.id,
        }
    }
}

/// A subscriber that receives messages matching its key expression.
///
/// Each subscriber owns a [`MessageQueue`] where delivered messages
/// accumulate until the subscriber consumes them.
#[derive(Debug)]
pub struct Subscriber {
    /// Subscription identity (matches the router's `SubscriptionId`).
    pub id: SubscriptionId,
    /// Key expression this subscriber is interested in.
    pub key_expr: KeyExpr,
    /// Internal message queue.
    queue: MessageQueue,
}

impl Subscriber {
    /// Create a new subscriber with an empty message queue.
    pub fn new(id: SubscriptionId, key_expr: KeyExpr) -> Self {
        Subscriber {
            id,
            key_expr,
            queue: MessageQueue::new(),
        }
    }

    /// Deliver a message into this subscriber's queue.
    ///
    /// # Errors
    ///
    /// Returns [`IpcError::BufferFull`] if the internal queue is at capacity.
    pub fn deliver(&mut self, msg: Message) -> Result<(), IpcError> {
        self.queue.push(msg)
    }

    /// Receive (pop) the oldest message from this subscriber's queue.
    ///
    /// Returns `None` if no messages are pending.
    pub fn receive(&mut self) -> Option<Message> {
        self.queue.pop()
    }

    /// Return the number of messages waiting in this subscriber's queue.
    pub fn pending(&self) -> usize {
        self.queue.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn ke(s: &str) -> KeyExpr {
        KeyExpr::new(s).unwrap()
    }

    // === Publisher ===

    #[test]
    fn publisher_create_message_fields() {
        let pub_id = PublisherId(1);
        let publisher = Publisher::new(pub_id, ke("sensor/temp"));
        let msg = publisher.create_message(vec![0xAA, 0xBB], 42);

        assert_eq!(msg.key, ke("sensor/temp"));
        assert_eq!(msg.payload, vec![0xAA, 0xBB]);
        assert_eq!(msg.timestamp, 42);
        assert_eq!(msg.publisher, PublisherId(1));
    }

    #[test]
    fn publisher_create_message_empty_payload() {
        let publisher = Publisher::new(PublisherId(2), ke("data"));
        let msg = publisher.create_message(vec![], 100);
        assert!(msg.payload.is_empty());
        assert_eq!(msg.timestamp, 100);
    }

    #[test]
    fn publisher_clone() {
        let publisher = Publisher::new(PublisherId(3), ke("a/b"));
        let cloned = publisher.clone();
        assert_eq!(cloned.id, publisher.id);
        assert_eq!(cloned.key_expr, publisher.key_expr);
    }

    // === MessageQueue ===

    #[test]
    fn queue_push_pop_fifo() {
        let mut q = MessageQueue::new();
        let msg1 = Message {
            key: ke("a"),
            payload: vec![1],
            timestamp: 1,
            publisher: PublisherId(1),
        };
        let msg2 = Message {
            key: ke("a"),
            payload: vec![2],
            timestamp: 2,
            publisher: PublisherId(1),
        };
        q.push(msg1.clone()).unwrap();
        q.push(msg2.clone()).unwrap();

        assert_eq!(q.pop(), Some(msg1));
        assert_eq!(q.pop(), Some(msg2));
    }

    #[test]
    fn queue_empty_pop_returns_none() {
        let mut q = MessageQueue::new();
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn queue_len_and_is_empty() {
        let mut q = MessageQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);

        q.push(Message {
            key: ke("x"),
            payload: vec![],
            timestamp: 0,
            publisher: PublisherId(0),
        })
        .unwrap();

        assert!(!q.is_empty());
        assert_eq!(q.len(), 1);
    }

    #[test]
    fn queue_full_error() {
        let mut q = MessageQueue::new();
        for i in 0..MAX_QUEUED {
            q.push(Message {
                key: ke("k"),
                payload: vec![i as u8],
                timestamp: i as u64,
                publisher: PublisherId(0),
            })
            .unwrap();
        }
        let result = q.push(Message {
            key: ke("k"),
            payload: vec![0xFF],
            timestamp: 999,
            publisher: PublisherId(0),
        });
        assert_eq!(result, Err(IpcError::BufferFull));
    }

    #[test]
    fn queue_push_after_pop_within_limit() {
        let mut q = MessageQueue::new();
        // Fill the queue.
        for i in 0..MAX_QUEUED {
            q.push(Message {
                key: ke("k"),
                payload: vec![i as u8],
                timestamp: i as u64,
                publisher: PublisherId(0),
            })
            .unwrap();
        }
        // Pop one to make room.
        let _ = q.pop();
        // Should succeed now.
        let result = q.push(Message {
            key: ke("k"),
            payload: vec![0],
            timestamp: 0,
            publisher: PublisherId(0),
        });
        assert!(result.is_ok());
    }

    // === Subscriber ===

    #[test]
    fn subscriber_deliver_and_receive() {
        let mut sub = Subscriber::new(SubscriptionId(1), ke("sensor/**"));
        let msg = Message {
            key: ke("sensor/temp"),
            payload: vec![42],
            timestamp: 10,
            publisher: PublisherId(1),
        };
        sub.deliver(msg.clone()).unwrap();
        assert_eq!(sub.pending(), 1);
        let received = sub.receive().unwrap();
        assert_eq!(received, msg);
        assert_eq!(sub.pending(), 0);
    }

    #[test]
    fn subscriber_receive_empty() {
        let mut sub = Subscriber::new(SubscriptionId(2), ke("x"));
        assert_eq!(sub.receive(), None);
        assert_eq!(sub.pending(), 0);
    }

    #[test]
    fn subscriber_deliver_order() {
        let mut sub = Subscriber::new(SubscriptionId(3), ke("data"));
        for i in 0u8..5 {
            sub.deliver(Message {
                key: ke("data"),
                payload: vec![i],
                timestamp: i as u64,
                publisher: PublisherId(1),
            })
            .unwrap();
        }
        for i in 0u8..5 {
            let msg = sub.receive().unwrap();
            assert_eq!(msg.payload, vec![i]);
        }
        assert!(sub.receive().is_none());
    }

    #[test]
    fn subscriber_buffer_full() {
        let mut sub = Subscriber::new(SubscriptionId(4), ke("full"));
        for i in 0..MAX_QUEUED {
            sub.deliver(Message {
                key: ke("full"),
                payload: vec![i as u8],
                timestamp: i as u64,
                publisher: PublisherId(0),
            })
            .unwrap();
        }
        let result = sub.deliver(Message {
            key: ke("full"),
            payload: vec![0xFF],
            timestamp: 999,
            publisher: PublisherId(0),
        });
        assert_eq!(result, Err(IpcError::BufferFull));
    }

    // === Message ===

    #[test]
    fn message_clone_and_eq() {
        let msg = Message {
            key: ke("a/b"),
            payload: vec![1, 2, 3],
            timestamp: 77,
            publisher: PublisherId(5),
        };
        let cloned = msg.clone();
        assert_eq!(msg, cloned);
    }

    #[test]
    fn message_not_equal_different_payload() {
        let msg1 = Message {
            key: ke("a"),
            payload: vec![1],
            timestamp: 0,
            publisher: PublisherId(1),
        };
        let msg2 = Message {
            key: ke("a"),
            payload: vec![2],
            timestamp: 0,
            publisher: PublisherId(1),
        };
        assert_ne!(msg1, msg2);
    }
}
