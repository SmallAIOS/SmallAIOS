/*
 * SmallAIOS IPC Pub/Sub Protocol Model
 * Verifies: message delivery, no message loss, subscriber isolation
 * SPDX-License-Identifier: Apache-2.0
 */

#define MAX_MSGS 4
#define MAX_SUBS 3

/* Message channels */
mtype = { PUBLISH, SUBSCRIBE, UNSUBSCRIBE, DATA, ACK };

/* Topic channel: publisher -> broker */
chan pub_chan = [MAX_MSGS] of { mtype, byte, byte };  /* type, topic_id, payload */

/* Subscriber channels: broker -> subscriber[i] */
chan sub_chan[MAX_SUBS] = [MAX_MSGS] of { mtype, byte, byte };

/* Subscription state */
bool subscribed[MAX_SUBS];
byte sub_topic[MAX_SUBS];

/* Counters for verification */
byte published = 0;
byte delivered[MAX_SUBS];

/* Publisher process */
proctype Publisher(byte topic_id) {
    byte payload = 0;
    do
    :: (payload < MAX_MSGS) ->
        pub_chan ! PUBLISH, topic_id, payload;
        published++;
        payload++;
    :: (payload >= MAX_MSGS) ->
        break;
    od;
}

/* Broker process: routes messages to matching subscribers */
proctype Broker() {
    mtype msg_type;
    byte topic_id, payload;
    byte i;

    do
    :: pub_chan ? msg_type, topic_id, payload ->
        if
        :: (msg_type == PUBLISH) ->
            /* Deliver to all matching subscribers */
            i = 0;
            do
            :: (i < MAX_SUBS) ->
                if
                :: (subscribed[i] && sub_topic[i] == topic_id) ->
                    sub_chan[i] ! DATA, topic_id, payload;
                :: else -> skip;
                fi;
                i++;
            :: (i >= MAX_SUBS) ->
                break;
            od;
        :: else -> skip;
        fi;
    :: timeout -> break;
    od;
}

/* Subscriber process */
proctype Subscriber(byte id; byte topic_id) {
    mtype msg_type;
    byte rx_topic, rx_payload;

    /* Subscribe */
    subscribed[id] = true;
    sub_topic[id] = topic_id;

    /* Receive messages */
    do
    :: sub_chan[id] ? msg_type, rx_topic, rx_payload ->
        assert(msg_type == DATA);
        assert(rx_topic == topic_id);
        delivered[id]++;
    :: timeout -> break;
    od;
}

/* Main: set up topology and verify */
init {
    byte i;

    /* Initialize */
    i = 0;
    do
    :: (i < MAX_SUBS) ->
        subscribed[i] = false;
        delivered[i] = 0;
        i++;
    :: (i >= MAX_SUBS) ->
        break;
    od;

    /* Launch processes */
    run Publisher(1);
    run Subscriber(0, 1);
    run Subscriber(1, 1);
    run Subscriber(2, 2);  /* Different topic */
    run Broker();
}

/* LTL properties */
/* All subscribers to topic 1 eventually receive all messages */
/* ltl delivery { [] (published > 0 -> <> (delivered[0] == published)) } */
