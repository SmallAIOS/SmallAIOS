/*
 * SmallAIOS QUIC 1-RTT Handshake Protocol Model
 * Verifies: every ClientHello eventually receives ServerHello or timeout
 * SPDX-License-Identifier: Apache-2.0
 *
 * Models the QUIC 1-RTT handshake with PQ hybrid key exchange.
 * Focuses on the protocol liveness property: handshake completion.
 */

#define MAX_RETRIES 3

mtype = {
    CLIENT_HELLO, SERVER_HELLO, ENCRYPTED_EXTENSIONS,
    CLIENT_FINISHED, HANDSHAKE_DONE,
    TIMEOUT, RESET
};

/* Network channels (lossy) */
chan client_to_server = [2] of { mtype };
chan server_to_client = [2] of { mtype };

/* State tracking */
bool client_done = false;
bool server_done = false;
bool handshake_complete = false;
bool client_timed_out = false;

/* Client process */
proctype Client() {
    byte retries = 0;

    /* Send ClientHello */
retry:
    client_to_server ! CLIENT_HELLO;

    /* Wait for ServerHello */
    if
    :: server_to_client ? SERVER_HELLO ->
        /* Receive EncryptedExtensions */
        if
        :: server_to_client ? ENCRYPTED_EXTENSIONS ->
            /* Send Finished */
            client_to_server ! CLIENT_FINISHED;
            /* Wait for HandshakeDone */
            if
            :: server_to_client ? HANDSHAKE_DONE ->
                client_done = true;
                handshake_complete = true;
            :: timeout ->
                /* Server didn't confirm */
                if
                :: (retries < MAX_RETRIES) ->
                    retries++;
                    goto retry;
                :: else ->
                    client_timed_out = true;
                fi;
            fi;
        :: timeout ->
            if
            :: (retries < MAX_RETRIES) ->
                retries++;
                goto retry;
            :: else ->
                client_timed_out = true;
            fi;
        fi;
    :: timeout ->
        /* No ServerHello — retry or give up */
        if
        :: (retries < MAX_RETRIES) ->
            retries++;
            goto retry;
        :: else ->
            client_timed_out = true;
        fi;
    fi;
}

/* Server process */
proctype Server() {
    mtype msg;

    /* Wait for ClientHello */
    client_to_server ? CLIENT_HELLO;

    /* Perform key exchange (ML-KEM-768 + X25519) */
    /* Model: non-deterministic success */
    if
    :: true ->
        /* Key exchange succeeds */
        server_to_client ! SERVER_HELLO;
        server_to_client ! ENCRYPTED_EXTENSIONS;

        /* Wait for client Finished */
        if
        :: client_to_server ? CLIENT_FINISHED ->
            server_to_client ! HANDSHAKE_DONE;
            server_done = true;
        :: timeout ->
            /* Client didn't finish */
            skip;
        fi;
    :: true ->
        /* Key exchange fails (e.g., bad parameters) — send reset */
        skip;
    fi;
}

init {
    run Client();
    run Server();
}

/* LTL: every client eventually completes handshake or times out */
ltl handshake_liveness {
    <> (handshake_complete || client_timed_out)
}

/* LTL: if handshake completes, both sides agree */
ltl handshake_agreement {
    [] (handshake_complete -> (client_done && server_done))
}
