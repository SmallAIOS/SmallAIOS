/*
 * SmallAIOS ONNX Inference Pipeline Model
 * Verifies: no deadlock under concurrent inference requests
 * SPDX-License-Identifier: Apache-2.0
 *
 * Models the inference pipeline: request arrival, model loading,
 * session creation, graph execution, and result delivery.
 * Verifies that concurrent requests do not deadlock.
 */

#define MAX_REQUESTS 3
#define MAX_SESSIONS 2

/* Pipeline stages */
mtype = { REQUEST, LOAD, EXECUTE, RESULT, DONE };

/* Request channel: client -> pipeline */
chan request_chan = [MAX_REQUESTS] of { mtype, byte };  /* type, request_id */

/* Result channel: pipeline -> client */
chan result_chan = [MAX_REQUESTS] of { mtype, byte };    /* type, request_id */

/* Model state */
bool model_loaded = false;

/* Session pool: tracks active sessions */
byte active_sessions = 0;

/* Counters for verification */
byte requests_submitted = 0;
byte requests_completed = 0;

/* Client process: submits inference requests */
proctype Client(byte id) {
    /* Submit request */
    request_chan ! REQUEST, id;
    requests_submitted++;

    /* Wait for result */
    mtype msg_type;
    byte result_id;
    result_chan ? msg_type, result_id;
    assert(msg_type == RESULT);
    assert(result_id == id);
    requests_completed++;
}

/* Model loader: ensures model is loaded before inference */
proctype ModelLoader() {
    /* Simulate model loading (one-time) */
    if
    :: (!model_loaded) ->
        model_loaded = true;
    :: (model_loaded) ->
        skip;
    fi;
}

/* Inference worker: processes requests through the pipeline */
proctype InferenceWorker() {
    mtype msg_type;
    byte req_id;

    do
    :: request_chan ? msg_type, req_id ->
        assert(msg_type == REQUEST);

        /* Ensure model is loaded */
        assert(model_loaded);

        /* Acquire session (bounded) */
        atomic {
            (active_sessions < MAX_SESSIONS) ->
            active_sessions++;
        }

        /* Execute inference (atomic to model computation) */
        /* In real system: graph traversal, operator execution */
        skip;

        /* Release session */
        atomic {
            active_sessions--;
        }

        /* Send result back */
        result_chan ! RESULT, req_id;

    :: timeout -> break;
    od;
}

/* Main initialization */
init {
    /* Load model first */
    run ModelLoader();

    /* Wait for model to load */
    (model_loaded);

    /* Start inference workers */
    run InferenceWorker();
    run InferenceWorker();

    /* Submit concurrent requests */
    run Client(1);
    run Client(2);
    run Client(3);
}

/*
 * Safety property: no deadlock
 * Verified by SPIN's default deadlock detection (-DNOREDUCE for full search)
 *
 * Liveness: all requests eventually complete
 * ltl all_complete { [] (requests_submitted == MAX_REQUESTS -> <> (requests_completed == MAX_REQUESTS)) }
 */
