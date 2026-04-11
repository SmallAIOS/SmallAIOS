/*
 * SmallAIOS Cooperative Scheduler Fairness Model
 * Verifies: no task starvation under cooperative scheduling
 * SPDX-License-Identifier: Apache-2.0
 *
 * Models N tasks sharing a single CPU with cooperative yields.
 * Each task yields at ONNX operator boundaries.
 * Verifies that every ready task eventually gets to run.
 */

#define N_TASKS 3
#define QUANTA 2  /* Operations before yield */

mtype = { READY, RUNNING, BLOCKED, DONE };

mtype task_state[N_TASKS];
byte current_task = 0;
byte run_count[N_TASKS];
bool all_done = false;

/* Scheduler: round-robin with cooperative yielding */
proctype Scheduler() {
    byte next;
    byte checked;

    do
    :: !all_done ->
        /* Find next ready task (round-robin) */
        checked = 0;
        next = (current_task + 1) % N_TASKS;
        do
        :: (checked < N_TASKS && task_state[next] != READY) ->
            next = (next + 1) % N_TASKS;
            checked++;
        :: (checked < N_TASKS && task_state[next] == READY) ->
            break;
        :: (checked >= N_TASKS) ->
            /* No ready tasks — check if all done */
            byte d = 0;
            byte i = 0;
            do
            :: (i < N_TASKS) ->
                if
                :: (task_state[i] == DONE) -> d++;
                :: else -> skip;
                fi;
                i++;
            :: (i >= N_TASKS) -> break;
            od;
            if
            :: (d == N_TASKS) -> all_done = true;
            :: else -> skip;
            fi;
            break;
        od;

        if
        :: (checked < N_TASKS) ->
            /* Dispatch task */
            current_task = next;
            task_state[next] = RUNNING;
        :: else -> skip;
        fi;
    :: all_done -> break;
    od;
}

/* Task: runs for some operations, then yields */
proctype Task(byte id; byte total_ops) {
    byte ops_done = 0;
    byte quantum;

    task_state[id] = READY;

    do
    :: (ops_done < total_ops) ->
        /* Wait to be scheduled */
        (task_state[id] == RUNNING);

        /* Execute a quantum of operations */
        quantum = 0;
        do
        :: (quantum < QUANTA && ops_done < total_ops) ->
            /* Simulate ONNX operator execution */
            ops_done++;
            quantum++;
            run_count[id]++;
        :: (quantum >= QUANTA || ops_done >= total_ops) ->
            break;
        od;

        /* Yield (cooperative) */
        if
        :: (ops_done < total_ops) ->
            task_state[id] = READY;
        :: (ops_done >= total_ops) ->
            task_state[id] = DONE;
        fi;
    :: (ops_done >= total_ops) -> break;
    od;
}

init {
    byte i = 0;
    do
    :: (i < N_TASKS) ->
        task_state[i] = BLOCKED;
        run_count[i] = 0;
        i++;
    :: (i >= N_TASKS) -> break;
    od;

    /* Launch tasks with different workloads */
    run Task(0, 4);  /* 4 operations */
    run Task(1, 6);  /* 6 operations */
    run Task(2, 2);  /* 2 operations */
    run Scheduler();
}

/* LTL: every task eventually completes (no starvation) */
ltl no_starvation_0 { <> (task_state[0] == DONE) }
ltl no_starvation_1 { <> (task_state[1] == DONE) }
ltl no_starvation_2 { <> (task_state[2] == DONE) }

/* LTL: all tasks eventually complete */
ltl all_complete { <> all_done }

/* LTL: fairness — every ready task eventually runs */
ltl fairness {
    [] ((task_state[0] == READY) -> <> (task_state[0] == RUNNING))
}
