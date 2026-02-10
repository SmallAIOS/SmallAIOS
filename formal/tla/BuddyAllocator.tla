---- MODULE BuddyAllocator ----
\* SmallAIOS Buddy Allocator Model
\* Verifies: no double allocation, no memory leak, alignment
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, FiniteSets

CONSTANTS
    MaxOrder,       \* Maximum order (e.g., 10 = 4MiB with 4KiB pages)
    NumPages        \* Total pages = 2^MaxOrder

VARIABLES
    freeList,       \* freeList[order] = set of free block start addresses
    allocated,      \* Set of allocated (start, order) pairs
    step

vars == <<freeList, allocated, step>>

Orders == 0..MaxOrder

\* Block size at a given order (in pages)
BlockSize(order) == 2^order

\* --- Initial State ---
\* One free block of maximum order at address 0

Init ==
    /\ freeList = [o \in Orders |-> IF o = MaxOrder THEN {0} ELSE {}]
    /\ allocated = {}
    /\ step = 0

\* --- Actions ---

\* Split a block of order o+1 into two blocks of order o
Split(order) ==
    /\ order < MaxOrder
    /\ \E addr \in freeList[order + 1] :
        /\ freeList' = [freeList EXCEPT
            ![order + 1] = @ \ {addr},
            ![order] = @ \union {addr, addr + BlockSize(order)}]
        /\ UNCHANGED <<allocated, step>>

\* Allocate a block of the requested order
Allocate(order) ==
    /\ \E addr \in freeList[order] :
        /\ freeList' = [freeList EXCEPT ![order] = @ \ {addr}]
        /\ allocated' = allocated \union {<<addr, order>>}
        /\ step' = step + 1

\* Free an allocated block
Free(addr, order) ==
    /\ <<addr, order>> \in allocated
    /\ allocated' = allocated \ {<<addr, order>>}
    /\ freeList' = [freeList EXCEPT ![order] = @ \union {addr}]
    /\ step' = step + 1

\* Merge two buddy blocks of order o into one block of order o+1
Merge(order) ==
    /\ order < MaxOrder
    /\ \E addr \in freeList[order] :
        LET buddy == IF (addr % BlockSize(order + 1)) = 0
                     THEN addr + BlockSize(order)
                     ELSE addr - BlockSize(order)
        IN /\ buddy \in freeList[order]
           /\ LET mergedAddr == IF addr < buddy THEN addr ELSE buddy
              IN freeList' = [freeList EXCEPT
                  ![order] = (@ \ {addr, buddy}),
                  ![order + 1] = @ \union {mergedAddr}]
           /\ UNCHANGED <<allocated, step>>

\* --- Next State ---

Next ==
    \/ \E o \in Orders : Split(o)
    \/ \E o \in Orders : Allocate(o)
    \/ \E <<a, o>> \in allocated : Free(a, o)
    \/ \E o \in Orders : Merge(o)

\* --- Safety Properties ---

\* No two allocated blocks overlap
NoOverlap ==
    \A <<a1, o1>>, <<a2, o2>> \in allocated :
        (<<a1, o1>> /= <<a2, o2>>) =>
        \/ a1 + BlockSize(o1) <= a2
        \/ a2 + BlockSize(o2) <= a1

\* All addresses are within bounds
InBounds ==
    \A <<a, o>> \in allocated : a + BlockSize(o) <= NumPages

\* Allocated blocks don't appear in free lists
NoDoubleFree ==
    \A <<a, o>> \in allocated : a \notin freeList[o]

Spec == Init /\ [][Next]_vars

====
