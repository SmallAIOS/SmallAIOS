---- MODULE SPIProtocol ----
\* SPI Full-Duplex Transfer Protocol Model
\* Verifies: data integrity, CS assertion, full-duplex symmetry
\* Models clock-driven shift register exchange per SPI specification
\* SPDX-License-Identifier: Apache-2.0

EXTENDS Naturals, Sequences

CONSTANTS
    WordSize        \* Number of bits per transfer (4 or 8)

VARIABLES
    master_state,   \* "idle" | "asserting_cs" | "transferring" | "done"
    slave_state,    \* "idle" | "selected" | "transferring" | "done"
    sclk,           \* Clock signal: 0 or 1
    mosi,           \* Master Out Slave In: current bit value
    miso,           \* Master In Slave Out: current bit value
    cs_n,           \* Chip Select (active low): TRUE = deasserted, FALSE = asserted
    bit_counter,    \* Number of bits shifted so far
    shift_reg_m,    \* Master shift register (sequence of bits)
    shift_reg_s,    \* Slave shift register (sequence of bits)
    captured_m,     \* Bits captured by master from MISO
    captured_s,     \* Bits captured by slave from MOSI
    tx_data_m,      \* Original data master wants to send
    tx_data_s       \* Original data slave wants to send

vars == <<master_state, slave_state, sclk, mosi, miso, cs_n,
          bit_counter, shift_reg_m, shift_reg_s, captured_m, captured_s,
          tx_data_m, tx_data_s>>

Bits == {0, 1}

\* Generate a fixed test pattern for a given role
\* Master sends: 1,0,1,0,...  Slave sends: 0,1,0,1,...
MasterPattern == [i \in 1..WordSize |-> (i + 1) % 2]
SlavePattern  == [i \in 1..WordSize |-> i % 2]

\* --- Type Invariant ---

TypeInvariant ==
    /\ master_state \in {"idle", "asserting_cs", "transferring", "done"}
    /\ slave_state \in {"idle", "selected", "transferring", "done"}
    /\ sclk \in Bits
    /\ mosi \in Bits
    /\ miso \in Bits
    /\ cs_n \in BOOLEAN
    /\ bit_counter \in 0..WordSize

\* --- Initial State ---

Init ==
    /\ master_state = "idle"
    /\ slave_state = "idle"
    /\ sclk = 0
    /\ mosi = 0
    /\ miso = 0
    /\ cs_n = TRUE          \* Deasserted (inactive high)
    /\ bit_counter = 0
    /\ shift_reg_m = MasterPattern
    /\ shift_reg_s = SlavePattern
    /\ captured_m = <<>>
    /\ captured_s = <<>>
    /\ tx_data_m = MasterPattern
    /\ tx_data_s = SlavePattern

\* --- Actions ---

\* Master asserts chip select (pulls CS low)
AssertCS ==
    /\ master_state = "idle"
    /\ cs_n = TRUE
    /\ cs_n' = FALSE
    /\ master_state' = "asserting_cs"
    /\ slave_state' = "selected"
    /\ UNCHANGED <<sclk, mosi, miso, bit_counter, shift_reg_m, shift_reg_s,
                   captured_m, captured_s, tx_data_m, tx_data_s>>

\* Setup phase: both sides drive their first bit onto the lines
SetupBits ==
    /\ master_state = "asserting_cs"
    /\ slave_state = "selected"
    /\ cs_n = FALSE
    /\ master_state' = "transferring"
    /\ slave_state' = "transferring"
    /\ mosi' = shift_reg_m[1]    \* Master drives first bit
    /\ miso' = shift_reg_s[1]    \* Slave drives first bit
    /\ sclk' = 0
    /\ UNCHANGED <<cs_n, bit_counter, shift_reg_m, shift_reg_s,
                   captured_m, captured_s, tx_data_m, tx_data_s>>

\* Rising clock edge: both sides sample the incoming line
ClockRise ==
    /\ master_state = "transferring"
    /\ slave_state = "transferring"
    /\ sclk = 0
    /\ bit_counter < WordSize
    /\ cs_n = FALSE
    /\ sclk' = 1
    \* Master samples MISO, slave samples MOSI
    /\ captured_m' = Append(captured_m, miso)
    /\ captured_s' = Append(captured_s, mosi)
    /\ bit_counter' = bit_counter + 1
    /\ UNCHANGED <<master_state, slave_state, cs_n, mosi, miso,
                   shift_reg_m, shift_reg_s, tx_data_m, tx_data_s>>

\* Falling clock edge: advance to next bit (if any remain)
ClockFall ==
    /\ master_state = "transferring"
    /\ sclk = 1
    /\ cs_n = FALSE
    /\ sclk' = 0
    /\ IF bit_counter < WordSize
       THEN /\ mosi' = shift_reg_m[bit_counter + 1]
            /\ miso' = shift_reg_s[bit_counter + 1]
            /\ UNCHANGED <<master_state, slave_state>>
       ELSE /\ master_state' = "done"
            /\ slave_state' = "done"
            /\ UNCHANGED <<mosi, miso>>
    /\ UNCHANGED <<cs_n, bit_counter, shift_reg_m, shift_reg_s,
                   captured_m, captured_s, tx_data_m, tx_data_s>>

\* Deassert CS after transfer completes
DeassertCS ==
    /\ master_state = "done"
    /\ cs_n = FALSE
    /\ cs_n' = TRUE
    /\ UNCHANGED <<master_state, slave_state, sclk, mosi, miso, bit_counter,
                   shift_reg_m, shift_reg_s, captured_m, captured_s,
                   tx_data_m, tx_data_s>>

\* --- Next-State Relation ---

Next ==
    \/ AssertCS
    \/ SetupBits
    \/ ClockRise
    \/ ClockFall
    \/ DeassertCS

Fairness == WF_vars(Next)

Spec == Init /\ [][Next]_vars /\ Fairness

\* --- Safety Properties ---

\* CS must be asserted (low) during any active transfer
CSAssertedDuringTransfer ==
    (master_state = "transferring") => (cs_n = FALSE)

\* Bit counter never exceeds word size
BitCountBound ==
    bit_counter <= WordSize

\* Data integrity: what master sent = what slave captured (when complete)
DataIntegrityStoM ==
    (master_state = "done" /\ Len(captured_m) = WordSize) =>
        captured_m = tx_data_s

DataIntegrityStoM2 ==
    (slave_state = "done" /\ Len(captured_s) = WordSize) =>
        captured_s = tx_data_m

\* Full duplex symmetry: both sides transfer same number of bits
FullDuplexSymmetry ==
    Len(captured_m) = Len(captured_s)

\* --- Liveness Properties ---

\* Transfer eventually completes
EventualCompletion ==
    (master_state = "idle") ~> (master_state = "done")

====
