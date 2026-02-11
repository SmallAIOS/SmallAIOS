Traceability Matrix
===================

This document provides DO-178C traceability from requirements through
design, implementation, and test cases.

Requirements to Design
----------------------

.. needtable::
   :types: req
   :columns: id, title, safety_level, satisfied_by
   :style: table

Design to Implementation
------------------------

.. needtable::
   :types: design
   :columns: id, title, implemented_by
   :style: table

Test Coverage
-------------

.. needtable::
   :types: test
   :columns: id, title, verifies, coverage
   :style: table

Traceability Flow
-----------------

.. needflow::
   :types: req, design, impl, test
   :show_link_names:

Cybersecurity Compliance Traceability
-------------------------------------

The following table maps cybersecurity requirements (REQ_040-REQ_047) through
specifications, implementations, tests, and formal verification artifacts.

.. list-table:: Cybersecurity Bidirectional Traceability Matrix
   :header-rows: 1
   :widths: 15 15 15 15 15 25

   * - Requirement
     - Specification
     - Implementation
     - Test
     - Verification
     - NIST Control
   * - REQ_040 (Audit)
     - SPEC_040
     - IMPL_040
     - TEST_040 (MC/DC)
     - VERIFY_040 (TLA+)
     - AU-2, AU-3, AU-9, AU-10, AU-12
   * - REQ_041 (Monitoring)
     - SPEC_041
     - IMPL_041
     - TEST_041
     - VERIFY_041 (TLA+)
     - CA-7, SI-4, SI-5
   * - REQ_042 (Incident)
     - SPEC_042
     - IMPL_042
     - TEST_042
     - —
     - IR-4, IR-5, IR-6, IR-8
   * - REQ_043 (Supply Chain)
     - —
     - supply_chain/
     - CI tests
     - —
     - SR-2, SR-3, SR-4, SR-11
   * - REQ_044 (OT/ICS)
     - SPEC_044
     - IMPL_044
     - TEST_044
     - —
     - SI-7, IEC 61508, ISO 26262, DO-178C
   * - REQ_045 (Info Flow)
     - SPEC_045
     - IMPL_045
     - boundary tests
     - VERIFY_045 (Lean 4)
     - AC-4, AC-5
   * - REQ_046 (Key Mgmt)
     - —
     - crypto/key_manager.rs
     - crypto tests
     - —
     - SC-12, SC-13
   * - REQ_047 (NIST Compliance)
     - —
     - compliance/nist_controls.rs
     - compliance tests
     - —
     - All families (SSP)
