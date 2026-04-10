// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end CAN backend integration tests for the container binary.
//!
//! These tests are scaffolded but `#[ignore]`-gated pending completion
//! of the `CanInferenceAdapter` (bus crate, task groups 1-4 of the
//! can-inference-bridge-v1 OpenSpec change). They document the expected
//! behavior once the bus-crate adapter and container wiring are both in.
//!
//! Activate with:
//!
//! ```text
//! cargo test -p smallaios-container --test e2e_can -- --ignored
//! ```

#![allow(clippy::needless_return)]

/// Once the CAN backend is fully wired, spawning the container with
/// `SMALLAIOS_BUS_BACKEND=can` and `SMALLAIOS_CAN_DEVICE=loopback`
/// should start the CAN dataflow runner AND keep the HTTP server
/// responsive on `/healthz`.
#[test]
#[ignore = "pending CanInferenceAdapter wiring (bus task groups 1-4)"]
fn test_bus_mode_can_loopback() {
    // TODO(can-inference-bridge-v1 §8.1):
    //   1. spawn the container binary as a subprocess with
    //      SMALLAIOS_BUS_BACKEND=can, SMALLAIOS_CAN_DEVICE=loopback,
    //      SMALLAIOS_CAN_ROUTING=examples/can-routes.toml
    //   2. poll GET /healthz until it returns 200
    //   3. publish 6 synthetic sensor frames on the loopback controller
    //   4. observe matching output frames produced by the adapter
    //   5. assert HTTP server is still responsive afterwards
}

/// The container must parse `SMALLAIOS_CAN_DEVICE` into a
/// `CanDeviceSpec` and log-and-disable on invalid input rather than
/// crashing (per container-inference-server spec).
#[test]
#[ignore = "pending container subprocess harness"]
fn test_can_device_parse() {
    // TODO(can-inference-bridge-v1 §5.1):
    //   exercise the three valid forms (loopback, mcp2515:<path>,
    //   axi:0x<hex>) plus at least one invalid form, and assert the
    //   container either boots cleanly or logs "invalid SMALLAIOS_CAN_DEVICE".
    //   Unit-level coverage for `parse_can_device` already lives in
    //   `container/src/main.rs` under `#[cfg(test)] mod tests`.
}

/// When `SMALLAIOS_CAN_ROUTING` points at a TOML file, the container
/// must log the routing file path during startup.
#[test]
#[ignore = "pending container subprocess harness"]
fn test_can_routing_file_load() {
    // TODO(can-inference-bridge-v1 §5.2):
    //   1. write a minimal routing TOML into a tempdir
    //   2. spawn the container with SMALLAIOS_CAN_ROUTING pointed at it
    //   3. capture stdout and assert the routing path appears in the
    //      "Starting CAN dataflow runner" line
    //   4. once the adapter exists, also assert the routes are loaded
    //      (check that an InferenceAdapter was attached via metrics
    //      or a debug endpoint).
}
