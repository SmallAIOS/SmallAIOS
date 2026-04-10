// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SmallAIOS container binary entry point.
//!
//! This is the Linux userspace binary that runs when SmallAIOS is deployed
//! as a container (Docker/K8s). It bootstraps the unikernel subsystems
//! using the host kernel's syscall interface via musl libc.

#[allow(dead_code)]
mod handlers;
#[allow(dead_code)]
mod json;
#[allow(dead_code)]
mod model_manager;
#[allow(dead_code)]
mod server;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn main() {
    // Health check mode: quick probe for liveness, then exit.
    if std::env::args().any(|a| a == "--health-check") {
        println!("ok");
        return;
    }

    println!("SmallAIOS {}", env!("CARGO_PKG_VERSION"));
    println!("Container mode: running as Linux userspace process");

    // Read configuration from environment variables.
    let model_dir = std::env::var("SMALLAIOS_MODEL_DIR").unwrap_or_else(|_| "/models".to_string());
    let port = std::env::var("SMALLAIOS_PORT").unwrap_or_else(|_| "8080".to_string());
    let gpu_backend = std::env::var("SMALLAIOS_GPU_BACKEND").unwrap_or_else(|_| "cpu".to_string());
    let bus_backend = std::env::var("SMALLAIOS_BUS_BACKEND").unwrap_or_else(|_| "none".to_string());

    println!(
        "Config: model_dir={}, port={}, gpu={}, bus_backend={}",
        model_dir, port, gpu_backend, bus_backend
    );

    // Boot phase: discover and validate models.
    println!("Loading models from '{}'...", model_dir);
    let mut manager = model_manager::ModelManager::new(&model_dir);
    let count = manager.load_directory();
    println!("Loaded {} model(s)", count);

    // Setup graceful shutdown via atomic flag.
    let shutdown = Arc::new(AtomicBool::new(false));
    setup_signal_handler(Arc::clone(&shutdown));

    // Wrap manager in Arc for sharing with route closures.
    let manager = Arc::new(manager);

    // Bus/dataflow runner startup (zenoh/dds/can/none).
    enable_dataflow_runner(&bus_backend, Arc::clone(&manager));

    // Build HTTP server and register routes.
    let addr = format!("0.0.0.0:{}", port);
    let mut http = match server::HttpServer::bind(&addr, Arc::clone(&shutdown)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR: failed to bind {}: {}", addr, e);
            std::process::exit(1);
        }
    };

    // Inference
    let mgr = Arc::clone(&manager);
    http.route_fn("POST", "/v1/inference", move |req| {
        handlers::handle_inference(req, &mgr)
    });

    // Model registry — specific model must be registered before the list
    // route so the longer prefix matches first.
    let mgr = Arc::clone(&manager);
    http.route_fn("GET", "/v1/models/", move |req| {
        handlers::handle_get_model(req, &mgr)
    });
    let mgr = Arc::clone(&manager);
    http.route_fn("GET", "/v1/models", move |_req| {
        handlers::handle_list_models(&mgr)
    });

    // Health / readiness / metrics
    http.route("GET", "/healthz", |_req| handlers::handle_health());
    let mgr = Arc::clone(&manager);
    http.route_fn("GET", "/readyz", move |_req| handlers::handle_ready(&mgr));
    http.route("GET", "/metrics", |_req| handlers::handle_metrics());

    println!("Ready. Listening on {}", addr);
    http.run();
    println!("Shutting down...");
}

/// Start the pub/sub dataflow runner for the configured bus backend.
///
/// Recognized values for `SMALLAIOS_BUS_BACKEND`:
/// - `none`  — HTTP only (default)
/// - `zenoh` — start a Zenoh-style pub/sub runner (topic
///   `smallaios/inference/<model>/{input,output,error}`)
/// - `dds`   — start a DDS runner via the bus::dds Zenoh adapter
///
/// This is currently a placeholder: the real runner lives behind the
/// `onnx` feature in the `ipc` crate (`dataflow_runner` module), which
/// is still being wired in a parallel change. Once that lands this
/// function will start the runner in a background thread sharing the
/// `ModelManager` Arc and hook its shutdown into the signal handler.
fn enable_dataflow_runner(bus_backend: &str, _manager: Arc<model_manager::ModelManager>) {
    match bus_backend {
        "none" => {}
        "zenoh" => {
            println!(
                "Bus: Zenoh dataflow runner requested \
                 (placeholder — enable once `smallaios-ipc` ships the `onnx` feature)"
            );
            println!("  Topics: smallaios/inference/<model>/{{input,output,error}}");
            // TODO(dataflow-inference-v1 §5.2): start_zenoh_dataflow_runner(_manager);
        }
        "dds" => {
            println!(
                "Bus: DDS dataflow runner requested \
                 (placeholder — enable once `smallaios-ipc` ships the `onnx` feature)"
            );
            println!(
                "  Topics: bridged via bus::dds::DdsZenohAdapter → smallaios/inference/<model>/..."
            );
            // TODO(dataflow-inference-v1 §5.2): start_dds_dataflow_runner(_manager);
        }
        "can" => {
            let device =
                std::env::var("SMALLAIOS_CAN_DEVICE").unwrap_or_else(|_| String::from("loopback"));
            let routing = std::env::var("SMALLAIOS_CAN_ROUTING").unwrap_or_default();
            println!(
                "Bus: CAN dataflow runner requested: device={}, routing={}",
                device,
                if routing.is_empty() {
                    "<none>"
                } else {
                    &routing
                }
            );
            match parse_can_device(&device) {
                Ok(spec) => {
                    println!("  CAN device parsed: {:?}", spec);
                    // TODO(can-inference-bridge-v1 §5.3): instantiate controller, attach adapter
                }
                Err(e) => {
                    eprintln!("ERROR: invalid SMALLAIOS_CAN_DEVICE: {}", e);
                }
            }
        }
        other => {
            eprintln!(
                "WARNING: unknown SMALLAIOS_BUS_BACKEND='{}', falling back to HTTP-only",
                other
            );
        }
    }
}

/// Register a signal handler that sets the shutdown flag on SIGTERM / SIGINT.
///
/// Uses raw `libc::signal` on Unix to avoid pulling in external crates.
/// On non-Unix platforms this is a no-op (the main loop can still be stopped
/// by other means).
fn setup_signal_handler(shutdown: Arc<AtomicBool>) {
    #[cfg(unix)]
    {
        // Store the Arc in a global so the C-style handler can reach it.
        // SAFETY: we only write once before any signal can fire, and reads
        // inside the handler use Relaxed ordering on the AtomicBool.
        use std::sync::OnceLock;
        static SHUTDOWN_FLAG: OnceLock<Arc<AtomicBool>> = OnceLock::new();
        SHUTDOWN_FLAG.get_or_init(|| shutdown);

        extern "C" fn handler(_sig: libc::c_int) {
            if let Some(flag) = SHUTDOWN_FLAG.get() {
                flag.store(true, Ordering::Relaxed);
            }
        }

        unsafe {
            libc::signal(libc::SIGTERM, handler as *const () as libc::sighandler_t);
            libc::signal(libc::SIGINT, handler as *const () as libc::sighandler_t);
        }
    }

    #[cfg(not(unix))]
    {
        let _ = shutdown; // suppress unused-variable warning
    }
}

/// Parsed form of `SMALLAIOS_CAN_DEVICE`.
#[derive(Debug, PartialEq, Eq)]
enum CanDeviceSpec {
    /// In-process loopback for testing / CI.
    Loopback,
    /// MCP2515 SPI controller; inner is the SPI device path.
    Mcp2515(String),
    /// Xilinx AXI CAN IP; inner is the MMIO base address.
    AxiCan(u64),
}

/// Parse `SMALLAIOS_CAN_DEVICE` into a [`CanDeviceSpec`].
///
/// Accepted forms:
/// - `loopback`
/// - `mcp2515:<spi-path>`  e.g. `mcp2515:/dev/spidev0.0`
/// - `axi:<hex-addr>`      e.g. `axi:0x40000000`
fn parse_can_device(spec: &str) -> Result<CanDeviceSpec, String> {
    if spec == "loopback" {
        return Ok(CanDeviceSpec::Loopback);
    }
    if let Some(path) = spec.strip_prefix("mcp2515:") {
        if path.is_empty() {
            return Err("mcp2515: requires an SPI device path".to_string());
        }
        return Ok(CanDeviceSpec::Mcp2515(path.to_string()));
    }
    if let Some(addr) = spec.strip_prefix("axi:") {
        let addr_clean = addr.trim_start_matches("0x").trim_start_matches("0X");
        let parsed = u64::from_str_radix(addr_clean, 16)
            .map_err(|e| format!("invalid hex address '{}': {}", addr, e))?;
        return Ok(CanDeviceSpec::AxiCan(parsed));
    }
    Err(format!("unknown device spec: '{}'", spec))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_can_device_loopback() {
        assert_eq!(
            parse_can_device("loopback").unwrap(),
            CanDeviceSpec::Loopback
        );
    }

    #[test]
    fn parse_can_device_mcp2515() {
        assert_eq!(
            parse_can_device("mcp2515:/dev/spidev0.0").unwrap(),
            CanDeviceSpec::Mcp2515("/dev/spidev0.0".to_string())
        );
    }

    #[test]
    fn parse_can_device_mcp2515_empty_path_errors() {
        assert!(parse_can_device("mcp2515:").is_err());
    }

    #[test]
    fn parse_can_device_axi_hex() {
        assert_eq!(
            parse_can_device("axi:0x40000000").unwrap(),
            CanDeviceSpec::AxiCan(0x4000_0000)
        );
    }

    #[test]
    fn parse_can_device_axi_bare_hex() {
        assert_eq!(
            parse_can_device("axi:deadbeef").unwrap(),
            CanDeviceSpec::AxiCan(0xdead_beef)
        );
    }

    #[test]
    fn parse_can_device_axi_bad_hex_errors() {
        assert!(parse_can_device("axi:not-hex").is_err());
    }

    #[test]
    fn parse_can_device_invalid_prefix_errors() {
        let err = parse_can_device("usb:/dev/ttyUSB0").unwrap_err();
        assert!(err.contains("unknown device spec"));
    }

    #[test]
    fn parse_can_device_empty_errors() {
        assert!(parse_can_device("").is_err());
    }
}
