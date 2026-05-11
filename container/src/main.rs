// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SmallAIOS container binary entry point.
//!
//! This is the Linux userspace binary that runs when SmallAIOS is deployed
//! as a container (Docker/K8s). It bootstraps the unikernel subsystems
//! using the host kernel's syscall interface via musl libc.

#[cfg(feature = "audit-export")]
#[allow(dead_code)]
mod audit_export_runtime;
#[allow(dead_code)]
mod handlers;
#[allow(dead_code)]
mod json;
#[allow(dead_code)]
mod model_manager;
#[allow(dead_code)]
mod server;

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use smallaios_bus::can::controller::{CanController, CanMode, MockCanController};
use smallaios_bus::can::inference_adapter::{AdapterConfig, CanInferenceAdapter};
use smallaios_ipc::dataflow_runner::{DataflowRunner, DataflowRunnerConfig};
use smallaios_onnx_rt::session::{self, Session, SessionConfig};

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

    // Initialize CUDA runtime if requested.
    #[cfg(feature = "cuda")]
    let cuda_runtime = if gpu_backend == "cuda" {
        let precision_str =
            std::env::var("SMALLAIOS_GPU_PRECISION").unwrap_or_else(|_| "tf32".to_string());
        let precision = smallaios_onnx_rt::cuda::GpuPrecision::from_str(&precision_str);
        println!("GPU precision mode: {:?}", precision);
        match smallaios_onnx_rt::cuda::CudaRuntime::init_with_precision(precision) {
            Ok(rt) => {
                println!(
                    "CUDA initialized: {} (compute {}.{}, {} MB VRAM, CUDA {})",
                    rt.device.name,
                    rt.device.compute_major,
                    rt.device.compute_minor,
                    rt.device.total_mem_bytes / (1024 * 1024),
                    rt.cuda_version / 1000,
                );
                Some(std::sync::Arc::new(rt))
            }
            Err(e) => {
                eprintln!(
                    "WARNING: GPU backend=cuda requested but CUDA init failed: {}",
                    e
                );
                eprintln!("  Falling back to CPU inference");
                None
            }
        }
    } else {
        None
    };

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

    // Build the inference session map for the HTTP handler. This
    // parses each .onnx file once at startup so request handling is
    // pure dispatch. Failures are logged by `load_sessions` itself.
    let http_sessions: Arc<BTreeMap<String, Session>> = Arc::new(load_sessions(
        &manager,
        #[cfg(feature = "cuda")]
        &cuda_runtime,
    ));
    println!("HTTP inference sessions ready: {}", http_sessions.len());

    // Bus/dataflow runner startup (zenoh/dds/can/none).
    let runner_handles =
        enable_dataflow_runner(&bus_backend, Arc::clone(&manager), Arc::clone(&shutdown));

    // Build HTTP server and register routes.
    let addr = format!("0.0.0.0:{}", port);
    let mut http = match server::HttpServer::bind(&addr, Arc::clone(&shutdown)) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR: failed to bind {}: {}", addr, e);
            std::process::exit(1);
        }
    };

    // Inference — use the real executor-backed handler if any session
    // loaded, otherwise fall back to the metadata-only handler that
    // still reports 404 / 400 for missing or malformed requests.
    if !http_sessions.is_empty() {
        let sessions = Arc::clone(&http_sessions);
        http.route_fn("POST", "/v1/inference", move |req| {
            handlers::handle_inference_exec(req, &sessions)
        });
    } else {
        let mgr = Arc::clone(&manager);
        http.route_fn("POST", "/v1/inference", move |req| {
            handlers::handle_inference(req, &mgr)
        });
    }

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

    // Ensure the shutdown flag is set so runner threads can observe it
    // (the HTTP server loop already sets it on SIGTERM, but be defensive
    // in case we fall out of `run()` via some other code path).
    shutdown.store(true, Ordering::Relaxed);
    for handle in runner_handles {
        let _ = handle.join();
    }
}

/// Load every `ModelInfo` tracked by the manager into an initialised
/// ONNX `Session`, keyed by model name.
///
/// Models that fail to read, parse, or initialise are logged and
/// skipped; the runner can still start with the remaining sessions.
fn load_sessions(
    manager: &model_manager::ModelManager,
    #[cfg(feature = "cuda")] cuda_runtime: &Option<
        std::sync::Arc<smallaios_onnx_rt::cuda::CudaRuntime>,
    >,
) -> BTreeMap<String, Session> {
    let mut sessions = BTreeMap::new();
    for info in manager.list_models() {
        if !info.loaded {
            continue;
        }

        match info.kind {
            model_manager::ModelKind::Onnx => {
                if let Some(sess) = load_onnx_session(
                    info,
                    #[cfg(feature = "cuda")]
                    cuda_runtime,
                ) {
                    sessions.insert(info.name.clone(), sess);
                }
            }
            model_manager::ModelKind::Safetensors => {
                // Safetensors sessions REQUIRE a live CudaRuntime. If
                // the container was built without `cuda` or failed to
                // init, we fail fast on this model but keep loading
                // the rest (§11.5).
                #[cfg(all(feature = "cuda", feature = "safetensors"))]
                {
                    if let Some(rt) = cuda_runtime.as_ref() {
                        match Session::from_safetensors(
                            std::path::Path::new(&info.file_path),
                            rt.clone(),
                        ) {
                            Ok(sess) => {
                                println!("  Session ready: {} [GPU/safetensors]", info.name);
                                sessions.insert(info.name.clone(), sess);
                            }
                            Err(e) => {
                                eprintln!(
                                    "  load_sessions: failed safetensors load for '{}': {}",
                                    info.name, e
                                );
                            }
                        }
                    } else {
                        eprintln!(
                            "  load_sessions: model '{}' is safetensors but no \
                             CudaRuntime is available — skipping (requires \
                             SMALLAIOS_GPU_BACKEND=cuda and a working GPU).",
                            info.name
                        );
                    }
                }
                #[cfg(not(all(feature = "cuda", feature = "safetensors")))]
                {
                    eprintln!(
                        "  load_sessions: model '{}' is safetensors but this \
                         container build lacks the `cuda,safetensors` features \
                         — skipping.",
                        info.name
                    );
                }
            }
        }
    }
    sessions
}

/// Load a single ONNX session from a `ModelInfo` of kind `Onnx`.
fn load_onnx_session(
    info: &model_manager::ModelInfo,
    #[cfg(feature = "cuda")] cuda_runtime: &Option<
        std::sync::Arc<smallaios_onnx_rt::cuda::CudaRuntime>,
    >,
) -> Option<Session> {
    let bytes = match std::fs::read(&info.file_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("  load_sessions: failed to read {}: {}", info.file_path, e);
            return None;
        }
    };
    let model = match session::load_model(&bytes) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("  load_sessions: failed to parse {}: {}", info.name, e);
            return None;
        }
    };
    let mut s = Session::new(SessionConfig::default());
    #[cfg(feature = "cuda")]
    {
        s.cuda_runtime = cuda_runtime.clone();
    }
    match s.initialize(&model) {
        Ok(()) => {
            #[cfg(feature = "cuda")]
            let gpu_tag = if cuda_runtime.is_some() { " [GPU]" } else { "" };
            #[cfg(not(feature = "cuda"))]
            let gpu_tag = "";
            println!("  Session ready: {}{}", info.name, gpu_tag);
            Some(s)
        }
        Err(e) => {
            eprintln!("  load_sessions: failed to initialize {}: {}", info.name, e);
            None
        }
    }
}

/// Build a [`DataflowRunner`] populated with sessions from the manager.
///
/// Returns `None` if no sessions could be loaded — callers use this as
/// a signal to skip spawning the runner thread entirely.
fn build_runner(manager: &model_manager::ModelManager) -> Option<DataflowRunner> {
    let sessions = load_sessions(
        manager,
        #[cfg(feature = "cuda")]
        &None::<std::sync::Arc<smallaios_onnx_rt::cuda::CudaRuntime>>,
    );
    if sessions.is_empty() {
        eprintln!("  no models loaded — runner will not start");
        return None;
    }
    let mut runner = DataflowRunner::new(DataflowRunnerConfig::default());
    for (name, sess) in sessions {
        runner.register_session(name, sess);
    }
    Some(runner)
}

/// Spawn a background thread that owns a Zenoh-backed
/// [`DataflowRunner`]. The runner currently runs in-process only: the
/// external Zenoh wire protocol wire-up is deferred to a follow-up
/// change. Until then the thread simply holds onto the runner and
/// polls the shutdown flag, proving that the container can bring the
/// runner up alongside the HTTP server without crashing.
fn start_zenoh_dataflow_runner(
    manager: Arc<model_manager::ModelManager>,
    shutdown: Arc<AtomicBool>,
) -> Option<JoinHandle<()>> {
    let runner = build_runner(&manager)?;
    let models = runner.model_names();
    println!(
        "  Zenoh runner ready: {} model(s) registered: {:?}",
        models.len(),
        models
    );
    Some(std::thread::spawn(move || {
        // The runner is held by this thread for its lifetime. External
        // pub/sub wiring (Zenoh TCP transport) is a follow-up change;
        // here we just keep the sessions resident until shutdown fires.
        let _runner = runner;
        while !shutdown.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(50));
        }
        println!("  Zenoh runner thread exiting");
    }))
}

/// Spawn a background thread for the DDS backend. For the initial
/// wire-up the DDS path mirrors the Zenoh path — both sit on the same
/// in-process runner. The real DDS RTPS wire protocol is a follow-up
/// change and, once it lands, this function will bridge it via
/// `bus::dds::DdsZenohAdapter`.
fn start_dds_dataflow_runner(
    manager: Arc<model_manager::ModelManager>,
    shutdown: Arc<AtomicBool>,
) -> Option<JoinHandle<()>> {
    let runner = build_runner(&manager)?;
    let models = runner.model_names();
    println!(
        "  DDS runner ready: {} model(s) registered: {:?}",
        models.len(),
        models
    );
    println!("  NOTE: DDS RTPS wire protocol is not active; runs in-process");
    Some(std::thread::spawn(move || {
        let _runner = runner;
        while !shutdown.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(50));
        }
        println!("  DDS runner thread exiting");
    }))
}

/// Spawn a background thread for the CAN backend. Uses a
/// [`MockCanController`] for loopback; for MCP2515 and AXI specs we
/// also fall back to the mock for now, logging a clear warning. Real
/// hardware bring-up is a separate change (see
/// `can-inference-bridge-v1` deferred tasks).
fn start_can_dataflow_runner(
    manager: Arc<model_manager::ModelManager>,
    shutdown: Arc<AtomicBool>,
    device: CanDeviceSpec,
) -> Option<JoinHandle<()>> {
    let runner = build_runner(&manager)?;
    let models = runner.model_names();
    println!(
        "  CAN runner ready: {} model(s) registered: {:?}",
        models.len(),
        models
    );

    // Always use MockCanController until real drivers are wired in.
    match device {
        CanDeviceSpec::Loopback => {}
        CanDeviceSpec::Mcp2515(ref path) => {
            eprintln!(
                "  WARNING: MCP2515 driver ({}) not wired; using in-process mock",
                path
            );
        }
        CanDeviceSpec::AxiCan(addr) => {
            eprintln!(
                "  WARNING: AXI CAN driver (0x{:x}) not wired; using in-process mock",
                addr
            );
        }
    }

    let mut controller = MockCanController::new();
    if let Err(e) = controller.init() {
        eprintln!("  CAN controller init failed: {:?}", e);
        return None;
    }
    if let Err(e) = controller.set_mode(CanMode::Loopback) {
        eprintln!("  CAN controller set_mode failed: {:?}", e);
        return None;
    }

    // Hard-coded empty routing table — real routing-file loading is a
    // follow-up (see task group 4.2 notes). With no routes, incoming
    // frames are counted under `unrouted_frames_total` and no batches
    // are ever flushed, which is fine for the initial wire-up: the
    // goal is to prove the thread starts cleanly and observes shutdown.
    let mut adapter = CanInferenceAdapter::new(AdapterConfig::default());

    Some(std::thread::spawn(move || {
        let mut timestamp_us: u64 = 0;
        while !shutdown.load(Ordering::Relaxed) {
            // Drain anything the controller has for us.
            loop {
                match controller.receive() {
                    Ok(Some(frame)) => {
                        if let Some((topic, payload)) = adapter.process_frame(&frame, timestamp_us)
                        {
                            // Try to run inference for the topic's model.
                            if let Ok(output_bytes) = runner.process_message(&topic, &payload) {
                                let output_topic = runner.output_topic(&topic);
                                let frames =
                                    adapter.on_inference_output(&output_topic, &output_bytes);
                                for frame in frames {
                                    let _ = controller.transmit(&frame);
                                }
                            }
                        }
                        timestamp_us = timestamp_us.wrapping_add(100);
                    }
                    Ok(None) => break,
                    Err(e) => {
                        eprintln!("  CAN receive error: {:?}", e);
                        break;
                    }
                }
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        println!("  CAN runner thread exiting");
    }))
}

/// Start the pub/sub dataflow runner(s) for the configured bus
/// backend. Returns the set of thread handles the caller must join
/// before exiting.
///
/// Recognised values for `SMALLAIOS_BUS_BACKEND`:
/// - `none`  — HTTP only (default)
/// - `zenoh` — Zenoh-style pub/sub runner
/// - `dds`   — DDS runner (mirrors zenoh in-process for now)
/// - `can`   — CAN dataflow bridge on top of a [`MockCanController`]
fn enable_dataflow_runner(
    bus_backend: &str,
    manager: Arc<model_manager::ModelManager>,
    shutdown: Arc<AtomicBool>,
) -> Vec<JoinHandle<()>> {
    let mut handles = Vec::new();
    match bus_backend {
        "none" => {}
        "zenoh" => {
            println!("Bus: starting Zenoh dataflow runner");
            println!("  Topics: smallaios/inference/<model>/{{input,output,error}}");
            if let Some(h) = start_zenoh_dataflow_runner(manager, shutdown) {
                handles.push(h);
            }
        }
        "dds" => {
            println!("Bus: starting DDS dataflow runner");
            println!(
                "  Topics: bridged via bus::dds::DdsZenohAdapter → smallaios/inference/<model>/..."
            );
            if let Some(h) = start_dds_dataflow_runner(manager, shutdown) {
                handles.push(h);
            }
        }
        "can" => {
            let device =
                std::env::var("SMALLAIOS_CAN_DEVICE").unwrap_or_else(|_| String::from("loopback"));
            let routing = std::env::var("SMALLAIOS_CAN_ROUTING").unwrap_or_default();
            println!(
                "Bus: starting CAN dataflow runner: device={}, routing={}",
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
                    if let Some(h) = start_can_dataflow_runner(manager, shutdown, spec) {
                        handles.push(h);
                    }
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
    handles
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
    use std::io::Write;
    use std::time::Instant;
    use tempfile::TempDir;

    /// Minimal valid ONNX Relu model — copied from
    /// `onnx-rt/tests/test_real_model.rs`. Parses cleanly and
    /// initialises a `Session` without hitting real hardware.
    fn minimal_relu_onnx_bytes() -> &'static [u8] {
        &[
            0x08, 0x07, 0x12, 0x0c, 0x62, 0x61, 0x63, 0x6b, 0x65, 0x6e, 0x64, 0x2d, 0x74, 0x65,
            0x73, 0x74, 0x3a, 0x4b, 0x0a, 0x0c, 0x0a, 0x01, 0x78, 0x12, 0x01, 0x79, 0x22, 0x04,
            0x52, 0x65, 0x6c, 0x75, 0x12, 0x09, 0x74, 0x65, 0x73, 0x74, 0x5f, 0x72, 0x65, 0x6c,
            0x75, 0x5a, 0x17, 0x0a, 0x01, 0x78, 0x12, 0x12, 0x0a, 0x10, 0x08, 0x01, 0x12, 0x0c,
            0x0a, 0x02, 0x08, 0x03, 0x0a, 0x02, 0x08, 0x04, 0x0a, 0x02, 0x08, 0x05, 0x62, 0x17,
            0x0a, 0x01, 0x79, 0x12, 0x12, 0x0a, 0x10, 0x08, 0x01, 0x12, 0x0c, 0x0a, 0x02, 0x08,
            0x03, 0x0a, 0x02, 0x08, 0x04, 0x0a, 0x02, 0x08, 0x05, 0x42, 0x04, 0x0a, 0x00, 0x10,
            0x09,
        ]
    }

    /// Helper: build a `ModelManager` rooted at a fresh temp directory
    /// containing a single `relu.onnx` file with a parseable ONNX
    /// Relu model. Returns the manager along with the `TempDir` guard
    /// the caller must keep alive for the duration of the test.
    fn make_temp_model_manager() -> (model_manager::ModelManager, TempDir) {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("relu.onnx");
        let mut f = std::fs::File::create(&path).expect("create .onnx");
        f.write_all(minimal_relu_onnx_bytes()).expect("write bytes");
        drop(f);
        let mut mgr = model_manager::ModelManager::new(dir.path().to_str().unwrap());
        assert_eq!(mgr.load_directory(), 1, "relu model should load");
        (mgr, dir)
    }

    /// Join a runner thread, failing the test if it doesn't exit
    /// within the given wall-clock budget. The runner loops at 5ms /
    /// 50ms sleep intervals so 2s is plenty.
    fn join_with_timeout(handle: JoinHandle<()>, budget: Duration) {
        let start = Instant::now();
        // Poll `is_finished` cheaply so we don't block the test on a
        // misbehaving thread.
        while !handle.is_finished() {
            if start.elapsed() > budget {
                panic!("runner thread did not exit within {:?}", budget);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        handle.join().expect("runner thread panicked");
    }

    #[test]
    fn load_sessions_empty_manager_returns_empty_map() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = model_manager::ModelManager::new(dir.path().to_str().unwrap());
        let sessions = load_sessions(
            &mgr,
            #[cfg(feature = "cuda")]
            &None::<std::sync::Arc<smallaios_onnx_rt::cuda::CudaRuntime>>,
        );
        assert!(sessions.is_empty());
    }

    #[test]
    fn load_sessions_nonexistent_dir_returns_empty_map() {
        // `ModelManager::new` does not touch the filesystem, and with
        // no `load_directory` call `list_models` is empty, so
        // `load_sessions` should return empty without panicking even
        // if the path doesn't exist.
        let mgr = model_manager::ModelManager::new("/nonexistent/smallaios/path/xyz");
        let sessions = load_sessions(
            &mgr,
            #[cfg(feature = "cuda")]
            &None::<std::sync::Arc<smallaios_onnx_rt::cuda::CudaRuntime>>,
        );
        assert!(sessions.is_empty());
    }

    #[test]
    fn load_sessions_with_real_relu_bytes_populates_map() {
        let (mgr, _guard) = make_temp_model_manager();
        let sessions = load_sessions(
            &mgr,
            #[cfg(feature = "cuda")]
            &None::<std::sync::Arc<smallaios_onnx_rt::cuda::CudaRuntime>>,
        );
        assert_eq!(sessions.len(), 1, "expected exactly one loaded session");
        assert!(
            sessions.contains_key("relu"),
            "session map should contain 'relu'"
        );
    }

    /// §11 integration: when the model_dir contains a safetensors
    /// directory but the container was built without the `cuda`
    /// feature (or CUDA init failed), that model must be skipped with
    /// a clear log line. Other models in the same directory should
    /// still load. Runs under default and `safetensors` features.
    #[test]
    fn load_sessions_skips_safetensors_without_cuda() {
        let root = tempfile::tempdir().expect("tempdir");

        // ONNX file that should load.
        std::fs::write(root.path().join("relu.onnx"), minimal_relu_onnx_bytes())
            .expect("write relu");

        // Safetensors directory that should be detected but skipped
        // because the test binary (even with `safetensors`) either
        // lacks `cuda` or has no GPU.
        let st = root.path().join("gemma-fake");
        std::fs::create_dir_all(&st).expect("mkdir");
        std::fs::write(st.join("config.json"), b"{}").expect("config");
        std::fs::write(st.join("model.safetensors"), b"hello").expect("weights");

        let mut mgr = model_manager::ModelManager::new(root.path().to_str().unwrap());
        assert_eq!(mgr.load_directory(), 2, "both models should be discovered");

        let sessions = load_sessions(
            &mgr,
            #[cfg(feature = "cuda")]
            &None::<std::sync::Arc<smallaios_onnx_rt::cuda::CudaRuntime>>,
        );
        // Only the ONNX session loads; the safetensors one is skipped
        // (no CUDA runtime or no `cuda,safetensors` feature combo).
        assert!(
            sessions.contains_key("relu"),
            "relu ONNX session should load"
        );
        assert!(
            !sessions.contains_key("gemma-fake"),
            "safetensors model must be skipped without a CudaRuntime"
        );
    }

    #[test]
    fn build_runner_with_no_models_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = model_manager::ModelManager::new(dir.path().to_str().unwrap());
        assert!(build_runner(&mgr).is_none());
    }

    #[test]
    fn build_runner_with_registered_session_returns_some() {
        let (mgr, _guard) = make_temp_model_manager();
        let runner = build_runner(&mgr).expect("runner should be built");
        let names = runner.model_names();
        assert_eq!(names.len(), 1);
        assert!(names.iter().any(|n| n == "relu"));
    }

    #[test]
    fn start_zenoh_dataflow_runner_shuts_down_cleanly() {
        let (mgr, _guard) = make_temp_model_manager();
        let shutdown = Arc::new(AtomicBool::new(true));
        let handle = start_zenoh_dataflow_runner(Arc::new(mgr), Arc::clone(&shutdown))
            .expect("runner should spawn");
        join_with_timeout(handle, Duration::from_secs(2));
    }

    #[test]
    fn start_zenoh_dataflow_runner_without_models_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = Arc::new(model_manager::ModelManager::new(
            dir.path().to_str().unwrap(),
        ));
        let shutdown = Arc::new(AtomicBool::new(true));
        assert!(start_zenoh_dataflow_runner(mgr, shutdown).is_none());
    }

    #[test]
    fn start_dds_dataflow_runner_shuts_down_cleanly() {
        let (mgr, _guard) = make_temp_model_manager();
        let shutdown = Arc::new(AtomicBool::new(true));
        let handle = start_dds_dataflow_runner(Arc::new(mgr), Arc::clone(&shutdown))
            .expect("runner should spawn");
        join_with_timeout(handle, Duration::from_secs(2));
    }

    #[test]
    fn start_dds_dataflow_runner_without_models_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = Arc::new(model_manager::ModelManager::new(
            dir.path().to_str().unwrap(),
        ));
        let shutdown = Arc::new(AtomicBool::new(true));
        assert!(start_dds_dataflow_runner(mgr, shutdown).is_none());
    }

    #[test]
    fn start_can_dataflow_runner_loopback_shuts_down_cleanly() {
        let (mgr, _guard) = make_temp_model_manager();
        let shutdown = Arc::new(AtomicBool::new(true));
        let handle = start_can_dataflow_runner(
            Arc::new(mgr),
            Arc::clone(&shutdown),
            CanDeviceSpec::Loopback,
        )
        .expect("CAN runner should spawn");
        join_with_timeout(handle, Duration::from_secs(2));
    }

    #[test]
    fn start_can_dataflow_runner_mcp2515_falls_back_to_mock() {
        // MCP2515 driver is not wired yet; the runner should still
        // spawn (using the in-process mock) and exit on shutdown.
        let (mgr, _guard) = make_temp_model_manager();
        let shutdown = Arc::new(AtomicBool::new(true));
        let handle = start_can_dataflow_runner(
            Arc::new(mgr),
            Arc::clone(&shutdown),
            CanDeviceSpec::Mcp2515("/dev/spidev0.0".into()),
        )
        .expect("CAN runner should spawn for mcp2515 stub");
        join_with_timeout(handle, Duration::from_secs(2));
    }

    #[test]
    fn start_can_dataflow_runner_axi_falls_back_to_mock() {
        let (mgr, _guard) = make_temp_model_manager();
        let shutdown = Arc::new(AtomicBool::new(true));
        let handle = start_can_dataflow_runner(
            Arc::new(mgr),
            Arc::clone(&shutdown),
            CanDeviceSpec::AxiCan(0x4000_0000),
        )
        .expect("CAN runner should spawn for axi stub");
        join_with_timeout(handle, Duration::from_secs(2));
    }

    #[test]
    fn start_can_dataflow_runner_without_models_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = Arc::new(model_manager::ModelManager::new(
            dir.path().to_str().unwrap(),
        ));
        let shutdown = Arc::new(AtomicBool::new(true));
        assert!(
            start_can_dataflow_runner(mgr, shutdown, CanDeviceSpec::Loopback).is_none(),
            "no models → no runner"
        );
    }

    #[test]
    fn enable_dataflow_runner_none_returns_empty_vec() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = Arc::new(model_manager::ModelManager::new(
            dir.path().to_str().unwrap(),
        ));
        let shutdown = Arc::new(AtomicBool::new(true));
        let handles = enable_dataflow_runner("none", mgr, shutdown);
        assert!(handles.is_empty());
    }

    #[test]
    fn enable_dataflow_runner_unknown_backend_returns_empty_vec() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = Arc::new(model_manager::ModelManager::new(
            dir.path().to_str().unwrap(),
        ));
        let shutdown = Arc::new(AtomicBool::new(true));
        let handles = enable_dataflow_runner("quic-over-pigeon", mgr, shutdown);
        assert!(handles.is_empty());
    }

    #[test]
    fn enable_dataflow_runner_zenoh_without_models_returns_empty_vec() {
        // `zenoh` + empty manager → start_zenoh_dataflow_runner returns
        // None → no handles pushed.
        let dir = tempfile::tempdir().unwrap();
        let mgr = Arc::new(model_manager::ModelManager::new(
            dir.path().to_str().unwrap(),
        ));
        let shutdown = Arc::new(AtomicBool::new(true));
        let handles = enable_dataflow_runner("zenoh", mgr, shutdown);
        assert!(handles.is_empty());
    }

    #[test]
    fn enable_dataflow_runner_dds_without_models_returns_empty_vec() {
        let dir = tempfile::tempdir().unwrap();
        let mgr = Arc::new(model_manager::ModelManager::new(
            dir.path().to_str().unwrap(),
        ));
        let shutdown = Arc::new(AtomicBool::new(true));
        let handles = enable_dataflow_runner("dds", mgr, shutdown);
        assert!(handles.is_empty());
    }

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
