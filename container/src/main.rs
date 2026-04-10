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

    println!(
        "Config: model_dir={}, port={}, gpu={}",
        model_dir, port, gpu_backend
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
