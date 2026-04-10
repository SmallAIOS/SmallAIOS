// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! HTTP endpoint handlers for the SmallAIOS container runtime.
//!
//! Provides inference, model registry, health, readiness, and metrics endpoints.
//! All handlers are standalone functions that accept an [`HttpRequest`] and
//! optional [`ModelManager`] reference, returning an [`HttpResponse`].

use crate::json::JsonValue;
use crate::model_manager::ModelManager;
use crate::server::{HttpRequest, HttpResponse};

/// POST /v1/inference — run inference on a named model.
///
/// Parses the JSON body to extract the `"model"` field, looks it up in the
/// manager, and returns model metadata. Full inference execution is pending
/// the protobuf parser implementation (`Session::load_model()` currently
/// returns `NotImplemented`).
pub fn handle_inference(req: &HttpRequest, manager: &ModelManager) -> HttpResponse {
    if req.method != "POST" {
        return HttpResponse::json(405, r#"{"error":"method not allowed, use POST"}"#);
    }

    let body_str = core::str::from_utf8(&req.body).unwrap_or("");
    let json = match JsonValue::parse(body_str) {
        Ok(j) => j,
        Err(e) => {
            return HttpResponse::json(400, &format!(r#"{{"error":"invalid JSON: {}"}}"#, e));
        }
    };

    let model_name = match json.get("model").and_then(|v| v.as_str()) {
        Some(n) => n,
        None => {
            return HttpResponse::json(400, r#"{"error":"missing 'model' field"}"#);
        }
    };

    let start = std::time::Instant::now();

    match manager.get_model(model_name) {
        Some(info) => {
            let elapsed_us = start.elapsed().as_micros();
            HttpResponse::json(
                200,
                &format!(
                    r#"{{"model":"{}","file_size":{},"status":"inference endpoint ready, model execution pending protobuf parser","latency_us":{}}}"#,
                    info.name, info.file_size, elapsed_us
                ),
            )
        }
        None => HttpResponse::json(
            404,
            &format!(r#"{{"error":"model '{}' not found"}}"#, model_name),
        ),
    }
}

/// GET /v1/models — list all loaded models.
pub fn handle_list_models(manager: &ModelManager) -> HttpResponse {
    let models = manager.list_models();
    let entries: Vec<String> = models
        .iter()
        .map(|m| {
            format!(
                r#"{{"name":"{}","file_size":{},"loaded":{}}}"#,
                m.name, m.file_size, m.loaded
            )
        })
        .collect();
    HttpResponse::json(200, &format!("[{}]", entries.join(",")))
}

/// GET /v1/models/{name} — get info for a specific model.
pub fn handle_get_model(req: &HttpRequest, manager: &ModelManager) -> HttpResponse {
    let name = req.path.strip_prefix("/v1/models/").unwrap_or("");
    if name.is_empty() {
        return HttpResponse::json(400, r#"{"error":"missing model name in path"}"#);
    }
    match manager.get_model(name) {
        Some(info) => HttpResponse::json(
            200,
            &format!(
                r#"{{"name":"{}","file_size":{},"loaded":{},"file_path":"{}"}}"#,
                info.name, info.file_size, info.loaded, info.file_path
            ),
        ),
        None => HttpResponse::json(404, &format!(r#"{{"error":"model '{}' not found"}}"#, name)),
    }
}

/// GET /healthz — liveness probe.
pub fn handle_health() -> HttpResponse {
    HttpResponse::json(200, r#"{"status":"healthy"}"#)
}

/// GET /readyz — readiness probe (ready when at least one model is loaded).
pub fn handle_ready(manager: &ModelManager) -> HttpResponse {
    if manager.model_count() > 0 {
        HttpResponse::json(200, r#"{"status":"ready"}"#)
    } else {
        HttpResponse::json(503, r#"{"status":"not_ready","reason":"no models loaded"}"#)
    }
}

/// GET /metrics — Prometheus-format metrics (stub).
pub fn handle_metrics() -> HttpResponse {
    HttpResponse::text(
        200,
        "# HELP models_loaded Number of loaded models\n\
         # TYPE models_loaded gauge\n\
         models_loaded 0\n\
         # HELP inference_requests_total Total inference requests\n\
         # TYPE inference_requests_total counter\n\
         inference_requests_total 0\n",
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an [`HttpRequest`] with the given method, path, and body.
    fn make_request(method: &str, path: &str, body: &[u8]) -> HttpRequest {
        HttpRequest {
            method: method.to_string(),
            path: path.to_string(),
            headers: vec![],
            body: body.to_vec(),
        }
    }

    /// Create a [`ModelManager`] with no models (empty temp directory).
    fn empty_manager() -> (ModelManager, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("create temp dir");
        let mut mgr = ModelManager::new(dir.path().to_str().unwrap());
        mgr.load_directory();
        (mgr, dir)
    }

    /// Create a [`ModelManager`] with one valid model.
    fn manager_with_model() -> (ModelManager, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("create temp dir");
        // Valid ONNX header: starts with 0x08 and >= 8 bytes.
        let data: Vec<u8> = vec![0x08, 0x07, 0x12, 0x04, b't', b'e', b's', b't'];
        std::fs::write(dir.path().join("resnet.onnx"), &data).unwrap();
        let mut mgr = ModelManager::new(dir.path().to_str().unwrap());
        mgr.load_directory();
        (mgr, dir)
    }

    // --- Inference ---

    #[test]
    fn test_handle_inference_missing_model_field() {
        let (mgr, _dir) = empty_manager();
        let req = make_request("POST", "/v1/inference", br#"{"data":"test"}"#);
        let resp = handle_inference(&req, &mgr);
        assert_eq!(resp.status, 400);
        let body = String::from_utf8_lossy(&resp.body);
        assert!(body.contains("missing 'model' field"));
    }

    #[test]
    fn test_handle_inference_invalid_json() {
        let (mgr, _dir) = empty_manager();
        let req = make_request("POST", "/v1/inference", b"not json");
        let resp = handle_inference(&req, &mgr);
        assert_eq!(resp.status, 400);
        let body = String::from_utf8_lossy(&resp.body);
        assert!(body.contains("invalid JSON"));
    }

    #[test]
    fn test_handle_inference_model_not_found() {
        let (mgr, _dir) = empty_manager();
        let req = make_request("POST", "/v1/inference", br#"{"model":"nonexistent"}"#);
        let resp = handle_inference(&req, &mgr);
        assert_eq!(resp.status, 404);
        let body = String::from_utf8_lossy(&resp.body);
        assert!(body.contains("not found"));
    }

    #[test]
    fn test_handle_inference_valid_request() {
        let (mgr, _dir) = manager_with_model();
        let req = make_request("POST", "/v1/inference", br#"{"model":"resnet"}"#);
        let resp = handle_inference(&req, &mgr);
        assert_eq!(resp.status, 200);
        let body = String::from_utf8_lossy(&resp.body);
        assert!(body.contains("resnet"));
        assert!(body.contains("inference endpoint ready"));
    }

    #[test]
    fn test_handle_inference_wrong_method() {
        let (mgr, _dir) = empty_manager();
        let req = make_request("GET", "/v1/inference", b"");
        let resp = handle_inference(&req, &mgr);
        assert_eq!(resp.status, 405);
    }

    // --- Model listing ---

    #[test]
    fn test_handle_list_models_empty() {
        let (mgr, _dir) = empty_manager();
        let resp = handle_list_models(&mgr);
        assert_eq!(resp.status, 200);
        let body = String::from_utf8_lossy(&resp.body);
        assert_eq!(body, "[]");
    }

    #[test]
    fn test_handle_list_models_with_model() {
        let (mgr, _dir) = manager_with_model();
        let resp = handle_list_models(&mgr);
        assert_eq!(resp.status, 200);
        let body = String::from_utf8_lossy(&resp.body);
        assert!(body.contains("resnet"));
        assert!(body.contains("file_size"));
    }

    // --- Get model ---

    #[test]
    fn test_handle_get_model_found() {
        let (mgr, _dir) = manager_with_model();
        let req = make_request("GET", "/v1/models/resnet", b"");
        let resp = handle_get_model(&req, &mgr);
        assert_eq!(resp.status, 200);
        let body = String::from_utf8_lossy(&resp.body);
        assert!(body.contains("resnet"));
        assert!(body.contains("file_path"));
    }

    #[test]
    fn test_handle_get_model_not_found() {
        let (mgr, _dir) = empty_manager();
        let req = make_request("GET", "/v1/models/nonexistent", b"");
        let resp = handle_get_model(&req, &mgr);
        assert_eq!(resp.status, 404);
    }

    #[test]
    fn test_handle_get_model_missing_name() {
        let (mgr, _dir) = empty_manager();
        let req = make_request("GET", "/v1/models/", b"");
        let resp = handle_get_model(&req, &mgr);
        assert_eq!(resp.status, 400);
    }

    // --- Health ---

    #[test]
    fn test_handle_health() {
        let resp = handle_health();
        assert_eq!(resp.status, 200);
        let body = String::from_utf8_lossy(&resp.body);
        assert!(body.contains("healthy"));
    }

    // --- Ready ---

    #[test]
    fn test_handle_ready_no_models() {
        let (mgr, _dir) = empty_manager();
        let resp = handle_ready(&mgr);
        assert_eq!(resp.status, 503);
        let body = String::from_utf8_lossy(&resp.body);
        assert!(body.contains("not_ready"));
    }

    #[test]
    fn test_handle_ready_with_models() {
        let (mgr, _dir) = manager_with_model();
        let resp = handle_ready(&mgr);
        assert_eq!(resp.status, 200);
        let body = String::from_utf8_lossy(&resp.body);
        assert!(body.contains("ready"));
    }

    // --- Metrics ---

    #[test]
    fn test_handle_metrics() {
        let resp = handle_metrics();
        assert_eq!(resp.status, 200);
        assert_eq!(resp.content_type, "text/plain");
        let body = String::from_utf8_lossy(&resp.body);
        assert!(body.contains("models_loaded"));
        assert!(body.contains("inference_requests_total"));
        assert!(body.contains("# HELP"));
        assert!(body.contains("# TYPE"));
    }
}
