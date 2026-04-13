// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Model manager for SmallAIOS container runtime.
//!
//! Scans a directory for `.onnx` files and HuggingFace safetensors
//! model directories, validates them, and tracks metadata. Uses `std`
//! for filesystem access -- this module lives on the binary side, not
//! in the `#![no_std]` library crate.

use std::collections::BTreeMap;
use std::fs;

/// Discriminator identifying the on-disk format of a discovered model.
///
/// `Onnx` is a single `.onnx` protobuf file (the historical path);
/// `Safetensors` is a HuggingFace model directory containing a
/// `config.json` plus either `model.safetensors` or
/// `model.safetensors.index.json` (Section 11 of
/// `safetensors-model-loader-v1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelKind {
    Onnx,
    Safetensors,
}

/// Metadata about a loaded (or failed-to-load) model.
pub struct ModelInfo {
    /// Model name derived from the file stem (for ONNX) or the
    /// directory name (for safetensors).
    pub name: String,
    /// Absolute path to the `.onnx` file (`ModelKind::Onnx`) or the
    /// safetensors model directory (`ModelKind::Safetensors`).
    pub file_path: String,
    /// Size in bytes of the ONNX file, or the sum of all safetensors
    /// shards' sizes for safetensors directories.
    pub file_size: u64,
    /// Whether the model was successfully discovered and validated.
    pub loaded: bool,
    /// Error message if loading/validation failed.
    pub error: Option<String>,
    /// On-disk format of the model.
    pub kind: ModelKind,
}

/// Manages ONNX model discovery and metadata tracking.
pub struct ModelManager {
    models: BTreeMap<String, ModelInfo>,
    model_dir: String,
}

impl ModelManager {
    /// Create a new `ModelManager` that will scan the given directory.
    pub fn new(model_dir: &str) -> Self {
        Self {
            models: BTreeMap::new(),
            model_dir: model_dir.to_string(),
        }
    }

    /// Scan `model_dir` for `.onnx` files, validate each, and return the
    /// number of models that loaded successfully.
    pub fn load_directory(&mut self) -> usize {
        let dir = match fs::read_dir(&self.model_dir) {
            Ok(d) => d,
            Err(e) => {
                eprintln!(
                    "WARNING: could not read model directory '{}': {}",
                    self.model_dir, e
                );
                return 0;
            }
        };

        let mut count = 0;
        for entry in dir.flatten() {
            let path = entry.path();

            // ONNX file path.
            if path.extension().map(|e| e == "onnx").unwrap_or(false) {
                let name = path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let file_size = entry.metadata().map(|m| m.len()).unwrap_or(0);

                // Real ONNX files start with a protobuf varint field tag;
                // field 1 (ir_version) wire-type 0 => tag byte 0x08.
                let (loaded, error) = match fs::read(&path) {
                    Ok(data) => {
                        if data.len() >= 8 && data[0] == 0x08 {
                            (true, None)
                        } else {
                            (false, Some("invalid ONNX magic byte".to_string()))
                        }
                    }
                    Err(e) => (false, Some(format!("read error: {}", e))),
                };

                if loaded {
                    count += 1;
                }
                println!(
                    "  Model '{}': {} bytes [onnx] {}",
                    name,
                    file_size,
                    if loaded { "OK" } else { "FAILED" }
                );

                self.models.insert(
                    name.clone(),
                    ModelInfo {
                        name,
                        file_path: path.to_string_lossy().to_string(),
                        file_size,
                        loaded,
                        error,
                        kind: ModelKind::Onnx,
                    },
                );
                continue;
            }

            // Safetensors model directory path.
            if path.is_dir() && is_safetensors_dir(&path) {
                let name = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let (total_size, error) = safetensors_total_size(&path);
                let loaded = error.is_none();
                if loaded {
                    count += 1;
                }
                println!(
                    "  Model '{}': {} bytes [safetensors] {}",
                    name,
                    total_size,
                    if loaded { "OK" } else { "FAILED" }
                );
                self.models.insert(
                    name.clone(),
                    ModelInfo {
                        name,
                        file_path: path.to_string_lossy().to_string(),
                        file_size: total_size,
                        loaded,
                        error,
                        kind: ModelKind::Safetensors,
                    },
                );
            }
        }
        count
    }

    /// Look up a successfully-loaded model by name.
    pub fn get_model(&self, name: &str) -> Option<&ModelInfo> {
        self.models.get(name).filter(|m| m.loaded)
    }

    /// Return all successfully-loaded models.
    pub fn list_models(&self) -> Vec<&ModelInfo> {
        self.models.values().filter(|m| m.loaded).collect()
    }

    /// Count of successfully-loaded models.
    pub fn model_count(&self) -> usize {
        self.models.values().filter(|m| m.loaded).count()
    }
}

/// Returns `true` if `path` looks like a HuggingFace safetensors model
/// directory: it must contain a `config.json` AND either a single
/// `model.safetensors` OR a `model.safetensors.index.json` manifest.
fn is_safetensors_dir(path: &std::path::Path) -> bool {
    if !path.is_dir() {
        return false;
    }
    let has_config = path.join("config.json").is_file();
    if !has_config {
        return false;
    }
    let has_single = path.join("model.safetensors").is_file();
    let has_index = path.join("model.safetensors.index.json").is_file();
    has_single || has_index
}

/// Sum the byte sizes of all weight files in a safetensors directory.
/// Returns `(total_bytes, error)`. On error, `total_bytes` is `0` and
/// `error` contains a human-readable reason.
fn safetensors_total_size(path: &std::path::Path) -> (u64, Option<String>) {
    let single = path.join("model.safetensors");
    let index = path.join("model.safetensors.index.json");
    if single.is_file() {
        match fs::metadata(&single) {
            Ok(m) => (m.len(), None),
            Err(e) => (0, Some(format!("stat model.safetensors: {}", e))),
        }
    } else if index.is_file() {
        // Sum every `*.safetensors` sibling file. We don't parse the
        // index JSON at discovery time — the loader will do that when
        // the session actually constructs.
        let mut total: u64 = 0;
        let rd = match fs::read_dir(path) {
            Ok(d) => d,
            Err(e) => return (0, Some(format!("read_dir shards: {}", e))),
        };
        for entry in rd.flatten() {
            let p = entry.path();
            if p.extension().map(|e| e == "safetensors").unwrap_or(false) {
                if let Ok(md) = entry.metadata() {
                    total = total.saturating_add(md.len());
                }
            }
        }
        (total, None)
    } else {
        (0, Some("no safetensors weights found".to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Helper: create a temp directory with optional .onnx files.
    fn make_temp_dir(files: &[(&str, &[u8])]) -> tempfile::TempDir {
        let dir = tempfile::tempdir().expect("create temp dir");
        for (name, data) in files {
            let path = dir.path().join(name);
            let mut f = fs::File::create(&path).expect("create file");
            f.write_all(data).expect("write file");
        }
        dir
    }

    /// A minimal valid header: starts with 0x08 and has >= 8 bytes.
    fn valid_onnx_bytes() -> Vec<u8> {
        let mut v = vec![0x08, 0x07]; // ir_version = 7
        v.extend_from_slice(&[0x12, 0x04, b't', b'e', b's', b't']); // producer = "test"
        v
    }

    #[test]
    fn load_empty_directory() {
        let dir = make_temp_dir(&[]);
        let mut mgr = ModelManager::new(dir.path().to_str().unwrap());
        assert_eq!(mgr.load_directory(), 0);
        assert_eq!(mgr.model_count(), 0);
        assert!(mgr.list_models().is_empty());
    }

    #[test]
    fn load_valid_model() {
        let data = valid_onnx_bytes();
        let dir = make_temp_dir(&[("resnet.onnx", &data)]);
        let mut mgr = ModelManager::new(dir.path().to_str().unwrap());
        assert_eq!(mgr.load_directory(), 1);
        assert_eq!(mgr.model_count(), 1);

        let m = mgr.get_model("resnet").expect("model should exist");
        assert_eq!(m.name, "resnet");
        assert!(m.loaded);
        assert!(m.error.is_none());
        assert_eq!(m.file_size, data.len() as u64);
    }

    #[test]
    fn load_invalid_model() {
        let dir = make_temp_dir(&[("bad.onnx", b"not-onnx")]);
        let mut mgr = ModelManager::new(dir.path().to_str().unwrap());
        assert_eq!(mgr.load_directory(), 0);
        assert_eq!(mgr.model_count(), 0);

        // get_model filters to loaded-only, so returns None
        assert!(mgr.get_model("bad").is_none());
    }

    #[test]
    fn load_too_small_model() {
        let dir = make_temp_dir(&[("tiny.onnx", &[0x08, 0x01, 0x00])]);
        let mut mgr = ModelManager::new(dir.path().to_str().unwrap());
        assert_eq!(mgr.load_directory(), 0);
    }

    #[test]
    fn load_mixed_files() {
        let good = valid_onnx_bytes();
        let dir = make_temp_dir(&[
            ("model_a.onnx", &good),
            ("model_b.onnx", b"bad"),
            ("readme.txt", b"ignore me"),
            ("model_c.onnx", &good),
        ]);
        let mut mgr = ModelManager::new(dir.path().to_str().unwrap());
        assert_eq!(mgr.load_directory(), 2);
        assert_eq!(mgr.model_count(), 2);
        assert_eq!(mgr.list_models().len(), 2);

        assert!(mgr.get_model("model_a").is_some());
        assert!(mgr.get_model("model_b").is_none());
        assert!(mgr.get_model("model_c").is_some());
    }

    #[test]
    fn nonexistent_directory() {
        let mut mgr = ModelManager::new("/tmp/nonexistent-smallaios-test-dir-xyz");
        assert_eq!(mgr.load_directory(), 0);
    }

    #[test]
    fn model_info_fields() {
        let data = valid_onnx_bytes();
        let dir = make_temp_dir(&[("test.onnx", &data)]);
        let mut mgr = ModelManager::new(dir.path().to_str().unwrap());
        mgr.load_directory();

        let m = mgr.get_model("test").unwrap();
        assert!(m.file_path.ends_with("test.onnx"));
        assert_eq!(m.file_size, data.len() as u64);
        assert_eq!(m.kind, ModelKind::Onnx);
    }

    // ── Section 11 — safetensors directory detection ─────────────────

    /// Build a minimal safetensors directory layout under `root` with a
    /// given name. Creates `config.json` and a zero-length
    /// `model.safetensors` (content is irrelevant at discovery time).
    fn make_safetensors_dir(root: &std::path::Path, name: &str, with_index: bool) {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).expect("mkdir safetensors");
        std::fs::write(dir.join("config.json"), b"{}").expect("write config.json");
        if with_index {
            std::fs::write(
                dir.join("model.safetensors.index.json"),
                b"{\"weight_map\":{}}",
            )
            .expect("write index");
            // Add at least one shard so total size > 0.
            std::fs::write(dir.join("model-00001-of-00001.safetensors"), b"shard")
                .expect("write shard");
        } else {
            std::fs::write(dir.join("model.safetensors"), b"hello").expect("write weights");
        }
    }

    #[test]
    fn detect_single_file_safetensors_dir() {
        let root = tempfile::tempdir().expect("tempdir");
        make_safetensors_dir(root.path(), "gemma-270m", false);
        let mut mgr = ModelManager::new(root.path().to_str().unwrap());
        assert_eq!(mgr.load_directory(), 1);

        let m = mgr.get_model("gemma-270m").expect("safetensors model");
        assert_eq!(m.kind, ModelKind::Safetensors);
        assert!(m.loaded);
        assert_eq!(m.file_size, 5); // "hello"
        assert!(m.file_path.ends_with("gemma-270m"));
    }

    #[test]
    fn detect_sharded_safetensors_dir() {
        let root = tempfile::tempdir().expect("tempdir");
        make_safetensors_dir(root.path(), "gemma-31b", true);
        let mut mgr = ModelManager::new(root.path().to_str().unwrap());
        assert_eq!(mgr.load_directory(), 1);

        let m = mgr.get_model("gemma-31b").expect("sharded model");
        assert_eq!(m.kind, ModelKind::Safetensors);
        assert!(m.loaded);
        assert!(m.file_size > 0, "shard sizes must be summed");
    }

    #[test]
    fn directory_without_config_is_not_safetensors() {
        let root = tempfile::tempdir().expect("tempdir");
        let dir = root.path().join("random-dir");
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(dir.join("model.safetensors"), b"x").expect("write");
        // Missing config.json -> should NOT be detected.
        let mut mgr = ModelManager::new(root.path().to_str().unwrap());
        assert_eq!(mgr.load_directory(), 0);
    }

    #[test]
    fn directory_with_config_but_no_weights_is_not_safetensors() {
        let root = tempfile::tempdir().expect("tempdir");
        let dir = root.path().join("configless");
        std::fs::create_dir_all(&dir).expect("mkdir");
        std::fs::write(dir.join("config.json"), b"{}").expect("write");
        let mut mgr = ModelManager::new(root.path().to_str().unwrap());
        assert_eq!(mgr.load_directory(), 0);
    }

    #[test]
    fn mixed_onnx_and_safetensors_coexist() {
        let root = tempfile::tempdir().expect("tempdir");
        // ONNX file
        let onnx = valid_onnx_bytes();
        std::fs::write(root.path().join("resnet.onnx"), &onnx).expect("onnx");
        // Safetensors dir
        make_safetensors_dir(root.path(), "gemma-4-270m", false);

        let mut mgr = ModelManager::new(root.path().to_str().unwrap());
        assert_eq!(mgr.load_directory(), 2);
        assert_eq!(mgr.model_count(), 2);

        let onnx_m = mgr.get_model("resnet").expect("onnx");
        assert_eq!(onnx_m.kind, ModelKind::Onnx);
        let st_m = mgr.get_model("gemma-4-270m").expect("safetensors");
        assert_eq!(st_m.kind, ModelKind::Safetensors);
    }
}
