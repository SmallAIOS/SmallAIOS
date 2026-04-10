// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Model manager for SmallAIOS container runtime.
//!
//! Scans a directory for `.onnx` files, validates them, and tracks metadata.
//! Uses `std` for filesystem access -- this module lives on the binary side,
//! not in the `#![no_std]` library crate.

use std::collections::BTreeMap;
use std::fs;

/// Metadata about a loaded (or failed-to-load) ONNX model.
pub struct ModelInfo {
    /// Model name derived from the file stem.
    pub name: String,
    /// Absolute path to the `.onnx` file.
    pub file_path: String,
    /// Size of the file in bytes.
    pub file_size: u64,
    /// Whether the model was successfully loaded and validated.
    pub loaded: bool,
    /// Error message if loading/validation failed.
    pub error: Option<String>,
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
            if path.extension().map(|e| e == "onnx").unwrap_or(false) {
                let name = path
                    .file_stem()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let file_size = entry.metadata().map(|m| m.len()).unwrap_or(0);

                // Try to read and do a basic header check.
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
                    "  Model '{}': {} bytes {}",
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
    }
}
