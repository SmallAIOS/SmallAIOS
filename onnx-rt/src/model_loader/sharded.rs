// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Multi-shard safetensors loader.
//!
//! Large models (e.g. Gemma 4 31B-it at ~62 GB) ship as several
//! safetensors files accompanied by a `model.safetensors.index.json`
//! manifest of the form:
//!
//! ```json
//! {
//!   "metadata": {"total_size": 12345678},
//!   "weight_map": {
//!     "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
//!     "model.layers.0.input_layernorm.weight": "model-00002-of-00002.safetensors"
//!   }
//! }
//! ```
//!
//! `MultiShardSafetensors` parses the index, eagerly opens every shard
//! referenced by `weight_map`, and presents a unified `tensor_view`
//! lookup. Lookups first consult the manifest to find which shard owns
//! the tensor, then delegate to that shard's [`SafetensorsFile`].

use super::safetensors::{parse_json_object, JsonValue, SafetensorsFile, TensorView};
use super::LoaderError;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use std::fs;
use std::path::{Path, PathBuf};

/// Fully-loaded multi-shard safetensors model.
#[derive(Debug)]
pub struct MultiShardSafetensors {
    /// Map from tensor name -> index into `shards`.
    weight_map: BTreeMap<String, usize>,
    /// Unique shard file names, in the order they were first encountered.
    shard_names: Vec<String>,
    /// Opened shards, indexed in parallel with `shard_names`.
    shards: Vec<SafetensorsFile>,
}

impl MultiShardSafetensors {
    /// Open a sharded model given the path to its directory. The
    /// directory is expected to contain `model.safetensors.index.json`
    /// plus every shard file referenced by the index.
    pub fn open_dir<P: AsRef<Path>>(dir: P) -> Result<Self, LoaderError> {
        let dir = dir.as_ref();
        let index_path = dir.join("model.safetensors.index.json");
        let text = fs::read_to_string(&index_path)
            .map_err(|e| LoaderError::Io(format!("read {}: {e}", index_path.display())))?;
        Self::open_with_index(dir, &text)
    }

    /// Open a sharded model given the directory and the raw JSON text
    /// of its `model.safetensors.index.json`. Exposed separately so
    /// tests can inject an in-memory index.
    pub fn open_with_index(dir: &Path, index_json: &str) -> Result<Self, LoaderError> {
        let top = parse_json_object(index_json)?;
        let weight_map_val = top
            .into_iter()
            .find(|(k, _)| k == "weight_map")
            .map(|(_, v)| v)
            .ok_or_else(|| LoaderError::InvalidHeader("index missing weight_map".to_string()))?;
        let weight_map_obj = match weight_map_val {
            JsonValue::Object(o) => o,
            _ => {
                return Err(LoaderError::InvalidHeader(
                    "weight_map is not an object".to_string(),
                ))
            }
        };

        let mut shard_name_to_idx: BTreeMap<String, usize> = BTreeMap::new();
        let mut shard_names: Vec<String> = Vec::new();
        let mut weight_map: BTreeMap<String, usize> = BTreeMap::new();

        for (tensor_name, shard_val) in weight_map_obj {
            let shard = match shard_val {
                JsonValue::String(s) => s,
                _ => {
                    return Err(LoaderError::InvalidHeader(format!(
                        "weight_map[{tensor_name}] is not a string"
                    )))
                }
            };
            let idx = if let Some(&idx) = shard_name_to_idx.get(&shard) {
                idx
            } else {
                let idx = shard_names.len();
                shard_name_to_idx.insert(shard.clone(), idx);
                shard_names.push(shard);
                idx
            };
            weight_map.insert(tensor_name, idx);
        }

        let mut shards: Vec<SafetensorsFile> = Vec::with_capacity(shard_names.len());
        for name in &shard_names {
            let path: PathBuf = dir.join(name);
            if !path.exists() {
                return Err(LoaderError::MissingShard(name.clone()));
            }
            let file = SafetensorsFile::open(&path)?;
            shards.push(file);
        }

        Ok(Self {
            weight_map,
            shard_names,
            shards,
        })
    }

    /// Number of distinct tensors across all shards.
    pub fn len(&self) -> usize {
        self.weight_map.len()
    }

    /// Whether the model exposes any tensors.
    pub fn is_empty(&self) -> bool {
        self.weight_map.is_empty()
    }

    /// Number of opened shards.
    pub fn num_shards(&self) -> usize {
        self.shards.len()
    }

    /// Iterate tensor names in lexicographic order.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.weight_map.keys().map(|s| s.as_str())
    }

    /// Return the name of the shard file that owns `tensor_name`.
    pub fn shard_of(&self, tensor_name: &str) -> Option<&str> {
        let idx = *self.weight_map.get(tensor_name)?;
        Some(self.shard_names[idx].as_str())
    }

    /// Zero-copy view into a tensor, resolving the owning shard first.
    pub fn tensor_view(&self, name: &str) -> Option<TensorView<'_>> {
        let idx = *self.weight_map.get(name)?;
        self.shards[idx].tensor_view(name)
    }
}

#[cfg(test)]
mod tests {
    use super::super::safetensors::tests::build_safetensors;
    use super::*;
    use crate::tensor::DataType;
    use std::io::Write;

    fn write_file(path: &Path, bytes: &[u8]) {
        let mut f = std::fs::File::create(path).expect("create");
        f.write_all(bytes).expect("write");
        f.sync_all().ok();
    }

    #[test]
    fn loads_two_shard_model() {
        // Build a scratch directory under std::env::temp_dir().
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut dir = std::env::temp_dir();
        dir.push(format!("smallaios-sharded-{nanos}"));
        std::fs::create_dir_all(&dir).expect("mkdir");

        // Shard A holds tensor "a" (F32 of shape [2]).
        let a_payload: [u8; 8] = [0, 0, 0x80, 0x3F, 0, 0, 0, 0x40]; // [1.0, 2.0]
        let a_header = r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
        let a_bytes = build_safetensors(a_header, &a_payload);
        let a_path = dir.join("model-00001-of-00002.safetensors");
        write_file(&a_path, &a_bytes);

        // Shard B holds tensor "b" (I64 of shape [3]).
        let mut b_payload = Vec::new();
        for v in [10i64, 20, 30] {
            b_payload.extend_from_slice(&v.to_le_bytes());
        }
        let b_header = r#"{"b":{"dtype":"I64","shape":[3],"data_offsets":[0,24]}}"#;
        let b_bytes = build_safetensors(b_header, &b_payload);
        let b_path = dir.join("model-00002-of-00002.safetensors");
        write_file(&b_path, &b_bytes);

        let index = r#"{
            "metadata": {"total_size": 32},
            "weight_map": {
                "a": "model-00001-of-00002.safetensors",
                "b": "model-00002-of-00002.safetensors"
            }
        }"#;
        let index_path = dir.join("model.safetensors.index.json");
        write_file(&index_path, index.as_bytes());

        let model = MultiShardSafetensors::open_dir(&dir).expect("open");
        assert_eq!(model.len(), 2);
        assert_eq!(model.num_shards(), 2);

        assert_eq!(
            model.shard_of("a"),
            Some("model-00001-of-00002.safetensors")
        );
        assert_eq!(
            model.shard_of("b"),
            Some("model-00002-of-00002.safetensors")
        );

        let a = model.tensor_view("a").expect("a");
        assert_eq!(a.dtype, DataType::Float);
        assert_eq!(a.shape, &[2i64]);
        assert_eq!(a.data, &a_payload[..]);

        let b = model.tensor_view("b").expect("b");
        assert_eq!(b.dtype, DataType::Int64);
        assert_eq!(b.shape, &[3i64]);
        assert_eq!(b.data.len(), 24);

        assert!(model.tensor_view("missing").is_none());

        // Best-effort cleanup.
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_shard_is_reported() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mut dir = std::env::temp_dir();
        dir.push(format!("smallaios-sharded-missing-{nanos}"));
        std::fs::create_dir_all(&dir).expect("mkdir");

        let index = r#"{
            "metadata": {},
            "weight_map": {"a": "does-not-exist.safetensors"}
        }"#;
        let err = MultiShardSafetensors::open_with_index(&dir, index).unwrap_err();
        assert!(matches!(err, LoaderError::MissingShard(_)));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
