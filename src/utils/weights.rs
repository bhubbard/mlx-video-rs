use crate::error::{Result, VideoError};
use mlx_rs::Array;
use std::collections::HashMap;
use std::path::Path;

/// Load all weights from a single safetensors file or a directory of safetensors shards.
pub fn load_weights(path: impl AsRef<Path>) -> Result<HashMap<String, Array>> {
    let p = path.as_ref();
    if !p.exists() {
        return Err(VideoError::Weight(format!(
            "Path does not exist: {}",
            p.display()
        )));
    }

    if p.is_file() {
        return Array::load_safetensors(p).map_err(Into::into);
    }

    // Directory of shards
    let mut combined = HashMap::new();
    let entries = std::fs::read_dir(p)?;

    for entry in entries {
        let entry = entry?;
        let entry_path = entry.path();
        if entry_path.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            let shard = Array::load_safetensors(&entry_path)?;
            for (k, v) in shard {
                combined.insert(k, v);
            }
        }
    }

    if combined.is_empty() {
        return Err(VideoError::Weight(format!(
            "No .safetensors files found in {}",
            p.display()
        )));
    }

    Ok(combined)
}

/// Print Apple Silicon Metal device memory info
pub fn print_device_info() {
    println!("MLX Device: Apple Silicon GPU (Metal Accelerated)");
}
