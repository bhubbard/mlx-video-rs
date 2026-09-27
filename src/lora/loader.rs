use super::types::{AppliedLoRA, LoRAConfig, LoRAWeights};
use crate::error::{Result, VideoError};
use mlx_rs::Array;
use std::collections::{HashMap, HashSet};
use std::path::Path;

pub fn load_lora_weights(lora_path: impl AsRef<Path>) -> Result<HashMap<String, LoRAWeights>> {
    let path = lora_path.as_ref();
    if !path.exists() {
        return Err(VideoError::LoRA(format!(
            "LoRA file not found: {}",
            path.display()
        )));
    }

    let all_weights = Array::load_safetensors(path)?;
    let mut module_names = HashSet::new();

    for key in all_weights.keys() {
        if let Some(prefix) = key.strip_suffix(".lora_A.weight") {
            module_names.insert(prefix.to_string());
        } else if let Some(prefix) = key.strip_suffix(".lora_B.weight") {
            module_names.insert(prefix.to_string());
        } else if let Some(prefix) = key.strip_suffix(".lora_down.weight") {
            module_names.insert(prefix.to_string());
        } else if let Some(prefix) = key.strip_suffix(".lora_up.weight") {
            module_names.insert(prefix.to_string());
        }
    }

    let mut lora_weights = HashMap::new();

    for module_name in module_names {
        let (key_a, key_b) = if all_weights.contains_key(&format!("{module_name}.lora_A.weight")) {
            (
                format!("{module_name}.lora_A.weight"),
                format!("{module_name}.lora_B.weight"),
            )
        } else if all_weights.contains_key(&format!("{module_name}.lora_down.weight")) {
            (
                format!("{module_name}.lora_down.weight"),
                format!("{module_name}.lora_up.weight"),
            )
        } else {
            continue;
        };

        let Some(lora_a) = all_weights.get(&key_a) else {
            continue;
        };
        let Some(lora_b) = all_weights.get(&key_b) else {
            continue;
        };

        let shape_a = lora_a.shape();
        let shape_b = lora_b.shape();

        if shape_a.len() != 2 || shape_b.len() != 2 {
            return Err(VideoError::LoRA(format!(
                "Invalid LoRA shape for {module_name}: A={shape_a:?}, B={shape_b:?}"
            )));
        }

        let rank = shape_a[0] as usize;
        let alpha = if let Some(alpha_arr) = all_weights.get(&format!("{module_name}.alpha")) {
            alpha_arr.item_cast::<f32>()
        } else {
            rank as f32
        };

        lora_weights.insert(
            module_name.clone(),
            LoRAWeights::new(lora_a.clone(), lora_b.clone(), rank, alpha, module_name),
        );
    }

    if lora_weights.is_empty() {
        return Err(VideoError::LoRA(format!(
            "No valid LoRA weights found in {}",
            path.display()
        )));
    }

    Ok(lora_weights)
}

pub fn load_multiple_loras(configs: &[LoRAConfig]) -> Result<HashMap<String, Vec<AppliedLoRA>>> {
    let mut applied_by_module: HashMap<String, Vec<AppliedLoRA>> = HashMap::new();

    for cfg in configs {
        let weights_map = load_lora_weights(&cfg.path)?;
        for (module_name, weights) in weights_map {
            if cfg
                .target_modules
                .as_ref()
                .is_some_and(|targets| !targets.iter().any(|t| module_name.contains(t)))
            {
                continue;
            }
            applied_by_module
                .entry(module_name)
                .or_default()
                .push(AppliedLoRA::new(weights, cfg.strength));
        }
    }

    Ok(applied_by_module)
}
