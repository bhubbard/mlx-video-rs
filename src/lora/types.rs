use crate::error::Result;
use mlx_rs::Array;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct LoRAWeights {
    pub lora_a: Array,
    pub lora_b: Array,
    pub rank: usize,
    pub alpha: f32,
    pub module_name: String,
}

impl LoRAWeights {
    pub fn new(lora_a: Array, lora_b: Array, rank: usize, alpha: f32, module_name: String) -> Self {
        Self {
            lora_a,
            lora_b,
            rank,
            alpha,
            module_name,
        }
    }

    pub fn scale(&self) -> f32 {
        if self.rank == 0 {
            1.0
        } else {
            self.alpha / self.rank as f32
        }
    }
}

#[derive(Debug, Clone)]
pub struct LoRAConfig {
    pub path: PathBuf,
    pub strength: f32,
    pub target_modules: Option<Vec<String>>,
}

impl LoRAConfig {
    pub fn new(path: impl AsRef<Path>, strength: f32) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            strength,
            target_modules: None,
        }
    }

    pub fn with_targets(mut self, targets: Vec<String>) -> Self {
        self.target_modules = Some(targets);
        self
    }
}

#[derive(Debug, Clone)]
pub struct AppliedLoRA {
    pub weights: LoRAWeights,
    pub strength: f32,
}

impl AppliedLoRA {
    pub fn new(weights: LoRAWeights, strength: f32) -> Self {
        Self { weights, strength }
    }

    pub fn compute_delta(&self) -> Result<Array> {
        let scale = self.weights.scale() * self.strength;
        let scale_arr = Array::from_f32(scale);
        let delta = self.weights.lora_b.matmul(&self.weights.lora_a)?;
        let scaled = delta.multiply(&scale_arr)?;
        Ok(scaled)
    }
}
