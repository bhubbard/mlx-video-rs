use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PipelineType {
    #[serde(rename = "distilled")]
    Distilled,
    #[serde(rename = "dev")]
    Dev,
    #[serde(rename = "dev-two-stage")]
    DevTwoStage,
    #[serde(rename = "dev-two-stage-hq")]
    DevTwoStageHq,
}

pub const STAGE_1_SIGMAS: [f32; 9] = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0,
];

pub const STAGE_2_SIGMAS: [f32; 4] = [0.909375, 0.725, 0.421875, 0.0];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LTXModelConfig {
    #[serde(default = "default_dim")]
    pub dim: usize,

    #[serde(default = "default_heads")]
    pub heads: usize,

    #[serde(default = "default_d_head")]
    pub d_head: usize,

    #[serde(default = "default_context_dim")]
    pub context_dim: usize,

    #[serde(default = "default_num_layers")]
    pub num_layers: usize,

    #[serde(default = "default_in_channels")]
    pub in_channels: usize,

    #[serde(default = "default_out_channels")]
    pub out_channels: usize,

    #[serde(default = "default_patch_size")]
    pub patch_size: (usize, usize, usize),

    #[serde(default = "default_pipeline")]
    pub pipeline_type: PipelineType,
}

fn default_dim() -> usize {
    2048
}
fn default_heads() -> usize {
    32
}
fn default_d_head() -> usize {
    64
}
fn default_context_dim() -> usize {
    4096
}
fn default_num_layers() -> usize {
    28
}
fn default_in_channels() -> usize {
    128
}
fn default_out_channels() -> usize {
    128
}
fn default_patch_size() -> (usize, usize, usize) {
    (1, 1, 1)
}
fn default_pipeline() -> PipelineType {
    PipelineType::Distilled
}

impl Default for LTXModelConfig {
    fn default() -> Self {
        Self {
            dim: 2048,
            heads: 32,
            d_head: 64,
            context_dim: 4096,
            num_layers: 28,
            in_channels: 128,
            out_channels: 128,
            patch_size: (1, 1, 1),
            pipeline_type: PipelineType::Distilled,
        }
    }
}
