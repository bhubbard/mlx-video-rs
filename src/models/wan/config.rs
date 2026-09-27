use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WanModelConfig {
    #[serde(default = "default_model_type")]
    pub model_type: String,

    #[serde(default = "default_model_version")]
    pub model_version: String,

    #[serde(default = "default_patch_size")]
    pub patch_size: (usize, usize, usize),

    #[serde(default = "default_text_len")]
    pub text_len: usize,

    #[serde(default = "default_in_dim")]
    pub in_dim: usize,

    #[serde(default = "default_dim")]
    pub dim: usize,

    #[serde(default = "default_ffn_dim")]
    pub ffn_dim: usize,

    #[serde(default = "default_freq_dim")]
    pub freq_dim: usize,

    #[serde(default = "default_text_dim")]
    pub text_dim: usize,

    #[serde(default = "default_out_dim")]
    pub out_dim: usize,

    #[serde(default = "default_num_heads")]
    pub num_heads: usize,

    #[serde(default = "default_num_layers")]
    pub num_layers: usize,

    #[serde(default = "default_window_size")]
    pub window_size: (i32, i32),

    #[serde(default = "default_true")]
    pub qk_norm: bool,

    #[serde(default = "default_true")]
    pub cross_attn_norm: bool,

    #[serde(default = "default_eps")]
    pub eps: f32,

    // VAE
    #[serde(default = "default_vae_stride")]
    pub vae_stride: (usize, usize, usize),

    #[serde(default = "default_vae_z_dim")]
    pub vae_z_dim: usize,

    // Inference
    #[serde(default = "default_true")]
    pub dual_model: bool,

    #[serde(default = "default_boundary")]
    pub boundary: f32,

    #[serde(default = "default_sample_shift")]
    pub sample_shift: f32,

    #[serde(default = "default_sample_steps")]
    pub sample_steps: usize,

    #[serde(default = "default_guide_scale")]
    pub sample_guide_scale: (f32, f32),

    #[serde(default = "default_num_train_timesteps")]
    pub num_train_timesteps: usize,

    #[serde(default = "default_sample_fps")]
    pub sample_fps: usize,

    #[serde(default = "default_frame_num")]
    pub frame_num: usize,

    #[serde(default = "default_neg_prompt")]
    pub sample_neg_prompt: String,

    #[serde(default)]
    pub max_area: usize,

    #[serde(default = "default_t5_vocab_size")]
    pub t5_vocab_size: usize,

    #[serde(default = "default_t5_dim")]
    pub t5_dim: usize,

    #[serde(default = "default_t5_dim_attn")]
    pub t5_dim_attn: usize,

    #[serde(default = "default_t5_dim_ffn")]
    pub t5_dim_ffn: usize,

    #[serde(default = "default_t5_num_heads")]
    pub t5_num_heads: usize,

    #[serde(default = "default_t5_num_layers")]
    pub t5_num_layers: usize,

    #[serde(default = "default_t5_num_buckets")]
    pub t5_num_buckets: usize,
}

fn default_model_type() -> String {
    "t2v".into()
}
fn default_model_version() -> String {
    "2.2".into()
}
fn default_patch_size() -> (usize, usize, usize) {
    (1, 2, 2)
}
fn default_text_len() -> usize {
    512
}
fn default_in_dim() -> usize {
    16
}
fn default_dim() -> usize {
    5120
}
fn default_ffn_dim() -> usize {
    13824
}
fn default_freq_dim() -> usize {
    256
}
fn default_text_dim() -> usize {
    4096
}
fn default_out_dim() -> usize {
    16
}
fn default_num_heads() -> usize {
    40
}
fn default_num_layers() -> usize {
    40
}
fn default_window_size() -> (i32, i32) {
    (-1, -1)
}
fn default_true() -> bool {
    true
}
fn default_eps() -> f32 {
    1e-6
}
fn default_vae_stride() -> (usize, usize, usize) {
    (4, 8, 8)
}
fn default_vae_z_dim() -> usize {
    16
}
fn default_boundary() -> f32 {
    0.875
}
fn default_sample_shift() -> f32 {
    12.0
}
fn default_sample_steps() -> usize {
    40
}
fn default_guide_scale() -> (f32, f32) {
    (3.0, 4.0)
}
fn default_num_train_timesteps() -> usize {
    1000
}
fn default_sample_fps() -> usize {
    16
}
fn default_frame_num() -> usize {
    81
}
fn default_t5_vocab_size() -> usize {
    256384
}
fn default_t5_dim() -> usize {
    4096
}
fn default_t5_dim_attn() -> usize {
    4096
}
fn default_t5_dim_ffn() -> usize {
    10240
}
fn default_t5_num_heads() -> usize {
    64
}
fn default_t5_num_layers() -> usize {
    24
}
fn default_t5_num_buckets() -> usize {
    32
}
fn default_neg_prompt() -> String {
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，\
    最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，\
    画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，\
    杂乱的背景，三条腿，背景人很多，倒着走"
        .into()
}

impl Default for WanModelConfig {
    fn default() -> Self {
        Self::wan22_t2v_14b()
    }
}

impl WanModelConfig {
    pub fn head_dim(&self) -> usize {
        self.dim / self.num_heads
    }

    /// Wan2.1 T2V 14B: single model, 40 layers, dim=5120.
    pub fn wan21_t2v_14b() -> Self {
        Self {
            model_type: "t2v".into(),
            model_version: "2.1".into(),
            patch_size: (1, 2, 2),
            text_len: 512,
            in_dim: 16,
            dim: 5120,
            ffn_dim: 13824,
            freq_dim: 256,
            text_dim: 4096,
            out_dim: 16,
            num_heads: 40,
            num_layers: 40,
            window_size: (-1, -1),
            qk_norm: true,
            cross_attn_norm: true,
            eps: 1e-6,
            vae_stride: (4, 8, 8),
            vae_z_dim: 16,
            dual_model: false,
            boundary: 0.0,
            sample_shift: 5.0,
            sample_steps: 50,
            sample_guide_scale: (5.0, 5.0),
            num_train_timesteps: 1000,
            sample_fps: 16,
            frame_num: 81,
            sample_neg_prompt: default_neg_prompt(),
            max_area: 0,
            t5_vocab_size: 256384,
            t5_dim: 4096,
            t5_dim_attn: 4096,
            t5_dim_ffn: 10240,
            t5_num_heads: 64,
            t5_num_layers: 24,
            t5_num_buckets: 32,
        }
    }

    /// Wan2.1 T2V 1.3B: single model, 30 layers, dim=1536.
    pub fn wan21_t2v_1_3b() -> Self {
        Self {
            model_type: "t2v".into(),
            model_version: "2.1".into(),
            patch_size: (1, 2, 2),
            text_len: 512,
            in_dim: 16,
            dim: 1536,
            ffn_dim: 8960,
            freq_dim: 256,
            text_dim: 4096,
            out_dim: 16,
            num_heads: 12,
            num_layers: 30,
            window_size: (-1, -1),
            qk_norm: true,
            cross_attn_norm: true,
            eps: 1e-6,
            vae_stride: (4, 8, 8),
            vae_z_dim: 16,
            dual_model: false,
            boundary: 0.0,
            sample_shift: 5.0,
            sample_steps: 50,
            sample_guide_scale: (5.0, 5.0),
            num_train_timesteps: 1000,
            sample_fps: 16,
            frame_num: 81,
            sample_neg_prompt: default_neg_prompt(),
            max_area: 0,
            t5_vocab_size: 256384,
            t5_dim: 4096,
            t5_dim_attn: 4096,
            t5_dim_ffn: 10240,
            t5_num_heads: 64,
            t5_num_layers: 24,
            t5_num_buckets: 32,
        }
    }

    /// Wan2.2 T2V 14B: dual model, 40 layers, dim=5120.
    pub fn wan22_t2v_14b() -> Self {
        Self {
            model_type: "t2v".into(),
            model_version: "2.2".into(),
            patch_size: (1, 2, 2),
            text_len: 512,
            in_dim: 16,
            dim: 5120,
            ffn_dim: 13824,
            freq_dim: 256,
            text_dim: 4096,
            out_dim: 16,
            num_heads: 40,
            num_layers: 40,
            window_size: (-1, -1),
            qk_norm: true,
            cross_attn_norm: true,
            eps: 1e-6,
            vae_stride: (4, 8, 8),
            vae_z_dim: 16,
            dual_model: true,
            boundary: 0.875,
            sample_shift: 12.0,
            sample_steps: 40,
            sample_guide_scale: (3.0, 4.0),
            num_train_timesteps: 1000,
            sample_fps: 16,
            frame_num: 81,
            sample_neg_prompt: default_neg_prompt(),
            max_area: 0,
            t5_vocab_size: 256384,
            t5_dim: 4096,
            t5_dim_attn: 4096,
            t5_dim_ffn: 10240,
            t5_num_heads: 64,
            t5_num_layers: 24,
            t5_num_buckets: 32,
        }
    }

    /// Wan2.2 I2V 14B: image-to-video, in_dim=36, out_dim=16.
    pub fn wan22_i2v_14b() -> Self {
        Self {
            model_type: "i2v".into(),
            model_version: "2.2".into(),
            patch_size: (1, 2, 2),
            text_len: 512,
            in_dim: 36,
            dim: 5120,
            ffn_dim: 13824,
            freq_dim: 256,
            text_dim: 4096,
            out_dim: 16,
            num_heads: 40,
            num_layers: 40,
            window_size: (-1, -1),
            qk_norm: true,
            cross_attn_norm: true,
            eps: 1e-6,
            vae_stride: (4, 8, 8),
            vae_z_dim: 16,
            dual_model: true,
            boundary: 0.900,
            sample_shift: 5.0,
            sample_steps: 40,
            sample_guide_scale: (3.5, 3.5),
            num_train_timesteps: 1000,
            sample_fps: 16,
            frame_num: 81,
            sample_neg_prompt: default_neg_prompt(),
            max_area: 704 * 1280,
            t5_vocab_size: 256384,
            t5_dim: 4096,
            t5_dim_attn: 4096,
            t5_dim_ffn: 10240,
            t5_num_heads: 64,
            t5_num_layers: 24,
            t5_num_buckets: 32,
        }
    }

    /// Wan2.2 TI2V 5B: text+image to video, 30 layers, dim=3072, in_dim=48.
    pub fn wan22_ti2v_5b() -> Self {
        Self {
            model_type: "ti2v".into(),
            model_version: "2.2".into(),
            patch_size: (1, 2, 2),
            text_len: 512,
            in_dim: 48,
            dim: 3072,
            ffn_dim: 14336,
            freq_dim: 256,
            text_dim: 4096,
            out_dim: 48,
            num_heads: 24,
            num_layers: 30,
            window_size: (-1, -1),
            qk_norm: true,
            cross_attn_norm: true,
            eps: 1e-6,
            vae_stride: (4, 16, 16),
            vae_z_dim: 48,
            dual_model: false,
            boundary: 0.0,
            sample_shift: 5.0,
            sample_steps: 40,
            sample_guide_scale: (5.0, 5.0),
            num_train_timesteps: 1000,
            sample_fps: 24,
            frame_num: 81,
            sample_neg_prompt: default_neg_prompt(),
            max_area: 704 * 1280,
            t5_vocab_size: 256384,
            t5_dim: 4096,
            t5_dim_attn: 4096,
            t5_dim_ffn: 10240,
            t5_num_heads: 64,
            t5_num_layers: 24,
            t5_num_buckets: 32,
        }
    }
}
