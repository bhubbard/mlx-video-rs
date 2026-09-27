use super::attention::WanLayerNorm;
use super::config::WanModelConfig;
use super::rope::{rope_params, rope_precompute_cos_sin};
use super::transformer::WanAttentionBlock;
use crate::error::Result;
use mlx_rs::Array;
use mlx_rs::ops::indexing::*;

pub fn sinusoidal_embedding_1d(dim: usize, positions: &[f32]) -> Result<Array> {
    assert!(
        dim.is_multiple_of(2),
        "dim must be even for sinusoidal embedding"
    );
    let half = dim / 2;
    let n = positions.len();

    let mut embeddings = Vec::with_capacity(n * dim);
    for &pos in positions {
        for i in 0..half {
            let inv_freq = (10000.0f64).powf(-(i as f64) / half as f64);
            let val = pos as f64 * inv_freq;
            embeddings.push(val.cos() as f32);
        }
        for i in 0..half {
            let inv_freq = (10000.0f64).powf(-(i as f64) / half as f64);
            let val = pos as f64 * inv_freq;
            embeddings.push(val.sin() as f32);
        }
    }

    Ok(Array::from_slice(&embeddings, &[n as i32, dim as i32]))
}

#[derive(Debug, Clone)]
pub struct Head {
    pub out_dim: usize,
    pub patch_size: (usize, usize, usize),
    pub norm: WanLayerNorm,
    pub head_weight: Array,
    pub head_bias: Option<Array>,
    pub modulation: Array, // [1, 2, dim]
}

impl Head {
    pub fn new(dim: usize, out_dim: usize, patch_size: (usize, usize, usize), eps: f32) -> Self {
        let proj_dim = patch_size.0 * patch_size.1 * patch_size.2 * out_dim;
        Self {
            out_dim,
            patch_size,
            norm: WanLayerNorm::new(dim, eps, false),
            head_weight: Array::zeros::<f32>(&[dim as i32, proj_dim as i32]).unwrap(),
            head_bias: None,
            modulation: Array::zeros::<f32>(&[1, 2, dim as i32]).unwrap(),
        }
    }

    pub fn forward(&self, x: &Array, e: &Array) -> Result<Array> {
        let dim = self.modulation.shape()[2];

        // Modulation [1, 2, dim] + e [B, 2, dim]
        let mod_vec = self.modulation.add(e)?;
        let e0 = mod_vec.index((.., 0..1, ..));
        let e1 = mod_vec.index((.., 1..2, ..));

        let ones = Array::ones::<f32>(&[1, 1, dim])?;
        let x_norm = self.norm.forward(x)?;
        let scale = ones.add(&e1)?;
        let x_mod = x_norm.multiply(&scale)?.add(&e0)?;

        let mut out = x_mod.matmul(&self.head_weight)?;
        if let Some(ref bias) = self.head_bias {
            out = out.add(bias)?;
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
pub struct WanModel {
    pub config: WanModelConfig,
    pub patch_embedding_weight: Array,
    pub patch_embedding_bias: Option<Array>,
    pub text_embedding_0_weight: Array,
    pub text_embedding_0_bias: Option<Array>,
    pub text_embedding_1_weight: Array,
    pub text_embedding_1_bias: Option<Array>,
    pub time_embedding_0_weight: Array,
    pub time_embedding_0_bias: Option<Array>,
    pub time_embedding_1_weight: Array,
    pub time_embedding_1_bias: Option<Array>,
    pub time_projection_weight: Array,
    pub time_projection_bias: Option<Array>,
    pub blocks: Vec<WanAttentionBlock>,
    pub head: Head,
    pub freqs: Array,
}

impl WanModel {
    pub fn new(config: WanModelConfig) -> Result<Self> {
        let dim = config.dim;
        let patch_dim =
            config.in_dim * config.patch_size.0 * config.patch_size.1 * config.patch_size.2;

        let blocks = (0..config.num_layers)
            .map(|_| {
                WanAttentionBlock::new(
                    dim,
                    config.ffn_dim,
                    config.num_heads,
                    config.qk_norm,
                    config.cross_attn_norm,
                    config.eps,
                )
            })
            .collect();

        let head = Head::new(dim, config.out_dim, config.patch_size, config.eps);
        let freqs = rope_params(1024, config.head_dim(), 10000.0)?;

        Ok(Self {
            patch_embedding_weight: Array::zeros::<f32>(&[patch_dim as i32, dim as i32])?,
            patch_embedding_bias: None,
            text_embedding_0_weight: Array::zeros::<f32>(&[config.text_dim as i32, dim as i32])?,
            text_embedding_0_bias: None,
            text_embedding_1_weight: Array::zeros::<f32>(&[dim as i32, dim as i32])?,
            text_embedding_1_bias: None,
            time_embedding_0_weight: Array::zeros::<f32>(&[config.freq_dim as i32, dim as i32])?,
            time_embedding_0_bias: None,
            time_embedding_1_weight: Array::zeros::<f32>(&[dim as i32, dim as i32])?,
            time_embedding_1_bias: None,
            time_projection_weight: Array::zeros::<f32>(&[dim as i32, (dim * 6) as i32])?,
            time_projection_bias: None,
            blocks,
            head,
            freqs,
            config,
        })
    }

    pub fn forward(
        &self,
        x: &Array,
        timestep: f32,
        context: &Array,
        grid_size: (usize, usize, usize),
    ) -> Result<Array> {
        let (f, h, w) = grid_size;
        let (p_t, p_h, p_w) = self.config.patch_size;
        let (grid_f, grid_h, grid_w) = (f / p_t, h / p_h, w / p_w);

        // Precompute 3D RoPE frequencies for this grid
        let rope_cos_sin = rope_precompute_cos_sin((grid_f, grid_h, grid_w), &self.freqs)?;

        // 1. Patch embedding
        let mut x_emb = x.matmul(&self.patch_embedding_weight)?;
        if let Some(ref bias) = self.patch_embedding_bias {
            x_emb = x_emb.add(bias)?;
        }

        // 2. Text embedding projection
        let mut text_h = context.matmul(&self.text_embedding_0_weight)?;
        if let Some(ref b) = self.text_embedding_0_bias {
            text_h = text_h.add(b)?;
        }
        let text_act = mlx_rs::nn::gelu_approximate(&text_h)?;
        let mut text_emb = text_act.matmul(&self.text_embedding_1_weight)?;
        if let Some(ref b) = self.text_embedding_1_bias {
            text_emb = text_emb.add(b)?;
        }

        // 3. Timestep embedding projection
        let t_sin = sinusoidal_embedding_1d(self.config.freq_dim, &[timestep])?;
        let mut t_h = t_sin.matmul(&self.time_embedding_0_weight)?;
        if let Some(ref b) = self.time_embedding_0_bias {
            t_h = t_h.add(b)?;
        }
        let t_act = mlx_rs::nn::silu(&t_h)?;
        let mut t_emb = t_act.matmul(&self.time_embedding_1_weight)?;
        if let Some(ref b) = self.time_embedding_1_bias {
            t_emb = t_emb.add(b)?;
        }

        // 4. Modulation projection (6 * dim)
        let t_proj_act = mlx_rs::nn::silu(&t_emb)?;
        let mut e_mod = t_proj_act.matmul(&self.time_projection_weight)?;
        if let Some(ref b) = self.time_projection_bias {
            e_mod = e_mod.add(b)?;
        }
        let b = x_emb.shape()[0];
        let e_reshaped = e_mod.reshape(&[b, 6, self.config.dim as i32])?;

        // 5. Pass through transformer blocks
        let mut hidden = x_emb;
        for block in &self.blocks {
            hidden = block.forward(
                &hidden,
                &e_reshaped,
                (grid_f, grid_h, grid_w),
                &rope_cos_sin,
                &text_emb,
            )?;
        }

        // 6. Output projection head (uses first 2 modulation vectors: e_head = e_reshaped[:, :2, :])
        let e_head = e_reshaped.index((.., 0..2, ..));
        let out = self.head.forward(&hidden, &e_head)?;

        Ok(out)
    }
}
