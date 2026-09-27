use super::attention::{WanCrossAttention, WanLayerNorm, WanSelfAttention};
use crate::error::Result;
use mlx_rs::Array;
use mlx_rs::ops::indexing::*;

#[derive(Debug, Clone)]
pub struct WanFFN {
    pub fc1_weight: Array,
    pub fc1_bias: Option<Array>,
    pub fc2_weight: Array,
    pub fc2_bias: Option<Array>,
}

impl WanFFN {
    pub fn new(dim: usize, ffn_dim: usize) -> Self {
        Self {
            fc1_weight: Array::zeros::<f32>(&[dim as i32, ffn_dim as i32]).unwrap(),
            fc1_bias: None,
            fc2_weight: Array::zeros::<f32>(&[ffn_dim as i32, dim as i32]).unwrap(),
            fc2_bias: None,
        }
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut h = x.matmul(&self.fc1_weight)?;
        if let Some(ref b) = self.fc1_bias {
            h = h.add(b)?;
        }
        // GELU with tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        let h_gelu = mlx_rs::nn::gelu_approximate(&h)?;
        let mut out = h_gelu.matmul(&self.fc2_weight)?;
        if let Some(ref b) = self.fc2_bias {
            out = out.add(b)?;
        }
        Ok(out)
    }
}

#[derive(Debug, Clone)]
pub struct WanAttentionBlock {
    pub norm1: WanLayerNorm,
    pub self_attn: WanSelfAttention,
    pub norm3: Option<WanLayerNorm>,
    pub cross_attn: WanCrossAttention,
    pub norm2: WanLayerNorm,
    pub ffn: WanFFN,
    pub modulation: Array, // [1, 6, dim]
}

impl WanAttentionBlock {
    pub fn new(
        dim: usize,
        ffn_dim: usize,
        num_heads: usize,
        qk_norm: bool,
        cross_attn_norm: bool,
        eps: f32,
    ) -> Self {
        let norm1 = WanLayerNorm::new(dim, eps, false);
        let self_attn = WanSelfAttention::new(dim, num_heads, qk_norm, eps);
        let norm3 = if cross_attn_norm {
            Some(WanLayerNorm::new(dim, eps, true))
        } else {
            None
        };
        let cross_attn = WanCrossAttention::new(dim, num_heads, qk_norm, eps);
        let norm2 = WanLayerNorm::new(dim, eps, false);
        let ffn = WanFFN::new(dim, ffn_dim);

        let modulation = Array::zeros::<f32>(&[1, 6, dim as i32]).unwrap();

        Self {
            norm1,
            self_attn,
            norm3,
            cross_attn,
            norm2,
            ffn,
            modulation,
        }
    }

    pub fn forward(
        &self,
        x: &Array,
        e: &Array,
        grid_size: (usize, usize, usize),
        rope_cos_sin: &(Array, Array),
        context: &Array,
    ) -> Result<Array> {
        let mod_vec = self.modulation.add(e)?;
        let dim = self.modulation.shape()[2];

        let e0 = mod_vec.index((.., 0..1, ..));
        let e1 = mod_vec.index((.., 1..2, ..));
        let e2 = mod_vec.index((.., 2..3, ..));
        let e3 = mod_vec.index((.., 3..4, ..));
        let e4 = mod_vec.index((.., 4..5, ..));
        let e5 = mod_vec.index((.., 5..6, ..));

        let ones = Array::ones::<f32>(&[1, 1, dim])?;

        // 1. Self-Attention with modulation
        let norm1_x = self.norm1.forward(x)?;
        let scale_sa = ones.add(&e1)?;
        let x_mod_sa = norm1_x.multiply(&scale_sa)?.add(&e0)?;
        let sa_out = self.self_attn.forward(&x_mod_sa, grid_size, rope_cos_sin)?;
        let mut x_acc = x.add(&sa_out.multiply(&e2)?)?;

        // 2. Cross-Attention
        let x_cross = if let Some(ref n3) = self.norm3 {
            n3.forward(&x_acc)?
        } else {
            x_acc.clone()
        };
        let ca_out = self.cross_attn.forward(&x_cross, context)?;
        x_acc = x_acc.add(&ca_out)?;

        // 3. FFN with modulation
        let norm2_x = self.norm2.forward(&x_acc)?;
        let scale_ffn = ones.add(&e4)?;
        let x_mod_ffn = norm2_x.multiply(&scale_ffn)?.add(&e3)?;
        let ffn_out = self.ffn.forward(&x_mod_ffn)?;
        let out = x_acc.add(&ffn_out.multiply(&e5)?)?;

        Ok(out)
    }
}
