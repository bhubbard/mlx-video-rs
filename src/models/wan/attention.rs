use super::rope::rope_apply;
use crate::error::Result;
use mlx_rs::Array;

#[derive(Debug, Clone)]
pub struct WanRMSNorm {
    pub weight: Array,
    pub eps: f32,
}

impl WanRMSNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        let weight = Array::ones::<f32>(&[dim as i32]).unwrap();
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        mlx_rs::fast::rms_norm(x, Some(&self.weight), self.eps).map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
pub struct WanLayerNorm {
    pub weight: Option<Array>,
    pub bias: Option<Array>,
    pub eps: f32,
    pub elementwise_affine: bool,
}

impl WanLayerNorm {
    pub fn new(dim: usize, eps: f32, elementwise_affine: bool) -> Self {
        let (weight, bias) = if elementwise_affine {
            (
                Some(Array::ones::<f32>(&[dim as i32]).unwrap()),
                Some(Array::zeros::<f32>(&[dim as i32]).unwrap()),
            )
        } else {
            (None, None)
        };
        Self {
            weight,
            bias,
            eps,
            elementwise_affine,
        }
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        mlx_rs::fast::layer_norm(x, self.weight.as_ref(), self.bias.as_ref(), self.eps)
            .map_err(Into::into)
    }
}

#[derive(Debug, Clone)]
pub struct WanSelfAttention {
    pub dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
    pub q_weight: Array,
    pub q_bias: Option<Array>,
    pub k_weight: Array,
    pub k_bias: Option<Array>,
    pub v_weight: Array,
    pub v_bias: Option<Array>,
    pub o_weight: Array,
    pub o_bias: Option<Array>,
    pub norm_q: Option<WanRMSNorm>,
    pub norm_k: Option<WanRMSNorm>,
}

impl WanSelfAttention {
    pub fn new(dim: usize, num_heads: usize, qk_norm: bool, eps: f32) -> Self {
        let head_dim = dim / num_heads;
        let scale = (head_dim as f32).powf(-0.5);

        let (norm_q, norm_k) = if qk_norm {
            (
                Some(WanRMSNorm::new(dim, eps)),
                Some(WanRMSNorm::new(dim, eps)),
            )
        } else {
            (None, None)
        };

        Self {
            dim,
            num_heads,
            head_dim,
            scale,
            q_weight: Array::zeros::<f32>(&[dim as i32, dim as i32]).unwrap(),
            q_bias: None,
            k_weight: Array::zeros::<f32>(&[dim as i32, dim as i32]).unwrap(),
            k_bias: None,
            v_weight: Array::zeros::<f32>(&[dim as i32, dim as i32]).unwrap(),
            v_bias: None,
            o_weight: Array::zeros::<f32>(&[dim as i32, dim as i32]).unwrap(),
            o_bias: None,
            norm_q,
            norm_k,
        }
    }

    pub fn forward(
        &self,
        x: &Array,
        grid_size: (usize, usize, usize),
        rope_cos_sin: &(Array, Array),
    ) -> Result<Array> {
        let shape = x.shape();
        let (b, s, _) = (shape[0], shape[1], shape[2]);
        let n = self.num_heads as i32;
        let d = self.head_dim as i32;

        let mut q = x.matmul(&self.q_weight)?;
        if let Some(ref b) = self.q_bias {
            q = q.add(b)?;
        }
        let mut k = x.matmul(&self.k_weight)?;
        if let Some(ref b) = self.k_bias {
            k = k.add(b)?;
        }
        let mut v = x.matmul(&self.v_weight)?;
        if let Some(ref b) = self.v_bias {
            v = v.add(b)?;
        }

        if let Some(ref nq) = self.norm_q {
            q = nq.forward(&q)?;
        }
        if let Some(ref nk) = self.norm_k {
            k = nk.forward(&k)?;
        }

        let q = q.reshape(&[b, s, n, d])?;
        let k = k.reshape(&[b, s, n, d])?;
        let v = v.reshape(&[b, s, n, d])?;

        // Apply 3D factorized RoPE
        let q_rot = rope_apply(&q, grid_size, rope_cos_sin)?;
        let k_rot = rope_apply(&k, grid_size, rope_cos_sin)?;

        // Transpose for multi-head attention: [B, num_heads, seq_len, head_dim]
        let q_t = q_rot.transpose_axes(&[0, 2, 1, 3])?;
        let k_t = k_rot.transpose_axes(&[0, 2, 1, 3])?;
        let v_t = v.transpose_axes(&[0, 2, 1, 3])?;

        let attn_out = mlx_rs::fast::scaled_dot_product_attention(
            &q_t,
            &k_t,
            &v_t,
            self.scale,
            None::<mlx_rs::fast::ScaledDotProductAttentionMask>,
            None::<&Array>,
        )?;

        // Transpose back: [B, seq_len, num_heads, head_dim] -> [B, seq_len, dim]
        let out_t = attn_out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, s, (n * d)])?;
        let mut output = out_t.matmul(&self.o_weight)?;
        if let Some(ref b) = self.o_bias {
            output = output.add(b)?;
        }
        Ok(output)
    }
}

#[derive(Debug, Clone)]
pub struct WanCrossAttention {
    pub dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub scale: f32,
    pub q_weight: Array,
    pub q_bias: Option<Array>,
    pub k_weight: Array,
    pub k_bias: Option<Array>,
    pub v_weight: Array,
    pub v_bias: Option<Array>,
    pub o_weight: Array,
    pub o_bias: Option<Array>,
    pub norm_q: Option<WanRMSNorm>,
    pub norm_k: Option<WanRMSNorm>,
}

impl WanCrossAttention {
    pub fn new(dim: usize, num_heads: usize, qk_norm: bool, eps: f32) -> Self {
        let head_dim = dim / num_heads;
        let scale = (head_dim as f32).powf(-0.5);

        let (norm_q, norm_k) = if qk_norm {
            (
                Some(WanRMSNorm::new(dim, eps)),
                Some(WanRMSNorm::new(dim, eps)),
            )
        } else {
            (None, None)
        };

        Self {
            dim,
            num_heads,
            head_dim,
            scale,
            q_weight: Array::zeros::<f32>(&[dim as i32, dim as i32]).unwrap(),
            q_bias: None,
            k_weight: Array::zeros::<f32>(&[dim as i32, dim as i32]).unwrap(),
            k_bias: None,
            v_weight: Array::zeros::<f32>(&[dim as i32, dim as i32]).unwrap(),
            v_bias: None,
            o_weight: Array::zeros::<f32>(&[dim as i32, dim as i32]).unwrap(),
            o_bias: None,
            norm_q,
            norm_k,
        }
    }

    pub fn forward(&self, x: &Array, context: &Array) -> Result<Array> {
        let x_shape = x.shape();
        let (b, s_x, _) = (x_shape[0], x_shape[1], x_shape[2]);
        let s_c = context.shape()[1];
        let n = self.num_heads as i32;
        let d = self.head_dim as i32;

        let mut q = x.matmul(&self.q_weight)?;
        if let Some(ref b) = self.q_bias {
            q = q.add(b)?;
        }
        let mut k = context.matmul(&self.k_weight)?;
        if let Some(ref b) = self.k_bias {
            k = k.add(b)?;
        }
        let mut v = context.matmul(&self.v_weight)?;
        if let Some(ref b) = self.v_bias {
            v = v.add(b)?;
        }

        if let Some(ref nq) = self.norm_q {
            q = nq.forward(&q)?;
        }
        if let Some(ref nk) = self.norm_k {
            k = nk.forward(&k)?;
        }

        let q_t = q.reshape(&[b, s_x, n, d])?.transpose_axes(&[0, 2, 1, 3])?;
        let k_t = k.reshape(&[b, s_c, n, d])?.transpose_axes(&[0, 2, 1, 3])?;
        let v_t = v.reshape(&[b, s_c, n, d])?.transpose_axes(&[0, 2, 1, 3])?;

        let attn_out = mlx_rs::fast::scaled_dot_product_attention(
            &q_t,
            &k_t,
            &v_t,
            self.scale,
            None::<mlx_rs::fast::ScaledDotProductAttentionMask>,
            None::<&Array>,
        )?;

        let out_t = attn_out
            .transpose_axes(&[0, 2, 1, 3])?
            .reshape(&[b, s_x, (n * d)])?;
        let mut output = out_t.matmul(&self.o_weight)?;
        if let Some(ref b) = self.o_bias {
            output = output.add(b)?;
        }
        Ok(output)
    }
}
