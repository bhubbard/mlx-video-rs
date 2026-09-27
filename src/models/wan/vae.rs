use crate::error::Result;
use mlx_rs::Array;

pub const VAE_MEAN: [f32; 16] = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517,
    -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
];

pub const VAE_STD: [f32; 16] = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579,
    1.6382, 1.1253, 2.8251, 1.9160,
];

/// Normalize latents using official Wan VAE channel statistics: (x - mean) / std
pub fn normalize_latents(latents: &Array) -> Result<Array> {
    let mean_arr = Array::from_slice(&VAE_MEAN, &[1, 16, 1, 1, 1]);
    let std_arr = Array::from_slice(&VAE_STD, &[1, 16, 1, 1, 1]);
    let centered = latents.subtract(&mean_arr)?;
    centered.divide(&std_arr).map_err(Into::into)
}

/// Unnormalize latents prior to VAE decoding: x * std + mean
pub fn unnormalize_latents(latents: &Array) -> Result<Array> {
    let mean_arr = Array::from_slice(&VAE_MEAN, &[1, 16, 1, 1, 1]);
    let std_arr = Array::from_slice(&VAE_STD, &[1, 16, 1, 1, 1]);
    let scaled = latents.multiply(&std_arr)?;
    scaled.add(&mean_arr).map_err(Into::into)
}

#[derive(Debug, Clone)]
pub struct CausalConv3d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: (usize, usize, usize),
    pub stride: (usize, usize, usize),
    pub causal_pad_t: usize,
    pub pad_h: usize,
    pub pad_w: usize,
    pub weight: Array, // [O, D, H, W, I]
    pub bias: Option<Array>,
}

impl CausalConv3d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
    ) -> Self {
        let causal_pad_t = kernel_size.0.saturating_sub(stride.0);
        let weight = Array::zeros::<f32>(&[
            out_channels as i32,
            kernel_size.0 as i32,
            kernel_size.1 as i32,
            kernel_size.2 as i32,
            in_channels as i32,
        ])
        .unwrap();

        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            causal_pad_t,
            pad_h: padding.1,
            pad_w: padding.2,
            weight,
            bias: None,
        }
    }

    /// Forward pass with causal padding: x: [B, C, T, H, W]
    pub fn forward(&self, x: &Array) -> Result<Array> {
        let shape = x.shape();
        let (b, c, _t, h, w) = (shape[0], shape[1], shape[2], shape[3], shape[4]);

        let mut padded = x.clone();
        if self.causal_pad_t > 0 {
            let pad_t = Array::zeros::<f32>(&[b, c, self.causal_pad_t as i32, h, w])?;
            padded = mlx_rs::ops::concatenate(&[&pad_t, &padded], 2)?;
        }

        // Channel-first [B, C, T, H, W] to channel-last [B, T, H, W, C] for MLX Conv3D
        let x_transposed = padded.transpose_axes(&[0, 2, 3, 4, 1])?;
        let conv_out = mlx_rs::ops::conv3d(
            &x_transposed,
            &self.weight,
            Some((
                self.stride.0 as i32,
                self.stride.1 as i32,
                self.stride.2 as i32,
            )),
            Some((0, self.pad_h as i32, self.pad_w as i32)),
            Some((1, 1, 1)),
            1,
        )?;

        let mut res = conv_out.transpose_axes(&[0, 4, 1, 2, 3])?;
        if let Some(ref bias) = self.bias {
            let b_reshaped = bias.reshape(&[1, self.out_channels as i32, 1, 1, 1])?;
            res = res.add(&b_reshaped)?;
        }
        Ok(res)
    }
}
