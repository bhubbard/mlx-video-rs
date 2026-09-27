use crate::error::Result;
use mlx_rs::Array;
use mlx_rs::ops::indexing::*;

/// Precompute RoPE frequency parameters as (cos, sin) pairs: [max_seq_len, dim / 2, 2].
pub fn rope_params(max_seq_len: usize, dim: usize, theta: f64) -> Result<Array> {
    assert!(dim.is_multiple_of(2), "dim must be even for RoPE");
    let half_d = dim / 2;

    let mut cos_sin = Vec::with_capacity(max_seq_len * half_d * 2);
    for pos in 0..max_seq_len {
        for i in 0..half_d {
            let freq = 1.0 / theta.powf((2 * i) as f64 / dim as f64);
            let val = pos as f64 * freq;
            cos_sin.push(val.cos() as f32);
            cos_sin.push(val.sin() as f32);
        }
    }

    Ok(Array::from_slice(
        &cos_sin,
        &[max_seq_len as i32, half_d as i32, 2],
    ))
}

/// Precompute cos/sin frequency tensors for a constant (F, H, W) grid.
///
/// Returns (cos_f, sin_f), each of shape [seq_len, 1, half_d].
pub fn rope_precompute_cos_sin(
    grid_size: (usize, usize, usize),
    freqs: &Array,
) -> Result<(Array, Array)> {
    let (f, h, w) = grid_size;
    let seq_len = f * h * w;
    let shape = freqs.shape();
    let half_d = shape[1] as usize;

    let d_t = half_d - 2 * (half_d / 3);
    let d_h = half_d / 3;
    let d_w = half_d / 3;

    let freqs_slice: &[f32] = freqs.as_slice();
    let _max_len = shape[0] as usize;

    let mut cos_vec = Vec::with_capacity(seq_len * half_d);
    let mut sin_vec = Vec::with_capacity(seq_len * half_d);

    for t_idx in 0..f {
        for h_idx in 0..h {
            for w_idx in 0..w {
                // Temporal components
                for c in 0..d_t {
                    let offset = (t_idx * half_d + c) * 2;
                    cos_vec.push(freqs_slice[offset]);
                    sin_vec.push(freqs_slice[offset + 1]);
                }
                // Height components
                for c in 0..d_h {
                    let offset = (h_idx * half_d + d_t + c) * 2;
                    cos_vec.push(freqs_slice[offset]);
                    sin_vec.push(freqs_slice[offset + 1]);
                }
                // Width components
                for c in 0..d_w {
                    let offset = (w_idx * half_d + d_t + d_h + c) * 2;
                    cos_vec.push(freqs_slice[offset]);
                    sin_vec.push(freqs_slice[offset + 1]);
                }
            }
        }
    }

    let cos_arr = Array::from_slice(&cos_vec, &[seq_len as i32, 1, half_d as i32]);
    let sin_arr = Array::from_slice(&sin_vec, &[seq_len as i32, 1, half_d as i32]);
    Ok((cos_arr, sin_arr))
}

/// Apply 3-way factorized RoPE to Q or K tensor using precomputed cos/sin.
///
/// x: [B, S, num_heads, head_dim]
/// precomputed_cos_sin: ([seq_len, 1, half_d], [seq_len, 1, half_d])
pub fn rope_apply(
    x: &Array,
    grid_size: (usize, usize, usize),
    precomputed: &(Array, Array),
) -> Result<Array> {
    let (f, h, w) = grid_size;
    let seq_len = f * h * w;
    let shape = x.shape();
    let (b, s, n, d) = (shape[0], shape[1], shape[2], shape[3]);
    let half_d = d / 2;

    let (cos_f, sin_f) = precomputed;

    // Index tokens up to seq_len
    let x_seq = x.index((.., 0..seq_len as i32, .., ..));
    let x_reshaped = x_seq.reshape(&[b, seq_len as i32, n, half_d, 2])?;

    // x_real = x_reshaped[..., 0], x_imag = x_reshaped[..., 1]
    let x_real = x_reshaped.index((Ellipsis, 0));
    let x_imag = x_reshaped.index((Ellipsis, 1));

    // Rotation: (a + bi) * (cos + i*sin) = (a*cos - b*sin) + i*(a*sin + b*cos)
    let term1 = x_real.multiply(cos_f)?;
    let term2 = x_imag.multiply(sin_f)?;
    let out_real = term1.subtract(&term2)?;

    let term3 = x_real.multiply(sin_f)?;
    let term4 = x_imag.multiply(cos_f)?;
    let out_imag = term3.add(&term4)?;

    let out_real_exp = out_real.expand_dims(-1)?;
    let out_imag_exp = out_imag.expand_dims(-1)?;

    let rotated = mlx_rs::ops::concatenate(&[&out_real_exp, &out_imag_exp], -1)?.reshape(&[
        b,
        seq_len as i32,
        n,
        d,
    ])?;

    if (seq_len as i32) < s {
        let remainder = x.index((.., seq_len as i32.., .., ..));
        mlx_rs::ops::concatenate(&[&rotated, &remainder], 1).map_err(Into::into)
    } else {
        Ok(rotated)
    }
}
