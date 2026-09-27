use crate::error::Result;
use mlx_rs::Array;
use std::path::Path;

/// Load, resize, center-crop, and normalize an image for I2V: [1, 1, H, W, 3] in [-1, 1].
pub fn preprocess_image(
    image_path: impl AsRef<Path>,
    target_width: u32,
    target_height: u32,
) -> Result<Array> {
    let img = image::open(image_path)?.to_rgb8();
    let (orig_w, orig_h) = img.dimensions();

    let scale = (target_width as f32 / orig_w as f32).max(target_height as f32 / orig_h as f32);
    let new_w = (orig_w as f32 * scale).round() as u32;
    let new_h = (orig_h as f32 * scale).round() as u32;

    let resized =
        image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::Lanczos3);

    let crop_x = (new_w.saturating_sub(target_width)) / 2;
    let crop_y = (new_h.saturating_sub(target_height)) / 2;

    let mut normalized_data = Vec::with_capacity((target_height * target_width * 3) as usize);

    for y in 0..target_height {
        for x in 0..target_width {
            let px = resized.get_pixel(crop_x + x, crop_y + y);
            for c in 0..3 {
                let v = (px[c] as f32 / 255.0) * 2.0 - 1.0;
                normalized_data.push(v);
            }
        }
    }

    Ok(Array::from_slice(
        &normalized_data,
        &[1, 1, target_height as i32, target_width as i32, 3],
    ))
}

/// Build temporal mask for I2V: first frame = 0, rest = 1.
/// Returns (mask [C, T, H, W], mask_tokens [1, L]).
pub fn build_i2v_mask(
    c: usize,
    t: usize,
    h: usize,
    w: usize,
    patch_size: (usize, usize, usize),
) -> Result<(Array, Array)> {
    let mut mask_vec = Vec::with_capacity(c * t * h * w);
    for _ in 0..c {
        // Frame 0: zeros
        mask_vec.resize(mask_vec.len() + h * w, 0.0f32);
        // Frames 1..T: ones
        mask_vec.resize(mask_vec.len() + (t - 1) * h * w, 1.0f32);
    }

    let mask = Array::from_slice(&mask_vec, &[c as i32, t as i32, h as i32, w as i32]);

    // Subsample to patch grid for token mask [1, L]
    let (pt, ph, pw) = patch_size;
    let (grid_t, grid_h, grid_w) = (t / pt, h / ph, w / pw);
    let mut token_vec = Vec::with_capacity(grid_t * grid_h * grid_w);

    for gt in 0..grid_t {
        let val = if gt == 0 { 0.0f32 } else { 1.0f32 };
        for _ in 0..(grid_h * grid_w) {
            token_vec.push(val);
        }
    }

    let mask_tokens = Array::from_slice(&token_vec, &[1, (grid_t * grid_h * grid_w) as i32]);
    Ok((mask, mask_tokens))
}
