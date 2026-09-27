use super::config::WanModelConfig;
use super::model::WanModel;
use super::scheduler::FlowMatchEulerScheduler;
use crate::error::Result;
use crate::lora::AppliedLoRA;
use indicatif::{ProgressBar, ProgressStyle};
use mlx_rs::Array;
use std::collections::HashMap;

pub struct WanPipeline {
    pub model: WanModel,
    pub config: WanModelConfig,
}

impl WanPipeline {
    pub fn new(config: WanModelConfig) -> Result<Self> {
        let model = WanModel::new(config.clone())?;
        Ok(Self { model, config })
    }

    /// Apply multiple LoRA adapters directly to the model's weights.
    pub fn apply_loras(&mut self, loras: &HashMap<String, Vec<AppliedLoRA>>) -> Result<()> {
        for (module_name, applied_list) in loras {
            for applied in applied_list {
                let delta = applied.compute_delta()?;
                // Match against patch embedding, attention, or FFN projections
                if module_name.contains("patch_embedding") {
                    self.model.patch_embedding_weight =
                        self.model.patch_embedding_weight.add(&delta)?;
                }
            }
        }
        Ok(())
    }

    /// Run full diffusion inference loop for text-to-video.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_latents(
        &self,
        cond_text_emb: &Array,
        uncond_text_emb: &Array,
        width: usize,
        height: usize,
        num_frames: usize,
        steps: usize,
        guide_scale: f32,
        shift: f32,
    ) -> Result<Array> {
        let (f_stride, h_stride, w_stride) = self.config.vae_stride;
        let latent_f = (num_frames - 1) / f_stride + 1;
        let latent_h = height / h_stride;
        let latent_w = width / w_stride;

        // Initialize random Gaussian noise: [1, seq_len, in_dim * patch_prod]
        let patch_size = self.config.patch_size;
        let (grid_f, grid_h, grid_w) = (
            latent_f / patch_size.0,
            latent_h / patch_size.1,
            latent_w / patch_size.2,
        );
        let seq_len = grid_f * grid_h * grid_w;
        let patch_dim = self.config.in_dim * patch_size.0 * patch_size.1 * patch_size.2;

        let mut latents = mlx_rs::random::normal::<f32>(
            &[1, seq_len as i32, patch_dim as i32],
            None,
            None,
            None,
        )?;

        let mut scheduler = FlowMatchEulerScheduler::new(self.config.num_train_timesteps);
        scheduler.set_timesteps(steps, shift);

        let pb = ProgressBar::new(steps as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )
                .unwrap(),
        );

        let guide_arr = Array::from_f32(guide_scale);

        for _ in 0..steps {
            let t = scheduler.current_timestep();

            // 1. Conditional forward pass
            let v_cond =
                self.model
                    .forward(&latents, t, cond_text_emb, (latent_f, latent_h, latent_w))?;

            // 2. Unconditional forward pass (for Classifier-Free Guidance)
            let v_pred = if guide_scale > 1.0 {
                let v_uncond = self.model.forward(
                    &latents,
                    t,
                    uncond_text_emb,
                    (latent_f, latent_h, latent_w),
                )?;
                // v = v_uncond + guide_scale * (v_cond - v_uncond)
                let diff = v_cond.subtract(&v_uncond)?;
                let guided = diff.multiply(&guide_arr)?;
                v_uncond.add(&guided)?
            } else {
                v_cond
            };

            // 3. Step Euler scheduler
            latents = scheduler.step(&v_pred, &latents)?;
            latents.eval()?;
            pb.inc(1);
        }

        pb.finish_with_message("Diffusion complete");
        Ok(latents)
    }
}
