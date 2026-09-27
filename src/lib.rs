pub mod error;
pub mod lora;
pub mod models;
pub mod utils;

pub use error::{Result, VideoError};
pub use lora::{AppliedLoRA, LoRAConfig, LoRAWeights, load_lora_weights, load_multiple_loras};
pub use models::ltx::{LTXModelConfig, LTXSampler, PipelineType};
pub use models::wan::{
    WanModelConfig, WanPipeline, build_i2v_mask, normalize_latents, preprocess_image,
    unnormalize_latents,
};
pub use utils::{load_weights, print_device_info};

#[cfg(test)]
mod tests {
    use super::*;
    use mlx_rs::Array;

    #[test]
    fn test_wan_config_presets() {
        let wan21_14b = WanModelConfig::wan21_t2v_14b();
        assert_eq!(wan21_14b.num_layers, 40);
        assert_eq!(wan21_14b.dim, 5120);
        assert_eq!(wan21_14b.model_version, "2.1");
        assert!(!wan21_14b.dual_model);

        let wan21_1_3b = WanModelConfig::wan21_t2v_1_3b();
        assert_eq!(wan21_1_3b.num_layers, 30);
        assert_eq!(wan21_1_3b.dim, 1536);

        let wan22_14b = WanModelConfig::wan22_t2v_14b();
        assert!(wan22_14b.dual_model);
        assert_eq!(wan22_14b.sample_shift, 12.0);

        let wan22_i2v = WanModelConfig::wan22_i2v_14b();
        assert_eq!(wan22_i2v.in_dim, 36);
        assert_eq!(wan22_i2v.model_type, "i2v");
    }

    #[test]
    fn test_ltx_config_presets() {
        let ltx = LTXModelConfig::default();
        assert_eq!(ltx.dim, 2048);
        assert_eq!(ltx.heads, 32);
        assert_eq!(ltx.pipeline_type, PipelineType::Distilled);
    }

    #[test]
    fn test_scheduler_sigmas() {
        let sigmas = models::wan::scheduler::compute_sigmas(40, 12.0, 1000);
        assert_eq!(sigmas.len(), 41);
        assert!((sigmas[0] - 0.999).abs() < 0.05);
        assert_eq!(sigmas[40], 0.0);
    }

    #[test]
    fn test_vae_normalization() {
        let fake_latents = Array::zeros::<f32>(&[1, 16, 2, 8, 8]).unwrap();
        let norm = normalize_latents(&fake_latents).unwrap();
        assert_eq!(norm.shape(), &[1, 16, 2, 8, 8]);

        let unnorm = unnormalize_latents(&norm).unwrap();
        assert_eq!(unnorm.shape(), &[1, 16, 2, 8, 8]);
    }

    #[test]
    fn test_i2v_mask() {
        let (mask, tokens) = build_i2v_mask(16, 8, 16, 16, (1, 2, 2)).unwrap();
        assert_eq!(mask.shape(), &[16, 8, 16, 16]);
        assert_eq!(tokens.shape(), &[1, 8 * 8 * 8]);
    }

    #[test]
    fn test_lora_delta_computation() {
        let lora_a = Array::ones::<f32>(&[4, 16]).unwrap();
        let lora_b = Array::ones::<f32>(&[32, 4]).unwrap();
        let weights = LoRAWeights::new(lora_a, lora_b, 4, 4.0, "test_module".to_string());
        assert_eq!(weights.scale(), 1.0);

        let applied = AppliedLoRA::new(weights, 1.0);
        let delta = applied.compute_delta().unwrap();
        assert_eq!(delta.shape(), &[32, 16]);
    }

    #[test]
    fn test_rope_computation() {
        use models::wan::rope::{rope_params, rope_precompute_cos_sin};
        let freqs = rope_params(16, 64, 10000.0).unwrap();
        assert_eq!(freqs.shape(), &[16, 32, 2]);

        let (cos_f, sin_f) = rope_precompute_cos_sin((2, 2, 2), &freqs).unwrap();
        assert_eq!(cos_f.shape(), &[8, 1, 32]);
        assert_eq!(sin_f.shape(), &[8, 1, 32]);
    }

    #[test]
    fn test_euler_scheduler_step() {
        use models::wan::scheduler::FlowMatchEulerScheduler;
        let mut scheduler = FlowMatchEulerScheduler::new(1000);
        scheduler.set_timesteps(10, 1.0);
        assert_eq!(scheduler.step_index, 0);

        let sample = Array::zeros::<f32>(&[1, 16, 2, 4, 4]).unwrap();
        let model_output = Array::ones::<f32>(&[1, 16, 2, 4, 4]).unwrap();
        let next_sample = scheduler.step(&model_output, &sample).unwrap();
        assert_eq!(next_sample.shape(), &[1, 16, 2, 4, 4]);
        assert_eq!(scheduler.step_index, 1);
    }

    #[test]
    fn test_wan_norms() {
        use models::wan::attention::{WanLayerNorm, WanRMSNorm};
        let x = Array::ones::<f32>(&[2, 32]).unwrap();
        let rms = WanRMSNorm::new(32, 1e-6);
        let rms_out = rms.forward(&x).unwrap();
        assert_eq!(rms_out.shape(), &[2, 32]);

        let ln = WanLayerNorm::new(32, 1e-6, true);
        let ln_out = ln.forward(&x).unwrap();
        assert_eq!(ln_out.shape(), &[2, 32]);
    }

    #[test]
    fn test_sinusoidal_embedding() {
        use models::wan::model::sinusoidal_embedding_1d;
        let timesteps = [100.0f32, 500.0f32];
        let emb = sinusoidal_embedding_1d(64, &timesteps).unwrap();
        assert_eq!(emb.shape(), &[2, 64]);
    }

    #[test]
    fn test_tiling_config() {
        use models::wan::tiling::TilingConfig;
        let cfg = TilingConfig::default();
        assert_eq!(cfg.tile_size_t, 16);
        assert_eq!(cfg.tile_size_h, 32);
        assert_eq!(cfg.tile_size_w, 32);
        assert_eq!(cfg.overlap_t, 4);
    }

    #[test]
    fn test_rope_apply() {
        use models::wan::rope::{rope_apply, rope_params, rope_precompute_cos_sin};
        let freqs = rope_params(8, 16, 10000.0).unwrap();
        let precomputed = rope_precompute_cos_sin((2, 2, 2), &freqs).unwrap();

        // [B, S, num_heads, head_dim]
        let x = Array::ones::<f32>(&[1, 8, 2, 16]).unwrap();
        let rotated = rope_apply(&x, (2, 2, 2), &precomputed).unwrap();
        assert_eq!(rotated.shape(), &[1, 8, 2, 16]);
    }

    #[test]
    fn test_causal_conv3d() {
        use models::wan::vae::CausalConv3d;
        let conv = CausalConv3d::new(2, 4, (3, 3, 3), (1, 1, 1), (0, 1, 1));
        let x = Array::zeros::<f32>(&[1, 2, 2, 4, 4]).unwrap();
        let out = conv.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 2, 4, 4]);
    }

    #[test]
    fn test_wan_ffn() {
        use models::wan::transformer::WanFFN;
        let ffn = WanFFN::new(16, 32);
        let x = Array::ones::<f32>(&[2, 16]).unwrap();
        let out = ffn.forward(&x).unwrap();
        assert_eq!(out.shape(), &[2, 16]);
    }

    #[test]
    fn test_ltx_sampler() {
        let s1 = LTXSampler::for_stage_1();
        assert_eq!(s1.step_index, 0);
        assert!(!s1.is_finished());
        assert!(s1.current_sigma() > 0.0);

        let s2 = LTXSampler::for_stage_2();
        assert_eq!(s2.step_index, 0);
        assert!(!s2.is_finished());

        let dev = LTXSampler::for_dev(10);
        assert_eq!(dev.sigmas.len(), 11);
    }

    #[test]
    fn test_safetensors_io() {
        let mut map = std::collections::HashMap::new();
        map.insert(
            "model.weight".to_string(),
            Array::ones::<f32>(&[4, 4]).unwrap(),
        );
        map.insert(
            "model.diffusion_model.lora_A.weight".to_string(),
            Array::ones::<f32>(&[2, 4]).unwrap(),
        );
        map.insert(
            "model.diffusion_model.lora_B.weight".to_string(),
            Array::zeros::<f32>(&[4, 2]).unwrap(),
        );

        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join("test.safetensors");
        Array::save_safetensors(&map, None, &file_path).unwrap();

        let loaded = load_weights(&file_path).unwrap();
        assert_eq!(loaded.len(), 3);

        let loras = load_lora_weights(&file_path).unwrap();
        assert_eq!(loras.len(), 1);
        let weight = loras.get("model.diffusion_model").unwrap();
        assert_eq!(weight.rank, 2);
    }
}
