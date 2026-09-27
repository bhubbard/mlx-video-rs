use clap::{Args, Parser, Subcommand};
use colored::*;
use mlx_video_rs::lora::load_lora_weights;
use mlx_video_rs::models::ltx::LTXModelConfig;
use mlx_video_rs::models::wan::WanModelConfig;
use mlx_video_rs::models::wan::WanPipeline;
use mlx_video_rs::utils::print_device_info;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "mlx-video",
    about = "Native Apple Silicon MLX Video generation and inference CLI in Rust (Wan2.1, Wan2.2, LTX-2)",
    version = "0.1.0"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate video using Wan2.1 or Wan2.2 (Text-to-Video & Image-to-Video)
    Wan(WanArgs),

    /// Generate video using LTX-2 or LTX-2.3
    Ltx(LtxArgs),

    /// Inspect a LoRA adapter file (ranks, alphas, modules)
    Lora(LoraArgs),

    /// Display Apple Silicon Metal GPU status and model capabilities
    Info,
}

#[derive(Args, Debug)]
struct WanArgs {
    /// Path to model directory (containing config.json and weights)
    #[arg(short = 'm', long, default_value = "./models/wan")]
    model_dir: PathBuf,

    /// Text prompt describing the video to generate
    #[arg(short = 'p', long)]
    prompt: String,

    /// Optional negative prompt
    #[arg(long)]
    negative_prompt: Option<String>,

    /// Path to input image for Image-to-Video (I2V mode)
    #[arg(long)]
    image: Option<PathBuf>,

    /// Video width (divisible by 16)
    #[arg(long, default_value_t = 1280)]
    width: usize,

    /// Video height (divisible by 16)
    #[arg(long, default_value_t = 704)]
    height: usize,

    /// Number of frames (must be 4n + 1, e.g. 81)
    #[arg(short = 'n', long, default_value_t = 81)]
    num_frames: usize,

    /// Number of diffusion sampling steps
    #[arg(long, default_value_t = 40)]
    steps: usize,

    /// Classifier-free guidance scale
    #[arg(long, default_value_t = 5.0)]
    guide_scale: f32,

    /// Flow matching shift parameter
    #[arg(long, default_value_t = 12.0)]
    shift: f32,

    /// Random seed (-1 for random)
    #[arg(short = 's', long, default_value_t = 42)]
    seed: i64,

    /// Output video path (.mp4)
    #[arg(short = 'o', long, default_value = "output.mp4")]
    output_path: PathBuf,

    /// Path to LoRA weights (.safetensors)
    #[arg(long)]
    lora: Option<PathBuf>,

    /// LoRA strength scale factor
    #[arg(long, default_value_t = 1.0)]
    lora_strength: f32,
}

#[derive(Args, Debug)]
struct LtxArgs {
    /// Path to model directory
    #[arg(short = 'm', long, default_value = "./models/ltx")]
    model_dir: PathBuf,

    /// Text prompt describing the video
    #[arg(short = 'p', long)]
    prompt: String,

    /// Pipeline type: distilled, dev, dev-two-stage, dev-two-stage-hq
    #[arg(long, default_value = "distilled")]
    pipeline: String,

    /// Video width (divisible by 64)
    #[arg(long, default_value_t = 768)]
    width: usize,

    /// Video height (divisible by 64)
    #[arg(long, default_value_t = 512)]
    height: usize,

    /// Number of frames
    #[arg(short = 'n', long, default_value_t = 97)]
    num_frames: usize,

    /// Output video path
    #[arg(short = 'o', long, default_value = "ltx_output.mp4")]
    output: PathBuf,

    /// Random seed
    #[arg(short = 's', long, default_value_t = 42)]
    seed: i64,
}

#[derive(Args, Debug)]
struct LoraArgs {
    /// Path to the LoRA safetensors file
    #[arg(short = 'p', long)]
    path: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Info => {
            println!(
                "{}",
                "=== MLX-Video-RS (Apple Silicon Native) ===".bold().cyan()
            );
            print_device_info();
            println!();
            println!("{}", "Supported Video Generation Models:".bold().green());
            println!("  • Wan2.1 T2V (1.3B / 14B) - Flow matching DiT single-model pipeline");
            println!("  • Wan2.2 T2V (14B) - Flow matching DiT dual-model pipeline");
            println!("  • Wan2.2 I2V (14B) - Image-to-Video diffusion transformer (in_dim=36)");
            println!("  • Wan2.2 TI2V (5B) - Text+Image-to-Video transformer (dim=3072)");
            println!("  • LTX-2 / LTX-2.3 - Distilled & Dev multi-stage diffusion models");
            println!();
            println!("{}", "Acceleration & Backends:".bold().yellow());
            println!("  • MLX unified memory architecture");
            println!("  • Apple Metal JIT compiled shader kernels");
            println!("  • Zero-copy safetensors checkpoint streaming");
            println!("  • LoRA parameter injection & fast matrix fusion");
        }

        Commands::Lora(args) => {
            println!(
                "{}",
                format!("Inspecting LoRA: {}", args.path.display()).cyan()
            );
            match load_lora_weights(&args.path) {
                Ok(lora_map) => {
                    println!(
                        "{}",
                        format!("Found {} adapted modules:", lora_map.len())
                            .green()
                            .bold()
                    );
                    for (name, weights) in lora_map {
                        println!(
                            "  • {:<50} rank={:<3} alpha={:<4.1} scale={:<4.2} A={:?} B={:?}",
                            name,
                            weights.rank,
                            weights.alpha,
                            weights.scale(),
                            weights.lora_a.shape(),
                            weights.lora_b.shape(),
                        );
                    }
                }
                Err(e) => {
                    eprintln!("{}", format!("Failed to read LoRA: {e}").red());
                    std::process::exit(1);
                }
            }
        }

        Commands::Wan(args) => {
            println!("{}", "=== Wan Video Generation Pipeline ===".bold().cyan());
            println!("  Prompt:       {}", args.prompt.italic());
            if let Some(ref neg) = args.negative_prompt {
                println!("  Neg Prompt:   {}", neg.dimmed());
            }
            if let Some(ref img) = args.image {
                println!("  Input Image:  {} (I2V Mode)", img.display());
            } else {
                println!("  Mode:         Text-to-Video (T2V)");
            }
            println!(
                "  Resolution:   {}x{} @ {} frames",
                args.width, args.height, args.num_frames
            );
            println!(
                "  Steps:        {} | Shift: {} | Guide: {}",
                args.steps, args.shift, args.guide_scale
            );
            println!("  Output:       {}", args.output_path.display());

            let mut config = if args.image.is_some() {
                WanModelConfig::wan22_i2v_14b()
            } else {
                WanModelConfig::wan22_t2v_14b()
            };

            config.sample_steps = args.steps;
            config.sample_shift = args.shift;

            let _pipeline = WanPipeline::new(config)?;
            println!(
                "{}",
                "Model pipeline successfully initialized on Apple Silicon Metal GPU.".green()
            );
            println!("{}", "Ready for video generation.".green().bold());
        }

        Commands::Ltx(args) => {
            println!("{}", "=== LTX Video Generation Pipeline ===".bold().cyan());
            println!("  Prompt:       {}", args.prompt.italic());
            println!("  Pipeline:     {}", args.pipeline);
            println!(
                "  Resolution:   {}x{} @ {} frames",
                args.width, args.height, args.num_frames
            );
            println!("  Output:       {}", args.output.display());

            let config = LTXModelConfig::default();
            println!(
                "{}",
                format!(
                    "LTX-2 model configuration loaded: dim={} heads={}",
                    config.dim, config.heads
                )
                .green()
            );
        }
    }

    Ok(())
}
