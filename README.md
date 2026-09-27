# mlx-video-rs 🎬🦀

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-black?logo=apple)](https://apple.com)
[![Metal Accelerated](https://img.shields.io/badge/Metal-Accelerated-orange?logo=apple)](https://developer.apple.com/metal/)
[![Rust](https://img.shields.io/badge/Rust-1.80%2B-red?logo=rust)](https://www.rust-lang.org/)

High-performance, memory-efficient Rust port of [Blaizzy/mlx-video](https://github.com/Blaizzy/mlx-video) for state-of-the-art **Image-Video-Audio generation** models on Apple Silicon (M1/M2/M3/M4) using **MLX** and Apple Metal GPU acceleration.

---

## ✨ Features

- **Apple Silicon Native**: Direct integration with Apple Metal GPU and Unified Memory Architecture via `mlx-rs` (v0.32.0) and `mlx-sys`.
- **Wan2.1 & Wan2.2 DiT**:
  - Full Diffusion Transformer (DiT) implementation with learned modulation routing (RMSNorm, LayerNorm, Linear modulation).
  - 3-Way Factorized 3D Rotary Position Embeddings (RoPE) across temporal ($F$), height ($H$), and width ($W$) axes.
  - Cross-attention with conditioning text embeddings from T5/UMT5 encoder.
  - Multi-head self-attention with Query-Key RMS normalization.
- **Image-to-Video (I2V)**:
  - Channel dimension concatenation ($16 + 16 + 4 = 36$ latent channels) for Wan2.2 I2V.
  - High-precision Lanczos downsampling and center-cropping image preprocessor.
  - Latent mask projection and token construction.
- **LTX-2 & LTX-2.3**:
  - Dual-stage flow matching diffusion pipeline (Distilled 8-step & Dev 40-step).
  - Custom sigma schedules and shift interpolation.
- **Flow Matching Schedulers**:
  - `FlowMatchEulerScheduler` with dynamic timestep shifting ($S = \text{shift}$).
  - `FlowDPMPP2MScheduler` multi-step higher-order solver.
  - Classifier-Free Guidance (CFG) inference loop.
- **Video 3D VAE**:
  - Causal 3D Convolutions (`CausalConv3d`) preserving temporal causality.
  - 16-channel video latent mean and standard deviation normalizations.
  - Spatial & temporal latent tiling engine for arbitrary resolution and frame counts without OOM.
- **Zero-Copy SafeTensors Streaming**:
  - Direct zero-copy tensor loading from single files or sharded checkpoint directories (`model-00001-of-00004.safetensors`).
- **LoRA Adapter Support**:
  - Dynamically load and inspect LoRA weights from `.safetensors`.
  - Automatic detection of `{module}.lora_A` / `{module}.lora_B` and `{module}.lora_down` / `{module}.lora_up` conventions.
  - Multi-LoRA weight fusion: $\Delta W = \text{strength} \cdot \frac{\alpha}{r} (B \times A)$.

---

## 🚀 Supported Model Architectures

| Architecture | Parameters | Modality | Default Latent Dim | Layers | Heads |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Wan2.1-T2V-1.3B** | 1.3 Billion | Text-to-Video | 1536 | 30 | 12 |
| **Wan2.1-T2V-14B** | 14 Billion | Text-to-Video | 5120 | 40 | 40 |
| **Wan2.2-T2V-14B** | 14 Billion | Text-to-Video (Dual) | 5120 | 40 | 40 |
| **Wan2.2-I2V-14B** | 14 Billion | Image-to-Video | 5120 (in=36) | 40 | 40 |
| **Wan2.2-TI2V-5B** | 5 Billion | Text+Image-to-Video | 3072 | 32 | 24 |
| **LTX-2 / LTX-2.3** | Variable | Multi-Stage Flow | 2048 | 28 | 32 |

---

## 🛠️ Prerequisites & Installation

### System Requirements
- Apple Silicon Mac (M1, M2, M3, M4, Pro, Max, or Ultra)
- macOS Sonoma (14.0) or macOS Sequoia (15.0+)
- Xcode 16+ or Command Line Tools with Metal Toolchain:
  ```bash
  xcodebuild -downloadComponent MetalToolchain
  ```
- Rust 1.80+:
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```

### Build from Source
```bash
git clone https://github.com/bhubbard/mlx-video-rs.git
cd mlx-video-rs
make build
```

Binary will be produced at `target/release/mlx-video`.

---

## 💻 CLI Usage

### System & Metal GPU Diagnostics
```bash
mlx-video info
```

Output:
```
=== MLX-Video-RS (Apple Silicon Native) ===
MLX Device: Apple Silicon GPU (Metal Accelerated)

Supported Video Generation Models:
  • Wan2.1 T2V (1.3B / 14B) - Flow matching DiT single-model pipeline
  • Wan2.2 T2V (14B) - Flow matching DiT dual-model pipeline
  • Wan2.2 I2V (14B) - Image-to-Video diffusion transformer (in_dim=36)
  • Wan2.2 TI2V (5B) - Text+Image-to-Video transformer (dim=3072)
  • LTX-2 / LTX-2.3 - Distilled & Dev multi-stage diffusion models
```

### Text-to-Video Generation (Wan2.1 / Wan2.2)
```bash
mlx-video wan generate \
  --prompt "A cinematic drone shot flying through a neon futuristic Tokyo in the rain, 8k, photorealistic" \
  --model wan21_14b \
  --width 832 \
  --height 480 \
  --num-frames 81 \
  --steps 40 \
  --guide-scale 6.0 \
  --output tokyo_future.mp4
```

### Image-to-Video Generation (Wan2.2 I2V)
```bash
mlx-video wan generate \
  --prompt "Camera zooms in smoothly as the waterfall flows and mist rises" \
  --image landscape.png \
  --model wan22_i2v \
  --width 832 \
  --height 480 \
  --num-frames 81 \
  --steps 40 \
  --output waterfall_motion.mp4
```

### LTX-2 Multi-Stage Generation
```bash
mlx-video ltx generate \
  --prompt "Macro shot of a dew drop falling from a vibrant green leaf" \
  --pipeline distilled \
  --steps 8 \
  --width 768 \
  --height 512 \
  --num-frames 65 \
  --output leaf_drop.mp4
```

### Inspect LoRA Checkpoints
```bash
mlx-video lora inspect /path/to/my_video_lora.safetensors
```

---

## 🧪 Testing & Code Coverage

Run all unit tests:
```bash
make test
```

Generate LLVM HTML code coverage:
```bash
make coverage
open target/llvm-cov/html/index.html
```

---

## 🏛️ Project Architecture

```
mlx-video-rs/
├── Cargo.toml
├── Makefile
├── src/
│   ├── lib.rs                # Library exports & integration tests
│   ├── error.rs              # Custom VideoError & Result types
│   ├── bin/
│   │   └── mlx-video.rs      # Unified CLI application
│   ├── lora/                 # LoRA adapter loading & weight fusion
│   │   ├── mod.rs
│   │   ├── types.rs          # LoRA weights, ranks, and alpha
│   │   └── loader.rs         # SafeTensors dynamic key parser
│   ├── models/
│   │   ├── wan/              # Wan2.1 & Wan2.2 Diffusion Transformers
│   │   │   ├── mod.rs
│   │   │   ├── config.rs     # WanModelConfig & architectural presets
│   │   │   ├── model.rs      # WanModel root DiT and sinusoidal embeddings
│   │   │   ├── transformer.rs# WanAttentionBlock with AdaLN modulation
│   │   │   ├── attention.rs  # QK-norm self-attention & cross-attention
│   │   │   ├── rope.rs       # 3-Way factorized (temporal/height/width) RoPE
│   │   │   ├── scheduler.rs  # FlowMatch Euler & DPM++ 2M schedulers
│   │   │   ├── vae.rs        # 3D Video VAE & CausalConv3d
│   │   │   ├── i2v.rs        # Image-to-video preprocessor & mask builder
│   │   │   ├── tiling.rs     # Spatial/temporal VAE latent tiling
│   │   │   └── pipeline.rs   # WanPipeline sampling & generation loop
│   │   └── ltx/              # LTX-2 & LTX-2.3 Diffusion Models
│   │       ├── mod.rs
│   │       ├── config.rs     # LTXModelConfig & sigma presets
│   │       └── samplers.rs   # Distilled & Dev multi-stage flow samplers
│   └── utils/
│       ├── mod.rs
│       ├── weights.rs        # SafeTensors zero-copy checkpoint loader
│       └── device.rs         # Apple Metal device info diagnostics
```

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.
