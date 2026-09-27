# MLX Rust Ecosystem & Roadmap 🦀🍏

This document tracks upcoming features, architectural improvements, and candidate MLX projects to convert to high-performance, native Apple Silicon Rust.

---

## 🎯 Current Project: `mlx-video-rs`

- [x] Initial port of Wan2.1 (1.3B & 14B) Diffusion Transformer (DiT).
- [x] Initial port of Wan2.2 (14B dual-model, 14B I2V 36-ch, 5B TI2V).
- [x] Initial port of LTX-2 & LTX-2.3 multi-stage flow matching.
- [x] 3-Way factorized 3D RoPE (temporal, height, width).
- [x] Flow matching Euler & 2nd-order DPM++ 2M schedulers.
- [x] Video 3D VAE with CausalConv3d and spatial/temporal tiling.
- [x] Dynamic LoRA adapter loading and on-the-fly matrix delta fusion ($\Delta W = s \cdot \frac{\alpha}{r} (B \times A)$).
- [x] Zero-copy SafeTensors streaming loader (single file & sharded).
- [x] CLI binary (`mlx-video`) with `wan`, `ltx`, `lora`, and `info` subcommands.
- [x] Comprehensive test suite (16 tests passing).
- [x] Automated LLVM code coverage (`cargo llvm-cov`) & Makefile.
- [x] GitHub Actions CI on macOS 15 Apple Silicon runners.
- [x] Interactive GitHub Pages landing site & CLI parameter playground.
- [ ] **Next for `mlx-video-rs`**:
  - [ ] Add 4-bit & 8-bit weight quantization support for 14B models on 16GB/24GB Macs.
  - [ ] Implement MMAudio joint video-to-audio flow matching for synchronized sound synthesis.
  - [ ] Add direct MP4/H.264 video encoding via macOS VideoToolbox hardware encoder.
  - [ ] Add interactive TUI (Terminal User Interface) with `ratatui` for live frame rendering.

---

## 🚀 Upcoming MLX Rust Ports (Tier 1 Candidates)

### 1. `mlx-audio-rs` (Port of [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio) • 7.9k ⭐)
> Ultra-low-latency, zero-dependency audio generation, transcription, and neural codecs. Eliminates Python GIL and GC audio dropouts.
- [ ] **TTS (Text-to-Speech)**:
  - [ ] Kokoro-82M (Sub-20ms first-audio-chunk time-to-first-token on Apple Silicon).
  - [ ] F5-TTS (Non-autoregressive flow matching speech synthesis).
  - [ ] ChatterBox & MeloTTS multilingual synthesis.
- [ ] **STT (Speech-to-Text)**:
  - [ ] OpenAI Whisper (tiny, base, small, medium, large-v3, turbo).
  - [ ] Distil-Whisper & Moonshine edge transcription.
- [ ] **VAD & Realtime Audio**:
  - [ ] Silero VAD native tensor processing.
  - [ ] Real-time mic stream capture & speaker output with `cpal` / `rodio`.
- [ ] **Neural Audio Codecs**:
  - [ ] SNAC (Multi-scale residual vector quantization).
  - [ ] EnCodec, Mimi, and DAC decoders/encoders.

---

### 2. `mflux-rs` (Port of [mflux-community/mflux](https://github.com/mflux-community/mflux) • 2.4k ⭐)
> State-of-the-art native image generation on Apple Silicon using MLX.
- [ ] **Core Diffusion Architectures**:
  - [ ] FLUX.1-schnell (4-step high-speed distillation).
  - [ ] FLUX.1-dev (guidance-distilled 12B DiT transformer).
  - [ ] FLUX.2 next-generation architectures.
- [ ] **Conditioning & Control**:
  - [ ] ControlNet integration (DepthPro, OpenPose, HED edge maps).
  - [ ] Inpainting & outpainting with masked latent flow matching.
  - [ ] Image-to-image latent perturbation pipelines.
- [ ] **Quantization**:
  - [ ] 4-bit (Q4_K, Q4_0) and 8-bit weight dequantization kernels.

---

### 3. `mlx-embeddings-rs` (Port of [Blaizzy/mlx-embeddings](https://github.com/Blaizzy/mlx-embeddings) • 445 ⭐)
> High-throughput text and vision embedding models that directly link into Rust vector databases (Qdrant, LanceDB) without IPC overhead.
- [ ] **Text Embeddings**:
  - [ ] BGE-M3 (dense, sparse, multi-lingual 8192-token context).
  - [ ] Nomic-Embed-Text (Matryoshka representation learning).
  - [ ] ModernBERT (flash-attention enabled bidirectional transformer).
- [ ] **Multimodal Embeddings**:
  - [ ] CLIP (ViT-B/32, ViT-L/14) and SigLIP vision-text contrastive encoders.
- [ ] **Rerankers**:
  - [ ] BGE-Reranker-large and ColBERT late-interaction token scoring.

---

### 4. `mlx-vlm-rs` (Port of [Blaizzy/mlx-vlm](https://github.com/Blaizzy/mlx-vlm) • 5.5k ⭐)
> Vision-Language Models (VLM) for zero-latency screen understanding, document OCR, and multimodal agent tools.
- [ ] **VLM Model Implementations**:
  - [ ] Qwen2-VL & Qwen2.5-VL (dynamic resolution patch grid).
  - [ ] Pixtral (12B multimodal transformer with native 128k context).
  - [ ] SmolVLM (compact edge-friendly vision model).
  - [ ] Gemma-Vision & Llama-3.2-Vision.
- [ ] **Optimizations**:
  - [ ] Vision token caching (skip re-encoding static screen/camera frames).
  - [ ] Cross-attention key-value compression.

---

### 5. `mlx-serve-rs` (Inspired by [ddalcu/mlx-serve](https://github.com/ddalcu/mlx-serve) • 1.6k ⭐ & [jundot/omlx](https://github.com/jundot/omlx) • 22.2k ⭐)
> Production-grade, zero-Python OpenAI & Anthropic API server built in Rust with `axum` and `tokio`.
- [ ] **Server Core**:
  - [ ] OpenAI `/v1/chat/completions` and `/v1/models` endpoints.
  - [ ] Anthropic `/v1/messages` endpoint with SSE streaming.
  - [ ] Structured JSON output enforcement via grammar-guided sampling.
- [ ] **Serving Engine**:
  - [ ] Continuous batching with dynamic token allocation.
  - [ ] Prefix caching (Radix tree) for multi-turn chats and system prompts.
  - [ ] Tool calling and MCP (Model Context Protocol) handler integration.

---

## 💡 Advanced Architectural Ideas & Features to Adopt

### 1. Speculative Decoding & Multi-Token Prediction (MTP)
- **Reference**: [youssofal/MTPLX](https://github.com/youssofal/MTPLX) (125 tok/s on Mac).
- [ ] Implement draft model / dual-head parallel speculative verification.
- [ ] Take advantage of Apple Silicon unified memory (zero-copy draft tensor verification).

### 2. SSD-Paged KV Disk Caching
- **Reference**: [jundot/omlx](https://github.com/jundot/omlx) & [ddalcu/mlx-serve/kv_disk_cache.zig](https://github.com/ddalcu/mlx-serve/blob/main/src/kv_disk_cache.zig).
- [ ] Memory-map attention KV caches to NVMe SSD using Apple APFS sparse files.
- [ ] Support 128k+ token context windows on 16GB–36GB Macs without OOM.

### 3. Apple Neural Engine (ANE) Hybrid Offload
- **Reference**: [ddalcu/mlx-serve/ane.zig](https://github.com/ddalcu/mlx-serve/blob/main/src/ane.zig).
- [ ] Offload fixed-dimension embeddings, VAE conv blocks, and VAD directly to the Apple Neural Engine.
- [ ] Free Metal GPU compute cores exclusively for heavy attention matmuls.

### 4. TurboQuant Dynamic KV Compression
- **Reference**: [Blaizzy/mlx-vlm/turboquant.py](https://github.com/Blaizzy/mlx-vlm/blob/main/mlx_vlm/turboquant.py).
- [ ] 4-bit and 8-bit dynamic quantization of attention keys and values during generation.
- [ ] Cut RAM usage by up to 60% during long generations.

---

## 📦 Packaging & Distribution

- [ ] Create Homebrew formula (`brew install bhubbard/tap/mlx-video`).
- [ ] Publish crate to crates.io (`cargo install mlx-video-rs`).
- [ ] Provide precompiled Universal macOS binaries (Apple Silicon arm64) via GitHub Releases.
