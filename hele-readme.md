# PersonaLive - Setup & Usage Notes (RTX 3090 eGPU on Windows)

## What is PersonaLive?

PersonaLive (CVPR 2026) is a real-time, streamable diffusion framework for expressive portrait animation. It takes a **reference image** (a face photo) and a **driving video** (someone moving/talking), and generates a video where the face in the photo moves like the person in the driving video. Requires only ~12GB VRAM.

## Repository Origins

The code comes from two upstream sources, now consolidated into a single local directory:

| Source | URL | Provides |
|--------|-----|----------|
| **GitHub** | `https://github.com/GVCLab/PersonaLive` | Source code, configs, demo files, requirements |
| **HuggingFace** | `https://huggingface.co/huaichang/PersonaLive` | Pretrained weights (~17GB of .pth, .onnx, .engine files via Git LFS) |

The `tools/download_weights.py` script also downloaded two additional base models:
- `sd-image-variations-diffusers` (~1.7GB) - Stable Diffusion image variations model
- `sd-vae-ft-mse` (~335MB) - VAE decoder

## Local Setup

### Directory Layout

```
c:\git\PersonaLive-code\         <-- Single directory with everything
  ├── .venv\                     <-- Python 3.11 virtual environment
  ├── pretrained_weights\        <-- Model weights (~21GB)
  ├── demo\                      <-- Demo reference image + driving video
  ├── src\                       <-- Model source code
  │   └── pipelines/
  │       ├── pipeline_pose2vid.py       <-- Original pipeline (DO NOT MODIFY)
  │       └── hele_pipeline_pose2vid.py  <-- Our copy with customizations
  ├── configs\                   <-- Config YAML files
  ├── inference_offline.py       <-- Original offline inference script (DO NOT MODIFY)
  ├── hele_inference_offline.py  <-- Our copy with customizations
  ├── inference_online.py        <-- Online/webcam inference script
  ├── hele-readme.md             <-- This file
  └── results\                   <-- Output videos go here
```

### Pretrained Weights Structure

```
pretrained_weights/
├── personalive/
│   ├── denoising_unet.pth       (4.7 GB)
│   ├── reference_unet.pth       (3.3 GB)
│   ├── temporal_module.pth      (1.7 GB)
│   ├── motion_encoder.pth       (235 MB)
│   ├── motion_extractor.pth     (107 MB)
│   └── pose_guider.pth          (4 MB)
├── sd-image-variations-diffusers/
│   ├── image_encoder/
│   │   ├── pytorch_model.bin
│   │   └── config.json
│   ├── unet/
│   │   ├── diffusion_pytorch_model.bin
│   │   └── config.json
│   └── model_index.json
├── sd-vae-ft-mse/
│   ├── diffusion_pytorch_model.bin
│   └── config.json
├── onnx/                        (not needed for offline inference)
└── tensorrt/                    (not needed for offline inference)
```

### Hardware

- **GPU:** NVIDIA GeForce RTX 3090 (24GB VRAM) via eGPU
- **CUDA Driver:** 591.74 (CUDA 13.1)
- **PyTorch CUDA:** 12.1
- **Architecture:** Ampere (sm_86) - fully supported by standard PyTorch + xformers

### Python Environment

Created with `python -m venv` (Python 3.11.9). Key packages:

| Package | Version | Notes |
|---------|---------|-------|
| torch | 2.1.0+cu121 | From PyTorch index, CUDA 12.1 |
| xformers | 0.0.22.post7 | Memory-efficient attention, works on RTX 3090 |
| diffusers | 0.27.0 | Specific version required |
| transformers | 4.36.2 | Specific version required |
| mediapipe | 0.10.11 | Face mesh detection |
| decord | 0.6.0 | Video reading |
| numpy | 1.26.4 | Must be <2 (mediapipe compat) |
| av | 17.0.0 | Video encoding/decoding |

## Commands

### Activate Environment

```powershell
# No conda needed - using venv directly
c:\git\PersonaLive-code\.venv\Scripts\activate
```

Or prefix commands with the full Python path:
```powershell
c:\git\PersonaLive-code\.venv\Scripts\python.exe
```

### Offline Inference (Picture + Driving Video → Animated Video)

**Using the included demo files:**
```powershell
cd c:\git\PersonaLive-code
.venv\Scripts\python.exe hele_inference_offline.py
```

**With your own files:**
```powershell
.venv\Scripts\python.exe hele_inference_offline.py --reference_image "path\to\your\photo.png" --driving_video "path\to\your\video.mp4"
```

**Quick test (fewer frames, faster):**
```powershell
.venv\Scripts\python.exe hele_inference_offline.py -L 20
```

**With TAESD (faster decoding, slightly lower quality):**
```powershell
.venv\Scripts\python.exe hele_inference_offline.py --use_taesd --reference_image "demo\guy2.png" --driving_video "demo\driving_video.mp4"
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--reference_image` | (from config) | Path to the face image you want to animate |
| `--driving_video` | (from config) | Path to the video with motions to transfer |
| `-L` | 100 | Number of frames to process |
| `-W` | 512 | Output width |
| `-H` | 512 | Output height |
| `--use_xformers` | True | Use xformers for memory efficiency (keep True for RTX 3090) |
| `--use_taesd` | False | Use Tiny AutoEncoder for ~27% faster decoding (flag, no value needed) |
| `--stream_gen` | True | Streaming generation to reduce VRAM usage |
| `--seed` | 42 | Random seed for reproducibility |

### Output

Results are saved to `results/<date>--personalive_offline/`:
- `concat_vid/` - Side-by-side video: reference | driving face | driving video | generated output
- `split_vid/` - Clean generated video only (with audio from driving video if available)

### Performance (RTX 3090)

- 20 frames: ~80 seconds
- ~1 second per denoising step per window
- xformers enabled for optimized attention

## File Strategy

We **never modify original repo files**. Instead, we create `hele_` prefixed copies:

| Original | Our Copy | Purpose |
|----------|----------|---------|
| `inference_offline.py` | `hele_inference_offline.py` | Entry point with our customizations |
| `src/pipelines/pipeline_pose2vid.py` | `src/pipelines/hele_pipeline_pose2vid.py` | Pipeline with our customizations |

`hele_inference_offline.py` imports from `hele_pipeline_pose2vid.py`. This way `git pull` won't create merge conflicts and the original code stays untouched.

## Notes

- **RTX 3090 (Ampere):** Standard PyTorch + xformers works perfectly. No nightly builds needed.
- **RTX 50-series (Blackwell):** Requires PyTorch nightly and `--use_xformers False` (xformers not yet compatible with sm_120).
- **TensorRT:** Optional acceleration (~2x speed). The included `.engine` file is for H100 - you'd need to rebuild with `python torch2trt.py` for your GPU. Not set up yet.
- **Triton warning:** "No module named 'triton'" is harmless - Triton is Linux-only and not needed on Windows.
- **Video length:** The `-L` parameter controls max frames. Videos are processed at the driving video's native resolution, cropped to faces. The `--stream_gen True` mode allows arbitrarily long videos within 12GB VRAM.

## TODO

- [ ] Test with custom reference images and driving videos
- [ ] Try longer video generation
- [ ] Explore online/webcam inference
- [ ] Consider TensorRT acceleration for RTX 3090
