# Hunyuan3D — Text-to-3D Pipeline with Texture

Generate textured 3D models from text prompts using **ComfyUI**, **SDXL**, and **Hunyuan3D 2.0** with the Paint pipeline.

This setup splits the generation into two phases to manage GPU memory efficiently, and runs on **Google Colab** (free T4 GPU) with a tunnel for browser access.

---

## Architecture

```
Phase 1 (Celda 2)                          Phase 2 (Celda 3)
┌─────────────────────────┐                ┌──────────────────────────────┐
│  Text Prompt             │                │  Load Image + Mesh            │
│       ↓                  │                │       ↓              ↓        │
│  SDXL (text → image)     │                │  Delight        UV Wrap       │
│       ↓                  │                │  (remove shadows)   ↓         │
│  Hunyuan3D DiT           │                │       ↓        Render 6 views │
│  (image → 3D shape)      │                │  Paint Model                  │
│       ↓                  │                │  (generate multiview textures)│
│  VAE Decode → Mesh       │   ──────►      │       ↓                       │
│       ↓                  │   .glb file    │  Bake → Apply → Export        │
│  Export .glb (no color)  │                │       ↓                       │
└─────────────────────────┘                │  Textured .glb                │
                                           └──────────────────────────────┘
```

---

## Files

| File | Description |
|------|-------------|
| `Comfy_Colab_Workflow.ipynb` | Google Colab notebook with 3 cells (setup, shape, paint) |
| `workflow_shape.json` | ComfyUI workflow — text → image → 3D mesh (no texture) |
| `workflow_paint.json` | ComfyUI workflow — mesh + image → textured 3D model |

---

## Prerequisites

- A **Google account** to use [Google Colab](https://colab.research.google.com)
- A modern web browser (Chrome, Firefox, Safari)
- The three files listed above downloaded to your local machine

---

## Setup

### 1. Download ComfyUI (local, optional)

If you want to run workflows locally (shape generation only — paint requires CUDA GPU):

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
python main.py
```

Open `http://127.0.0.1:8188` in your browser and drag any `.json` workflow file onto the canvas.

> **Note:** Local execution on macOS/Apple Silicon supports shape generation but not texture painting due to CUDA dependencies in the rasterizer. Use the Colab notebook for the full pipeline with texture.

### 2. Google Colab Setup (recommended — full pipeline)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `Comfy_Colab_Workflow.ipynb`
3. Go to **Runtime → Change runtime type → T4 GPU**

---

## Usage

The notebook contains **3 cells** that must be executed in order. Each cell serves a specific purpose and manages GPU memory independently.

### Cell 1 — Setup

Installs all dependencies and downloads models (~25 GB total, runs once per session).

**What it installs:**
- ComfyUI
- ComfyUI-Hunyuan3DWrapper (kijai) with CUDA rasterizer
- Cloudflare tunnel for browser access
- pyfqmr for mesh operations

**Models downloaded:**
| Model | Size | Purpose |
|-------|------|---------|
| SDXL Base 1.0 | 6.5 GB | Text-to-image generation |
| Hunyuan3D DiT v2-0-fast | 4.9 GB | Image-to-3D shape generation |
| Hunyuan3D VAE v2-0 | 430 MB | Latent-to-voxel decoding |
| Hunyuan3D Paint v2-0 | 8.6 GB | Multiview texture generation |
| Hunyuan3D Delight v2-0 | 4.0 GB | Shadow/highlight removal |

**Action:** Execute and wait for `SETUP COMPLETO` message.

---

### Cell 2 — Generate Image + 3D Shape

Starts ComfyUI with a Cloudflare tunnel. Only loads SDXL + Hunyuan3D DiT (~9 GB RAM).

**Action:**
1. Execute the cell
2. Wait for the tunnel URL to appear (e.g., `https://xxx.trycloudflare.com`)
3. Open the URL in your browser
4. Drag `workflow_shape.json` onto the ComfyUI canvas
5. Edit the **Prompt Positivo** node with your desired text description
6. Click **Ejecutar** (Execute)
7. Wait for completion — outputs are saved to:
   - `output/generated/` — generated image (.png)
   - `output/mesh/` — 3D mesh without texture (.glb)
8. **Stop this cell** before proceeding to Cell 3

> **Tip:** For best 3D results, include "centered, white background, full body, studio lighting" in your prompt.

---

### Cell 3 — Texturize Mesh

Starts a fresh ComfyUI instance. Only loads Paint + Delight models (~10 GB RAM).

Automatically copies the outputs from Cell 2 into the input folder so the paint workflow can access them.

**Action:**
1. **Ensure Cell 2 is stopped** (this frees RAM for the paint models)
2. Execute the cell
3. Open the new tunnel URL in your browser
4. Drag `workflow_paint.json` onto the ComfyUI canvas
5. In the **Cargar Imagen de Referencia** node, select your generated image
6. In the **Cargar Mesh** node, select your generated mesh
7. Click **Ejecutar** (Execute)
8. Wait for completion — outputs are saved to:
   - `output/multiview/` — 6 texture views generated by the paint model
   - `output/textures/` — UV texture map
   - `output/mesh/con_textura` — final textured 3D model (.glb)

---

## Important Notes

### Memory Management
The pipeline is split into two phases because Colab's free tier has ~12 GB system RAM. Loading all models simultaneously (SDXL + DiT + Paint + Delight ≈ 24 GB) exceeds this limit. Each cell starts a fresh ComfyUI process with only the required models.

### Session Limits
- Colab free tier sessions last approximately 4–12 hours
- Models must be re-downloaded each new session (~15 minutes)
- Files in `output/` persist within the same session but are lost on disconnect

### Tunnel Stability
Cloudflare's free tunnel service (`trycloudflare.com`) may occasionally disconnect. If you see "Reconnecting" in the ComfyUI interface, wait a few seconds or refresh the page. If the connection doesn't recover, restart the current cell.

### Known Limitations
- Texture seams may be visible where the 6 multiview projections meet — this is a known limitation of Hunyuan3D Paint v2.0
- The paint model requires CUDA (NVIDIA GPU) and cannot run on CPU or Apple Silicon with acceptable quality
- Mesh generation works on any platform (macOS, Linux, Windows) but texture painting requires the Colab GPU setup

---

## Workflow Parameters

### workflow_shape.json

| Node | Parameter | Default | Description |
|------|-----------|---------|-------------|
| KSampler (SDXL) | steps | 25 | Diffusion steps for image generation |
| KSampler (SDXL) | cfg | 7.0 | Classifier-free guidance scale |
| Hy3DGenerateMesh | guidance_scale | 5.5 | 3D shape guidance |
| Hy3DGenerateMesh | steps | 30 | Diffusion steps for shape generation |
| Hy3DVAEDecode | octree_resolution | 256 | Voxel grid resolution |

### workflow_paint.json

| Node | Parameter | Default | Description |
|------|-----------|---------|-------------|
| Hy3DDelightImage | steps | 50 | Denoising steps for shadow removal |
| Hy3DRenderMultiView | render_size | 1024 | Resolution for normal/position maps |
| Hy3DRenderMultiView | texture_size | 1024 | Output texture resolution |
| Hy3DSampleMultiView | steps | 30 | Diffusion steps for texture generation |
| Hy3DSampleMultiView | view_size | 512 | Per-view render resolution |

---

## Models & References

- [Hunyuan3D-2](https://huggingface.co/tencent/Hunyuan3D-2) — Tencent's 3D generation model
- [SDXL Base 1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) — Stability AI's text-to-image model
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI) — Node-based AI workflow engine
- [ComfyUI-Hunyuan3DWrapper](https://github.com/kijai/ComfyUI-Hunyuan3DWrapper) — Hunyuan3D integration for ComfyUI

---

## License

The models used in this pipeline are subject to their respective licenses:
- Hunyuan3D-2: [Tencent Hunyuan Community License](https://huggingface.co/tencent/Hunyuan3D-2/blob/main/LICENSE)
- SDXL: [OpenRAIL++](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md)
- ComfyUI: [GPL-3.0](https://github.com/comfyanonymous/ComfyUI/blob/master/LICENSE)
