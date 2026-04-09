# 🏭 Industrial Defect Detection — ViT + β-VAE + DDPM

**MSc Thesis Project | University of Debrecen — Faculty of Informatics | Data Science MSc**

> *Developing High-Level AI Architectures and Systems for Industrial Defect Detection using Transformers, VAEs, and Diffusion Models*

**Author:** Paudel Dipak &nbsp;|&nbsp; **Supervisor:** Dr. Robert Lakatos, Assistant Professor

---

## 🚀 Live Demo

Try the deployed application directly in your browser — no installation required:

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-blue?style=for-the-badge)](https://huggingface.co/spaces/dipakpaudel333/Industrial-Defect-Detection)

**→ [https://huggingface.co/spaces/dipakpaudel333/Industrial-Defect-Detection](https://huggingface.co/spaces/dipakpaudel333/Industrial-Defect-Detection)**

Upload any steel surface image and the system will:
- **Classify** the defect type with confidence score
- **Localise** the anomaly with a colour-coded heatmap (blue = normal, red = anomaly)
- **Quantify** severity: NORMAL / LOW / MODERATE / HIGH
- **Visualise** the DDPM forward diffusion process in Generation mode

---

## 📋 Overview

This project implements a unified industrial surface defect detection pipeline combining three deep learning paradigms, each addressing a distinct limitation of the others:

| Component | Architecture | Role |
|-----------|-------------|------|
| **ViT-B/16** | Vision Transformer (86M params) | Multi-class defect classification |
| **β-VAE** | Convolutional VAE, β=4.0 (25M params) | Unsupervised pixel-level anomaly localisation |
| **DDPM** | U-Net denoiser, T=1000 (13M params) | Synthetic defect image generation |
| **Fusion Classifier** | 3-stream MLP (1M params) | Late fusion of all three component outputs |

**Dataset:** NEU Steel Surface Defect Database — 1,800 greyscale images across 6 defect classes

**Results:**
- ✅ **100% test-set accuracy** (270/270 correct predictions)
- ✅ **Weighted F1-score: 1.0000**
- ✅ **Severity-proportional anomaly scoring** confirmed on live inference (crazing: 80.6% HIGH, inclusion: 26.1% MODERATE)
- ✅ **GPU-accelerated inference: 0.5–1.5 seconds per image**

---

## 🏗️ System Architecture

```
Input Image (224×224 RGB)
        │
        ├──────────────────────────┐
        │                          │
        ▼                          ▼
  ┌─────────────┐          ┌─────────────┐
  │  ViT-B/16   │          │   β-VAE     │
  │  ImageNet   │          │  [-1,+1]    │
  │  Norm       │          │  Norm       │
  └──────┬──────┘          └──────┬──────┘
         │                        │
   768-d [CLS]             256-d latent μ
   feature vector          + reconstruction x̂
         │                        │
         │                   Anomaly Score
         │                  A(x) = mean|x−x̂|
         │                        │
         └──────────┬─────────────┘
                    │
                    ▼
           ┌─────────────────┐
           │ Fusion Classifier│
           │ 769-d → MLP → C │
           └────────┬────────┘
                    │
                    ▼
            Class Prediction
           + Confidence %
           + Severity Label
           + Heatmap Overlay
```

The **DDPM** component is used offline for synthetic data augmentation and powers the **Generation mode** in the Gradio application.

---

## 📂 Repository Structure

```
├── Industrial_Defect_Detection.ipynb   # Main Colab notebook (run top-to-bottom)
├── app.py                              # Gradio application (deployed on HF Spaces)
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## ⚡ Quick Start

### Option A — Run in Google Colab (Recommended)

1. Open the notebook in Colab:

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/industrial-defect-detection/blob/main/Industrial_Defect_Detection.ipynb)

2. Set runtime to **GPU**: `Runtime → Change runtime type → GPU (T4 or A100)`

3. Run all cells top-to-bottom — the notebook handles everything:
   - Package installation
   - Dataset download (NEU Steel via Kaggle API or gdown)
   - Training all four components
   - Evaluation and live inference
   - Checkpoint saving

### Option B — Use the Live Demo

Visit the Hugging Face Space directly: **[dipakpaudel333/Industrial-Defect-Detection](https://huggingface.co/spaces/dipakpaudel333/Industrial-Defect-Detection)**

No setup required — just upload an image and click **Run Analysis**.

---

## 🗃️ Supported Datasets

| `DATASET_CHOICE` | Dataset | Images | Download |
|-----------------|---------|--------|---------|
| `'neu_steel'` | **NEU Steel Surface Defects** *(default)* | 1,800 | Auto via gdown / Kaggle |
| `'mvtec'` | MVTec Anomaly Detection | ~5,000 | Auto from mvtec.com |
| `'dagm'` | DAGM 2007 Industrial Textures | ~14,000 | Auto via gdown |
| `'kaggle_steel'` | Severstal Steel Defect Detection | 12,568 | Requires `kaggle.json` |
| `'custom'` | Your own images | Any | Upload a ZIP |

Edit `DATASET_CHOICE` in **Cell 3** of the notebook to switch datasets.

---

## 🔬 NEU Steel — Defect Classes

| Class | Description | Images |
|-------|-------------|--------|
| Crazing | Diffuse network of fine surface cracks | 300 |
| Inclusion | Foreign particles embedded in metal | 300 |
| Patches | Large surface discolouration zones | 300 |
| Pitted Surface | Localised material-loss cavities | 300 |
| Rolled-in Scale | Elongated streaks from hot rolling | 300 |
| Scratches | Fine abrasion marks from handling | 300 |

---

## 🧠 Model Details

### ViT-B/16 — Vision Transformer
- **Backbone:** `vit_base_patch16_224` pretrained on ImageNet-21k (14M images)
- **Progressive unfreezing:** Blocks 0–7 frozen in Phase 1 (epochs 1–10); all 86M parameters unfrozen in Phase 2 (epoch 10+)
- **Critical fix:** `CosineAnnealingLR` replaces `OneCycleLR` — the original scheduler caused NaN losses when new parameter groups were added at the unfreezing boundary
- **Custom head:** `768 → 512 → 256 → NUM_CLASSES` with LayerNorm, GELU, and Dropout

### β-VAE — Anomaly Detector
- **Architecture:** 5-stage convolutional encoder–decoder (224×224 → 7×7 → 256-d latent → 224×224)
- **β = 4.0** — enforces structured disentangled latent space
- **Anomaly score:** `A(x) = mean|x − x̂|` → calibrated to percentage → NORMAL / LOW / MODERATE / HIGH
- **Critical fix:** AMP (mixed precision) is **disabled** — `ConvTranspose2d` + `BatchNorm` + AMP silently corrupts weights without visible loss changes. Training in float32 resolves this.

### DDPM — Diffusion Model
- **Noise schedule:** Linear β₁=0.0001 → β_T=0.02, T=1,000 steps
- **Architecture:** Lightweight U-Net with sinusoidal timestep conditioning, operates at 56×56 internally
- **Generates** photorealistic defect images for data augmentation

### Fusion Classifier
- **Input streams:** ViT [CLS] features (768-d) + VAE latent mean μ (256-d) + scalar anomaly score (1-d) = 769-d
- **MLP:** `769 → 512 → 256 → NUM_CLASSES` with GELU and Dropout
- Only **~1% of total parameters** — the heavy lifting is done by the pretrained backbones

---

## 📊 Results

### Test Set Performance (NEU Steel)

| Metric | Score |
|--------|-------|
| Accuracy | **100.00%** (270/270) |
| Weighted F1 | **1.0000** |
| AUC-ROC | Macro OvR |

### Live Inference (April 2026)

| Image | Prediction | Confidence | Anomaly Score |
|-------|-----------|-----------|--------------|
| Crazing (flat surface) | ✅ Crazing | 66.53% | **80.6% HIGH** |
| Crazing (curved surface) | ✅ Crazing | 54.46% | **58.5% HIGH** |
| Inclusion | ✅ Inclusion | 90.24% | **26.1% MODERATE** |

The anomaly scores correctly reflect **physical severity**: crazing (diffuse crack network across the full surface) scores higher than inclusion (compact localised spot).

---

## ⚙️ Training Configuration

| Hyperparameter | ViT Phase 1 | ViT Phase 2 | β-VAE | DDPM | Fusion |
|---------------|------------|------------|-------|------|--------|
| Epochs | 10 | 20 | 40 | 50 | 20 |
| Learning Rate | 1×10⁻⁴ | 5×10⁻⁶ | 2×10⁻⁴ | 2×10⁻⁴ | 5×10⁻⁵ |
| Optimiser | AdamW | AdamW (rebuilt) | Adam | AdamW | AdamW |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR | CosineAnnealingLR |
| AMP | ✅ Enabled | ✅ Enabled | ❌ Disabled | ✅ Enabled | ✅ Enabled |
| Batch Size | 32 | 32 | 32 | 16 | 32 |
| Grad Clip | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |

**Data split:** 70% train / 15% validation / 15% test (seed=42, `WeightedRandomSampler` for class balance)

---

## 🔧 Known Issues & Fixes

### Fix 1 — ViT: OneCycleLR + Progressive Unfreezing → NaN Loss
**Problem:** `OneCycleLR` pre-computes its step schedule at initialisation. Adding new parameter groups at epoch 10 (`unfreeze_all()`) made the internal state inconsistent, causing NaN losses within 2 epochs.

**Fix:** Switch to `CosineAnnealingLR` and completely **rebuild** the optimiser and scheduler as new instances at epoch 10.

### Fix 2 — β-VAE: Silent NaN Weights from AMP
**Problem:** PyTorch AMP (`autocast`) with `ConvTranspose2d` + `BatchNorm2d` in the decoder silently corrupts model weights — the training loss stays finite, but reconstruction outputs collapse to near-uniform grey.

**Fix:** Disable AMP entirely for the VAE. Train exclusively in `float32`. ~15% slower but completely stable.

---

## 🖥️ Application Modes

### Detection Mode
1. Upload a surface image
2. ViT classifies the defect type and confidence
3. β-VAE generates a pixel-level anomaly heatmap
4. Results: class label, confidence %, anomaly severity, full probability distribution

### Generation Mode
1. Upload any surface image
2. DDPM forward diffusion visualised at t = 0, 200, 400, 600, 800, 999
3. Demonstrates how the model progressively corrupts the input to pure Gaussian noise (and learns to reverse this process to generate new defect images)

---

## 📦 Dependencies

```txt
torch
torchvision
timm
diffusers
transformers
accelerate
einops
torchmetrics
scikit-learn
matplotlib
seaborn
tqdm
kaggle
gdown
Pillow
opencv-python-headless
onnxscript
gradio
```

Install all at once:
```bash
pip install timm diffusers transformers accelerate einops torchmetrics \
            scikit-learn matplotlib seaborn tqdm kaggle gdown \
            Pillow opencv-python-headless onnxscript gradio
```

---

## 📁 Checkpoints

After training, the notebook saves the following files to `/content/checkpoints/`:

| File | Size | Contents |
|------|------|----------|
| `vit_best.pt` | ~334 MB | ViT-B/16 + head (best val_acc) |
| `vae_best.pt` | ~99 MB | β-VAE encoder + decoder (best ELBO) |
| `ddpm_best.pt` | ~52 MB | DDPM U-Net |
| `full_pipeline_best.pt` | ~449 MB | ViT + VAE + Fusion combined |
| `vit_neu_steel.onnx` | — | ViT exported to ONNX (opset 14) |

Download the primary checkpoint with the last notebook cell:
```python
from google.colab import files
files.download('/content/checkpoints/full_pipeline_best.pt')
```

---

## 📖 Citation

If you use this work, please cite:

```bibtex
@mastersthesis{paudel2026defect,
  author    = {Paudel, Dipak},
  title     = {Developing High-Level AI Architectures and Systems for Industrial
               Defect Detection using Transformers, VAEs, and Diffusion Models},
  school    = {University of Debrecen, Faculty of Informatics},
  year      = {2026},
  type      = {Master's Thesis},
  supervisor = {Dr. Robert Lakatos}
}
```

---

## 📚 Key References

| # | Paper |
|---|-------|
| [1] | Dosovitskiy et al. — *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale* (ICLR 2021) |
| [2] | Higgins et al. — *β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework* (ICLR 2017) |
| [3] | Ho et al. — *Denoising Diffusion Probabilistic Models* (NeurIPS 2020) |
| [4] | Song & Yan — *NEU Surface Defect Database* (Applied Surface Science, 2013) |
| [5] | Bergmann et al. — *MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection* (CVPR 2019) |

---

## 📜 License

This project is released for academic and research purposes.  
© 2026 Paudel Dipak — University of Debrecen, Faculty of Informatics.

---

<div align="center">

**Built with PyTorch · timm · Gradio · Hugging Face**

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Live%20Demo-Hugging%20Face-blue?style=flat-square)](https://huggingface.co/spaces/dipakpaudel333/Industrial-Defect-Detection)
&nbsp;&nbsp;
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
&nbsp;&nbsp;
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square&logo=pytorch)
&nbsp;&nbsp;
![License](https://img.shields.io/badge/License-Academic-green?style=flat-square)

*University of Debrecen · Faculty of Informatics · Data Science MSc · 2026*

</div>
