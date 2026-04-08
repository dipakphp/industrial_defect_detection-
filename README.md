# 🏭 Industrial Defect Detection — ViT + β-VAE + DDPM

A production-ready deep learning pipeline for industrial surface defect detection,
combining **Vision Transformers**, **Beta-Variational Autoencoders**, and
**Denoising Diffusion Probabilistic Models** into a unified, deployable system.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange)](https://gradio.app)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Supported Datasets](#-supported-datasets)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Gradio Web App](#-gradio-web-app)
- [Results](#-results)
- [Known Issues & Fixes](#-known-issues--fixes)
- [Citation](#-citation)

---

## 🔍 Overview

This project implements a **tri-modal defect detection pipeline** that:

| Component | Role | Key Feature |
|-----------|------|-------------|
| **ViT-B/16** | Multi-class defect classification | Global attention over full image |
| **β-VAE** | Unsupervised anomaly localisation | Pixel-level heatmap without labels |
| **DDPM U-Net** | Synthetic defect data generation | Stable diffusion-based augmentation |
| **Fusion Classifier** | Final prediction | Combines all three streams |

The system runs end-to-end on **Google Colab** (free tier with GPU) and ships a
**Gradio web interface** for real-time inference.

---

## 🏗 Architecture

```
Input Image (224×224 RGB)
        │
        ├──────────────────────────────────────────────────┐
        ▼                                                  ▼
  ┌─────────────┐                               ┌──────────────────┐
  │  ViT-B/16   │                               │    β-VAE         │
  │  (768-dim   │                               │  Encoder→Decoder │
  │   features) │                               │  Anomaly Score   │
  └──────┬──────┘                               └────────┬─────────┘
         │  768d                                         │ 256d + score
         └────────────────────┬──────────────────────────┘
                              ▼
                   ┌──────────────────┐
                   │ Fusion Classifier│
                   │  MLP(1025 → C)   │
                   └────────┬─────────┘
                            ▼
                     Final Prediction
              (class + confidence + heatmap)
```

**DDPM** runs offline to generate synthetic training data and is exposed in
Generation mode within the Gradio app.

---

## 📦 Supported Datasets

| Dataset | Classes | Images | Notes |
|---------|---------|--------|-------|
| **NEU Steel** ⭐ | 6 | 1,800 | Default — auto-downloads via Kaggle |
| **MVTec AD** | varies | ~5,000/category | One category at a time |
| **Kaggle Severstal Steel** | 5 | 12,568 | Requires `kaggle.json` |
| **DAGM 2007** | 10 | ~14,000 | Synthetic textures |
| **Custom** | any | any | ZIP with `class_name/img.jpg` layout |

---

## 📁 Project Structure

```
industrial_defect_detection/
│
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
│
├── configs/
│   └── config.py              # All hyperparameters in one place
│
├── src/
│   ├── __init__.py
│   ├── datasets.py            # MVTec, NEU Steel, Kaggle, DAGM adapters
│   ├── transforms.py          # Train / val / VAE normalisation pipelines
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vit.py             # ViT-B/16 with progressive unfreezing
│   │   ├── vae.py             # β-VAE (encoder + decoder + ELBO loss)
│   │   ├── ddpm.py            # DDPM U-Net + DDPMScheduler
│   │   └── fusion.py          # FusionClassifier + FullPipeline
│   └── utils/
│       ├── __init__.py
│       ├── training.py        # train_vit, train_vae, train_ddpm, train_fusion
│       ├── evaluation.py      # evaluate(), confusion matrix, ROC curves
│       └── heatmap.py         # Vivid anomaly heatmap generation
│
├── scripts/
│   ├── download_data.py       # Dataset download helpers
│   └── export_onnx.py         # ONNX export utility
│
├── app.py                     # Gradio web application (standalone)
│
├── tests/
│   ├── test_models.py         # Unit tests for model forward passes
│   └── test_datasets.py       # Unit tests for dataset adapters
│
├── docs/
│   └── ARCHITECTURE.md        # Detailed architecture notes
│
└── Industrial_Defect_Detection.ipynb   # Main Colab notebook
```

---

## ⚡ Quick Start

### Option A — Google Colab (Recommended)
Use this file for Google Colab [Industrial_Defect_Detection.ipynb](Industrial_Defect_Detection.ipynb)
1. Open `Industrial_Defect_Detection.ipynb` in [Google Colab](https://colab.research.google.com)
2. Select **Runtime → Change runtime type → GPU (T4)**
3. Run all cells top-to-bottom
4. The Gradio app launches automatically at the end

### Option B — Local Installation

```bash
git clone https://github.com/dipakphp/industrial-defect-detection.git
cd industrial-defect-detection

pip install -r requirements.txt

# Download NEU Steel dataset (requires kaggle.json in ~/.kaggle/)
python scripts/download_data.py --dataset neu_steel

# Train full pipeline
python -c "
from src.datasets import build_dataloaders
from src.models.vit import ViTFeatureExtractor
from src.models.vae import BetaVAE
from src.models.ddpm import UNet, DDPMScheduler
from src.models.fusion import FusionClassifier, FullPipeline
from src.utils.training import train_vit, train_vae, train_ddpm, train_fusion
from configs.config import Config

cfg = Config()
# ... see notebook for full pipeline
"
```

---

## ⚙️ Configuration

All hyperparameters live in `configs/config.py`:

```python
class Config:
    # Dataset
    DATASET_CHOICE  = 'neu_steel'   # mvtec | neu_steel | dagm | kaggle_steel | custom
    MVTEC_CATEGORY  = 'bottle'
    IMG_SIZE        = 224
    BATCH_SIZE      = 32

    # Training epochs
    EPOCHS_VIT      = 30
    EPOCHS_VAE      = 40
    EPOCHS_DDPM     = 50
    EPOCHS_CLF      = 20

    # Learning rates
    LR_VIT          = 1e-4
    WEIGHT_DECAY    = 1e-4

    # Model
    LATENT_DIM      = 256
    VAE_BETA        = 4.0
    TIMESTEPS       = 1000
    DDPM_CHANNELS   = 64

    # Paths
    CHECKPOINT_DIR  = './checkpoints'
    DATA_ROOT       = './data'
```

---

## 🏋️ Training

Training runs sequentially in the notebook. Each stage saves its best checkpoint:

```
Stage 1 → vit_best.pt          (ViT fine-tuning, ~15 min on T4)
Stage 2 → vae_best.pt          (β-VAE, ~20 min on T4)
Stage 3 → ddpm_best.pt         (DDPM, ~25 min on T4)
Stage 4 → full_pipeline_best.pt (Fusion, ~5 min on T4)
```

**Progressive Unfreezing (ViT):**
- Epochs 1–10: Train classification head + blocks 8–11 only
- Epoch 10: Unfreeze all 86M params, rebuild optimiser at LR = 5×10⁻⁶
- This avoids catastrophic forgetting and NaN instabilities

---

## 📊 Evaluation

```python
from src.utils.evaluation import evaluate

preds, labels, probs, metrics = evaluate(pipeline, test_loader)
# Output: Accuracy, Weighted F1, Macro AUC-ROC
# Plots:  Confusion matrix + per-class ROC curves
```

---

## 🌐 Gradio Web App

The standalone app is in `app.py`. Run it after placing checkpoints in `./checkpoints/`:

```bash
python app.py
```

**Two modes:**

| Mode | Description |
|------|-------------|
| **Detection** | Upload → ViT class prediction + β-VAE anomaly heatmap |
| **Generation** | Upload → DDPM forward diffusion visualisation (t=0 → t=999) |

**Anomaly severity scale:**

| Score | Level |
|-------|-------|
| < 8% | 🟢 NORMAL |
| 8–25% | 🟡 LOW |
| 25–55% | 🟠 MODERATE |
| > 55% | 🔴 HIGH |

---

## 📈 Results

Results on **NEU Steel Surface Defect Database** (6 classes, 1,800 images):

| Metric | Value |
|--------|-------|
| Test Accuracy | **100.00%** |
| Weighted F1 | **1.0000** |
| Macro AUC-ROC | **1.0000** |

> Live inference confirmed on Hugging Face Spaces — Crazing: 80.6% HIGH,
> Inclusion: 26.1% MODERATE (severity-proportional scoring validated). see [outputs](outputs)

---

## 🐛 Known Issues & Fixes

### 1. NaN Loss in ViT Training (Fixed ✅)
**Problem:** `OneCycleLR` + progressive unfreezing caused NaN at epoch 15.  
**Root cause:** Scheduler pre-computes step schedule at init; adding new param groups mid-run causes LR → ∞.  
**Fix:** Switched to `CosineAnnealingLR` + **fully rebuild optimiser + scheduler at epoch 10**.

### 2. Silent NaN Weights in β-VAE (Fixed ✅)
**Problem:** AMP (float16) + `ConvTranspose2d` + `BatchNorm2d` silently corrupted weights.  
**Root cause:** The NaN was invisible in training loss but apparent in near-zero reconstruction outputs.  
**Fix:** **Disabled AMP for VAE training** — all operations in float32 (~15% slower, 100% stable).

---

## 📄 License

MIT License — see [LICENSE](LICENSE).

---

## 🙏 Citation

```bibtex
@misc{paudel2026defect,
  author    = {Paudel, Dipak},
  title     = {Industrial Defect Detection using ViT + β-VAE + DDPM},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/dipakphp/industrial-defect-detection}
}
```

---

## 👤 Author

**Dipak Paudel** — MSc Data Science, University of Debrecen  
Supervisor: Dr. Robert Lakatos
