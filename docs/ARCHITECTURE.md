# Architecture Notes

## Model Pipeline

```
Input Image [B, 3, 224, 224]
        │
        ├─────────────────────────────────────────────┐
        │  ViT Branch (ImageNet norm)                 │  VAE Branch ([-1,+1] norm)
        ▼                                             ▼
  ViT-B/16 Backbone                         ConvEncoder (5-layer)
  196 patch tokens (16×16)                  224→112→56→28→14→7 px
  12 Transformer Encoder Blocks             3→32→64→128→256→512 ch
  LayerNorm + FFN + Residual               Flatten → 25,088-dim
        │                                             │
  [CLS] token → 768-dim features            μ ∈ ℝ²⁵⁶, logσ² ∈ ℝ²⁵⁶
  Classification head → logits                        │
                                            Reparameterise z = μ + σε
                                                       │
                                            ConvDecoder (5-layer)
                                            7→14→28→56→112→224 px
                                            512→256→128→64→32→3 ch
                                            Tanh → [-1, +1]
                                                       │
                                            diff = |x - x̂|
                                            score = mean(diff)
        │                                             │
        └──────────────────────┬──────────────────────┘
                               ▼
                    Fusion Classifier
                    LayerNorm + Linear projections
                    Concat [768 + 256 + 1] = 769-dim
                    MLP (769→512→256→C)
                               │
                               ▼
                        Class Logits [B, C]
```

## Known Issues & Engineering Fixes

### Fix 1: NaN in ViT Training (OneCycleLR + Progressive Unfreezing)

**Problem:**  
`OneCycleLR` pre-computes its complete step schedule at `__init__` time based on the number of steps and the initial set of parameter groups. When `unfreeze_all()` adds new parameter groups at epoch 10, the scheduler has no record of them and assigns divergent (often infinite) learning rates to the new groups.

**Symptoms:**  
- Loss → NaN within 2 epochs of the phase boundary
- No PyTorch warning or error

**Fix applied (in `src/utils/training.py → train_vit`):**  
1. Switch from `OneCycleLR` to `CosineAnnealingLR`
2. At epoch `UNFREEZE_EPOCH`: call `model.unfreeze_all()`, then **create brand-new `AdamW` and `CosineAnnealingLR` instances** from scratch — do not extend the existing objects.

---

### Fix 2: Silent NaN in β-VAE (AMP + ConvTranspose2d + BatchNorm)

**Problem:**  
PyTorch Automatic Mixed Precision (float16) + `ConvTranspose2d` + `BatchNorm2d` in the VAE decoder silently corrupt model weights with NaN values on some GPU/CUDA combinations. The training ELBO metric continues to display plausible numeric values — the NaN is invisible in the loss curve.

**Symptoms:**  
- Reconstruction outputs collapse to near-uniform grey (~0.0 everywhere)
- `torch.isfinite(model_weights).all()` returns `False`
- ELBO loss shows no obvious anomaly

**Fix applied (in `src/utils/training.py → train_vae`):**  
Completely disable AMP for VAE training. All computations run in float32. This is ~15% slower but produces fully stable and reproducible results.

---

## Normalisation Pipelines

Three separate pipelines must never be mixed:

| Pipeline | Used for | Mean | Std | Range |
|----------|----------|------|-----|-------|
| `train_tf` | ViT training | [0.485, 0.456, 0.406] | [0.229, 0.224, 0.225] | ~[-2.1, +2.6] |
| `val_tf`   | ViT val/test | same as above | same | same |
| `vae_tf`   | β-VAE input | [0.5, 0.5, 0.5] | [0.5, 0.5, 0.5] | [-1, +1] |

**Why this matters for the VAE:**  
The decoder ends with `Tanh`, whose output range is exactly `[-1, +1]`. If the input is ImageNet-normalised (range `~[-2.5, +2.5]`), the reconstruction error `|x − x̂|` is systematically inflated for all pixels — anomaly scores become meaningless as triage measures.

## Progressive Unfreezing Schedule

```
Epoch  1–10: Only ViT blocks 8–11 + classifier head are trainable
             LR = 1e-4

Epoch    10: call model.unfreeze_all()
             Rebuild optimizer (new AdamW instance, all params, LR=5e-6)
             Rebuild scheduler (new CosineAnnealingLR instance)

Epoch 11–30: All 86M ViT params trainable
             LR decays from 5e-6 → 1e-8 (cosine)
```

## Checkpoint Files

| File | Size | Contents |
|------|------|----------|
| `vit_best.pt` | ~334 MB | ViT state dict (best val epoch) |
| `vae_best.pt` | ~100 MB | β-VAE state dict (best ELBO) |
| `ddpm_best.pt` | ~52 MB | DDPM U-Net state dict |
| `full_pipeline_best.pt` | ~450 MB | ViT + VAE + Fusion + class names |
| `full_system_v1.pt` | ~500 MB | All four models + metrics |
