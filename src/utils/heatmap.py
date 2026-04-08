"""
src/utils/heatmap.py
────────────────────
Anomaly heatmap generation from β-VAE reconstruction error.

Pipeline (8 stages)
--------------------
1. Collapse 3-channel diff to 2D [H, W] via mean
2. Sanitise NaN / Inf → 0
3. Log-scale amplification: log(1 + diff × 50)
4. Percentile contrast stretch (1st–99th percentile → [0, 1])
5. Gamma correction (γ = 0.6, brightens mid-tones)
6. JET colourmap: blue = low deviation, red = high deviation
7. Alpha-blend with original (35% original + 65% heatmap)
8. Colourbar legend strip at bottom

Anomaly score levels
--------------------
< 8%    NORMAL   — no significant anomaly
8–25%   LOW      — anomaly detected
25–55%  MODERATE — possible defect
> 55%   HIGH     — defect confirmed
"""

from typing import Tuple

import cv2
import numpy as np
import torch
from PIL import Image


def compute_anomaly_score(
    diff_tensor: torch.Tensor,
    scale: float = 0.3,
) -> Tuple[float, float, str]:
    """
    Convert a reconstruction-error tensor to a scalar severity score.

    Parameters
    ----------
    diff_tensor : Tensor  Absolute pixel-wise difference |x − x̂|.
    scale       : float   Denominator for percentage mapping (0.3 = 100% at raw=0.3).

    Returns
    -------
    raw   : float  Mean absolute reconstruction error.
    pct   : float  Percentage score in [0, 100].
    level : str    Human-readable severity label.
    """
    d = diff_tensor.numpy().astype(np.float64)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    raw = float(np.mean(np.abs(d)))
    pct = min(raw / scale * 100.0, 100.0)

    if pct < 8:
        level = "NORMAL — no significant anomaly"
    elif pct < 25:
        level = "LOW anomaly detected"
    elif pct < 55:
        level = "MODERATE anomaly — possible defect"
    else:
        level = "HIGH anomaly — defect confirmed"

    return raw, pct, level


def tensor_to_vivid_heatmap(
    diff_tensor:  torch.Tensor,
    original_pil: Image.Image,
    alpha_heatmap: float = 0.65,
    gamma:         float = 0.6,
) -> Image.Image:
    """
    Convert a [C, H, W] or [H, W] difference tensor to a vivid heatmap.

    Parameters
    ----------
    diff_tensor    : Tensor  Reconstruction error (absolute values expected).
    original_pil   : PIL     Original input image for context blending.
    alpha_heatmap  : float   Heatmap blend weight (default 0.65).
    gamma          : float   Gamma correction exponent (< 1 brightens mid-tones).

    Returns
    -------
    PIL.Image  Final blended heatmap with colourbar legend.
    """
    # Stage 1: collapse to [H, W]
    if diff_tensor.dim() == 3:
        diff_np = diff_tensor.mean(0).numpy().astype(np.float32)
    else:
        diff_np = diff_tensor.numpy().astype(np.float32)

    # Stage 2: sanitise
    diff_np = np.nan_to_num(np.abs(diff_np), nan=0.0, posinf=1.0, neginf=0.0)

    # Stage 3: log amplification
    amplified = np.log1p(diff_np * 50.0)

    # Stage 4: percentile contrast stretch
    p1  = float(np.percentile(amplified, 1))
    p99 = float(np.percentile(amplified, 99))
    if p99 - p1 < 1e-6:
        p1, p99 = 0.0, max(float(amplified.max()) * 0.5, 1e-6)
    stretched = np.clip((amplified - p1) / (p99 - p1), 0.0, 1.0)

    # Stage 5: gamma correction
    gamma_corrected = np.power(stretched, gamma)

    # Stage 6: JET colourmap
    heatmap_jet = cv2.applyColorMap(
        (gamma_corrected * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_rgb = cv2.cvtColor(heatmap_jet, cv2.COLOR_BGR2RGB)

    # Stage 7: blend with original
    orig_np = np.array(
        original_pil.resize((diff_np.shape[1], diff_np.shape[0]))
    ).astype(np.float32)
    blended = np.clip(
        (1 - alpha_heatmap) * orig_np + alpha_heatmap * heatmap_rgb.astype(np.float32),
        0, 255,
    ).astype(np.uint8)

    # Stage 8: colourbar legend
    legend_h = 20
    legend   = np.zeros((legend_h, blended.shape[1], 3), dtype=np.uint8)
    for xi in range(blended.shape[1]):
        val   = xi / blended.shape[1]
        color = cv2.applyColorMap(
            np.array([[[int(val * 255)]]], dtype=np.uint8), cv2.COLORMAP_JET
        )[0, 0]
        legend[:, xi] = cv2.cvtColor(
            np.array([[color]], dtype=np.uint8), cv2.COLOR_BGR2RGB
        )[0, 0]

    cv2.putText(legend, "LOW",  (2, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.putText(legend, "HIGH", (blended.shape[1] - 35, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    final = np.vstack([blended, legend])
    return Image.fromarray(final)
