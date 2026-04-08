"""
src/models/fusion.py
────────────────────
Fusion Classifier and FullPipeline — combine ViT, β-VAE, and anomaly score.

FusionClassifier
----------------
Input streams:
    vit_f  : [B, 768]  ViT [CLS] features
    vae_mu : [B, 256]  β-VAE posterior mean
    score  : [B]       Scalar anomaly score

Projection → concatenation → MLP → class logits.

FullPipeline
------------
End-to-end module for inference:
    x_vit (ImageNet-normalised) → ViT → features
                                → VAE (re-normalised to [-1,+1]) → μ + score
                                → Fusion → logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vit import ViTFeatureExtractor
from .vae import BetaVAE


class FusionClassifier(nn.Module):
    """
    Three-stream fusion MLP.

    Parameters
    ----------
    vit_dim    : int  ViT feature dimension (768 for ViT-B/16).
    vae_dim    : int  VAE latent dimension.
    num_classes: int  Number of output classes.
    """

    def __init__(
        self,
        vit_dim: int     = 768,
        vae_dim: int     = 256,
        num_classes: int = 6,
    ):
        super().__init__()
        # Independent projection for each stream
        self.vit_proj = nn.Sequential(
            nn.LayerNorm(vit_dim),
            nn.Linear(vit_dim, 512),
            nn.GELU(),
        )
        self.vae_proj = nn.Sequential(
            nn.LayerNorm(vae_dim),
            nn.Linear(vae_dim, 256),
            nn.GELU(),
        )
        # Fusion MLP — input = 512 + 256 + 1 = 769
        fused_dim = 512 + 256 + 1
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512,       256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        vit_f:  torch.Tensor,
        vae_mu: torch.Tensor,
        score:  torch.Tensor,
    ) -> torch.Tensor:
        v = self.vit_proj(vit_f)
        a = self.vae_proj(vae_mu)
        s = score.unsqueeze(1) if score.dim() == 1 else score  # [B, 1]
        return self.fusion(torch.cat([v, a, s], dim=1))


class FullPipeline(nn.Module):
    """
    End-to-end inference pipeline.

    Input : x_vit — ImageNet-normalised image [B, 3, 224, 224]
    Output: class logits [B, C]

    Internally converts x_vit to the VAE's [-1,+1] normalisation.
    VAE gradients are detached — the VAE is always frozen during inference.
    """

    # ImageNet stats (used to undo normalisation before VAE re-normalisation)
    _INET_MEAN = torch.tensor([0.485, 0.456, 0.406])
    _INET_STD  = torch.tensor([0.229, 0.224, 0.225])

    def __init__(
        self,
        vit:    ViTFeatureExtractor,
        vae:    BetaVAE,
        fusion: FusionClassifier,
    ):
        super().__init__()
        self.vit    = vit
        self.vae    = vae
        self.fusion = fusion

    def _imagenet_to_vae(self, x: torch.Tensor) -> torch.Tensor:
        """Convert ImageNet-normalised tensor to VAE's [-1, +1] range."""
        device = x.device
        mean = self._INET_MEAN.to(device).view(1, 3, 1, 1)
        std  = self._INET_STD.to(device).view(1, 3, 1, 1)
        x_01 = x * std + mean          # Undo ImageNet normalisation → [0, 1]
        return x_01 * 2.0 - 1.0        # Scale to [-1, +1]

    def forward(self, x_vit: torch.Tensor) -> torch.Tensor:
        # ── ViT branch ────────────────────────────────────────────────────
        _, vit_features = self.vit(x_vit)          # [B, 768]

        # ── VAE branch ────────────────────────────────────────────────────
        x_vae = self._imagenet_to_vae(x_vit)
        with torch.no_grad():
            recon, mu, _, _ = self.vae(x_vae)

        score = F.mse_loss(recon, x_vae, reduction="none").mean(dim=[1, 2, 3])

        # ── Fusion ────────────────────────────────────────────────────────
        return self.fusion(vit_features, mu.detach(), score.detach())
