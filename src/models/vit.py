"""
src/models/vit.py
─────────────────
ViT-B/16 feature extractor with progressive unfreezing.

Architecture
------------
- Backbone: vit_base_patch16_224 (timm), pretrained on ImageNet-21k
- Feature dim: 768
- Classification head: LayerNorm → Linear(768→512) → GELU → Dropout(0.25)
                       → Linear(512→256) → GELU → Dropout(0.1) → Linear(256→C)
- Forward returns: (logits [B, C], features [B, 768])

Progressive unfreezing
----------------------
Phase 1 (epochs 1–10):  Encoder blocks 0–7 frozen.
Phase 2 (epoch 10+):    All params unfrozen; optimiser + scheduler rebuilt.

⚠️  Critical: Do NOT extend an OneCycleLR scheduler when adding new param
    groups via unfreeze_all().  OneCycleLR pre-computes its step schedule at
    init, so adding groups mid-run causes LR → ∞ → NaN loss within 2 epochs.
    Use CosineAnnealingLR and rebuild the optimiser completely at epoch 10.
"""

import torch
import torch.nn as nn
import timm


class ViTFeatureExtractor(nn.Module):
    """
    Fine-tuned ViT-B/16 for industrial defect classification.

    Parameters
    ----------
    num_classes  : int   Number of output classes.
    freeze_blocks: int   Number of ViT encoder blocks to freeze initially
                         (0-indexed from bottom). Default = 8 (blocks 0–7 frozen).
    """

    def __init__(self, num_classes: int, freeze_blocks: int = 8):
        super().__init__()

        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=True,
            num_classes=0,        # Remove ImageNet head
            drop_rate=0.1,
            attn_drop_rate=0.05,
        )
        self.feature_dim: int = self.backbone.num_features  # 768

        # Freeze early encoder blocks (Phase 1)
        for name, param in self.backbone.named_parameters():
            block_num = None
            for i in range(12):
                if f"blocks.{i}." in name:
                    block_num = i
                    break
            if block_num is not None and block_num < freeze_blocks:
                param.requires_grad = False

        # Task-specific classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Tensor [B, 3, 224, 224]  ImageNet-normalised input.

        Returns
        -------
        logits   : Tensor [B, C]
        features : Tensor [B, 768]   Global [CLS] token representation.
        """
        features = self.backbone(x)           # [B, 768]
        logits   = self.classifier(features)  # [B, C]
        return logits, features

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters (call at Phase 2 boundary)."""
        for param in self.parameters():
            param.requires_grad = True
        print("✅ ViT: all parameters unfrozen")

    @property
    def n_total(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
