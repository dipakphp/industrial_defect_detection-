"""
src/transforms.py
─────────────────
Three strictly separated normalisation pipelines.

⚠️  NEVER mix these pipelines across model components:
    - train_tf / val_tf  → ViT branch  (ImageNet stats)
    - vae_tf             → VAE branch  (symmetric [-1, +1])

Mixing pipelines produces silent errors:
  • Using ImageNet norm for the VAE decoder (Tanh output = [-1,+1]) inflates
    reconstruction errors uniformly → anomaly scores become meaningless.
  • Confirmed safe input range for VAE: [-0.961, +1.000] on NEU Steel.
"""

from torchvision import transforms


# ── ViT training augmentation ─────────────────────────────────────────────────
train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    transforms.ToTensor(),
    # ImageNet statistics — mandatory for ViT-B/16 pretrained on ImageNet-21k
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── ViT validation / test (no augmentation) ───────────────────────────────────
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── β-VAE pipeline: symmetric [-1, +1] ───────────────────────────────────────
# Required because the VAE decoder ends with Tanh (output range [-1, +1]).
vae_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
