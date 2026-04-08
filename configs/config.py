"""
configs/config.py
─────────────────
Single source of truth for all hyperparameters and paths.
Edit DATASET_CHOICE here to switch datasets without touching any other file.
"""

import os


class Config:
    # ── Dataset ──────────────────────────────────────────────────────────────
    # Options: 'neu_steel' | 'mvtec' | 'dagm' | 'kaggle_steel' | 'custom'
    DATASET_CHOICE: str  = "neu_steel"

    # MVTec: which category to use (ignored for other datasets)
    # Options: 'bottle','cable','capsule','carpet','grid','hazelnut','leather',
    #          'metal_nut','pill','screw','tile','toothbrush','transistor','wood','zipper'
    MVTEC_CATEGORY: str  = "bottle"

    # Custom dataset: path to ZIP (subfolders per class)
    CUSTOM_ZIP_PATH: str = "./my_defect_dataset.zip"

    # ── Image & loader ────────────────────────────────────────────────────────
    IMG_SIZE:   int = 224
    BATCH_SIZE: int = 32

    # Train / val / test split ratios (val and test each get VAL_FRAC of total)
    VAL_FRAC:  float = 0.15
    TEST_FRAC: float = 0.15

    SEED: int = 42

    # ── Training epochs ───────────────────────────────────────────────────────
    EPOCHS_VIT:  int = 30
    EPOCHS_VAE:  int = 40
    EPOCHS_DDPM: int = 50
    EPOCHS_CLF:  int = 20

    # ── Learning rates ────────────────────────────────────────────────────────
    LR_VIT:       float = 1e-4
    LR_VIT_PHASE2: float = 5e-6   # LR after progressive unfreezing at epoch 10
    LR_VAE:       float = 2e-4
    LR_DDPM:      float = 2e-4
    LR_FUSION:    float = 5e-5
    WEIGHT_DECAY: float = 1e-4

    # Epoch at which to unfreeze all ViT blocks and rebuild the optimiser
    UNFREEZE_EPOCH: int = 10
    # Number of ViT encoder blocks frozen during Phase 1 (0-indexed from bottom)
    FREEZE_BLOCKS:  int = 8

    # ── Model architecture ────────────────────────────────────────────────────
    LATENT_DIM:    int   = 256    # β-VAE latent dimensionality
    VAE_BETA:      float = 4.0    # β-VAE KL weight (disentanglement strength)
    TIMESTEPS:     int   = 1000   # DDPM noise steps
    DDPM_CHANNELS: int   = 64     # DDPM U-Net base channel count
    DDPM_T_DIM:    int   = 256    # Sinusoidal timestep embedding dim

    # ── Paths ─────────────────────────────────────────────────────────────────
    # In Colab these point to /content/; locally they default to ./
    CHECKPOINT_DIR: str = os.getenv("CHECKPOINT_DIR", "./checkpoints")
    DATA_ROOT:      str = os.getenv("DATA_ROOT",      "./data")

    # ── Checkpoint filenames ──────────────────────────────────────────────────
    CKPT_VIT:      str = "vit_best.pt"
    CKPT_VAE:      str = "vae_best.pt"
    CKPT_DDPM:     str = "ddpm_best.pt"
    CKPT_PIPELINE: str = "full_pipeline_best.pt"
    CKPT_SYSTEM:   str = "full_system_v1.pt"

    def __post_init__(self):
        os.makedirs(self.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(self.DATA_ROOT,      exist_ok=True)

    def vit_ckpt_path(self)  -> str: return os.path.join(self.CHECKPOINT_DIR, self.CKPT_VIT)
    def vae_ckpt_path(self)  -> str: return os.path.join(self.CHECKPOINT_DIR, self.CKPT_VAE)
    def ddpm_ckpt_path(self) -> str: return os.path.join(self.CHECKPOINT_DIR, self.CKPT_DDPM)
    def pipe_ckpt_path(self) -> str: return os.path.join(self.CHECKPOINT_DIR, self.CKPT_PIPELINE)


# Convenience singleton — import this throughout the project
cfg = Config()
os.makedirs(cfg.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(cfg.DATA_ROOT,      exist_ok=True)
