"""
src/models/vae.py
─────────────────
Beta-Variational Autoencoder for unsupervised anomaly detection.

Architecture
------------
Encoder: 5 × (Conv2d → BatchNorm2d → LeakyReLU)
         224×224×3 → 112→56→28→14→7 pixels, channels 3→32→64→128→256→512
         Flatten → Linear → μ ∈ ℝ²⁵⁶, logσ² ∈ ℝ²⁵⁶

Decoder: Linear → Reshape → 5 × (ConvTranspose2d → BatchNorm2d → ReLU) → Tanh
         ⚠️  Tanh output = [-1, +1] → input MUST be normalised to [-1, +1]

Anomaly detection
-----------------
- Train on all training images (not limited to "normal" class).
- At inference: reconstruction error |x − x̂|² gives pixel-level heatmap.
- Scalar anomaly score = mean|x − x̂|, calibrated to 0–100%.

AMP disabled
------------
PyTorch AMP (float16) + ConvTranspose2d + BatchNorm2d can produce
silent NaN weights on some GPU/CUDA combinations. The NaN is invisible
in the training loss but makes reconstruction outputs collapse to uniform
grey. All VAE computations run in float32 only.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):
    """Convolutional encoder: image → (μ, logσ²)."""

    def __init__(self, in_ch: int = 3, latent_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,   32, 4, 2, 1), nn.BatchNorm2d(32),  nn.LeakyReLU(0.2),
            nn.Conv2d(32,      64, 4, 2, 1), nn.BatchNorm2d(64),  nn.LeakyReLU(0.2),
            nn.Conv2d(64,     128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128,    256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256,    512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
            nn.Flatten(),                     # → [B, 512*7*7]
        )
        self.fc_mu  = nn.Linear(512 * 7 * 7, latent_dim)
        self.fc_var = nn.Linear(512 * 7 * 7, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.fc_mu(h), self.fc_var(h)


class ConvDecoder(nn.Module):
    """Convolutional decoder: z → reconstructed image in [-1, +1]."""

    def __init__(self, latent_dim: int = 256, out_ch: int = 3):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 512 * 7 * 7)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128,  64, 4, 2, 1), nn.BatchNorm2d(64),  nn.ReLU(),
            nn.ConvTranspose2d( 64,  32, 4, 2, 1), nn.BatchNorm2d(32),  nn.ReLU(),
            nn.ConvTranspose2d( 32, out_ch, 4, 2, 1),
            nn.Tanh(),  # output range [-1, +1]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(self.fc(z).view(-1, 512, 7, 7))


class BetaVAE(nn.Module):
    """
    β-VAE for unsupervised anomaly detection.

    Parameters
    ----------
    latent_dim : int   Dimensionality of the latent space.
    beta       : float KL weight (> 1 enforces disentanglement).
    """

    def __init__(self, latent_dim: int = 256, beta: float = 4.0):
        super().__init__()
        self.encoder    = ConvEncoder(latent_dim=latent_dim)
        self.decoder    = ConvDecoder(latent_dim=latent_dim)
        self.beta       = beta
        self.latent_dim = latent_dim

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick. Deterministic at inference time."""
        if self.training:
            return mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        return mu  # deterministic → stable anomaly scores

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : Tensor [B, 3, 224, 224]  Normalised to [-1, +1].

        Returns
        -------
        recon  : Tensor [B, 3, 224, 224]
        mu     : Tensor [B, latent_dim]
        log_var: Tensor [B, latent_dim]
        z      : Tensor [B, latent_dim]
        """
        mu, log_var = self.encoder(x)
        z           = self.reparameterize(mu, log_var)
        recon       = self.decoder(z)
        return recon, mu, log_var, z

    def elbo_loss(
        self,
        x: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ):
        """
        ELBO = reconstruction loss + β × KL divergence.

        Returns
        -------
        elbo   : scalar
        recon_l: scalar  (MSE reconstruction term)
        kl_l   : scalar  (KL divergence term)
        """
        recon_l = F.mse_loss(recon, x, reduction="sum") / x.size(0)
        kl_l    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        return recon_l + self.beta * kl_l, recon_l, kl_l

    def anomaly_score(self, x: torch.Tensor) -> torch.Tensor:
        """Per-image anomaly score = mean pixel-wise reconstruction error."""
        self.eval()
        with torch.no_grad():
            recon, _, _, _ = self.forward(x)
            return F.mse_loss(recon, x, reduction="none").mean(dim=[1, 2, 3])
