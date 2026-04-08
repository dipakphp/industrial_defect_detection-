"""
src/models/ddpm.py
──────────────────
Denoising Diffusion Probabilistic Model (DDPM) for synthetic defect generation.

Components
----------
SinusoidalPosEmb : Timestep embedding
ResBlock         : Residual block with timestep injection
UNet             : Encoder-bottleneck-decoder denoising network
DDPMScheduler    : Linear noise schedule + forward/reverse sampling

Training objective
------------------
L = E[‖ε − ε_θ(x_t, t)‖²]
where x_t = √ā_t · x_0 + √(1−ā_t) · ε,  ε ~ N(0,I).

Usage
-----
Forward diffusion  : scheduler.q_sample(x0, t)
Reverse (sampling) : scheduler.sample(model, shape, device)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


# ── Sinusoidal Timestep Embedding ─────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        d = self.dim // 2
        e = math.log(10_000) / (d - 1)
        e = torch.exp(torch.arange(d, device=t.device) * -e)
        e = t[:, None].float() * e[None, :]
        return torch.cat([e.sin(), e.cos()], dim=-1)


# ── Residual Block with Timestep Injection ────────────────────────────────────

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, groups: int = 8):
        super().__init__()
        self.t_mlp = nn.Sequential(nn.SiLU(), nn.Linear(t_dim, out_ch))
        self.b1 = nn.Sequential(
            nn.GroupNorm(min(groups, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
        )
        self.b2 = nn.Sequential(
            nn.GroupNorm(min(groups, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
        )
        self.res = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.b1(x) + self.t_mlp(t)[:, :, None, None]
        return self.b2(h) + self.res(x)


# ── U-Net Denoiser ─────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Lightweight U-Net denoiser operating at 56×56 internal resolution.

    Parameters
    ----------
    ch   : int  Input/output channels (3 for RGB).
    base : int  Base channel multiplier.
    t_dim: int  Timestep embedding dimension.
    """

    def __init__(self, ch: int = 3, base: int = 64, t_dim: int = 256):
        super().__init__()
        c = base

        # Timestep MLP
        self.t_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            nn.Linear(t_dim, t_dim * 2),
            nn.GELU(),
            nn.Linear(t_dim * 2, t_dim),
        )

        # Encoder
        self.e1 = ResBlock(ch,    c,    t_dim)   # 56×56, c
        self.e2 = ResBlock(c,     c*2,  t_dim)   # 28×28, c*2
        self.e3 = ResBlock(c*2,   c*4,  t_dim)   # 14×14, c*4
        self.d1 = nn.MaxPool2d(2)
        self.d2 = nn.MaxPool2d(2)

        # Bottleneck
        self.m1 = ResBlock(c*4, c*8, t_dim)
        self.m2 = ResBlock(c*8, c*8, t_dim)

        # Decoder — skip connections double the channel count
        self.u2   = nn.ConvTranspose2d(c*8, c*4, 2, 2)
        self.dec3 = ResBlock(c*4 + c*2, c*4, t_dim)  # u2(c*4) + e2(c*2)

        self.u1   = nn.ConvTranspose2d(c*4, c*2, 2, 2)
        self.dec2 = ResBlock(c*2 + c,   c*2, t_dim)  # u1(c*2) + e1(c)

        self.dec1 = ResBlock(c*2 + c,   c,   t_dim)

        self.out = nn.Sequential(
            nn.GroupNorm(8, c),
            nn.SiLU(),
            nn.Conv2d(c, ch, 1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.t_mlp(t)
        xs = F.interpolate(x, 56, mode="bilinear", align_corners=False)

        e1 = self.e1(xs, te)                       # [B, c,   56, 56]
        e2 = self.e2(self.d1(e1), te)              # [B, c*2, 28, 28]
        e3 = self.e3(self.d2(e2), te)              # [B, c*4, 14, 14]

        m  = self.m2(self.m1(e3, te), te)          # [B, c*8, 14, 14]

        d3 = self.dec3(torch.cat([self.u2(m), e2], 1), te)
        d2 = self.dec2(torch.cat([self.u1(d3), e1], 1), te)
        d1 = self.dec1(torch.cat([d2, e1], 1), te)
        out = self.out(d1)
        return F.interpolate(out, size=x.shape[2], mode="bilinear", align_corners=False)


# ── DDPM Noise Schedule + Sampling ────────────────────────────────────────────

class DDPMScheduler:
    """
    Linear noise schedule from β₁=1e-4 to β_T=0.02 over T steps.

    Methods
    -------
    q_sample(x0, t)          : Sample noisy image at timestep t (forward process).
    p_sample(model, xt, t_s) : Single reverse denoising step.
    sample(model, shape, ...)  : Full reverse-process image generation.
    """

    def __init__(self, T: int = 1000):
        self.T = T
        betas       = torch.linspace(1e-4, 0.02, T)
        self.betas  = betas
        self.alpha  = 1.0 - betas
        self.abar   = torch.cumprod(self.alpha, dim=0)  # ā_t

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor = None,
    ):
        """
        Forward diffusion: x_t = √ā_t · x_0 + √(1−ā_t) · ε

        Returns
        -------
        noisy  : Tensor  x_t
        noise  : Tensor  the actual noise added (target for the U-Net)
        """
        if noise is None:
            noise = torch.randn_like(x0)
        ab = self.abar.to(x0.device)[t][:, None, None, None]
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise, noise

    @torch.no_grad()
    def p_sample(
        self,
        model: nn.Module,
        xt: torch.Tensor,
        t_s: int,
    ) -> torch.Tensor:
        """Single reverse denoising step from x_t → x_{t-1}."""
        model.eval()
        tb    = torch.full((xt.size(0),), t_s, device=xt.device, dtype=torch.long)
        b     = self.betas[t_s].to(xt.device)
        a     = self.alpha[t_s].to(xt.device)
        ab    = self.abar[t_s].to(xt.device)
        eps   = model(xt, tb)
        mean  = (1 / torch.sqrt(a)) * (xt - (b / torch.sqrt(1.0 - ab)) * eps)
        noise = torch.sqrt(b) * torch.randn_like(xt) if t_s > 0 else 0.0
        return mean + noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple,
        device: torch.device,
        steps: int = 200,
    ) -> torch.Tensor:
        """
        Generate synthetic images via the reverse diffusion process.

        Parameters
        ----------
        steps : int  Number of denoising steps (< T for faster sampling).
        """
        x    = torch.randn(shape, device=device)
        step = self.T // steps
        for t in tqdm(
            reversed(range(0, self.T, step)),
            desc="DDPM sampling",
            total=steps,
            leave=False,
        ):
            x = self.p_sample(model, x, t)
        return x.clamp(-1.0, 1.0)
