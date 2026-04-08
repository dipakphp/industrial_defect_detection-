"""
tests/test_models.py
────────────────────
Unit tests for model forward passes.

Run with:  pytest tests/test_models.py -v
"""

import torch
import pytest

from configs.config import Config
from src.models.vit    import ViTFeatureExtractor
from src.models.vae    import BetaVAE
from src.models.ddpm   import UNet, DDPMScheduler
from src.models.fusion import FusionClassifier, FullPipeline

cfg = Config()
B   = 2   # small batch for speed


@pytest.fixture(scope="module")
def device():
    return torch.device("cpu")  # tests always run on CPU


# ── ViT ───────────────────────────────────────────────────────────────────────

class TestViT:
    def test_output_shapes(self, device):
        model = ViTFeatureExtractor(num_classes=6).to(device)
        x     = torch.randn(B, 3, 224, 224)
        logits, features = model(x)
        assert logits.shape   == (B, 6),   f"Expected (B,6), got {logits.shape}"
        assert features.shape == (B, 768), f"Expected (B,768), got {features.shape}"

    def test_outputs_finite(self, device):
        model = ViTFeatureExtractor(num_classes=6).to(device)
        x     = torch.randn(B, 3, 224, 224)
        logits, features = model(x)
        assert torch.isfinite(logits).all(),   "ViT logits contain NaN/Inf"
        assert torch.isfinite(features).all(), "ViT features contain NaN/Inf"

    def test_unfreeze_all(self, device):
        model = ViTFeatureExtractor(num_classes=6, freeze_blocks=8).to(device)
        n_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model.unfreeze_all()
        n_after  = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_after > n_before, "unfreeze_all() should increase trainable params"

    def test_param_counts(self, device):
        model = ViTFeatureExtractor(num_classes=6).to(device)
        assert model.n_total > 86_000_000, "ViT-B/16 should have ~86M total params"


# ── β-VAE ─────────────────────────────────────────────────────────────────────

class TestVAE:
    def test_output_shapes(self, device):
        model = BetaVAE(latent_dim=256, beta=4.0).to(device)
        x     = torch.randn(B, 3, 224, 224)
        recon, mu, log_var, z = model(x)
        assert recon.shape   == (B, 3, 224, 224)
        assert mu.shape      == (B, 256)
        assert log_var.shape == (B, 256)
        assert z.shape       == (B, 256)

    def test_outputs_finite(self, device):
        model = BetaVAE().to(device)
        x     = torch.randn(B, 3, 224, 224).clamp(-1, 1)
        recon, mu, log_var, z = model(x)
        for name, t in [("recon", recon), ("mu", mu), ("log_var", log_var), ("z", z)]:
            assert torch.isfinite(t).all(), f"VAE {name} contains NaN/Inf"

    def test_elbo_loss_positive(self, device):
        model = BetaVAE().to(device)
        x     = torch.randn(B, 3, 224, 224).clamp(-1, 1)
        recon, mu, log_var, _ = model(x)
        elbo, recon_l, kl_l   = model.elbo_loss(x, recon, mu, log_var)
        assert torch.isfinite(elbo),   "ELBO is NaN/Inf"
        assert elbo.item() > 0,        "ELBO should be positive"

    def test_deterministic_at_inference(self, device):
        model = BetaVAE().to(device)
        model.eval()
        x     = torch.randn(1, 3, 224, 224).clamp(-1, 1)
        with torch.no_grad():
            _, mu1, _, z1 = model(x)
            _, mu2, _, z2 = model(x)
        assert torch.allclose(z1, z2), "Inference should be deterministic (z = mu)"

    def test_anomaly_score_shape(self, device):
        model = BetaVAE().to(device)
        x     = torch.randn(B, 3, 224, 224).clamp(-1, 1)
        score = model.anomaly_score(x)
        assert score.shape == (B,), f"Expected ({B},), got {score.shape}"
        assert (score >= 0).all(), "Anomaly scores should be non-negative"


# ── DDPM ──────────────────────────────────────────────────────────────────────

class TestDDPM:
    def test_unet_output_shape(self, device):
        model = UNet(ch=3, base=32, t_dim=64).to(device)  # small for speed
        x     = torch.randn(B, 3, 56, 56)
        t     = torch.randint(0, 1000, (B,)).long()
        out   = model(x, t)
        assert out.shape == x.shape, f"U-Net output shape mismatch: {out.shape}"

    def test_scheduler_q_sample(self, device):
        sched = DDPMScheduler(T=1000)
        x0    = torch.randn(B, 3, 56, 56)
        t     = torch.randint(0, 1000, (B,)).long()
        noisy, noise = sched.q_sample(x0, t)
        assert noisy.shape == x0.shape
        assert noise.shape == x0.shape
        assert torch.isfinite(noisy).all()

    def test_scheduler_p_sample(self, device):
        sched = DDPMScheduler(T=100)
        model = UNet(ch=3, base=16, t_dim=32).to(device)
        xt    = torch.randn(B, 3, 56, 56)
        out   = sched.p_sample(model, xt, t_s=50)
        assert out.shape == xt.shape


# ── Fusion Classifier ─────────────────────────────────────────────────────────

class TestFusion:
    def test_output_shape(self, device):
        model   = FusionClassifier(vit_dim=768, vae_dim=256, num_classes=6).to(device)
        vit_f   = torch.randn(B, 768)
        vae_mu  = torch.randn(B, 256)
        score   = torch.rand(B)
        logits  = model(vit_f, vae_mu, score)
        assert logits.shape == (B, 6), f"Expected (B,6), got {logits.shape}"

    def test_full_pipeline_forward(self, device):
        vit    = ViTFeatureExtractor(num_classes=6).to(device)
        vae    = BetaVAE(latent_dim=256).to(device)
        fusion = FusionClassifier(num_classes=6).to(device)
        pipe   = FullPipeline(vit, vae, fusion).to(device)
        x      = torch.randn(B, 3, 224, 224)
        logits = pipe(x)
        assert logits.shape == (B, 6)
        assert torch.isfinite(logits).all()
