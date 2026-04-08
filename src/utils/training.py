"""
src/utils/training.py
─────────────────────
Four independent training functions, one per model stage:

    train_vit    : ViT-B/16 with progressive unfreezing + NaN guard
    train_vae    : β-VAE in float32 (AMP disabled — prevents silent NaN)
    train_ddpm   : DDPM U-Net noise predictor
    train_fusion : Fusion Classifier over frozen VAE + partially-unfrozen ViT
"""

import os
from collections import Counter
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_labels(subset) -> List[int]:
    """Extract all labels from a Subset or Dataset that exposes .samples."""
    try:
        return [subset.dataset.samples[i][1] for i in subset.indices]
    except AttributeError:
        return []


def _weighted_criterion(labels: List[int], num_classes: int, device: torch.device):
    """Class-weighted cross-entropy loss."""
    if not labels:
        return nn.CrossEntropyLoss(label_smoothing=0.1)
    cnt = Counter(labels)
    w   = torch.tensor(
        [1.0 / max(cnt.get(i, 1), 1) for i in range(num_classes)],
        dtype=torch.float, device=device,
    )
    w = w / w.sum() * num_classes
    return nn.CrossEntropyLoss(weight=w, label_smoothing=0.1)


# ── 1. ViT Training ───────────────────────────────────────────────────────────

def train_vit(
    model,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg,
    device:       torch.device,
    num_classes:  int,
) -> Dict:
    """
    Train ViT-B/16 with two-phase progressive unfreezing.

    Phase 1 (epochs 1–cfg.UNFREEZE_EPOCH):
        Only encoder blocks 8–11 + classifier head are trainable.

    Phase 2 (epoch cfg.UNFREEZE_EPOCH onward):
        All 86M params unfrozen; optimiser + scheduler rebuilt at reduced LR.

    ⚠️  Critical — OneCycleLR is NOT used here.
        OneCycleLR pre-computes its full step schedule at __init__ time.
        Adding new parameter groups via unfreeze_all() mid-training causes
        LR → ∞ → NaN loss within 2 epochs.
        Fix: use CosineAnnealingLR and rebuild the optimiser completely.
    """
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LR_VIT, weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS_VIT, eta_min=1e-7,
    )
    criterion = _weighted_criterion(_get_labels(train_loader.dataset), num_classes, device)

    history   = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc  = 0.0
    best_ckpt = cfg.vit_ckpt_path()
    scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(cfg.EPOCHS_VIT):

        # ── Progressive unfreezing at phase boundary ──────────────────────
        if epoch == cfg.UNFREEZE_EPOCH:
            model.unfreeze_all()
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=cfg.LR_VIT_PHASE2,
                weight_decay=cfg.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg.EPOCHS_VIT - epoch, eta_min=1e-8,
            )
            print(f"  Epoch {epoch+1}: full unfreeze, optimiser rebuilt, "
                  f"LR={cfg.LR_VIT_PHASE2:.2e}")

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        t_loss, t_correct, n = 0.0, 0, 0
        nan_epoch = False
        pbar = tqdm(train_loader, desc=f"ViT {epoch+1}/{cfg.EPOCHS_VIT}", leave=False)

        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits, _ = model(imgs)
                loss      = criterion(logits, labels)

            if not torch.isfinite(loss):
                print(f"  ⚠️  Non-finite loss at epoch {epoch+1} — skipping batch")
                nan_epoch = True
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            t_loss    += loss.item() * imgs.size(0)
            t_correct += (logits.argmax(1) == labels).sum().item()
            n         += imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{t_correct/max(n,1):.3f}")

        # NaN-only epoch → rollback
        if nan_epoch and n == 0:
            print(f"  ROLLBACK: epoch {epoch+1} all-NaN — restoring best checkpoint")
            if os.path.exists(best_ckpt):
                model.load_state_dict(torch.load(best_ckpt, map_location=device))
            for pg in optimizer.param_groups:
                pg["lr"] *= 0.1
            history["train_loss"].append(float("nan"))
            history["val_loss"].append(float("nan"))
            history["train_acc"].append(0.0)
            history["val_acc"].append(0.0)
            continue

        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        v_loss, v_correct, nv = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits, _    = model(imgs)
                vl = criterion(logits, labels)
                if torch.isfinite(vl):
                    v_loss    += vl.item() * imgs.size(0)
                v_correct += (logits.argmax(1) == labels).sum().item()
                nv        += imgs.size(0)

        ta = t_correct / max(n,  1)
        va = v_correct / max(nv, 1)
        history["train_loss"].append(t_loss / max(n,  1))
        history["val_loss"].append(  v_loss / max(nv, 1))
        history["train_acc"].append(ta)
        history["val_acc"].append(va)

        print(f"Epoch {epoch+1:3d}/{cfg.EPOCHS_VIT} | "
              f"Train Loss:{t_loss/max(n,1):.4f} Acc:{ta:.4f} | "
              f"Val Loss:{v_loss/max(nv,1):.4f} Acc:{va:.4f}")

        if va > best_acc:
            best_acc = va
            torch.save(model.state_dict(), best_ckpt)
            print(f"  ✅ Best ViT saved → val_acc:{va:.4f}")

    # Always restore best checkpoint
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print(f"\nViT training complete — best val_acc: {best_acc:.4f}")
    return history


# ── 2. β-VAE Training ─────────────────────────────────────────────────────────

def train_vae(
    model,
    loader:  DataLoader,
    cfg,
    device:  torch.device,
) -> Dict:
    """
    Train β-VAE entirely in float32.

    ⚠️  AMP (float16) is explicitly disabled here.
        PyTorch AMP + ConvTranspose2d + BatchNorm2d can silently corrupt
        weights on some GPU/CUDA combinations. The NaN appears in model
        weights but is invisible in the scalar ELBO — only visible by
        inspecting the reconstruction outputs (near-uniform grey).
        Training in float32 is ~15% slower but fully stable.
    """
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.LR_VAE, betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS_VAE, eta_min=1e-6,
    )
    history   = {"elbo": [], "recon": [], "kl": []}
    best_loss = float("inf")
    best_ckpt = cfg.vae_ckpt_path()
    nan_count = 0

    for epoch in range(cfg.EPOCHS_VAE):
        model.train()
        te, tr, tk, n = 0.0, 0.0, 0.0, 0

        for imgs, _ in tqdm(loader, desc=f"VAE {epoch+1}/{cfg.EPOCHS_VAE}", leave=False):
            imgs = imgs.to(device)   # float32, [-1, +1]

            optimizer.zero_grad()
            # NO autocast — float32 only to prevent silent NaN in VAE decoder
            recon, mu, log_var, _ = model(imgs)
            elbo, rl, kl = model.elbo_loss(imgs, recon, mu, log_var)

            if not torch.isfinite(elbo):
                nan_count += 1
                if nan_count > 5:
                    print("  Too many NaN losses — reducing LR ×0.1")
                    for pg in optimizer.param_groups:
                        pg["lr"] *= 0.1
                    nan_count = 0
                continue

            elbo.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            te += elbo.item() * imgs.size(0)
            tr += rl.item()   * imgs.size(0)
            tk += kl.item()   * imgs.size(0)
            n  += imgs.size(0)

        scheduler.step()

        if n == 0:
            print(f"  Epoch {epoch+1}: all NaN — skipping")
            continue

        avg_elbo = te / n
        history["elbo"].append(avg_elbo)
        history["recon"].append(tr / n)
        history["kl"].append(tk / n)
        print(f"Epoch {epoch+1:3d}/{cfg.EPOCHS_VAE} | "
              f"ELBO:{avg_elbo:.2f}  Recon:{tr/n:.2f}  KL:{tk/n:.4f}")

        if avg_elbo < best_loss:
            best_loss = avg_elbo
            torch.save(model.state_dict(), best_ckpt)

    # Restore + verify
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, 3, cfg.IMG_SIZE, cfg.IMG_SIZE).clamp(-1, 1).to(device)
        recon_t, mu_t, _, _ = model(dummy)
        ok = torch.isfinite(recon_t).all() and torch.isfinite(mu_t).all()

    print(f"\nVAE weight check: {'✅ PASS' if ok else '❌ FAIL — NaN detected'}")
    print(f"VAE best ELBO: {best_loss:.2f}")
    return history


# ── 3. DDPM Training ──────────────────────────────────────────────────────────

def train_ddpm(
    model,
    scheduler,
    loader:  DataLoader,
    cfg,
    device:  torch.device,
) -> List[float]:
    """Train DDPM U-Net noise predictor."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.LR_DDPM)
    lr_sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS_DDPM,
    )
    criterion = nn.MSELoss()
    history   = []
    scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(cfg.EPOCHS_DDPM):
        model.train()
        total_loss, n = 0.0, 0

        for imgs, *_ in tqdm(loader, desc=f"DDPM {epoch+1}/{cfg.EPOCHS_DDPM}", leave=False):
            imgs = imgs.to(device)
            t    = torch.randint(0, scheduler.T, (imgs.size(0),), device=device).long()
            noisy, noise = scheduler.q_sample(imgs, t)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                pred_noise = model(noisy, t)
                loss       = criterion(pred_noise, noise)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)
            n          += imgs.size(0)

        lr_sched.step()
        avg = total_loss / n
        history.append(avg)
        print(f"Epoch {epoch+1:3d}/{cfg.EPOCHS_DDPM} | DDPM Loss: {avg:.6f}")

    torch.save(model.state_dict(), cfg.ddpm_ckpt_path())
    print(f"\n✅ DDPM training complete — final loss: {history[-1]:.6f}")
    return history


# ── 4. Fusion Classifier Training ─────────────────────────────────────────────

def train_fusion(
    pipeline,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    cfg,
    device:       torch.device,
    vit_model,
    vae_model,
    fusion_clf,
    class_names:  List[str],
    dataset_name: str,
) -> Dict:
    """
    Train the Fusion Classifier.

    Strategy
    --------
    - β-VAE is fully frozen throughout.
    - ViT encoder blocks 10–11 + classifier head remain trainable.
    - Fusion MLP is fully trainable.

    Critical: reload the best ViT checkpoint before fusion training.
    In-memory ViT weights from the end of Phase 2 may show minor drift
    compared to the epoch-14 (or best) checkpoint.
    """
    # Reload best ViT checkpoint
    best_vit = cfg.vit_ckpt_path()
    if os.path.exists(best_vit):
        pipeline.vit.load_state_dict(torch.load(best_vit, map_location=device))
        print(f"  Reloaded best ViT: {best_vit}")
    else:
        print("  ⚠️  No ViT checkpoint found — using current weights")

    # Freeze VAE; partially unfreeze ViT
    for p in pipeline.vae.parameters():
        p.requires_grad = False
    for name, p in pipeline.vit.named_parameters():
        p.requires_grad = (
            "blocks.10" in name
            or "blocks.11" in name
            or "classifier" in name
        )

    trainable = [p for p in pipeline.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.LR_FUSION, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.EPOCHS_CLF,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    history   = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_acc  = 0.0
    best_ckpt = cfg.pipe_ckpt_path()
    scaler    = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for epoch in range(cfg.EPOCHS_CLF):
        pipeline.train()
        tl, tc, n = 0.0, 0, 0

        for imgs, labels in tqdm(train_loader, desc=f"Fusion {epoch+1}/{cfg.EPOCHS_CLF}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                logits = pipeline(imgs)
                loss   = criterion(logits, labels)

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()

            tl += loss.item() * imgs.size(0)
            tc += (logits.argmax(1) == labels).sum().item()
            n  += imgs.size(0)

        scheduler.step()

        pipeline.eval()
        vl, vc, nv = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = pipeline(imgs)
                loss_v = criterion(logits, labels)
                if torch.isfinite(loss_v):
                    vl += loss_v.item() * imgs.size(0)
                vc += (logits.argmax(1) == labels).sum().item()
                nv += imgs.size(0)

        ta = tc / max(n,  1)
        va = vc / max(nv, 1)
        history["train_loss"].append(tl / max(n, 1))
        history["val_loss"].append(  vl / max(nv, 1))
        history["train_acc"].append(ta)
        history["val_acc"].append(va)
        print(f"Epoch {epoch+1:3d}/{cfg.EPOCHS_CLF} | Train Acc:{ta:.4f} | Val Acc:{va:.4f}")

        if va > best_acc:
            best_acc = va
            torch.save(
                {
                    "vit":     vit_model.state_dict(),
                    "vae":     vae_model.state_dict(),
                    "fusion":  fusion_clf.state_dict(),
                    "classes": class_names,
                    "dataset": dataset_name,
                },
                best_ckpt,
            )
            print(f"  ✅ Best pipeline saved → val_acc:{va:.4f}")

    print(f"\nFusion training complete — best val_acc: {best_acc:.4f}")
    return history
