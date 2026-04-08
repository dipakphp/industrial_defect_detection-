"""
app.py
──────
Standalone Gradio web application for Industrial Defect Detection.

Usage
-----
    # Local
    python app.py

    # After training, place checkpoints in ./checkpoints/:
    #   full_pipeline_best.pt  (required for Detection mode)
    #   ddpm_best.pt           (required for Generation mode)

Modes
-----
    Detection  : Upload image → ViT class prediction + β-VAE anomaly heatmap
    Generation : Upload image → DDPM forward diffusion strip (t=0 → t=999)

Anomaly severity scale
----------------------
    < 8%    NORMAL
    8–25%   LOW
    25–55%  MODERATE
    > 55%   HIGH
"""

import os
import traceback

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from configs.config import Config
from src.models.ddpm import DDPMScheduler, UNet
from src.models.fusion import FusionClassifier, FullPipeline
from src.models.vae import BetaVAE
from src.models.vit import ViTFeatureExtractor
from src.transforms import val_tf, vae_tf
from src.utils.heatmap import compute_anomaly_score, tensor_to_vivid_heatmap

# ── Configuration & device ────────────────────────────────────────────────────
cfg    = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ── Load checkpoint ───────────────────────────────────────────────────────────
pipeline_ckpt = cfg.pipe_ckpt_path()
ddpm_ckpt     = cfg.ddpm_ckpt_path()

if not os.path.exists(pipeline_ckpt):
    raise FileNotFoundError(
        f"Pipeline checkpoint not found: {pipeline_ckpt}\n"
        "Run the training notebook first, then place full_pipeline_best.pt "
        f"in {cfg.CHECKPOINT_DIR}/"
    )

ck          = torch.load(pipeline_ckpt, map_location=DEVICE)
CLASS_NAMES = ck.get("classes", [f"Class {i}" for i in range(cfg.NUM_CLASSES
                                                               if hasattr(cfg, "NUM_CLASSES") else 6)])
NUM_CLASSES = len(CLASS_NAMES)

# Instantiate models
vit_model  = ViTFeatureExtractor(num_classes=NUM_CLASSES).to(DEVICE)
vae_model  = BetaVAE(latent_dim=cfg.LATENT_DIM, beta=cfg.VAE_BETA).to(DEVICE)
fusion_clf = FusionClassifier(
    vit_dim=vit_model.feature_dim,
    vae_dim=cfg.LATENT_DIM,
    num_classes=NUM_CLASSES,
).to(DEVICE)
pipeline = FullPipeline(vit_model, vae_model, fusion_clf).to(DEVICE)

vit_model.load_state_dict(ck["vit"])
vae_model.load_state_dict(ck["vae"])
fusion_clf.load_state_dict(ck["fusion"])
pipeline.eval()
vit_model.eval()
vae_model.eval()
print(f"✅ Pipeline loaded — {NUM_CLASSES} classes: {CLASS_NAMES}")

# DDPM (optional — only needed for Generation mode)
ddpm_model     = UNet(ch=3, base=cfg.DDPM_CHANNELS, t_dim=cfg.DDPM_T_DIM).to(DEVICE)
ddpm_scheduler = DDPMScheduler(T=cfg.TIMESTEPS)
if os.path.exists(ddpm_ckpt):
    ddpm_model.load_state_dict(torch.load(ddpm_ckpt, map_location=DEVICE))
    ddpm_model.eval()
    print(f"✅ DDPM loaded: {ddpm_ckpt}")
else:
    print(f"⚠️  DDPM checkpoint not found ({ddpm_ckpt}) — Generation mode disabled")


# ── Inference logic ───────────────────────────────────────────────────────────

def industrial_app_logic(input_img, mode: str):
    """Core inference function called by Gradio."""
    if input_img is None:
        return None, "Please upload an image first."

    try:
        # Normalise input
        if isinstance(input_img, np.ndarray):
            pil_img = Image.fromarray(input_img.astype(np.uint8))
        else:
            pil_img = input_img
        pil_img  = pil_img.convert("RGB").resize((cfg.IMG_SIZE, cfg.IMG_SIZE))
        img_vit  = val_tf(pil_img).unsqueeze(0).to(DEVICE)   # ImageNet norm
        img_vae  = vae_tf(pil_img).unsqueeze(0).to(DEVICE)   # [-1, +1] norm

        # ── Detection ─────────────────────────────────────────────────────
        if mode == "Detection (ViT + VAE)":
            with torch.no_grad():
                logits, features = vit_model(img_vit)

                if not torch.isfinite(logits).all():
                    return None, (
                        "⚠️  ViT weights contain NaN.\n"
                        "Reload the checkpoint:\n"
                        "  vit_model.load_state_dict(torch.load("
                        "cfg.vit_ckpt_path(), map_location=DEVICE))"
                    )

                probs    = torch.softmax(logits, dim=1)[0]
                pred_idx = int(probs.argmax().item())
                conf     = float(probs.max().item())
                label    = CLASS_NAMES[pred_idx] if pred_idx < NUM_CLASSES else str(pred_idx)

                recon, mu, _, _ = vae_model(img_vae)

                if not torch.isfinite(recon).all():
                    return None, (
                        "⚠️  VAE weights contain NaN.\n"
                        "Retrain the VAE using the fixed training code "
                        "(AMP disabled) or reload vae_best.pt."
                    )

                diff = (img_vae.squeeze(0).cpu() - recon.squeeze(0).cpu()).abs()

            raw_score, pct_score, level = compute_anomaly_score(diff)
            heatmap_pil = tensor_to_vivid_heatmap(diff, pil_img)

            prob_lines = "\n".join([
                f"  {CLASS_NAMES[i]:22s}: {probs[i].item() * 100:.2f}%"
                for i in range(NUM_CLASSES)
            ])
            result_str = (
                f"Predicted Class  : {label}\n"
                f"Confidence       : {conf * 100:.2f}%\n"
                f"Anomaly Score    : {pct_score:.1f}%  ({level})\n"
                f"Raw Recon Error  : {raw_score:.6f}\n"
                f"{'─' * 44}\n"
                f"All Class Probabilities:\n{prob_lines}"
            )
            return heatmap_pil, result_str

        # ── Generation ────────────────────────────────────────────────────
        else:
            if not os.path.exists(ddpm_ckpt):
                return None, "DDPM checkpoint not found. Run DDPM training first."

            ddpm_model.eval()
            timesteps_viz = [0, 200, 400, 600, 800, 999]
            frames = []
            for t_val in timesteps_viz:
                t_tensor = torch.tensor([t_val], device=DEVICE)
                with torch.no_grad():
                    noisy, _ = ddpm_scheduler.q_sample(img_vae, t_tensor)
                frame    = ((noisy.squeeze().cpu() + 1) / 2).clamp(0, 1)
                frame_np = (frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                frames.append(frame_np)

            h, w      = frames[0].shape[:2]
            label_h   = 22
            canvas    = np.zeros((h + label_h, w * len(frames), 3), dtype=np.uint8)
            for fi, (frame, t_val) in enumerate(zip(frames, timesteps_viz)):
                x = fi * w
                canvas[label_h:, x:x + w] = frame
                cv2.putText(
                    canvas, f"t={t_val}", (x + 3, label_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1,
                )

            result_pil = Image.fromarray(canvas)
            return result_pil, (
                "DDPM Forward Diffusion Process\n"
                "t=0   : Original input image\n"
                "t=999 : Pure Gaussian noise\n\n"
                "The model learns to REVERSE this process\n"
                "to synthesise new realistic defect images.\n"
            )

    except Exception as e:
        return None, f"Error: {str(e)}\n\n{traceback.format_exc()}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(theme=gr.themes.Soft(), title="Industrial Defect Detection") as demo:
        gr.Markdown("# 🏭 Industrial AI: ViT + β-VAE + DDPM")
        gr.Markdown(
            f"**Classes ({NUM_CLASSES}):** {', '.join(CLASS_NAMES)}  |  "
            f"**Device:** `{DEVICE}`"
        )

        with gr.Row():
            with gr.Column(scale=1):
                in_img   = gr.Image(
                    type="pil",
                    label="Upload Surface Image",
                    sources=["upload", "webcam"],
                )
                mode_sel = gr.Radio(
                    choices=["Detection (ViT + VAE)", "Generation (DDPM)"],
                    value="Detection (ViT + VAE)",
                    label="Select Mode",
                )
                run_btn = gr.Button("Run Analysis", variant="primary", size="lg")

            with gr.Column(scale=1):
                out_img = gr.Image(
                    label="Anomaly Heatmap / Diffusion Visualisation",
                    type="pil",
                )
                out_txt = gr.Textbox(
                    label="Results & Metrics",
                    lines=16,
                    max_lines=20,
                )

        gr.Markdown(
            "**Detection:** ViT classifies defect + β-VAE heatmap "
            "(blue = normal, red = anomaly)  |  "
            "**Generation:** DDPM forward diffusion t=0 → t=999"
        )

        run_btn.click(
            fn=industrial_app_logic,
            inputs=[in_img, mode_sel],
            outputs=[out_img, out_txt],
            queue=False,
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(share=False, server_port=7860)
