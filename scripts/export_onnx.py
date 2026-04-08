"""
scripts/export_onnx.py
──────────────────────
Export the trained ViT backbone to ONNX format.

Usage
-----
    python scripts/export_onnx.py
    python scripts/export_onnx.py --ckpt ./checkpoints/vit_best.pt --out ./checkpoints/vit.onnx
"""

import argparse

import torch

from configs.config import Config
from src.models.vit import ViTFeatureExtractor


def export_vit_onnx(
    ckpt_path: str,
    out_path:  str,
    num_classes: int,
    img_size:    int = 224,
    opset:       int = 14,
) -> None:
    device = torch.device("cpu")  # ONNX export on CPU for portability
    model  = ViTFeatureExtractor(num_classes=num_classes)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    dummy = torch.randn(1, 3, img_size, img_size)

    torch.onnx.export(
        model,
        dummy,
        out_path,
        input_names=["image"],
        output_names=["logits", "features"],
        dynamic_axes={"image": {0: "batch_size"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"✅ ONNX model exported to {out_path}")
    print(f"   Input  : image [batch, 3, {img_size}, {img_size}]")
    print(f"   Outputs: logits [batch, {num_classes}], features [batch, 768]")


def main():
    cfg = Config()
    parser = argparse.ArgumentParser(description="Export ViT to ONNX")
    parser.add_argument("--ckpt",        default=cfg.vit_ckpt_path())
    parser.add_argument("--out",         default=cfg.CHECKPOINT_DIR + "/vit_model.onnx")
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--img-size",    type=int, default=cfg.IMG_SIZE)
    parser.add_argument("--opset",       type=int, default=14)
    args = parser.parse_args()

    export_vit_onnx(
        ckpt_path=args.ckpt,
        out_path=args.out,
        num_classes=args.num_classes,
        img_size=args.img_size,
        opset=args.opset,
    )


if __name__ == "__main__":
    main()
