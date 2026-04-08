"""
scripts/download_data.py
────────────────────────
Dataset download helper.

Usage
-----
    python scripts/download_data.py --dataset neu_steel
    python scripts/download_data.py --dataset mvtec --category bottle
    python scripts/download_data.py --dataset dagm
    python scripts/download_data.py --dataset kaggle_steel
"""

import argparse
import os
import shutil
import tarfile
import zipfile
from pathlib import Path


DATA_ROOT = os.environ.get("DATA_ROOT", "./data")


def download_neu_steel(data_root: str) -> str:
    """
    NEU Steel Surface Defect Database.
    Primary source: Kaggle (requires kaggle.json in ~/.kaggle/)
    Fallback:       gdown from public Google Drive mirror.
    """
    neu_dir = os.path.join(data_root, "neu_steel")
    os.makedirs(neu_dir, exist_ok=True)

    # Check if already downloaded
    neu_classes = ["crazing", "inclusion", "patches",
                   "pitted_surface", "rolled-in_scale", "scratches"]
    if any(os.path.isdir(os.path.join(neu_dir, c)) for c in neu_classes):
        print(f"✅ NEU Steel already exists at {neu_dir}")
        return neu_dir

    print("Downloading NEU Steel Surface Defect dataset …")

    # Method 1: Kaggle API
    try:
        import subprocess
        result = subprocess.run(
            ["kaggle", "datasets", "download",
             "-d", "kaustubhdikshit/neu-surface-defect-database",
             "-p", neu_dir, "--unzip"],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print("  ✅ Downloaded via Kaggle API")
            _restructure_neu(neu_dir)
            return neu_dir
        print(f"  Kaggle failed: {result.stderr.strip()}")
    except FileNotFoundError:
        print("  kaggle CLI not found — trying gdown …")

    # Method 2: gdown (public mirror)
    try:
        import gdown
        file_id  = "1qrdZlaDi272-9CWEFfb5OLQ5bkGGhGCq"
        zip_path = os.path.join(data_root, "neu_steel.zip")
        gdown.download(f"https://drive.google.com/uc?id={file_id}", zip_path, quiet=False)
        if os.path.exists(zip_path) and os.path.getsize(zip_path) > 0:
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(neu_dir)
            os.remove(zip_path)
            print("  ✅ Downloaded via gdown")
            _restructure_neu(neu_dir)
            return neu_dir
    except Exception as e:
        print(f"  gdown failed: {e}")

    print(
        "\n⚠️  Auto-download failed.\n"
        "Manual steps:\n"
        "  1. Go to https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database\n"
        "  2. Download and extract into data/neu_steel/\n"
        "  3. Ensure structure: data/neu_steel/<class_name>/<image>.jpg"
    )
    return neu_dir


def _restructure_neu(base_dir: str) -> None:
    """Ensure NEU images are in ImageFolder format: base_dir/class_name/*.jpg"""
    NEU_CLASSES = ["crazing", "inclusion", "patches",
                   "pitted_surface", "rolled-in_scale", "scratches"]

    # Already structured?
    if all(
        os.path.isdir(os.path.join(base_dir, c)) and
        any(f.endswith((".jpg", ".bmp"))
            for f in os.listdir(os.path.join(base_dir, c)))
        for c in NEU_CLASSES
        if os.path.isdir(os.path.join(base_dir, c))
    ):
        return

    # Handle Kaggle's nested extraction: base_dir/NEU-DET/NEU-DET/
    for sub in ["NEU-DET/NEU-DET", "NEU-DET"]:
        candidate = os.path.join(base_dir, sub)
        if os.path.exists(candidate):
            for item in os.listdir(candidate):
                src = os.path.join(candidate, item)
                dst = os.path.join(base_dir, item)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
            break

    # Move flat images into class subdirectories
    for cls in NEU_CLASSES:
        os.makedirs(os.path.join(base_dir, cls), exist_ok=True)

    for fname in os.listdir(base_dir):
        if not fname.lower().endswith((".jpg", ".bmp", ".png")):
            continue
        for cls in NEU_CLASSES:
            if fname.lower().startswith(cls[:3]):
                src = os.path.join(base_dir, fname)
                dst = os.path.join(base_dir, cls, fname)
                if not os.path.exists(dst):
                    shutil.move(src, dst)
                break

    print(f"  ✅ NEU Steel restructured in {base_dir}")


def download_mvtec(data_root: str, category: str = "bottle") -> str:
    """Download a single MVTec AD category."""
    url_map = {
        "bottle":     "https://www.mvtec.com/fileadmin/Redaktion/mvtec-ad/bottle.tar.gz",
        "cable":      "https://www.mvtec.com/fileadmin/Redaktion/mvtec-ad/cable.tar.gz",
        "capsule":    "https://www.mvtec.com/fileadmin/Redaktion/mvtec-ad/capsule.tar.gz",
        "carpet":     "https://www.mvtec.com/fileadmin/Redaktion/mvtec-ad/carpet.tar.gz",
        "grid":       "https://www.mvtec.com/fileadmin/Redaktion/mvtec-ad/grid.tar.gz",
        "hazelnut":   "https://www.mvtec.com/fileadmin/Redaktion/mvtec-ad/hazelnut.tar.gz",
        "leather":    "https://www.mvtec.com/fileadmin/Redaktion/mvtec-ad/leather.tar.gz",
        "metal_nut":  "https://www.mvtec.com/fileadmin/Redaktion/mvtec-ad/metal_nut.tar.gz",
        "pill":       "https://www.mvtec.com/fileadmin/Redaktion/mvtec-ad/pill.tar.gz",
        "screw":      "https://www.mvtec.com/fileadmin/Redaktion/mvtec-ad/screw.tar.gz",
        "tile":       "https://www.mvtec.com/fileadmin/Redaktion/mvtec-ad/tile.tar.gz",
        "toothbrush": "https://www.mvtec.com/fileadmin/Redaktion/mvtec-ad/toothbrush.tar.gz",
        "transistor": "https://www.mvtec.com/fileadmin/Redaktion/mvtec-ad/transistor.tar.gz",
        "wood":       "https://www.mvtec.com/fileadmin/Redaktion/mvtec-ad/wood.tar.gz",
        "zipper":     "https://www.mvtec.com/fileadmin/Redaktion/mvtec-ad/zipper.tar.gz",
    }
    cat_dir = os.path.join(data_root, "mvtec", category)
    if os.path.exists(cat_dir):
        print(f"✅ MVTec '{category}' already exists at {cat_dir}")
        return cat_dir

    url = url_map.get(category)
    if not url:
        raise ValueError(f"Unknown MVTec category: {category}. "
                         f"Choose from: {sorted(url_map)}")

    os.makedirs(os.path.join(data_root, "mvtec"), exist_ok=True)
    tar_path = os.path.join(data_root, "mvtec", f"{category}.tar.gz")
    print(f"Downloading MVTec '{category}' …")
    import urllib.request
    urllib.request.urlretrieve(url, tar_path)
    with tarfile.open(tar_path) as tf:
        tf.extractall(os.path.join(data_root, "mvtec"))
    os.remove(tar_path)
    print(f"✅ MVTec '{category}' extracted to {cat_dir}")
    return cat_dir


def download_kaggle_steel(data_root: str) -> str:
    """Download Severstal Kaggle Steel Defect Detection (requires kaggle.json)."""
    steel_dir = os.path.join(data_root, "kaggle_steel")
    if os.path.exists(steel_dir) and len(os.listdir(steel_dir)) > 2:
        print(f"✅ Kaggle Steel already exists at {steel_dir}")
        return steel_dir

    os.makedirs(steel_dir, exist_ok=True)
    print("Downloading Severstal Steel from Kaggle …")
    import subprocess
    result = subprocess.run(
        ["kaggle", "competitions", "download",
         "-c", "severstal-steel-defect-detection", "-p", steel_dir],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Kaggle download failed:\n{result.stderr}\n"
            "Make sure kaggle.json is at ~/.kaggle/kaggle.json"
        )
    zip_f = os.path.join(steel_dir, "severstal-steel-defect-detection.zip")
    if os.path.exists(zip_f):
        with zipfile.ZipFile(zip_f) as z:
            z.extractall(steel_dir)
        os.remove(zip_f)
    print(f"✅ Kaggle Steel extracted to {steel_dir}")
    return steel_dir


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download industrial defect detection datasets"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["neu_steel", "mvtec", "kaggle_steel", "dagm"],
        help="Which dataset to download",
    )
    parser.add_argument(
        "--category",
        default="bottle",
        help="MVTec category (only used with --dataset mvtec)",
    )
    parser.add_argument(
        "--data-root",
        default=DATA_ROOT,
        help=f"Root directory for data (default: {DATA_ROOT})",
    )
    args = parser.parse_args()

    os.makedirs(args.data_root, exist_ok=True)

    if args.dataset == "neu_steel":
        download_neu_steel(args.data_root)
    elif args.dataset == "mvtec":
        download_mvtec(args.data_root, args.category)
    elif args.dataset == "kaggle_steel":
        download_kaggle_steel(args.data_root)
    elif args.dataset == "dagm":
        print(
            "DAGM 2007 requires manual download:\n"
            "  https://conferences.mpi-inf.mpg.de/dagm/2007/prizes.html\n"
            "Extract to: data/dagm/<class_name>/<image>.png"
        )


if __name__ == "__main__":
    main()
