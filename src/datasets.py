"""
src/datasets.py
───────────────
Universal dataset adapters for industrial defect detection.

Supported formats
-----------------
MVTecDataset          : MVTec Anomaly Detection (one category at a time)
NEUDataset            : NEU Steel Surface Defect Database (6 classes)
KaggleSteelDataset    : Severstal Kaggle Steel Defect Detection
GenericDefectDataset  : Any ImageFolder-compatible structure
NormalOnlySubset      : Wraps any dataset, returns only normal-class images
DefectClassSubset     : Wraps any dataset, returns only a specific defect class

All adapters expose a unified (image_tensor, label) interface
compatible with torch.utils.data.DataLoader.
"""

import os
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset


# ── MVTec Anomaly Detection ───────────────────────────────────────────────────

class MVTecDataset(Dataset):
    """
    MVTec AD directory layout:
        {category}/
            train/good/*.png          ← normal training images
            test/good/*.png           ← normal test images
            test/{defect_type}/*.png  ← anomalous test images
            ground_truth/...          ← pixel-level masks (not loaded here)

    Labels: 0 = good, 1..N = defect types (alphabetical order).
    """

    IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

    def __init__(self, root: str, split: str = "train", transform=None):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.class_names: List[str] = []

        root = Path(root)

        if split == "train":
            self.class_names = ["good"]
            good_dir = root / "train" / "good"
            for p in sorted(good_dir.iterdir()):
                if p.suffix.lower() in self.IMG_EXTS:
                    self.samples.append((str(p), 0))
        else:
            test_root = root / "test"
            defect_types = sorted(os.listdir(test_root))
            self.class_names = defect_types
            label_map = {d: (0 if d == "good" else i) for i, d in enumerate(defect_types)}
            for defect in defect_types:
                for p in sorted((test_root / defect).iterdir()):
                    if p.suffix.lower() in self.IMG_EXTS:
                        self.samples.append((str(p), label_map[defect]))

        print(f"  MVTecDataset [{split}]: {len(self.samples)} images | "
              f"classes: {sorted(set(s[1] for s in self.samples))}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── NEU Steel Surface Defect Database ────────────────────────────────────────

class NEUDataset(Dataset):
    """
    NEU Surface Defect Database:
    - 6 classes: crazing, inclusion, patches, pitted_surface,
                 rolled-in_scale, scratches
    - 300 images per class, 200×200 greyscale (loaded as RGB)

    Expected layout:
        root/
            crazing/       *.jpg
            inclusion/     *.jpg
            patches/       *.jpg
            pitted_surface/*.jpg
            rolled-in_scale/*.jpg
            scratches/     *.jpg
    """

    CLASSES = [
        "crazing", "inclusion", "patches",
        "pitted_surface", "rolled-in_scale", "scratches",
    ]
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.class_names: List[str] = []

        # Discover class directories
        found_classes = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])
        self.class_names = found_classes if found_classes else self.CLASSES
        label_map = {c: i for i, c in enumerate(self.class_names)}

        for cls in self.class_names:
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if Path(fname).suffix.lower() in self.IMG_EXTS:
                    self.samples.append((os.path.join(cls_dir, fname), label_map[cls]))

        # Flat-structure fallback (images named e.g. 'crazing_01.jpg' in root)
        if not self.samples:
            self.class_names = self.CLASSES
            label_map = {c: i for i, c in enumerate(self.CLASSES)}
            for fname in os.listdir(root):
                if Path(fname).suffix.lower() not in self.IMG_EXTS:
                    continue
                for cls in self.CLASSES:
                    if fname.lower().startswith(cls[:3]):
                        self.samples.append(
                            (os.path.join(root, fname), label_map[cls])
                        )
                        break

        print(f"  NEUDataset: {len(self.samples)} images | "
              f"{len(self.class_names)} classes: {self.class_names}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Severstal Kaggle Steel ─────────────────────────────────────────────────────

class KaggleSteelDataset(Dataset):
    """
    Severstal Steel Defect Detection (Kaggle competition).
    Requires:
        root/train_images/*.jpg
        root/train.csv  (columns: ImageId, ClassId, EncodedPixels)

    Label per image = highest defect ClassId present (0 = no defect).
    """

    CLASSES = ["No Defect", "Class 1", "Class 2", "Class 3", "Class 4"]

    def __init__(self, root: str, transform=None, max_samples: int = 5000):
        import pandas as pd

        self.transform = transform
        self.class_names = self.CLASSES
        self.samples: List[Tuple[str, int]] = []

        img_dir  = os.path.join(root, "train_images")
        csv_path = os.path.join(root, "train.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"train.csv not found in {root}")

        df = pd.read_csv(csv_path)
        df["ClassId"] = df["ClassId"].astype(int)

        img_labels: dict = {}
        for _, row in df.iterrows():
            img_id = row["ImageId"]
            cls    = row["ClassId"] if pd.notna(row["EncodedPixels"]) else 0
            img_labels[img_id] = max(img_labels.get(img_id, 0), cls)

        for fname in sorted(os.listdir(img_dir))[:max_samples]:
            if fname.endswith(".jpg"):
                lbl = img_labels.get(fname, 0)
                self.samples.append((os.path.join(img_dir, fname), lbl))

        print(f"  KaggleSteelDataset: {len(self.samples)} images")
        cnt = Counter(s[1] for s in self.samples)
        for k, v in sorted(cnt.items()):
            print(f"    {self.CLASSES[k]:12s}: {v}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Generic ImageFolder-compatible ────────────────────────────────────────────

class GenericDefectDataset(Dataset):
    """
    Works with any dataset organised as:
        root/
            class_name_1/img1.jpg ...
            class_name_2/img1.jpg ...

    Compatible with DAGM 2007, custom uploads, and any other
    standard ImageFolder-style layout.
    """

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

    def __init__(
        self,
        root: str,
        transform=None,
        split_frac: Optional[float] = None,
        split: str = "train",
        seed: int = 42,
    ):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.class_names: List[str] = []

        classes = sorted([
            d for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))
        ])
        self.class_names = classes
        label_map = {c: i for i, c in enumerate(classes)}

        for cls in classes:
            cls_dir = os.path.join(root, cls)
            for fname in os.listdir(cls_dir):
                if Path(fname).suffix.lower() in self.IMG_EXTS:
                    self.samples.append((os.path.join(cls_dir, fname), label_map[cls]))

        # Optional deterministic train/val split
        if split_frac is not None:
            random.seed(seed)
            random.shuffle(self.samples)
            cut = int(len(self.samples) * split_frac)
            self.samples = self.samples[:cut] if split == "train" else self.samples[cut:]

        print(f"  GenericDefectDataset [{split}]: {len(self.samples)} images | "
              f"{len(classes)} classes: {classes}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ── Normal-only subset (for β-VAE training) ───────────────────────────────────

class NormalOnlySubset(Dataset):
    """
    Collects file paths from a Subset/Dataset and applies the VAE transform
    at load time.

    Fixed design:
    - Operates on file paths only (no tensor re-normalisation) to avoid
      double-normalisation bugs.
    - use_all=True  → use all classes (appropriate for NEU Steel, DAGM, etc.)
    - use_all=False → use normal_label only (appropriate for MVTec AD)
    """

    def __init__(
        self,
        base_dataset,
        vae_transform=None,
        use_all: bool = False,
        normal_label: int = 0,
    ):
        self.transform = vae_transform
        self.paths: List[str] = []

        # Resolve Subset → base dataset
        if hasattr(base_dataset, "indices"):
            indices = list(base_dataset.indices)
            src = base_dataset.dataset
        else:
            indices = list(range(len(base_dataset)))
            src = base_dataset

        for idx in indices:
            try:
                path, lbl = src.samples[idx]
                if use_all or lbl == normal_label:
                    self.paths.append(path)
            except Exception:
                pass  # Skip samples without a file path attribute

        label_desc = "all classes" if use_all else f"label={normal_label}"
        print(f"  NormalOnlySubset: {len(self.paths)} paths ({label_desc})")

        if len(self.paths) == 0:
            raise RuntimeError(
                "NormalOnlySubset found 0 images! "
                "Ensure your dataset exposes a .samples attribute."
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


# ── Defect-class subset (for DDPM training) ───────────────────────────────────

class DefectClassSubset(Dataset):
    """
    Subset of a training dataset filtered to a single defect class.
    Used to train the DDPM on the most representative defect category.

    Falls back to the full training set if the target class has fewer
    than `min_samples` images.
    """

    def __init__(
        self,
        base_dataset,
        target_label: int = 1,
        vae_transform=None,
        min_samples: int = 20,
    ):
        self.transform = vae_transform
        self.samples: List[str] = []

        try:
            indices = base_dataset.indices
            src = base_dataset.dataset
        except AttributeError:
            indices = range(len(base_dataset))
            src = base_dataset

        for idx in indices:
            try:
                path, lbl = src.samples[idx]
                if lbl == target_label:
                    self.samples.append(path)
            except Exception:
                pass

        # Fallback to all training images if target class is too small
        if len(self.samples) < min_samples:
            print(f"  ⚠️  Only {len(self.samples)} images for class {target_label}. "
                  f"Falling back to full training set for DDPM.")
            self.samples = []
            for idx in indices:
                try:
                    path, _ = src.samples[idx]
                    self.samples.append(path)
                except Exception:
                    pass

        print(f"  DefectClassSubset: {len(self.samples)} images (class {target_label})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0
