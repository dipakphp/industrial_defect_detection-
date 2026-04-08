"""
tests/test_datasets.py
──────────────────────
Unit tests for dataset adapters using temporary directories.

Run with:  pytest tests/test_datasets.py -v
"""

import os
import tempfile

import pytest
import torch
from PIL import Image

from src.datasets import (
    GenericDefectDataset,
    NormalOnlySubset,
    DefectClassSubset,
)
from src.transforms import train_tf, vae_tf


def _make_fake_dataset(root: str, classes: list, n_per_class: int = 10):
    """Create a minimal ImageFolder-style directory for testing."""
    fake_samples = []
    for i, cls in enumerate(classes):
        cls_dir = os.path.join(root, cls)
        os.makedirs(cls_dir, exist_ok=True)
        for j in range(n_per_class):
            img_path = os.path.join(cls_dir, f"{cls}_{j:03d}.jpg")
            Image.new("RGB", (64, 64), color=(i * 40, j * 20, 100)).save(img_path)
            fake_samples.append((img_path, i))
    return fake_samples


class TestGenericDefectDataset:
    def test_loads_all_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            classes = ["good", "scratch", "crack"]
            _make_fake_dataset(tmp, classes, n_per_class=5)
            ds = GenericDefectDataset(tmp, transform=train_tf)
            assert len(ds.class_names) == 3
            assert len(ds) == 15

    def test_item_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_fake_dataset(tmp, ["a", "b"], n_per_class=3)
            ds  = GenericDefectDataset(tmp, transform=train_tf)
            img, label = ds[0]
            assert img.shape == torch.Size([3, 224, 224])
            assert isinstance(label, int)

    def test_split_frac(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_fake_dataset(tmp, ["x", "y"], n_per_class=10)
            train = GenericDefectDataset(tmp, transform=train_tf,
                                         split_frac=0.8, split="train")
            val   = GenericDefectDataset(tmp, transform=train_tf,
                                         split_frac=0.8, split="val")
            assert len(train) + len(val) == 20


class TestNormalOnlySubset:
    def test_filters_by_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_fake_dataset(tmp, ["good", "defect"], n_per_class=8)
            ds     = GenericDefectDataset(tmp, transform=vae_tf)
            normal = NormalOnlySubset(ds, vae_transform=vae_tf,
                                      use_all=False, normal_label=0)
            # All normal_only images should have label 0 in the base dataset
            assert len(normal) == 8

    def test_use_all(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_fake_dataset(tmp, ["a", "b", "c"], n_per_class=6)
            ds  = GenericDefectDataset(tmp, transform=vae_tf)
            all_sub = NormalOnlySubset(ds, vae_transform=vae_tf, use_all=True)
            assert len(all_sub) == 18

    def test_item_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_fake_dataset(tmp, ["good", "bad"], n_per_class=4)
            ds     = GenericDefectDataset(tmp, transform=vae_tf)
            normal = NormalOnlySubset(ds, vae_transform=vae_tf, use_all=True)
            img, label = normal[0]
            assert img.shape == torch.Size([3, 224, 224])


class TestDefectClassSubset:
    def test_filters_target_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_fake_dataset(tmp, ["class0", "class1", "class2"], n_per_class=10)
            ds  = GenericDefectDataset(tmp, transform=vae_tf)
            sub = DefectClassSubset(ds, target_label=1, vae_transform=vae_tf,
                                    min_samples=1)
            assert len(sub) == 10

    def test_fallback_on_empty_class(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_fake_dataset(tmp, ["only_class"], n_per_class=12)
            ds  = GenericDefectDataset(tmp, transform=vae_tf)
            # target_label=5 doesn't exist → should fallback to full dataset
            sub = DefectClassSubset(ds, target_label=5, vae_transform=vae_tf,
                                    min_samples=5)
            assert len(sub) == 12
