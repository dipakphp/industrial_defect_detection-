"""
src/utils/evaluation.py
───────────────────────
Evaluation utilities:

    evaluate()         : Full test-set evaluation (accuracy, F1, AUC-ROC)
    plot_confusion()   : Confusion matrix heatmap
    plot_roc_curves()  : Per-class ROC curves with AUC annotations
"""

from itertools import cycle
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def evaluate(
    pipeline,
    loader:      DataLoader,
    num_classes: int,
    class_names: List[str],
    device:      torch.device,
    dataset_name: str = "",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Evaluate the full pipeline on a test DataLoader.

    Returns
    -------
    preds      : [N]        Predicted class indices
    labels_all : [N]        Ground-truth class indices
    probs_all  : [N, C]     Softmax probabilities
    metrics    : dict       accuracy, f1, auc
    """
    pipeline.eval()
    preds, labels_all, probs_all = [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating"):
            logits    = pipeline(imgs.to(device))
            probs_all.extend(F.softmax(logits, dim=1).cpu().numpy())
            preds.extend(logits.argmax(1).cpu().numpy())
            labels_all.extend(labels.numpy())

    p  = np.array(preds)
    l  = np.array(labels_all)
    pr = np.array(probs_all)

    acc = float((p == l).mean())
    f1  = float(f1_score(l, p, average="weighted", zero_division=0))

    try:
        yb  = label_binarize(l, classes=list(range(num_classes)))
        auc = float(roc_auc_score(yb, pr, average="macro", multi_class="ovr"))
    except Exception as e:
        print(f"  ⚠️  AUC-ROC skipped: {e}")
        auc = float("nan")

    tag = f" — {dataset_name.upper()}" if dataset_name else ""
    print("=" * 55)
    print(f"  TEST RESULTS{tag}")
    print("=" * 55)
    print(f"  Accuracy : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"  F1 Score : {f1:.4f}  (weighted)")
    print(f"  AUC-ROC  : {auc:.4f}  (macro OvR)")
    print("=" * 55)
    print(
        classification_report(
            l, p,
            labels=list(range(num_classes)),
            target_names=class_names[:num_classes],
            digits=4,
            zero_division=0,
        )
    )

    return p, l, pr, {"accuracy": acc, "f1": f1, "auc": auc}


def plot_confusion(
    preds:       np.ndarray,
    labels_all:  np.ndarray,
    class_names: List[str],
    accuracy:    float,
    save_path:   str = "confusion_matrix.png",
) -> None:
    """Plot and save a labelled confusion matrix."""
    cm = confusion_matrix(labels_all, preds)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 1.2),
                                     max(6, len(class_names) * 1.0)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix  —  Acc = {accuracy:.4f}")
    ax.tick_params(axis="x", rotation=40)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved: {save_path}")


def plot_roc_curves(
    labels_all:  np.ndarray,
    probs_all:   np.ndarray,
    class_names: List[str],
    num_classes: int,
    macro_auc:   float,
    save_path:   str = "roc_curves.png",
) -> None:
    """Plot per-class ROC curves with AUC annotations."""
    yb     = label_binarize(labels_all, classes=list(range(num_classes)))
    colors = cycle(plt.cm.tab10.colors)

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, (col, name) in enumerate(zip(colors, class_names[:num_classes])):
        try:
            fpr, tpr, _ = roc_curve(yb[:, i], probs_all[:, i])
            auc_i = roc_auc_score(yb[:, i], probs_all[:, i])
            ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC={auc_i:.3f})")
        except Exception:
            pass

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves  —  Macro AUC = {macro_auc:.4f}")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved: {save_path}")
