"""
Supervised (cross-entropy) training of the SAME CIFAR-adapted ResNet-18 on
CIFAR-10, at matched scale/hyperparameters/augmentation to train_contrastive.py,
reproducing Betser, Gofer, Levi, Gilboa, "InfoNCE Induces Gaussian Distribution"
(ICLR 2026, arXiv:2602.24012) Table 1's "Supervised" vs "Contrastive" comparison:
identical architecture, data, and augmentation - only the training OBJECTIVE
differs - to isolate whether Gaussianity comes from InfoNCE itself.

Gaussianity diagnostics are run on the same backbone feature space (512-dim,
pre-classifier) evaluated for the contrastive arm, for a fair comparison.
"""
import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from data import eval_transform, simclr_augment
from diagnostics import run_diagnostics
from model import ResNet18Cifar

# ---------------- Config (matched to train_contrastive.py) ----------------
DATA_ROOT = "./data"
SUBSET_SIZE = None
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-4
N_TEST = 2000          # eval pool size; individual tests subsample N_TEST_SAMPLES from this
N_TEST_SAMPLES = 250   # samples per individual per-coordinate/projection test
N_PROJECTIONS = 100
N_VAL = 500
NUM_WORKERS = 2
SEED = 0
RUN_PER_EPOCH_DIAGNOSTICS = False
PER_EPOCH_DIAGNOSTICS_PATH = "gaussianity_per_epoch_supervised.csv"


torch.manual_seed(SEED)
np.random.seed(SEED)


class LabeledAugmentedDataset(torch.utils.data.Dataset):
    """Wrap a CIFAR-10 dataset of raw PIL images and return one augmented view
    plus the original class label for supervised training."""

    def __init__(self, base_dataset, transform):
        self.base = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        return self.transform(img), label


def build_train_loader():
    """Build the augmented CIFAR-10 train loader used for cross-entropy."""
    base = CIFAR10(root=DATA_ROOT, train=True, download=True)

    if SUBSET_SIZE is not None:
        idx = np.random.RandomState(SEED).choice(
            len(base),
            size=SUBSET_SIZE,
            replace=False,
        )
        base = Subset(base, idx)

    ds = LabeledAugmentedDataset(base, simclr_augment())

    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )


def build_val_loader():
    """Build a small held-out validation loader for per-epoch progress checks."""
    base = CIFAR10(root=DATA_ROOT, train=False, download=True)
    idx = np.random.RandomState(SEED + 2).choice(len(base), size=N_VAL, replace=False)
    base = Subset(base, idx)
    ds = LabeledAugmentedDataset(base, simclr_augment())
    return DataLoader(
        ds,
        batch_size=min(BATCH_SIZE, N_VAL),
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )


@torch.no_grad()
def validate(model, head, device, val_loader):
    """Compute validation cross-entropy loss and top-1 accuracy."""
    model.eval()
    head.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        logits = head(model(x))
        loss = F.cross_entropy(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_n += x.size(0)
    model.train()
    head.train()
    return total_loss / total_n, total_correct / total_n


def save_epoch_stats(epoch, space, stats, path=PER_EPOCH_DIAGNOSTICS_PATH):
    """Append Gaussianity diagnostics for one epoch/representation to CSV."""
    file_exists = os.path.exists(path)

    with open(path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "epoch",
                "space",
                "cv",
                "coord_ad_score",
                "coord_ad_compliance_pct",
                "coord_dp_avg_p",
                "coord_dp_compliance_pct",
                "proj_ad_score",
                "proj_ad_compliance_pct",
                "proj_dp_avg_p",
                "proj_dp_compliance_pct",
            ])

        writer.writerow([
            epoch,
            space,
            stats["cv"],
            stats["ad_score"],
            stats["ad_compliance_pct"],
            stats["dp_score"],
            stats["dp_compliance_pct"],
            stats["proj_ad_score"],
            stats["proj_ad_compliance_pct"],
            stats["proj_dp_score"],
            stats["proj_dp_compliance_pct"],
        ])


@torch.no_grad()
def evaluate(model, device):
    """Run Gaussianity diagnostics on the supervised backbone feature space."""
    test_base = CIFAR10(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=eval_transform(),
    )

    if N_TEST is not None and N_TEST < len(test_base):
        idx = np.random.RandomState(SEED + 1).choice(
            len(test_base),
            size=N_TEST,
            replace=False,
        )
        test_base = Subset(test_base, idx)

    loader = DataLoader(
        test_base,
        batch_size=256,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model.eval()
    feats = [model(x.to(device)).cpu().numpy() for x, _ in loader]
    z = np.concatenate(feats, axis=0)
    stats = run_diagnostics(
        z,
        "Supervised backbone (512-dim, pre-classifier)",
        N_TEST_SAMPLES,
        N_PROJECTIONS,
    )
    model.train()
    return stats


def train():
    """Train the ResNet-18 backbone and linear classifier with cross-entropy."""
    loader = build_train_loader()
    val_loader = build_val_loader()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet18Cifar().to(device)
    head = nn.Linear(512, 10).to(device)
    opt = torch.optim.Adam(
        list(model.parameters()) + list(head.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    for epoch in range(1, EPOCHS + 1):
        epoch_loss, n_correct, n_seen, n_batches = 0.0, 0, 0, 0
        t0 = time.time()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = head(model(x))
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_correct += (logits.argmax(dim=1) == y).sum().item()
            n_seen += x.size(0)
            n_batches += 1

        val_loss, val_acc = validate(model, head, device, val_loader)
        print(
            f"epoch {epoch:3d}/{EPOCHS}  train_loss={epoch_loss / n_batches:.4f}  "
            f"train_acc={n_correct / n_seen * 100:.1f}%  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc * 100:.1f}%  ({time.time() - t0:.1f}s)"
        )

        if RUN_PER_EPOCH_DIAGNOSTICS:
            stats = evaluate(model, device)
            save_epoch_stats(epoch, "supervised_backbone", stats)

    return model, device


def main():
    """Run supervised training, save the checkpoint, and print final diagnostics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"device={device}  subset_size={SUBSET_SIZE}  epochs={EPOCHS}  "
        f"batch_size={BATCH_SIZE}  per_epoch_diagnostics={RUN_PER_EPOCH_DIAGNOSTICS}"
    )
    model, device = train()
    torch.save({"model": model.state_dict()}, "supervised_checkpoint.pt")
    evaluate(model, device)


if __name__ == "__main__":
    main()
