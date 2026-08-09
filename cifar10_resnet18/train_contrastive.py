"""
Contrastive (InfoNCE) training of a CIFAR-adapted ResNet-18 on CIFAR-10,
reproducing the contrastive arm of Betser, Gofer, Levi, Gilboa, "InfoNCE
Induces Gaussian Distribution" (ICLR 2026, arXiv:2602.24012), Appendix E.1
CIFAR-10 experiments, Table 1's "Contrastive" column.
"""
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10

from data import ContrastivePairDataset, eval_transform, simclr_augment
from diagnostics import run_diagnostics
from model import ProjectionHead, ResNet18Cifar

# ---------------- Config ----------------
DATA_ROOT = "./data"
SUBSET_SIZE = None
EPOCHS = 100
BATCH_SIZE = 256
TEMPERATURE = 0.1
LR = 1e-3
WEIGHT_DECAY = 1e-4
PROJECTION_DIM = 128  # SimCLR-style projector output; loss is computed here, then discarded
N_TEST = 2000         # eval pool size; individual tests subsample N_TEST_SAMPLES from this
N_VAL = 500           # held-out CIFAR test images for per-epoch progress checks
NUM_WORKERS = 2

N_TEST_SAMPLES = 250   # samples per individual per-coordinate/projection test
N_PROJECTIONS = 100
SEED = 0
RUN_PER_EPOCH_DIAGNOSTICS = False
PER_EPOCH_DIAGNOSTICS_PATH = "gaussianity_per_epoch_contrastive.csv"


torch.manual_seed(SEED)
np.random.seed(SEED)


def info_nce_loss(z1, z2, temperature=TEMPERATURE):
    """Symmetric InfoNCE loss on L2-normalized representations."""
    u, v = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    logits = u @ v.t() / temperature
    labels = torch.arange(u.size(0), device=u.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def build_train_loader():
    """Build the augmented CIFAR-10 train loader used for contrastive pairs."""
    base = CIFAR10(root=DATA_ROOT, train=True, download=True)
    if SUBSET_SIZE is not None:
        idx = np.random.RandomState(SEED).choice(len(base), size=SUBSET_SIZE, replace=False)
        base = Subset(base, idx)
    ds = ContrastivePairDataset(base, simclr_augment())
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
        pin_memory=False,
    )


def build_val_loader():
    """Build a held-out validation loader used only for quick progress checks."""
    base = CIFAR10(root=DATA_ROOT, train=False, download=True)
    idx = np.random.RandomState(SEED + 2).choice(len(base), size=N_VAL, replace=False)
    base = Subset(base, idx)
    ds = ContrastivePairDataset(base, simclr_augment())
    return DataLoader(
        ds,
        batch_size=min(BATCH_SIZE, N_VAL),
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )


def build_eval_loader():
    """Build the plain CIFAR-10 evaluation pool for Gaussianity diagnostics."""
    test_base = CIFAR10(root=DATA_ROOT, train=False, download=True, transform=eval_transform())
    if N_TEST is not None and N_TEST < len(test_base):
        idx = np.random.RandomState(SEED + 1).choice(len(test_base), size=N_TEST, replace=False)
        test_base = Subset(test_base, idx)
    return DataLoader(test_base, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)


@torch.no_grad()
def compute_features(model, projector, device, eval_loader):
    """Return backbone and projector features for the evaluation pool."""
    model.eval()
    projector.eval()
    backbone_feats, proj_feats = [], []
    for x, _ in eval_loader:
        h = model(x.to(device))
        backbone_feats.append(h.cpu().numpy())
        proj_feats.append(projector(h).cpu().numpy())
    model.train()
    projector.train()
    return np.concatenate(backbone_feats, axis=0), np.concatenate(proj_feats, axis=0)


@torch.no_grad()
def validate(model, projector, device, val_loader):
    """Compute validation InfoNCE loss and positive-pair retrieval accuracy."""
    model.eval()
    projector.eval()
    total_loss, total_correct, total_n = 0.0, 0, 0
    for x1, x2 in val_loader:
        x1, x2 = x1.to(device), x2.to(device)
        u = F.normalize(projector(model(x1)), dim=1)
        v = F.normalize(projector(model(x2)), dim=1)
        logits = u @ v.t() / TEMPERATURE
        labels = torch.arange(u.size(0), device=device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
        total_loss += loss.item() * u.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_n += u.size(0)
    model.train()
    projector.train()
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
def evaluate(model, projector, device, eval_loader):
    """Run Gaussianity diagnostics on both backbone and projector spaces."""
    z_backbone, z_proj = compute_features(model, projector, device, eval_loader)
    backbone_stats = run_diagnostics(
        z_backbone,
        "Contrastive backbone (512-dim, pre-projector)",
        N_TEST_SAMPLES,
        N_PROJECTIONS,
    )
    projector_stats = run_diagnostics(
        z_proj,
        "Projector output (128-dim, InfoNCE space)",
        N_TEST_SAMPLES,
        N_PROJECTIONS,
    )
    return backbone_stats, projector_stats


def train():
    """Train the ResNet-18 backbone and projection head with InfoNCE."""
    loader = build_train_loader()
    val_loader = build_val_loader()
    eval_loader = build_eval_loader()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResNet18Cifar().to(device)
    projector = ProjectionHead(out_dim=PROJECTION_DIM).to(device)
    opt = torch.optim.Adam(
        list(model.parameters()) + list(projector.parameters()),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    for epoch in range(1, EPOCHS + 1):
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()
        for x1, x2 in loader:
            x1, x2 = x1.to(device), x2.to(device)
            loss = info_nce_loss(projector(model(x1)), projector(model(x2)))
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1

        val_loss, val_acc = validate(model, projector, device, val_loader)
        print(
            f"epoch {epoch:3d}/{EPOCHS}  train_loss={epoch_loss / n_batches:.4f}  "
            f"val_loss={val_loss:.4f}  val_retrieval_acc={val_acc * 100:.1f}%  "
            f"({time.time() - t0:.1f}s)"
        )

        if RUN_PER_EPOCH_DIAGNOSTICS:
            backbone_stats, projector_stats = evaluate(model, projector, device, eval_loader)
            save_epoch_stats(epoch, "contrastive_backbone", backbone_stats)
            save_epoch_stats(epoch, "projector", projector_stats)

    return model, projector, device, eval_loader


def main():
    """Run contrastive training, save the checkpoint, and print final diagnostics."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"device={device}  subset_size={SUBSET_SIZE}  epochs={EPOCHS}  "
        f"batch_size={BATCH_SIZE}  per_epoch_diagnostics={RUN_PER_EPOCH_DIAGNOSTICS}"
    )
    model, projector, device, eval_loader = train()
    torch.save(
        {"model": model.state_dict(), "projector": projector.state_dict()},
        "contrastive_checkpoint.pt",
    )
    evaluate(model, projector, device, eval_loader)


if __name__ == "__main__":
    main()
