"""
Trains a linear encoder with InfoNCE on synthetic Laplace(0, 1) data and
checks whether the resulting representations look Gaussian.

Gaussianity is checked per output coordinate and via random 1-D projections,
using the Anderson-Darling (AD) and D'Agostino-Pearson (DP) normality tests.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

# ---------------- Config ----------------
D_DATA = 1024      # input (Laplace) dimension
D_EMBED = 256       # output representation dimension
BATCH_SIZE = 256
EPOCHS = 100        # real epochs over a fixed training pool
N_TRAIN = 20000
N_TEST = 5000

TEMPERATURE = 0.1   # InfoNCE temperature
A_CORR = 0.6        # positive-pair correlation (paper Eq. 53)
NOISE_STD = 0.2     # additive Gaussian noise on each view
DROPOUT_P = 0.1     # feature dropout probability on each view
SCALE_STD = 0.1     # std of the log-normal multiplicative scaling
LR = 1e-3

AD_THRESHOLD = 0.752  # Anderson-Darling: fail to reject normality below this
DP_ALPHA = 0.05       # D'Agostino-Pearson: fail to reject normality above this p
N_PROJECTIONS = 100
N_PROJ_SAMPLES = 250
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)


def sample_laplace(n, dim=D_DATA):
    """Draw n i.i.d. Laplace(0, 1) vectors."""
    return torch.from_numpy(np.random.laplace(0.0, 1.0, size=(n, dim)).astype(np.float32))


def augment_pair(x, A=A_CORR, noise_std=NOISE_STD, dropout_p=DROPOUT_P, scale_std=SCALE_STD):
    """Build two correlated 'views' of x: Gaussian mixing at correlation A,
    then independent additive noise, feature dropout, and multiplicative scaling."""
    mix = float(np.sqrt(1 - A ** 2))
    x1 = A * x + mix * torch.randn_like(x)
    x2 = A * x + mix * torch.randn_like(x)
    x1 = x1 + noise_std * torch.randn_like(x1)
    x2 = x2 + noise_std * torch.randn_like(x2)
    keep1 = (torch.rand_like(x1) >= dropout_p).float()
    keep2 = (torch.rand_like(x2) >= dropout_p).float()
    x1, x2 = x1 * keep1, x2 * keep2
    scale1 = torch.exp(scale_std * torch.randn(x.size(0), 1))
    scale2 = torch.exp(scale_std * torch.randn(x.size(0), 1))
    return x1 * scale1, x2 * scale2


def info_nce_loss(z1, z2, temperature=TEMPERATURE):
    """Symmetric InfoNCE loss on L2-normalized representations."""
    u, v = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    logits = u @ v.t() / temperature
    labels = torch.arange(u.size(0))
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def ad_stat(sample):
    """Anderson-Darling normality statistic for a 1-D sample."""
    return float(stats.anderson(sample, dist="norm").statistic)


def dp_stat(sample):
    """D'Agostino-Pearson normality test for a 1-D sample: (statistic, p-value)."""
    stat, p = stats.normaltest(sample)
    return float(stat), float(p)


def train():
    """Train a linear InfoNCE encoder over EPOCHS passes of a fixed Laplace training pool."""
    x_train = sample_laplace(N_TRAIN)
    enc = nn.Linear(D_DATA, D_EMBED)
    opt = torch.optim.Adam(enc.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        perm = torch.randperm(N_TRAIN)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N_TRAIN - BATCH_SIZE + 1, BATCH_SIZE):
            idx = perm[start:start + BATCH_SIZE]
            x1, x2 = augment_pair(x_train[idx])
            loss = info_nce_loss(enc(x1), enc(x2))
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch:3d}/{EPOCHS}  InfoNCE loss = {epoch_loss / n_batches:.4f}")

    return enc


@torch.no_grad()
def evaluate(enc):
    """Compute norm CV and per-coordinate / random-projection AD & DP diagnostics."""
    x_test = sample_laplace(N_TEST)
    z = enc(x_test).numpy()  # unnormalized representations, as in the paper's Table 1

    norms = np.linalg.norm(z, axis=1)
    cv = norms.std() / norms.mean()

    ad_stats = np.array([ad_stat(z[:, k]) for k in range(D_EMBED)])
    dp_results = [dp_stat(z[:, k]) for k in range(D_EMBED)]
    dp_ps = np.array([p for _, p in dp_results])

    ad_pass_frac = (ad_stats < AD_THRESHOLD).mean()
    dp_pass_frac = (dp_ps > DP_ALPHA).mean()

    # random 1-D projections: same tests, applied off-axis instead of per-coordinate
    proj_ad, proj_dp_p = [], []
    for _ in range(N_PROJECTIONS):
        w = np.random.randn(D_EMBED)
        w /= np.linalg.norm(w)
        idx = np.random.choice(N_TEST, N_PROJ_SAMPLES, replace=False)
        proj = z[idx] @ w
        proj_ad.append(ad_stat(proj))
        proj_dp_p.append(dp_stat(proj)[1])
    proj_ad, proj_dp_p = np.array(proj_ad), np.array(proj_dp_p)
    proj_ad_pass_frac = (proj_ad < AD_THRESHOLD).mean()
    proj_dp_pass_frac = (proj_dp_p > DP_ALPHA).mean()

    print(f"\n=== Config: d={D_EMBED}, N={BATCH_SIZE}, epochs={EPOCHS} ===")
    print(f"CV (norm coefficient of variation): {cv:.4f}")
    print()
    print("Per-coordinate diagnostics (n_coords={}, n_samples={}):".format(D_EMBED, N_TEST))
    print(f"  AD avg statistic = {ad_stats.mean():.4f}   AD pass rate = {ad_pass_frac*100:.1f}%  (<{AD_THRESHOLD})")
    print(f"  DP avg p-value   = {dp_ps.mean():.4f}   DP pass rate = {dp_pass_frac*100:.1f}%  (>{DP_ALPHA})")
    print()
    print(f"Random-projection diagnostics (n_projections={N_PROJECTIONS}, n_samples={N_PROJ_SAMPLES}):")
    print(f"  AD avg statistic = {proj_ad.mean():.4f}   AD pass rate = {proj_ad_pass_frac*100:.1f}%")
    print(f"  DP avg p-value   = {proj_dp_p.mean():.4f}   DP pass rate = {proj_dp_pass_frac*100:.1f}%")


def main():
    """Train the encoder and print its Gaussianity diagnostics."""
    print(f"Training linear InfoNCE encoder: d_data={D_DATA} -> d={D_EMBED}, "
          f"batch_size={BATCH_SIZE}, epochs={EPOCHS}")
    enc = train()
    evaluate(enc)


if __name__ == "__main__":
    main()
