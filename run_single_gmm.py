"""
Trains a linear encoder with InfoNCE on synthetic Gaussian-mixture data and
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
D_DATA = 1024       # input dimension
D_EMBED = 256       # output representation dimension
BATCH_SIZE = 256
EPOCHS = 100        # real epochs over a fixed training pool
N_TRAIN = 20000
N_TEST = 5000

N_COMPONENTS = 25
COMPONENT_STD = 1.0    # shared isotropic std of each mixture component
MEAN_SCALE = 0.05      # std of the (fixed, random) component means - see module docstring

TEMPERATURE = 0.1      # InfoNCE temperature
LR = 1e-3

AD_THRESHOLD = 0.752  # Anderson-Darling: fail to reject normality below this
DP_ALPHA = 0.05       # D'Agostino-Pearson: fail to reject normality above this p
N_PROJECTIONS = 100
N_PROJ_SAMPLES = 250
SEED = 0

torch.manual_seed(SEED)
np.random.seed(SEED)

# component means are drawn once and fixed for the whole run (paper's setup)
COMPONENT_MEANS = np.random.randn(N_COMPONENTS, D_DATA).astype(np.float32) * MEAN_SCALE


def sample_from_components(comp_idx):
    """One i.i.d. draw per given component index (array of ints)."""
    means = COMPONENT_MEANS[comp_idx]
    x = means + COMPONENT_STD * np.random.randn(*means.shape).astype(np.float32)
    return torch.from_numpy(x.astype(np.float32))


def sample_mixture(n):
    """Draw n samples from the mixture, each from a uniformly chosen component."""
    comp_idx = np.random.randint(0, N_COMPONENTS, size=n)
    return sample_from_components(comp_idx)


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
    """Train a linear InfoNCE encoder over EPOCHS passes of a fixed component-assignment pool."""
    # fixed pool of "base examples", each identified only by its component id
    train_components = np.random.randint(0, N_COMPONENTS, size=N_TRAIN)
    enc = nn.Linear(D_DATA, D_EMBED)
    opt = torch.optim.Adam(enc.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        perm = np.random.permutation(N_TRAIN)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N_TRAIN - BATCH_SIZE + 1, BATCH_SIZE):
            comps = train_components[perm[start:start + BATCH_SIZE]]
            x1 = sample_from_components(comps)
            x2 = sample_from_components(comps)  # independent second view, same component
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
    x_test = sample_mixture(N_TEST)
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

    print(f"\n=== Config: GMM input, d={D_EMBED}, N={BATCH_SIZE}, epochs={EPOCHS} ===")
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
    print(f"Training linear InfoNCE encoder on {N_COMPONENTS}-component Gaussian mixture: "
          f"d_data={D_DATA} -> d={D_EMBED}, batch_size={BATCH_SIZE}, epochs={EPOCHS}")
    enc = train()
    evaluate(enc)


if __name__ == "__main__":
    main()
