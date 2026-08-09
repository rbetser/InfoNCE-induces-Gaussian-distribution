"""Shared Gaussianity diagnostics: Anderson-Darling and D'Agostino-Pearson
normality tests, applied per-coordinate and via random 1-D projections."""
import numpy as np
from scipy import stats

AD_THRESHOLD = 0.752  # Anderson-Darling: fail to reject normality below this
DP_ALPHA = 0.05        # D'Agostino-Pearson: fail to reject normality above this p


def ad_stat(sample):
    """Anderson-Darling normality statistic for a 1-D sample. NaN if the sample
    has zero variance (e.g. a dead ReLU unit) - not testable, not Gaussian."""
    if np.std(sample) == 0:
        return float("nan")
    return float(stats.anderson(sample, dist="norm").statistic)


def dp_stat(sample):
    """D'Agostino-Pearson normality test for a 1-D sample: (statistic, p-value).
    NaN if the sample has zero variance (e.g. a dead ReLU unit)."""
    if np.std(sample) == 0:
        return float("nan"), float("nan")
    stat, p = stats.normaltest(sample)
    return float(stat), float(p)


def run_diagnostics(z, label, n_test_samples, n_projections, log_fn=print):
    """CV + per-coordinate/random-projection AD & DP diagnostics for a feature
    matrix z (n_samples, d). Each individual test subsamples n_test_samples
    rows from the eval pool. log_fn defaults to print; pass a caller-specific
    logger to route output to the right log file."""
    n, d = z.shape
    norms = np.linalg.norm(z, axis=1)
    cv = norms.std() / norms.mean()

    ad_stats, dp_ps = [], []
    for k in range(d):
        idx = np.random.choice(n, size=min(n_test_samples, n), replace=False)
        sample = z[idx, k]
        ad_stats.append(ad_stat(sample))
        dp_ps.append(dp_stat(sample)[1])
    ad_stats, dp_ps = np.array(ad_stats), np.array(dp_ps)
    n_degenerate_coord = int(np.isnan(ad_stats).sum())
    # NaN (zero-variance/degenerate) entries fail both comparisons automatically -
    # they are not Gaussian, they're untestable, which should count against pass rate.
    ad_pass_frac = (ad_stats < AD_THRESHOLD).mean()
    dp_pass_frac = (dp_ps > DP_ALPHA).mean()

    # random 1-D projections: same tests, applied off-axis instead of per-coordinate
    proj_ad, proj_dp_p = [], []
    for _ in range(n_projections):
        w = np.random.randn(d)
        w /= np.linalg.norm(w)
        idx = np.random.choice(n, size=min(n_test_samples, n), replace=False)
        proj = z[idx] @ w
        proj_ad.append(ad_stat(proj))
        proj_dp_p.append(dp_stat(proj)[1])
    proj_ad, proj_dp_p = np.array(proj_ad), np.array(proj_dp_p)
    n_degenerate_proj = int(np.isnan(proj_ad).sum())

    degenerate_note = (f"   [{n_degenerate_coord} degenerate/zero-variance coords excluded from avg]"
                        if n_degenerate_coord else "")
    log_fn(f"\n[{label}]  d={d}  CV (norm coefficient of variation) = {cv:.4f}")
    log_fn(f"  Per-coordinate: AD avg={np.nanmean(ad_stats):.4f} pass={ad_pass_frac * 100:.1f}%   "
           f"DP avg p={np.nanmean(dp_ps):.4f} pass={dp_pass_frac * 100:.1f}%{degenerate_note}")
    log_fn(f"  Random-projection: AD avg={np.nanmean(proj_ad):.4f} pass={(proj_ad < AD_THRESHOLD).mean() * 100:.1f}%   "
           f"DP avg p={np.nanmean(proj_dp_p):.4f} pass={(proj_dp_p > DP_ALPHA).mean() * 100:.1f}%"
           + (f"   [{n_degenerate_proj} degenerate projections excluded from avg]" if n_degenerate_proj else ""))
    
    return {
        "ad_score": float(np.nanmean(ad_stats)),
        "ad_compliance_pct": float(ad_pass_frac * 100),
    
        "dp_score": float(np.nanmean(dp_ps)),
        "dp_compliance_pct": float(dp_pass_frac * 100),
    
        "proj_ad_score": float(np.nanmean(proj_ad)),
        "proj_ad_compliance_pct": float((proj_ad < AD_THRESHOLD).mean() * 100),
    
        "proj_dp_score": float(np.nanmean(proj_dp_p)),
        "proj_dp_compliance_pct": float((proj_dp_p > DP_ALPHA).mean() * 100),
    
        "cv": float(cv),
    }