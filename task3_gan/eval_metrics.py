"""
Shared Task 3 metric functions — every member uses these so numbers are comparable.

All distribution metrics (FID, KID, precision/recall, density/coverage) are
computed from the same 2048-d Inception-v3 pool features (the standard
FID network, via torchmetrics / torch-fidelity).
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy import linalg

_INCEPTION = None
_LPIPS = None


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """(N,3,H,W) in [-1,1] -> uint8 [0,255]."""
    return ((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8)


@torch.no_grad()
def inception_features(images: torch.Tensor, device: str = "cpu", batch_size: int = 32) -> np.ndarray:
    global _INCEPTION
    if _INCEPTION is None:
        from torchmetrics.image.fid import NoTrainInceptionV3

        _INCEPTION = NoTrainInceptionV3(name="inception-v3-compat", features_list=["2048"]).eval()
    net = _INCEPTION.to(device)
    feats = [net(to_uint8(images[i : i + batch_size]).to(device)).reshape(-1, 2048).cpu().double() for i in range(0, len(images), batch_size)]
    return torch.cat(feats).numpy()


def fid(real_f: np.ndarray, fake_f: np.ndarray) -> float:
    mu1, mu2 = real_f.mean(0), fake_f.mean(0)
    s1, s2 = np.cov(real_f, rowvar=False), np.cov(fake_f, rowvar=False)
    covmean, _ = linalg.sqrtm(s1.dot(s2), disp=False)
    if not np.isfinite(covmean).all():
        eps = np.eye(s1.shape[0]) * 1e-6
        covmean = linalg.sqrtm((s1 + eps).dot(s2 + eps))
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def kid(real_f: np.ndarray, fake_f: np.ndarray, n_subsets: int = 100, subset_size: int = 1000, seed: int = 0):
    """Unbiased MMD^2 with the cubic polynomial kernel (Binkowski et al. 2018). Returns (mean, std)."""
    rng = np.random.default_rng(seed)
    d = real_f.shape[1]
    m = min(subset_size, len(real_f), len(fake_f))
    vals = []
    for _ in range(n_subsets):
        x = real_f[rng.choice(len(real_f), m, replace=False)]
        y = fake_f[rng.choice(len(fake_f), m, replace=False)]
        kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
        mmd = (kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean()
        vals.append(mmd)
    return float(np.mean(vals)), float(np.std(vals))


def _pairwise(a, b):
    a2, b2 = (a ** 2).sum(1)[:, None], (b ** 2).sum(1)[None, :]
    return np.sqrt(np.maximum(a2 + b2 - 2 * a @ b.T, 0))


def prdc(real_f: np.ndarray, fake_f: np.ndarray, k: int = 5) -> dict:
    """Improved precision/recall (Kynkaanniemi et al. 2019) and density/coverage (Naeem et al. 2020)."""
    k = min(k, len(real_f) - 1, len(fake_f) - 1)
    d_rr, d_ff, d_rf = _pairwise(real_f, real_f), _pairwise(fake_f, fake_f), _pairwise(real_f, fake_f)
    r_real = np.sort(d_rr, axis=1)[:, k]  # column 0 is the point itself
    r_fake = np.sort(d_ff, axis=1)[:, k]
    inside_real = d_rf <= r_real[:, None]  # (real, fake): fake inside real j's ball
    return {
        "precision": float(inside_real.any(axis=0).mean()),
        "recall": float((d_rf <= r_fake[None, :]).any(axis=1).mean()),
        "density": float(inside_real.sum() / (k * len(fake_f))),
        "coverage": float((d_rf.min(axis=1) <= r_real).mean()),
        "k": int(k),
    }


def l1_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    """Mean absolute error in [-1,1] pixel space."""
    return float((a - b).abs().mean())


@torch.no_grad()
def lpips_distance(a: torch.Tensor, b: torch.Tensor, device: str = "cpu", batch_size: int = 16) -> float:
    global _LPIPS
    if _LPIPS is None:
        import lpips

        _LPIPS = lpips.LPIPS(net="alex", verbose=False).eval()
    net = _LPIPS.to(device)
    ds = [net(a[i : i + batch_size].to(device), b[i : i + batch_size].to(device)).flatten().cpu() for i in range(0, len(a), batch_size)]
    return float(torch.cat(ds).mean())


def content_cosine(feat_in: np.ndarray, feat_out: np.ndarray) -> float:
    """Mean cosine similarity between Inception features of each input and its own translation."""
    a = torch.from_numpy(feat_in)
    b = torch.from_numpy(feat_out)
    return float(F.cosine_similarity(a, b, dim=1).mean())


def cohens_kappa(r1, r2, weights=None) -> float:
    from sklearn.metrics import cohen_kappa_score

    return float(cohen_kappa_score(r1, r2, weights=weights))
