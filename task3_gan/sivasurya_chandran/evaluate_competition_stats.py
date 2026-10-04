"""
Competition-stats scorer (the scorer used for the v1-v3 Kaggle entries): FID / MiFID of a folder of
generated Monet-style images against the competition's own `real_stats.npz` (300 real Monet paintings).
The instructor allows any evaluation mechanism; `evaluate_local.py` is the instructor's
Part3_Evaluation_Script.ipynb (both directions, first 300 images by filename) for comparison.

Metric definitions follow the competition's Evaluation page. Features are Inception-v3
(torchvision ImageNet weights, fc removed -> 2048-d pool features) on images resized to
299x299 with ImageNet normalisation. This pipeline reproduces the instructor's
`real_stats.npz` exactly (feats_real for all 300 Monet paintings, cosine 1.0000).
  FID   = ||mu_r - mu_g||^2 + Tr(S_r + S_g - 2 (S_r S_g)^(1/2))   vs mu_real / sigma_real
  MiFID = mean_i (1 - cos(f_g_i, f_r_i)) after subsampling both sets to equal size
          (course definition: mean cosine distance, not Kaggle's nearest-neighbour MiFID)
  leaderboard score = (FID + MiFID) / 2  (lower is better)

The images must be the direct output of your own G_AB (src/translate.py); no edits.

  python evaluate_competition_stats.py --gen outputs/pred_B2A --out submission.csv   # all 7,038 translated photos
  python evaluate_competition_stats.py --gen ../data/monet_jpg --no_csv                # sanity check: FID ~ 0
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torchvision
from PIL import Image
from scipy import linalg
from torchvision import transforms as T

HERE = os.path.dirname(os.path.abspath(__file__))
TRANSFORM = T.Compose([T.Resize((299, 299)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def inception(dev):
    net = torchvision.models.inception_v3(weights="IMAGENET1K_V1", aux_logits=True)
    net.fc = torch.nn.Identity()
    return net.eval().to(dev)


@torch.no_grad()
def features(files, dev, batch_size=50, net=None) -> np.ndarray:
    net = net or inception(dev)
    out = []
    for i in range(0, len(files), batch_size):
        x = torch.stack([TRANSFORM(Image.open(f).convert("RGB")) for f in files[i : i + batch_size]]).to(dev)
        out.append(net(x).cpu().double())
    return torch.cat(out).numpy()


def fid(mu_r, sigma_r, feats_g) -> float:
    mu_g, sigma_g = feats_g.mean(0), np.cov(feats_g, rowvar=False)
    covmean, _ = linalg.sqrtm(sigma_r.dot(sigma_g), disp=False)
    if not np.isfinite(covmean).all():
        eps = np.eye(sigma_r.shape[0]) * 1e-6
        covmean = linalg.sqrtm((sigma_r + eps).dot(sigma_g + eps))
    return float(((mu_r - mu_g) ** 2).sum() + np.trace(sigma_r) + np.trace(sigma_g) - 2 * np.trace(covmean.real))


def mifid(feats_r, feats_g, seed) -> tuple:
    """Course MiFID: subsample both sets to the same size n, pair them, average cosine distance.
    Also returns the all-pairs mean, which the paired estimate approximates."""
    rng = np.random.default_rng(seed)
    n = min(len(feats_r), len(feats_g))
    r = feats_r[rng.choice(len(feats_r), n, replace=False)]
    g = feats_g[rng.choice(len(feats_g), n, replace=False)]
    unit = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)  # noqa: E731
    with np.errstate(all="ignore"):  # macOS Accelerate BLAS emits spurious matmul warnings for large float64 products
        paired = float(np.mean(1 - np.sum(unit(g) * unit(r), axis=1)))
        all_pairs = float(np.mean(1 - unit(feats_g) @ unit(feats_r).T))
    return paired, all_pairs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="outputs/pred_B2A", help="folder of generated Monet-style images")
    ap.add_argument("--stats", default=os.path.join(HERE, "..", "data", "competition_files", "real_stats.npz"))
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--id", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_csv", action="store_true")
    a = ap.parse_args()

    ref = np.load(a.stats)
    files = sorted(f for e in ("*.jpg", "*.jpeg", "*.png") for f in glob.glob(os.path.join(a.gen, e)))
    assert files, f"no images in {a.gen}"
    dev = device()
    print(f"{len(files)} generated images | device {dev}")
    fg = features(files, dev)
    f_val = fid(ref["mu_real"].astype(np.float64), ref["sigma_real"].astype(np.float64), fg)
    m_val, m_all = mifid(ref["feats_real"].astype(np.float64), fg, a.seed)
    result = {"n_generated": len(files), "FID": f_val, "MiFID": m_val, "MiFID_all_pairs": m_all, "score_mean": (f_val + m_val) / 2}
    print(json.dumps(result, indent=2))
    if not a.no_csv:
        with open(a.out, "w") as f:
            f.write(f"ID,FID,MiFID\n{a.id},{f_val:.3f},{m_val:.3f}\n")
        with open(os.path.splitext(a.out)[0] + "_details.json", "w") as f:
            json.dump({**result, "gen_dir": a.gen, "stats": os.path.relpath(a.stats, HERE), "seed": a.seed}, f, indent=2)
        print(open(a.out).read())
