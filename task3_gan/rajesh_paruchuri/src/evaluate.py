"""Task 3 evaluation (Rajesh Paruchuri). Measurement only: pretrained Inception-v3 / VGG16 / LPIPS networks
score images that my own CycleGAN already produced; they never generate or modify a submitted image.

Metrics, both directions (photo->Monet = B2A, Monet->photo = A2B):
  FID, KID (unbiased polynomial-kernel MMD, subset bootstrap), improved precision/recall (k-NN, Kynkaanniemi 2019),
  density/coverage (Naeem 2020), local MiFID-style memorisation distance, cycle-reconstruction L1,
  LPIPS (input vs reconstruction and input vs translation), content cosine similarity (VGG16 relu4_3).
"""

import json
import os
import time
from pathlib import Path

import certifi
import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("SSL_CERT_FILE", certifi.where())  # python.org builds on macOS lack root certs

import cyclegan as cg  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# --------------------------------------------------------------------------
# feature extractors
# --------------------------------------------------------------------------
def inception(device):
    import torchvision.models as tvm
    m = tvm.inception_v3(weights=tvm.Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
    m.fc = torch.nn.Identity()  # 2048-d pool features
    return m.eval().to(device)


def vgg_relu4_3(device):
    import torchvision.models as tvm
    v = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features[:23]  # up to relu4_3
    return v.eval().to(device)


def load_batch(files, size=256):
    """uint8 files -> float [0,1] tensor."""
    return torch.stack([(cg.to_tensor(cg.load_image(p, size)) + 1) / 2 for p in files])


@torch.no_grad()
def inception_features(files, model, device, batch=32):
    feats = []
    for s in range(0, len(files), batch):
        x = load_batch(files[s:s + batch]).to(device)
        x = F.interpolate(x, size=(299, 299), mode="bilinear", align_corners=False)
        x = (x - MEAN.to(device)) / STD.to(device)
        feats.append(model(x).float().cpu())
    return torch.cat(feats).numpy().astype(np.float64)


# --------------------------------------------------------------------------
# distribution metrics
# --------------------------------------------------------------------------
def fid(f1, f2):
    from scipy import linalg
    mu1, mu2 = f1.mean(0), f2.mean(0)
    s1, s2 = np.cov(f1, rowvar=False), np.cov(f2, rowvar=False)
    covmean, _ = linalg.sqrtm(s1 @ s2, disp=False)
    covmean = covmean.real
    return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def kid(f_real, f_fake, n_subsets=50, subset_size=100, seed=0):
    rng = np.random.default_rng(seed)
    d = f_real.shape[1]
    m = min(subset_size, len(f_real), len(f_fake))
    vals = []
    for _ in range(n_subsets):
        x = f_real[rng.choice(len(f_real), m, replace=False)]
        y = f_fake[rng.choice(len(f_fake), m, replace=False)]
        kxx, kyy, kxy = ((x @ x.T) / d + 1) ** 3, ((y @ y.T) / d + 1) ** 3, ((x @ y.T) / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1))
                    - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def _dist(a, b):
    """Euclidean distance matrix via |a|^2 + |b|^2 - 2ab (no (n, m, d) intermediate)."""
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T
    return np.sqrt(np.maximum(d2, 0))


def _knn_radii(f, k):
    return np.sort(_dist(f, f), axis=1)[:, k]  # column 0 is the point itself


def precision_recall_density_coverage(f_real, f_fake, k=3, max_n=1000, seed=0):
    rng = np.random.default_rng(seed)
    r = f_real[rng.choice(len(f_real), min(max_n, len(f_real)), replace=False)]
    g = f_fake[rng.choice(len(f_fake), min(max_n, len(f_fake)), replace=False)]
    rr, rg = _knn_radii(r, k), _knn_radii(g, k)
    d_gr = _dist(g, r)  # (n_g, n_r)
    precision = float((d_gr <= rr[None, :]).any(1).mean())
    recall = float((d_gr.T <= rg[None, :]).any(1).mean())
    density = float((d_gr <= rr[None, :]).sum(1).mean() / k)
    coverage = float((d_gr.min(0) <= rr).mean())
    return {"precision": precision, "recall": recall, "density": density, "coverage": coverage,
            "n_real": len(r), "n_fake": len(g), "k": k}


def memorisation_distance(f_real, f_fake):
    """MiFID-style term: mean over fakes of min cosine distance to any real image (low = possible copying)."""
    a = f_fake / np.linalg.norm(f_fake, axis=1, keepdims=True)
    b = f_real / np.linalg.norm(f_real, axis=1, keepdims=True)
    return float((1 - a @ b.T).min(1).mean())


# --------------------------------------------------------------------------
# per-image content / cycle metrics
# --------------------------------------------------------------------------
@torch.no_grad()
def pairwise_metrics(G, H, files, device, lpips_model, vgg, batch=8):
    """For each input x: y = G(x), r = H(y). Returns per-image L1(x,r), LPIPS(x,r), LPIPS(x,y),
    cosine(VGG(x), VGG(y)), and a reference L1 between x and an unrelated image of the same domain."""
    out = {k: [] for k in ("cycle_l1", "cycle_lpips", "translation_lpips", "content_cosine", "translation_l1",
                           "unrelated_l1")}
    for s in range(0, len(files), batch):
        x = load_batch(files[s:s + batch]).to(device)            # [0,1]
        xs = x * 2 - 1                                          # [-1,1] model space
        y = G(xs)
        r = H(y)
        out["cycle_l1"] += ((r - xs).abs().mean([1, 2, 3]) / 2).cpu().tolist()   # in [0,1] pixel units
        out["translation_l1"] += ((y - xs).abs().mean([1, 2, 3]) / 2).cpu().tolist()
        out["unrelated_l1"] += ((xs.roll(1, 0) - xs).abs().mean([1, 2, 3]) / 2).cpu().tolist()
        out["cycle_lpips"] += lpips_model(r, xs).flatten().cpu().tolist()
        out["translation_lpips"] += lpips_model(y, xs).flatten().cpu().tolist()
        fx = vgg((x - MEAN.to(device)) / STD.to(device)).flatten(1)
        fy = vgg((((y + 1) / 2) - MEAN.to(device)) / STD.to(device)).flatten(1)
        out["content_cosine"] += F.cosine_similarity(fx, fy, dim=1).cpu().tolist()
    return out


def summarize(d):
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "n": len(v)} for k, v in d.items()}


def run(ckpt_path, cfg, paths, device, tag, max_pairwise=300, use_ema=True):
    import lpips
    root = cg.data_dir(cfg, paths)
    monet, photo = cg.list_images(root / "monet_jpg"), cg.list_images(root / "photo_jpg")
    if tag == "smoke":
        monet, photo = monet[:cfg["smoke_n_monet"] * 2], photo[:cfg["smoke_n_photo"] * 2]
    G_AB, G_BA, ck = cg.load_generators(ckpt_path, cfg, device, use_ema)
    # folder names expected by the course script Part3_Evaluation_Script.ipynb (and git-ignored for the full run)
    sfx = "" if tag == "full" else f"_{tag}"
    out_b2a, out_a2b = paths["out"] / f"pred_B2A{sfx}", paths["out"] / f"pred_A2B{sfx}"
    res = {"checkpoint": cg.rel(ckpt_path, paths["repo"]), "checkpoint_epoch": ck["epoch"], "generator": "EMA" if use_ema else "raw"}
    print("translating", len(photo), "photos -> Monet and", len(monet), "Monet -> photo at 256x256", flush=True)
    res["gen_images_per_sec_b2a"] = cg.translate_folder(G_BA, photo, out_b2a, device)
    res["gen_images_per_sec_a2b"] = cg.translate_folder(G_AB, monet, out_a2b, device)
    fake_monet, fake_photo = cg.list_images(out_b2a), cg.list_images(out_a2b)

    t0 = time.time()
    inc = inception(device)
    f_monet = inception_features(monet, inc, device)
    f_photo = inception_features(photo, inc, device)
    f_fake_monet = inception_features(fake_monet, inc, device)
    f_fake_photo = inception_features(fake_photo, inc, device)
    np.savez_compressed(paths["out"] / f"inception_features_{tag}.npz", real_monet=f_monet, real_photo=f_photo,
                        fake_monet=f_fake_monet, fake_photo=f_fake_photo)
    print("inception features in %.0fs" % (time.time() - t0), flush=True)
    for name, fr, ff in (("photo2monet_B2A", f_monet, f_fake_monet), ("monet2photo_A2B", f_photo, f_fake_photo)):
        k_mean, k_std = kid(fr, ff, cfg["kid_subsets"], cfg["kid_subset_size"])
        res[name] = {"fid": fid(fr, ff), "kid_mean": k_mean, "kid_std": k_std,
                     "memorisation_min_cosine_distance": memorisation_distance(fr, ff),
                     **precision_recall_density_coverage(fr, ff, cfg["pr_k"]),
                     "n_real": len(fr), "n_fake": len(ff)}
    # reference point: FID between the two real domains (how far apart Monet and photos start)
    res["reference_fid_real_photo_vs_real_monet"] = fid(f_monet, f_photo)

    lp = lpips.LPIPS(net="vgg", verbose=False).to(device)
    vgg = vgg_relu4_3(device)
    rng = np.random.default_rng(cfg["seed"])
    sub_photo = [photo[i] for i in sorted(rng.choice(len(photo), min(max_pairwise, len(photo)), replace=False))]
    res["cycle_B_photo_monet_photo"] = summarize(pairwise_metrics(G_BA, G_AB, sub_photo, device, lp, vgg))
    res["cycle_A_monet_photo_monet"] = summarize(pairwise_metrics(G_AB, G_BA, monet, device, lp, vgg))
    with open(paths["out"] / f"eval_metrics_{tag}.json", "w") as f:
        json.dump(res, f, indent=1)
    return res
