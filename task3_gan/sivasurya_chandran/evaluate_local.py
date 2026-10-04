"""
Kaggle submission scorer: the instructor's Part3_Evaluation_Script.ipynb as a script (same functions, same
defaults), with paths as arguments. Writes submission.csv (ID, FID, MiFID) and submission_details.json.

Folder roles (competition naming: A = Monet, B = photo):
  real Monet  (domain A)  ../data/monet_jpg
  real photos (domain B)  ../data/photo_jpg
  pred_A2B    generated photos from the 300 Monet paintings (our G_BA)
  pred_B2A    generated Monet from the photos (our G_AB; all 7,038, so nothing is hand-picked)

Exactly as in the instructor's script:
  - each folder is sorted by filename and capped at the first N_EVAL = 300 images
  - Inception-v3 (torchvision IMAGENET1K_V1, transform_input=False, fc removed), Resize(299) + CenterCrop(299),
    ImageNet normalisation, batch 32
  - per direction: real/generated lists are truncated to the same length; FID from the two feature sets;
    MiFID = mean cosine distance between real[i] and generated[i] (paired by sorted index)
  - submission FID = (FID_A2B + FID_B2A) / 2, MiFID = (MiFID_A2B + MiFID_B2A) / 2; Kaggle shows -(FID + MiFID) / 2

  python evaluate_local.py --pred_dir outputs --out submission.csv
"""
import argparse
import glob
import json
import os

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from scipy.spatial.distance import cosine

HERE = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def list_images(folder):
    exts = (".jpg", ".jpeg", ".png")
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    paths = sorted(list(set(paths)))
    return paths


def take_n(paths, n):
    if n is None:
        return paths
    return paths[: min(n, len(paths))]


def get_inception_model():
    inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)
    inception.fc = nn.Identity()
    inception.to(device)
    inception.eval()
    return inception


INCEPTION_TF = T.Compose([
    T.Resize(299),
    T.CenterCrop(299),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def load_batch(paths):
    imgs = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        imgs.append(INCEPTION_TF(img))
    return torch.stack(imgs, dim=0)


@torch.no_grad()
def get_activations(model, image_paths, batch_size=32):
    feats = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        x = load_batch(batch_paths).to(device)
        f = model(x).detach().cpu().numpy()
        feats.append(f)
    return np.concatenate(feats, axis=0)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def fid_mifid_from_acts(real_act, gen_act):
    mu_r, sig_r = real_act.mean(axis=0), np.cov(real_act, rowvar=False)
    mu_g, sig_g = gen_act.mean(axis=0), np.cov(gen_act, rowvar=False)
    fid = frechet_distance(mu_r, sig_r, mu_g, sig_g)
    m = min(len(real_act), len(gen_act))
    mifid = float(np.mean([cosine(real_act[i], gen_act[i]) for i in range(m)]))
    return fid, mifid


def calculate_fid_mifid(real_paths, gen_paths, batch_size=32, subsample_to_match=True, model=None):
    real_paths, gen_paths = sorted(real_paths), sorted(gen_paths)
    if subsample_to_match:
        n = min(len(real_paths), len(gen_paths))
        real_paths, gen_paths = real_paths[:n], gen_paths[:n]
    model = model or get_inception_model()
    return fid_mifid_from_acts(get_activations(model, real_paths, batch_size), get_activations(model, gen_paths, batch_size))


def evaluate(real_monet_dir, real_photo_dir, gen_a2b_dir, gen_b2a_dir, n_eval=300, batch_size=32):
    for d in (real_monet_dir, real_photo_dir, gen_a2b_dir, gen_b2a_dir):
        assert os.path.isdir(d), f"Missing folder: {d}"
    real_monet = take_n(list_images(real_monet_dir), n_eval)
    real_photo = take_n(list_images(real_photo_dir), n_eval)
    gen_a2b = take_n(list_images(gen_a2b_dir), n_eval)  # Monet->Photo (generated photos)
    gen_b2a = take_n(list_images(gen_b2a_dir), n_eval)  # Photo->Monet (generated Monet)
    model = get_inception_model()
    fid_b2a, mifid_b2a = calculate_fid_mifid(real_monet, gen_b2a, batch_size, model=model)
    fid_a2b, mifid_a2b = calculate_fid_mifid(real_photo, gen_a2b, batch_size, model=model)
    return {
        "FID": (fid_a2b + fid_b2a) / 2, "MiFID": (mifid_a2b + mifid_b2a) / 2,
        "FID_B2A_photo_to_monet": fid_b2a, "MiFID_B2A_photo_to_monet": mifid_b2a,
        "FID_A2B_monet_to_photo": fid_a2b, "MiFID_A2B_monet_to_photo": mifid_a2b,
        "counts": {"real_monet": len(real_monet), "gen_b2a": len(gen_b2a), "real_photo": len(real_photo), "gen_a2b": len(gen_a2b)},
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real_monet", default=os.path.join(HERE, "..", "data", "monet_jpg"))
    ap.add_argument("--real_photo", default=os.path.join(HERE, "..", "data", "photo_jpg"))
    ap.add_argument("--pred_dir", default="outputs", help="folder holding pred_A2B/ and pred_B2A/")
    ap.add_argument("--gen_a2b", default=None, help="override <pred_dir>/pred_A2B")
    ap.add_argument("--gen_b2a", default=None, help="override <pred_dir>/pred_B2A")
    ap.add_argument("--n_eval", type=int, default=300, help="images per set, as in the instructor's script (0 = all)")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--id", type=int, default=1)
    ap.add_argument("--no_csv", action="store_true")
    a = ap.parse_args()

    r = evaluate(a.real_monet, a.real_photo, a.gen_a2b or os.path.join(a.pred_dir, "pred_A2B"),
                 a.gen_b2a or os.path.join(a.pred_dir, "pred_B2A"), a.n_eval or None)
    r["score_mean"] = (r["FID"] + r["MiFID"]) / 2
    print(f"device {device} | counts {r['counts']}")
    print(f"[Photo->Monet] FID={r['FID_B2A_photo_to_monet']:.3f}  MiFID={r['MiFID_B2A_photo_to_monet']:.4f}")
    print(f"[Monet->Photo] FID={r['FID_A2B_monet_to_photo']:.3f}  MiFID={r['MiFID_A2B_monet_to_photo']:.4f}")
    print(f"submission: FID={r['FID']:.3f} MiFID={r['MiFID']:.4f} -> leaderboard -{r['score_mean']:.4f}")
    if not a.no_csv:
        with open(a.out, "w") as f:  # same columns as the instructor's pandas to_csv
            f.write(f"ID,FID,MiFID\n{a.id},{r['FID']},{r['MiFID']}\n")
        with open(os.path.splitext(a.out)[0] + "_details.json", "w") as f:
            json.dump({**r, "pred_dir": a.pred_dir, "n_eval": a.n_eval, "scorer": "instructor Part3_Evaluation_Script.ipynb"}, f, indent=2)
        print(open(a.out).read())
