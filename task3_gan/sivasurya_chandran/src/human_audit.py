"""
Blinded human audit of 30 fixed samples with 2 raters.

  prepare: picks 30 fixed held-out photos (fixed seed), translates them, saves
           input|output pairs under anonymised IDs, and writes one blank rating
           CSV per rater plus a private key file.
  score:   reads the two filled CSVs and reports mean scores per criterion,
           Cohen's kappa (unweighted and quadratic-weighted) and % agreement.

Ratings are 1-5 for: style (looks like a Monet), content (scene preserved),
artifacts (5 = no visible artifacts).

Run:
    python src/human_audit.py prepare --config config.yaml
    # rater1 and rater2 fill audit/ratings_rater1.csv, audit/ratings_rater2.csv independently
    python src/human_audit.py score --config config.yaml
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import yaml
from torchvision.utils import save_image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from common.utils import resolve_device  # noqa: E402
from dataset import ImageFolderList  # noqa: E402
from eval_metrics import cohens_kappa  # noqa: E402
from models import build_generator  # noqa: E402

CRITERIA = ["style", "content", "artifacts"]
N_SAMPLES = 30
AUDIT_SEED = 266  # fixed so every member audits comparable samples


@torch.no_grad()
def prepare(cfg):
    out_dir = cfg["train"]["out_dir"]
    audit_dir = os.path.join(out_dir, "audit")
    for r in ("rater1", "rater2"):  # never wipe ratings people have already entered
        sheet = os.path.join(audit_dir, f"ratings_{r}.csv")
        if os.path.exists(sheet) and any(row[1:4] != ["", "", ""] for row in list(csv.reader(open(sheet)))[1:]):
            raise SystemExit(f"{sheet} already has ratings; move the audit folder away before preparing a new one")
    os.makedirs(os.path.join(audit_dir, "images"), exist_ok=True)
    with open(os.path.join(out_dir, "split.json")) as f:
        pool = json.load(f)["holdout_a"]
    rng = np.random.default_rng(AUDIT_SEED)
    chosen = [pool[i] for i in sorted(rng.choice(len(pool), min(N_SAMPLES, len(pool)), replace=False))]
    ids = [f"S{int(x):03d}" for x in rng.permutation(len(chosen)) + 1]

    device = resolve_device(cfg["train"]["device"])
    G = build_generator(cfg["model"]).to(device)
    G.load_state_dict(torch.load(os.path.join(out_dir, "G_AB.pt"), map_location=device))
    G.eval()

    ds = ImageFolderList(chosen, cfg["data"]["image_size"])
    key = {}
    for i, sid in enumerate(ids):
        x, name = ds[i]
        fake = G(x.unsqueeze(0).to(device)).cpu()
        save_image(torch.cat([x.unsqueeze(0), fake]) * 0.5 + 0.5, os.path.join(audit_dir, "images", f"{sid}.png"), nrow=2)
        key[sid] = name
    with open(os.path.join(audit_dir, "key_DO_NOT_SHOW_RATERS.json"), "w") as f:
        json.dump(key, f, indent=2)
    for r in ("rater1", "rater2"):
        with open(os.path.join(audit_dir, f"ratings_{r}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["sample_id"] + CRITERIA + ["notes"])
            for sid in sorted(ids):
                w.writerow([sid, "", "", "", ""])
    print(f"wrote {len(ids)} blinded pairs to {audit_dir}/images (left=input, right=translation)")
    print("each rater fills their CSV independently with integers 1-5")


def read(path):
    with open(path) as f:
        return {row["sample_id"]: row for row in csv.DictReader(f)}


def score(cfg):
    audit_dir = os.path.join(cfg["train"]["out_dir"], "audit")
    r1, r2 = read(os.path.join(audit_dir, "ratings_rater1.csv")), read(os.path.join(audit_dir, "ratings_rater2.csv"))
    ids = sorted(set(r1) & set(r2))
    out = {"n_samples": len(ids)}
    all1, all2 = [], []
    for c in CRITERIA:
        a = [int(r1[i][c]) for i in ids]
        b = [int(r2[i][c]) for i in ids]
        all1 += a
        all2 += b
        out[c] = {
            "mean_rater1": float(np.mean(a)),
            "mean_rater2": float(np.mean(b)),
            "mean_both": float(np.mean(a + b)),
            "cohens_kappa": cohens_kappa(a, b),
            "cohens_kappa_quadratic": cohens_kappa(a, b, "quadratic"),
            "exact_agreement_pct": float(np.mean(np.array(a) == np.array(b)) * 100),
            "within_1_agreement_pct": float(np.mean(np.abs(np.array(a) - np.array(b)) <= 1) * 100),
        }
    out["overall_human_audit_score_1to5"] = float(np.mean(all1 + all2))
    out["overall_cohens_kappa_quadratic"] = cohens_kappa(all1, all2, "quadratic")
    out["overall_exact_agreement_pct"] = float(np.mean(np.array(all1) == np.array(all2)) * 100)
    with open(os.path.join(audit_dir, "audit_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["prepare", "score"])
    ap.add_argument("--config", default="config.yaml")
    a = ap.parse_args()
    with open(a.config) as f:
        cfg = yaml.safe_load(f)
    prepare(cfg) if a.action == "prepare" else score(cfg)
