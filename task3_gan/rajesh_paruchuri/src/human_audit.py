"""Blinded human audit of 30 fixed photo->Monet translations, scored by 2 human raters.

make_sheet(): picks 30 photo indices with a fixed seed (before looking at any output), writes each as an anonymous
  side-by-side image (left = input photo, right = translation) under outputs/human_audit/, shuffles the order, and
  writes a blank CSV for two raters. The key mapping anonymous IDs -> photo files is kept in a separate file so the
  raters never see file names, epochs or which model made the image.
score(): reads the filled CSV and reports per-criterion mean scores, % exact agreement, % within-1 agreement,
  Cohen's kappa (unweighted) and linear-weighted kappa between the two raters.
The ratings themselves must come from two real people; this script never fills them in.
"""

import csv
import json
import random
from pathlib import Path

import numpy as np

CRITERIA = ("style", "content", "artifacts")
SCALE = (1, 2, 3, 4, 5)
GUIDE = """# Human audit — rater guide (Task 3, Rajesh Paruchuri)

Each image `audit_XX.png` shows **left: the input photo, right: the model's Monet-style translation**.
Rate the right-hand image on three 1–5 scales and write the scores in `audit_sheet.csv` under your column.
Work alone and do not discuss scores with the other rater until both of you are done.

| Score | style (looks like a Monet painting?) | content (same scene as the photo?) | artifacts (how clean?) |
|---|---|---|---|
| 5 | clearly Monet: brushwork, palette, soft light | every object and the layout preserved | no visible artifacts |
| 4 | mostly painterly, a few photo-like regions | layout kept, minor detail loss | 1–2 small artifacts |
| 3 | partly stylised / just a colour filter | main objects recognisable, some distortion | noticeable artifacts in places |
| 2 | barely stylised | objects hard to recognise | artifacts in large areas |
| 1 | no Monet style / broken | scene unrecognisable | image dominated by artifacts |

For `artifacts`, a higher score is **better** (cleaner image).
"""


def make_sheet(photo_files, translate_fn, out_dir, n=30, seed=266):
    """translate_fn(path) -> HxWx3 uint8 translation of that photo."""
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    picks = sorted(rng.sample(range(len(photo_files)), n))
    order = picks[:]
    rng.shuffle(order)
    key = []
    for k, idx in enumerate(order, 1):
        src = np.asarray(Image.open(photo_files[idx]).convert("RGB").resize((256, 256)))
        out = translate_fn(photo_files[idx])
        gap = np.full((256, 8, 3), 255, np.uint8)
        Image.fromarray(np.concatenate([src, gap, out], 1)).save(out_dir / f"audit_{k:02d}.png")
        key.append({"audit_id": f"audit_{k:02d}", "photo_index": idx, "photo_file": Path(photo_files[idx]).name})
    with open(out_dir / "audit_key_DO_NOT_SHOW_RATERS.json", "w") as f:
        json.dump({"seed": seed, "n": n, "key": key}, f, indent=1)
    with open(out_dir / "audit_sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audit_id"] + [f"rater{r}_{c}" for r in (1, 2) for c in CRITERIA] + ["rater1_notes", "rater2_notes"])
        for k in range(1, n + 1):
            w.writerow([f"audit_{k:02d}"] + [""] * (2 * len(CRITERIA) + 2))
    (out_dir / "RATER_GUIDE.md").write_text(GUIDE)
    return picks


def cohen_kappa(a, b, weights=None, labels=SCALE):
    a, b = np.asarray(a), np.asarray(b)
    k = len(labels)
    idx = {l: i for i, l in enumerate(labels)}
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    O /= O.sum()
    E = np.outer(O.sum(1), O.sum(0))
    i, j = np.meshgrid(range(k), range(k), indexing="ij")
    W = (np.abs(i - j) / (k - 1)) if weights == "linear" else (i != j).astype(float)
    denom = (W * E).sum()
    return float(1 - (W * O).sum() / denom) if denom > 0 else 1.0


def score(sheet_path):
    rows = list(csv.DictReader(open(sheet_path)))
    filled = [r for r in rows if all(r[f"rater{x}_{c}"].strip() for x in (1, 2) for c in CRITERIA)]
    if len(filled) < len(rows):
        return {"status": f"incomplete: {len(filled)}/{len(rows)} rows have both raters' scores"}
    res = {"n_samples": len(rows)}
    for c in CRITERIA:
        a = [int(r[f"rater1_{c}"]) for r in rows]
        b = [int(r[f"rater2_{c}"]) for r in rows]
        res[c] = {"rater1_mean": float(np.mean(a)), "rater2_mean": float(np.mean(b)),
                  "mean_both": float(np.mean(a + b)),
                  "exact_agreement": float(np.mean(np.array(a) == np.array(b))),
                  "within1_agreement": float(np.mean(np.abs(np.array(a) - np.array(b)) <= 1)),
                  "cohen_kappa": cohen_kappa(a, b), "cohen_kappa_linear_weighted": cohen_kappa(a, b, "linear")}
    res["overall_human_score_mean_1to5"] = float(np.mean([res[c]["mean_both"] for c in CRITERIA]))
    return res
