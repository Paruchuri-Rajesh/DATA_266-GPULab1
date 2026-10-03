"""Writes part3_cyclegan.ipynb (unexecuted). Run: python build_notebook.py"""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# DATA 266 Lab 1 — Task 3: CycleGAN, photo ↔ Monet
**Member:** Rajesh Paruchuri · **Data:** class Kaggle competition images (`monet_jpg/` = domain A, `photo_jpg/` = domain B)

Two generators (G_AB: Monet→photo, G_BA: photo→Monet, the Kaggle direction) and two PatchGAN discriminators,
written from scratch in `cyclegan.py`. No pretrained network touches training or generation. Inception-v3, VGG16 and
LPIPS appear only in `evaluate.py`, to *measure* images the CycleGAN has already produced.

Run (repo root): `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 task3_gan/rajesh_paruchuri/src/part3_cyclegan.ipynb`
· smoke test: prefix with `LAB_SMOKE=1`.
""")

code("""
import json, sys, time, csv, subprocess
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path.cwd()) if (Path.cwd() / "cyclegan.py").exists() else str(Path.cwd() / "task3_gan/rajesh_paruchuri/src"))
import cyclegan as cg

P = cg.find_paths()
R = lambda p: cg.rel(p, P["repo"])
CFG = cg.load_config(P)
SMOKE = bool(CFG["smoke"])
TAG = "smoke" if SMOKE else "full"
device = cg.pick_device()
print("smoke:", SMOKE, "| seed:", CFG["seed"], "| device:", device, "|", cg.hardware_string(device))
import torchvision
print("torch", torch.__version__, "| torchvision", torchvision.__version__, "| python", sys.version.split()[0])
""")

md("""
## 3.1 Data: two unpaired domains
""")

code("""
root = cg.data_dir(CFG, P)
monet, photo = cg.list_images(root / "monet_jpg"), cg.list_images(root / "photo_jpg")
sizes = {Image.open(p).size for p in monet[:50] + photo[:50]}
print("domain A (Monet paintings):", len(monet), "| domain B (photos):", len(photo), "| image sizes seen:", sizes)
print("unpaired: there is no correspondence between any Monet file and any photo file")
fig, ax = plt.subplots(2, 6, figsize=(15, 5.4))
for i in range(6):
    ax[0, i].imshow(Image.open(monet[i * 37 % len(monet)])); ax[0, i].axis("off")
    ax[1, i].imshow(Image.open(photo[i * 911 % len(photo)])); ax[1, i].axis("off")
ax[0, 0].set_title("A: Monet", loc="left"); ax[1, 0].set_title("B: photos", loc="left")
fig.tight_layout(); fig.savefig(P["out"] / f"domains_{TAG}.png", dpi=110); plt.show()
""")

md("""
## 3.1 Model
| Part | My choice | Why |
|---|---|---|
| Generators (×2) | ResNet: c7s1-48 → 2 stride-2 downsamples → **6 residual blocks** → 2 × (bilinear ×2 + 3×3 conv) → c7s1-3, tanh; InstanceNorm, reflection padding | 6 blocks and 48 base channels (the standard CycleGAN uses 9 and 64) halve the cost on a laptop GPU. Resize+conv upsampling avoids the checkerboard artifacts of transposed convolutions |
| Discriminators (×2) | 70×70 PatchGAN C64-C128-C256-C512-1 with **spectral normalisation**, no norm layers | judges local texture, which is what style is. Spectral norm bounds D's Lipschitz constant and stabilises the min-max game |
| Adversarial loss | LSGAN (MSE to 1 / 0) | smoother gradients than the log loss when D is confident |
| Cycle loss | L1(G_BA(G_AB(a)), a) + L1(G_AB(G_BA(b)), b), λ = 10 | the constraint that makes unpaired training possible: a translation must be invertible |
| Identity loss | L1(G_BA(a), a) + L1(G_AB(b), b), λ = 5 | stops the generators from shifting colours that are already in the target domain |
| **DiffAugment** | colour + translation + cutout on every real *and* fake D input | only 300 Monet paintings, so without augmentation D memorises them and G gets useless gradients |
| Replay buffer | 50 past fakes | D sees a mix of old and new fakes, which reduces oscillation |
| **EMA generators** | decay 0.999, used for all outputs | averaging weights smooths GAN noise between steps |
| Training input | random **128×128 crops** + flips, batch 1 | about 4× cheaper than full images. The model is fully convolutional, so inference runs on full 256×256 images |
| Schedule | Adam 2e-4, β = (0.5, 0.999); 80 epochs × 300 iterations; constant for 40 epochs, then linear decay to 0 | the standard CycleGAN recipe |
""")

code("""
G_AB, G_BA, D_A, D_B = cg.build_models(CFG, device)
print("G params each: {:,} | D params each: {:,} | total: {:,}".format(cg.count_params(G_AB), cg.count_params(D_A),
                                                                         cg.count_params(G_AB, G_BA, D_A, D_B)))
x = torch.randn(1, 3, 256, 256, device=device)
with torch.no_grad():
    print("G(256x256) ->", tuple(G_BA(x).shape), "| D(256x256) -> patch map", tuple(D_A(x).shape),
          "| D(128x128 crop) ->", tuple(D_A(x[..., :128, :128]).shape))
del G_AB, G_BA, D_A, D_B
""")

md("""
## 3.1 Training
Training runs in its own process (`python cyclegan.py full`) and writes an unedited raw log to
`reproducibility/raw_logs/rajesh_paruchuri/task3_gan/`. If a finished checkpoint and history already exist, this
cell reuses them instead of retraining (delete them to retrain).
""")

code("""
ckpt = P["ckpt"] / f"cyclegan_{TAG}.pt"
hist_path = P["out"] / f"train_history_{TAG}.json"
log_path = P["logs"] / f"train_{TAG}.log"
if not (ckpt.exists() and hist_path.exists()):
    run = subprocess.run([sys.executable, str(P["src"] / "cyclegan.py"), TAG], capture_output=True, text=True)
    print(run.stdout[-4000:])
    if run.returncode != 0:
        raise RuntimeError(run.stderr[-3000:])
else:
    print("reusing finished run:", R(ckpt))
lines = open(log_path).read().splitlines()
print("\\n".join(lines[:4] + ["..."] + [l for l in lines if l.startswith("EPOCH")][-6:] + lines[-2:]))
H = json.load(open(hist_path))
hist, summary = H["hist"], H["summary"]
""")

md("""
## 3.1 Training behaviour: convergence and stability
Each point is the mean over a 50-iteration window.
""")

code("""
it = np.array(hist["iter"]); ep = it / CFG["iters_per_epoch"] if not SMOKE else it / CFG["smoke_iters_per_epoch"]
fig, ax = plt.subplots(2, 3, figsize=(17, 8))
ax[0, 0].plot(ep, hist["loss_G_adv"], label="G adversarial (both)"); ax[0, 0].plot(ep, hist["loss_D_A"], label="D_A (Monet)")
ax[0, 0].plot(ep, hist["loss_D_B"], label="D_B (photo)"); ax[0, 0].set(title="Adversarial losses (LSGAN)", xlabel="epoch"); ax[0, 0].legend()
ax[0, 1].plot(ep, hist["loss_cycle"], label="cycle L1 (A+B)"); ax[0, 1].plot(ep, hist["loss_identity"], label="identity L1 (A+B)")
ax[0, 1].set(title="Cycle-consistency and identity losses (unweighted)", xlabel="epoch"); ax[0, 1].legend()
ax[0, 2].plot(ep, hist["loss_G"], color="k"); ax[0, 2].set(title="Total generator loss", xlabel="epoch")
for k in ("D_A_real", "D_A_fake", "D_B_real", "D_B_fake"):
    ax[1, 0].plot(ep, hist[k], label=k)
ax[1, 0].axhline(1, ls=":", c="gray"); ax[1, 0].axhline(0, ls=":", c="gray")
ax[1, 0].set(title="Mean D output (1 = 'real', 0 = 'fake')", xlabel="epoch"); ax[1, 0].legend(fontsize=8)
ax[1, 1].plot(ep, hist["grad_norm_G"], label="G"); ax[1, 1].plot(ep, hist["grad_norm_D"], label="D")
ax[1, 1].set(title="Gradient L2 norm (no clipping)", xlabel="epoch", yscale="log"); ax[1, 1].legend()
ax[1, 2].plot(ep, hist["lr"]); ax[1, 2].set(title="Learning rate", xlabel="epoch")
for a in ax.flat:
    a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(P["out"] / f"loss_curves_{TAG}.png", dpi=120); plt.show()
print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()})
""")

code("""
prev = sorted((P["out"] / "previews").glob(f"{TAG}_epoch*.png"))
show = [prev[i] for i in sorted({0, len(prev) // 4, len(prev) // 2, len(prev) - 1}) if prev]
fig, ax = plt.subplots(len(show), 1, figsize=(15, 5.2 * len(show)))
for a, p in zip(np.atleast_1d(ax), show):
    a.imshow(Image.open(p)); a.axis("off"); a.set_title(p.stem + "   rows: Monet → photo → Monet | photo → Monet → photo", loc="left")
fig.tight_layout(); plt.show()
""")

md("""
## 3.1 Translation and 3.2 evaluation
Both EMA generators translate every image at full 256×256. Metrics are computed in both directions:
* **FID** and **KID** (Inception-v3 pool features), translated images vs the *real* images of the target domain.
* **Generative precision / recall** (k = 3 nearest neighbours) and **density / coverage**.
* **Cycle-reconstruction L1** (pixels in [0,1]), plus **LPIPS** for input vs reconstruction and input vs translation.
* **Content-preservation cosine similarity** between VGG16 relu4_3 features of the input and its translation.
""")

code("""
import evaluate as ev
t0 = time.time()
eval_json = P["out"] / f"eval_metrics_{TAG}.json"
if eval_json.exists() and eval_json.stat().st_mtime > ckpt.stat().st_mtime:
    E = json.load(open(eval_json)); print("reusing", R(eval_json), "(newer than the checkpoint; delete it to recompute)")
else:
    E = ev.run(ckpt, CFG, P, device, TAG)
print("evaluation took %.0fs" % (time.time() - t0))
rows = []
for d in ("photo2monet_B2A", "monet2photo_A2B"):
    m = E[d]
    rows.append([d, m["fid"], m["kid_mean"], m["kid_std"], m["precision"], m["recall"], m["density"], m["coverage"],
                 m["memorisation_min_cosine_distance"], m["n_real"], m["n_fake"]])
import pandas as pd
display(pd.DataFrame(rows, columns=["direction", "FID", "KID", "KID std", "precision", "recall", "density", "coverage",
                                    "min cos-dist to real", "n real", "n fake"]).round(4))
print("reference FID real photos vs real Monet: %.1f" % E["reference_fid_real_photo_vs_real_monet"])
cyc = {d: {k: v["mean"] for k, v in E[d].items()} for d in ("cycle_B_photo_monet_photo", "cycle_A_monet_photo_monet")}
display(pd.DataFrame(cyc).T.round(4))
""")

md("""
### Verifying the cycle-consistency constraint
If the constraint is working, the reconstruction G_AB(G_BA(x)) should be much closer to x than the translation
G_BA(x) is, and closer than two unrelated images of the same domain are. The figure shows x, G(x) and F(G(x)) side by side.
""")

code("""
for d in ("cycle_B_photo_monet_photo", "cycle_A_monet_photo_monet"):
    m = E[d]
    print(f"{d}: L1(x, reconstruction) = {m['cycle_l1']['mean']:.4f} | L1(x, translation) = {m['translation_l1']['mean']:.4f} "
          f"| L1(x, unrelated image) = {m['unrelated_l1']['mean']:.4f} | LPIPS(x, recon) = {m['cycle_lpips']['mean']:.3f} "
          f"vs LPIPS(x, translation) = {m['translation_lpips']['mean']:.3f}")
G_AB, G_BA, _ = cg.load_generators(ckpt, CFG, device)
rng = np.random.default_rng(CFG["seed"])
pick_b = [photo[i] for i in rng.choice(len(photo), 4, replace=False)]
pick_a = [monet[i] for i in rng.choice(len(monet), 4, replace=False)]
fig, ax = plt.subplots(4, 6, figsize=(15, 10.5))
with torch.no_grad():
    for r, (pb, pa) in enumerate(zip(pick_b, pick_a)):
        xb = cg.to_tensor(cg.load_image(pb)).unsqueeze(0).to(device); yb = G_BA(xb); rb = G_AB(yb)
        xa = cg.to_tensor(cg.load_image(pa)).unsqueeze(0).to(device); ya = G_AB(xa); ra = G_BA(ya)
        for c, (img, title) in enumerate([(xb, "photo"), (yb, "→ Monet"), (rb, "→ photo (cycle)"),
                                          (xa, "Monet"), (ya, "→ photo"), (ra, "→ Monet (cycle)")]):
            ax[r, c].imshow(cg.to_uint8(img[0])); ax[r, c].axis("off")
            if r == 0: ax[r, c].set_title(title)
fig.tight_layout(); fig.savefig(P["out"] / f"translation_cycle_grid_{TAG}.png", dpi=110); plt.show()
""")

md("""
## 3.2 Kaggle submission (`submission.csv`: ID, FID, MiFID)
The values come from the **course-provided `Part3_Evaluation_Script.ipynb`**, run unmodified except for its folder
paths, which the script asks you to set. It compares the first 300 real Monet with the first 300 of my photo→Monet
outputs (`outputs/pred_B2A/`), and the first 300 real photos with my 300 Monet→photo outputs (`outputs/pred_A2B/`).
It then writes the average of the two directions. The executed copy is kept as `src/Part3_Evaluation_Script_run.ipynb`.
""")

code("""
import re, shutil, nbformat
official = P["task"] / "Part3_Evaluation_Script.ipynb"
run_nb = P["src"] / f"Part3_Evaluation_Script_run{'_smoke' if SMOKE else ''}.ipynb"
sfx = "" if TAG == "full" else "_smoke"
nb = nbformat.read(official, as_version=4)
for c in nb.cells:
    if c.cell_type == "code" and 'BASE = "Part 3/Data"' in c.source:
        c.source = (c.source.replace('BASE = "Part 3/Data"', 'BASE = "../../data"')
                    .replace('os.path.join(BASE, "pred_A2B")', f'"../outputs/pred_A2B{sfx}"')
                    .replace('os.path.join(BASE, "pred_B2A")', f'"../outputs/pred_B2A{sfx}"'))
        print("path cell after edit:\\n" + c.source)
nbformat.write(nb, run_nb)
r = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=-1", str(run_nb)], capture_output=True, text=True)
assert r.returncode == 0, r.stderr[-3000:]
done = nbformat.read(run_nb, as_version=4)
for c in done.cells:
    for o in c.get("outputs", []):
        t = o.get("text", "") or o.get("data", {}).get("text/plain", "")
        if "FID" in t or "Counts" in t:
            print(t.strip())
sub_src = P["src"] / "submission.csv"
sub = P["member"] / ("submission.csv" if TAG == "full" else "submission_smoke.csv")
shutil.move(sub_src, sub)
print("\\n" + R(sub) + ":\\n" + open(sub).read())
""")

md("""
## 3.2 Blinded human audit (30 fixed samples, 2 raters)
The 30 photos are chosen with a fixed seed before any output is inspected. Each audit image is anonymous
(`audit_XX.png`: input on the left, translation on the right) and the order is shuffled. Two raters score style,
content and artifacts (1–5) independently in `outputs/human_audit/audit_sheet.csv`, using `RATER_GUIDE.md`.
Agreement is reported as % exact agreement, % within-1 agreement, and Cohen's kappa (unweighted and linear-weighted).
""")

code("""
import human_audit as ha
aud = P["out"] / "human_audit"
if not (aud / "audit_sheet.csv").exists():
    @torch.no_grad()
    def tr(p):
        return cg.to_uint8(G_BA(cg.to_tensor(cg.load_image(p)).unsqueeze(0).to(device))[0])
    picks = ha.make_sheet(photo, tr, aud, CFG["audit_n"], CFG["seed"])
    print("audit sheet created for photo indices:", picks)
HA = ha.score(aud / "audit_sheet.csv")
print(json.dumps(HA, indent=1))
""")

md("""
## Metrics report
""")

code("""
b2a, a2b = E["photo2monet_B2A"], E["monet2photo_A2B"]
cB, cA = E["cycle_B_photo_monet_photo"], E["cycle_A_monet_photo_monet"]
last = lambda k: float(np.mean(hist[k][-10:]))
rows = [
 ("fid_photo2monet_B2A", b2a["fid"], f"{b2a['n_fake']} translated photos vs {b2a['n_real']} real Monet; torchvision Inception-v3"),
 ("fid_monet2photo_A2B", a2b["fid"], f"{a2b['n_fake']} translated Monet vs {a2b['n_real']} real photos"),
 ("kid_photo2monet_B2A", b2a["kid_mean"], f"std {b2a['kid_std']:.5f}; {CFG['kid_subsets']} subsets of {CFG['kid_subset_size']}"),
 ("kid_monet2photo_A2B", a2b["kid_mean"], f"std {a2b['kid_std']:.5f}"),
 ("precision_photo2monet_B2A", b2a["precision"], "k=3 improved precision"), ("recall_photo2monet_B2A", b2a["recall"], "k=3 improved recall"),
 ("density_photo2monet_B2A", b2a["density"], ""), ("coverage_photo2monet_B2A", b2a["coverage"], ""),
 ("precision_monet2photo_A2B", a2b["precision"], ""), ("recall_monet2photo_A2B", a2b["recall"], ""),
 ("density_monet2photo_A2B", a2b["density"], ""), ("coverage_monet2photo_A2B", a2b["coverage"], ""),
 ("memorisation_cosdist_photo2monet_B2A", b2a["memorisation_min_cosine_distance"], "mean min cosine distance to a real Monet (MiFID-style term, local)"),
 ("cycle_l1_photo_monet_photo_B", cB["cycle_l1"]["mean"], f"pixels in [0,1], {cB['cycle_l1']['n']} photos"),
 ("cycle_l1_monet_photo_monet_A", cA["cycle_l1"]["mean"], f"{cA['cycle_l1']['n']} Monet"),
 ("lpips_cycle_B", cB["cycle_lpips"]["mean"], "LPIPS-VGG input vs reconstruction"), ("lpips_cycle_A", cA["cycle_lpips"]["mean"], ""),
 ("lpips_translation_photo2monet", cB["translation_lpips"]["mean"], "LPIPS-VGG input vs translation"),
 ("lpips_translation_monet2photo", cA["translation_lpips"]["mean"], ""),
 ("content_cosine_photo2monet", cB["content_cosine"]["mean"], "VGG16 relu4_3 cosine, input vs translation"),
 ("content_cosine_monet2photo", cA["content_cosine"]["mean"], ""),
 ("loss_G_adv_final", last("loss_G_adv"), "mean of last 10 log windows"), ("loss_D_A_final", last("loss_D_A"), ""),
 ("loss_D_B_final", last("loss_D_B"), ""), ("loss_cycle_final", last("loss_cycle"), "unweighted L1 sum A+B"),
 ("loss_identity_final", last("loss_identity"), "unweighted L1 sum A+B"),
 ("grad_norm_G_mean", summary["grad_norm_G_mean"], "window means, no clipping"), ("grad_norm_G_max", summary["grad_norm_G_max"], ""),
 ("grad_norm_D_mean", summary["grad_norm_D_mean"], ""), ("grad_norm_D_max", summary["grad_norm_D_max"], ""),
 ("nan_count", summary["nan_count"], "non-finite steps (skipped)"),
 ("param_count_total", summary["params_total"], f"G {summary['params_G_each']:,} x2 + D {summary['params_D_each']:,} x2"),
 ("training_time_sec", summary["total_training_time_sec"], ""), ("train_images_per_sec", summary["images_per_sec"], "Monet+photo crops per second"),
 ("inference_images_per_sec_photo2monet", E["gen_images_per_sec_b2a"], "256x256, batch 8"),
 ("peak_mps_tensor_memory_mb", summary["peak_mps_tensor_memory_mb"], ""), ("peak_mps_driver_memory_mb", summary["peak_mps_driver_memory_mb"], ""),
 ("peak_process_rss_mb", summary["peak_process_rss_mb"], ""),
 ("human_audit_score_1to5", HA.get("overall_human_score_mean_1to5", "pending"), "mean of style/content/artifacts, both raters"),
]
for c in ("style", "content", "artifacts"):
    rows.append((f"human_{c}_cohen_kappa", HA.get(c, {}).get("cohen_kappa", "pending"), "unweighted"))
    rows.append((f"human_{c}_exact_agreement", HA.get(c, {}).get("exact_agreement", "pending"), ""))
sub_vals = list(csv.DictReader(open(P["member"] / "submission.csv")))[0] if (P["member"] / "submission.csv").exists() else {}
done_nb = nbformat.read(P["src"] / "Part3_Evaluation_Script_run.ipynb", as_version=4)
per_dir = re.findall(r"\\[(Photo->Monet|Monet->Photo)\\] FID=([0-9.]+)  MiFID=([0-9.]+)",
                     "".join(o.get("text", "") for c in done_nb.cells for o in c.get("outputs", [])))
for d, f_, m_ in per_dir:
    key = "photo2monet_B2A" if d.startswith("Photo") else "monet2photo_A2B"
    rows += [(f"official_script_fid_{key}", float(f_), "Part3_Evaluation_Script.ipynb, first 300 vs 300"),
             (f"official_script_mifid_{key}", float(m_), "script's MiFID = mean paired cosine distance")]
rows += [("kaggle_submission_fid", float(sub_vals.get("FID", "nan")), "submission.csv = mean of both directions (official script)"),
         ("kaggle_submission_mifid", float(sub_vals.get("MiFID", "nan")), "submission.csv"),
         ]
kr_path = P["out"] / "kaggle_result.json"   # written after the Kaggle upload
kr = json.load(open(kr_path)) if kr_path.exists() else {}
rows += [("kaggle_public_score", kr.get("public_score", "pending"), f"my submission ref {kr.get('submission_ref', '-')}, team {kr.get('team', '-')}"),
         ("kaggle_private_score", kr.get("private_score") or "not shown until the competition closes", ""),
         ("kaggle_rank", kr.get("team_public_rank", "pending"), "team public rank (Kaggle ranks a team by its best submission)"),
         ("hardware", cg.hardware_string(device), "")]
csv_path = P["member"] / "metrics_report.csv" if not SMOKE else P["out"] / "metrics_report_smoke.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["metric", "value", "notes"]); w.writerows(rows)
print("saved", R(csv_path))
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for k, c in enumerate(nb["cells"]):
    c["id"] = f"c{k:02d}"
Path(__file__).with_name("part3_cyclegan.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote part3_cyclegan.ipynb with", len(cells), "cells")
