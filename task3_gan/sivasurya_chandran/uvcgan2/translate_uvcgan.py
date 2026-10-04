"""
Generate the competition's two prediction folders with the UVCGAN-v2-style generators, then score them
with the instructor's exact evaluator (evaluate_local.py).

Domain convention (unchanged):
    A = photos, B = Monet, G_AB = photo->Monet, G_BA = Monet->photo
Output convention (unchanged, what Part3_Evaluation_Script.ipynb reads):
    <pred_dir>/pred_B2A = photo->Monet  (all 7,038 photos, written with G_AB)
    <pred_dir>/pred_A2B = Monet->photo  (all 300 paintings, written with G_BA)

    python uvcgan2/translate_uvcgan.py --config uvcgan2/config_uvcgan.yaml \
        --checkpoint_dir uvcgan2/checkpoints --out uvcgan2/outputs --score

Writes into freshly emptied folders so no stale file can ever be scored.
"""
import argparse
import os
import shutil
import subprocess
import sys

import torch
import yaml
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
MEMBER = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(MEMBER, "src"))
sys.path.insert(0, os.path.join(MEMBER, "..", ".."))
sys.path.insert(0, HERE)

from dataset import ImageFolderList, list_images, save_jpegs  # noqa: E402
from models_uvcgan import build_uvc_generator  # noqa: E402

from common.utils import resolve_device  # noqa: E402


@torch.no_grad()
def translate(G, files, out_dir, size, device, batch=16):
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)           # never score a stale file
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for x, names in DataLoader(ImageFolderList(files, size), batch_size=batch):
        n += len(save_jpegs(G(x.to(device)), names, out_dir))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config_uvcgan.yaml"))
    ap.add_argument("--checkpoint_dir", default="uvcgan2/checkpoints")
    ap.add_argument("--out", default="uvcgan2/outputs")
    ap.add_argument("--limit_photos", type=int, default=None, help="pilot: only translate the first N photos")
    ap.add_argument("--score", action="store_true", help="run evaluate_local.py afterwards")
    ap.add_argument("--csv", default=None, help="write submission.csv here (implies --score)")
    a = ap.parse_args()

    os.chdir(MEMBER)
    cfg = yaml.safe_load(open(a.config))
    dcfg, mcfg = cfg["data"], {**cfg["model"], "image_size": cfg["data"]["image_size"]}
    device = resolve_device(cfg["train"].get("device", "auto"))
    size, batch = dcfg["image_size"], cfg.get("eval", {}).get("batch_size", 16)

    photos, monet = list_images(dcfg["domain_a_dir"]), list_images(dcfg["domain_b_dir"])
    if a.limit_photos:
        photos = photos[: a.limit_photos]

    gens = {}
    for name in ("G_AB", "G_BA"):
        G = build_uvc_generator(mcfg).to(device)
        G.load_state_dict(torch.load(os.path.join(a.checkpoint_dir, f"{name}.pt"), map_location=device))
        gens[name] = G.eval()

    n_b2a = translate(gens["G_AB"], photos, os.path.join(a.out, "pred_B2A"), size, device, batch)
    n_a2b = translate(gens["G_BA"], monet, os.path.join(a.out, "pred_A2B"), size, device, batch)
    print(f"pred_B2A (photo->Monet): {n_b2a} images\npred_A2B (Monet->photo): {n_a2b} images", flush=True)

    if a.score or a.csv:
        cmd = [sys.executable, "evaluate_local.py", "--real_monet", dcfg["domain_b_dir"],
               "--real_photo", dcfg["domain_a_dir"], "--pred_dir", a.out]
        cmd += ["--out", a.csv] if a.csv else ["--no_csv"]
        subprocess.run(cmd, check=False)


if __name__ == "__main__":
    main()
