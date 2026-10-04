"""
Translate a folder of images with one of your own trained generators and write JPEGs
(dataset.save_jpegs: the same writer the evaluation and checkpoint selection use).

Run (from the member folder):
    # photo -> Monet for ALL photos into outputs/pred_B2A (what the instructor's evaluation script reads)
    python src/translate.py --config config.yaml --input ../data/photo_jpg --output outputs/pred_B2A --flat
    # Monet -> photo with G_BA (src/evaluate.py already writes outputs/pred_A2B for all 300 paintings)
    python src/translate.py --config config.yaml --generator G_BA --input ../data/monet_jpg --output outputs/pred_A2B --flat
    # without --flat: <output>/images/ plus <output>/images.zip
"""
import argparse
import os
import sys
import zipfile

import torch
import yaml
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from common.utils import resolve_device  # noqa: E402
from dataset import ImageFolderList, list_images, save_jpegs  # noqa: E402
from models import build_generator  # noqa: E402


@torch.no_grad()
def main(cfg_path, checkpoint, generator, input_dir, output_dir, flat, limit):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    mcfg, dcfg = cfg["model"], cfg["data"]
    device = resolve_device(cfg["train"]["device"])
    checkpoint = checkpoint or os.path.join(cfg["train"]["out_dir"], f"{generator}.pt")

    G = build_generator(mcfg).to(device)
    G.load_state_dict(torch.load(checkpoint, map_location=device))
    G.eval()

    files = list_images(input_dir)[: limit or None]
    img_dir = output_dir if flat else os.path.join(output_dir, "images")
    paths = []
    for x, names in DataLoader(ImageFolderList(files, dcfg["image_size"]), batch_size=16):
        paths += save_jpegs(G(x.to(device)), names, img_dir)
    if not flat:
        zip_path = os.path.join(output_dir, "images.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in paths:
                zf.write(p, arcname=os.path.basename(p))
    print(f"translated {len(paths)} images with {checkpoint} -> {img_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", default=None, help="default: <out_dir>/<generator>.pt")
    ap.add_argument("--generator", default="G_AB", choices=["G_AB", "G_BA"], help="G_AB = photo->Monet, G_BA = Monet->photo")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="outputs/pred_B2A")
    ap.add_argument("--flat", action="store_true", help="write JPEGs directly into --output (no images/ subfolder, no zip)")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    main(a.config, a.checkpoint, a.generator, a.input, a.output, a.flat, a.limit)
