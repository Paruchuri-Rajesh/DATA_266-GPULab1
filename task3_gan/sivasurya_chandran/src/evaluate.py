"""
Task 3 evaluation in both directions on held-out images:
  FID, KID, precision/recall, density/coverage, cycle-reconstruction L1,
  LPIPS(input, translation), content cosine similarity, identity L1.

Photo->Monet fakes are compared against all real Monet paintings (the
Monet set is small); Monet->Photo fakes are compared against held-out real photos.

Run (after train.py):
    python src/evaluate.py --config config.yaml
"""
import argparse
import json
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from common.utils import resolve_device, set_seed  # noqa: E402
from dataset import ImageFolderList, save_jpegs  # noqa: E402
from eval_metrics import content_cosine, fid, inception_features, kid, l1_distance, lpips_distance, prdc  # noqa: E402
from models import build_generator  # noqa: E402


def load_images(files, size, bs):
    xs = [x for x, _ in DataLoader(ImageFolderList(files, size), batch_size=bs)]
    return torch.cat(xs) if xs else torch.empty(0, 3, size, size)


@torch.no_grad()
def run(G, x, device, bs):
    return torch.cat([G(x[i : i + bs].to(device)).cpu() for i in range(0, len(x), bs)])


def save_translations(fake, names, folder):
    """Writes every translation of this direction as a JPEG (dataset.save_jpegs, the same writer as translate.py)."""
    save_jpegs(fake, names, folder)


def direction_metrics(name, x_in, G_fwd, G_back, G_idt, real_ref, device, bs, out_dir, names=None, pred_dir=None):
    fake = run(G_fwd, x_in, device, bs)
    if pred_dir:
        save_translations(fake, names, pred_dir)
    rec = run(G_back, fake, device, bs)
    idt = run(G_idt, x_in, device, bs)
    f_in, f_fake, f_ref = (inception_features(t, device) for t in (x_in, fake, real_ref))
    kid_m, kid_s = kid(f_ref, f_fake)
    n = min(8, len(x_in))
    save_image(torch.cat([x_in[:n], fake[:n], rec[:n]]) * 0.5 + 0.5, os.path.join(out_dir, f"eval_{name}_input_fake_rec.png"), nrow=n)
    return {
        "n_inputs": len(x_in),
        "n_reference_real": len(real_ref),
        "fid": fid(f_ref, f_fake),
        "kid_mean": kid_m,
        "kid_std": kid_s,
        **prdc(f_ref, f_fake),
        "cycle_reconstruction_l1": l1_distance(x_in, rec),
        "lpips_input_vs_translation": lpips_distance(x_in, fake, device),
        "lpips_input_vs_reconstruction": lpips_distance(x_in, rec, device),
        "content_cosine_input_vs_translation": content_cosine(f_in, f_fake),
        "identity_l1": l1_distance(x_in, idt),
    }


def main(cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    dcfg, mcfg, tcfg, ecfg = cfg["data"], cfg["model"], cfg["train"], cfg["eval"]
    set_seed(dcfg["seed"])
    device = resolve_device(tcfg["device"])
    out_dir, size, bs = tcfg["out_dir"], dcfg["image_size"], ecfg["batch_size"]

    with open(os.path.join(out_dir, "split.json")) as f:
        split = json.load(f)
    cap = ecfg.get("max_images")

    def capped(files):
        return files[:cap] if cap else files

    photo_files, monet_files = capped(split["holdout_a"]), capped(split["holdout_b"] + split["train_b"])
    photos_hold = load_images(photo_files, size, bs)
    monet_all = load_images(monet_files, size, bs)
    base = [os.path.basename(f) for f in photo_files], [os.path.basename(f) for f in monet_files]
    pred_root = ecfg.get("pred_dir", "outputs")

    gens = {}
    for n in ("G_AB", "G_BA"):
        g = build_generator(mcfg).to(device)
        g.load_state_dict(torch.load(os.path.join(out_dir, f"{n}.pt"), map_location=device))
        gens[n] = g.eval()

    # Output folders follow the COMPETITION's domain names (A = Monet, B = photo), which are the reverse of
    # this code's internal names (A = photo, B = Monet; G_AB = photo->Monet):
    #   outputs/pred_A2B = Monet->photo for all 300 monet_jpg images (host: "run it on all 300 Monet images")
    #   outputs/pred_B2A = photo->Monet for ALL 7,038 photos, written by src/translate.py --flat (the instructor's
    #   script scores the first 300 by filename, so every photo is translated and nothing is picked)
    results = {
        # identity for A->B direction: G_BA applied to a real photo should leave it unchanged
        "photo_to_monet": direction_metrics("photo2monet", photos_hold, gens["G_AB"], gens["G_BA"], gens["G_BA"], monet_all, device, bs, out_dir,
                                            base[0], None),
        "monet_to_photo": direction_metrics("monet2photo", monet_all, gens["G_BA"], gens["G_AB"], gens["G_AB"], photos_hold, device, bs, out_dir,
                                            base[1], os.path.join(pred_root, "pred_A2B")),
    }
    with open(os.path.join(out_dir, "metrics_eval.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
