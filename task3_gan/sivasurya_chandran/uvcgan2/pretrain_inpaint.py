"""
Stage 1 of the UVCGAN-v2-style run: generator inpainting pretraining.

Idea from UVCGAN (Torbunov et al., https://github.com/LS4GAN/uvcgan2): before any adversarial
training, teach each generator to reconstruct images with masked-out patches. The generator learns
image statistics of both domains, so the translation stage starts from a sensible initialisation
instead of random weights. No external weights are used: this trains OUR generators on OUR data
(the competition's photo and Monet images only).

    python uvcgan2/pretrain_inpaint.py --config uvcgan2/config_uvcgan.yaml --minutes 45

Writes <out_dir>/pretrain/G_AB.pt and G_BA.pt (both generators are pretrained on the union of both
domains, then the translation stage loads them as its starting point).
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
MEMBER = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(MEMBER, "src"))
sys.path.insert(0, os.path.join(MEMBER, "..", ".."))
from dataset import list_images, split_files, eval_transform  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, HERE)
from models_uvcgan import build_uvc_generator  # noqa: E402

from common.utils import resolve_device, set_seed  # noqa: E402


class FlatImages(Dataset):
    def __init__(self, files, size):
        self.files, self.tf = files, eval_transform(size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert("RGB"))


def random_mask(x, patch=32, prob=0.40):
    """Zero out `patch`x`patch` blocks, each kept with probability (1 - prob). Returns (masked, mask)."""
    b, _, h, w = x.shape
    gh, gw = h // patch, w // patch
    keep = (torch.rand(b, 1, gh, gw, device=x.device) >= prob).float()
    mask = F.interpolate(keep, size=(h, w), mode="nearest")      # 1 = visible, 0 = masked out
    return x * mask, mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config_uvcgan.yaml"))
    ap.add_argument("--minutes", type=float, default=45.0, help="wall-clock budget for pretraining")
    ap.add_argument("--steps", type=int, default=None, help="override: fixed number of steps")
    a = ap.parse_args()

    os.chdir(MEMBER)
    cfg = yaml.safe_load(open(a.config))
    dcfg, tcfg, mcfg = cfg["data"], cfg["train"], cfg["model"]
    pcfg = cfg.get("pretrain", {})
    device = resolve_device(tcfg.get("device", "auto"))
    set_seed(dcfg.get("seed", 1337))

    photos = list_images(dcfg["domain_a_dir"])
    monet = list_images(dcfg["domain_b_dir"])
    train_a, _ = split_files(photos, dcfg.get("holdout_a", 1000), dcfg.get("seed", 1337))
    files = sorted(train_a) + sorted(monet)          # both domains, training photos only
    print(f"inpainting pretraining on {len(files)} images ({len(train_a)} photo + {len(monet)} monet)", flush=True)

    size = dcfg["image_size"]
    bs = pcfg.get("batch_size", 4)
    dl = DataLoader(FlatImages(files, size), batch_size=bs, shuffle=True,
                    num_workers=tcfg.get("num_workers", 3), drop_last=True, persistent_workers=True)

    gens = {"G_AB": build_uvc_generator({**mcfg, "image_size": size}).to(device),
            "G_BA": build_uvc_generator({**mcfg, "image_size": size}).to(device)}
    opts = {k: torch.optim.Adam(g.parameters(), lr=pcfg.get("lr", 2.0e-4), betas=(0.5, 0.999))
            for k, g in gens.items()}
    amp = tcfg.get("amp", True)
    dtype = torch.bfloat16 if tcfg.get("amp_dtype", "bf16") == "bf16" else torch.float16

    out_dir = os.path.join(tcfg["out_dir"], "pretrain")
    os.makedirs(out_dir, exist_ok=True)
    budget = a.minutes * 60
    t0, step, done = time.time(), 0, False
    patch, prob = pcfg.get("mask_patch", 32), pcfg.get("mask_prob", 0.40)

    while not done:
        for x in dl:
            x = x.to(device, non_blocking=True)
            masked, mask = random_mask(x, patch, prob)
            losses = {}
            for k, g in gens.items():
                g.train()
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=amp and device == "cuda"):
                    rec = g(masked)
                    # reconstruct everywhere, but weight the masked (unseen) region higher
                    l_vis = F.l1_loss(rec * mask, x * mask)
                    l_hid = F.l1_loss(rec * (1 - mask), x * (1 - mask))
                    loss = l_vis + pcfg.get("hidden_weight", 2.0) * l_hid
                opts[k].zero_grad(set_to_none=True)
                loss.float().backward()
                torch.nn.utils.clip_grad_norm_(g.parameters(), pcfg.get("clip", 5.0))
                opts[k].step()
                losses[k] = float(loss)
            step += 1
            if step % pcfg.get("log_every", 100) == 0:
                el = time.time() - t0
                print(f"[pretrain] step={step} G_AB_L1={losses['G_AB']:.4f} G_BA_L1={losses['G_BA']:.4f} "
                      f"elapsed={el/60:.1f}m / {a.minutes:.0f}m", flush=True)
            if (a.steps and step >= a.steps) or (not a.steps and time.time() - t0 > budget):
                done = True
                break

    for k, g in gens.items():
        torch.save(g.state_dict(), os.path.join(out_dir, f"{k}.pt"))
    print(f"[pretrain] done: {step} steps in {(time.time()-t0)/60:.1f} min -> {out_dir}/", flush=True)


if __name__ == "__main__":
    main()
