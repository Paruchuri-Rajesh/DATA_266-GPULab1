"""
CycleGAN training (A = photo, B = Monet): LSGAN adversarial loss, cycle-consistency loss, identity loss,
50-image history pool for D, constant-then-linear-decay LR. Optional: DiffAugment on every D input,
multi-scale discriminators, an EMA copy of both generators, bf16/fp16 autocast, a wall-clock time budget
that sizes the schedule to the GPU, and checkpoint selection by FID on held-out photos. Writes the raw log,
loss curves, sample grids, metrics JSON and a reproducibility manifest.

Run (from the member folder):
    python src/train.py --config config.yaml --variant A_base --budget_hours 2.6   # one variant, sized to 2.6 h
    python src/train.py --config config.yaml --variant A_base --resume             # continue after an interruption
    python src/run_final.py --hours 3       # all variants in parallel + selection + submission (the graded run)
"""
import argparse
import copy
import itertools
import json
import logging
import math
import os
import random
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from common.utils import hardware_info, peak_memory_mb, reset_peak_memory, resolve_device, set_seed, write_manifest  # noqa: E402
from cfg import load_config  # noqa: E402
from dataset import ImageFolderList, UnpairedImageDataset, list_images, save_jpegs, split_files  # noqa: E402
from diffaugment import diff_augment  # noqa: E402
from models import build_discriminator, build_generator, count_params, init_weights  # noqa: E402


class ImagePool:
    """Returns a mix of current and previously generated images (Shrivastava et al.), as in the CycleGAN paper."""

    def __init__(self, size: int):
        self.size, self.images = size, []

    def query(self, images):
        if self.size == 0:
            return images
        out = []
        for img in images.detach():
            img = img.unsqueeze(0)
            if len(self.images) < self.size:
                self.images.append(img)
                out.append(img)
            elif random.random() > 0.5:
                i = random.randrange(self.size)
                out.append(self.images[i].clone())
                self.images[i] = img
            else:
                out.append(img)
        return torch.cat(out, 0)


def grad_norm(params) -> float:
    norms = [p.grad.detach().float().norm(2) for p in params if p.grad is not None]
    return torch.stack(norms).norm(2).item() if norms else 0.0  # one GPU sync per call


def lsgan(pred, target: float):
    """LSGAN loss; a multi-scale discriminator returns one map per scale and the scales are averaged."""
    preds = pred if isinstance(pred, list) else [pred]
    return sum(((p.float() - target) ** 2).mean() for p in preds) / len(preds)


@torch.no_grad()
def ema_update(ema: nn.Module, model: nn.Module, decay: float):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.lerp_(p.detach().float(), 1.0 - decay)
    for be, b in zip(ema.buffers(), model.buffers()):
        be.copy_(b)


class OfficialSelector:
    """Checkpoint selection with the instructor's metric (evaluate_local.py = Part3_Evaluation_Script.ipynb:
    torchvision Inception-v3 with transform_input=False, 300 images per set, both directions averaged):
      photo->Monet: G_AB on held-out photos vs the 300 real Monet paintings
      Monet->photo: G_BA on the 300 Monet paintings vs held-out real photos
    score = (FID + MiFID) / 2 with FID and MiFID averaged over the two directions, as on the leaderboard.
    The held-out photos exclude the first 300 photos by filename (the ones the official script scores), and the
    score is averaged over `n_sets` disjoint sets of `n_images` photos to reduce the noise of 300-image FID.
    Outputs are written with dataset.save_jpegs and read back, exactly like the submitted folders."""

    def __init__(self, scfg, hold_a, monet_files, photo_dir, image_size, device, tmp_dir):
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        import evaluate_local  # member folder: the instructor's scorer

        self.el, self.device, self.tmp_dir, self.size = evaluate_local, device, tmp_dir, image_size
        n, n_sets = scfg.get("n_images", 300), scfg.get("n_sets", 2)
        eval_photos = {os.path.basename(p) for p in evaluate_local.take_n(evaluate_local.list_images(photo_dir), 300)}
        pool = sorted(p for p in hold_a if os.path.basename(p) not in eval_photos) or sorted(hold_a)  # fallback: tiny test sets
        n = min(n, len(pool))
        self.sets = [pool[i * n:(i + 1) * n] for i in range(max(1, min(n_sets, len(pool) // max(1, n))))]
        self.monet = sorted(monet_files)[:300]
        self.n_photos = sum(len(x) for x in self.sets)
        self.model = evaluate_local.get_inception_model()
        self.real_monet = evaluate_local.get_activations(self.model, self.monet)
        self.real_photo = [evaluate_local.get_activations(self.model, x) for x in self.sets]

    @torch.no_grad()
    def _translate(self, G, files, tag):
        paths = []
        for x, names in DataLoader(ImageFolderList(files, self.size), batch_size=16):
            paths += save_jpegs(G(x.to(self.device)), names, os.path.join(self.tmp_dir, tag))
        return sorted(paths)

    def _pair(self, real_act, gen_act):
        m = min(len(real_act), len(gen_act))  # the official script truncates both sorted lists to the same length
        return self.el.fid_mifid_from_acts(real_act[:m], gen_act[:m])

    def __call__(self, G_AB, G_BA) -> dict:
        modes = (G_AB.training, G_BA.training)
        G_AB.eval(), G_BA.eval()
        b2a = [self._pair(self.real_monet, self.el.get_activations(self.model, self._translate(G_AB, x, f"b2a{k}"))) for k, x in enumerate(self.sets)]
        gen_a2b = self.el.get_activations(self.model, self._translate(G_BA, self.monet, "a2b"))
        a2b = [self._pair(r, gen_a2b) for r in self.real_photo]
        G_AB.train(modes[0]), G_BA.train(modes[1])
        r = {"fid_b2a": float(np.mean([f for f, _ in b2a])), "mifid_b2a": float(np.mean([m for _, m in b2a])),
             "fid_a2b": float(np.mean([f for f, _ in a2b])), "mifid_a2b": float(np.mean([m for _, m in a2b]))}
        r["fid"], r["mifid"] = (r["fid_b2a"] + r["fid_a2b"]) / 2, (r["mifid_b2a"] + r["mifid_a2b"]) / 2
        r["score"] = (r["fid"] + r["mifid"]) / 2
        return r


def plot_losses(h, path, fid_history=None):
    n = 4 if fid_history else 3
    fig, axes = plt.subplots(1, n, figsize=(5.3 * n, 4))
    s = h["step"]
    axes[0].plot(s, h["loss_G_adv"], label="G adversarial (AB+BA)")
    axes[0].plot(s, h["loss_D_A"], label="D_A")
    axes[0].plot(s, h["loss_D_B"], label="D_B")
    axes[0].set_title("Adversarial losses")
    axes[1].plot(s, h["loss_cycle"], label="cycle (L1, A+B)")
    axes[1].plot(s, h["loss_identity"], label="identity (L1, A+B)")
    axes[1].set_title("Cycle-consistency / identity")
    axes[2].plot(s, h["grad_norm_G"], label="||grad G||")
    axes[2].plot(s, h["grad_norm_D"], label="||grad D||")
    axes[2].set_yscale("log")
    axes[2].set_title("Gradient norms")
    if fid_history:
        for kind in ("raw", "ema"):
            pts = [(r["step"], r[kind]) for r in fid_history if r.get(kind) is not None and not r.get("calibration")]
            if pts:
                axes[3].plot(*zip(*pts), marker="o", label=f"G_AB {kind}")
        axes[3].set_title("Held-out official score (lower is better)")
    for ax in axes:
        ax.set_xlabel("step")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fixed_schedule(tcfg, scfg):
    n = tcfg["epochs"]
    return {"n": n, "decay": tcfg["decay_start_epoch"], "from": scfg.get("from_epoch", 0), "every": scfg.get("every", 1), "calibrated": True}


def budget_schedule(tcfg, scfg, budget_s, elapsed_s, epoch_s, eval_s, select_on):
    """Size the run to the time budget from the measured speed of epoch 0 (and of one FID check)."""
    n_evals = scfg.get("n_evals", 12) if select_on else 0
    usable = 0.95 * (budget_s - elapsed_s - n_evals * eval_s)  # 5% margin for speed drift
    n = 1 + int(usable // epoch_s)
    n = max(tcfg.get("min_epochs", 10), min(tcfg.get("max_epochs", 400), n))
    decay = int(round(tcfg.get("decay_frac", 0.5) * n))
    start = int(math.ceil(scfg.get("from_frac", 0.5) * n))
    every = max(1, int(math.ceil((n - 1 - start) / max(1, n_evals - 1))))
    return {"n": n, "decay": decay, "from": start, "every": every, "calibrated": True}


def main(cfg_path: str, variant: str = None, resume: bool = False, budget_hours: float = None):
    cfg = load_config(cfg_path, variant)
    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]
    scfg = cfg.get("select") or {}
    set_seed(dcfg["seed"])
    random.seed(dcfg["seed"])
    device = resolve_device(tcfg["device"])
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_num_threads(2)  # several variant processes share the CPU with their data-loader workers
    out_dir = tcfg["out_dir"]
    sample_dir = os.path.join(out_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(os.path.dirname(tcfg["log_file"]), exist_ok=True)
    logging.basicConfig(filename=tcfg["log_file"], level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("task3")
    log.info(f"config={json.dumps(cfg)}")
    log.info(f"hardware={hardware_info(device)}")

    for d in (dcfg["domain_a_dir"], dcfg["domain_b_dir"]):
        if not list_images(d):
            raise SystemExit(f"no images found in {os.path.abspath(d)}: download the data first (python ../download_data.py)")
    train_a, hold_a = split_files(list_images(dcfg["domain_a_dir"]), dcfg["holdout_a"], dcfg["seed"])
    train_b, hold_b = split_files(list_images(dcfg["domain_b_dir"]), dcfg["holdout_b"], dcfg["seed"])
    with open(os.path.join(out_dir, "split.json"), "w") as f:
        json.dump({"train_a": train_a, "holdout_a": hold_a, "train_b": train_b, "holdout_b": hold_b}, f, indent=1)
    print(f"[{variant or 'base'}] device={device} train A={len(train_a)} B={len(train_b)} | holdout A={len(hold_a)} B={len(hold_b)}", flush=True)

    ds = UnpairedImageDataset(train_a, train_b, dcfg["image_size"], augment=True)
    nw = tcfg["num_workers"]
    dl = DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=True, num_workers=nw, drop_last=True,
                    pin_memory=device == "cuda", persistent_workers=nw > 0)

    G_AB, G_BA = build_generator(mcfg).to(device), build_generator(mcfg).to(device)
    D_A, D_B = build_discriminator(mcfg).to(device), build_discriminator(mcfg).to(device)
    for m in (G_AB, G_BA, D_A, D_B):
        m.apply(init_weights)
    params = {"G_AB": count_params(G_AB), "G_BA": count_params(G_BA), "D_A": count_params(D_A), "D_B": count_params(D_B)}
    params["total"] = sum(params.values())
    log.info(f"params={params}")

    l1 = nn.L1Loss()
    g_params = list(itertools.chain(G_AB.parameters(), G_BA.parameters()))
    d_params = list(itertools.chain(D_A.parameters(), D_B.parameters()))
    opt_G = torch.optim.Adam(g_params, lr=tcfg["lr"], betas=(tcfg["beta1"], 0.999))
    opt_D = torch.optim.Adam(d_params, lr=tcfg["lr"], betas=(tcfg["beta1"], 0.999))

    # Schedule (epochs, LR-decay start, which epochs get an FID check). With a time budget it is sized after
    # epoch 0 from the measured speed; the lambda reads the dict, so the sizing applies from then on.
    select_on = bool(scfg.get("enabled"))
    budget_s = budget_hours * 3600 if budget_hours else None
    sched = fixed_schedule(tcfg, scfg)
    if budget_s:
        sched["calibrated"] = False
        sched["n"] = max(sched["n"], 2)

    def lr_lambda(epoch):
        return 1.0 - max(0, epoch - sched["decay"]) / float(max(1, sched["n"] - sched["decay"]))

    sch_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sch_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)

    # Mixed precision on CUDA: bf16 where supported (RTX 30/40 series, A100: no loss scaling needed), else fp16 + GradScaler
    amp = device == "cuda" and tcfg.get("amp", True)
    use_bf16 = amp and tcfg.get("amp_dtype", "fp16") == "bf16" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler_G = torch.amp.GradScaler("cuda", enabled=amp and not use_bf16)
    scaler_D = torch.amp.GradScaler("cuda", enabled=amp and not use_bf16)
    log.info(f"amp={amp} dtype={'bf16' if use_bf16 else 'fp16' if amp else 'fp32'}")
    pool_A, pool_B = ImagePool(tcfg["pool_size"]), ImagePool(tcfg["pool_size"])
    policy = tcfg.get("diffaugment", "")  # DiffAugment on every D input (real and fake); "" = off
    aug = lambda x: diff_augment(x, policy)  # noqa: E731
    ema_decay = tcfg.get("ema_decay", 0.0)
    # Cycle weight schedule ("CycleGAN with Better Cycles", Wang et al. 2024): lambda_cycle moves linearly to
    # lambda_cycle_end over the run; the identity weight keeps its ratio to the cycle weight
    lam_c0 = tcfg["lambda_cycle"]
    lam_c1 = tcfg.get("lambda_cycle_end", lam_c0)

    def loss_weights(epoch):
        frac = min(1.0, epoch / max(1, sched["n"] - 1))
        lam_c = lam_c0 + (lam_c1 - lam_c0) * frac
        return lam_c, tcfg["lambda_identity"] * lam_c / lam_c0

    ema = {"G_AB": copy.deepcopy(G_AB).eval().requires_grad_(False), "G_BA": copy.deepcopy(G_BA).eval().requires_grad_(False)} if ema_decay else {}
    log.info(f"diffaugment={policy!r} upsample={mcfg.get('upsample', 'deconv')} d_scales={mcfg.get('d_scales', 1)} "
             f"lambda_cycle={lam_c0}->{lam_c1} lambda_identity={tcfg['lambda_identity']} (scaled with cycle) ema_decay={ema_decay} budget_hours={budget_hours}")
    selector = OfficialSelector(scfg, hold_a, train_b + hold_b, dcfg["domain_a_dir"], dcfg["image_size"], device,
                                os.path.join(out_dir, "select_tmp")) if select_on else None
    best = {"score": float("inf"), "epoch": None, "weights": None}
    fid_history = []

    keys = ["step", "epoch", "loss_G", "loss_G_adv", "loss_D_A", "loss_D_B", "loss_cycle", "loss_identity", "grad_norm_G", "grad_norm_D"]
    history = {k: [] for k in keys}
    all_gn_G, all_gn_D = [], []
    nan_count = 0
    fixed = next(iter(DataLoader(UnpairedImageDataset(hold_a or train_a, hold_b or train_b, dcfg["image_size"], augment=False), batch_size=4)))

    nets = {"G_AB": G_AB, "G_BA": G_BA, "D_A": D_A, "D_B": D_B}
    state_path = os.path.join(out_dir, "train_state.pt")
    start_epoch, n_images, step, prev_elapsed = 0, 0, 0, 0.0
    init_from = tcfg.get("init_from")  # warm start all four: "{name}" is replaced by G_AB/G_BA/D_A/D_B
    init_gen = tcfg.get("init_generators_from")  # warm start the generators only; discriminators start fresh
    if (init_from or init_gen) and not (resume and os.path.exists(state_path)):
        path_tpl, load_names = (init_from, list(nets)) if init_from else (init_gen, ["G_AB", "G_BA"])
        for n in load_names:
            nets[n].load_state_dict(torch.load(path_tpl.format(name=n), map_location=device))
        for k, m in ema.items():  # ema was deep-copied above, before these weights were loaded: re-sync it
            m.load_state_dict(nets[k].state_dict())
        log.info(f"initialised {', '.join(load_names)} from {path_tpl}"
                 f"{'; discriminators initialised fresh' if init_gen else ''}"
                 f"{'; EMA re-synced to the loaded generators' if ema else ''} (optimizers and LR schedule start fresh)")
    if resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=device, weights_only=False)
        for n, m in nets.items():
            m.load_state_dict(st[n])
        for obj, k in ((opt_G, "opt_G"), (opt_D, "opt_D"), (sch_G, "sch_G"), (sch_D, "sch_D"), (scaler_G, "scaler_G"), (scaler_D, "scaler_D")):
            obj.load_state_dict(st[k])
        for k, m in ema.items():
            if f"ema_{k}" in st:
                m.load_state_dict(st[f"ema_{k}"])
        start_epoch, step, n_images, prev_elapsed = st["epoch"] + 1, st["step"], st["n_images"], st["elapsed"]
        history, all_gn_G, all_gn_D, nan_count = st["history"], st["all_gn_G"], st["all_gn_D"], st["nan_count"]
        best, fid_history = st.get("best", best), st.get("fid_history", fid_history)
        sched.update(st.get("sched", {}))
        log.info(f"resumed from {state_path} at epoch={start_epoch} step={step} schedule={sched}")
        print(f"[{variant or 'base'}] resumed at epoch {start_epoch}, step {step}", flush=True)

    def score_and_keep(epoch, tag=""):
        """Official-metric score of G_AB and G_BA, each independently raw or EMA (4 pairs when EMA is on, since the
        two directions can peak at different times): the best pair becomes {name}.pt if it beats the best so far."""
        t_sel = time.time()
        pairs = {"raw_raw": (G_AB, G_BA)}
        if ema:
            pairs["raw_ema"] = (G_AB, ema["G_BA"])
            pairs["ema_raw"] = (ema["G_AB"], G_BA)
            pairs["ema_ema"] = (ema["G_AB"], ema["G_BA"])
        parts = {k: selector(*g) for k, g in pairs.items()}
        rec = {"epoch": epoch, "step": step, "raw": parts["raw_raw"]["score"],
               "ema": parts["ema_ema"]["score"] if ema else None, "parts": parts, "calibration": bool(tag)}
        fid_history.append(rec)
        kind = min(parts, key=lambda k: parts[k]["score"])
        f_val = parts[kind]["score"]
        is_best = f_val < best["score"] and not tag
        if is_best:
            best.update(score=f_val, epoch=epoch, step=step, weights=kind, **{k: v for k, v in parts[kind].items() if k != "score"})
            ab_src, ba_src = (ema["G_AB"] if "ema" == kind.split("_")[0] else G_AB), (ema["G_BA"] if "ema" == kind.split("_")[1] else G_BA)
            torch.save(ab_src.state_dict(), os.path.join(out_dir, "G_AB.pt"))
            torch.save(ba_src.state_dict(), os.path.join(out_dir, "G_BA.pt"))
            for name in ("D_A", "D_B"):
                torch.save(nets[name].state_dict(), os.path.join(out_dir, f"{name}.pt"))
        p = parts[kind]
        ema_txt = f" ema_ema={rec['ema']:.3f}" if ema else ""
        msg = (f"epoch={epoch} select_score={f_val:.3f} (FID {p['fid']:.2f}: photo->Monet {p['fid_b2a']:.2f}, Monet->photo {p['fid_a2b']:.2f}; "
               f"MiFID {p['mifid']:.4f}) raw_raw={rec['raw']:.3f}{ema_txt} [{kind}, {selector.n_photos} held-out photos, {time.time() - t_sel:.0f}s]{tag} "
               f"best={best['score']:.3f}@epoch{best['epoch']}({best['weights']}){' NEW BEST' if is_best else ''}")
        log.info(msg)
        print(f"[{variant or 'base'}] {msg}", flush=True)
        return time.time() - t_sel

    reset_peak_memory(device)
    t0 = time.time() - prev_elapsed
    max_steps = tcfg.get("max_steps_per_epoch")
    steps_per_epoch = min(len(dl), max_steps) if max_steps else len(dl)
    epoch = start_epoch
    while epoch < sched["n"]:
        t_epoch = time.time()
        lam_cyc, lam_idt = loss_weights(epoch)
        for i, batch in enumerate(dl):
            if max_steps and i >= max_steps:
                break
            real_a, real_b = batch["A"].to(device, non_blocking=True), batch["B"].to(device, non_blocking=True)

            # Generators (discriminators frozen)
            for p in d_params:
                p.requires_grad_(False)
            opt_G.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
                fake_b, fake_a = G_AB(real_a), G_BA(real_b)
                rec_a, rec_b = G_BA(fake_b), G_AB(fake_a)
                loss_adv = lsgan(D_B(aug(fake_b)), 1.0) + lsgan(D_A(aug(fake_a)), 1.0)
                loss_cyc = l1(rec_a.float(), real_a) + l1(rec_b.float(), real_b)
                if lam_idt > 0:
                    loss_idt = l1(G_BA(real_a).float(), real_a) + l1(G_AB(real_b).float(), real_b)
                else:
                    loss_idt = torch.zeros((), device=device)
                loss_G = loss_adv + lam_cyc * loss_cyc + lam_idt * loss_idt
            if not torch.isfinite(loss_G):
                nan_count += 1
                log.info(f"step={step} non-finite generator loss, skipping")
                step += 1
                continue
            scaler_G.scale(loss_G).backward()
            scaler_G.unscale_(opt_G)
            gn_G = grad_norm(g_params)
            scaler_G.step(opt_G)
            scaler_G.update()
            for k, m in ema.items():
                ema_update(m, nets[k], ema_decay)
            for p in d_params:
                p.requires_grad_(True)

            # Discriminators on real vs pooled fakes
            opt_D.zero_grad(set_to_none=True)
            fa, fb = pool_A.query(fake_a), pool_B.query(fake_b)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp):
                loss_D_A = 0.5 * (lsgan(D_A(aug(real_a)), 1.0) + lsgan(D_A(aug(fa)), 0.0))
                loss_D_B = 0.5 * (lsgan(D_B(aug(real_b)), 1.0) + lsgan(D_B(aug(fb)), 0.0))
                loss_D = loss_D_A + loss_D_B
            if not torch.isfinite(loss_D):
                nan_count += 1
                log.info(f"step={step} non-finite discriminator loss, skipping")
                step += 1
                continue
            scaler_D.scale(loss_D).backward()
            scaler_D.unscale_(opt_D)
            gn_D = grad_norm(d_params)
            scaler_D.step(opt_D)
            scaler_D.update()

            all_gn_G.append(gn_G)
            all_gn_D.append(gn_D)
            n_images += real_a.size(0)

            if step % tcfg["log_every"] == 0:
                vals = [step, epoch, loss_G.item(), loss_adv.item(), loss_D_A.item(), loss_D_B.item(), loss_cyc.item(), loss_idt.item(), gn_G, gn_D]
                for k, v in zip(keys, vals):
                    history[k].append(v)
                el = time.time() - t0
                total = sched["n"] * steps_per_epoch
                msg = (
                    f"epoch={epoch}/{sched['n']} step={step} lr={sch_G.get_last_lr()[0]:.2e} lam_cyc={lam_cyc:.2f} loss_G={loss_G.item():.4f} adv={loss_adv.item():.4f} "
                    f"cyc={loss_cyc.item():.4f} idt={loss_idt.item():.4f} D_A={loss_D_A.item():.4f} D_B={loss_D_B.item():.4f} "
                    f"gnG={gn_G:.3f} gnD={gn_D:.3f} elapsed={el / 60:.1f}m eta={el / (step + 1) * max(0, total - step - 1) / 60:.1f}m"
                )
                log.info(msg)
                print(f"[{variant or 'base'}] {msg}", flush=True)

            if step % tcfg["sample_every"] == 0:
                with torch.no_grad():
                    G_AB.eval(), G_BA.eval()
                    xa = fixed["A"].to(device)
                    fb_ = G_AB(xa)
                    grid = torch.cat([xa, fb_, G_BA(fb_)], 0) * 0.5 + 0.5
                    save_image(grid, os.path.join(sample_dir, f"step{step:07d}_A_fakeB_recA.png"), nrow=xa.size(0))
                    G_AB.train(), G_BA.train()
            step += 1
        epoch_s = time.time() - t_epoch

        if not sched["calibrated"]:
            # Time-budget sizing after epoch 0: one FID check measures the evaluation cost too (logged, never kept as best)
            eval_s = score_and_keep(epoch, tag=" [calibration]") if selector else 0.0
            sched.update(budget_schedule(tcfg, scfg, budget_s, time.time() - t0, epoch_s, eval_s, select_on))
            msg = (f"budget {budget_hours:.2f} h: epoch 0 took {epoch_s:.0f}s ({epoch_s / steps_per_epoch:.3f} s/step), one FID check {eval_s:.0f}s "
                   f"-> {sched['n']} epochs x {steps_per_epoch} steps, LR decay from epoch {sched['decay']}, FID checks from epoch {sched['from']} every {sched['every']}")
            log.info(msg)
            print(f"[{variant or 'base'}] {msg}", flush=True)

        sch_G.step()
        sch_D.step()
        if selector is None:
            for name, m in nets.items():
                torch.save(m.state_dict(), os.path.join(out_dir, f"{name}.pt"))
        else:
            # {name}.pt = the epoch (raw or EMA weights) with the lowest held-out FID; {name}_last.pt = latest epoch (raw)
            for name, m in nets.items():
                torch.save(m.state_dict(), os.path.join(out_dir, f"{name}_last.pt"))
            due = epoch >= sched["from"] and (epoch - sched["from"]) % sched["every"] == 0
            if due or epoch == sched["n"] - 1:
                score_and_keep(epoch)
        state = {n: m.state_dict() for n, m in nets.items()}
        for k, m in ema.items():
            state[f"ema_{k}"] = m.state_dict()
        for obj, k in ((opt_G, "opt_G"), (opt_D, "opt_D"), (sch_G, "sch_G"), (sch_D, "sch_D"), (scaler_G, "scaler_G"), (scaler_D, "scaler_D")):
            state[k] = obj.state_dict()
        state.update(epoch=epoch, step=step, n_images=n_images, elapsed=time.time() - t0, history=history, sched=dict(sched),
                     all_gn_G=all_gn_G, all_gn_D=all_gn_D, nan_count=nan_count, best=best, fid_history=fid_history)
        torch.save(state, state_path)
        log.info(f"epoch={epoch} done, checkpoints + train_state saved")
        epoch += 1

    elapsed = time.time() - t0
    plot_path = os.path.join(out_dir, "loss_curves.png")
    plot_losses(history, plot_path, fid_history)

    def last_mean(k, n=10):
        v = history[k][-n:]
        return float(np.mean(v)) if v else None

    # fp16 loss-scale overflows give inf grad norms; the scalers skip those steps, so stats use finite norms only
    rawG, rawD = np.array(all_gn_G or [0.0]), np.array(all_gn_D or [0.0])
    gG = rawG[np.isfinite(rawG)] if np.isfinite(rawG).any() else np.array([0.0])
    gD = rawD[np.isfinite(rawD)] if np.isfinite(rawD).any() else np.array([0.0])
    metrics = {
        "variant": variant,
        "checkpoints": {n: os.path.join(out_dir, f"{n}.pt") for n in ("G_AB", "G_BA", "D_A", "D_B")},
        "param_count": params,
        "epochs": sched["n"],
        "steps": step,
        "schedule": sched,
        "budget_hours": budget_hours,
        "training_time_sec": elapsed,
        "images_per_sec": n_images / elapsed,
        "peak_memory_mb": peak_memory_mb(device),
        "nan_count": nan_count,
        "amp_overflow_skipped_steps": {"G": int((~np.isfinite(rawG)).sum()), "D": int((~np.isfinite(rawD)).sum())},
        "amp_dtype": "bf16" if use_bf16 else "fp16" if amp else "fp32",
        "grad_norm_G": {"mean": float(gG.mean()), "max": float(gG.max()), "p95": float(np.percentile(gG, 95))},
        "grad_norm_D": {"mean": float(gD.mean()), "max": float(gD.max()), "p95": float(np.percentile(gD, 95))},
        "final_loss_G": last_mean("loss_G"),
        "final_loss_G_adv": last_mean("loss_G_adv"),
        "final_loss_D_A": last_mean("loss_D_A"),
        "final_loss_D_B": last_mean("loss_D_B"),
        "final_cycle_loss": last_mean("loss_cycle"),
        "final_identity_loss": last_mean("loss_identity"),
        "loss_curve_plot": plot_path,
        "checkpoint_selection": {"method": "lowest held-out score with the instructor's metric, both directions (raw or EMA generator pair)",
                                 "best": best, "n_heldout_photos": selector.n_photos, "fid_history": fid_history} if selector else {"method": "last epoch"},
        "history": history,
    }
    metrics_path = os.path.join(out_dir, "metrics_train.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    write_manifest(os.path.join(out_dir, "manifest.json"), cfg, device, metrics["checkpoints"], metrics_path)
    log.info(f"final={json.dumps({k: v for k, v in metrics.items() if k != 'history'})}")
    done = f"best held-out score {best['score']:.3f} at epoch {best['epoch']} ({best['weights']})" if selector else "weights of the last epoch kept"
    print(f"[{variant or 'base'}] done: {done}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--variant", default=None, help="a name from the config's variants: section")
    ap.add_argument("--resume", action="store_true", help="continue from <out_dir>/train_state.pt")
    ap.add_argument("--budget_hours", type=float, default=None, help="size the schedule to this many hours (measured after epoch 0)")
    a = ap.parse_args()
    main(a.config, a.variant, a.resume, a.budget_hours)
