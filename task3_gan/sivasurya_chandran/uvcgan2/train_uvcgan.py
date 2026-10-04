"""
Stage 2 of the UVCGAN-v2-style run: unpaired translation training (CycleGAN-shaped).

Two generators (G_AB photo->Monet, G_BA Monet->photo) and two discriminators, LSGAN adversarial +
cycle-consistency + identity, plus the UVCGAN-v2-inspired additions:
  - hybrid U-Net/ViT generators with source-driven style modulation (models_uvcgan.py)
  - spectral-normalised discriminators with a batch-statistics head
  - zero-centred R1 gradient penalty on real images (lazy: every r1_every D steps)
  - low-resolution 32x32 consistency loss (keeps composition while texture changes)
  - optional discriminator-feature cycle loss (config flag, default off)
  - EMA generator weights; separate G/D learning rates; linear LR decay; image replay pool
Checkpoint selection reuses src/train.py's OfficialSelector (the instructor's exact metric) over all
four raw/EMA generator pairings.

    python uvcgan2/train_uvcgan.py --config uvcgan2/config_uvcgan.yaml --budget_hours 9
"""
import argparse
import copy
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torchvision.utils import save_image

HERE = os.path.dirname(os.path.abspath(__file__))
MEMBER = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(MEMBER, "src"))
sys.path.insert(0, os.path.join(MEMBER, "..", ".."))
sys.path.insert(0, HERE)

from dataset import UnpairedImageDataset, list_images, split_files  # noqa: E402
from diffaugment import diff_augment  # noqa: E402
from train import ImagePool, ema_update, grad_norm, lsgan  # noqa: E402
from models_uvcgan import build_uvc_discriminator, build_uvc_generator  # noqa: E402
from dual_selector import DualSelector  # noqa: E402

from common.utils import hardware_info, peak_memory_mb, reset_peak_memory, resolve_device, set_seed  # noqa: E402

log = logging.getLogger("uvcgan2")


def setup_log(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log.setLevel(logging.INFO)
    h = logging.FileHandler(path)           # append-only: raw logs are never rewritten
    h.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    log.addHandler(h)


def r1_penalty(D, real):
    """Zero-centred gradient penalty on real samples: E[||grad_x D(x)||^2]."""
    real = real.detach().requires_grad_(True)
    with torch.autocast(device_type="cuda", enabled=False):
        out = D(real.float()).sum()
        g = torch.autograd.grad(out, real, create_graph=True)[0]
    return g.pow(2).flatten(1).sum(1).mean()


def main(cfg_path, budget_hours=None, resume=False, max_steps=None):
    os.chdir(MEMBER)
    cfg = yaml.safe_load(open(cfg_path))
    dcfg, tcfg, mcfg, scfg = cfg["data"], cfg["train"], cfg["model"], cfg.get("select", {})
    device = resolve_device(tcfg.get("device", "auto"))
    set_seed(dcfg.get("seed", 1337))
    out_dir = tcfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    setup_log(tcfg["log_file"])

    size = dcfg["image_size"]
    photos, monet = list_images(dcfg["domain_a_dir"]), list_images(dcfg["domain_b_dir"])
    train_a, hold_a = split_files(photos, dcfg.get("holdout_a", 1000), dcfg["seed"])
    train_b, hold_b = split_files(monet, dcfg.get("holdout_b", 0), dcfg["seed"])
    print(f"[uvcgan] device={device} train A={len(train_a)} B={len(train_b)} | holdout A={len(hold_a)}", flush=True)
    log.info(f"device={device} train A={len(train_a)} B={len(train_b)} holdout A={len(hold_a)} {hardware_info(device)}")

    ds = UnpairedImageDataset(train_a, train_b, size, augment=True)
    dl = DataLoader(ds, batch_size=tcfg.get("batch_size", 1), shuffle=True,
                    num_workers=tcfg.get("num_workers", 3), drop_last=True, persistent_workers=True)

    mm = {**mcfg, "image_size": size}
    G_AB, G_BA = build_uvc_generator(mm).to(device), build_uvc_generator(mm).to(device)
    D_A, D_B = build_uvc_discriminator(mm).to(device), build_uvc_discriminator(mm).to(device)
    nets = {"G_AB": G_AB, "G_BA": G_BA, "D_A": D_A, "D_B": D_B}

    ema_decay = tcfg.get("ema_decay", 0.0)
    ema = {k: copy.deepcopy(nets[k]).eval().requires_grad_(False) for k in ("G_AB", "G_BA")} if ema_decay else {}

    # stage-1 warm start: inpainting-pretrained generators (our own weights, trained on our own data)
    init_gen = tcfg.get("init_generators_from")
    state_path = os.path.join(out_dir, "train_state.pt")
    if init_gen and not (resume and os.path.exists(state_path)):
        missing = [n for n in ("G_AB", "G_BA") if not os.path.exists(init_gen.format(name=n))]
        if missing:
            print(f"[uvcgan] NOTE: no pretrained generators at {init_gen} ({missing}); starting from random init", flush=True)
            log.info(f"no pretrained generators at {init_gen}; random init")
        else:
            for n in ("G_AB", "G_BA"):
                nets[n].load_state_dict(torch.load(init_gen.format(name=n), map_location=device))
            for k, m in ema.items():
                m.load_state_dict(nets[k].state_dict())     # EMA must start from the loaded weights
            log.info(f"initialised G_AB,G_BA from {init_gen}; discriminators fresh; EMA re-synced")
            print(f"[uvcgan] warm start: generators from {init_gen}; D fresh; EMA synced", flush=True)

    opt_G = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()),
                             lr=tcfg["lr_g"], betas=(tcfg["beta1"], tcfg.get("beta2", 0.999)))
    opt_D = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()),
                             lr=tcfg["lr_d"], betas=(tcfg["beta1"], tcfg.get("beta2", 0.999)))

    steps_per_epoch = min(len(dl), tcfg.get("max_steps_per_epoch", 1000))
    sched = {"n": tcfg.get("max_epochs", 1000), "decay": 0, "from": 0, "every": 1, "calibrated": not budget_hours}
    if not budget_hours:
        sched.update(n=tcfg.get("max_epochs", 60), decay=int(0.4 * tcfg.get("max_epochs", 60)),
                     **{"from": scfg.get("from_epoch", 30), "every": scfg.get("every", 2)})

    def lr_factor(epoch):
        return 1.0 - max(0, epoch - sched["decay"]) / float(max(1, sched["n"] - sched["decay"]))

    sch_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_factor)
    sch_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_factor)

    pool_A, pool_B = ImagePool(tcfg.get("pool_size", 50)), ImagePool(tcfg.get("pool_size", 50))
    policy = tcfg.get("diffaugment", "")
    amp, dtype = tcfg.get("amp", True), torch.bfloat16 if tcfg.get("amp_dtype") == "bf16" else torch.float16
    autocast = lambda: torch.autocast(device_type="cuda", dtype=dtype, enabled=amp and device == "cuda")  # noqa: E731

    select_on = scfg.get("enabled", True) and len(hold_a) > 0
    selector = DualSelector(scfg, hold_a, train_b + hold_b, dcfg["domain_a_dir"], size, device,
                            os.path.join(out_dir, "select_tmp")) if select_on else None
    best = {"score": float("inf"), "epoch": None, "weights": None}
    fid_history, history = [], {k: [] for k in
                                ["step", "epoch", "loss_G", "loss_G_adv", "loss_D_A", "loss_D_B",
                                 "loss_cycle", "loss_identity", "loss_lowres", "r1", "grad_norm_G", "grad_norm_D"]}
    lam_c0, lam_c1 = tcfg["lambda_cycle"], tcfg.get("lambda_cycle_end", tcfg["lambda_cycle"])
    start_epoch, step, nan_count, prev_elapsed = 0, 0, 0, 0.0

    if resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=device, weights_only=False)
        for n, m in nets.items():
            m.load_state_dict(st[n])
        for k, m in ema.items():
            m.load_state_dict(st[f"ema_{k}"])
        opt_G.load_state_dict(st["opt_G"]); opt_D.load_state_dict(st["opt_D"])
        sch_G.load_state_dict(st["sch_G"]); sch_D.load_state_dict(st["sch_D"])
        start_epoch, step, prev_elapsed = st["epoch"] + 1, st["step"], st["elapsed"]
        best, fid_history, history, sched = st["best"], st["fid_history"], st["history"], st["sched"]
        print(f"[uvcgan] resumed at epoch {start_epoch}, step {step}", flush=True)

    def score_and_keep(epoch, tag=""):
        """Official metric over all four raw/EMA generator pairings; best pair is saved."""
        t0s = time.time()
        pairs = {"raw_raw": (G_AB, G_BA)}
        if ema:
            pairs.update(raw_ema=(G_AB, ema["G_BA"]), ema_raw=(ema["G_AB"], G_BA),
                         ema_ema=(ema["G_AB"], ema["G_BA"]))
        parts = {k: selector(*g) for k, g in pairs.items()}
        kind = min(parts, key=lambda k: parts[k]["score"])
        val = parts[kind]["score"]
        fid_history.append({"epoch": epoch, "step": step, "parts": parts, "calibration": bool(tag)})
        is_best = val < best["score"] and not tag
        if is_best:
            best.update(score=val, epoch=epoch, step=step, weights=kind,
                        **{k: v for k, v in parts[kind].items() if k != "score"})
            ab = ema["G_AB"] if kind.split("_")[0] == "ema" else G_AB
            ba = ema["G_BA"] if kind.split("_")[1] == "ema" else G_BA
            torch.save(ab.state_dict(), os.path.join(out_dir, "G_AB.pt"))
            torch.save(ba.state_dict(), os.path.join(out_dir, "G_BA.pt"))
            torch.save(D_A.state_dict(), os.path.join(out_dir, "D_A.pt"))
            torch.save(D_B.state_dict(), os.path.join(out_dir, "D_B.pt"))
        o, h = parts[kind]["official"], parts[kind]["heldout"]
        msg = (f"epoch={epoch} OFFICIAL score={o['score']:.3f} "
               f"(FID {o['fid']:.2f} = photo->Monet {o['fid_b2a']:.2f} / Monet->photo {o['fid_a2b']:.2f}; "
               f"MiFID {o['mifid']:.4f} = {o['mifid_b2a']:.4f} / {o['mifid_a2b']:.4f}) | "
               f"HELDOUT score={h['score']:.3f} (FID {h['fid']:.2f} = {h['fid_b2a']:.2f} / {h['fid_a2b']:.2f}) | "
               + " ".join(f"{k}={v['score']:.3f}" for k, v in parts.items())
               + f" [{kind}, official={selector.n_official} heldout={selector.n_heldout} photos, "
                 f"{time.time()-t0s:.0f}s]{tag} best={best['score']:.3f}@epoch{best['epoch']}({best['weights']})"
                 f"{' NEW BEST' if is_best else ''}")
        log.info(msg)
        print(f"[uvcgan] {msg}", flush=True)
        return time.time() - t0s

    sample_dir = os.path.join(out_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    reset_peak_memory(device)
    t0 = time.time() - prev_elapsed
    budget_s = budget_hours * 3600 if budget_hours else None
    print(f"[uvcgan] steps/epoch={steps_per_epoch} lr_g={tcfg['lr_g']} lr_d={tcfg['lr_d']} "
          f"r1_gamma={tcfg['r1_gamma']} lambda_lowres={tcfg['lambda_lowres']} diffaug={policy!r}", flush=True)

    for epoch in range(start_epoch, sched["n"]):
        t_epoch = time.time()
        frac = min(1.0, epoch / max(1, sched["n"] - 1))
        lam_c = lam_c0 + (lam_c1 - lam_c0) * frac
        lam_i = tcfg["lambda_identity"] * lam_c / lam_c0
        G_AB.train(), G_BA.train(), D_A.train(), D_B.train()
        it = iter(dl)
        for i in range(steps_per_epoch):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(dl); batch = next(it)
            xa = batch["A"].to(device, non_blocking=True)
            xb = batch["B"].to(device, non_blocking=True)

            # ---- generators
            with autocast():
                fb, fa = G_AB(xa), G_BA(xb)
                rec_a, rec_b = G_BA(fb), G_AB(fa)
                idt_b, idt_a = G_AB(xb), G_BA(xa)
                adv = lsgan(D_B(diff_augment(fb, policy)), 1.0) + lsgan(D_A(diff_augment(fa, policy)), 1.0)
                cyc = F.l1_loss(rec_a, xa) + F.l1_loss(rec_b, xb)
                idt = F.l1_loss(idt_b, xb) + F.l1_loss(idt_a, xa)
                # low-resolution consistency: composition must survive the translation
                low = (F.l1_loss(F.interpolate(fb, size=32, mode="area"), F.interpolate(xa, size=32, mode="area"))
                       + F.l1_loss(F.interpolate(fa, size=32, mode="area"), F.interpolate(xb, size=32, mode="area")))
                loss_G = adv + lam_c * cyc + lam_i * idt + tcfg.get("lambda_lowres", 0.0) * low
            opt_G.zero_grad(set_to_none=True)
            loss_G.float().backward()
            gnG = grad_norm(list(G_AB.parameters()) + list(G_BA.parameters()))
            torch.nn.utils.clip_grad_norm_(list(G_AB.parameters()) + list(G_BA.parameters()), tcfg.get("clip", 5.0))
            opt_G.step()

            # ---- discriminators
            with autocast():
                fb_p, fa_p = pool_B.query(fb.detach()), pool_A.query(fa.detach())
                loss_D_B = 0.5 * (lsgan(D_B(diff_augment(xb, policy)), 1.0) + lsgan(D_B(diff_augment(fb_p, policy)), 0.0))
                loss_D_A = 0.5 * (lsgan(D_A(diff_augment(xa, policy)), 1.0) + lsgan(D_A(diff_augment(fa_p, policy)), 0.0))
                loss_D = loss_D_A + loss_D_B
            opt_D.zero_grad(set_to_none=True)
            loss_D.float().backward()
            r1_val = 0.0
            if tcfg.get("r1_gamma", 0) > 0 and step % tcfg.get("r1_every", 16) == 0:
                r1 = 0.5 * tcfg["r1_gamma"] * tcfg.get("r1_every", 16) * (r1_penalty(D_A, xa) + r1_penalty(D_B, xb))
                r1.backward()
                r1_val = float(r1)
            gnD = grad_norm(list(D_A.parameters()) + list(D_B.parameters()))
            torch.nn.utils.clip_grad_norm_(list(D_A.parameters()) + list(D_B.parameters()), tcfg.get("clip", 5.0))
            opt_D.step()

            if ema:
                for k in ema:
                    ema_update(ema[k], nets[k], ema_decay)
            if not np.isfinite(float(loss_G)) or not np.isfinite(float(loss_D)):
                nan_count += 1

            if tcfg.get("sample_every") and step % tcfg["sample_every"] == 0:
                # input / translation / cycle-reconstruction grids, both directions: lets a human see
                # whether outputs have structure and vary between samples (mode collapse check)
                with torch.no_grad():
                    G_AB.eval(), G_BA.eval()
                    sb = G_AB(xa); sa = G_BA(xb)
                    grid = torch.cat([xa, sb, G_BA(sb), xb, sa, G_AB(sa)], 0).float() * 0.5 + 0.5
                    save_image(grid.clamp(0, 1), os.path.join(sample_dir, f"step{step:07d}.png"), nrow=max(1, xa.size(0)))
                    G_AB.train(), G_BA.train()

            if step % tcfg.get("log_every", 200) == 0:
                for k, v in zip(history, [step, epoch, float(loss_G), float(adv), float(loss_D_A), float(loss_D_B),
                                          float(cyc), float(idt), float(low), r1_val, gnG, gnD]):
                    history[k].append(v)
                el = time.time() - t0
                msg = (f"epoch={epoch}/{sched['n']} step={step} lr_g={opt_G.param_groups[0]['lr']:.2e} "
                       f"lam_cyc={lam_c:.2f} loss_G={float(loss_G):.4f} adv={float(adv):.4f} cyc={float(cyc):.4f} "
                       f"idt={float(idt):.4f} low={float(low):.4f} r1={r1_val:.4f} "
                       f"D_A={float(loss_D_A):.4f} D_B={float(loss_D_B):.4f} gnG={gnG:.3f} gnD={gnD:.3f} "
                       f"elapsed={el/60:.1f}m")
                log.info(msg)
                print(f"[uvcgan] {msg}", flush=True)
            step += 1
            if max_steps and step >= max_steps:
                break

        epoch_s = time.time() - t_epoch
        if not sched["calibrated"]:
            eval_s = score_and_keep(epoch, tag=" [calibration]") if selector else 0.0
            n_ev = scfg.get("n_evals", 8)
            usable = 0.95 * (budget_s - (time.time() - t0) - n_ev * eval_s)
            n = int(max(tcfg.get("min_epochs", 10), min(tcfg.get("max_epochs", 1000), usable // max(1.0, epoch_s) + 1)))
            sched.update(n=n, decay=int(round(tcfg.get("decay_frac", 0.4) * n)),
                         **{"from": max(1, int(scfg.get("from_frac", 0.1) * n)),
                            "every": max(1, (n - int(scfg.get("from_frac", 0.1) * n)) // max(1, n_ev))},
                         calibrated=True)
            msg = (f"budget {budget_hours:.2f} h: epoch 0 took {epoch_s:.0f}s ({epoch_s/steps_per_epoch:.3f} s/step), "
                   f"one eval {eval_s:.0f}s -> {n} epochs x {steps_per_epoch} steps, LR decay from epoch "
                   f"{sched['decay']}, evals from epoch {sched['from']} every {sched['every']}")
            log.info(msg)
            print(f"[uvcgan] {msg}", flush=True)

        sch_G.step(), sch_D.step()
        if selector:
            for n_, m_ in nets.items():
                torch.save(m_.state_dict(), os.path.join(out_dir, f"{n_}_last.pt"))
            due = epoch >= sched["from"] and (epoch - sched["from"]) % sched["every"] == 0
            if due or epoch == sched["n"] - 1:
                score_and_keep(epoch)

        state = {n_: m_.state_dict() for n_, m_ in nets.items()}
        state.update({f"ema_{k}": m.state_dict() for k, m in ema.items()})
        state.update(opt_G=opt_G.state_dict(), opt_D=opt_D.state_dict(), sch_G=sch_G.state_dict(),
                     sch_D=sch_D.state_dict(), epoch=epoch, step=step, elapsed=time.time() - t0,
                     best=best, fid_history=fid_history, history=history, sched=sched)
        torch.save(state, state_path)
        if max_steps and step >= max_steps:
            break
        if budget_s and time.time() - t0 > budget_s:
            print("[uvcgan] budget reached", flush=True)
            break

    total = time.time() - t0
    metrics = {"model": "uvcgan2_style", "epochs": epoch + 1, "steps": step, "training_time_sec": total,
               "nan_count": nan_count, "peak_memory_mb": peak_memory_mb(device),
               "checkpoint_selection": {"best": best, "fid_history": fid_history}, "history": history,
               "config": cfg}
    with open(os.path.join(out_dir, "metrics_train.json"), "w") as f:
        json.dump(metrics, f, indent=2, default=float)
    done = f"best score {best['score']:.3f} at epoch {best['epoch']} ({best['weights']})" if selector else "no selection"
    log.info(f"done: {step} steps in {total/3600:.2f} h; {done}")
    print(f"[uvcgan] done: {step} steps in {total/3600:.2f} h; {done}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "config_uvcgan.yaml"))
    ap.add_argument("--budget_hours", type=float, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_steps", type=int, default=None, help="smoke test: stop after N steps")
    a = ap.parse_args()
    main(a.config, a.budget_hours, a.resume, a.max_steps)
