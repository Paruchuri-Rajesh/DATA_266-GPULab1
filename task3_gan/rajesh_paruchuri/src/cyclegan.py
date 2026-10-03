"""CycleGAN for Monet <-> photo (Rajesh Paruchuri), written from scratch.

Domains: A = Monet paintings (monet_jpg/), B = photos (photo_jpg/).
G_AB: Monet -> photo, G_BA: photo -> Monet (the Kaggle-scored direction).
D_A judges Monet images, D_B judges photos.

Design (differs from the standard 9-block / ngf-64 / transposed-conv CycleGAN of Zhu et al. 2017):
  * ResNet generator, ngf=48, 6 residual blocks, bilinear-upsample + conv decoder
  * 70x70 PatchGAN discriminators with spectral normalisation (no norm layers)
  * LSGAN adversarial loss + L1 cycle (lambda 10) + L1 identity (lambda 5)
  * DiffAugment (colour / translation / cutout) on every discriminator input,
    because the Monet domain has only 300 images and D overfits to them fast
  * 50-image replay buffer, EMA copies of both generators used for inference
  * trained on random 128x128 crops (fully convolutional -> inference at 256x256)
No pretrained network is used anywhere in training or generation.
"""

import json
import math
import os
import platform
import random
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# --------------------------------------------------------------------------
# paths / config / environment
# --------------------------------------------------------------------------
def find_paths(start=None):
    cwd = Path(start or Path.cwd()).resolve()
    candidates = [cwd, cwd / "src", cwd / "task3_gan" / "rajesh_paruchuri" / "src"]
    src = next((c for c in candidates if (c / "config.json").exists() and (c / "cyclegan.py").exists()), None)
    if src is None:
        raise FileNotFoundError("run from the repo root or task3_gan/rajesh_paruchuri/src")
    member = src.parent
    task = member.parent
    repo = task.parent
    paths = {"src": src, "member": member, "task": task, "repo": repo, "data": task / "data",
             "data_proc": member / "data_processed", "ckpt": member / "checkpoints", "out": member / "outputs",
             "logs": repo / "reproducibility" / "raw_logs" / member.name / "task3_gan"}
    for k in ("data_proc", "ckpt", "out", "logs"):
        paths[k].mkdir(parents=True, exist_ok=True)
    return paths


def rel(p, repo):
    p = Path(p).resolve()
    try:
        return str(p.relative_to(Path(repo).resolve()))
    except ValueError:
        return p.name


def load_config(paths):
    with open(paths["src"] / "config.json") as f:
        cfg = json.load(f)
    if os.environ.get("LAB_SMOKE") is not None:
        cfg["smoke"] = os.environ["LAB_SMOKE"] not in ("0", "false", "False", "")
    if os.environ.get("LAB_DATA_DIR"):  # optional override, e.g. synthetic smoke data
        cfg["data_dir_override"] = os.environ["LAB_DATA_DIR"]
    return cfg


def data_dir(cfg, paths):
    return Path(cfg["data_dir_override"]) if cfg.get("data_dir_override") else paths["data"]


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hardware_string(device):
    if device.type == "cuda":
        return "CUDA: " + torch.cuda.get_device_name(0)
    chip = platform.machine()
    if sys.platform == "darwin":
        import subprocess
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True).stdout.strip() or chip
        except Exception:
            pass
    return ("MPS (Apple GPU) on " if device.type == "mps" else "CPU: ") + chip


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def device_memory_mb(device, driver=False):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device.type == "mps":
        return (torch.mps.driver_allocated_memory() if driver else torch.mps.current_allocated_memory()) / 2**20
    return 0.0


def process_peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
IMG_EXT = (".jpg", ".jpeg", ".png")


def list_images(folder):
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXT)


def load_image(path, size=256):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    if im.size != (size, size):
        im = im.resize((size, size), Image.BICUBIC)
    return im


def to_tensor(im):
    """PIL -> float tensor in [-1, 1], shape (3, H, W)."""
    a = np.asarray(im, dtype=np.float32) / 127.5 - 1.0
    return torch.from_numpy(a).permute(2, 0, 1).contiguous()


def to_uint8(t):
    """(3, H, W) in [-1, 1] -> HxWx3 uint8."""
    return ((t.clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()


class UnpairedCrops(torch.utils.data.Dataset):
    """Each item: one random Monet crop and one random photo crop (independent indices, so unpaired).
    The 300 Monet images are cached as uint8; photos are decoded on demand (caching all 7,038 would cost ~1.4 GB RAM)."""

    def __init__(self, files_a, files_b, crop, length, seed):
        self.a = [np.asarray(load_image(p), dtype=np.uint8) for p in files_a]
        self.files_b = files_b
        self.crop, self.length = crop, length
        self.rng = random.Random(seed)

    def __len__(self):
        return self.length

    def _photo(self, i):
        return np.asarray(load_image(self.files_b[i]), dtype=np.uint8)

    def _aug(self, arr):
        H, W, _ = arr.shape
        c = self.crop
        y, x = self.rng.randint(0, H - c), self.rng.randint(0, W - c)
        arr = arr[y:y + c, x:x + c]
        if self.rng.random() < 0.5:
            arr = arr[:, ::-1]
        t = torch.from_numpy(np.ascontiguousarray(arr)).float().permute(2, 0, 1) / 127.5 - 1.0
        return t

    def __getitem__(self, _):
        a = self.a[self.rng.randrange(len(self.a))]
        b = self._photo(self.rng.randrange(len(self.files_b)))
        return self._aug(a), self._aug(b)


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.body = nn.Sequential(
            nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c), nn.ReLU(True),
            nn.ReflectionPad2d(1), nn.Conv2d(c, c, 3), nn.InstanceNorm2d(c))

    def forward(self, x):
        return x + self.body(x)


class Generator(nn.Module):
    """c7s1-48, d96, d192, 6 x R192, u96, u48, c7s1-3 (tanh).
    Upsampling is bilinear resize + 3x3 conv instead of transposed conv, which avoids checkerboard artifacts."""

    def __init__(self, ngf=48, n_blocks=6):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), nn.Conv2d(3, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True)]
        c = ngf
        for _ in range(2):
            layers += [nn.Conv2d(c, c * 2, 3, 2, 1), nn.InstanceNorm2d(c * 2), nn.ReLU(True)]
            c *= 2
        layers += [ResBlock(c) for _ in range(n_blocks)]
        for _ in range(2):
            layers += [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                       nn.Conv2d(c, c // 2, 3, 1, 1), nn.InstanceNorm2d(c // 2), nn.ReLU(True)]
            c //= 2
        layers += [nn.ReflectionPad2d(3), nn.Conv2d(c, 3, 7), nn.Tanh()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class PatchDiscriminator(nn.Module):
    """70x70 PatchGAN (C64-C128-C256-C512 -> 1) with spectral norm; outputs one real/fake score per patch."""

    def __init__(self, ndf=64):
        super().__init__()
        sn = spectral_norm
        self.net = nn.Sequential(
            sn(nn.Conv2d(3, ndf, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(ndf, ndf * 2, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(ndf * 4, ndf * 8, 4, 1, 1)), nn.LeakyReLU(0.2, True),
            sn(nn.Conv2d(ndf * 8, 1, 4, 1, 1)))

    def forward(self, x):
        return self.net(x)


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def build_models(cfg, device):
    G_AB = Generator(cfg["ngf"], cfg["n_blocks"])
    G_BA = Generator(cfg["ngf"], cfg["n_blocks"])
    D_A, D_B = PatchDiscriminator(cfg["ndf"]), PatchDiscriminator(cfg["ndf"])
    for m in (G_AB, G_BA, D_A, D_B):
        m.apply(init_weights)
    return [m.to(device) for m in (G_AB, G_BA, D_A, D_B)]


def count_params(*models):
    return sum(p.numel() for m in models for p in m.parameters())


# --------------------------------------------------------------------------
# DiffAugment (Zhao et al., 2020): differentiable, applied identically to real and fake D inputs
# --------------------------------------------------------------------------
def diff_augment(x, policy=("color", "translation", "cutout")):
    if "color" in policy:
        B = x.size(0)
        x = x + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5)                      # brightness
        m = x.mean(dim=1, keepdim=True)
        x = (x - m) * (torch.rand(B, 1, 1, 1, device=x.device) * 2) + m              # saturation
        m = x.mean(dim=[1, 2, 3], keepdim=True)
        x = (x - m) * (torch.rand(B, 1, 1, 1, device=x.device) + 0.5) + m            # contrast
    if "translation" in policy:
        B, C, H, W = x.shape
        sh, sw = int(H * 0.125 + 0.5), int(W * 0.125 + 0.5)
        ty = torch.randint(-sh, sh + 1, (B, 1, 1), device=x.device)
        tx = torch.randint(-sw, sw + 1, (B, 1, 1), device=x.device)
        gy, gx = torch.meshgrid(torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij")
        gy = torch.clamp(gy.unsqueeze(0) + ty + 1, 0, H + 1)
        gx = torch.clamp(gx.unsqueeze(0) + tx + 1, 0, W + 1)
        xp = F.pad(x, [1, 1, 1, 1])
        x = xp.permute(0, 2, 3, 1).contiguous()[torch.arange(B, device=x.device)[:, None, None], gy, gx]
        x = x.permute(0, 3, 1, 2).contiguous()
    if "cutout" in policy:
        B, C, H, W = x.shape
        ch, cw = int(H * 0.5 + 0.5), int(W * 0.5 + 0.5)
        oy = torch.randint(0, H + (1 - ch % 2), (B, 1, 1), device=x.device)
        ox = torch.randint(0, W + (1 - cw % 2), (B, 1, 1), device=x.device)
        gy, gx = torch.meshgrid(torch.arange(ch, device=x.device), torch.arange(cw, device=x.device), indexing="ij")
        gy = torch.clamp(gy.unsqueeze(0) + oy - ch // 2, 0, H - 1)
        gx = torch.clamp(gx.unsqueeze(0) + ox - cw // 2, 0, W - 1)
        mask = torch.ones(B, H, W, device=x.device)
        mask[torch.arange(B, device=x.device)[:, None, None], gy, gx] = 0
        x = x * mask.unsqueeze(1)
    return x


class ReplayBuffer:
    """Shrivastava et al. 2017: show D a mix of new and past fakes so it cannot just track the latest G."""

    def __init__(self, size=50):
        self.size, self.data = size, []

    def push_pop(self, x):
        out = []
        for img in x.detach():
            img = img.unsqueeze(0)
            if len(self.data) < self.size:
                self.data.append(img)
                out.append(img)
            elif random.random() < 0.5:
                i = random.randrange(self.size)
                out.append(self.data[i].clone())
                self.data[i] = img
            else:
                out.append(img)
        return torch.cat(out, 0)


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, p in zip(ema.parameters(), model.parameters()):
        pe.mul_(decay).add_(p.detach(), alpha=1 - decay)
    for be, b in zip(ema.buffers(), model.buffers()):
        be.copy_(b)


def grad_norm(params):
    norms = [p.grad.detach().norm(2) for p in params if p.grad is not None]
    return float(torch.norm(torch.stack(norms), 2)) if norms else 0.0


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------
def train(cfg, paths, device, tag):
    import copy
    smoke = tag == "smoke"
    root = data_dir(cfg, paths)
    files_a, files_b = list_images(root / "monet_jpg"), list_images(root / "photo_jpg")
    if smoke:
        files_a, files_b = files_a[:cfg["smoke_n_monet"]], files_b[:cfg["smoke_n_photo"]]
    iters_per_epoch = cfg["iters_per_epoch"] if not smoke else cfg["smoke_iters_per_epoch"]
    epochs = cfg["epochs"] if not smoke else cfg["smoke_epochs"]
    decay_start = cfg["decay_start_epoch"] if not smoke else max(1, epochs // 2)
    total = epochs * iters_per_epoch
    set_seed(cfg["seed"])
    ds = UnpairedCrops(files_a, files_b, cfg["crop"], total * cfg["batch_size"], cfg["seed"])
    dl = torch.utils.data.DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    G_AB, G_BA, D_A, D_B = build_models(cfg, device)
    E_AB, E_BA = copy.deepcopy(G_AB).eval(), copy.deepcopy(G_BA).eval()
    opt_G = torch.optim.Adam(list(G_AB.parameters()) + list(G_BA.parameters()), cfg["lr"], betas=tuple(cfg["betas"]))
    opt_D = torch.optim.Adam(list(D_A.parameters()) + list(D_B.parameters()), cfg["lr"], betas=tuple(cfg["betas"]))
    lr_lambda = lambda it: 1.0 if it < decay_start * iters_per_epoch else max(
        0.0, 1.0 - (it - decay_start * iters_per_epoch) / max(1, total - decay_start * iters_per_epoch))
    sch_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda)
    sch_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_lambda)
    buf_A, buf_B = ReplayBuffer(cfg["pool_size"]), ReplayBuffer(cfg["pool_size"])
    mse, l1 = nn.MSELoss(), nn.L1Loss()
    lam_c, lam_i = cfg["lambda_cycle"], cfg["lambda_identity"]
    aug = (lambda x: diff_augment(x)) if cfg["diffaugment"] else (lambda x: x)

    # fixed preview images (full 256x256, never seen as training crops in this exact form)
    prev_a = torch.stack([to_tensor(load_image(p)) for p in files_a[:3]]).to(device)
    prev_b = torch.stack([to_tensor(load_image(p)) for p in files_b[:3]]).to(device)

    hist = {k: [] for k in ("iter", "loss_G", "loss_G_adv", "loss_cycle", "loss_identity", "loss_D_A", "loss_D_B",
                            "D_A_real", "D_A_fake", "D_B_real", "D_B_fake", "grad_norm_G", "grad_norm_D", "lr")}
    nan_count, peak, peak_drv, t_train = 0, 0.0, 0.0, 0.0
    log_path = paths["logs"] / f"train_{tag}.log"
    run = {k: 0.0 for k in hist if k not in ("iter", "lr")}
    n_run = 0
    t0 = time.time()
    with open(log_path, "w") as log:
        def emit(m):
            print(m, flush=True)
            log.write(m + "\n")
            log.flush()

        emit(f"run_start={time.strftime('%Y-%m-%d %H:%M:%S')} member={cfg['member']} smoke={smoke}")
        emit(f"device={device} hardware={hardware_string(device)} torch={torch.__version__} python={platform.python_version()}")
        emit(f"n_monet={len(files_a)} n_photo={len(files_b)} crop={cfg['crop']} batch={cfg['batch_size']} epochs={epochs} "
             f"iters_per_epoch={iters_per_epoch} total_iters={total} decay_start_epoch={decay_start}")
        emit(f"G params (each)={count_params(G_AB)} D params (each)={count_params(D_A)} "
             f"total={count_params(G_AB, G_BA, D_A, D_B)} config={json.dumps({k: cfg[k] for k in ('ngf','n_blocks','ndf','lr','betas','lambda_cycle','lambda_identity','pool_size','diffaugment','ema_decay')})}")
        it = 0
        t_ep = time.time()
        for real_a, real_b in dl:
            real_a, real_b = real_a.to(device), real_b.to(device)
            # ---- generators ----
            for p in list(D_A.parameters()) + list(D_B.parameters()):
                p.requires_grad_(False)
            fake_b, fake_a = G_AB(real_a), G_BA(real_b)
            rec_a, rec_b = G_BA(fake_b), G_AB(fake_a)
            pa, pb = D_A(aug(fake_a)), D_B(aug(fake_b))
            adv = mse(pa, torch.ones_like(pa)) + mse(pb, torch.ones_like(pb))
            cyc = l1(rec_a, real_a) + l1(rec_b, real_b)
            idt = (l1(G_BA(real_a), real_a) + l1(G_AB(real_b), real_b)) if lam_i > 0 else torch.zeros((), device=device)
            loss_G = adv + lam_c * cyc + lam_i * idt
            opt_G.zero_grad(set_to_none=True)
            loss_G.backward()
            gn_G = grad_norm(list(G_AB.parameters()) + list(G_BA.parameters()))
            # ---- discriminators ----
            for p in list(D_A.parameters()) + list(D_B.parameters()):
                p.requires_grad_(True)
            fa, fb = buf_A.push_pop(fake_a), buf_B.push_pop(fake_b)
            ra, fa_s = D_A(aug(real_a)), D_A(aug(fa))
            rb, fb_s = D_B(aug(real_b)), D_B(aug(fb))
            loss_DA = 0.5 * (mse(ra, torch.ones_like(ra)) + mse(fa_s, torch.zeros_like(fa_s)))
            loss_DB = 0.5 * (mse(rb, torch.ones_like(rb)) + mse(fb_s, torch.zeros_like(fb_s)))
            opt_D.zero_grad(set_to_none=True)
            (loss_DA + loss_DB).backward()
            gn_D = grad_norm(list(D_A.parameters()) + list(D_B.parameters()))
            vals = [v.item() for v in (loss_G, adv, cyc, idt, loss_DA, loss_DB)]
            if not all(math.isfinite(v) for v in vals + [gn_G, gn_D]):
                nan_count += 1
                opt_G.zero_grad(set_to_none=True)
                opt_D.zero_grad(set_to_none=True)
                emit(f"NAN_OR_INF iter={it} values={vals} gn_G={gn_G} gn_D={gn_D} (step skipped)")
            else:
                opt_G.step()
                opt_D.step()
                ema_update(E_AB, G_AB, cfg["ema_decay"])
                ema_update(E_BA, G_BA, cfg["ema_decay"])
            sch_G.step()
            sch_D.step()
            row = dict(zip(("loss_G", "loss_G_adv", "loss_cycle", "loss_identity", "loss_D_A", "loss_D_B"), vals))
            row.update(D_A_real=float(ra.mean()), D_A_fake=float(fa_s.mean()), D_B_real=float(rb.mean()),
                       D_B_fake=float(fb_s.mean()), grad_norm_G=gn_G, grad_norm_D=gn_D)
            for k, v in row.items():
                run[k] += v
            n_run += 1
            it += 1
            if it % cfg["log_every"] == 0 or it == total:
                peak = max(peak, device_memory_mb(device))
                peak_drv = max(peak_drv, device_memory_mb(device, driver=True))
                hist["iter"].append(it)
                hist["lr"].append(sch_G.get_last_lr()[0])
                for k in run:
                    hist[k].append(run[k] / n_run)
                emit(f"iter={it}/{total} " + " ".join(f"{k}={run[k] / n_run:.4f}" for k in run)
                     + f" lr={sch_G.get_last_lr()[0]:.6f} elapsed={time.time() - t0:.0f}s")
                run = {k: 0.0 for k in run}
                n_run = 0
            if it % iters_per_epoch == 0:
                sync(device)
                ep = it // iters_per_epoch
                ep_sec = time.time() - t_ep
                t_train += ep_sec
                emit(f"EPOCH {ep}/{epochs} epoch_sec={ep_sec:.1f} img_per_sec={iters_per_epoch * cfg['batch_size'] * 2 / ep_sec:.2f} "
                     f"nan={nan_count} peak_mps_tensor_mb={peak:.0f}")
                if ep % cfg["preview_every"] == 0 or ep == epochs:
                    save_preview(E_AB, E_BA, prev_a, prev_b, paths["out"] / "previews" / f"{tag}_epoch{ep:03d}.png")
                if ep % cfg["ckpt_every"] == 0 or ep == epochs:
                    save_ckpt(paths["ckpt"] / f"cyclegan_{tag}.pt", G_AB, G_BA, D_A, D_B, E_AB, E_BA, cfg, ep, it)
                if device.type == "mps":
                    torch.mps.empty_cache()
                t_ep = time.time()
        summary = {"total_training_time_sec": time.time() - t0, "train_loop_time_sec": t_train,
                   "iterations": total, "epochs": epochs, "images_per_sec": total * cfg["batch_size"] * 2 / t_train,
                   "nan_count": nan_count, "peak_mps_tensor_memory_mb": peak, "peak_mps_driver_memory_mb": peak_drv,
                   "peak_process_rss_mb": process_peak_rss_mb(), "params_G_each": count_params(G_AB),
                   "params_D_each": count_params(D_A), "params_total": count_params(G_AB, G_BA, D_A, D_B),
                   "grad_norm_G_mean": float(np.mean(hist["grad_norm_G"])), "grad_norm_G_max": float(np.max(hist["grad_norm_G"])),
                   "grad_norm_D_mean": float(np.mean(hist["grad_norm_D"])), "grad_norm_D_max": float(np.max(hist["grad_norm_D"])),
                   "hardware": hardware_string(device)}
        emit("SUMMARY " + json.dumps(summary))
        emit(f"run_end={time.strftime('%Y-%m-%d %H:%M:%S')}")
    with open(paths["out"] / f"train_history_{tag}.json", "w") as f:
        json.dump({"hist": hist, "summary": summary}, f)
    return hist, summary


@torch.no_grad()
def save_preview(E_AB, E_BA, a, b, path):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for x, G, H in ((a, E_AB, E_BA), (b, E_BA, E_AB)):
        y = G(x)
        r = H(y)
        rows.append(torch.cat([torch.cat([x[i], y[i], r[i]], 2) for i in range(x.size(0))], 2))
    Image.fromarray(to_uint8(torch.cat(rows, 1))).save(path)


def save_ckpt(path, G_AB, G_BA, D_A, D_B, E_AB, E_BA, cfg, epoch, it):
    torch.save({"G_AB": G_AB.state_dict(), "G_BA": G_BA.state_dict(), "D_A": D_A.state_dict(), "D_B": D_B.state_dict(),
                "EMA_G_AB": E_AB.state_dict(), "EMA_G_BA": E_BA.state_dict(), "config": cfg, "epoch": epoch, "iter": it}, path)


def load_generators(path, cfg, device, use_ema=True):
    ck = torch.load(path, map_location=device)
    G_AB, G_BA = Generator(cfg["ngf"], cfg["n_blocks"]).to(device), Generator(cfg["ngf"], cfg["n_blocks"]).to(device)
    G_AB.load_state_dict(ck["EMA_G_AB" if use_ema else "G_AB"])
    G_BA.load_state_dict(ck["EMA_G_BA" if use_ema else "G_BA"])
    return G_AB.eval(), G_BA.eval(), ck


@torch.no_grad()
def translate_folder(G, files, out_dir, device, batch=8):
    """Translate every image at full 256x256 and save as JPEG with the same file name."""
    from PIL import Image
    out_dir.mkdir(parents=True, exist_ok=True)
    t0, n = time.time(), 0
    for s in range(0, len(files), batch):
        x = torch.stack([to_tensor(load_image(p)) for p in files[s:s + batch]]).to(device)
        y = G(x)
        for p, img in zip(files[s:s + batch], y):
            Image.fromarray(to_uint8(img)).save(out_dir / (p.stem + ".jpg"), quality=95)
            n += 1
    sync(device)
    return n / (time.time() - t0)


if __name__ == "__main__":
    # python cyclegan.py <full|smoke>
    tag = sys.argv[1] if len(sys.argv) > 1 else "full"
    P = find_paths(Path(__file__).resolve().parent)
    CFG = load_config(P)
    CFG["smoke"] = tag == "smoke"
    train(CFG, P, pick_device(), tag)
