"""
Task 1 training: cross-entropy loss, linear LR warmup + cosine decay,
unedited raw log, loss-curve plot, metrics JSON and reproducibility manifest.

Run:
    python src/train.py --config config.yaml            # from the member folder
    python src/train.py --config config.yaml --resume   # continue after an interruption (from checkpoints/train_state.pt)
"""
import argparse
import json
import logging
import math
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from common.utils import (  # noqa: E402
    hardware_info,
    peak_memory_mb,
    reset_peak_memory,
    resolve_device,
    set_seed,
    write_manifest,
)
from data import CharSequences  # noqa: E402
from model import GPT  # noqa: E402


def get_batch(data: np.ndarray, block_size: int, batch_size: int, device: str):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


def lr_at_step(step: int, warmup_steps: int, base_lr: float, total_steps: int, min_lr_ratio: float = 0.1) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
    min_lr = base_lr * min_lr_ratio
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss_and_acc(model, data, block_size, batch_size, device, iters, amp=False):
    model.eval()
    losses, correct, total = [], 0, 0
    for _ in range(iters):
        x, y = get_batch(data, block_size, batch_size, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits, loss = model(x, y)
        losses.append(loss.item())
        correct += (logits.argmax(-1) == y).sum().item()
        total += y.numel()
    model.train()
    return float(np.mean(losses)), correct / total


def plot_curves(history: dict, path: str, zoom_from: int = 2000):
    """Left: the whole run. Right: from `zoom_from` steps on, where the two curves sit close together.
    Training is drawn last and dashed so it stays visible where it overlaps validation."""
    steps, tr, va = np.array(history["step"]), np.array(history["train_loss"]), np.array(history["val_loss"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, keep in zip(axes, (np.ones_like(steps, dtype=bool), steps >= zoom_from)):
        ax.plot(steps[keep], va[keep], color="tab:orange", lw=1.6, label="validation")
        ax.plot(steps[keep], tr[keep], color="tab:blue", lw=1.3, ls="--", label="train")
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("cross-entropy loss (nats)")
    axes[0].set_title("Whole run")
    axes[1].set_title(f"From step {zoom_from:,} (zoomed)")
    axes[0].legend()
    fig.suptitle("Task 1 — training vs validation loss")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main(cfg_path: str, resume: bool = False):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]
    set_seed(dcfg["seed"])
    device = resolve_device(tcfg["device"])

    out_dir = tcfg["out_dir"]
    os.makedirs(os.path.dirname(tcfg["log_file"]), exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    logging.basicConfig(filename=tcfg["log_file"], level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("task1")
    log.info(f"config={json.dumps(cfg)}")
    log.info(f"hardware={hardware_info(device)}")

    data_dir = dcfg.get("data_dir", "data")
    with open(os.path.join(data_dir, "char_to_idx.json")) as f:
        vocab_size = len(json.load(f))
    train_data = np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint16)
    val_data = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint16)

    model = GPT(vocab_size, dcfg["block_size"], mcfg["n_embd"], mcfg["n_head"], mcfg["n_layer"], mcfg["dropout"]).to(device)
    param_count = model.num_params()
    log.info(f"param_count={param_count} vocab_size={vocab_size} device={device}")
    print(f"device={device} param_count={param_count:,} vocab_size={vocab_size}")

    # fp16 autocast + loss scaling on CUDA (the T4 has fp16 tensor cores but no bf16)
    amp = device == "cuda" and tcfg.get("amp", True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    log.info(f"amp_fp16={amp}")

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": tcfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
        lr=tcfg["lr"],
        betas=(0.9, 0.95),
    )

    train_seqs = CharSequences(train_data, dcfg["block_size"])
    rng = np.random.default_rng(dcfg["seed"])
    tokens_per_step = tcfg["batch_size"] * dcfg["block_size"]
    steps_per_epoch = train_seqs.steps_per_epoch(tcfg["batch_size"])
    total_steps = steps_per_epoch * tcfg["epochs"]
    eval_iters = tcfg.get("eval_iters", 50)
    log.info(f"steps_per_epoch={steps_per_epoch} total_steps={total_steps}")

    history = {"step": [], "epoch": [], "lr": [], "train_loss": [], "val_loss": [], "grad_norm": []}
    grad_norms, nan_steps, loss_spikes = [], 0, 0
    ema_loss = None
    best_val = float("inf")

    state_path = os.path.join(out_dir, "train_state.pt")
    start_epoch, step, prev_elapsed = 0, 0, 0.0
    if resume and os.path.exists(state_path):
        st = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"])
        optimizer.load_state_dict(st["optimizer"])
        scaler.load_state_dict(st["scaler"])
        rng.bit_generator.state = st["np_rng"]
        torch.set_rng_state(st["torch_rng"].cpu())  # map_location may have moved it to the GPU
        start_epoch, step, prev_elapsed = st["epoch"] + 1, st["step"], st["elapsed"]
        history, grad_norms, nan_steps, loss_spikes = st["history"], st["grad_norms"], st["nan_steps"], st["loss_spikes"]
        ema_loss, best_val = st["ema_loss"], st["best_val"]
        log.info(f"resumed from {state_path} at epoch={start_epoch} step={step}")
        print(f"resumed at epoch {start_epoch}, step {step}")

    reset_peak_memory(device)
    t0 = time.time() - prev_elapsed
    for epoch in range(start_epoch, tcfg["epochs"]):
        for x, y in train_seqs.epoch_batches(tcfg["batch_size"], rng, device):
            lr = lr_at_step(step, tcfg["warmup_steps"], tcfg["lr"], total_steps)
            for g in optimizer.param_groups:
                g["lr"] = lr

            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                _, loss = model(x, y)

            if not torch.isfinite(loss):
                nan_steps += 1
                log.info(f"step={step} non-finite loss, skipping update")
                optimizer.zero_grad(set_to_none=True)
                step += 1
                continue

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"]).item()
            grad_norms.append(gn)
            scaler.step(optimizer)  # skipped automatically if fp16 grads overflowed
            scaler.update()

            lv = loss.item()
            if ema_loss is not None and lv > 1.5 * ema_loss:
                loss_spikes += 1
                log.info(f"step={step} loss spike {lv:.4f} vs ema {ema_loss:.4f}")
            ema_loss = lv if ema_loss is None else 0.98 * ema_loss + 0.02 * lv

            if step % tcfg["eval_interval"] == 0 or step == total_steps - 1:
                tr_loss, _ = estimate_loss_and_acc(model, train_data, dcfg["block_size"], tcfg["batch_size"], device, eval_iters, amp)
                va_loss, _ = estimate_loss_and_acc(model, val_data, dcfg["block_size"], tcfg["batch_size"], device, eval_iters, amp)
                for k, v in zip(history, [step, epoch, lr, tr_loss, va_loss, gn]):
                    history[k].append(v)
                msg = (
                    f"epoch={epoch} step={step}/{total_steps} lr={lr:.2e} batch_loss={lv:.4f} "
                    f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} grad_norm={gn:.3f}"
                )
                el = time.time() - t0
                msg += f" elapsed={el / 60:.1f}m eta={el / (step + 1) * (total_steps - step - 1) / 60:.1f}m"
                log.info(msg)
                print(msg, flush=True)
                if va_loss < best_val:
                    best_val = va_loss
                    torch.save(model.state_dict(), os.path.join(out_dir, "gpt_best.pt"))
            step += 1

        torch.save(model.state_dict(), os.path.join(out_dir, f"gpt_epoch{epoch}.pt"))
        torch.save(
            {
                "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "np_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(), "epoch": epoch, "step": step,
                "elapsed": time.time() - t0, "history": history, "grad_norms": grad_norms, "nan_steps": nan_steps,
                "loss_spikes": loss_spikes, "ema_loss": ema_loss, "best_val": best_val,
            },
            state_path,
        )
        log.info(f"epoch={epoch} done, train_state saved")

    elapsed = time.time() - t0
    torch.save(model.state_dict(), os.path.join(out_dir, "gpt_final.pt"))

    final_iters = tcfg.get("final_eval_iters", 200)
    train_loss, train_acc = estimate_loss_and_acc(model, train_data, dcfg["block_size"], tcfg["batch_size"], device, final_iters, amp)
    val_loss, val_acc = estimate_loss_and_acc(model, val_data, dcfg["block_size"], tcfg["batch_size"], device, final_iters, amp)

    plot_path = os.path.join(out_dir, "loss_curves.png")
    plot_curves(history, plot_path)

    # fp16 loss-scale overflows give inf grad norms; the scaler skips those steps, so stats use finite norms only
    gn_all = np.array(grad_norms) if grad_norms else np.array([0.0])
    gn_arr = gn_all[np.isfinite(gn_all)] if np.isfinite(gn_all).any() else np.array([0.0])
    metrics = {
        "checkpoint": os.path.join(out_dir, "gpt_final.pt"),
        "param_count": param_count,
        "train_cross_entropy": train_loss,
        "val_cross_entropy": val_loss,
        "perplexity_val": math.exp(val_loss),
        "perplexity_train": math.exp(train_loss),
        "bits_per_character_val": val_loss / math.log(2),
        "generalization_gap": val_loss - train_loss,
        "top1_next_char_accuracy_val": val_acc,
        "top1_next_char_accuracy_train": train_acc,
        "grad_norm_mean": float(gn_arr.mean()),
        "grad_norm_max": float(gn_arr.max()),
        "grad_norm_p95": float(np.percentile(gn_arr, 95)),
        "nan_steps": nan_steps,
        "amp_overflow_skipped_steps": int((~np.isfinite(gn_all)).sum()),
        "loss_spikes": loss_spikes,
        "training_time_sec": elapsed,
        "training_tokens_per_sec": step * tokens_per_step / elapsed,
        "peak_memory_mb": peak_memory_mb(device),
        "epochs": tcfg["epochs"],
        "amp_fp16": amp,
        "total_steps": total_steps,
        "loss_curve_plot": plot_path,
        "history": history,
    }
    metrics_path = os.path.join(out_dir, "metrics_train.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    write_manifest(
        os.path.join(out_dir, "manifest.json"),
        cfg,
        device,
        {"final": metrics["checkpoint"], "best_val": os.path.join(out_dir, "gpt_best.pt")},
        metrics_path,
    )
    log.info(f"final_metrics={json.dumps({k: v for k, v in metrics.items() if k != 'history'})}")
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--resume", action="store_true", help="continue from checkpoints/train_state.pt")
    ap.add_argument("--replot", action="store_true", help="only redraw loss_curves.png from the saved metrics_train.json history")
    a = ap.parse_args()
    if a.replot:
        with open(a.config) as f:
            out = yaml.safe_load(f)["train"]["out_dir"]
        plot_curves(json.load(open(os.path.join(out, "metrics_train.json")))["history"], os.path.join(out, "loss_curves.png"))
    else:
        main(a.config, a.resume)
