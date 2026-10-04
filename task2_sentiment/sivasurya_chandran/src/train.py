"""
Task 2: trains baseline + 2 experimental models with early stopping on
validation macro-F1, scores each on the test set with the shared
eval_metrics.full_report, runs McNemar (baseline vs each experimental),
and writes metrics, plots, predictions and a reproducibility manifest.

Run:
    python src/train.py --config config.yaml [--only baseline]
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from common.utils import hardware_info, peak_memory_mb, reset_peak_memory, resolve_device, set_seed, write_manifest  # noqa: E402
from eval_metrics import full_report, mcnemar_test, save_plots  # noqa: E402
from model import build_model  # noqa: E402


def load_split(data_dir, name):
    return (
        torch.from_numpy(np.load(os.path.join(data_dir, f"X_{name}.npy"))).long(),
        torch.from_numpy(np.load(os.path.join(data_dir, f"len_{name}.npy"))).long(),
        torch.from_numpy(np.load(os.path.join(data_dir, f"y_{name}.npy"))).long(),
    )


def batches(X, L, y, bs, shuffle, gen=None):
    idx = torch.randperm(len(X), generator=gen) if shuffle else torch.arange(len(X))
    for i in range(0, len(X), bs):
        j = idx[i : i + bs]
        maxlen = int(L[j].max())
        yield X[j, :maxlen], L[j], y[j]


@torch.no_grad()
def predict(model, X, L, y, bs, device):
    model.eval()
    probs = []
    t0 = time.time()
    for xb, lb, _ in batches(X, L, y, bs, shuffle=False):
        probs.append(torch.softmax(model(xb.to(device), lb.to(device)), dim=1)[:, 1].float().cpu())
    elapsed = time.time() - t0
    model.train()
    return torch.cat(probs).numpy(), len(X) / elapsed


def train_one(name, mcfg, tcfg, splits, vocab_size, device, log, out_dir, seed):
    set_seed(seed)
    (Xtr, Ltr, ytr), (Xva, Lva, yva) = splits["train"], splits["val"]
    model = build_model(mcfg, vocab_size).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=mcfg.get("lr", 1e-3), weight_decay=tcfg.get("weight_decay", 0.0))
    loss_fn = nn.CrossEntropyLoss()
    gen = torch.Generator().manual_seed(seed)
    ckpt = os.path.join(out_dir, f"{name}.pt")

    log.info(f"[{name}] config={json.dumps(mcfg)} params={params}")
    print(f"\n=== {name} ({mcfg['type']}) params={params:,} ===", flush=True)
    reset_peak_memory(device)
    history, best_f1, bad_epochs = [], -1.0, 0
    t0, seen = time.time(), 0
    for epoch in range(tcfg["epochs"]):
        tot, n = 0.0, 0
        for xb, lb, yb in batches(Xtr, Ltr, ytr, tcfg["batch_size"], True, gen):
            xb, lb, yb = xb.to(device), lb.to(device), yb.to(device)
            loss = loss_fn(model(xb, lb), yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), tcfg.get("grad_clip", 5.0))
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)
            seen += len(yb)
        vprob, _ = predict(model, Xva, Lva, yva, 512, device)
        vpred = (vprob >= 0.5).astype(int)
        vf1 = f1_score(yva.numpy(), vpred, average="macro")
        vacc = float((vpred == yva.numpy()).mean())
        history.append({"epoch": epoch, "train_loss": tot / n, "val_acc": vacc, "val_macro_f1": vf1})
        msg = f"[{name}] epoch={epoch} train_loss={tot / n:.4f} val_acc={vacc:.4f} val_macro_f1={vf1:.4f}"
        log.info(msg)
        print(msg, flush=True)
        if vf1 > best_f1:
            best_f1, bad_epochs = vf1, 0
            torch.save(model.state_dict(), ckpt)
        else:
            bad_epochs += 1
            if bad_epochs >= tcfg.get("patience", 2):
                log.info(f"[{name}] early stop at epoch {epoch}")
                break
    train_time = time.time() - t0
    mem = peak_memory_mb(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    return model, {
        "type": mcfg["type"],
        "hyperparameters": mcfg,
        "checkpoint": ckpt,
        "param_count": params,
        "training_time_sec": train_time,
        "train_examples_per_sec": seen / train_time,
        "peak_memory_mb": mem,
        "best_val_macro_f1": best_f1,
        "history": history,
    }


def main(cfg_path, only):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    dcfg, tcfg = cfg["data"], cfg["train"]
    device = resolve_device(tcfg["device"])
    data_dir, out_dir = dcfg.get("data_dir", "data"), tcfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(tcfg["log_file"]), exist_ok=True)
    logging.basicConfig(filename=tcfg["log_file"], level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger("task2")
    hw = hardware_info(device)
    log.info(f"hardware={hw}")

    with open(os.path.join(data_dir, "vocab.json")) as f:
        vocab_size = len(json.load(f))
    splits = {s: load_split(data_dir, s) for s in ("train", "val", "test")}
    with open(os.path.join(data_dir, "test_slices.json")) as f:
        slices = json.load(f)
    Xte, Lte, yte = splits["test"]
    y_true = yte.numpy()

    names = [only] if only else list(cfg["models"])
    reports, curves, preds = {}, {}, {}
    for name in names:
        model, info = train_one(name, cfg["models"][name], tcfg, splits, vocab_size, device, log, out_dir, dcfg["seed"])
        prob, infer_eps = predict(model, Xte, Lte, yte, 512, device)
        pred = (prob >= 0.5).astype(int)
        rep = full_report(y_true, pred, prob, slices)
        rep.update(info)
        rep["inference_examples_per_sec"] = infer_eps
        rep["hardware"] = hw
        reports[name] = rep
        curves[name] = (y_true, prob)
        preds[name] = pred
        np.savez(os.path.join(out_dir, f"predictions_{name}.npz"), y_true=y_true, y_prob=prob)
        log.info(f"[{name}] test={json.dumps({k: v for k, v in rep.items() if k not in ('history', 'slices')})}")
        print(f"[{name}] test acc={rep['accuracy']:.4f} macroF1={rep['f1_macro']:.4f} AUC={rep['roc_auc']:.4f} MCC={rep['mcc']:.4f} ECE={rep['ece']:.4f}")

    mcnemar = {}
    base_file = os.path.join(out_dir, "predictions_baseline.npz")
    if os.path.exists(base_file):
        base_pred = (np.load(base_file)["y_prob"] >= 0.5).astype(int)
        for name in preds:
            if name != "baseline":
                mcnemar[f"baseline_vs_{name}"] = mcnemar_test(y_true, base_pred, preds[name])

    save_plots(curves, os.path.join(out_dir, "test" if not only else f"test_{only}"))
    metrics_path = os.path.join(out_dir, "metrics_task2.json" if not only else f"metrics_task2_{only}.json")
    with open(metrics_path, "w") as f:
        json.dump({"models": reports, "mcnemar": mcnemar}, f, indent=2)
    write_manifest(os.path.join(out_dir, "manifest.json"), cfg, device, {n: r["checkpoint"] for n, r in reports.items()}, metrics_path)
    print(json.dumps(mcnemar, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--only", default=None, help="train a single model by name")
    a = ap.parse_args()
    main(a.config, a.only)
