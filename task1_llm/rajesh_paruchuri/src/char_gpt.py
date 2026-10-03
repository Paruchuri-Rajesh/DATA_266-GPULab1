"""Character-level GPT on TinyStories, written from scratch (Rajesh Paruchuri).

Everything a Transformer needs is built here by hand: LayerNorm, causal
multi-head self-attention, the feed-forward block, residual wiring, token and
positional embeddings and the LM head. No nn.Transformer*, nn.MultiheadAttention
or F.scaled_dot_product_attention is used anywhere.

The notebook part1_llm.ipynb drives this module; every function here is
config-driven and writes only repo-relative paths.
"""

import json
import math
import os
import platform
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EOS = "\x04"  # end-of-story marker appended after every story
UNK = "\x00"  # any character rarer than min_char_count
DISPLAY = {EOS: "<|eos|>", UNK: "<unk>"}

# TinyStories mixes curly quotes / dashes with ASCII ones; fold them so the
# vocab is not split across visually identical characters.
CHAR_FOLD = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "…": "...", " ": " ",
}
# ~6% of stories in the shard carry UTF-8-read-as-cp1252 mojibake ("â€œ" for a quote).
# Longer sequences first; the bare "â€" (closing quote whose last byte was lost) last.
MOJIBAKE_FOLD = [
    ("â€œ", '"'), ("â€\u009d", '"'), ("â€™", "'"),
    ("â€˜", "'"), ("â€“", "-"), ("â€”", "-"),
    ("â€¦", "..."), ("â€‹", ""), ("â€Š", ""),
    ("â€", '"'),
]


# --------------------------------------------------------------------------
# paths / config / environment
# --------------------------------------------------------------------------
def find_paths(start=None):
    """Locate member/task/repo folders from wherever the notebook is opened."""
    cwd = Path(start or Path.cwd()).resolve()
    candidates = [cwd, cwd / "src", cwd / "task1_llm" / "rajesh_paruchuri" / "src"]
    src = next((c for c in candidates if (c / "config.json").exists() and (c / "char_gpt.py").exists()), None)
    if src is None:
        raise FileNotFoundError("run from the repo root or task1_llm/rajesh_paruchuri/src")
    member = src.parent
    task = member.parent
    repo = task.parent
    paths = {
        "src": src, "member": member, "task": task, "repo": repo,
        "data_raw": task / "data", "data_proc": member / "data_processed",
        "ckpt": member / "checkpoints", "out": member / "outputs",
        "logs": repo / "reproducibility" / "raw_logs" / member.name / "task1_llm",
    }
    for k in ("data_raw", "data_proc", "ckpt", "out", "logs"):
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
    return cfg


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hardware_string(device):
    if device.type == "cuda":
        return "CUDA: " + torch.cuda.get_device_name(0)
    chip = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        try:
            import subprocess
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip() or chip
        except Exception:
            pass
    return ("MPS (Apple GPU) on " if device.type == "mps" else "CPU: ") + chip


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# --------------------------------------------------------------------------
# 1.1 data preprocessing
# --------------------------------------------------------------------------
def download_if_missing(cfg, paths):
    raw_path = paths["data_raw"] / cfg["dataset_file"]
    if not raw_path.exists():
        import urllib.request
        print("downloading", cfg["dataset_file"])
        try:
            urllib.request.urlretrieve(cfg["dataset_url"], raw_path)
        except Exception:
            # python.org builds on macOS often lack root certs; curl uses the system store
            import subprocess
            subprocess.run(["curl", "-sL", "-o", str(raw_path), cfg["dataset_url"]], check=True)
    return raw_path


def load_stories(raw_path, min_chars):
    import pandas as pd
    df = pd.read_parquet(raw_path, columns=["text"])
    n_raw = len(df)
    texts = df["text"].astype(str).tolist()
    stories, n_short, n_mojibake = [], 0, 0
    for t in texts:
        if "\u00e2\u20ac" in t:
            n_mojibake += 1
            for a, b in MOJIBAKE_FOLD:
                t = t.replace(a, b)
        for a, b in CHAR_FOLD.items():
            t = t.replace(a, b)
        t = t.replace(EOS, "").replace(UNK, "").strip()
        if len(t) < min_chars:
            n_short += 1
            continue
        stories.append(t)
    return stories, {"n_raw": n_raw, "n_mojibake_repaired": n_mojibake,
                     "n_dropped_short_or_empty": n_short, "n_kept": len(stories)}


def build_streams(stories, n_train, n_val, block_size, seed):
    """Story-disjoint split. Shuffle stories, fill val first, then train."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(stories))
    win = block_size + 1
    need_val, need_train = n_val * win, n_train * win

    def take(start, need):
        parts, total, i = [], 0, start
        while total < need:
            if i >= len(order):
                raise ValueError("not enough stories for the requested windows")
            s = stories[order[i]] + EOS
            parts.append(s)
            total += len(s)
            i += 1
        return "".join(parts), i

    val_text, nxt = take(0, need_val)
    train_text, end = take(nxt, need_train)
    info = {"val_stories": nxt, "train_stories": end - nxt,
            "val_chars": len(val_text), "train_chars": len(train_text)}
    return train_text, val_text, info


def build_vocab(train_text, min_count):
    counts = Counter(train_text)
    chars = sorted(c for c, n in counts.items() if n >= min_count and c not in (EOS, UNK))
    chars = [UNK, EOS] + chars
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    rare = {c: n for c, n in counts.items() if n < min_count}
    return char_to_idx, idx_to_char, rare


def encode(text, char_to_idx):
    unk = char_to_idx[UNK]
    # vectorised lookup through a code-point table (fast for ~20M chars)
    cps = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    table = np.full(int(cps.max()) + 1, unk, dtype=np.int64)
    for c, i in char_to_idx.items():
        if ord(c) < len(table):
            table[ord(c)] = i
    return table[cps]


def decode(ids, idx_to_char, show_special=True):
    out = []
    for i in ids:
        c = idx_to_char[int(i)]
        out.append(DISPLAY.get(c, c) if show_special else c)
    return "".join(out)


def make_windows(ids, n_windows, block_size):
    """Non-overlapping (block_size+1) chunks -> x = chunk[:-1], y = chunk[1:]."""
    win = block_size + 1
    chunks = ids[: n_windows * win].reshape(n_windows, win)
    return chunks[:, :-1].copy(), chunks[:, 1:].copy()


# --------------------------------------------------------------------------
# 1.2 model
# --------------------------------------------------------------------------
class LayerNorm(nn.Module):
    """y = (x - mean) / sqrt(var + eps) * gamma + beta over the last dim."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.gamma + self.beta


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0, "n_embd must divide by n_head"
        self.n_head, self.head_dim = n_head, n_embd // n_head
        self.qkv = nn.Linear(n_embd, 3 * n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        # True above the diagonal = a future position that must be hidden
        self.register_buffer("future", torch.triu(torch.ones(block_size, block_size, dtype=torch.bool), 1),
                             persistent=False)

    def forward(self, x, return_attn=False):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, h, T, d)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, h, T, T)
        scores = scores.masked_fill(self.future[:T, :T], float("-inf"))
        weights = torch.softmax(scores.float(), dim=-1).to(q.dtype)
        y = self.attn_drop(weights) @ v  # (B, h, T, d)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_drop(self.proj(y))
        return (y, weights) if return_attn else y


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd)
        self.proj = nn.Linear(4 * n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.proj(F.gelu(self.fc(x))))


class Block(nn.Module):
    """Pre-norm block: x + Attn(LN(x)), then x + FFN(LN(x))."""

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = LayerNorm(n_embd)
        self.ffn = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class CharGPT(nn.Module):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying
        self.apply(self._init)
        # GPT-2: shrink the residual-branch output projections by 1/sqrt(2L)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, "sequence longer than block_size"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def num_params(self):
        return sum(p.numel() for p in self.parameters())  # tied weight counted once


def causal_mask_check(model, device, vocab_size):
    """Changing a future token must not change earlier logits."""
    model.eval()
    T = min(32, model.block_size)
    a = torch.randint(0, vocab_size, (1, T), device=device)
    b = a.clone()
    b[0, T // 2:] = torch.randint(0, vocab_size, (T - T // 2,), device=device)
    with torch.no_grad():
        la, _ = model(a)
        lb, _ = model(b)
    return float((la[0, : T // 2] - lb[0, : T // 2]).abs().max())


# --------------------------------------------------------------------------
# 1.3 training
# --------------------------------------------------------------------------
def lr_at(step, warmup, total, peak, floor):
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def make_optimizer(model, cfg):
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg["peak_lr"], betas=tuple(cfg["betas"]))


def device_memory_mb(device):
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    return 0.0


def process_peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024  # bytes on macOS, KB on Linux


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def evaluate(model, X, Y, device, batch_size=128):
    """Mean CE (nats/char) and top-1 next-char accuracy over every position."""
    model.eval()
    tot_loss, tot_correct, tot_n = 0.0, 0, 0
    for s in range(0, X.size(0), batch_size):
        xb, yb = X[s:s + batch_size].to(device), Y[s:s + batch_size].to(device)
        logits, _ = model(xb)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1), reduction="sum")
        tot_loss += float(loss)
        tot_correct += int((logits.argmax(-1) == yb).sum())
        tot_n += yb.numel()
    return tot_loss / tot_n, tot_correct / tot_n


def train(model, data, cfg, device, log_path, ckpt_path, smoke):
    X_tr, Y_tr, X_va, Y_va = data
    bs = cfg["batch_size"]
    epochs = cfg["epochs_smoke"] if smoke else cfg["epochs_full"]
    warmup = cfg["warmup_steps_smoke"] if smoke else cfg["warmup_steps"]
    steps_per_epoch = math.ceil(X_tr.size(0) / bs)
    total_steps = epochs * steps_per_epoch
    opt = make_optimizer(model, cfg)

    # fixed train subset, scored with dropout off, for an honest train-vs-val gap
    g = torch.Generator().manual_seed(cfg["seed"])
    sub = torch.randperm(X_tr.size(0), generator=g)[: cfg["train_eval_subset"]]
    X_sub, Y_sub = X_tr[sub], Y_tr[sub]

    hist = {"step": [], "step_loss": [], "grad_norm": [], "lr": [],
            "epoch": [], "train_loss_running": [], "train_ce_eval": [], "val_ce": [],
            "val_acc": [], "epoch_time_sec": []}
    nan_count, spike_count, ema = 0, 0, None
    best_val, peak_dev_mb = float("inf"), 0.0
    train_compute_sec, tokens_seen, step = 0.0, 0, 0
    t_start = time.time()

    with open(log_path, "w") as log:
        def emit(msg):
            print(msg)
            log.write(msg + "\n")
            log.flush()

        emit(f"run_start={time.strftime('%Y-%m-%d %H:%M:%S')} member={cfg['member']} smoke={smoke}")
        emit(f"device={device} hardware={hardware_string(device)} torch={torch.__version__} python={platform.python_version()}")
        emit(f"n_layer={cfg['n_layer']} n_head={cfg['n_head']} n_embd={cfg['n_embd']} block_size={cfg['block_size']} "
             f"dropout={cfg['dropout']} params={model.num_params()}")
        emit(f"n_train={X_tr.size(0)} n_val={X_va.size(0)} batch_size={bs} epochs={epochs} steps={total_steps} "
             f"peak_lr={cfg['peak_lr']} min_lr={cfg['min_lr']} warmup={warmup} wd={cfg['weight_decay']} clip={cfg['grad_clip']}")

        for epoch in range(1, epochs + 1):
            model.train()
            t_ep = time.time()
            order = torch.randperm(X_tr.size(0))
            run_sum, run_n = 0.0, 0
            for s in range(0, X_tr.size(0), bs):
                ix = order[s:s + bs]
                xb, yb = X_tr[ix].to(device), Y_tr[ix].to(device)
                lr = lr_at(step, warmup, total_steps, cfg["peak_lr"], cfg["min_lr"])
                for gp in opt.param_groups:
                    gp["lr"] = lr
                _, loss = model(xb, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]))
                lval = loss.item()
                if not (math.isfinite(lval) and math.isfinite(gnorm)):
                    nan_count += 1  # skip the update, keep the weights finite
                    opt.zero_grad(set_to_none=True)
                else:
                    opt.step()
                    if ema is not None and step > warmup and lval > 1.5 * ema:
                        spike_count += 1
                        emit(f"LOSS_SPIKE step={step} loss={lval:.4f} ema={ema:.4f}")
                    ema = lval if ema is None else 0.98 * ema + 0.02 * lval
                    run_sum += lval * xb.size(0)
                    run_n += xb.size(0)
                tokens_seen += xb.numel()
                hist["step"].append(step)
                hist["step_loss"].append(lval)
                hist["grad_norm"].append(gnorm)
                hist["lr"].append(lr)
                if step % cfg["log_every"] == 0:
                    peak_dev_mb = max(peak_dev_mb, device_memory_mb(device))
                    emit(f"epoch={epoch} step={step}/{total_steps} loss={lval:.4f} grad_norm={gnorm:.3f} lr={lr:.6f} "
                         f"elapsed={time.time() - t_start:.0f}s")
                step += 1
            sync(device)
            ep_train_sec = time.time() - t_ep
            train_compute_sec += ep_train_sec
            peak_dev_mb = max(peak_dev_mb, device_memory_mb(device))

            if device.type == "mps":
                torch.mps.empty_cache()  # hand cached blocks back; unified memory is shared with the OS
            tr_ce, _ = evaluate(model, X_sub, Y_sub, device)
            va_ce, va_acc = evaluate(model, X_va, Y_va, device)
            hist["epoch"].append(epoch)
            hist["train_loss_running"].append(run_sum / max(1, run_n))
            hist["train_ce_eval"].append(tr_ce)
            hist["val_ce"].append(va_ce)
            hist["val_acc"].append(va_acc)
            hist["epoch_time_sec"].append(ep_train_sec)
            if device.type == "mps":
                torch.mps.empty_cache()
            is_best = va_ce < best_val
            if is_best:
                best_val = va_ce
                torch.save({"model": model.state_dict(), "config": cfg, "epoch": epoch,
                            "val_ce": va_ce, "vocab_size": model.tok_emb.num_embeddings}, ckpt_path)
            emit(f"EPOCH {epoch} train_running_ce={run_sum / max(1, run_n):.4f} train_eval_ce={tr_ce:.4f} "
                 f"val_ce={va_ce:.4f} val_ppl={math.exp(va_ce):.3f} val_bpc={va_ce / math.log(2):.4f} "
                 f"val_top1={va_acc:.4f} epoch_sec={ep_train_sec:.1f} tok_per_sec={X_tr.numel() / ep_train_sec:.0f} "
                 f"nan={nan_count} spikes={spike_count} best={'yes' if is_best else 'no'}")

        total_sec = time.time() - t_start
        summary = {
            "total_training_time_sec": total_sec,
            "train_compute_time_sec": train_compute_sec,
            "train_tokens_per_sec": tokens_seen / train_compute_sec,
            "tokens_seen": tokens_seen,
            "nan_count": nan_count,
            "loss_spike_count": spike_count,
            "grad_norm_mean": float(np.mean(hist["grad_norm"])),
            "grad_norm_max": float(np.max(hist["grad_norm"])),
            "grad_norm_p99": float(np.percentile(hist["grad_norm"], 99)),
            "peak_device_memory_mb": peak_dev_mb,
            "peak_process_rss_mb": process_peak_rss_mb(),
            "best_val_ce": best_val,
            "epochs": epochs,
        }
        emit("SUMMARY " + json.dumps(summary))
        emit(f"run_end={time.strftime('%Y-%m-%d %H:%M:%S')}")
    return hist, summary


# --------------------------------------------------------------------------
# generation + generation metrics
# --------------------------------------------------------------------------
@torch.no_grad()
def generate(model, prompt_ids, max_new, device, temperature=0.0, eos_id=None, generator=None):
    """temperature == 0 -> greedy; otherwise sample from softmax(logits / T)."""
    model.eval()
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    new = 0
    for _ in range(max_new):
        logits, _ = model(idx[:, -model.block_size:])
        logits = logits[:, -1, :].float()
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1).cpu()
            nxt = torch.multinomial(probs, 1, generator=generator).to(device)
        idx = torch.cat([idx, nxt], dim=1)
        new += 1
        if eos_id is not None and int(nxt) == eos_id:
            break
    return idx[0].tolist(), new


def words(text):
    import re
    return re.findall(r"[a-z']+", text.lower())


def distinct_n(tokens, n):
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def repeated_ngram_rate(tokens, n=4):
    """Share of n-gram occurrences whose n-gram already appeared earlier in the same text."""
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    if not grams:
        return 0.0
    seen, rep = set(), 0
    for g in grams:
        rep += g in seen
        seen.add(g)
    return rep / len(grams)


def diversity_metrics(texts):
    """Distinct-n is pooled over all samples; repeated-4gram rate is the per-sample mean."""
    w = [words(t) for t in texts]
    pooled = [x for ws in w for x in ws]
    chars = list("".join(texts))
    return {
        "distinct_1": distinct_n(pooled, 1), "distinct_2": distinct_n(pooled, 2), "distinct_3": distinct_n(pooled, 3),
        "repeated_4gram_rate": float(np.mean([repeated_ngram_rate(ws, 4) for ws in w])),
        "char_distinct_3": distinct_n(chars, 3),
        "n_words": len(pooled),
    }
