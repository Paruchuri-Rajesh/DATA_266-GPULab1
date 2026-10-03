"""Yelp polarity bake-off (Rajesh Paruchuri): preprocessing, three from-scratch models, metrics.

No pretrained embeddings or language models: every nn.Embedding starts from random init and is
trained with its classifier. The notebook part2_sentiment.ipynb drives this module.
"""

import html
import json
import math
import os
import platform
import re
import resource
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# paths / config / environment
# --------------------------------------------------------------------------
def find_paths(start=None):
    cwd = Path(start or Path.cwd()).resolve()
    candidates = [cwd, cwd / "src", cwd / "task2_sentiment" / "rajesh_paruchuri" / "src"]
    src = next((c for c in candidates if (c / "config.json").exists() and (c / "sentiment_lib.py").exists()), None)
    if src is None:
        raise FileNotFoundError("run from the repo root or task2_sentiment/rajesh_paruchuri/src")
    member = src.parent
    task = member.parent
    repo = task.parent
    paths = {"src": src, "member": member, "task": task, "repo": repo,
             "data_raw": task / "data", "data_proc": member / "data_processed",
             "ckpt": member / "checkpoints", "out": member / "outputs",
             "logs": repo / "reproducibility" / "raw_logs" / member.name / "task2_sentiment"}
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
        import subprocess
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True).stdout.strip() or chip
        except Exception:
            pass
    return ("MPS (Apple GPU) on " if device.type == "mps" else "CPU: ") + chip


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def device_memory_mb(device, driver=False):
    """Tensor memory currently allocated on the GPU (driver=True: incl. the MPS cache pool)."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device.type == "mps":
        return (torch.mps.driver_allocated_memory() if driver else torch.mps.current_allocated_memory()) / 2**20
    return 0.0


def process_peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024


def download_if_missing(cfg, paths):
    out = {}
    for key in ("train", "test"):
        p = paths["data_raw"] / cfg[f"{key}_file"]
        if not p.exists():
            import subprocess
            print("downloading", cfg[f"{key}_file"])
            subprocess.run(["curl", "-sL", "-o", str(p), cfg[f"{key}_url"]], check=True)
        out[key] = p
    return out


# --------------------------------------------------------------------------
# 2.1 preprocessing
# --------------------------------------------------------------------------
# NLTK's English stopword list (nltk.corpus.stopwords, 179 words), embedded so the run
# needs no corpus download.
NLTK_STOPWORDS = set("""
i me my myself we our ours ourselves you you're you've you'll you'd your yours yourself yourselves he him
his himself she she's her hers herself it it's its itself they them their theirs themselves what which who
whom this that that'll these those am is are was were be been being have has had having do does did doing a
an the and but if or because as until while of at by for with about against between into through during
before after above below to from up down in out on off over under again further then once here there when
where why how all any both each few more most other some such no nor not only own same so than too very s t
can will just don don't should should've now d ll m o re ve y ain aren aren't couldn couldn't didn didn't
doesn doesn't hadn hadn't hasn hasn't haven haven't isn isn't ma mightn mightn't mustn mustn't needn needn't
shan shan't shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't
""".split())
# Words that flip or scale sentiment. Removing them turns "not good" into "good", so they stay.
KEEP_WORDS = {"not", "no", "nor", "but", "against", "very", "too", "few", "more", "most", "only", "off",
              "over", "under", "again", "down", "up", "never"}
STOPWORDS = NLTK_STOPWORDS - KEEP_WORDS
NEGATION_RE = re.compile(r"\b(?:not|no|never|nor|nothing|nobody|none|neither|cannot)\b|n't\b", re.I)
CONTRAST_RE = re.compile(r"\b(?:but|however|although|though|yet|except)\b", re.I)

_UESC = re.compile(r"\\u([0-9a-fA-F]{4})")
# Non-English reviews in Yelp polarity are mostly French (Montreal) with some Spanish
_FR = {"le", "la", "les", "et", "est", "je", "pas", "une", "des", "du", "très", "mais", "pour", "avec", "qui",
       "que", "nous", "vous", "ce", "était", "au", "sur", "dans", "il", "elle", "on", "y", "à"}
_EN = {"the", "and", "is", "was", "i", "to", "it", "of", "for", "this", "we", "you", "they", "my", "with"}


def is_non_english(text):
    w = re.findall(r"[a-zàâçéèêëîïôûùüÿœ']+", text.lower())
    fr = sum(x in _FR for x in w)
    return fr >= 5 and fr > 2 * sum(x in _EN for x in w)


_URL = re.compile(r"(?:https?://|www\.)\S+")
_TAG = re.compile(r"<[^>]+>")
_NT = re.compile(r"n't\b")
_NON_ALPHA = re.compile(r"[^a-z\s]")
_SPACE = re.compile(r"\s+")
_STEMMER = None
_STEM_CACHE = {}


def repair(text):
    """Fix the malformed bits in the raw Yelp dump: literal '\\n', escaped quotes, HTML entities."""
    text = text.replace("\\n", " ").replace('\\""', '"').replace('\\"', '"')
    text = _UESC.sub(lambda m: chr(int(m.group(1), 16)), text)  # literal "\u00e9" -> "é"
    return html.unescape(text)


def clean_tokens(text):
    """lowercase -> strip URLs/HTML -> n't -> ' not' -> drop punctuation/digits -> stopwords -> Snowball stem."""
    global _STEMMER
    if _STEMMER is None:
        from nltk.stem.snowball import SnowballStemmer
        _STEMMER = SnowballStemmer("english")
    t = repair(text).lower()
    t = _URL.sub(" ", t)
    t = _TAG.sub(" ", t)
    t = t.replace("’", "'")
    t = t.replace("won't", "will not").replace("can't", "can not").replace("cannot", "can not")
    t = _NT.sub(" not", t)
    t = t.replace("'", "")
    t = _NON_ALPHA.sub(" ", t)
    out = []
    for w in _SPACE.split(t):
        if len(w) < 2 and w not in ("no",):
            continue
        if w in STOPWORDS:
            continue
        s = _STEM_CACHE.get(w)
        if s is None:
            s = _STEMMER.stem(w)
            _STEM_CACHE[w] = s
        out.append(s)
    return out


def _clean_chunk(texts):
    return [clean_tokens(t) for t in texts]


def clean_all(texts, n_workers=4, chunk=5000):
    chunks = [texts[i:i + chunk] for i in range(0, len(texts), chunk)]
    if n_workers <= 1:
        return [toks for c in chunks for toks in _clean_chunk(c)]
    with Pool(n_workers) as pool:
        return [toks for part in pool.imap(_clean_chunk, chunks) for toks in part]


def malformed_report(texts):
    s = np.array(texts, dtype=object)
    f = lambda pat: int(sum(pat in t for t in s))
    return {"null_or_nonstring": int(sum(not isinstance(t, str) for t in s)),
            "blank": int(sum(isinstance(t, str) and not t.strip() for t in s)),
            "literal_backslash_n": f("\\n"), "escaped_quotes": f('\\"'),
            "html_entities": int(sum(bool(re.search(r"&[a-z]+;|&#\d+;", t)) for t in s)),
            "literal_unicode_escapes": int(sum(bool(_UESC.search(t)) for t in s)),
            "urls": int(sum(bool(_URL.search(t)) for t in s))}


PAD, UNK = 0, 1


def build_vocab(token_lists, min_freq, max_vocab):
    counts = Counter(w for toks in token_lists for w in toks)
    words = [w for w, c in counts.most_common(max_vocab - 2) if c >= min_freq]
    stoi = {"<pad>": PAD, "<unk>": UNK}
    for w in words:
        stoi[w] = len(stoi)
    return stoi, counts


def numericalize(token_lists, stoi, max_len, head):
    """Head + tail truncation: keep the first `head` and last `max_len - head` tokens.
    Reviews often open with context and close with the verdict, so both ends are kept."""
    tail = max_len - head
    ids, lens = np.zeros((len(token_lists), max_len), dtype=np.int32), np.zeros(len(token_lists), dtype=np.int32)
    for i, toks in enumerate(token_lists):
        seq = [stoi.get(w, UNK) for w in toks] or [UNK]
        if len(seq) > max_len:
            seq = seq[:head] + seq[-tail:]
        ids[i, :len(seq)] = seq
        lens[i] = len(seq)
    return ids, lens


# --------------------------------------------------------------------------
# batching: length-bucketed, dynamically padded
# --------------------------------------------------------------------------
def bucket_batches(lens, batch_size, rng, shuffle=True, pool_batches=50):
    idx = rng.permutation(len(lens)) if shuffle else np.arange(len(lens))
    batches = []
    pool = batch_size * pool_batches
    for s in range(0, len(idx), pool):
        chunk = idx[s:s + pool]
        chunk = chunk[np.argsort(lens[chunk], kind="stable")]
        batches += [chunk[i:i + batch_size] for i in range(0, len(chunk), batch_size)]
    if shuffle:
        order = rng.permutation(len(batches))
        batches = [batches[i] for i in order]
    return batches


def make_batch(ids, lens, labels, ix, device, round_to=32):
    # pad each batch up to a multiple of 32 tokens: a handful of distinct shapes keeps the
    # MPS graph/buffer cache bounded (fully dynamic lengths grew it past 16 GB)
    L = min(ids.shape[1], -(-int(lens[ix].max()) // round_to) * round_to)
    x = torch.from_numpy(ids[ix, :L].astype(np.int64)).to(device)
    ln = torch.from_numpy(lens[ix].astype(np.int64))
    y = torch.from_numpy(labels[ix].astype(np.float32)).to(device)
    return x, ln, y


# --------------------------------------------------------------------------
# 2.2 models
# --------------------------------------------------------------------------
class FastTextBigram(nn.Module):
    """Baseline. Mean of learned unigram and hashed-bigram embeddings -> linear.
    Order-blind except for adjacent pairs, which is enough to see 'not good' vs 'good'."""

    def __init__(self, vocab_size, buckets, emb_dim, dropout):
        super().__init__()
        self.vocab_size, self.buckets = vocab_size, buckets
        self.emb = nn.Embedding(vocab_size + buckets, emb_dim, padding_idx=PAD)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(emb_dim, 1)
        nn.init.uniform_(self.emb.weight, -0.05, 0.05)
        with torch.no_grad():
            self.emb.weight[PAD].zero_()

    def forward(self, x, lens):
        mask = (x != PAD)
        a, b = x[:, :-1], x[:, 1:]
        big = self.vocab_size + (a * 1_000_003 + b) % self.buckets
        bmask = mask[:, :-1] & mask[:, 1:]
        big = torch.where(bmask, big, torch.zeros_like(big))
        tok = torch.cat([x, big], dim=1)
        m = torch.cat([mask, bmask], dim=1).unsqueeze(-1).float()
        h = (self.emb(tok) * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.out(self.drop(h)).squeeze(-1)


class BiGRUAttention(nn.Module):
    """Experimental A. Packed BiGRU reads the review both ways; additive attention
    (Bahdanau-style) picks the hidden states that carry the verdict instead of the last state."""

    def __init__(self, vocab_size, emb_dim, hidden, attn_dim, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.emb_drop = nn.Dropout(dropout)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True, bidirectional=True)
        self.att_w = nn.Linear(2 * hidden, attn_dim)
        self.att_v = nn.Linear(attn_dim, 1, bias=False)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(2 * hidden, 1)

    def forward(self, x, lens, return_attn=False):
        from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
        e = self.emb_drop(self.emb(x))
        packed = pack_padded_sequence(e, lens, batch_first=True, enforce_sorted=False)
        h, _ = self.gru(packed)
        h, _ = pad_packed_sequence(h, batch_first=True, total_length=x.size(1))
        scores = self.att_v(torch.tanh(self.att_w(h))).squeeze(-1)
        scores = scores.masked_fill(x == PAD, float("-inf"))
        alpha = torch.softmax(scores, dim=1)
        ctx = (alpha.unsqueeze(-1) * h).sum(1)
        logit = self.out(self.drop(ctx)).squeeze(-1)
        return (logit, alpha) if return_attn else logit


class EncoderSelfAttention(nn.Module):
    def __init__(self, d, n_head, dropout):
        super().__init__()
        self.h, self.dk = n_head, d // n_head
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = [t.view(B, T, self.h, self.dk).transpose(1, 2) for t in (q, k, v)]
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        s = s.masked_fill(pad_mask[:, None, None, :], float("-inf"))  # no attending to padding
        a = self.drop(torch.softmax(s, dim=-1))
        return self.proj((a @ v).transpose(1, 2).reshape(B, T, C))


class EncoderBlock(nn.Module):
    def __init__(self, d, n_head, ffn, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = EncoderSelfAttention(d, n_head, dropout)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        x = x + self.drop(self.attn(self.ln1(x), pad_mask))
        return x + self.drop(self.ffn(self.ln2(x)))


class TransformerClassifier(nn.Module):
    """Experimental B. Bidirectional (unmasked) self-attention encoder with a learned [CLS] vector.
    Every token can attend to every other token, so a negation can reach a far-away adjective in one hop."""

    def __init__(self, vocab_size, max_len, emb_dim, n_layer, n_head, ffn_dim, dropout):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        self.pos = nn.Embedding(max_len + 1, emb_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([EncoderBlock(emb_dim, n_head, ffn_dim, dropout) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(emb_dim)
        self.out = nn.Linear(emb_dim, 1)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls, 0.0, 0.02)

    def forward(self, x, lens):
        B, T = x.shape
        h = torch.cat([self.cls.expand(B, 1, -1), self.emb(x)], dim=1)
        h = self.drop(h + self.pos(torch.arange(T + 1, device=x.device)))
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), x == PAD], dim=1)
        for blk in self.blocks:
            h = blk(h, pad)
        return self.out(self.ln(h[:, 0])).squeeze(-1)


def build_model(name, mcfg, vocab_size, cfg):
    if name.startswith("baseline"):
        return FastTextBigram(vocab_size, cfg["bigram_buckets"], mcfg["emb_dim"], mcfg["dropout"])
    if "gru" in name:
        return BiGRUAttention(vocab_size, mcfg["emb_dim"], mcfg["hidden"], mcfg["attn_dim"], mcfg["dropout"])
    return TransformerClassifier(vocab_size, cfg["max_len"], mcfg["emb_dim"], mcfg["n_layer"], mcfg["n_head"],
                                 mcfg["ffn_dim"], mcfg["dropout"])


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------
# training / inference
# --------------------------------------------------------------------------
@torch.no_grad()
def predict_proba(model, ids, lens, device, batch_size=512):
    model.eval()
    labels = np.zeros(len(lens), dtype=np.float32)
    out = np.zeros(len(lens), dtype=np.float64)
    for ix in bucket_batches(lens, batch_size, None, shuffle=False):
        x, ln, _ = make_batch(ids, lens, labels, ix, device)
        out[ix] = torch.sigmoid(model(x, ln).float()).cpu().numpy()
    return out


def train_model(name, model, data, cfg, device, log_path, ckpt_path):
    mcfg = cfg["models"][name]
    tr_ids, tr_lens, tr_y, va_ids, va_lens, va_y = data
    bs, epochs = cfg["batch_size"], mcfg["epochs"]
    rng = np.random.default_rng(cfg["seed"])
    steps_per_epoch = math.ceil(len(tr_lens) / bs)
    total = epochs * steps_per_epoch
    warm = max(1, int(mcfg["warmup_frac"] * total))
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and "emb" not in n]
    other = [p for n, p in model.named_parameters() if not (p.dim() >= 2 and "emb" not in n)]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": mcfg["weight_decay"]},
                             {"params": other, "weight_decay": 0.0}], lr=mcfg["lr"])
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.05, (total - s) / max(1, total - warm)))
    hist = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "val_macro_f1": [], "epoch_sec": []}
    best, step, nan_count, peak, peak_drv = float("inf"), 0, 0, 0.0, 0.0
    t_all, train_sec, seen = time.time(), 0.0, 0
    from sklearn.metrics import f1_score
    with open(log_path, "w") as log:
        def emit(m):
            print(m)
            log.write(m + "\n")
            log.flush()
        emit(f"run_start={time.strftime('%Y-%m-%d %H:%M:%S')} model={name} smoke={cfg['smoke']} member={cfg['member']}")
        emit(f"device={device} hardware={hardware_string(device)} torch={torch.__version__} python={platform.python_version()}")
        emit(f"config={json.dumps(mcfg)} params={count_params(model)} n_train={len(tr_lens)} n_val={len(va_lens)} "
             f"batch_size={bs} steps={total} warmup={warm}")
        for ep in range(1, epochs + 1):
            model.train()
            t0, run, n = time.time(), 0.0, 0
            for ix in bucket_batches(tr_lens, bs, rng):
                x, ln, y = make_batch(tr_ids, tr_lens, tr_y, ix, device)
                loss = F.binary_cross_entropy_with_logits(model(x, ln), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                lv = loss.item()
                if not (math.isfinite(lv) and math.isfinite(gn)):
                    nan_count += 1
                    opt.zero_grad(set_to_none=True)
                else:
                    opt.step()
                sched.step()
                run += lv * len(ix)
                n += len(ix)
                seen += len(ix)
                if step % 25 == 0:
                    peak = max(peak, device_memory_mb(device))
                    peak_drv = max(peak_drv, device_memory_mb(device, driver=True))
                if step % 400 == 0:
                    emit(f"epoch={ep} step={step}/{total} loss={lv:.4f} grad_norm={gn:.3f} lr={sched.get_last_lr()[0]:.6f} "
                         f"elapsed={time.time() - t_all:.0f}s")
                step += 1
            sync(device)
            ep_sec = time.time() - t0
            train_sec += ep_sec
            peak = max(peak, device_memory_mb(device))
            if device.type == "mps":
                torch.mps.empty_cache()
            p = predict_proba(model, va_ids, va_lens, device)
            pc = np.clip(p, 1e-7, 1 - 1e-7)
            vl = float(-np.mean(va_y * np.log(pc) + (1 - va_y) * np.log(1 - pc)))
            va_acc = float(np.mean((p >= 0.5) == va_y))
            va_f1 = float(f1_score(va_y, p >= 0.5, average="macro"))
            for k, v in zip(hist, [ep, run / n, vl, va_acc, va_f1, ep_sec]):
                hist[k].append(v)
            is_best = vl < best
            if is_best:
                best = vl
                torch.save({"model": model.state_dict(), "name": name, "config": mcfg, "epoch": ep, "val_loss": vl}, ckpt_path)
            emit(f"EPOCH {ep} train_loss={run / n:.4f} val_loss={vl:.4f} val_acc={va_acc:.4f} val_macro_f1={va_f1:.4f} "
                 f"epoch_sec={ep_sec:.1f} ex_per_sec={n / ep_sec:.0f} nan={nan_count} best={'yes' if is_best else 'no'}")
        summary = {"train_time_sec": train_sec, "wall_time_sec": time.time() - t_all,
                   "train_examples_per_sec": seen / train_sec, "nan_count": nan_count,
                   "peak_device_memory_mb": peak, "peak_mps_driver_memory_mb": peak_drv, "peak_process_rss_mb": process_peak_rss_mb(),
                   "best_val_loss": best, "params": count_params(model), "hardware": hardware_string(device)}
        emit("SUMMARY " + json.dumps(summary))
        emit(f"run_end={time.strftime('%Y-%m-%d %H:%M:%S')}")
    return hist, summary


# --------------------------------------------------------------------------
# 2.2 evaluation metrics
# --------------------------------------------------------------------------
def ece_score(y, p, n_bins=15):
    """Expected calibration error on the confidence of the predicted class (equal-width bins)."""
    pred = (p >= 0.5).astype(int)
    conf = np.where(pred == 1, p, 1 - p)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        gap = abs(correct[m].mean() - conf[m].mean())
        ece += m.mean() * gap
        rows.append((float(conf[m].mean()), float(correct[m].mean()), int(m.sum())))
    return float(ece), rows


def _binary_stats(tp, fp, fn, tn):
    """Vectorised accuracy, macro-F1 and MCC from confusion counts (arrays allowed)."""
    tp, fp, fn, tn = (np.asarray(v, dtype=np.float64) for v in (tp, fp, fn, tn))
    n = tp + fp + fn + tn
    acc = (tp + tn) / n
    f1_pos = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1e-12), 0.0)
    f1_neg = np.where(2 * tn + fn + fp > 0, 2 * tn / np.maximum(2 * tn + fn + fp, 1e-12), 0.0)
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.where(denom > 0, (tp * tn - fp * fn) / np.maximum(denom, 1e-12), 0.0)
    return acc, (f1_pos + f1_neg) / 2, mcc


def bootstrap_ci(y, pred, n_boot=1000, seed=0):
    """Percentile bootstrap over test examples, resampled with replacement."""
    rng = np.random.default_rng(seed)
    n = len(y)
    counts = []
    for s in range(0, n_boot, 100):  # chunks keep memory near 100 x n
        idx = rng.integers(0, n, size=(min(100, n_boot - s), n))
        yy, pp = y[idx], pred[idx]
        counts.append(np.stack([((yy == 1) & (pp == 1)).sum(1), ((yy == 0) & (pp == 1)).sum(1),
                                ((yy == 1) & (pp == 0)).sum(1), ((yy == 0) & (pp == 0)).sum(1)]))
    tp, fp, fn, tn = np.concatenate(counts, axis=1)
    acc, f1, mcc = _binary_stats(tp, fp, fn, tn)
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"accuracy": q(acc), "macro_f1": q(f1), "mcc": q(mcc)}


def mcnemar(y, pred_a, pred_b):
    """b = A right & B wrong, c = A wrong & B right. Exact binomial p and chi-square with continuity."""
    from scipy.stats import binomtest, chi2
    ra, rb = pred_a == y, pred_b == y
    b, c = int((ra & ~rb).sum()), int((~ra & rb).sum())
    stat = (abs(b - c) - 1) ** 2 / (b + c) if b + c > 0 else 0.0
    return {"b_baseline_right_model_wrong": b, "c_baseline_wrong_model_right": c,
            "chi2_cc": float(stat), "p_chi2": float(chi2.sf(stat, 1)),
            "p_exact": float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c > 0 else 1.0}


def full_metrics(y, p, n_boot, seed, n_bins):
    from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                                 confusion_matrix, matthews_corrcoef, precision_recall_fscore_support,
                                 roc_auc_score)
    pred = (p >= 0.5).astype(int)
    m = {"accuracy": accuracy_score(y, pred)}
    for avg in ("macro", "micro", "weighted"):
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average=avg, zero_division=0)
        m[f"precision_{avg}"], m[f"recall_{avg}"], m[f"f1_{avg}"] = pr, rc, f1
    m["roc_auc"] = roc_auc_score(y, p)
    m["pr_auc"] = average_precision_score(y, p)
    m["mcc"] = matthews_corrcoef(y, pred)
    m["brier"] = brier_score_loss(y, p)
    m["ece"], _ = ece_score(y, p, n_bins)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    m["tn"], m["fp"], m["fn"], m["tp"] = (int(v) for v in cm.ravel())
    ci = bootstrap_ci(y, pred, n_boot, seed)
    for k, (lo, hi) in ci.items():
        m[f"{k}_ci95_low"], m[f"{k}_ci95_high"] = lo, hi
    return {k: float(v) if not isinstance(v, int) else v for k, v in m.items()}, cm


def slice_masks(raw_texts, raw_word_counts, quartiles):
    """raw_texts should already be repair()-ed so the language check sees real accents."""
    q1, q2, q3 = quartiles
    w = np.asarray(raw_word_counts)
    neg = np.array([bool(NEGATION_RE.search(t)) for t in raw_texts])
    con = np.array([bool(CONTRAST_RE.search(t)) for t in raw_texts])
    foreign = np.array([is_non_english(t) for t in raw_texts])
    return {
        f"len_Q1_<={q1:.0f}w": w <= q1,
        f"len_Q2_{q1:.0f}-{q2:.0f}w": (w > q1) & (w <= q2),
        f"len_Q3_{q2:.0f}-{q3:.0f}w": (w > q2) & (w <= q3),
        f"len_Q4_>{q3:.0f}w": w > q3,
        "has_negation": neg, "no_negation": ~neg,
        "has_contrast(but/however/...)": con, "no_contrast": ~con,
        "negation_and_contrast": neg & con,
        "non_english": foreign,
    }


def slice_metrics(y, p, masks):
    from sklearn.metrics import f1_score
    pred = (p >= 0.5).astype(int)
    rows = {}
    for name, m in masks.items():
        rows[name] = {"n": int(m.sum()), "macro_f1": float(f1_score(y[m], pred[m], average="macro")),
                      "error_rate": float((pred[m] != y[m]).mean())}
    return rows
