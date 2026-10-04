"""
Task 2 preprocessing: EDA, cleaning, tokenisation, from-scratch vocabulary.
No pretrained embeddings or pretrained tokenizers are used.

Outputs (in data_dir):
  X_{train,val,test}.npy  int32 padded token ids
  len_{train,val,test}.npy true (unpadded) lengths
  y_{train,val,test}.npy  labels (0 = negative, 1 = positive)
  test_raw.json           raw test reviews (for manual error review)
  test_slices.json        boolean slice masks for robustness metrics
  vocab.json, preprocess_stats.json, eda/*.png

Run:
    python src/preprocess.py --config config.yaml
"""
import argparse
import json
import os
import re
from collections import Counter
from functools import lru_cache

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from datasets import load_dataset, load_from_disk

import nltk
from nltk.stem import PorterStemmer

nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords  # noqa: E402

# Negations and intensifiers flip or scale sentiment, so they are kept even though NLTK lists them as stopwords.
KEEP = {
    "not", "no", "nor", "never", "don", "don't", "didn", "didn't", "doesn", "doesn't", "isn", "isn't",
    "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't", "shouldn", "shouldn't",
    "couldn", "couldn't", "aren", "aren't", "hasn", "hasn't", "haven", "haven't", "hadn", "hadn't",
    "mightn", "mustn", "needn", "ain", "too", "very", "most", "more", "against", "but", "only",
}
STOPWORDS = set(stopwords.words("english")) - KEEP
STEMMER = PorterStemmer()
# Yelp has ~74M tokens but only a few hundred thousand distinct words, so caching stems cuts preprocessing from ~20 min to ~2 min.
stem = lru_cache(maxsize=None)(STEMMER.stem)
NEGATION_RE = re.compile(r"\b(not|no|never|n't|nothing|nobody|none|neither|nor)\b", re.I)


def clean_text(text: str, cfg: dict) -> list:
    text = text.replace("\\n", " ").replace("\\\"", "\"")
    if cfg.get("lowercase", True):
        text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"[^a-z0-9\s]" if cfg.get("lowercase", True) else r"[^A-Za-z0-9\s]", " ", text)
    tokens = text.split()
    if cfg.get("remove_stopwords", True):
        tokens = [t for t in tokens if t not in STOPWORDS]
    if cfg.get("stem", True):
        tokens = [stem(t) for t in tokens]
    return tokens


def build_vocab(token_lists, max_vocab_size: int, min_freq: int):
    counter = Counter(t for toks in token_lists for t in toks)
    vocab = ["<pad>", "<unk>"]
    for tok, freq in counter.most_common():
        if freq < min_freq or len(vocab) >= max_vocab_size:
            break
        vocab.append(tok)
    return {tok: i for i, tok in enumerate(vocab)}, counter


def encode(token_lists, word_to_idx, max_len):
    X = np.zeros((len(token_lists), max_len), dtype=np.int32)
    lengths = np.zeros(len(token_lists), dtype=np.int32)
    for r, toks in enumerate(token_lists):
        ids = [word_to_idx.get(t, 1) for t in toks[:max_len]] or [1]
        X[r, : len(ids)] = ids
        lengths[r] = len(ids)
    return X, lengths


def is_malformed(text) -> bool:
    return not isinstance(text, str) or len(text.strip()) == 0 or not re.search(r"[A-Za-z]", text)


def slice_masks(raw_texts, lengths_words):
    lw = np.asarray(lengths_words)
    q1, q3 = np.percentile(lw, [25, 75])
    return {
        "short_reviews(<=p25_words)": (lw <= q1).tolist(),
        "long_reviews(>=p75_words)": (lw >= q3).tolist(),
        "contains_negation": [bool(NEGATION_RE.search(t)) for t in raw_texts],
        "contains_but_contrast": [bool(re.search(r"\bbut\b|\bhowever\b|\balthough\b", t, re.I)) for t in raw_texts],
        "exclamation_heavy(>=3 '!')": [t.count("!") >= 3 for t in raw_texts],
    }


def eda(train_texts, train_labels, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    lengths = np.array([len(t.split()) for t in train_texts])
    labels = np.asarray(train_labels)

    plt.figure(figsize=(7, 4))
    for lab, name in [(0, "negative"), (1, "positive")]:
        plt.hist(np.clip(lengths[labels == lab], 0, 800), bins=60, alpha=0.6, label=name)
    plt.xlabel("review length (words, clipped at 800)")
    plt.ylabel("count")
    plt.title("Review length distribution by class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "review_length_distribution.png"), dpi=120)
    plt.close()

    counts = Counter(labels.tolist())
    plt.figure(figsize=(4, 4))
    plt.bar(["negative (0)", "positive (1)"], [counts.get(0, 0), counts.get(1, 0)])
    plt.title("Class distribution (train)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "class_distribution.png"), dpi=120)
    plt.close()

    return {
        "n_train": int(len(labels)),
        "class_counts": {str(k): int(v) for k, v in counts.items()},
        "class_balance_ratio_min_over_max": float(min(counts.values()) / max(counts.values())),
        "length_words_mean": float(lengths.mean()),
        "length_words_median": float(np.median(lengths)),
        "length_words_p95": float(np.percentile(lengths, 95)),
        "length_words_max": int(lengths.max()),
        "length_by_class_mean": {str(k): float(lengths[labels == k].mean()) for k in counts},
    }


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["data"]
    rng = np.random.default_rng(cfg["seed"])
    data_dir = cfg.get("data_dir", "data")
    os.makedirs(data_dir, exist_ok=True)

    # Prefer the team's shared raw copy (task2_sentiment/data/, made by tools/download_raw_data.py), else the HF Hub
    local = cfg.get("local_dataset_dir")
    if local and os.path.isdir(local):
        ds = load_from_disk(local)
        print(f"loaded raw dataset from {local}")
    else:
        ds = load_dataset(cfg["dataset_name"])
    train_raw, test_raw = ds["train"], ds["test"]
    if cfg.get("train_subset"):
        train_raw = train_raw.select(rng.permutation(len(train_raw))[: cfg["train_subset"]].tolist())
    if cfg.get("test_subset"):
        test_raw = test_raw.select(rng.permutation(len(test_raw))[: cfg["test_subset"]].tolist())

    tr_texts, tr_labels = list(train_raw["text"]), list(train_raw["label"])
    te_texts, te_labels = list(test_raw["text"]), list(test_raw["label"])

    stats = {"eda": eda(tr_texts, tr_labels, os.path.join(data_dir, "eda"))}

    def filter_bad(texts, labels, split):
        keep = [i for i, (t, l) in enumerate(zip(texts, labels)) if not is_malformed(t) and l in (0, 1)]
        stats[f"removed_malformed_{split}"] = len(texts) - len(keep)
        dup = len(keep) - len({texts[i] for i in keep})
        stats[f"exact_duplicates_{split}"] = dup
        return [texts[i] for i in keep], [labels[i] for i in keep]

    tr_texts, tr_labels = filter_bad(tr_texts, tr_labels, "train")
    te_texts, te_labels = filter_bad(te_texts, te_labels, "test")

    perm = rng.permutation(len(tr_texts))
    n_val = int(len(tr_texts) * cfg.get("val_fraction", 0.1))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    print("cleaning/tokenising ...", flush=True)
    tr_tok_all = [clean_text(t, cfg) for t in tr_texts]
    te_tok = [clean_text(t, cfg) for t in te_texts]
    tr_tok = [tr_tok_all[i] for i in tr_idx]
    va_tok = [tr_tok_all[i] for i in val_idx]

    word_to_idx, counter = build_vocab(tr_tok, cfg["max_vocab_size"], cfg["min_freq"])
    max_len = cfg["max_seq_len"]
    tok_lens = np.array([len(t) for t in tr_tok])
    stats.update(
        {
            "vocab_size": len(word_to_idx),
            "raw_unique_tokens": len(counter),
            "token_len_mean_after_cleaning": float(tok_lens.mean()),
            "pct_truncated_at_max_seq_len": float((tok_lens > max_len).mean() * 100),
            "unk_rate_test": float(np.mean([t not in word_to_idx for toks in te_tok for t in toks])),
            "splits": {"train": len(tr_idx), "val": len(val_idx), "test": len(te_tok)},
            "example_cleaning": {"raw": te_texts[0][:300], "tokens": te_tok[0][:40]},
        }
    )

    for name, toks, labels in [
        ("train", tr_tok, [tr_labels[i] for i in tr_idx]),
        ("val", va_tok, [tr_labels[i] for i in val_idx]),
        ("test", te_tok, te_labels),
    ]:
        X, L = encode(toks, word_to_idx, max_len)
        np.save(os.path.join(data_dir, f"X_{name}.npy"), X)
        np.save(os.path.join(data_dir, f"len_{name}.npy"), L)
        np.save(os.path.join(data_dir, f"y_{name}.npy"), np.asarray(labels, dtype=np.int64))

    with open(os.path.join(data_dir, "vocab.json"), "w") as f:
        json.dump(word_to_idx, f)
    with open(os.path.join(data_dir, "test_raw.json"), "w") as f:
        json.dump(te_texts, f)
    with open(os.path.join(data_dir, "test_slices.json"), "w") as f:
        json.dump(slice_masks(te_texts, [len(t.split()) for t in te_texts]), f)
    with open(os.path.join(data_dir, "preprocess_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "example_cleaning"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    main(ap.parse_args().config)
