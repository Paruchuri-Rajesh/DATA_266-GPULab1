"""
Task 1 data preprocessing: char-level tokenization of TinyStories.

Produces:
  - char_to_idx.json / idx_to_char.json
  - train.bin / val.bin  (uint16 numpy arrays of integer-encoded characters)

Run:
    python src/preprocess.py --config config.yaml
"""
import argparse
import json
import os

import numpy as np
import yaml
from datasets import load_dataset, load_from_disk


def build_vocab(text: str):
    chars = sorted(set(text))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return char_to_idx, idx_to_char


def encode(text: str, char_to_idx: dict) -> np.ndarray:
    return np.array([char_to_idx[c] for c in text if c in char_to_idx], dtype=np.uint16)


def main(cfg_path: str):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)["data"]

    rng = np.random.default_rng(cfg["seed"])
    # Prefer the team's shared raw copy (task1_llm/data/, made by tools/download_raw_data.py), else the HF Hub
    local = cfg.get("local_dataset_dir")
    if local and os.path.isdir(local):
        ds = load_from_disk(local)["train"]
        print(f"loaded raw dataset from {local}")
    else:
        ds = load_dataset(cfg["dataset_name"], split="train")

    # Each member draws their own shuffled split (own seed) per the assignment spec.
    n_total = cfg["train_size"] + cfg["val_size"]
    indices = rng.permutation(len(ds))[:n_total]
    texts = [t.strip() for t in ds.select(indices.tolist())["text"]]
    n_empty = sum(1 for t in texts if not t)
    texts = [t for t in texts if t]
    print(f"dropped {n_empty} empty stories")

    full_text = "\n".join(texts)
    char_to_idx, idx_to_char = build_vocab(full_text)

    n_val = min(cfg["val_size"], len(texts) // 10)
    train_texts = texts[: len(texts) - n_val]
    val_texts = texts[len(texts) - n_val :]

    train_ids = encode("\n".join(train_texts), char_to_idx)
    val_ids = encode("\n".join(val_texts), char_to_idx)

    data_dir = cfg.get("data_dir", "data")
    os.makedirs(data_dir, exist_ok=True)
    train_ids.tofile(os.path.join(data_dir, "train.bin"))
    val_ids.tofile(os.path.join(data_dir, "val.bin"))
    with open(os.path.join(data_dir, "char_to_idx.json"), "w") as f:
        json.dump(char_to_idx, f)
    with open(os.path.join(data_dir, "idx_to_char.json"), "w") as f:
        json.dump(idx_to_char, f)

    print(f"vocab_size={len(char_to_idx)}  train_chars={len(train_ids)}  val_chars={len(val_ids)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    main(args.config)
