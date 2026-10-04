"""
Task 1 generation + generation metrics:
  - greedy and temperature sampling from several prompts
  - distinct-1/2/3 and repeated 4-gram rate (word level, over all samples)
  - generation tokens/sec
  - failure-case candidates (repetition, invented words) for the failure analysis

Run:
    python src/generate.py --config config.yaml --checkpoint checkpoints/gpt_final.pt
"""
import argparse
import json
import os
import re
import time
from collections import Counter

import numpy as np
import torch
import yaml

import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from common.utils import resolve_device, set_seed  # noqa: E402
from model import GPT  # noqa: E402

PROMPTS = [
    "Once upon a time",
    "One day, a little girl named Lily",
    "Tom and his dog went to the park",
    "The sun was shining and",
    "Mom said, \"",
]


def words(text: str):
    return re.findall(r"[a-zA-Z']+|[.,!?\"]", text.lower())


def distinct_n(token_lists, n: int) -> float:
    ngrams = [tuple(t[i : i + n]) for t in token_lists for i in range(len(t) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 0.0


def repeated_ngram_rate(tokens, n: int = 4) -> float:
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    if not ngrams:
        return 0.0
    counts = Counter(ngrams)
    return sum(c - 1 for c in counts.values()) / len(ngrams)


def main(cfg_path: str, checkpoint: str, samples_per_prompt: int):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    dcfg, mcfg, gcfg = cfg["data"], cfg["model"], cfg["generate"]
    set_seed(dcfg["seed"])
    device = resolve_device(cfg["train"]["device"])
    out_dir = os.path.dirname(checkpoint) or "."

    data_dir = dcfg.get("data_dir", "data")
    with open(os.path.join(data_dir, "char_to_idx.json")) as f:
        char_to_idx = json.load(f)
    with open(os.path.join(data_dir, "idx_to_char.json")) as f:
        idx_to_char = {int(k): v for k, v in json.load(f).items()}

    model = GPT(len(char_to_idx), dcfg["block_size"], mcfg["n_embd"], mcfg["n_head"], mcfg["n_layer"], mcfg["dropout"])
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device).eval()

    train_text = "".join(idx_to_char[int(i)] for i in np.fromfile(os.path.join(data_dir, "train.bin"), dtype=np.uint16)[:2_000_000])
    train_vocab = set(words(train_text))

    def encode(s):
        return torch.tensor([[char_to_idx[c] for c in s if c in char_to_idx]], dtype=torch.long, device=device)

    def decode(ids):
        return "".join(idx_to_char[i] for i in ids)

    samples = []
    gen_tokens, gen_time = 0, 0.0
    for prompt in PROMPTS:
        runs = [("greedy", 0.0)] + [("temperature", gcfg["temperature"])] * samples_per_prompt
        for mode, temp in runs:
            idx = encode(prompt)
            t0 = time.time()
            out = model.generate(idx, gcfg["max_new_tokens"], temperature=temp, top_k=gcfg.get("top_k"), greedy=(mode == "greedy"))
            if device == "cuda":
                torch.cuda.synchronize()
            gen_time += time.time() - t0
            gen_tokens += gcfg["max_new_tokens"]
            text = decode(out[0].tolist())
            continuation = text[len(prompt) :]
            w = words(continuation)
            oov = [x for x in w if x.isalpha() and x not in train_vocab]
            samples.append(
                {
                    "prompt": prompt,
                    "mode": mode,
                    "temperature": temp,
                    "text": text,
                    "repeated_4gram_rate": repeated_ngram_rate(w, 4),
                    "invented_word_rate": len(oov) / max(1, len(w)),
                    "invented_words": sorted(set(oov))[:15],
                }
            )

    sampled = [s for s in samples if s["mode"] == "temperature"]
    greedy = [s for s in samples if s["mode"] == "greedy"]
    sampled_tokens = [words(s["text"][len(s["prompt"]) :]) for s in sampled]
    all_sampled = [t for toks in sampled_tokens for t in toks]

    metrics = {
        "checkpoint": checkpoint,
        "decoding": {"temperature": gcfg["temperature"], "top_k": gcfg.get("top_k"), "max_new_tokens": gcfg["max_new_tokens"]},
        "distinct_1": distinct_n(sampled_tokens, 1),
        "distinct_2": distinct_n(sampled_tokens, 2),
        "distinct_3": distinct_n(sampled_tokens, 3),
        "repeated_4gram_rate": float(np.mean([s["repeated_4gram_rate"] for s in sampled])) if sampled else 0.0,
        "repeated_4gram_rate_pooled": repeated_ngram_rate(all_sampled, 4),
        "repeated_4gram_rate_greedy": float(np.mean([s["repeated_4gram_rate"] for s in greedy])),
        "invented_word_rate": float(np.mean([s["invented_word_rate"] for s in sampled])) if sampled else 0.0,
        "generation_tokens_per_sec": gen_tokens / gen_time,
        "num_samples": len(samples),
    }

    candidates = {
        "most_repetitive": max(samples, key=lambda s: s["repeated_4gram_rate"]),
        "most_invented_words": max(samples, key=lambda s: s["invented_word_rate"]),
        "greedy_example": greedy[0],
    }

    with open(os.path.join(out_dir, "generation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "samples.json"), "w") as f:
        json.dump(samples, f, indent=2)
    with open(os.path.join(out_dir, "failure_candidates.json"), "w") as f:
        json.dump(candidates, f, indent=2)
    with open(os.path.join(out_dir, "samples.txt"), "w") as f:
        for s in samples:
            f.write(f"=== [{s['mode']} T={s['temperature']}] rep4={s['repeated_4gram_rate']:.3f} invented={s['invented_word_rate']:.3f}\n{s['text']}\n\n")

    print(json.dumps(metrics, indent=2))
    print("\n--- sample (temperature) ---\n" + (sampled[0]["text"] if sampled else greedy[0]["text"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--checkpoint", default="checkpoints/gpt_final.pt")
    ap.add_argument("--samples_per_prompt", type=int, default=3)
    a = ap.parse_args()
    main(a.config, a.checkpoint, a.samples_per_prompt)
