"""
Selects 20 of a model's test errors for manual review:
  5 confident false positives, 5 confident false negatives,
  5 near-threshold errors, 5 errors from the model's worst data slice.

Writes error_review_<model>.csv and .md with the review text, the model's
probability, a heuristic *suggested* error type (to be confirmed/overridden
by hand) and empty columns for your final error type and proposed fix.

Run (after train.py):
    python src/error_analysis.py --config config.yaml --model_name experimental_2
"""
import argparse
import csv
import json
import os
import re

import numpy as np
import yaml

NEG = re.compile(r"\b(not|no|never|n't|nothing|nobody|none|neither|nor)\b", re.I)
CONTRAST = re.compile(r"\b(but|however|although|though|except|otherwise)\b", re.I)
SARCASM = re.compile(r"(\bsure\b|\byeah right\b|\bthanks a lot\b|\bgreat job\b|\bwow\b|\"[a-z]+\")", re.I)


def suggest_error_type(text: str, n_words: int, max_len_words: int) -> str:
    if n_words > max_len_words:
        return "truncation (key sentiment after max_seq_len cut-off)"
    if CONTRAST.search(text):
        return "mixed sentiment / contrast clause"
    if NEG.search(text):
        return "negation scope"
    if SARCASM.search(text):
        return "sarcasm / irony"
    if n_words < 15:
        return "too little signal (very short review)"
    return "implicit sentiment / domain-specific wording"


def main(cfg_path: str, model_name: str, seed: int = 1337):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    data_dir, out_dir = cfg["data"].get("data_dir", "data"), cfg["train"]["out_dir"]
    max_len = cfg["data"]["max_seq_len"]

    d = np.load(os.path.join(out_dir, f"predictions_{model_name}.npz"))
    y, p = d["y_true"], d["y_prob"]
    pred = (p >= 0.5).astype(int)
    with open(os.path.join(data_dir, "test_raw.json")) as f:
        texts = json.load(f)
    with open(os.path.join(data_dir, "test_slices.json")) as f:
        slices = {k: np.asarray(v, dtype=bool) for k, v in json.load(f).items()}
    token_lens = np.load(os.path.join(data_dir, "len_test.npy"))

    wrong = pred != y
    rng = np.random.default_rng(seed)
    used = set()

    def take(candidates, k, order=None):
        c = [i for i in (candidates if order is None else candidates[order]) if i not in used]
        chosen = c[:k] if order is not None else list(rng.permutation(c)[:k]) if c else []
        used.update(int(i) for i in chosen)
        return [int(i) for i in chosen]

    fp = np.where(wrong & (y == 0))[0]
    fn = np.where(wrong & (y == 1))[0]
    near = np.where(wrong)[0]
    worst_slice = max(slices, key=lambda s: (wrong[slices[s]].mean() if slices[s].any() else -1))
    slice_err = np.where(wrong & slices[worst_slice])[0]

    buckets = [
        ("confident_false_positive", take(fp, 5, np.argsort(-p[fp]))),
        ("confident_false_negative", take(fn, 5, np.argsort(p[fn]))),
        ("near_threshold", take(near, 5, np.argsort(np.abs(p[near] - 0.5)))),
        (f"slice:{worst_slice}", take(slice_err, 5)),
    ]

    rows = []
    for bucket, idxs in buckets:
        for i in idxs:
            rows.append(
                {
                    "bucket": bucket,
                    "test_index": i,
                    "true_label": "positive" if y[i] == 1 else "negative",
                    "pred_label": "positive" if pred[i] == 1 else "negative",
                    "p_positive": round(float(p[i]), 4),
                    "n_tokens_after_cleaning": int(token_lens[i]),
                    "suggested_error_type": suggest_error_type(texts[i], len(texts[i].split()), max_len),
                    "final_error_type": "",
                    "proposed_testable_fix": "",
                    "review_text": texts[i].replace("\\n", " ")[:1200],
                }
            )

    csv_path = os.path.join(out_dir, f"error_review_{model_name}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md_path = os.path.join(out_dir, f"error_review_{model_name}.md")
    with open(md_path, "w") as f:
        f.write(f"# Error review — {model_name}\n\nWorst slice: `{worst_slice}` (error rate {wrong[slices[worst_slice]].mean():.3f})\n\n")
        for r in rows:
            f.write(
                f"### {r['bucket']} — test #{r['test_index']}\n"
                f"- true: **{r['true_label']}**, predicted: **{r['pred_label']}** (P(pos)={r['p_positive']})\n"
                f"- suggested error type: {r['suggested_error_type']}\n"
                f"- final error type: _\n- proposed testable fix: _\n\n> {r['review_text'][:600]}\n\n"
            )

    summary = {}
    for r in rows:
        summary[r["suggested_error_type"]] = summary.get(r["suggested_error_type"], 0) + 1
    print(f"wrote {len(rows)} errors -> {csv_path} and {md_path}")
    print("suggested error-type counts:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model_name", required=True)
    a = ap.parse_args()
    main(a.config, a.model_name)
