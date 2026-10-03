"""Writes part2_sentiment.ipynb (unexecuted). Run: python build_notebook.py"""
import json
from pathlib import Path

cells = []


def md(s):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})


def code(s):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})


md("""
# DATA 266 Lab 1 — Task 2: Yelp Polarity sentiment bake-off
**Member:** Rajesh Paruchuri · **Data:** `fancyzhx/yelp_polarity` (560,000 train / 38,000 test, binary)

No pretrained embeddings or language models: every embedding table starts from random init and is learned with its classifier.

| | Model | Core idea |
|---|---|---|
| Baseline | **fastText-style bag of unigrams + hashed bigrams** | order-blind, but adjacent pairs ("not good") get their own vectors |
| Experimental A | **BiGRU + additive attention pooling** | reads the review in both directions; attention weights the verdict tokens |
| Experimental B | **2-layer Transformer encoder written from scratch, [CLS] pooling** | every token attends to every other token in one hop |

Run (repo root): `jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 task2_sentiment/rajesh_paruchuri/src/part2_sentiment.ipynb`
· smoke test: prefix with `LAB_SMOKE=1`.
""")

code("""
import json, math, sys, time, csv
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path.cwd()) if (Path.cwd() / "sentiment_lib.py").exists() else str(Path.cwd() / "task2_sentiment/rajesh_paruchuri/src"))
import sentiment_lib as sl

P = sl.find_paths()
R = lambda p: sl.rel(p, P["repo"])
CFG = sl.load_config(P)
SMOKE = bool(CFG["smoke"])
TAG = "smoke" if SMOKE else "full"
SEED = CFG["seed"]
sl.set_seed(SEED)
device = sl.pick_device()
HW = sl.hardware_string(device)
print("smoke:", SMOKE, "| seed:", SEED, "| device:", device, "|", HW)
import sklearn, nltk
print("torch", torch.__version__, "| sklearn", sklearn.__version__, "| nltk", nltk.__version__, "| python", sys.version.split()[0])
""")

md("""
## 2.1 Data preprocessing
### 2.1.1 Load, then check for missing and malformed entries
""")

code("""
files = sl.download_if_missing(CFG, P)
train_df = pd.read_parquet(files["train"])
test_df = pd.read_parquet(files["test"])
print("raw train", train_df.shape, "| raw test", test_df.shape)
print("label meaning: 0 = negative (1-2 stars), 1 = positive (3-4 stars)")
print("missing values:\\n", pd.concat([train_df.isna().sum().rename("train"), test_df.isna().sum().rename("test")], axis=1))
print("exact duplicate texts in train:", int(train_df.text.duplicated().sum()),
      "| test texts that also appear in train (leakage):", int(test_df.text.isin(set(train_df.text)).sum()))
print("malformed-pattern counts (train):", sl.malformed_report(train_df.text.tolist()))
print("malformed-pattern counts (test): ", sl.malformed_report(test_df.text.tolist()))
pos = train_df.text.str.find('\\\\n')
ex = train_df.text[(pos > 0) & (pos < 120) & train_df.text.str.contains('\\\\u00', regex=False)].iloc[0]
print("\\nexample raw entry with escapes:", repr(ex[:240]))
print("after repair():                 ", repr(sl.repair(ex)[:240]))
""")

md("""
### 2.1.2 Class balance and review-length distribution
""")

code("""
for df in (train_df, test_df):
    df["n_words"] = df.text.str.split().str.len()
    df["n_chars"] = df.text.str.len()
print("class counts train:", train_df.label.value_counts().sort_index().to_dict(),
      "| test:", test_df.label.value_counts().sort_index().to_dict())
print("balance ratio (minority/majority) train: %.3f" % (train_df.label.value_counts().min() / train_df.label.value_counts().max()))
desc = train_df.groupby("label").n_words.describe(percentiles=[.05, .25, .5, .75, .95]).round(1)
desc.index = ["negative", "positive"]
display(desc)

fig, ax = plt.subplots(1, 3, figsize=(16, 4))
ax[0].bar(["negative", "positive"], train_df.label.value_counts().sort_index().values, color=["#c0504d", "#4f81bd"])
ax[0].set(title="Class distribution (train)", ylabel="reviews")
bins = np.logspace(0, np.log10(train_df.n_words.max()), 60)
for lab, name, col in [(0, "negative", "#c0504d"), (1, "positive", "#4f81bd")]:
    ax[1].hist(train_df.n_words[train_df.label == lab], bins=bins, alpha=0.55, label=name, color=col)
ax[1].set(xscale="log", title="Review length (words, log x)", xlabel="words", ylabel="reviews"); ax[1].legend()
ax[2].boxplot([train_df.n_words[train_df.label == 0], train_df.n_words[train_df.label == 1]], showfliers=False)
ax[2].set_xticks([1, 2], ["negative", "positive"])
ax[2].set(title="Length by class (no outliers)", ylabel="words")
fig.tight_layout(); fig.savefig(P["out"] / f"eda_{TAG}.png", dpi=130); plt.show()
print("negative reviews are longer on average: %.0f vs %.0f words" % tuple(train_df.groupby("label").n_words.mean()))
""")

md("""
### 2.1.3 Split
* **Test:** the official 38,000-row test file, untouched until final evaluation.
* **Validation:** 20,000 reviews held out from the train file, stratified by label (seed 266). Used for checkpoint selection only.
* **Train:** the remaining 540,000 reviews.
""")

code("""
from sklearn.model_selection import train_test_split
tr_df, va_df = train_test_split(train_df, test_size=CFG["val_size"], stratify=train_df.label, random_state=SEED)
te_df = test_df
if SMOKE:
    tr_df = tr_df.sample(CFG["n_train_smoke"], random_state=SEED)
    va_df = va_df.sample(CFG["n_val_smoke"], random_state=SEED)
    te_df = test_df.sample(CFG["n_test_smoke"], random_state=SEED)
tr_df, va_df, te_df = (d.reset_index(drop=True) for d in (tr_df, va_df, te_df))
print("train", len(tr_df), tr_df.label.value_counts().to_dict(), "| val", len(va_df), va_df.label.value_counts().to_dict(),
      "| test", len(te_df), te_df.label.value_counts().to_dict())
""")

md("""
### 2.1.4 Text preprocessing and tokenisation
Steps in `sentiment_lib.clean_tokens`, in order:
1. **Repair** literal `\\n`, escaped `\\"` quotes and HTML entities (the malformed entries found above).
2. **Lowercase**, strip URLs and HTML tags.
3. **Negation-aware contractions**: `won't → will not`, `can't → can not`, `*n't → * not`.
4. **Remove punctuation, digits and special characters** (keep `a–z` only).
5. **Stopword removal** with NLTK's 179-word English list, **minus 18 sentiment-bearing words**
   (`not, no, nor, never, but, very, too, only, against, more, most, few, off, over, under, again, up, down`).
   Dropping "not" would turn "not good" into "good", which is the exact error a sentiment model must avoid.
6. **Stemming** with the Snowball (Porter2) stemmer: `amazing/amazed → amaz`. It merges inflections without needing a downloaded lexicon.
7. **Whitespace tokenisation**, then a vocabulary built from the **train split only** (min frequency 3, cap 60,000).
   Indices: 0 = `<pad>`, 1 = `<unk>`.
8. **Head+tail truncation** to 256 tokens (first 160 + last 96), because reviews often end with the verdict.
""")

code("""
for t in ["I didn't love it... the food wasn't GREAT but service was amazing!!! 5/5 http://yelp.com",
          tr_df.text[3][:300]]:
    print("RAW:   ", t[:300].replace("\\n", " "))
    print("TOKENS:", sl.clean_tokens(t)[:60], "\\n")

cache = P["data_proc"] / f"train_arrays_{TAG}.npz"
vocab_path = P["data_proc"] / ("vocab.json" if not SMOKE else "vocab_smoke.json")
t0 = time.time()
tok_tr = sl.clean_all(tr_df.text.tolist(), CFG["n_workers"])
tok_va = sl.clean_all(va_df.text.tolist(), CFG["n_workers"])
tok_te = sl.clean_all(te_df.text.tolist(), CFG["n_workers"])
print("preprocessed %d reviews in %.0fs" % (len(tok_tr) + len(tok_va) + len(tok_te), time.time() - t0))

# entries that become empty after cleaning carry no signal: drop from train, keep (as <unk>) in val/test
empty_tr = np.array([len(t) == 0 for t in tok_tr])
print("empty after cleaning -> train:", int(empty_tr.sum()), "| val:", sum(len(t) == 0 for t in tok_va),
      "| test:", sum(len(t) == 0 for t in tok_te))
keep = ~empty_tr
tr_df = tr_df[keep].reset_index(drop=True)
tok_tr = [t for t, k in zip(tok_tr, keep) if k]

stoi, counts = sl.build_vocab(tok_tr, CFG["min_freq"], CFG["max_vocab"])
V = len(stoi)
with open(vocab_path, "w") as f:
    json.dump({"stoi": stoi, "min_freq": CFG["min_freq"], "max_vocab": CFG["max_vocab"], "built_from": "train split only"}, f)
tr_ids, tr_lens = sl.numericalize(tok_tr, stoi, CFG["max_len"], CFG["head_tokens"])
va_ids, va_lens = sl.numericalize(tok_va, stoi, CFG["max_len"], CFG["head_tokens"])
te_ids, te_lens = sl.numericalize(tok_te, stoi, CFG["max_len"], CFG["head_tokens"])
tr_y, va_y, te_y = (d.label.values.astype(np.int64) for d in (tr_df, va_df, te_df))
np.savez_compressed(cache, tr_ids=tr_ids, tr_lens=tr_lens, tr_y=tr_y, va_ids=va_ids, va_lens=va_lens, va_y=va_y,
                    te_ids=te_ids, te_lens=te_lens, te_y=te_y, vocab_size=V)

clean_len = np.array([len(t) for t in tok_tr])
oov = lambda toks: sum(w not in stoi for t in toks for w in t) / max(1, sum(len(t) for t in toks))
print("unique train tokens: %d -> vocab size %d (incl. pad/unk)" % (len(counts), V))
print("OOV rate  val %.4f | test %.4f" % (oov(tok_va), oov(tok_te)))
print("tokens/review after cleaning: mean %.1f | median %.0f | p95 %.0f | truncated (>256): %.2f%%" % (
    clean_len.mean(), np.median(clean_len), np.percentile(clean_len, 95), 100 * (clean_len > CFG["max_len"]).mean()))
print("top-25 train tokens:", [w for w, _ in counts.most_common(25)])
print("saved", R(vocab_path), "and", R(cache))
import gc
del tok_tr, tok_va, tok_te, train_df
gc.collect()
""")

md("""
### 2.1.5 Embeddings learned from scratch
None of the three models loads external vectors. Each has its own `nn.Embedding(V, d)`, randomly initialised and trained
end-to-end by back-propagating the classification loss:
* **Baseline:** 64-dim, over V unigrams plus 262,144 hashed bigram buckets. Low dimension, because the classifier on top is only linear.
* **Exp A / Exp B:** 128-dim word embeddings. Exp B also learns a 257×128 position-embedding table and a [CLS] vector.

After training (section 2.2), nearest-neighbour lookups in the learned embedding spaces show the vectors picked up sentiment.
""")

md("""
## 2.2 Model training and evaluation
Shared setup: AdamW, linear warm-up then linear decay, gradient clipping at 1.0, binary cross-entropy on one logit,
batch 256 with length-bucketed dynamic padding. The checkpoint with the lowest validation log-loss is kept.
Each model writes its own unedited raw log.

| Model | Architecture | Key hyperparameters | Why |
|---|---|---|---|
| Baseline | Embedding(V+262,144 buckets, 64) → masked mean over unigrams+bigrams → dropout 0.2 → Linear(64,1) | lr 2e-3, 4 epochs, no weight decay | Joulin et al. 2017: a near-linear n-gram model is a strong, cheap floor. Bigrams capture short negation, which a unigram mean pool cannot |
| Exp A | Embedding(V,128) → BiGRU(128/dir) → additive attention over 256-d states → dropout 0.3 → Linear(256,1) | lr 1e-3, 3 epochs, wd 0.01 | GRU has fewer gates than LSTM (faster, less overfitting); attention pooling lets the verdict sentence dominate a long review instead of only the last state |
| Exp B | Embedding(V,128) + learned positions + [CLS] → 2× pre-norm encoder block (4 heads, FFN 256, GELU) → LN → Linear(128,1) | lr 5e-4, 5% warm-up, 3 epochs, dropout 0.1 | Self-attention links negations and contrasts to distant words in one step; written from scratch (no `nn.Transformer`) |
""")

code("""
# each model trains in its own process (src/train_one.py) so MPS memory is fully released between models
import subprocess
results, histories, summaries, models = {}, {}, {}, {}
for name, mcfg in CFG["models"].items():
    print("=" * 100, "\\n", name, "—", mcfg["description"])
    ckpt_path = P["ckpt"] / f"{name}_{TAG}.pt"
    hist_path = P["out"] / f"{name}_history_{TAG}.json"
    if not (ckpt_path.exists() and hist_path.exists()):
        run = subprocess.run([sys.executable, str(P["src"] / "train_one.py"), name, TAG], capture_output=True, text=True)
        print(run.stdout[-6000:])
        if run.returncode != 0:
            raise RuntimeError(run.stderr[-3000:])
    else:
        print("already trained; reusing", R(ckpt_path), "(delete it to retrain)")
        print(open(P["logs"] / f"{name}_{TAG}.log").read()[-3000:])
    with open(hist_path) as f:
        h = json.load(f)
    histories[name], summaries[name] = h["hist"], h["summary"]
    model = sl.build_model(name, mcfg, V, CFG).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    models[name] = model
    print("parameters: {:,} | raw log: {} | checkpoint: {}".format(
        sl.count_params(model), R(P["logs"] / f"{name}_{TAG}.log"), R(ckpt_path)))
with open(P["out"] / f"train_histories_{TAG}.json", "w") as f:
    json.dump({"histories": histories, "summaries": summaries}, f, indent=1)
""")

code("""
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for name, h in histories.items():
    l, = ax[0].plot(h["epoch"], h["train_loss"], "o--", label=f"{name} train")
    ax[0].plot(h["epoch"], h["val_loss"], "s-", color=l.get_color(), label=f"{name} val")
    ax[1].plot(h["epoch"], h["val_macro_f1"], "s-", label=name)
ax[0].set(xlabel="epoch", ylabel="BCE log-loss", title="Training vs validation loss"); ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
ax[1].set(xlabel="epoch", ylabel="macro-F1", title="Validation macro-F1"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
fig.tight_layout(); fig.savefig(P["out"] / f"training_curves_{TAG}.png", dpi=130); plt.show()
""")

md("""
### Test-set evaluation (all required metrics)
Every metric is computed on the 38,000-review official test set at threshold 0.5.
* **ECE** uses 15 equal-width bins on the confidence of the predicted class.
* **Confidence intervals** come from a 1,000-resample percentile bootstrap.
* **McNemar** is paired on the same test reviews; I report both the exact binomial p and the continuity-corrected χ² p.
""")

code("""
probs, infer_eps = {}, {}
for name, model in models.items():
    sl.sync(device); t0 = time.time()
    probs[name] = sl.predict_proba(model, te_ids, te_lens, device)
    sl.sync(device); infer_eps[name] = len(te_lens) / (time.time() - t0)
    np.savez_compressed(P["out"] / f"{name}_test_predictions_{TAG}.npz", prob_pos=probs[name], label=te_y)

all_metrics, cms = {}, {}
for name in models:
    m, cm = sl.full_metrics(te_y, probs[name], CFG["bootstrap_resamples"], SEED, CFG["ece_bins"])
    all_metrics[name], cms[name] = m, cm

base = "baseline_fasttext"
mcn = {name: sl.mcnemar(te_y, (probs[base] >= .5).astype(int), (probs[name] >= .5).astype(int))
       for name in models if name != base}

q = np.percentile(te_df.n_words, [25, 50, 75])
masks = sl.slice_masks([sl.repair(t) for t in te_df.text], te_df.n_words.values, q)
slices = {name: sl.slice_metrics(te_y, probs[name], masks) for name in models}

show = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_micro", "f1_weighted", "roc_auc", "pr_auc", "mcc", "brier", "ece"]
display(pd.DataFrame(all_metrics).T[show].round(4))
ci = pd.DataFrame({n: {k: f"[{m[k + '_ci95_low']:.4f}, {m[k + '_ci95_high']:.4f}]" for k in ("accuracy", "macro_f1", "mcc")}
                   for n, m in all_metrics.items()}).T
print("95% bootstrap CIs"); display(ci)
print("McNemar vs baseline"); display(pd.DataFrame(mcn).T)
""")

code("""
sl_df = pd.concat({n: pd.DataFrame(s).T for n, s in slices.items()}, axis=1)
print("Per-slice macro-F1 / error rate (test)")
display(sl_df.round(4))
cost = pd.DataFrame({n: {"params": summaries[n]["params"], "train_time_sec": summaries[n]["train_time_sec"],
                         "train_examples_per_sec": summaries[n]["train_examples_per_sec"],
                         "test_inference_examples_per_sec": infer_eps[n],
                         "peak_device_memory_mb": summaries[n]["peak_device_memory_mb"],
                         "peak_mps_driver_memory_mb": summaries[n]["peak_mps_driver_memory_mb"],
                         "peak_process_rss_mb": summaries[n]["peak_process_rss_mb"],
                         "nan_steps": summaries[n]["nan_count"], "hardware": summaries[n]["hardware"]} for n in models}).T
display(cost)
""")

code("""
from sklearn.metrics import roc_curve, precision_recall_curve
fig, ax = plt.subplots(1, 3 + 3, figsize=(26, 4))
for i, name in enumerate(models):
    cm = cms[name]
    ax[i].imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        ax[i].text(c, r, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    ax[i].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["neg", "pos"], yticklabels=["neg", "pos"],
              xlabel="predicted", ylabel="true", title=f"{name}")
for name in models:
    fpr, tpr, _ = roc_curve(te_y, probs[name]); ax[3].plot(fpr, tpr, label=f"{name} AUC={all_metrics[name]['roc_auc']:.4f}")
    pr, rc, _ = precision_recall_curve(te_y, probs[name]); ax[4].plot(rc, pr, label=f"{name} AP={all_metrics[name]['pr_auc']:.4f}")
    _, rows = sl.ece_score(te_y, probs[name], CFG["ece_bins"])
    ax[5].plot([r[0] for r in rows], [r[1] for r in rows], "o-", ms=3, label=f"{name} ECE={all_metrics[name]['ece']:.4f}")
ax[3].set(title="ROC", xlabel="FPR", ylabel="TPR"); ax[4].set(title="Precision-recall", xlabel="recall", ylabel="precision", ylim=(0.8, 1.0))
ax[5].plot([0.5, 1], [0.5, 1], "k--", lw=.8); ax[5].set(title="Reliability (predicted-class confidence)", xlabel="confidence", ylabel="accuracy")
for a in ax[3:]:
    a.legend(fontsize=7); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(P["out"] / f"confusion_roc_pr_calibration_{TAG}.png", dpi=120); plt.show()
for name in models:
    f, a = plt.subplots(figsize=(3.6, 3.2)); cm = cms[name]; a.imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        a.text(c, r, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    a.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["neg", "pos"], yticklabels=["neg", "pos"], xlabel="predicted", ylabel="true", title=name)
    f.tight_layout(); f.savefig(P["out"] / f"{name}_confusion_matrix_{TAG}.png", dpi=130); plt.close(f)
""")

code("""
# one CSV row per model with every required metric
rows = []
for name in models:
    m, s = all_metrics[name], summaries[name]
    r = {"model": name, "description": CFG["models"][name]["description"], **{k: m[k] for k in m}}
    r["mcnemar_b_vs_baseline"] = mcn.get(name, {}).get("b_baseline_right_model_wrong", "")
    r["mcnemar_c_vs_baseline"] = mcn.get(name, {}).get("c_baseline_wrong_model_right", "")
    r["mcnemar_chi2_vs_baseline"] = mcn.get(name, {}).get("chi2_cc", "")
    r["mcnemar_p_exact_vs_baseline"] = mcn.get(name, {}).get("p_exact", "")
    for sname, v in slices[name].items():
        r[f"slice_macro_f1[{sname}]"] = v["macro_f1"]
        r[f"slice_error_rate[{sname}]"] = v["error_rate"]
        r[f"slice_n[{sname}]"] = v["n"]
    r.update({"param_count": s["params"], "train_time_sec": s["train_time_sec"],
              "train_examples_per_sec": s["train_examples_per_sec"], "test_inference_examples_per_sec": infer_eps[name],
              "peak_device_memory_mb": s["peak_device_memory_mb"],
              "peak_mps_driver_memory_mb": s["peak_mps_driver_memory_mb"], "peak_process_rss_mb": s["peak_process_rss_mb"],
              "nan_steps": s["nan_count"], "epochs": CFG["models"][name]["epochs"], "best_val_loss": s["best_val_loss"],
              "hardware": s["hardware"], "checkpoint": f"task2_sentiment/rajesh_paruchuri/checkpoints/{name}_{TAG}.pt",
              "raw_log": f"reproducibility/raw_logs/rajesh_paruchuri/task2_sentiment/{name}_{TAG}.log"})
    rows.append(r)
csv_path = P["member"] / "metrics_report.csv" if not SMOKE else P["out"] / "metrics_report_smoke.csv"
pd.DataFrame(rows).to_csv(csv_path, index=False)
with open(P["out"] / f"full_metrics_{TAG}.json", "w") as f:
    json.dump({"metrics": all_metrics, "confusion_matrices": {k: v.tolist() for k, v in cms.items()}, "mcnemar": mcn,
               "slices": slices, "slice_length_quartiles_words": q.tolist(), "cost": cost.to_dict(orient="index")}, f, indent=1, default=str)
print("saved", R(csv_path))
""")

md("""
### Learned-embedding sanity check
Cosine nearest neighbours in each model's trained embedding table. Words are shown as stems.
""")

code("""
itos = {i: w for w, i in stoi.items()}
def neighbours(E, word, k=8):
    if word not in stoi:
        return []
    E = torch.nn.functional.normalize(E[:V].float(), dim=1)
    sims = E @ E[stoi[word]]
    return [itos[int(i)] for i in sims.topk(k + 1).indices[1:]]
for name, model in models.items():
    E = model.emb.weight.detach().cpu()
    print(name)
    for w in ["delici", "terribl", "rude", "recommend", "slow"]:
        print(f"   {w:10s} -> {neighbours(E, w)}")
""")

md("""
### Manual error review — selection of 20 errors
The model under review is the **experimental model with the best validation macro-F1**, chosen on validation data, not test.
The 20 errors are:
* 5 confident false positives (true negative, highest p(pos))
* 5 confident false negatives (true positive, lowest p(pos))
* 5 near-threshold errors (smallest |p − 0.5|)
* 5 slice-specific failures, drawn at random (seed 266) from the remaining errors in the slice where the model's error rate is highest

My manual annotations (error type and a testable fix) are in `../failure_analysis.md`.
""")

code("""
exp_names = [n for n in models if n != base]
val_f1 = {n: histories[n]["val_macro_f1"][int(np.argmin(histories[n]["val_loss"]))] for n in exp_names}
review = max(val_f1, key=val_f1.get)
p = probs[review]; pred = (p >= .5).astype(int); err = np.where(pred != te_y)[0]
fp_idx = [i for i in err[np.argsort(-p[err])] if te_y[i] == 0][:5]
fn_idx = [i for i in err[np.argsort(p[err])] if te_y[i] == 1][:5]
used = set(fp_idx) | set(fn_idx)
nt_idx = [i for i in err[np.argsort(np.abs(p[err] - .5))] if i not in used][:5]
used |= set(nt_idx)
worst = max(slices[review], key=lambda s: slices[review][s]["error_rate"] if slices[review][s]["n"] >= 100 else -1)
pool = [i for i in err if masks[worst][i] and i not in used]
sl_idx = list(np.random.default_rng(SEED).choice(pool, size=min(5, len(pool)), replace=False))
cases = []
for group, idxs in [("confident_false_positive", fp_idx), ("confident_false_negative", fn_idx),
                    ("near_threshold", nt_idx), (f"slice:{worst}", sl_idx)]:
    for i in idxs:
        i = int(i)
        cases.append({"group": group, "test_index": i, "true": int(te_y[i]), "pred": int(pred[i]), "p_pos": float(p[i]),
                      "n_words": int(te_df.n_words[i]), "text": sl.repair(te_df.text[i]),
                      "tokens_seen_by_model": [itos[int(t)] for t in te_ids[i, :te_lens[i]]][:80],
                      "other_models_p_pos": {n: float(probs[n][i]) for n in models if n != review}})
with open(P["out"] / f"error_cases_{review}_{TAG}.json", "w") as f:
    json.dump({"model": review, "worst_slice": worst, "val_macro_f1": val_f1, "cases": cases}, f, indent=1)
print("model reviewed:", review, "| val macro-F1:", {k: round(v, 4) for k, v in val_f1.items()}, "| worst slice:", worst)
for k, c in enumerate(cases, 1):
    print(f"\\n[{k:02d}] {c['group']} | true={c['true']} pred={c['pred']} p(pos)={c['p_pos']:.3f} | {c['n_words']} words | "
          f"others: {{{', '.join(f'{n}: {v:.2f}' for n, v in c['other_models_p_pos'].items())}}}")
    print("    ", c["text"][:700].replace("\\n", " "))
""")

md("""
## 2.3 Comparative analysis
The written comparison of my three models,
strengths, weaknesses, limitations and future work are in `../results.md`.
""")

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for k, c in enumerate(nb["cells"]):
    c["id"] = f"c{k:02d}"
Path(__file__).with_name("part2_sentiment.ipynb").write_text(json.dumps(nb, indent=1))
print("wrote part2_sentiment.ipynb with", len(cells), "cells")
