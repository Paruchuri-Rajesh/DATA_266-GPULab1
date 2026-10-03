# Task 2 — Yelp Polarity sentiment bake-off (Rajesh Paruchuri)

Code: `src/sentiment_lib.py` (preprocessing, models, metrics) and `src/train_one.py` (one model per process), driven by `src/part2_sentiment.ipynb` (executed, with outputs).
Config: `src/config.json`. Seed: 266. **No pretrained embeddings or language models anywhere:** every embedding table is randomly initialised and learned with its classifier.

## 2.1 Data and preprocessing
- **Data:** `fancyzhx/yelp_polarity`, 560,000 train / 38,000 test.
  Exactly balanced (280K/280K and 19K/19K).
  Labels: 0 = 1–2 stars, 1 = 3–4 stars.
- **Checks:**
  - no missing values, no blank texts, no duplicate texts, no train/test overlap;
  - the malformed entries are escaped text in the raw dump: 269,718 training reviews contain literal `\n`, 102,352 escaped quotes `\"`, 12,521 literal unicode escapes (`é`) and 26 HTML entities.
    `repair()` fixes all four;
  - 59 training reviews were empty after cleaning and were dropped (val/test rows are never dropped).
- **Length:** median 97 words (mean 133, max 1,052).
  Negative reviews are longer (mean 152 vs 115 words), so length alone carries weak signal.
  Plot: `outputs/eda_full.png`.
- **Split:**
  - 20,000 stratified validation reviews held out from the train file, used for checkpoint selection only;
  - 539,941 training reviews;
  - the official 38,000 test reviews, used only for final metrics.
- **Pipeline (in this order):**
  1. repair escapes;
  2. lowercase;
  3. strip URLs and HTML;
  4. expand `n't` to ` not`;
  5. keep `a–z` only (removes punctuation, digits and special characters);
  6. remove NLTK's 179 English stopwords **except 18 sentiment-bearing words** (not, no, nor, never, but, very, too, only, …);
  7. Snowball (Porter2) stemming;
  8. whitespace tokenisation.
  - Vocabulary: train split only, min frequency 3, giving **54,408 types** (from 139,713). Val/test OOV rate is 0.36% / 0.34%.
  - Head+tail truncation to 256 tokens (first 160 + last 96); only 2.33% of reviews are longer than that.
- **Embeddings:** learned from scratch (`nn.Embedding`), 64-d for the baseline and 128-d for the experimental models.
  Nearest neighbours after training show sentiment structure. In the baseline, `delici` is closest to amaz, awesom, fantast, excel, perfect; `terribl` is closest to overpr, worst, flavorless, poor.
  The BiGRU's input embeddings are noisier (its recurrent layer does more of the work).

## 2.2 Models and why
| | Architecture | Params | Hyperparameters | Justification |
|---|---|---|---|---|
| **Baseline** — fastText bigram | Embedding(54,408 + 262,144 hashed-bigram buckets, 64) → masked mean of unigram+bigram vectors → dropout 0.2 → Linear(64,1) | 20.26M (almost all embedding rows) | AdamW lr 2e-3, 4 epochs, 2% warm-up, linear decay | A near-linear n-gram model is a strong and cheap floor (Joulin et al., 2017). Hashed bigrams give "not good" its own vector, which a unigram mean-pool cannot |
| **Exp A** — BiGRU + attention | Embedding(V,128) → dropout 0.3 → packed 1-layer BiGRU (128/dir) → additive (Bahdanau) attention over the 256-d states → dropout → Linear(256,1) | 7.20M | lr 1e-3, wd 0.01, 3 epochs | Reads word order in both directions. GRU has fewer gates than LSTM (cheaper, less overfitting). Attention pooling lets the verdict sentence dominate a long review instead of the last hidden state. |
| **Exp B** — Transformer encoder (from scratch) | Embedding(V,128) + learned positions (257) + [CLS] → 2× pre-norm block (4-head self-attention with padding mask, FFN 256, GELU, dropout 0.1) → LN → Linear(128,1) | 7.26M | lr 5e-4, 5% warm-up, wd 0.01, 3 epochs | Global self-attention links a negation or "but" to any word in one hop. It tests whether attention beats recurrence without pre-training. No `nn.Transformer` / `nn.MultiheadAttention` |

All three: binary cross-entropy on one logit, batch 256 with length bucketing (padding rounded to multiples of 32), gradient clipping at 1.0.
The checkpoint with the lowest validation log-loss is kept.

## 2.2 Test metrics (38,000 reviews, threshold 0.5)
| Metric | Baseline (fastText) | Exp A (BiGRU+attn) | Exp B (Transformer) |
|---|---|---|---|
| Accuracy | 0.9515 | **0.9523** | 0.9323 |
| Precision / Recall / F1 — macro | 0.9515 / 0.9515 / 0.9515 | **0.9523 / 0.9523 / 0.9523** | 0.9323 / 0.9323 / 0.9323 |
| Precision / Recall / F1 — micro | 0.9515 / 0.9515 / 0.9515 | **0.9523 / 0.9523 / 0.9523** | 0.9323 / 0.9323 / 0.9323 |
| Precision / Recall / F1 — weighted | 0.9515 / 0.9515 / 0.9515 | **0.9523 / 0.9523 / 0.9523** | 0.9323 / 0.9323 / 0.9323 |
| Confusion [TN FP; FN TP] | [18,132 868; 975 18,025] | [18,070 930; 882 18,118] | [17,779 1,221; 1,353 17,647] |
| ROC-AUC | 0.9874 | **0.9903** | 0.9830 |
| PR-AUC (average precision) | 0.9872 | **0.9906** | 0.9837 |
| MCC | 0.9030 | **0.9046** | 0.8645 |
| Brier score | 0.0382 | **0.0363** | 0.0494 |
| ECE (15 bins) | 0.0145 | 0.0147 | **0.0079** |
| Accuracy 95% CI | [0.9496, 0.9535] | [0.9502, 0.9545] | [0.9298, 0.9346] |
| Macro-F1 95% CI | [0.9496, 0.9535] | [0.9502, 0.9545] | [0.9298, 0.9346] |
| MCC 95% CI | [0.8992, 0.9071] | [0.9005, 0.9090] | [0.8596, 0.8691] |
| McNemar vs baseline (b / c, exact p) | — | 745 / 776, **p = 0.44** | 1,197 / 466, **p = 4×10⁻⁷⁴** |
| Parameters | 20,259,393 | 7,195,649 | 7,262,593 |
| Training time (s) | 1,195 | 17,403 | 9,084 |
| Training examples/s | 1,807 | 93 | 178 |
| Test inference examples/s | 56,946 | 7,178 | 4,467 |
| Peak memory: MPS tensors / MPS driver pool / process RSS (MB) | 310 / 2,787 / 1,873 | 127 / 4,032 / 1,963 | 130 / 10,130 / 1,296 |
| Best epoch (by val loss) | 1 of 4 | 3 of 3 | 1 of 3 |

Macro, micro and weighted scores coincide because the test set is exactly balanced and both classes have near-equal error counts.
For a balanced binary task, micro-F1 equals accuracy by definition.
McNemar: b = baseline right / model wrong, c = baseline wrong / model right.
Figures:
- `outputs/confusion_roc_pr_calibration_full.png` (all three confusion matrices plus ROC, PR and reliability curves);
- `outputs/*_confusion_matrix_full.png`;
- `outputs/training_curves_full.png`.

All columns are in `metrics_report.csv`.

### Robustness: macro-F1 / error rate per slice (test)
| Slice (n) | Baseline | Exp A | Exp B |
|---|---|---|---|
| Length Q1 ≤ 51 words (9,590) | **0.948** / 4.95% | 0.947 / 5.12% | 0.929 / 6.81% |
| Length Q2 51–97 (9,450) | 0.952 / 4.80% | **0.955** / 4.50% | 0.932 / 6.79% |
| Length Q3 97–173 (9,502) | 0.952 / 4.81% | **0.955** / 4.55% | 0.930 / 6.97% |
| Length Q4 > 173 (9,458) | **0.949** / 4.83% | 0.949 / 4.91% | 0.931 / 6.52% |
| Has negation (28,375) | 0.948 / 5.06% | **0.950** / 4.89% | 0.925 / 7.25% |
| No negation (9,625) | **0.941** / 4.23% | 0.939 / 4.42% | 0.925 / 5.36% |
| Has contrast (but/however/…) (22,626) | 0.945 / 5.44% | **0.947** / 5.25% | 0.923 / 7.62% |
| No contrast (15,374) | **0.959** / 3.99% | 0.959 / 4.06% | 0.944 / 5.53% |
| Negation + contrast (19,814) | 0.943 / 5.49% | **0.945** / 5.30% | 0.920 / 7.78% |
| **Non-English** (133) | **0.875** / 11.3% | 0.810 / 17.3% | 0.781 / 20.3% |

### Training behaviour
| | Val log-loss by epoch | Val macro-F1 by epoch | Train loss by epoch |
|---|---|---|---|
| Baseline | 0.145 → 0.146 → 0.155 → 0.161 | 0.949 → 0.949 → 0.949 → 0.948 | 0.223 → 0.099 → 0.065 → 0.048 |
| Exp A | 0.144 → 0.133 → 0.133 | 0.944 → 0.950 → 0.952 | 0.220 → 0.144 → 0.129 |
| Exp B | 0.171 → 0.174 → 0.186 | 0.932 → 0.931 → 0.932 | 0.221 → 0.155 → 0.131 |

- The **baseline memorises**: its 20M embedding rows let train loss fall to 0.05 while val loss rises after epoch 1. Best-checkpoint selection keeps epoch 1.
- **Exp A** is the only model still improving at its last epoch.
- **Exp B** peaks at epoch 1 and then overfits, even though its training loss stays above the BiGRU's.
- No run produced a NaN/Inf step.

## 2.2 Manual error review
I reviewed 20 BiGRU errors: 5 confident FP, 5 confident FN, 5 near-threshold, and 5 from the worst slice (non-English).
Each has an error type and one testable fix; see `failure_analysis.md`.
The main finding: **6 of the 20 errors come from my own preprocessing** (deleted prices, case and emphasis, the stopword "just", and split accents).
A further 3 are label noise or text–rating mismatch.

## 2.3 Comparative analysis

### My three models
- **Exp A (BiGRU + attention) is my best model on every discrimination metric** (accuracy 0.9523, MCC 0.905, ROC-AUC 0.990, PR-AUC 0.991, Brier 0.036).
  But the gain over the baseline is **not statistically significant**: McNemar p = 0.44, and the bootstrap CIs overlap almost completely.
  The models disagree on 1,521 reviews, and those disagreements split almost evenly (745 vs 776).
  So they make *different* mistakes, not fewer ones: ensembling is worth testing.
  The BiGRU's ranking quality is measurably better (AUC 0.990 vs 0.987), and it helps most on mid-length reviews (Q2/Q3 error 4.5% vs 4.8%) and on negation + contrast reviews.
- **The fastText-bigram baseline is the efficiency winner:** 15× faster to train, 8× faster at inference, within 0.1 pp of the best accuracy, and the most robust on non-English text.
  The likely reason is that its averaged n-gram vectors are less thrown off by a few garbled French tokens than a sequence model that reads them in order; this is not tested.
- **Exp B (Transformer) is significantly worse** (−1.9 pp accuracy, McNemar p ≈ 10⁻⁷⁴), but it is the **best calibrated** (ECE 0.008, about half the others').
  Without pre-training, a 2-layer encoder over stemmed, stopword-stripped tokens has a weaker inductive bias than recurrence.
  Stopword removal also scrambles the positional structure it relies on, and it overfit after one epoch.
  Its strength is calibrated probabilities, useful if downstream decisions need a confidence threshold.
- **Shared weaknesses:**
  - contrast reviews ("but/however") are the hardest large slice for all three (5.3–7.6% error vs 4.0–5.5%);
  - non-English is the hardest slice overall;
  - mixed 2-vs-3-star reviews make up the near-threshold errors, and the label scheme itself puts that boundary at "it was okay".

### Team comparison
The comparison with my Team 6 teammate's models belongs in the team report.

## Limitations
- **Hardware made timing numbers noisy.** All three models trained on the same laptop, which was memory-constrained and running other apps.
  The BiGRU benchmarked at about 870 examples/s in a clean process, but averaged 93 during the full run; two epochs coincided with heavy system memory pressure.
  Training times and examples/s are real measurements, but they are not representative throughput.
- The first full attempt ran all three models in one kernel and was aborted when the MPS memory pool grew past 16 GB.
  Its partial logs are kept unedited (`*_aborted_attempt1.log`). The reported run trains each model in a fresh process.
- Each model was trained with one seed; the bootstrap CIs capture test-set sampling noise, not training randomness.
- Error review covers the BiGRU only; the other models' errors are visible in the per-slice table.

## Future work (each tied to an observed error)
1. Preprocessing ablation: keep digits/`$` as `<price>`, `<allcaps>`, letter grades and `just`; use a Unicode-aware tokeniser with per-language stemming (fixes 6 of 20 reviewed errors).
2. Train on 5-class stars with an ordinal loss, then collapse to binary (targets the 2-vs-3-star boundary errors).
3. Ensemble the baseline and BiGRU; their McNemar disagreement is large and balanced. A post-hoc check (averaging the saved test probabilities) already gives **0.9572 macro-F1 / 0.9572 accuracy**, +0.5 pp over the best single model. It is not in the main table because it was evaluated on test without validation-based selection.
4. Label-noise audit plus label smoothing.
5. Tune the decision threshold on validation instead of fixing it at 0.5.

## Hardware (disclosure)
All three models trained on an **Apple MacBook Air, M4 chip** (10-core CPU, 10-core GPU, 16 GB unified memory).
Software: PyTorch 2.11.0, MPS backend, fp32, macOS 26.5.2, Python 3.13.7. Preprocessing ran on 4 CPU worker processes.

Raw logs (unedited): `reproducibility/raw_logs/rajesh_paruchuri/task2_sentiment/{baseline_fasttext,exp_a_bigru_attn,exp_b_transformer}_full.log`.
