# Task 2 — Yelp Polarity Sentiment Classification — Sivasurya Chandran

The numbers here come from `checkpoints/metrics_task2.json`, `data_processed/preprocess_stats.json` (also copied to `outputs/eda/`) and `checkpoints/manifest.json`. I generated the tables with `tools/make_tables.py`.

**Hardware.** All three models were trained on an Apple M5 (10-core CPU, integrated GPU through PyTorch MPS), macOS 26.6, Python 3.9, PyTorch 2.8.0. The manifests from this run only say "Apple Silicon (MPS)" for the GPU; `common/utils.py` now records the actual chip name.

**Where things are:**
- Checkpoints: `checkpoints/{baseline,experimental_1,experimental_2}.pt`
- Raw log: `logs/train_raw.log`, with a byte-identical copy at `reproducibility/raw_logs/sivasurya_chandran/task2_sentiment/train_raw.log`
- Notebook: `src/task2_sentiment.ipynb`
- Plots: `outputs/test_curves.png`, `outputs/test_confusion.png` and `outputs/eda/`

## Data and preprocessing

**Dataset.** I used all of Yelp Polarity (`fancyzhx/yelp_polarity`):
- The official 560,000 training reviews are split 90/10 with seed 1337 into **503,979 train** and **55,997 validation** reviews.
- The official **38,000 test** reviews are only used for the final numbers.

**EDA** (`outputs/eda/`):
- The classes are exactly balanced, 280,000 each.
- Reviews average 133 words: median 97, 95th percentile 372, longest 1,052.
- Negative reviews tend to be longer than positive ones, 152 vs 115 words on average.

**Malformed entries.** I removed 24 training reviews that were empty or had no letters at all. There were no exact duplicates, and nothing was removed from the test set.

**Cleaning:**
- lowercase everything, strip URLs and HTML, and turn `n't` into `not`;
- remove punctuation and special characters;
- remove NLTK stopwords, **except** negations and intensifiers (not, no, never, but, too, very, only, …). Dropping those can flip or weaken the sentiment of a sentence;
- Porter stemming.

**Tokens and vocabulary:**
- Tokens are split on whitespace after cleaning.
- The vocabulary is built from the training split only: the 50,000 most frequent tokens that appear at least 3 times, plus `<pad>` = 0 and `<unk>` = 1.
- After cleaning, a review averages 72 tokens. 2.3% are cut off at 256 tokens, and 0.4% of test tokens are out of vocabulary.

**Embeddings.** Every model learns its own `nn.Embedding(50000, 128)` from a random start. I didn't use any pretrained vectors or language models.

## Models and hyperparameters

All three use Adam, batch size 128, at most 5 epochs and cross-entropy loss, with gradient clipping at 5. I used early stopping on validation macro-F1 with patience 2 and kept the best checkpoint.

| Model | Architecture | Key hyperparameters | Params |
|---|---|---|---|
| Baseline | Word embeddings averaged → MLP (fastText-style) | embed 128, hidden 128, dropout 0.3, lr 2e-3 | 6,416,770 |
| Experimental 1 | CNN with kernel widths 3/4/5 × 128 filters, ReLU, max-over-time pooling (Kim 2014) | embed 128, dropout 0.5, lr 1e-3 | 6,597,762 |
| Experimental 2 | BiLSTM (1 layer, 128 per direction, packed sequences) + additive attention pooling | embed 128, dropout 0.3, lr 1e-3 | 6,697,730 |

**Why these three.** I wanted a clear ladder from "ignores word order" to "reads the whole sequence".

- **Baseline.** Averaging embeddings tells the model which words are present but not their order. It gives a strong lower bound and trains in about three minutes.
- **CNN.** It adds local word order. Filters of width 3 to 5 act like learned n-gram detectors ("not good", "highly recommend"), which should help with negation and short phrases.
- **BiLSTM with attention.** It reads the full review in both directions. Attention pooling lets it focus on the sentence that actually decides the review, instead of averaging everything, and the attention weights can be inspected afterwards.
- **Same embedding everywhere.** All three use the same 128-d embedding, so the comparison is about architecture and not embedding size. The embedding table alone is 6.4M parameters, between 95.5% (BiLSTM) and 99.7% (baseline) of each model.

### How this differs from my teammate's models

Rajesh Paruchuri built a different set of three models, with a different pipeline:

| | Mine | Rajesh's |
|---|---|---|
| Baseline | mean-pooled unigram embeddings → MLP | fastText with hashed bigrams → linear |
| Experimental models | multi-width CNN; BiLSTM + attention | BiGRU + attention; 2-layer Transformer encoder |
| Selection | early stopping on val macro-F1 | lowest val log-loss |
| Truncation | first 256 tokens | head + tail (first 160 + last 96 tokens) |
| Validation split | 10% (55,997 reviews) | 20,000 reviews |
| Best test accuracy | 0.9581 (BiLSTM) | 0.9523 (BiGRU) |

Rajesh's bigram baseline is much stronger than my unigram one (0.9515 vs 0.9332). That confirms that bigrams such as "not good" carry most of what word order adds. My recurrent model ends up about 0.6 points ahead of his, probably because of the longer LSTM training (5 epochs vs 3) and model selection on F1 rather than loss. I haven't tested either explanation.

## Metrics (38,000 test reviews)

| Metric | Baseline (bag) | Exp. 1 (CNN) | Exp. 2 (BiLSTM) |
|---|---|---|---|
| Accuracy | 0.9332 | 0.9502 | **0.9581** |
| Precision / Recall / F1, macro | 0.9332 / 0.9332 / 0.9332 | 0.9502 / 0.9502 / 0.9502 | **0.9581 / 0.9581 / 0.9581** |
| Precision / Recall / F1, micro | 0.9332 / 0.9332 / 0.9332 | 0.9502 / 0.9502 / 0.9502 | **0.9581 / 0.9581 / 0.9581** |
| Precision / Recall / F1, weighted | 0.9332 / 0.9332 / 0.9332 | 0.9502 / 0.9502 / 0.9502 | **0.9581 / 0.9581 / 0.9581** |
| Confusion matrix [[TN, FP], [FN, TP]] | [[17739, 1261], [1276, 17724]] | [[18002, 998], [895, 18105]] | [[18196, 804], [790, 18210]] |
| ROC-AUC | 0.9812 | 0.9883 | **0.9921** |
| PR-AUC | 0.9816 | 0.9882 | **0.9923** |
| MCC | 0.8665 | 0.9004 | **0.9161** |
| Brier score | 0.0503 | 0.0384 | **0.0323** |
| Expected calibration error | **0.0082** | 0.0105 | 0.0124 |
| Accuracy 95% CI | [0.9309, 0.9359] | [0.9479, 0.9523] | [0.9560, 0.9601] |
| Macro-F1 95% CI | [0.9309, 0.9359] | [0.9479, 0.9523] | [0.9560, 0.9601] |
| MCC 95% CI | [0.8618, 0.8717] | [0.8957, 0.9046] | [0.9121, 0.9202] |
| McNemar vs baseline (b / c / p) | — | 688 / 1332 / p < 1e-16 | 542 / 1485 / p < 1e-16 |
| Parameters | 6,416,770 | 6,597,762 | 6,697,730 |
| Training time (s), as logged | 173 | 791 | 19,092 * |
| Train examples/sec, as logged | 14,535 | 3,188 | 132 * |
| Inference examples/sec | 105,134 | 8,354 | 2,959 |
| Peak memory (MB, MPS driver) | 2,810 | 2,828 | 4,006 |

\* **The BiLSTM's logged time includes about 8,500 s of laptop sleep.** The laptop slept from 20:03 to 22:25 (from `pmset -g log`) while it was training, which paused the process. Its real compute time was about **10,600 s**: roughly 2.9 hours, 35 minutes per epoch, or 240 examples per second. I left the logged numbers as they are.

**Notes on the metrics:**
- **Precision, recall and F1 are identical across averaging methods.** Macro, micro and weighted come out the same because the test set is exactly balanced (19,000 per class) and the errors are nearly symmetric.
- **Confidence intervals** are percentiles from 1,000 bootstrap resamples.
- **McNemar** uses the chi-squared test with continuity correction. Here b is the number of reviews the baseline got right and the other model got wrong, and c is the reverse. Both p-values are so small they round to 0.
- **ECE** uses 15 equal-width confidence bins.

### Validation macro-F1 per epoch

| Model | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| Baseline | 0.9279 | 0.9293 | **0.9302** | 0.9302 | 0.9300 |
| CNN | 0.9387 | 0.9420 | 0.9452 | 0.9455 | **0.9462** |
| BiLSTM | 0.9454 | 0.9495 | 0.9528 | **0.9547** | 0.9541 |

The baseline levels off after three epochs, the CNN is still improving slowly at epoch 5, and the BiLSTM peaks at epoch 4. None of them triggered early stopping within 5 epochs.

### Robustness: per-slice macro-F1 / error rate

| Slice | Baseline | CNN | BiLSTM |
|---|---|---|---|
| Short reviews (≤ 25th-percentile words) | 0.9304 / 0.0668 | 0.9483 / 0.0495 | **0.9541 / 0.0441** |
| Long reviews (≥ 75th-percentile words) | 0.9305 / 0.0661 | 0.9459 / 0.0516 | **0.9543 / 0.0435** |
| Contains negation | 0.9247 / 0.0705 | 0.9467 / 0.0499 | **0.9553 / 0.0419** |
| Contains "but/however/although" | 0.9241 / 0.0751 | 0.9450 / 0.0544 | **0.9530 / 0.0465** |
| Exclamation-heavy (≥ 3 "!") | 0.9520 / 0.0468 | 0.9656 / 0.0334 | **0.9727 / 0.0265** |

## Comparison

**Ranking.** The BiLSTM is the best of the three on every accuracy-type metric and on every slice, the CNN is second and the baseline last. The 95% confidence intervals don't overlap, and McNemar's test clearly rejects "same error rate as the baseline" for both experimental models.

**Word order helps most where you'd expect.** The baseline's two worst slices are reviews with a contrast word (7.5% error) and reviews with a negation (7.1%). Both depend on word order, and both improve the most:

| Slice | Baseline | CNN | BiLSTM |
|---|---|---|---|
| Contrast | 7.5% | 5.4% | 4.65% |
| Negation | 7.1% | 5.0% | 4.2% |

**Calibration.** The baseline has the best ECE (0.008), even though it's the least accurate. The BiLSTM has the best Brier score but a slightly higher ECE (0.012), so it's a bit over-confident. You can see that in the error review, where some of its mistakes have probabilities above 0.999. Temperature scaling on the validation set would be a cheap way to fix this.

**Cost.** The accuracy gains are not free:
- The CNN trains about 4.5× slower than the baseline.
- The BiLSTM trains about 60× slower, because the recurrence is sequential and that doesn't suit MPS well, and it is 35× slower at inference.

If I had to deploy one of these at scale, I'd pick the CNN as the best balance of accuracy and cost.

**Limitations:**
- **Label noise.** Labels come from star ratings, so some "errors" are really noisy labels: 4 of the 20 I reviewed.
- **Truncation.** Cutting reviews at 256 tokens hides verdicts that come at the end of long reviews.
- **Preprocessing.** Stemming and stopword removal throw away some nuance.
- **Training length.** The BiLSTM only had 5 epochs. Its validation score was flat at the end (95.47% after epoch 4, 95.41% after epoch 5), so more epochs probably wouldn't add much.

**What I'd try next:**
- head + tail truncation, or a longer `max_seq_len` of 512;
- marking contrast and negation scope during preprocessing;
- temperature scaling for calibration;
- a 2-layer BiLSTM or a small Transformer encoder trained from scratch;
- filtering out likely mislabeled training examples (confident learning).

## Error analysis (20 errors)

I went through 20 of the BiLSTM's mistakes by hand:
- the 5 most confident false positives;
- the 5 most confident false negatives;
- the 5 closest to the 0.5 threshold;
- 5 from the worst slice.

Each one has an error type and a fix that could be tested. They're all in [`failure_analysis.md`](failure_analysis.md).
