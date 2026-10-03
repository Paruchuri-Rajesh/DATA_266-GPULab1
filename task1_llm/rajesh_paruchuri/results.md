# Task 1 — Character-level GPT from scratch on TinyStories (Rajesh Paruchuri)

Code: `src/char_gpt.py` (model, data, training, metrics) driven by `src/part1_llm.ipynb` (executed, with outputs).
Config: `src/config.json`. Seed: 266.

## What I built
A decoder-only, pre-norm GPT where every Transformer part is written by hand:
- LayerNorm (mean/variance normalisation with learnable γ, β);
- causal multi-head self-attention (fused QKV, `QKᵀ/√d`, upper-triangular `-inf` mask, softmax);
- a GELU feed-forward block;
- residual connections;
- learnable token and positional embeddings;
- a language-model head tied to the token embedding.

No `nn.Transformer*`, `nn.MultiheadAttention` or `F.scaled_dot_product_attention` is used.
The notebook also checks the causal mask: changing future tokens changes earlier logits by exactly 0.0.

| Hyperparameter | Value | Why |
|---|---|---|
| Layers / heads / width | 6 / 6 / 192 (head dim 32) | Six blocks give depth for sentence-level structure while keeping the model under 3M parameters, small enough to train 10 epochs on a laptop GPU |
| Context (block size) | 192 characters | Covers 2–3 TinyStories sentences; attention cost grows with T², so 192 balances context against speed |
| FFN | 192 → 768 → 192, GELU | standard 4× expansion |
| Dropout | 0.1 (embeddings, attention weights, residual branches) | light regularisation; 100K windows is a small dataset for 2.7M params |
| Init | N(0, 0.02), residual output projections × 1/√(2·6) | GPT-2 recipe; keeps the residual stream variance stable with depth |
| Parameters | **2,719,488** (LM head tied, counted once) | |
| Optimiser | AdamW, β = (0.9, 0.95), weight decay 0.1 on matrices only, grad clip 1.0 | |
| LR schedule | linear warm-up 500 steps → peak 1e-3 → cosine decay to 1e-4 | warm-up avoids the early gradient blow-up (pre-clip norm reached 8.7 at step 0) |
| Batch / epochs / steps | 64 / 10 / 15,630 | |

## Data (1.1)
- **Source:** TinyStories train shard 0 (`roneneldan/TinyStories`, 529,930 stories), downloaded by the notebook.
- **Cleaning:**
  - repaired cp1252 mojibake in 31,850 stories (`â€œ` → `"`, `â€™` → `'`, and so on);
  - folded curly quotes and dashes to ASCII;
  - dropped 56 empty or very short (<50-char) stories.
- **My split is story-disjoint.** Stories are shuffled with seed 266. The first 2,141 fill the validation stream (1.93M chars) and the next 21,320 fill the training stream (19.3M chars), so no story appears in both.
  Each story ends with an `<|eos|>` character.
  Each stream is cut into non-overlapping 193-character windows (x = w[:-1], y = w[1:]), giving **100,000 train and 10,000 validation sequences**.
- **Tokeniser:** my own `char_to_idx` / `idx_to_char`, built from the train stream only.
  Vocab is 68: `<unk>`, `<|eos|>` and 66 characters seen at least 25 times.
- Saved under `data_processed/` (`char_tokenizer.json`, `sequences.npz`).

## Results (best checkpoint = epoch 10, `checkpoints/best.pt`)
| Metric | Value |
|---|---|
| Train cross-entropy (eval mode, 10K train windows) | 0.6920 |
| Train cross-entropy (running, dropout on) | 0.7420 |
| **Validation cross-entropy** | **0.7111** |
| Validation perplexity | 2.036 |
| Validation bits per character | 1.026 |
| Generalisation gap (val − train, both eval mode) | +0.019 |
| Top-1 next-character accuracy (val) | 77.46 % |
| Distinct-1 / 2 / 3, T = 0.7 (word level) | 0.387 / 0.801 / 0.935 |
| Repeated 4-gram rate, T = 0.7 | 0.005 |
| Distinct-1 / 2 / 3, greedy | 0.183 / 0.385 / 0.481 |
| Repeated 4-gram rate, greedy | 0.396 |
| Distinct-1 / 2 / 3, T = 1.0 | 0.481 / 0.879 / 0.972 |
| Repeated 4-gram rate, T = 1.0 | 0.003 |
| Grad norm, pre-clip (mean / p99 / max) | 0.386 / 1.157 / 8.75 (max at step 0) |
| NaN/Inf steps · loss spikes (>1.5× EMA) | 0 · 0 |
| Training throughput | 18,282 chars/s (training compute only) |
| Generation throughput | 100 chars/s (batch 1, no KV cache) |
| Peak memory | 3,308 MB MPS driver-allocated · 2,223 MB process RSS |
| Total training time | 11,173 s (3.1 h) incl. per-epoch evaluation |

Every value is in `metrics_report.csv`. Curves: `outputs/loss_curves_full.png`. Samples: `outputs/generated_samples_full.txt`.

**Training behaviour.** Validation CE fell every epoch: 0.985 → 0.863 → 0.812 → 0.782 → 0.762 → 0.746 → 0.733 → 0.722 → 0.714 → 0.711.
Each epoch produced a new best checkpoint, and the gains shrink as the cosine schedule reaches its floor, so the model is close to converged for this size.

The running training loss is higher than validation loss only because dropout is on during training.
Scored the same way (eval mode), train is 0.019 below val: a small, healthy gap with no overfitting.
Training was stable throughout. The only gradient norms above the clip came in the first ~600 steps (warm-up); afterwards they settled at about 0.3–0.4.

**Decoding.**
- Greedy decoding is fluent for 1–3 sentences, then loops.
- T = 0.7 gives the best balance.
- T = 1.0 adds invented words.

Details in `failure_analysis.md`.

## Hardware (disclosure)
- **Apple MacBook Air, M4 chip** (10-core CPU, 10-core GPU, 16 GB unified memory), PyTorch 2.11.0 MPS backend, fp32, macOS 26.5.2, Python 3.13.7.
- The laptop was memory-constrained during the run (other apps open, about 8 GB swap). Throughput was 18.3K chars/s, against about 29K chars/s for this same configuration on an unloaded machine earlier the same day.
  Wall time and tokens/sec therefore reflect a contended laptop, not the model's best speed.
- Raw, unedited log: `reproducibility/raw_logs/rajesh_paruchuri/task1_llm/train_full.log`.

## Notes on the design
- The story-disjoint split makes validation harder than a random-window split: none of the validation stories was seen during training, so the 0.711 val CE measures generalisation to new stories.
- The explicit `<|eos|>` token lets generation stop at a story boundary instead of running on into an unrelated story.
- The comparison with my Team 6 teammate's model belongs in the team report.

## Limitations and next steps
- No KV cache, so generation is slow (100 chars/s). Caching keys and values would make each new character O(T) instead of re-running the whole window.
- Character tokens spend most of the capacity on spelling. A BPE tokeniser at the same context would see about 4× more text and should help with Case 2 coherence.
- The val curve was still falling slowly at epoch 10. More epochs or more of the 530K-story shard would probably lower CE further.
