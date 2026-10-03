# DATA266 Lab 1 — Team 6 Report (DRAFT)

> **Draft status.** Rajesh Paruchuri's sections are complete and their numbers come from the committed `metrics_report.csv` files.
> Every cell marked **⟨Siva⟩** must be filled from Siva Surya Chandran's own metrics files, and the joint analysis must be written together.
> Export to `report/DATA266_Lab1_Report_Team_6.pdf` when finished.

**Team:** PairProgramming_Team_06 — Rajesh Paruchuri, Siva Surya Chandran
**Repository:** https://github.com/Paruchuri-Rajesh/DATA_266-GPULab1 ⟨replace with the shared team repo if different⟩

## Team ownership statement
Each member independently designed, coded, trained and evaluated their own models for all three tasks.
**Rajesh Paruchuri** built:
- a 6-layer character-level GPT for TinyStories (Task 1);
- a fastText-bigram baseline, a BiGRU with attention and a from-scratch Transformer encoder for Yelp polarity (Task 2);
- a CycleGAN with 6-block / ngf-48 generators, spectral-norm PatchGAN discriminators, DiffAugment and EMA generators (Task 3).

**Siva Surya Chandran** built ⟨Siva: one sentence per task⟩.
The comparison tables and analyses below were written jointly ⟨confirm⟩.

---

## Task 1 — GPT-style LLM from scratch (TinyStories)

| Metric | Rajesh Paruchuri | Siva Surya Chandran |
|---|---|---|
| Architecture | Decoder-only GPT, hand-written LayerNorm / causal MHA / GELU FFN, pre-norm, weight-tied head | ⟨Siva⟩ |
| Layers / heads / width / context | 6 / 6 / 192 / 192 chars | ⟨Siva⟩ |
| Data split | Story-disjoint: 100,000 train / 10,000 val non-overlapping windows, seed 266 | ⟨Siva⟩ |
| Hyperparameters | AdamW (0.9, 0.95), wd 0.1, lr 1e-3, 500-step warm-up + cosine to 1e-4, batch 64, dropout 0.1, 10 epochs | ⟨Siva⟩ |
| Parameters | 2,719,488 | ⟨Siva⟩ |
| Train CE (eval mode) / Val CE | 0.6920 / **0.7111** | ⟨Siva⟩ |
| Perplexity / bits per char | 2.036 / 1.026 | ⟨Siva⟩ |
| Generalisation gap (val − train) | +0.019 | ⟨Siva⟩ |
| Top-1 next-char accuracy | 0.7746 | ⟨Siva⟩ |
| Distinct-1/2/3, T = 0.7 | 0.387 / 0.801 / 0.935 | ⟨Siva⟩ |
| Repeated 4-gram rate (greedy / T = 0.7) | 0.396 / 0.005 | ⟨Siva⟩ |
| Grad norm mean / max · NaNs · loss spikes | 0.386 / 8.75 · 0 · 0 | ⟨Siva⟩ |
| Train / generation tokens/s | 18,282 / 100 | ⟨Siva⟩ |
| Peak memory · train time | 3,308 MB (MPS) · 11,173 s | ⟨Siva⟩ |
| Hardware | Apple M4 MacBook Air, MPS | ⟨Siva⟩ |

**Joint analysis** ⟨write together⟩.
Rajesh's points:
- The story-disjoint split makes validation a true out-of-story test.
- Training was stable (no spikes; gradient norm about 0.35 after warm-up).
- Greedy decoding loops (repeated 4-gram rate 0.40), while T = 0.7 does not.
- Coherence breaks down beyond about 2 sentences (speaker and entity drift).
- Next steps: KV cache for generation, BPE tokens, longer training.

---

## Task 2 — Yelp polarity sentiment classification

| Metric | Rajesh: fastText bigram (baseline) | Rajesh: BiGRU + attention | Rajesh: Transformer | Siva: baseline | Siva: exp 1 | Siva: exp 2 |
|---|---|---|---|---|---|---|
| Architecture | uni + hashed-bigram embedding bag → linear | BiGRU 128/dir + additive attention | 2-layer encoder, 4 heads, [CLS] | ⟨Siva⟩ | ⟨Siva⟩ | ⟨Siva⟩ |
| Key hyperparameters | emb 64, lr 2e-3, 4 ep | emb 128, lr 1e-3, 3 ep | emb 128, lr 5e-4, 3 ep | ⟨Siva⟩ | ⟨Siva⟩ | ⟨Siva⟩ |
| Accuracy | 0.9515 | **0.9523** | 0.9323 | ⟨Siva⟩ | ⟨Siva⟩ | ⟨Siva⟩ |
| Precision / recall / F1 (macro) | 0.9515 / 0.9515 / 0.9515 | 0.9523 / 0.9523 / 0.9523 | 0.9323 / 0.9323 / 0.9323 | ⟨Siva⟩ | ⟨Siva⟩ | ⟨Siva⟩ |
| F1 micro / weighted | 0.9515 / 0.9515 | 0.9523 / 0.9523 | 0.9323 / 0.9323 | ⟨Siva⟩ | ⟨Siva⟩ | ⟨Siva⟩ |
| ROC-AUC / PR-AUC | 0.9874 / 0.9872 | 0.9903 / 0.9906 | 0.9830 / 0.9837 | ⟨Siva⟩ | ⟨Siva⟩ | ⟨Siva⟩ |
| MCC | 0.9030 | 0.9046 | 0.8645 | ⟨Siva⟩ | ⟨Siva⟩ | ⟨Siva⟩ |
| Brier / ECE | 0.0382 / 0.0145 | 0.0363 / 0.0147 | 0.0494 / 0.0079 | ⟨Siva⟩ | ⟨Siva⟩ | ⟨Siva⟩ |
| 95% CI accuracy | [0.9496, 0.9535] | [0.9502, 0.9545] | [0.9298, 0.9346] | ⟨Siva⟩ | ⟨Siva⟩ | ⟨Siva⟩ |
| McNemar vs baseline (exact p) | — | 0.44 | 4×10⁻⁷⁴ | — | ⟨Siva⟩ | ⟨Siva⟩ |
| Worst slice macro-F1 (non-English, n = 133) | 0.875 | 0.810 | 0.781 | ⟨Siva⟩ | ⟨Siva⟩ | ⟨Siva⟩ |
| Params · train time | 20.26M · 1,195 s | 7.20M · 17,403 s | 7.26M · 9,084 s | ⟨Siva⟩ | ⟨Siva⟩ | ⟨Siva⟩ |
| Peak memory (MPS tensors / driver) | 310 / 2,787 MB | 127 / 4,032 MB | 130 / 10,130 MB | ⟨Siva⟩ | ⟨Siva⟩ | ⟨Siva⟩ |
| Hardware | Apple M4, MPS | Apple M4, MPS | Apple M4, MPS | ⟨Siva⟩ | ⟨Siva⟩ | ⟨Siva⟩ |

Full per-slice tables, confusion matrices and bootstrap CIs for MCC and macro-F1 are in each member's `metrics_report.csv`.

**Joint analysis** ⟨write together⟩.
Rajesh's points:
- The BiGRU's lead over the bigram baseline is not significant (McNemar p = 0.44), but their errors differ: averaging the two gives 0.9572 (post hoc).
- The from-scratch Transformer overfits after epoch 1 but is the best calibrated model.
- 6 of 20 reviewed errors come from preprocessing (deleted prices, case and emphasis, the stopword "just", split accents).
- Non-English reviews are the weakest slice.
- Next steps: ordinal 5-star training, a Unicode-aware tokeniser, an ensemble.

---

## Task 3 — CycleGAN photo ↔ Monet

| Metric | Rajesh Paruchuri | Siva Surya Chandran |
|---|---|---|
| Generators / discriminators | ResNet, 6 blocks, ngf 48, bilinear+conv decoder / spectral-norm 70×70 PatchGAN | ⟨Siva: v3 = warm start from v2, resize-conv upsampling, translation-only DiffAugment (from his Kaggle notes)⟩ |
| Training | DiffAugment (colour / translation / cutout), EMA, 128-px crops, 80 epochs × 300 iterations, LSGAN, λ_cyc 10, λ_id 5 | ⟨Siva⟩ |
| Parameters | 14,355,336 | ⟨Siva⟩ |
| FID photo→Monet / Monet→photo (all images) | 105.72 / 148.80 | ⟨Siva⟩ |
| KID photo→Monet / Monet→photo | 0.0308 / 0.0776 | ⟨Siva⟩ |
| Precision / recall (photo→Monet) | 0.247 / 0.523 | ⟨Siva⟩ |
| Cycle L1 (photo / Monet cycle) | 0.0774 / 0.0775 | ⟨Siva⟩ |
| LPIPS input vs translation (photo→Monet) | 0.523 | ⟨Siva⟩ |
| Content cosine (photo→Monet) | 0.373 | ⟨Siva⟩ |
| Final losses G-adv / D_A / D_B / cycle / identity | 0.645 / 0.214 / 0.216 / 0.358 / 0.373 | ⟨Siva⟩ |
| Grad norm G / D (mean) · NaNs | 44.2 / 5.5 · 0 | ⟨Siva⟩ |
| Course-script submission FID / MiFID | 149.48 / 0.414 | ⟨Siva⟩ |
| **Kaggle public score** | **−74.9486** | **−45.9695** (v3; v2 −46.3530; v1 −49.2570) |
| Human audit score · Cohen's kappa | ⟨pending: 2 raters⟩ | ⟨Siva⟩ |
| Train time · images/s · peak memory | 9,417 s · 5.1 · 409 MB tensors / 1,009 MB pool | ⟨Siva⟩ |
| Hardware | Apple M4, MPS | ⟨Siva⟩ |

**Kaggle:** team PairProgramming_Team_06, public rank **11** (ranked on the team's best submission, Siva's v3).

**Joint analysis** ⟨write together⟩.
Rajesh's points:
- Training was stable (no D collapse; D outputs about 0.57 real / 0.43 fake), but under-trained: cycle loss was still falling at epoch 80.
- The cycle constraint verifies: reconstructions are 3.5× closer to the input than unrelated images are. But dark-blob "hidden codes" partly satisfy it.
- Monet→photo is much weaker than photo→Monet.
- Next steps: longer training, full-resolution fine-tuning, noise in the cycle path, a larger Monet→photo generator.
- Both members used resize-conv upsampling and DiffAugment; the remaining design differences should be stated here.

---

## Reproducibility
- Each member's manifest is in `reproducibility/manifests/` and their raw logs in `reproducibility/raw_logs/<member>/`.
- Smoke test for any run: `LAB_SMOKE=1 jupyter nbconvert --to notebook --execute <notebook>`. The README gives the per-task commands.
