# DATA266 Lab 1 — Rajesh Paruchuri (Team 6)

**Member:** Rajesh Paruchuri · **Kaggle team:** PairProgramming_Team_06 · **GitHub:** https://github.com/Paruchuri-Rajesh/DATA_266-GPULab1

This repository contains my individual work for the three lab tasks, laid out as the lab PDF specifies.
All runs used seed 266 on an Apple M4 MacBook Air (PyTorch MPS backend).

## Setup

```bash
git clone https://github.com/Paruchuri-Rajesh/DATA_266-GPULab1.git
cd DATA_266-GPULab1
python -m venv .venv && source .venv/bin/activate
pip install -r reproducibility/manifests/rajesh_paruchuri_task1_requirements.txt \
            -r reproducibility/manifests/rajesh_paruchuri_task2_requirements.txt \
            -r reproducibility/manifests/rajesh_paruchuri_task3_requirements.txt
```

## Repo layout

```
├── README.md
├── task1_llm/
│   ├── data/                       # TinyStories shard (downloaded by the notebook, not in git)
│   └── rajesh_paruchuri/
├── task2_sentiment/
│   ├── data/                       # Yelp polarity parquet (downloaded by the notebook, not in git)
│   └── rajesh_paruchuri/
├── task3_gan/
│   ├── Part3_Evaluation_Script.ipynb   # course-provided evaluation script
│   ├── data/monet_jpg/  photo_jpg/     # Kaggle competition data (not in git)
│   └── rajesh_paruchuri/
├── reproducibility/
│   ├── manifests/                  # environment files + run manifest
│   └── raw_logs/rajesh_paruchuri/  # unedited training logs
└── report/
```

Each `rajesh_paruchuri/` folder has:
- `src/`: code plus the executed notebook;
- `data_processed/`;
- `checkpoints/`;
- `outputs/`;
- `metrics_report.csv`, `failure_analysis.md`, `results.md`.

Every run is config-driven (`src/config.json`). Setting `LAB_SMOKE=1` runs a small smoke test without editing the config.

**Smoke tests verified on 2026-10-03 with the final code** (all three notebooks ran without errors under `LAB_SMOKE=1`):

| Task | Wall time on the Apple M4 |
|---|---|
| Task 1 | 33 min, mostly loading the TinyStories shard while the machine was under memory pressure |
| Task 2 | 10 min |
| Task 3 | 2.5 min, reusing the smoke checkpoint |

Smoke outputs are written to separate `*_smoke` files, so they never overwrite the full-run results.

---

## Task 1 — GPT-style character LM from scratch (TinyStories)
`task1_llm/rajesh_paruchuri/`

- Hand-written LayerNorm, causal multi-head attention and a GPT in `src/char_gpt.py`; executed notebook `src/part1_llm.ipynb`.
- 6 layers / 6 heads / width 192 / context 192, 2.72M parameters, story-disjoint 100K / 10K split, 10 epochs.
- **Val CE 0.711, perplexity 2.04, 1.026 bits/char, top-1 next-char accuracy 0.775, no NaNs.**
- `checkpoints/best.pt` (10.9 MB).

```bash
LAB_SMOKE=1 jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  --output part1_llm_smoke.ipynb task1_llm/rajesh_paruchuri/src/part1_llm.ipynb      # smoke test
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
  task1_llm/rajesh_paruchuri/src/part1_llm.ipynb                                       # full run
```

## Task 2 — Yelp polarity sentiment bake-off
`task2_sentiment/rajesh_paruchuri/`

- Three models with embeddings learned from scratch:
  - **baseline:** fastText-style unigram + hashed-bigram bag;
  - **Exp A:** BiGRU + additive attention;
  - **Exp B:** 2-layer Transformer encoder written from scratch.
- 539,941 train / 20,000 val / 38,000 test reviews.
- Test macro-F1: baseline **0.9515**, BiGRU **0.9523** (best), Transformer **0.9323**.
- Full metric suite (bootstrap CIs, McNemar, calibration, per-slice results) and a 20-error manual review.

```bash
LAB_SMOKE=1 jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  --output part2_sentiment_smoke.ipynb task2_sentiment/rajesh_paruchuri/src/part2_sentiment.ipynb
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
  task2_sentiment/rajesh_paruchuri/src/part2_sentiment.ipynb
```

## Task 3 — CycleGAN photo ↔ Monet
`task3_gan/rajesh_paruchuri/`

- CycleGAN written from scratch:
  - 6-block / ngf-48 generators;
  - spectral-norm PatchGAN discriminators;
  - DiffAugment and EMA generators;
  - 14.4M parameters, 80 epochs.
- Photo→Monet FID 105.7 on all 7,038 photos; cycle L1 0.077; no NaNs.
- Course evaluation script: **`submission.csv` FID 149.48 / MiFID 0.414**.
- **Kaggle (team PairProgramming_Team_06): my submission scored −74.9486; team public rank 11.**
- The 30-image blinded human-audit kit is in `outputs/human_audit/`; ratings are pending.
- `checkpoints/cyclegan_full.pt` (95 MB) is not in git; see `checkpoints/README.md`.

```bash
kaggle competitions download -c data-266-fall-2026-gan-image-style-transfer   # then unzip monet_jpg/, photo_jpg/ into task3_gan/data/
python task3_gan/rajesh_paruchuri/src/cyclegan.py full                        # train (~2.6 h on Apple M4)
jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 \
  task3_gan/rajesh_paruchuri/src/part3_cyclegan.ipynb                         # evaluate, official score, audit kit
```

---

## Reproducibility
| Item | Location |
|---|---|
| Run manifest (versions, commands, checkpoint → result) | `reproducibility/manifests/rajesh_paruchuri.md` |
| Pinned environments | `reproducibility/manifests/rajesh_paruchuri_task{1,2,3}_requirements.txt` |
| Raw, unedited training logs | `reproducibility/raw_logs/rajesh_paruchuri/` |

Datasets, the Task 3 checkpoint, generated images and large caches are excluded by `.gitignore`. The notebooks download or regenerate them.
