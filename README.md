# DATA266 Lab 1 — Team 6

**Members:** Rajesh Paruchuri, Siva Surya Chandran · **Kaggle team:** PairProgramming_Team_06 · **GitHub:** https://github.com/Paruchuri-Rajesh/DATA_266-GPULab1

This repository contains both members' individual work for the three lab tasks, laid out as the lab PDF specifies, and the team report.
Each member has their own folder in every task (`rajesh_paruchuri/`, `sivasurya_chandran/`), with their own code, config, checkpoints, outputs, logs and `results.md`.

| | Rajesh Paruchuri | Siva Surya Chandran |
|---|---|---|
| Seed | 266 | 1337 |
| Hardware | Apple M4 MacBook Air (MPS) for all tasks | Task 1: NVIDIA T4 (Colab); Task 2: Apple M5 (MPS); Task 3: NVIDIA RTX 4090 (lab PC) |
| Task 1 val CE | 0.711 | **0.639** |
| Task 2 best test accuracy | 0.9523 (BiGRU + attention) | **0.9581** (BiLSTM + attention) |
| Task 3 course-script FID / MiFID | 149.48 / 0.414 | **101.91 / 0.408** |
| Kaggle public score | −74.9486 | −51.1585 (final model) |

**Reports:**
- `report/DATA266_Lab1_Report_Team_6.pdf`: the combined team report, with comparison tables, joint analyses and both members' failure analyses.
- `report/DATA266_Lab1_Report_Sivasurya_Chandran.pdf`: Siva's individual write-up of all three tasks.

## Setup

**Rajesh's runs:**

```bash
git clone https://github.com/Paruchuri-Rajesh/DATA_266-GPULab1.git
cd DATA_266-GPULab1
python -m venv .venv && source .venv/bin/activate
pip install -r reproducibility/manifests/rajesh_paruchuri_task1_requirements.txt \
            -r reproducibility/manifests/rajesh_paruchuri_task2_requirements.txt \
            -r reproducibility/manifests/rajesh_paruchuri_task3_requirements.txt
```

**Siva's runs** (same clone; config-driven, each task run from its member folder):

```bash
pip install -r reproducibility/manifests/sivasurya_chandran_requirements.txt   # pinned, Python 3.9 (Tasks 1-2)
pip install -r task3_gan/requirements-task3.txt                                 # Task 3; install a CUDA build of torch first on a GPU machine
python task3_gan/download_data.py              # Kaggle images -> task3_gan/data/ (KAGGLE_API_TOKEN, or --zip <dataset.zip>)
```

Tasks 1 and 2 download TinyStories / Yelp polarity from the Hugging Face Hub the first time `src/preprocess.py` runs, or you can fetch both with `python tools/download_raw_data.py`.

## Repo layout

```
├── README.md
├── task1_llm/
│   ├── data/                       # TinyStories shard (downloaded by the notebook, not in git)
│   ├── rajesh_paruchuri/
│   └── sivasurya_chandran/
├── task2_sentiment/
│   ├── data/                       # Yelp polarity parquet (downloaded by the notebook, not in git)
│   ├── eval_metrics.py             # Siva's shared classification metrics
│   ├── rajesh_paruchuri/
│   └── sivasurya_chandran/
├── task3_gan/
│   ├── Part3_Evaluation_Script.ipynb   # course-provided evaluation script
│   ├── data/monet_jpg/  photo_jpg/     # Kaggle competition data (not in git)
│   ├── download_data.py  eval_metrics.py  requirements-task3.txt   # Siva's Task 3 helpers
│   ├── rajesh_paruchuri/
│   └── sivasurya_chandran/
├── common/  tools/                 # Siva's shared helpers (device/seed/manifest; metric tables, log export)
├── reproducibility/
│   ├── manifests/                  # environment files + run manifests (both members)
│   └── raw_logs/<member>/          # unedited training logs
└── report/                         # DATA266_Lab1_Report_Team_6.pdf (+ Siva's individual PDF)
```

Each member folder has:
- `src/`: code plus the executed notebook;
- `data_processed/`;
- `checkpoints/`;
- `outputs/`;
- `metrics_report.csv`, `failure_analysis.md`, `results.md`.

Rajesh's runs are config-driven (`src/config.json`). Setting `LAB_SMOKE=1` runs a small smoke test without editing the config. Siva's runs are driven by `config.yaml` in each of his folders.

**Smoke tests verified on 2026-10-03 with the final code** (all three notebooks ran without errors under `LAB_SMOKE=1`):

| Task | Wall time on the Apple M4 |
|---|---|
| Task 1 | 33 min, mostly loading the TinyStories shard while the machine was under memory pressure |
| Task 2 | 10 min |
| Task 3 | 2.5 min, reusing the smoke checkpoint |

Smoke outputs are written to separate `*_smoke` files, so they never overwrite the full-run results.

---

## Task 1 — GPT-style character LM from scratch (TinyStories)

### Rajesh Paruchuri — `task1_llm/rajesh_paruchuri/`

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

### Siva Surya Chandran — `task1_llm/sivasurya_chandran/`

- Character-level GPT written from scratch in `src/model.py` (causal multi-head attention, LayerNorm, GELU feed-forward, pre-norm), driven by `config.yaml`; executed notebook `src/task1_gpt.ipynb`.
- 6 layers / 8 heads / width 256 / context 128, 4.83M parameters, 100K / 10K stories (seed 1337), 10 epochs on a Colab T4 with fp16.
- **Val CE 0.639, perplexity 1.89, 0.921 bits/char, top-1 next-char accuracy 0.796, no NaNs.**
- `checkpoints/gpt_final.pt` and `gpt_best.pt` (19 MB each).

```bash
cd task1_llm/sivasurya_chandran
python src/preprocess.py --config config.yaml    # 100K/10K split, char vocab -> data_processed/
python src/train.py --config config.yaml         # 10 epochs; --resume continues; --replot redraws the loss curves
python src/generate.py --config config.yaml --checkpoint checkpoints/gpt_final.pt
```

## Task 2 — Yelp polarity sentiment bake-off

### Rajesh Paruchuri — `task2_sentiment/rajesh_paruchuri/`

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

### Siva Surya Chandran — `task2_sentiment/sivasurya_chandran/`

- Three models with embeddings learned from scratch:
  - **baseline:** mean-pooled word embeddings → MLP;
  - **experimental 1:** multi-width CNN (kernels 3/4/5);
  - **experimental 2:** BiLSTM + additive attention.
- 503,979 train / 55,997 val / 38,000 test reviews, on an Apple M5 (MPS).
- Test accuracy: baseline **0.9332**, CNN **0.9502**, BiLSTM **0.9581** (best).
- Full metric suite (bootstrap CIs, McNemar, calibration, per-slice results) and a 20-error manual review.

```bash
cd task2_sentiment/sivasurya_chandran
python src/preprocess.py --config config.yaml       # EDA, cleaning, vocab, slices -> data_processed/
python src/train.py --config config.yaml            # all three models -> checkpoints/metrics_task2.json, plots
python src/error_analysis.py --config config.yaml --model_name experimental_2   # 20-error review
```

## Task 3 — CycleGAN photo ↔ Monet

### Rajesh Paruchuri — `task3_gan/rajesh_paruchuri/`

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

### Siva Surya Chandran — `task3_gan/sivasurya_chandran/`

- CycleGAN written from scratch:
  - 9-block / ngf-64 generators with resize-convolution upsampling;
  - InstanceNorm PatchGAN discriminators;
  - translation DiffAugment and EMA generators;
  - 28.3M parameters.
- Final run on the lab RTX 4090: v2's weights trained for 270K more steps (6.9 h, bf16). The best checkpoint was picked on held-out photos in both directions.
- Held-out FID 83.0 (photo→Monet) / 89.7 (Monet→photo); cycle L1 0.105 / 0.084; no NaNs.
- Course evaluation script (executed copy `src/Part3_Evaluation_Script_run.ipynb`): **`submission.csv` FID 101.91 / MiFID 0.408**.
- **Kaggle: final submission scored −51.1585; team public rank 11.**
- Blinded human audit (30 samples, 2 raters): **3.78 / 5, quadratic κ 0.24**. The results are in `checkpoints/audit/`.
- The final weights `checkpoints/{G_AB,G_BA,D_A,D_B}.pt`, both prediction folders (`outputs/pred_A2B`, `outputs/pred_B2A`) and the v1–v3 history (`HISTORY.md`) are committed.

```bash
cd task3_gan/sivasurya_chandran
python src/run_final.py --hours 13 --variants V2_CONTINUE      # train (needs a CUDA GPU; ~7 h on an RTX 4090), then evaluate, translate, score
python src/evaluate.py --config config.yaml                    # metrics in both directions + outputs/pred_A2B
python src/translate.py --config config.yaml --input ../data/photo_jpg --output outputs/pred_B2A --flat
python evaluate_local.py --pred_dir outputs --out submission.csv   # the course evaluation script as a script
python src/human_audit.py score --config config.yaml           # audit score and inter-rater agreement
```

---

## Reproducibility
| Item | Location |
|---|---|
| Run manifest (versions, commands, checkpoint → result) | `reproducibility/manifests/rajesh_paruchuri.md` |
| Pinned environments | `reproducibility/manifests/rajesh_paruchuri_task{1,2,3}_requirements.txt` |
| Raw, unedited training logs | `reproducibility/raw_logs/rajesh_paruchuri/` |
| Siva: run manifest (versions, commands, checkpoint → result) | `reproducibility/manifests/sivasurya_chandran.md` and `sivasurya_chandran_task{1,2,3}_manifest.json` |
| Siva: environments | `reproducibility/manifests/sivasurya_chandran_requirements.txt` (pinned, Tasks 1–2), `sivasurya_chandran_task3_requirements.txt` |
| Siva: raw, unedited training logs | `reproducibility/raw_logs/sivasurya_chandran/` |

Datasets, Rajesh's Task 3 checkpoint, his generated images and large caches are excluded by `.gitignore`. The notebooks download or regenerate them. Siva's processed data (`data_processed/`), resume states and per-epoch checkpoints are not committed either; `src/preprocess.py` and the training scripts regenerate them.
